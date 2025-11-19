# client_backend/model/base_llm.py

import collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft.tuners.lora import LoraLayer

from common.config import (
    LLM_NAME,
    HF_CACHE_DIR,
    BNB_COMPUTE_DTYPE,
    DEVICE,
    TARGET_LAYER_INDEX,
    REPRESENTATIVE_LORA_TARGET_MODULE,
)
from common.model_utils import get_bnb_config, get_lora_config

# proj별 델타를 관리하는 전역 dict
GLOBAL_INJECTED_LORA_OUTPUT_DELTAS = {}


class BaseLLMLoader:
    """
    - 토크나이저 / Base LLM 로딩
    - QLoRA 래핑
    - xL 캡처 / 델타 주입용 Hook 등록
    """

    def __init__(self):
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._hidden_size = None
        self._eos_token_id = None

        # 마지막 토큰의 xL 입력들을 순서대로 쌓아두는 큐
        self._lora_xL_inputs_queue = collections.deque()

        # 등록된 hook 핸들 보관
        self._xL_pre_hooks = []
        self._output_post_hooks = []

    # ------------------------------------------------------------------
    #  LayerNorm / final_ln 찾기 (PostProcessor에서 사용)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_final_layernorm(module: torch.nn.Module):
        """
        lm_head 직전 최종 LayerNorm 추정해서 반환
        (StarCoder2 기준으로 마지막 LayerNorm 후보들 중 마지막 것을 선택)
        """
        candidates = []
        for name, m in module.named_modules():
            n_lower = name.lower()
            if "ln_f" in n_lower or "layernorm" in n_lower or "norm" in n_lower:
                candidates.append((name, m))
        if candidates:
            return candidates[-1][1]
        return None

    # ------------------------------------------------------------------
    #  모델 로딩
    # ------------------------------------------------------------------
    def load_model(self):
        print(f"[BaseLLM] 토크나이저 로딩 중: {LLM_NAME}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME,
            cache_dir=HF_CACHE_DIR,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print("[BaseLLM] 토크나이저 로딩 완료.")

        print("[BaseLLM] Base LLM 로딩 중 (4bit / bfloat16 compute)...")
        bnb_config = get_bnb_config()
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            quantization_config=bnb_config,
            device_map={"": DEVICE},
            torch_dtype=BNB_COMPUTE_DTYPE,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True,
        )

        # vocab 크기 맞추기 (학습 시 special tokens 추가된 경우 대비)
        need_vocab = len(self._tokenizer)
        have_vocab = base_model.get_input_embeddings().weight.shape[0]
        if have_vocab != need_vocab:
            print(f"[BaseLLM] resize_token_embeddings: {have_vocab} -> {need_vocab}")
            base_model.resize_token_embeddings(need_vocab)

        base_model.eval()
        print("[BaseLLM] Base LLM 로딩 완료.")

        # LayerNorm / Norm 계열은 float32로 캐스팅
        for _name, module in base_model.named_modules():
            if isinstance(module, torch.nn.LayerNorm) or "Norm" in module.__class__.__name__:
                module.float()

        # lm_head도 float32로
        if hasattr(base_model, "lm_head"):
            base_model.lm_head = base_model.lm_head.to(torch.float32)

        self._base_model = base_model

        # QLoRA 슬롯 생성 (학습과 동일 target_modules)
        lora_config = get_lora_config()
        self._peft_model = get_peft_model(self._base_model, lora_config)

        # LoRA 가중치 0으로 초기화 (실제 효과는 FHE 델타로만 반영)
        self.reset_lora_weights()
        print("[BaseLLM] LoRA 래핑 완료 및 가중치 0 초기화.")

        # xL 캡처 / 델타 주입용 Hook 등록
        self._register_lora_hooks()

        self._hidden_size = self._peft_model.config.hidden_size
        self._eos_token_id = self._tokenizer.eos_token_id
        print(f"[BaseLLM] Hidden Size = {self._hidden_size}, EOS ID = {self._eos_token_id}")

    # ------------------------------------------------------------------
    #  LoRA 관련 유틸
    # ------------------------------------------------------------------
    def reset_lora_weights(self):
        """모든 LoRA 어댑터 가중치를 0으로 초기화"""
        if self._peft_model is None:
            return
        for name, param in self._peft_model.named_parameters():
            if "lora_" in name:
                param.data.zero_()
        print("[BaseLLM] 모든 LoRA 어댑터 가중치를 0으로 리셋했습니다.")

    def _register_lora_hooks(self):
        """
        하나의 레이어(TARGET_LAYER_INDEX)의 self_attn.{q,k,v,o}_proj에 대해 hook 등록.
        - q_proj: x_L 캡처 + delta 주입
        - k/v/o_proj: delta 주입만
        """
        self.clear_lora_hooks()

        TARGET_LAYER = TARGET_LAYER_INDEX
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        def save_xL_input_hook(module, args):
            # args[0]: (batch, seq, hidden)
            x = args[0]
            last_token_input = x[:, -1, :].detach().clone()
            self._lora_xL_inputs_queue.append(last_token_input)
            print(f"[HOOK] xL captured: {last_token_input.shape}")

        def make_inject_delta_hook(proj_key: str):
            def _inject(module, input, output):
                global GLOBAL_INJECTED_LORA_OUTPUT_DELTAS
                delta = GLOBAL_INJECTED_LORA_OUTPUT_DELTAS.get(proj_key)
                if delta is None:
                    return output

                # delta: (H,) 또는 (1, H) 허용
                if delta.dim() == 2 and delta.shape[0] == 1:
                    delta_use = delta[0]
                else:
                    delta_use = delta

                if delta_use.shape[-1] != output.shape[-1]:
                    print(
                        f"[DELTA] Skip {proj_key}: delta={delta_use.shape}, output={output.shape}"
                    )
                    return output

                out = output.clone()
                out[:, -1, :] = out[:, -1, :] + delta_use.to(out.dtype)
                print(
                    f"[DELTA] Injected delta for {proj_key} at module={module.__class__.__name__}, "
                    f"delta_shape={delta_use.shape}, output_shape={out.shape}"
                )
                return out

            return _inject

        for name, module in self._peft_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            if f".layers.{TARGET_LAYER}.self_attn." not in name:
                continue

            for proj_key in TARGET_MODULES:
                if f".self_attn.{proj_key}" in name:
                    # q_proj 하나에서만 x_L 캡처
                    if proj_key == "q_proj":
                        pre_hook = module.register_forward_pre_hook(save_xL_input_hook)
                        self._xL_pre_hooks.append(pre_hook)

                    hook = module.register_forward_hook(make_inject_delta_hook(proj_key))
                    self._output_post_hooks.append(hook)

                    print(f"[HOOK] Registered hooks at: {name} ({proj_key})")
                    break


    def clear_lora_hooks(self):
        """등록된 hook 제거 및 큐 초기화"""
        for h in self._xL_pre_hooks + self._output_post_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._xL_pre_hooks.clear()
        self._output_post_hooks.clear()
        self._lora_xL_inputs_queue.clear()

    def get_lora_xL_input(self) -> torch.Tensor:
        """
        PreProcessor에서 xL을 꺼갈 때 사용.
        - 반환 shape: (batch, hidden)
        """
        if not self._lora_xL_inputs_queue:
            raise RuntimeError("캡처된 xL 입력이 없습니다.")
        return self._lora_xL_inputs_queue.popleft()

    def set_global_lora_output_deltas(self, delta_dict: dict):
        """
        delta_dict: {"q_proj": tensor(1, H), ...}
        """
        global GLOBAL_INJECTED_LORA_OUTPUT_DELTAS
        GLOBAL_INJECTED_LORA_OUTPUT_DELTAS = {}
        for k, v in delta_dict.items():
            GLOBAL_INJECTED_LORA_OUTPUT_DELTAS[k] = v.detach().to(torch.float32)

        keys = ", ".join(
            f"{k}:{tuple(v.shape)}" for k, v in GLOBAL_INJECTED_LORA_OUTPUT_DELTAS.items()
        )
        print(f"[BaseLLM] Global LoRA deltas set: {keys}")

    def clear_global_lora_output_deltas(self):
        global GLOBAL_INJECTED_LORA_OUTPUT_DELTAS
        GLOBAL_INJECTED_LORA_OUTPUT_DELTAS.clear()

    # ------------------------------------------------------------------
    #  프로퍼티 / LM Head 가중치
    # ------------------------------------------------------------------
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def peft_model(self):
        return self._peft_model

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def get_lm_head_weights(self):
        """
        PostProcessor에서 사용할 LM Head 가중치 (float32) 반환
        """
        lm_head = self._peft_model.base_model.lm_head
        w = lm_head.weight.data.to(torch.float32)
        b = getattr(lm_head, "bias", None)
        if b is None:
            b = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
        else:
            b = b.data.to(torch.float32)
        return w, b
