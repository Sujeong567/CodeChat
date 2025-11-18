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

# 서버에서 계산한 LoRA 델타를 임시로 담아두는 전역 변수
GLOBAL_INJECTED_LORA_OUTPUT_DELTA = None


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
        - 대표 LoRA 레이어(예: model.layers.0.self_attn.q_proj)에만 Hook 1쌍 등록
        - pre-hook: 마지막 토큰의 입력 xL 캡처
        - post-hook: 서버에서 받은 델타를 출력의 마지막 토큰 위치에 더함
        """
        self.clear_lora_hooks()

        # StarCoder2에서 q_proj 이름 패턴 (Peft 래핑 후 prefix 포함)을 부분 매칭으로 찾기
        # ex) "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default"
        target_signature = f"layers.{TARGET_LAYER_INDEX}.self_attn.{REPRESENTATIVE_LORA_TARGET_MODULE}"

        # --- hook 내부 함수들 (반드시 _register_lora_hooks 안에 중첩 함수로 정의!) ---
        def save_xL_input_hook(module, args):
            """
            LoraLayer 실행 직전 입력 캡처.
            args[0] shape: (batch, seq_len, hidden)
            우리는 마지막 토큰의 hidden만 사용 → (batch, hidden) 형태로 큐에 push.
            """
            x = args[0]  # (B, S, H)
            try:
                last_token_input = x[:, -1, :].detach().clone()  # (B, H)
                self._lora_xL_inputs_queue.append(last_token_input)
                print(f"[HOOK] xL captured: {last_token_input.shape}")
            except Exception as e:
                print(f"[BaseLLM] WARN: xL 캡처 실패: {e}")

        def inject_delta_output_hook(module, input, output):
            """
            LoraLayer 실행 직후 호출.
            GLOBAL_INJECTED_LORA_OUTPUT_DELTA에 저장된 델타를
            출력 텐서의 마지막 토큰 위치에 더해준다.
            output shape: (batch, seq_len, hidden)
            """
            global GLOBAL_INJECTED_LORA_OUTPUT_DELTA
            if GLOBAL_INJECTED_LORA_OUTPUT_DELTA is None:
                return output

            delta = GLOBAL_INJECTED_LORA_OUTPUT_DELTA
            try:
                # delta: (B, H) 또는 (H,)
                if delta.dim() == 1:
                    # (H,) -> (1, H)
                    delta = delta.unsqueeze(0)
                if delta.dim() == 2:
                    # (B, H) -> (B, 1, H) : seq_len 차원에 브로드캐스트 가능하게
                    delta = delta.unsqueeze(1)

                if delta.shape[-1] != output.shape[-1]:
                    print(
                        f"[BaseLLM] WARN: LoRA 델타 주입 실패: "
                        f"delta={delta.shape}, output={output.shape}"
                    )
                    return output

                delta = delta.to(device=output.device, dtype=output.dtype)
                # 마지막 토큰 위치에만 델타 추가: output[:, -1:, :] shape (B, 1, H)
                output[:, -1:, :] = output[:, -1:, :] + delta
                print(
                    f"[DELTA] Injected delta at module={module.__class__.__name__}, "
                    f"delta_shape={delta.shape}, output_shape={output.shape}"
                )
            except Exception as e:
                print(f"[BaseLLM] WARN: LoRA 델타 주입 예외: {e}")
            return output

        # 실제로는 대표 레이어 하나에만 Hook 등록
        for name, module in self._peft_model.named_modules():
            if isinstance(module, LoraLayer) and target_signature in name:
                pre_h = module.register_forward_pre_hook(save_xL_input_hook)
                post_h = module.register_forward_hook(inject_delta_output_hook)
                self._xL_pre_hooks.append(pre_h)
                self._output_post_hooks.append(post_h)
                print(f"[HOOK] Registered hooks at: {name}")
                break
        else:
            print(f"[BaseLLM] WARN: 대상 LoRA 레이어를 찾지 못했습니다: {target_signature}")

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

    def set_global_lora_output_delta(self, delta: torch.Tensor):
        """
        서버에서 복호화한 LoRA 델타를 전역 변수에 세팅.
        - 기대 shape: (1, hidden) 또는 (hidden,)
        """
        global GLOBAL_INJECTED_LORA_OUTPUT_DELTA
        GLOBAL_INJECTED_LORA_OUTPUT_DELTA = delta
        print(f"[BaseLLM] Global LoRA delta set: {tuple(delta.shape)}")

    def clear_global_lora_output_delta(self):
        """한 토큰 step에서 델타 사용이 끝난 뒤 반드시 호출"""
        global GLOBAL_INJECTED_LORA_OUTPUT_DELTA
        GLOBAL_INJECTED_LORA_OUTPUT_DELTA = None

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
