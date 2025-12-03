import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft.tuners.lora import LoraLayer

from common.config import (
    LLM_NAME,
    HF_CACHE_DIR,
    BNB_COMPUTE_DTYPE,
    DEVICE,
    FHE_LAYERS,
    FHE_MODULES,
)
from common.model_utils import get_bnb_config, get_lora_config
from common.protocol import (
    encode_bytes_to_base64,
    decode_base64_to_bytes,
)


class BaseLLMLoader:
    """
    - 토크나이저 / Base LLM 로딩
    - QLoRA 래핑
    - same-token FHE-LoRA hooks (multi-layer × multi-module)
    """

    def __init__(self):
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._hidden_size = None
        self._eos_token_id = None

        # FHE client (attach_fhe_client에서 설정)
        self._ckks_manager = None
        self._http_session: requests.Session | None = None
        self._server_url: str | None = None

        # hook 핸들 보관
        self._pre_hooks = []
        self._post_hooks = []

    # ------------------------------------------------------------------
    #  LayerNorm / final_ln 찾기
    # ------------------------------------------------------------------
    @staticmethod
    def _find_final_layernorm(module: torch.nn.Module):
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

        need_vocab = len(self._tokenizer)
        have_vocab = base_model.get_input_embeddings().weight.shape[0]
        if have_vocab != need_vocab:
            print(f"[BaseLLM] resize_token_embeddings: {have_vocab} -> {need_vocab}")
            base_model.resize_token_embeddings(need_vocab)

        base_model.eval()
        print("[BaseLLM] Base LLM 로딩 완료.")

        # LayerNorm / Norm은 float32
        for _name, module in base_model.named_modules():
            if isinstance(module, torch.nn.LayerNorm) or "Norm" in module.__class__.__name__:
                module.float()

        if hasattr(base_model, "lm_head"):
            base_model.lm_head = base_model.lm_head.to(torch.float32)

        self._base_model = base_model

        # QLoRA 래핑
        lora_config = get_lora_config()
        self._peft_model = get_peft_model(self._base_model, lora_config)

        # LoRA 가중치 0 초기화
        self.reset_lora_weights()
        print("[BaseLLM] LoRA 래핑 완료 및 가중치 0 초기화.")

        # 아직 FHE 클라이언트 정보 없음 → hook은 attach 이후 등록
        self._hidden_size = self._peft_model.config.hidden_size
        self._eos_token_id = self._tokenizer.eos_token_id
        print(f"[BaseLLM] Hidden Size = {self._hidden_size}, EOS ID = {self._eos_token_id}")

    # ------------------------------------------------------------------
    #  FHE client attach
    # ------------------------------------------------------------------
    def attach_fhe_client(self, ckks_manager, http_session: requests.Session, server_url: str):
        self._ckks_manager = ckks_manager
        self._http_session = http_session
        self._server_url = server_url

        self._register_fhe_hooks()

    # ------------------------------------------------------------------
    #  LoRA 관련 유틸
    # ------------------------------------------------------------------
    def reset_lora_weights(self):
        if self._peft_model is None:
            return
        for name, param in self._peft_model.named_parameters():
            if "lora_" in name:
                param.data.zero_()
        print("[BaseLLM] 모든 LoRA 어댑터 가중치를 0으로 리셋했습니다.")

    def clear_fhe_hooks(self):
        for h in self._pre_hooks + self._post_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._pre_hooks.clear()
        self._post_hooks.clear()

    def _register_fhe_hooks(self):
        """
        FHE_LAYERS × FHE_MODULES 조합에 대해
        same-token FHE-LoRA hook을 등록.
        각 (layer, module)마다:
          - 입력 x_L 암호화 → 서버에 layer/module 정보 포함해서 delta 요청
          - delta 복호화 후 해당 모듈 output에만 추가
        """
        self.clear_fhe_hooks()

        if (
            self._ckks_manager is None
            or self._http_session is None
            or self._server_url is None
        ):
            print("[BaseLLM] FHE client not attached yet. Skip FHE hooks.")
            return

        ckks = self._ckks_manager
        session = self._http_session
        server_url = self._server_url

        def parse_layer_and_module(path: str):
            # 예: "base_model.model.model.layers.15.self_attn.q_proj.lora_A"
            parts = path.split(".")
            # ... layers.{idx}.self_attn.{mod} ...
            layer_idx = int(parts[4])
            module_name = parts[7] if len(parts) > 7 else parts[-1]
            return layer_idx, module_name

        def make_post_hook(layer_idx: int, module_name: str):
            def hook(module, inputs, output):
                try:
                    x = inputs[0]
                    if x.dim() == 3:
                        xL = x[:, -1, :].detach().clone()
                    else:
                        xL = x.detach().clone()

                    xL_vec = xL.squeeze(0)  # (H,)

                    # 1) 암호화
                    enc_bytes = ckks.encrypt_tensor(xL_vec)
                    payload = {
                        "layer_idx": layer_idx,
                        "module_name": module_name,
                        "enc_hidden_state_bytes": encode_bytes_to_base64(enc_bytes),
                    }

                    # 2) 서버 호출
                    res = session.post(server_url, json=payload)
                    res.raise_for_status()
                    resp_json = res.json()
                    enc_delta_b64 = resp_json["enc_lora_delta_bytes"]

                    # 3) delta 복호화
                    delta_bytes = decode_base64_to_bytes(enc_delta_b64)
                    delta_vec = ckks.decrypt_tensor(delta_bytes)  # (H,)
                    delta = delta_vec.unsqueeze(0).to(output.device)  # (1, H)

                    # 4) same-token output에 delta 주입
                    if output.dim() == 3:
                        if delta.shape[-1] != output.shape[-1]:
                            print(
                                f"[WARN] delta mismatch ({layer_idx},{module_name}): "
                                f"delta={delta.shape}, out={output.shape}"
                            )
                            return output
                        output[:, -1, :] = output[:, -1, :] + delta.to(output.dtype)
                    elif output.dim() == 2:
                        if delta.shape[-1] != output.shape[-1]:
                            print(
                                f"[WARN] delta mismatch ({layer_idx},{module_name}): "
                                f"delta={delta.shape}, out={output.shape}"
                            )
                            return output
                        output[:, :] = output[:, :] + delta.to(output.dtype)
                    else:
                        print(f"[WARN] Unexpected output dim: {output.shape}")

                    return output

                except Exception as e:
                    import traceback
                    print(f"[HOOK] FHE-LoRA hook error at ({layer_idx},{module_name}): {e}")
                    traceback.print_exc()
                    # 에러 나도 원본 output 그대로
                    return output

            return hook

        for name, module in self._peft_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue

            try:
                layer_idx, mod = parse_layer_and_module(name)
            except Exception:
                continue

            if layer_idx in FHE_LAYERS and mod in FHE_MODULES:
                post = module.register_forward_hook(make_post_hook(layer_idx, mod))
                self._post_hooks.append(post)
                print(f"[HOOK] Registered same-token FHE hook at (layer={layer_idx}, mod={mod})")

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
        lm_head = self._peft_model.base_model.lm_head
        w = lm_head.weight.data.to(torch.float32)
        b = getattr(lm_head, "bias", None)
        if b is None:
            b = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
        else:
            b = b.data.to(torch.float32)
        return w, b
