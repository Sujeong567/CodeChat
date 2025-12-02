# client_backend/model/base_llm.py

import collections  # (지금은 크게 안 쓰지만 남겨둬도 무방)
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
    TARGET_LAYER_INDEX,  # 15
)
from common.model_utils import get_bnb_config, get_lora_config
from common.protocol import (
    EncryptedInferenceRequest,
    encode_bytes_to_base64,
    decode_base64_to_bytes,
)


class BaseLLMLoader:
    """
    - 토크나이저 / Base LLM 로딩
    - QLoRA 래핑
    - q_proj same-token FHE-LoRA hook 등록
    """

    def __init__(self):
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._hidden_size = None
        self._eos_token_id = None

        # 등록된 hook 핸들 보관
        self._xL_pre_hooks = []
        self._output_post_hooks = []

        # FHE 클라이언트 (나중에 attach_fhe_client 에서 주입)
        self._ckks_manager = None
        self._http_session = None
        self._server_url = None

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

        # 여기서는 아직 FHE 클라이언트 정보가 없을 수 있음
        # attach_fhe_client 호출 이후 다시 _register_lora_hooks()를 부를 것
        self._register_lora_hooks()

        self._hidden_size = self._peft_model.config.hidden_size
        self._eos_token_id = self._tokenizer.eos_token_id
        print(f"[BaseLLM] Hidden Size = {self._hidden_size}, EOS ID = {self._eos_token_id}")

    # ------------------------------------------------------------------
    #  FHE 클라이언트 주입
    # ------------------------------------------------------------------
    def attach_fhe_client(self, ckks_manager, http_session: requests.Session, server_url: str):
        """
        ckks_manager: CKKSClientManager 인스턴스
        http_session: requests.Session
        server_url: FHE-LoRA 서버의 /compute_lora URL
        """
        self._ckks_manager = ckks_manager
        self._http_session = http_session
        self._server_url = server_url

        # FHE 클라이언트 정보가 준비된 상태에서 hook 다시 등록
        self._register_lora_hooks()

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
        레이어 TARGET_LAYER_INDEX 의 self_attn.q_proj에만
        same-token FHE-LoRA hook 등록
        """
        # 기존 hook 제거
        self.clear_lora_hooks()

        # FHE 클라이언트가 아직 attach 안 됐다면 hook을 붙이지 않음
        if self._ckks_manager is None or self._http_session is None or self._server_url is None:
            print("[BaseLLM] FHE client not attached yet, skip LoRA hooks for now.")
            return

        target_layer = TARGET_LAYER_INDEX
        sig_q = f"base_model.model.model.layers.{target_layer}.self_attn.q_proj"

        ckks = self._ckks_manager
        session = self._http_session
        server_url = self._server_url

        def q_post_hook(module, inputs, output):
            """
            same-token q_proj hook:
            1) 입력 x_L를 가져와서
            2) CKKS 암호화 → 서버 /compute_lora → delta 복호화
            3) 현재 q_proj output에 delta를 더한 뒤 반환
            """
            try:
                x = inputs[0]  # (B,T,H) 또는 (B,H)
                if x.dim() == 3:
                    xL = x[:, -1, :].detach().clone()
                else:
                    xL = x.detach().clone()

                # (1,H) → (H,)
                xL_vec = xL.squeeze(0)

                # 1) 암호화
                enc_bytes = ckks.encrypt_tensor(xL_vec)
                payload = {
                    "enc_hidden_state_bytes": encode_bytes_to_base64(enc_bytes)
                }

                # 2) 서버 호출
                res = session.post(server_url, json=payload)
                res.raise_for_status()
                resp_json = res.json()
                enc_delta_b64 = resp_json["enc_lora_delta_bytes"]

                # 3) delta 복호화
                delta_bytes = decode_base64_to_bytes(enc_delta_b64)
                delta_vec = ckks.decrypt_tensor(delta_bytes)  # (H,)
                delta = delta_vec.unsqueeze(0).to(output.device)  # (1,H)

                # 4) 현재 토큰의 q_proj output에 delta 주입
                if output.dim() == 3:
                    if delta.shape[-1] != output.shape[-1]:
                        print(
                            f"[WARN] q_proj delta mismatch: delta={delta.shape}, output={output.shape}"
                        )
                        return output
                    output[:, -1, :] = output[:, -1, :] + delta.to(output.dtype)
                elif output.dim() == 2:
                    if delta.shape[-1] != output.shape[-1]:
                        print(
                            f"[WARN] q_proj delta mismatch: delta={delta.shape}, output={output.shape}"
                        )
                        return output
                    output[:, :] = output[:, :] + delta.to(output.dtype)
                else:
                    print(f"[WARN] Unexpected q_proj output dim: {output.shape}")

                return output

            except Exception as e:
                import traceback
                print("[HOOK] q_proj FHE-LoRA hook error:", e)
                traceback.print_exc()
                # 실패해도 모델이 죽지 않도록, 원본 output 그대로 반환
                return output

        # 실제 hook 등록
        for name, module in self._peft_model.named_modules():
            if isinstance(module, LoraLayer) and sig_q in name:
                post_hook = module.register_forward_hook(q_post_hook)
                self._output_post_hooks.append(post_hook)
                print(f"[HOOK] Registered same-token q_proj hook at: {name}")
                break

    def clear_lora_hooks(self):
        """등록된 hook 제거"""
        for h in self._xL_pre_hooks + self._output_post_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._xL_pre_hooks.clear()
        self._output_post_hooks.clear()

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
