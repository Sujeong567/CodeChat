import collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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


# (layer, module) → delta tensor
GLOBAL_DELTA_DICT = {}


class BaseLLMLoader:
    """
    Multi-layer × Multi-module Same-token FHE-LoRA 지원 LLM Loader
    """

    def __init__(self):
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._hidden_size = None
        self._eos_token_id = None

        # (layer,module) → x_L queue
        self._xl_queue = {
            (l, m): collections.deque() for l in FHE_LAYERS for m in FHE_MODULES
        }

        self._pre_hooks = []
        self._post_hooks = []

        # FHE 클라이언트 관련
        self._ckks_manager = None
        self._http_session = None
        self._server_url = None

    # ------------------------------------------------------------------
    #  LayerNorm / final_ln 찾기 (PostProcessor에서 사용)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_final_layernorm(module: torch.nn.Module):
        """
        LLM 구조에서 lm_head 직전에 있는 Final LayerNorm을 자동으로 탐색하여 반환
        (StarCoder2 기준으로 마지막 LayerNorm 후보들 중 마지막 것을 선택)
        """
        candidates = []
        for name, m in module.named_modules():
            lname = name.lower()
            if "ln_f" in lname or "layernorm" in lname or "norm" in lname:
                candidates.append((name, m))

        if candidates:
            # 마지막 LayerNorm이 실제 final_ln일 확률이 높음
            return candidates[-1][1]

        print("[WARN] Final LayerNorm을 찾지 못했습니다.")
        return None

    # ------------------------------------------------------------------
    # 모델 로딩
    # ------------------------------------------------------------------
    def load_model(self):
        print(f"[BaseLLM] Loading tokenizer: {LLM_NAME}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME,
            cache_dir=HF_CACHE_DIR,
            use_fast=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print("[BaseLLM] Tokenizer loaded.")

        print("[BaseLLM] Loading base model (4bit)...")
        bnb_config = get_bnb_config()
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            quantization_config=bnb_config,
            device_map={"": DEVICE},
            torch_dtype=BNB_COMPUTE_DTYPE,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True,
        )

        # vocab resize 대응
        need_vocab = len(self._tokenizer)
        have_vocab = base_model.get_input_embeddings().weight.shape[0]
        if need_vocab != have_vocab:
            base_model.resize_token_embeddings(need_vocab)

        base_model.eval()
        print("[BaseLLM] Base model loaded.")

        # LayerNorm/Norm 계열 float32
        for name, module in base_model.named_modules():
            if isinstance(module, torch.nn.LayerNorm) or "Norm" in module.__class__.__name__:
                module.float()

        # lm_head float32
        if hasattr(base_model, "lm_head"):
            base_model.lm_head = base_model.lm_head.to(torch.float32)

        self._base_model = base_model

        # QLoRA 래핑
        lora_config = get_lora_config()
        self._peft_model = get_peft_model(self._base_model, lora_config)
        self.reset_lora_weights()
        print("[BaseLLM] LoRA wrapping done.")

        # hook은 FHE 클라이언트 attach 후 다시 등록됨
        self._register_hooks()

        self._hidden_size = self._peft_model.config.hidden_size
        self._eos_token_id = self._tokenizer.eos_token_id
        print(f"[BaseLLM] Hidden={self._hidden_size}, EOS={self._eos_token_id}")

    # ------------------------------------------------------------------
    # FHE 클라이언트 등록
    # ------------------------------------------------------------------
    def attach_fhe_client(self, ckks_manager, http_session, server_url):
        self._ckks_manager = ckks_manager
        self._http_session = http_session
        self._server_url = server_url

        # hook 다시 등록
        self._register_hooks()

    # ------------------------------------------------------------------
    # LoRA 가중치 0으로 초기화
    # ------------------------------------------------------------------
    def reset_lora_weights(self):
        if self._peft_model is None:
            return
        for name, param in self._peft_model.named_parameters():
            if "lora_" in name:
                param.data.zero_()
        print("[BaseLLM] LoRA weights reset.")

    # ------------------------------------------------------------------
    # delta 관리
    # ------------------------------------------------------------------
    def reset_delta(self):
        GLOBAL_DELTA_DICT.clear()

    def set_delta(self, delta_dict):
        """
        delta_dict: key "(15,'q_proj')" → tensor
        """
        for key, val in delta_dict.items():
            layer_idx, mod = eval(key)
            GLOBAL_DELTA_DICT[(layer_idx, mod)] = val

    # ------------------------------------------------------------------
    # Hook 등록
    # ------------------------------------------------------------------
    def _register_hooks(self):
        self.clear_hooks()

        if self._ckks_manager is None:
            print("[BaseLLM] FHE client not attached yet. skip hooks.")
            return

        def parse_layer(path):
            # e.g. "base_model.model.model.layers.15.self_attn.q_proj"
            parts = path.split(".")
            return int(parts[4]), parts[-1]

        def make_pre_hook(layer_idx, module_name):
            def hook(module, args):
                x = args[0]
                last = x[:, -1, :] if x.dim() == 3 else x
                self._xl_queue[(layer_idx, module_name)].append(last.detach().clone())
            return hook

        def make_post_hook(layer_idx, module_name):
            def hook(module, inp, out):
                delta = GLOBAL_DELTA_DICT.get((layer_idx, module_name), None)
                if delta is None:
                    return out
                if out.dim() == 3:
                    out[:, -1, :] += delta.to(out.dtype)
                else:
                    out += delta.to(out.dtype)
                return out
            return hook

        for name, module in self._peft_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue

            layer_idx, mod = parse_layer(name)

            if layer_idx in FHE_LAYERS and mod in FHE_MODULES:
                pre = module.register_forward_pre_hook(make_pre_hook(layer_idx, mod))
                post = module.register_forward_hook(make_post_hook(layer_idx, mod))
                self._pre_hooks.append(pre)
                self._post_hooks.append(post)
                print(f"[HOOK] Registered ({layer_idx}, {mod})")

    def clear_hooks(self):
        for h in self._pre_hooks + self._post_hooks:
            try:
                h.remove()
            except:
                pass
        self._pre_hooks.clear()
        self._post_hooks.clear()

    # ------------------------------------------------------------------
    # x_L 배치 전달 (모든 layer,module)
    # ------------------------------------------------------------------
    def get_xl_batch(self):
        batch = {}
        for key, q in self._xl_queue.items():
            if not q:
                raise RuntimeError(f"x_L missing: {key}")
            batch[str(key)] = q.popleft()
        return batch

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
