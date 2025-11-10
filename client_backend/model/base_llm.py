# codechat/client-backend/model/base_llm.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer
import gc
import collections

from common.config import (
    LLM_NAME, HF_CACHE_DIR, BNB_COMPUTE_DTYPE, DEVICE,
    REPRESENTATIVE_LORA_TARGET_MODULE, LORA_INJECTION_MODULES
) 
from common.model_utils import get_bnb_config, get_lora_config

GLOBAL_INJECTED_LORA_OUTPUT_DELTA = None

class BaseLLMLoader:
    """
    모델 및 토크나이저 로딩 및 관리
    """

    @staticmethod
    def _find_final_layernorm(module: torch.nn.Module):
        """모델 내부에서 final layernorm 추정 → 수백만 개의 행렬 곱으로 널뛰는 값 정규화"""
        candidates = []
        for name, m in module.named_modules():
            n_lower = name.lower()
            if ("ln_f" in n_lower or "norm" in n_lower or "layernorm" in n_lower):
                candidates.append((name, m))
        return candidates[-1][1] if candidates else None

    def __init__(self):
        self._tokenizer = None
        self._base_model = None
        self._peft_model = None
        self._hidden_size = None
        self._eos_token_id = None
        self._lora_xL_inputs_queue = collections.defaultdict(collections.deque)
        self._xL_pre_hooks = []
        self._output_post_hooks = []

    def load_model(self):
        print(f"[BaseLLM] 토크나이저 로딩 중: {LLM_NAME}")
        self._tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, cache_dir=HF_CACHE_DIR)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self._tokenizer.chat_template is None:
            self._tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}\n{% endif %}{% endfor %}"
        print("[BaseLLM] 토크나이저 로딩 완료.")

        print(f"[BaseLLM] Base LLM 로딩 중 (4-bit)...")
        # float16으로 로드 → 메모리 절약
        bnb_config = get_bnb_config()
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, quantization_config=bnb_config, device_map="auto",
            torch_dtype=BNB_COMPUTE_DTYPE, cache_dir=HF_CACHE_DIR
        )
        base_model.eval()  # 모델 추론 모드 설정
        print("[BaseLLM] Base LLM 로딩 완료.")

        # Dtype 패치 → float32로 캐스팅
        if hasattr(base_model, "lm_head"):
            base_model.lm_head = base_model.lm_head.to(torch.float32)

        final_ln = self._find_final_layernorm(base_model)
        if final_ln is not None:
            final_ln.float()

        self._base_model = base_model
        lora_config = get_lora_config()
        self._peft_model = get_peft_model(self._base_model, lora_config)

        self.reset_lora_weights()
        print("[BaseLLM] LoRA 모델로 Wrapping 완료.")

        self._register_lora_hooks()

        self._hidden_size = self._peft_model.config.hidden_size
        self._eos_token_id = self._tokenizer.eos_token_id
        print(f"[BaseLLM] Hidden Size: {self.hidden_size}")

        gc.collect()
        torch.cuda.empty_cache()
    
    def reset_lora_weights(self):
        """서버가 델타를 주입하므로, 클라이언트의 LoRA 가중치는 0으로 리셋"""
        for name, param in self._peft_model.named_parameters():
            if 'lora_' in name:
                param.data.zero_()
        print("[BaseLLM] 모든 LoRA 어댑터 가중치를 0으로 리셋했습니다.")
    
    def _register_lora_hooks(self):
        self.clear_lora_hooks()

        def save_xL_input_hook(module, args):
            last_token_input = args[0][:, -1, :]    # (Batch, Hidden)
            self._lora_xL_inputs_queue[module].append(last_token_input.detach().clone())

        def inject_delta_output_hook(module, input, output):
            global GLOBAL_INJECTED_LORA_OUTPUT_DELTA
            if GLOBAL_INJECTED_LORA_OUTPUT_DELTA is not None:
                try:
                    # 델타를 출력 Dtype에 맞춤
                    delta_to_add = GLOBAL_INJECTED_LORA_OUTPUT_DELTA.to(output.dtype)

                    # 출력의 마지막 토큰에만 델타를 더함
                    output[:, -1, :] = output[:, -1, :] + delta_to_add
                    return output
                except Exception as e:
                    print(f"WARN: LoRA 델타 주입 실패")
            return output
        
        for name, module in self._peft_model.named_modules():
            if isinstance(module, LoraLayer):
                if REPRESENTATIVE_LORA_TARGET_MODULE in name:
                    hook = module.register_forward_pre_hook(save_xL_input_hook)
                    self._xL_pre_hooks.append(hook)

                if any(target in name for target in LORA_INJECTION_MODULES):
                    hook = module.register_forward_hook(inject_delta_output_hook)
                    self._output_post_hooks.append(hook)

    def clear_lora_hooks(self):
        for hook in self._xL_pre_hooks: hook.remove()
        for hook in self._output_post_hooks: hook.remove()
        self._xL_pre_hooks.clear()
        self._output_post_hooks.clear()
        self._lora_xL_inputs_queue.clear()

    def get_lora_xL_input(self) -> torch.Tensor:
        first_module_queue = next(iter(self._lora_xL_inputs_queue.values()))
        if not first_module_queue:
            raise RuntimeError(f"LoRA XL 입력이 캡처되지 않았습니다.")
        return first_module_queue.popleft() # (Batch, Hidden) 반환
    
    def set_global_lora_output_delta(self, delta: torch.Tensor):
        global GLOBAL_INJECTED_LORA_OUTPUT_DELTA
        GLOBAL_INJECTED_LORA_OUTPUT_DELTA = delta

    def clear_global_lora_output_delta(self):
        global GLOBAL_INJECTED_LORA_OUTPUT_DELTA
        GLOBAL_INJECTED_LORA_OUTPUT_DELTA = None

    @property
    def tokenizer(self): return self._tokenizer
    @property
    def peft_model(self): return self._peft_model
    @property
    def hidden_size(self) -> int: return self._hidden_size
    @property
    def eos_token_id(self) -> int: return self._eos_token_id

    def get_lm_head_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """LM Head 가중치 및 편향 추출 (float32로 → 연산 안정성)"""
        lm_head_weight = self._peft_model.base_model.model.lm_head.weight.data.to(torch.float32).to(DEVICE)
        lm_head_bias = getattr(self._peft_model.base_model.model.lm_head, "bias", None)
        if lm_head_bias is None:
            lm_head_bias = torch.zeros(lm_head_weight.shape[0], dtype=torch.float32, device=DEVICE)
        else:
            lm_head_bias = lm_head_bias.to(torch.float32).to(DEVICE)
        return lm_head_weight, lm_head_bias