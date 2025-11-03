import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

from common.config import LLM_NAME, BNB_COMPUTE_DTYPE, DEVICE, HF_CACHE_DIR
from common.lora_utils import get_bnb_config

class BaseLLMLoader:
    def __init__(self):
        self.tokenizer = None
        self.base_model = None
        self.hidden_size = None
        self.eos_token_id = None

    def load_model(self):
        """
        LLM 토크나이저와 Base 모델 로드 및 초기화
        모델에 4비트 양자화 적용 ← GPU 메모리 효율성 향상
        """

        # Tokenizer: LLM 텍스트-토큰 변환
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, cache_dir=HF_CACHE_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # DeepSeek Coder 템플릿 설정
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role']" \
            "== 'user' %}### Instruction:\n{{ message['content] }}\n{% elif message['role'] == 'assistant'" \
            "%}### Response:\n{{ message['content'] }}\n{% endif %}{% endfor %}"
        print(f"[Client-Backend][BaseLLMLoader] 토크나이저 로드 완료: {LLM_NAME}")

        # Base Model
        bnb_config = get_bnb_config()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=BNB_COMPUTE_DTYPE,
            cache_dir=HF_CACHE_DIR
        ).to(DEVICE)
        self.base_model.eval() # 모델을 추론 모드로 설정
        print(f"[Client-Backend][BaseLLMLoader] Base LLM 로드 완료 (4비트 양자화, {DEVICE} 사용)")

        # Model Info
        self.hidden_size = self.base_model.config.hidden_size
        self.eos_token_id = self.tokenizer.eos_token_id

        print(f"[Client-Backend][BaseLLMLoader] LLM Hidden Size: {self.hidden_size}")
        print(f"[Client-Backend][BaseLLMLoader] EOS Token ID: {self.eos_token_id}")

        gc.collect()
        torch.cuda.empty_cache()
    
    def get_lm_head_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LLM의 lm_head 가중치와 편향을 float32 Tensor로 반환 ← LM head 연산에 사용
        """
        if self.base_model is None:
            raise RuntimeError("Base model not loaded. Call load_model() first.")
        
        lm_head_weight = self.base_model.lm_head.weight.data.to(torch.float32) # shape (vocab_size, hidden_size)
        if self.base_model.lm_head.bias is not None:
            lm_head_bias = self.base_model.lm_head.bias.data.to(torch.float32) # shape (vocab_size,)
        else:
            lm_head_bias = torch.zeros(lm_head_weight.shape[0], dtype=torch.float32, device=DEVICE)

        return lm_head_weight, lm_head_bias