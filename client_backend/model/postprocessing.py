# client_backend/model/postprocessing.py
import torch

from common.config import DEVICE
from .base_llm import BaseLLMLoader

class LLMPostProcessor:
    def __init__(self, llm_loader: BaseLLMLoader):
        self.tokenizer = llm_loader.tokenizer
        self.lm_head_weight, self.lm_head_bias = llm_loader.get_lm_head_weights()
        self.eos_token_id = llm_loader.eos_token_id

        self.target_device = self.lm_head_weight.device
        self.final_ln = llm_loader._find_final_layernorm(llm_loader.peft_model.base_model)
        if self.final_ln is not None:
            self.final_ln = self.final_ln.to(self.target_device)
        else:
            print("[PostProcessor] WARN: final LayerNorm을 찾지 못했습니다.")

    def integrate_lora_delta_and_predict_token(
        self,
        current_llm_hidden_state: torch.Tensor,
    ):
        # hidden -> float32 -> final_ln -> lm_head -> argmax
        h = current_llm_hidden_state.to(torch.float32).to(self.target_device)

        if self.final_ln is not None:
            try:
                h = self.final_ln(h)
            except Exception as e:
                print(f"[PostProcessor] WARN: final LayerNorm 적용 실패: {e}")

        logits = h @ self.lm_head_weight.T + self.lm_head_bias
        next_token_id = torch.argmax(logits, dim=-1).item()
        next_token_char = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
        return next_token_id, next_token_char

    def decode_final_output(self, generated_ids: list) -> str:
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text
