import torch
from common.config import DEVICE
from .base_llm import BaseLLMLoader


class LLMPostProcessor:
    def __init__(self, llm_loader: BaseLLMLoader):
        self.tokenizer = llm_loader.tokenizer
        self.lm_head_weight, self.lm_head_bias = llm_loader.get_lm_head_weights()
        self.eos_token_id = llm_loader.eos_token_id

        self.final_ln = llm_loader._find_final_layernorm(llm_loader.peft_model.base_model)
        if self.final_ln is not None:
            self.final_ln = self.final_ln.to(self.lm_head_weight.device)

    def integrate_lora_delta_and_predict_token(self, hidden):
        h = hidden.to(torch.float32).to(self.lm_head_weight.device)

        if self.final_ln is not None:
            try:
                h = self.final_ln(h)
            except Exception as e:
                print("[PostProcessor] WARN ln_f:", e)

        logits = h @ self.lm_head_weight.T + self.lm_head_bias
        next_id = torch.argmax(logits, dim=-1).item()
        next_char = self.tokenizer.decode([next_id], skip_special_tokens=False)
        return next_id, next_char

    def decode_final_output(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
