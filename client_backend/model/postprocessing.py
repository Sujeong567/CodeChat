# codechat/client-backend/model/postprocessing.py
import torch
import torch.nn.functional as F

from .base_llm import BaseLLMLoader
from common.config import DEVICE

class LLMPostProcessor:
    """
    델타 주입, 토큰 예측, 최종 디코딩
    - float16 LayerNorm을 float32로 캐스팅
    """
    def __init__(self, llm_loader: BaseLLMLoader):
        self.tokenizer = llm_loader._tokenizer
        self.lm_head_weight, self.lm_head_bias = llm_loader.get_lm_head_weights()
        self.eos_token_id = llm_loader.eos_token_id
        self.device = DEVICE

        self.final_ln_float32 = llm_loader._find_final_layernorm(llm_loader.peft_model.base_model)
        if self.final_ln_float32 is None:
            print("[PostProcessor] WARN: final LayerNorm을 찾지 못했습니다.")

    def integrate_lora_delta_and_predict_token(
        self,
        current_llm_hidden_state: torch.Tensor, # (bfloat16)
    ) -> tuple[int, str]:
        
        # 1. 델타 주입 (bfloat16 -> float32) + (float32) = (float32)
        normed = current_llm_hidden_state.to(torch.float32).to(self.device)
        
        if self.final_ln_float32 is not None:
            try:
                normed = self.final_ln_float32(normed)
            except Exception as e:
                print(f"WARN: final LayerNorm 적용 실패: {e}")

        final_logits = normed @ self.lm_head_weight.T + self.lm_head_bias

        next_token_id = torch.argmax(final_logits, dim=-1).item()

        next_token_char = self.tokenizer.decode([next_token_id], skip_special_token=False)
        return next_token_id, next_token_char
 
        # 4. Top-k 샘플링
        """
        생성된 final_logits 중 다음 토큰 1개 선택
        - Top-k (확률적 샘플링)
        - argmax (결정론적 샘플링)
            # next_token_index = torch.argmax(probabilities, dim=-1).item()
            # next_token_char = tokenizer.decode([next_token_index], skip_special_tokens=False)
        
        TODO: 코드 생성 후 EOS 대신 brer...이 선택되어 랜덤값이 출력되는 것으로 보임
                → argmax()로 변경해보기
        
        logits_for_softmax = final_logits / float(max(1e-6, temperature))
        probs = F.softmax(logits_for_softmax.unsqueeze(0), dim=-1)
        
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
            topk_idx = topk_idx[0]
            topk_vals = topk_vals[0]
            topk_probs = topk_vals / topk_vals.sum()
            sampled_idx = torch.multinomial(topk_probs, num_samples=1).item()
            next_token_id = int(topk_idx[sampled_idx].item())
        else:
            next_token_id = torch.multinomial(probs[0], num_samples=1).item()

        next_token_char = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
        
        return next_token_id, next_token_char
        """

    def decode_final_output(self, generated_ids: list) -> str:
        """디코딩"""
        final_output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_start_tag = "### Response:\n"
        if response_start_tag in final_output_text:
            return final_output_text.split(response_start_tag)[1].strip()
        return final_output_text