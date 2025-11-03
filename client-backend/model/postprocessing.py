import torch
import torch.nn.functional as F

from common.config import SCALE_FACTOR_FOR_LM_INPUT_INJECTION
from model.base_llm import BaseLLMLoader
   
class LLMPostprocessor:
    def __init__(self, llm_loader: BaseLLMLoader):
        self.llm_loader = llm_loader
        self.lm_head_weight, self.lm_head_bias = self.llm_loader.get_lm_head_weights()
        self.eos_token_id = self.llm_loader.eos_token_id
        self.base_model = self.llm_loader.base_model
        self.tokenizer = self.llm_loader.tokenizer
        self.scale_injection = SCALE_FACTOR_FOR_LM_INPUT_INJECTION
    
    @staticmethod
    def _find_final_layernorm(module: torch.nn.Module):
        """
        모델 내부에서 마지막 LayerNorm 탐색 → LM Head에 로짓을 전달하기 전 적용되는 LayerNorm을 찾음
        """
        candidates = []
        for name, m in module.named_modules():
            n = name.lower()
            if isinstance(m, torch.nn.LayerNorm) or "ln_f" in n or "norm" in n or "layernorm" in n:
                candidates.append((name, m))
        if not candidates:
            return None
        return candidates[-1][1]  # 마지막 candidate 사용
    
    def integrate_lora_delta_and_predict_token(
            self,
            dec_lora_delta: torch.Tensor,
            current_llm_hidden_state: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 50,           
    ) -> tuple[int, str]:
        """
        복호화된 LoRA 델타를 LLM의 lm_head 연산에 통합하여 다음 토큰 예측
        """
        # LoRA delta 통합
        adjusted = current_llm_hidden_state.to(torch.float32) + (dec_lora_delta * self.scale_injection)

        # 선택적 final LayerNorm 적용
        final_ln = self._find_final_layernorm(self.base_model)
        if final_ln is not None:
            try:
                normed = final_ln(adjusted)
            except Exception as e:
                print(f"[Client-Backend][PostProcessor] WARN: final LayerNorm 적용 실패:", e)
                normed = adjusted
        else:
            print(f"[Client-Backend][PostProcessor] WARN: No final LayerNorm found; proceeding without it.")
        
        # LM Head로 로짓 계산
        final_logits = normed @ self.lm_head_weight.T + self.lm_head_bias   # -> (vocab_size,) 형태, float32

        # Softmax + temperature
        logits_for_softmax = final_logits / float(max(1e-6, temperature))
        probs = F.softmax(logits_for_softmax.unsqueeze(0), dim=-1)  # -> (1, vocab_size)
        
        # 로짓 안정화
        # CLIP_VALUE = 5.0
        # final_logits = torch.clamp(logits, -CLIP_VALUE, CLIP_VALUE)

        # toP-K 샘플링
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
            topk_idx = topk_idx[0]
            topk_vals = topk_vals[0]
            # renormalize
            topk_probs = topk_vals / topk_vals.sum()
            # sample
            sampled_idx = torch.multinomial(topk_probs, num_samples=1).item()
            next_token_id = int(topk_idx[sampled_idx].item())
        else:
            next_token_id = torch.multinomial(probs[0], num_sampels=1).item()

        # 토큰 디코딩
        next_token_str = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
        return next_token_id, next_token_str, final_logits
    
    def decode_final_output(self, generated_ids: list) -> str:
        """
        토큰 ID 리스트를 텍스트로 디코딩
        """
        final_output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_start_tag = "### Response:\n"
        if response_start_tag in final_output_text:
            return final_output_text.split(response_start_tag)[1].strip()
        return final_output_text