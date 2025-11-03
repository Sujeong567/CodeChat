# deploy_llm.py
import torch
from common.config import MAX_GEN_LENGTH, DEVICE
from model.base_llm import BaseLLMLoader
from model.preprocessing import LLMPreprocessor
from model.postprocessing import LLMPostprocessor

class LLMService:
    def __init__(self):
        self.llm_loader = BaseLLMLoader()
        self.llm_loader.load_model()
        self.preprocessor = LLMPreprocessor(self.llm_loader)
        self.postprocessor = LLMPostprocessor(self.llm_loader)
        self.max_gen_length = MAX_GEN_LENGTH
        self.device = DEVICE

    def generate_text(self, prompt: str, temperature: float = 1.0, top_k: int = 50) -> str:
        """
        배포용 텍스트 생성 함수.
        내부에서 preprocessing → LoRA 통합 → 토큰 생성 → 최종 텍스트 반환
        """
        # --- 1. 초기 상태 가져오기 ---
        states = self.preprocessor.get_initial_states(prompt)
        current_state = states["current_llm_hidden_state"]
        generated_ids = states["generated_ids"].copy()

        # --- 2. 토큰별 생성 루프 ---
        for step in range(self.max_gen_length):
            # HE/LoRA 연산 후 다음 토큰 예측
            # 여기서는 dummy_lora_delta 사용 예시, 실제 서비스에서는 HE 연산 결과로 대체
            dummy_lora_delta = torch.zeros_like(current_state, dtype=torch.float32)
            
            next_token_id, _, _ = self.postprocessor.integrate_lora_delta_and_predict_token(
                dec_lora_delta=dummy_lora_delta,
                current_llm_hidden_state=current_state,
                temperature=temperature,
                top_k=top_k
            )

            # 상태 업데이트
            states = self.preprocessor.get_next_token_states(next_token_id, states)
            current_state = states["current_llm_hidden_state"]
            generated_ids.append(next_token_id)

            # EOS 도달 시 종료
            if next_token_id == self.llm_loader.eos_token_id:
                break

        # --- 3. 최종 텍스트 디코딩 ---
        final_text = self.postprocessor.decode_final_output(generated_ids)
        return final_text


# ============================
# 실행 코드
# ============================
if __name__ == "__main__":
    service = LLMService()
    prompt = "Write a Python function that returns the factorial of a number."
    result = service.generate_text(prompt)
    print("===== Generated Output =====")
    print(result)
