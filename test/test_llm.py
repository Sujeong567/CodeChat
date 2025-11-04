import torch
import gc

from common.config import MAX_GEN_LENGTH, DEVICE
from model.base_llm import BaseLLMLoader
from model.preprocessing import LLMPreprocessor
from model.postprocessing import LLMPostprocessor

class LLMService:
    def __init__(self):
        """
        Base LLM 서비스 초기화
        """
        self.llm_loader = BaseLLMLoader()
        self.llm_loader.load_model()
        self.preprocessor = LLMPreprocessor(self.llm_loader)
        self.postprocessor = LLMPostprocessor(self.llm_loader)
        self.max_gen_length = MAX_GEN_LENGTH
        self.device = DEVICE

        print("[Client-Backend][LLMService] 서비스 초기화 완료.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_text(self, prompt: str, temperature: float = 1.0, top_k: int = 50) -> str:
        """
        사용자 프롬프트에 대해 Base LLM 실행
        내부에서 preprocessing → LoRA 통합 → 토큰 생성 → 최종 텍스트 반환
        """
        generated_ids = []

        # --- 1. 초기 상태 가져오기 ---
        llm_states = self.preprocessor.get_initial_states(prompt)
        generated_ids.extend(llm_states["generated_ids"])

        # --- 2. 토큰별 생성 루프 ---
        for step in range(self.max_gen_length):
            # HE/LoRA 연산 후 다음 토큰 예측
            # 여기서는 dummy_lora_delta 사용 예시, 실제 서비스에서는 HE 연산 결과로 대체
            dummy_lora_delta = torch.zeros_like(
                llm_states["current_llm_hidden_state"], dtype=torch.float32, device=self.device
            )
            
            next_token_id, next_token_char, final_logits = self.postprocessor.integrate_lora_delta_and_predict_token(
                dec_lora_delta=dummy_lora_delta,
                current_llm_hidden_state=llm_states["current_llm_hidden_state"],
                temperature=temperature,
                top_k=top_k
            )

            # LLM 상태 업데이트 (다음 루프를 위한 전처리)
            llm_states = self.preprocessor.get_next_token_states(next_token_id, llm_states)
            
            # 생성된 토큰 ID를 전체 리스트에 추가
            generated_ids.append(next_token_id)

            # EOS 도달 시 종료
            if next_token_id == self.llm_loader.eos_token_id:
                break

        # --- 3. 최종 텍스트 디코딩 ---
        final_generated_text = self.postprocessor.decode_final_output(generated_ids)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_generated_text


# ============================
# 실행 코드 (테스트용)
# ============================
if __name__ == "__main__":
    service = LLMService()
    prompt = "Write a Python function that returns the factorial of a number."
    result = service.generate_text(prompt)
    print("===== Generated Output =====")
    print(result)
