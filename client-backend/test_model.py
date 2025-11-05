import sys
import os
import torch
import gc

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model.base_llm import BaseLLMLoader

# BaseLLMLoader 클래스 정의 아래에 추가될 코드입니다.

def run_llm_query_test(llm_loader: BaseLLMLoader, query: str):
    """
    로드된 Base LLM을 사용하여 주어진 쿼리에 대한 응답을 생성하고 결과를 출력합니다.
    
    Args:
        llm_loader: 로드된 BaseLLMLoader 인스턴스.
        query: LLM에게 전달할 텍스트 쿼리.
    """
    if llm_loader.base_model is None:
        print("❌ Error: Base model not loaded. Call load_model() first.")
        return

    # 1. 입력 프롬프트 생성
    prompt = f"### Instruction:\n{query}\n\n### Response:"
    print(f"\n--- LLM Test Start ---")
    print(f"**Input Query:** {query}")
    print(f"**Full Prompt Sent to LLM:**\n{prompt}")

    # 2. 토큰화
    # PyTorch 텐서 반환 및 모델 장치로 이동 (CPU 또는 GPU)
    inputs = llm_loader.tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(llm_loader.base_model.device)

    # 3. 모델 추론 (Generation)
    with torch.no_grad():
        output_tokens = llm_loader.base_model.generate(
            **inputs,
            max_new_tokens=150,           # 코드 생성을 위해 토큰 수를 늘립니다.
            num_beams=1,
            do_sample=False,
            pad_token_id=llm_loader.tokenizer.pad_token_id,
            eos_token_id=llm_loader.tokenizer.eos_token_id
        )

    # 4. 결과 디코딩 및 출력
    generated_text = llm_loader.tokenizer.decode(
        output_tokens[0], 
        skip_special_tokens=True
    )
    
    # 생성된 텍스트에서 입력 프롬프트 부분 제거
    # deepseek-coder의 프롬프트 구조(### Instruction: ... ### Response: ...)를 가정하고 응답 부분만 추출
    response = generated_text[len(prompt):].strip()
    
    # 생성된 텍스트 출력
    print(f"\n**LLM Generated Response (Extracted):**")
    print("=" * 50)
    print(response)
    print("=" * 50)
    print("----------------------")


if __name__ == "__main__":
    # 1. LLM 로더 초기화 및 모델 로드
    llm_loader = BaseLLMLoader()
    try:
        print("Starting LLM Model Loading...")
        llm_loader.load_model()
        print("Model Loading Complete.")
    except Exception as e:
        print(f"LLM Load Error: {e}")
        # LLM 로드에 실패하면 테스트를 진행하지 않음
        exit()

    # 2. 코드 생성 테스트 실행
    code_query = "Write a Python function named add that takes two arguments a and b and returns their sum."
    run_llm_query_test(llm_loader, query=code_query)
    
    # 3. 추가적인 정리 작업
    del llm_loader
    gc.collect()
    torch.cuda.empty_cache()
    print("\nTest finished and resources cleaned up.")
