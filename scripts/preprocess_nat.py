import os
import glob
import numpy as np
from transformers import AutoTokenizer # <--- 핵심 라이브러리

# 프로젝트 상대 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "dec") 

# LoRA 학습에 사용된 LLM 모델 이름 (예시)
# LoRA 어댑터가 부착된 기본 모델과 동일한 토크나이저를 사용해야 합니다.
LLM_MODEL_NAME = "distilbert-base-uncased" # 또는 "llama/llama-2-7b", "mistralai/Mistral-7B-v0.1" 등

# ====================================================================
# 1. Softmax 및 토크나이저 로딩 헬퍼 함수
# ====================================================================

def softmax(x):
    """Softmax 함수 계산"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 토크나이저를 로드하는 함수
def load_llm_tokenizer(model_name):
    """지정된 모델 이름으로 Hugging Face 토크나이저를 로드하고 어휘 크기를 반환"""
    print(f"Loading tokenizer for: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return None

# ====================================================================
# 2. 메인 디코딩 로직
# ====================================================================
def main(input_dir=DEC_OUTPUT_DIR, model_name=LLM_MODEL_NAME):
    
    # 1. 토크나이저 로드 (가장 먼저 수행)
    tokenizer = load_llm_tokenizer(model_name)
    if not tokenizer:
        return
    
    VOCAB_SIZE = tokenizer.vocab_size
    print(f"Tokenizer loaded. Actual Vocabulary Size: {VOCAB_SIZE}")

    print("\n--- 3. Applying Softmax and Decoding ---")
    
    decrypted_files = sorted(glob.glob(os.path.join(input_dir, "dummy_code_ckks_enc_*_lora_out_*.txt")))
    
    if len(decrypted_files) == 0:
        print(f"Error: No decrypted Logits files (*.txt) found in {input_dir}.")
        return

    for idx, logits_path in enumerate(decrypted_files):
        filename = os.path.basename(logits_path)
        print(f"\n[{idx+1}/{len(decrypted_files)}] Processing {filename}...")
        
        try:
            # 1. Logits 벡터 로드
            logits_vector = np.loadtxt(logits_path, dtype=np.float64)
            
            # (중요) Logits 벡터 크기를 실제 LLM의 어휘 크기로 조정 (필요 시)
            if len(logits_vector) > VOCAB_SIZE:
                 logits_vector = logits_vector[:VOCAB_SIZE]
            elif len(logits_vector) < VOCAB_SIZE:
                 # 벡터가 어휘 크기보다 작다면 에러 또는 패딩 로직 필요 (CKKS 연산 설계 확인 필요)
                 print("Warning: Logits vector size is smaller than Vocabulary size.")

            
            # 2. Softmax 연산 적용 (확률 분포 생성)
            probability_distribution = softmax(logits_vector)
            
            # 3. 토큰 디코딩 (가장 높은 확률의 인덱스 선택)
            best_token_index = np.argmax(probability_distribution)
            max_probability = probability_distribution[best_token_index]
            
            # 4. 실제 토크나이저를 사용하여 인덱스를 문자열로 변환
            selected_token_string = tokenizer.decode(best_token_index)
            
            # 5. 결과 출력
            print("  > Decoding successful.")
            print(f"  > **Predicted Next Token:** '{selected_token_string.strip()}'")
            print(f"  > Token ID (Logits Index): {best_token_index}")
            print(f"  > Probability: {max_probability:.6f}")
            
        except Exception as e:
            print(f"  > FAILED to process {filename}: {e}")
            continue

    print("\n--- LLM Tokenizer Decoding Complete ---")


if __name__ == "__main__":
    main()