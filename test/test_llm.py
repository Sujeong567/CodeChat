import torch
import gc
import sys
import os
import collections
import time

print("[DEBUG] 1. 스크립트 시작")
time.sleep(0.1) # 출력 버퍼 비우기용

# --- 1. 프로젝트 루트 설정 ---
# 이 스크립트가 있는 루트 디렉토리를 Python 경로에 추가
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
    print(f"[DEBUG] 2. PROJECT_ROOT 설정 완료: {PROJECT_ROOT}")
except Exception as e:
    print(f"[FATAL] PROJECT_ROOT 설정 실패: {e}")
    sys.exit(1)

# --- 2. 테스트 대상 모듈 임포트 ---
# from client_backend.model.base_llm import BaseLLMLoader
# from client_backend.model.preprocessing import LLMPreProcessor
# from client_backend.model.postprocessing import LLMPostProcessor
print("[DEBUG] 3. client_backend 모듈 임포트 시도...")
try:
    from client_backend.model.base_llm import BaseLLMLoader
    print("  - BaseLLMLoader 임포트 성공")
    from client_backend.model.preprocessing import LLMPreProcessor
    print("  - LLMPreProcessor 임포트 성공")
    from client_backend.model.postprocessing import LLMPostProcessor
    print("  - LLMPostProcessor 임포트 성공")
except ImportError as e:
    print(f"\n[FATAL] 모듈 임포트 실패! 폴더 이름이 'client_backend'(언더스코어)인지 확인하세요.")
    print(f"에러 메시지: {e}")
    print(f"현재 sys.path: {sys.path}\n")
    sys.exit(1)
except Exception as e:
    print(f"[FATAL] 예상치 못한 임포트 에러: {e}")
    sys.exit(1)

# --- 3. 의존성 모듈 임포트 (설정값) ---
# from common.config import (
#    MAX_GEN_LENGTH, DEVICE, HF_CACHE_DIR, 
#    LLM_NAME, BNB_COMPUTE_DTYPE, R_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
#    REPRESENTATIVE_LORA_TARGET_MODULE, LORA_INJECTION_MODULES
#)
print("[DEBUG] 4. common 모듈 임포트 시도...")
try:
    import torch
    import gc
    from common.config import (
        MAX_GEN_LENGTH, DEVICE, HF_CACHE_DIR, 
        LLM_NAME, BNB_COMPUTE_DTYPE
    )
    print("  - common 모듈 임포트 성공")
except ImportError as e:
    print(f"[FATAL] common 모듈 임포트 실패: {e}")
    sys.exit(1)

def run_llm_test():
    """
    [LLM 단독 테스트]
    FHE/서버 연동 없이, LoRA 훅 아키텍처(base_llm, preprocessing, postprocessing)가
    '0-델타' 주입 시 정상적으로 작동하는지 테스트합니다.
    """
    print("--- 🚀 LLM 모듈 단독 테스트 시작 ---")
    gc.collect()
    torch.cuda.empty_cache()

    # 1. 모듈 초기화 (테스트 대상)
    print("[Test] 1/3: LLM 로더 (훅 포함) 초기화 중...")
    llm_loader = BaseLLMLoader()
    llm_loader.load_model()
    
    print("[Test] 2/3: 전/후처리기 초기화 중...")
    preprocessor = LLMPreProcessor(llm_loader=llm_loader)
    postprocessor = LLMPostProcessor(llm_loader=llm_loader)
    
    print("[Test] 3/3: 테스트 프롬프트 설정...")
    prompt = "Write a Python function that returns the factorial of a number."
    generated_ids = []

    try:
        # --- 1. 초기 상태 가져오기 ---
        print("\n--- [Test] Step 1: LLM 초기 상태 (xL 포함) 가져오기 ---")
        llm_loader.reset_lora_weights() # LoRA 가중치 0으로 리셋
        llm_states = preprocessor.get_initial_states(prompt)
        generated_ids.extend(llm_states["generated_ids"])
        
        current_llm_hidden_state = llm_states["current_llm_hidden_state"]
        xL_tensor = llm_states["lora_xL_input"] # (Batch, Hidden)

        # --- 2. 토큰별 생성 루프 ---
        for i in range(MAX_GEN_LENGTH):
            print(f"\n--- [Test] Step 2.{i+1}: 토큰 {i+1} 생성 ---")
            
            # --- [0-델타 시뮬레이션] ---
            # 'xL_tensor' (Batch, Hidden)와 동일한 shape의 0-텐서를 생성합니다.
            # 이것이 'FHE 노이즈가 낀 0-델타' (dec_lora_output_delta)를 대체합니다.
            
            print(f"  (3-5) [Sim] 0-델타 생성 (Shape: {xL_tensor.shape})...")
            dummy_delta = torch.zeros_like(xL_tensor).to(DEVICE)
            
            # (5b) 0-델타를 훅에 주입하기 위해 전역 변수에 설정
            llm_loader.set_global_lora_output_delta(dummy_delta)
            
            # --- [클라이언트 로직 실행] ---
            # (6) 다음 토큰 예측
            print("  (6) [Client] 다음 토큰 예측 (델타는 다음 스텝에 주입됨)...")
            next_token_id, next_token_char = postprocessor.integrate_lora_delta_and_predict_token(
                current_llm_hidden_state=current_llm_hidden_state
            )
            
            generated_ids.append(next_token_id)
            print(f"  -> 생성: {repr(next_token_char)}")

            if next_token_id == llm_loader.eos_token_id:
                print("\n  [Test] EOS 토큰 감지. 생성 종료.")
                break
            
            # (7) 상태 업데이트 (이때 'inject_delta_output_hook'이 0-델타를 주입함)
            print("  (7) [Client] 상태 업데이트 (훅을 통해 0-델타 주입)...")
            llm_states = preprocessor.get_next_token_states(next_token_id, llm_states)
            
            # (7b) 다음 루프를 위해 변수 업데이트
            llm_loader.clear_global_lora_output_delta() # 주입 완료 후 델타 초기화
            current_llm_hidden_state = llm_states["current_llm_hidden_state"]
            xL_tensor = llm_states["lora_xL_input"] # 새 xL

        # --- 3. 최종 텍스트 디코딩 ---
        final_generated_text = postprocessor.decode_final_output(generated_ids)

        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n" + "="*30)
        print("    ✅ 최종 생성 결과 (LLM 단독 테스트)")
        print("="*30)
        print(final_generated_text)
        print("="*30)

    except Exception as e:
        print(f"\n[Test] 🚨 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'llm_loader' in locals():
            llm_loader.clear_lora_hooks()
        print("\n--- 🧹 테스트 완료. 리소스 정리 ---")

if __name__ == "__main__":
    run_llm_test()