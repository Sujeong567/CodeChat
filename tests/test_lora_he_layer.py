import os
import numpy as np
import tenseal as ts
from model.lora_he_layer import encrypt_lora_input
from model.fhe_ckks_local import create_context

# 프로젝트 루트 디렉터리
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCA_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "lora_zero_input.npy")

def test_lora_encryption_function_only():
    print("\n--- encrypt_lora_input 함수 단독 테스트 시작 ---")

    try:
        test_input_vector = np.load(LOCA_INPUT_PATH)
    except FileNotFoundError:
        print(f"오류: '{LOCA_INPUT_PATH}' 파일을 찾을 수 없습니다.")
        print("테스트를 실행하기 전에 'preprocess_lora_zero_input.py' 스크립트를 먼저 실행하여")
        print("해당 파일을 생성해주세요.")
        return

    if test_input_vector.ndim > 1:
        print(f"경고: 로드된 데이터가 2차원({test_input_vector.shape})입니다. 첫 번째 샘플을 사용합니다.")
        test_input_vector = test_input_vector[0]
    
    print(f"lora_input.npy 에서 로드된 입력 벡터 (길이 {len(test_input_vector)}):")
    print(f"앞 부분: {test_input_vector[:5]}...")
    if len(test_input_vector) > 5:
        print(f"뒤 부분: ...{test_input_vector[-5:]}")

    # 테스트용 Context 객체 생성 (비밀키 포함)
    encryption_context = create_context() # <- 여기서 Context를 생성합니다.

    print("\nencrypt_lora_input 함수 호출 중...")
    # 생성된 Context 객체를 encrypt_lora_input 함수에 전달합니다.
    encrypted_vec = encrypt_lora_input(test_input_vector, encryption_context)
    print("encrypt_lora_input 함수 호출 완료.")

    assert isinstance(encrypted_vec, ts.CKKSVector), "암호화 결과가 TenSEAL CKKSVector 객체가 아닙니다!"
    print(f"암호화된 벡터 타입 확인: {type(encrypted_vec)} (올바른 타입)")

    # 복호화 테스트 (Context에 비밀키가 남아있어야 동작)
    try:
        decrypted_vec = encrypted_vec.decrypt() # 암호화 시 사용된 Context로 자동 복호화 시도
        print("\n--- 복호화 테스트 ---")
        print("원본 벡터 (앞 5개):", test_input_vector[:5])
        print("복호화된 벡터 (앞 5개):", decrypted_vec[:5])
        
        if np.allclose(decrypted_vec, test_input_vector, atol=1e-2):
            print("복호화된 데이터가 원본과 허용 오차 범위 내에서 일치합니다.")
        else:
            print("경고: 복호화된 데이터가 원본과 일치하지 않습니다. 차이:", np.max(np.abs(decrypted_vec - test_input_vector)))

    except ValueError as e:
        print(f"\n복호화 중 오류 발생: {e}")
        print("이 오류는 대개 Context에 비밀키가 없을 때 발생합니다.")
        print("model/fhe_ckks_local.py의 create_context()에서 'make_context_public()' 줄을 제거했는지 다시 확인해주세요.")
    except Exception as e:
        print(f"\n복호화 중 예상치 못한 오류 발생: {e}")

    print("\n--- encrypt_lora_input 함수 단독 테스트 완료 ---")
    print("주의: 'WARNING: The input does not fit in a single ciphertext...'와 같은 경고는")
    print("현재 TenSEAL 컨텍스트 설정(poly_modulus_degree)이 입력 데이터 크기보다 작을 때 발생합니다.")
    print("이는 암호화 자체는 성공했을 수 있지만, 나중에 특정 동형 연산이 제한될 수 있음을 의미합니다.")

if __name__ == "__main__":
    test_lora_encryption_function_only()
    