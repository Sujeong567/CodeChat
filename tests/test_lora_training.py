import numpy as np
import tenseal as ts

from model.lora_he_layer import encrypt_lora_input, apply_he_lora
from model.fhe_ckks_local import create_context

def test_lora_pipeline():
    print("\n--- FHE LoRA 전체 파이프라인 테스트 시작 ---")
    
    # 1. 테스트 입력 벡터 불러오기 (없으면 무작위 벡터 생성)
    try:
        x = np.load("data/processed/lora_zero_input.npy")
        if x.ndim > 1:
            print(f"로드된 입력 shape={x.shape}, 첫 샘플 사용")
            x = x[0]
    except:
        print("입력 벡터 파일 없음: 무작위 입력 생성")
        x = np.random.normal(size=768).astype(np.float32)
    print("입력 벡터 shape:", x.shape)
    
    # 2. Encryption context 생성
    context = create_context()
    
    # 3. 암호화 입력 생성
    enc_x = encrypt_lora_input(x, context)
    assert isinstance(enc_x, ts.CKKSVector)
    print("입력 암호화 완료.")
    
    # 4. 저랭크 행렬 A, B 생성 (예시: r=8, m=4)
    n = x.shape[0]
    r = 8
    m = 4
    A = np.random.normal(0, 0.02, (r, n)).astype(np.float32)
    B = np.random.normal(0, 0.02, (m, r)).astype(np.float32)
    print(f"LoRA 저랭크 행렬 생성: A={A.shape}, B={B.shape}")
    
    # 5. FHE 저랭크 곱셈 pipeline (LoRA)
    enc_out = apply_he_lora(enc_x, A, B, context)
    print(f"FHE Encrypted Output 벡터 개수: {len(enc_out)} (m={m})")

    # 6. 복호화 & 결과 비교
    out_dec = []
    for i, v in enumerate(enc_out):
        dec = v.decrypt()
        out_dec.append(float(dec[0]) if isinstance(dec, (list, np.ndarray)) else float(dec))
    out_dec = np.array(out_dec)
    print("복호화된 FHE 결과 (앞 5개):", out_dec[:5])
    
    # 7. 평문 LoRA 결과 계산 및 비교
    lora_out = B @ (A @ x)
    print("평문 LoRA 결과 (앞 5개):", lora_out[:5])
    max_err = np.max(np.abs(out_dec - lora_out))
    print("최대 오차:", max_err)
    if np.allclose(out_dec, lora_out, atol=0.1):
        print("테스트 통과: 복호화 결과와 평문 연산이 허용 오차 내에서 일치합니다.")
    else:
        print("경고: FHE 복호화 결과와 평문 결과가 다릅니다.")

    print("--- FHE LoRA 전체 파이프라인 테스트 완료 ---")

if __name__ == "__main__":
    test_lora_pipeline()
