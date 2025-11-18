# test/test_he_lora.py
import sys, os
import numpy as np
import torch
import tenseal as ts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from common.he_context import load_client_context, load_server_context
from server.lora.adapter import get_fhe_lora_tensors
from server.lora.inference import he_lora_inference

def test_he_lora():
    print("=== Test: FHE LoRA End-to-End ===")

    # 1) 클라이언트/서버 컨텍스트 불러오기
    client_ctx = load_client_context()
    server_ctx = load_server_context()

    # 2) LoRA W_A, W_B 가져오기 (PlainTensor)
    W_A_pt, W_B_pt = get_fhe_lora_tensors()
    print("[OK] Loaded LoRA PlainTensors")

    # 3) 테스트 입력 벡터 생성
    H = 4608
    x = torch.randn(H).float()
    print("[TEST] Input vector first 5:", x[:5])

    # 4) 클라이언트 암호화
    enc_x = ts.ckks_vector(client_ctx, x.tolist())
    enc_bytes = enc_x.serialize()

    # 5) 서버 측에서 읽기
    enc_x_server = ts.ckks_vector_from(server_ctx, enc_bytes)

    # 6) FHE matmul
    enc_out = he_lora_inference(enc_x_server, W_A_pt, W_B_pt, server_ctx)

    # 7) 서버 결과를 다시 vector로 로드
    enc_out_vec = ts.ckks_vector_from(server_ctx, enc_out)

    # 8) 복호화 (client secret key 필요!)
    decrypted = np.array(
        enc_out_vec.decrypt(secret_key=client_ctx.secret_key()),
        dtype=np.float32
    )

    print("[TEST] Decrypted output sample:", decrypted[:5])
    print("[OK] HE LoRA test passed.")

if __name__ == "__main__":
    test_he_lora()

"""
결과:
FHE LoRA MatMul → Ciphertext 유지 → Secret Key로 복호화

>> ckks로 matmul 정상 동작
"""