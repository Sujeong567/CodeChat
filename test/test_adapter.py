# test/test_adapter.py
import sys
import os
import tenseal as ts

# 프로젝트 루트 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from server.lora.adapter import get_fhe_lora_tensors

def main():
    print("=== LoRA 서버측 가중치 로딩 테스트 ===")

    W_A, W_B = get_fhe_lora_tensors()

    print("W_A type:", type(W_A))
    print("W_B type:", type(W_B))

    # 1) PlainTensor -> Python list 변환
    W_A_list = W_A.tolist()
    W_B_list = W_B.tolist()

    # 2) shape 추론 (row, col)
    rows_A = len(W_A_list)
    cols_A = len(W_A_list[0]) if rows_A > 0 else 0

    rows_B = len(W_B_list)
    cols_B = len(W_B_list[0]) if rows_B > 0 else 0

    print(f"W_A shape (rows, cols): ({rows_A}, {cols_A})")
    print(f"W_B shape (rows, cols): ({rows_B}, {cols_B})")

    # 3) 일부 값 샘플 확인
    if rows_A > 0 and cols_A > 0:
        print("W_A sample row 0 first 5:", W_A_list[0][:5])
    else:
        print("W_A is empty")

    if rows_B > 0 and cols_B > 0:
        print("W_B sample row 0 first 5:", W_B_list[0][:5])
    else:
        print("W_B is empty")

    print("\n[OK] LoRA 가중치 로딩 및 변환 성공!")

if __name__ == "__main__":
    main()

"""
결과:
(1) LoRA A/B 가중치 Torch Shape 정상
W_A: torch.Size([16, 4608]) (r, hidden)
W_B: torch.Size([4608, 16]) (hidden, r)

(2) TenSEAL 변환 후 PlainTensor Shape
W_A shape (rows, cols): (4608, 16)     # W_A.T
W_B shape (rows, cols): (16, 4608)     # W_B.T
    >> client가 보낸 xL shape → (4608,)
    >> server 계산:
        · xL(1x4608) x W_A_pt(4608x16) → (1x16)
        · (1x16) x W_B_pt(16x4608) → (1x4608)

(3) 값 샘플도 정상적인 LoRA 값
W_A sample row 0 first 5: [-0.0052, -0.0084, 0.01087, -0.00288, 0.01273]
W_B sample row 0 first 5: [0.006937, -0.00546, 0.00418, -0.00596, 0.00599]
    >> 학습된 LoRA weights가 float32로 나옴
"""