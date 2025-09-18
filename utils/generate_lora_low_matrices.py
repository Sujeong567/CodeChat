import numpy as np
from .generate_random_matrix import generate_random_matrix
from .generate_zero_matrix import generate_zero_matrix


def generate_lora_low_rank_matrices(d, r, std=0.02):
    """
    LoRA 방식 저랭크 행렬 A (Gaussian random), B (zero) 생성
    d: 입력 차원
    r: rank(저랭크 차원)
    std: A행렬 표준편차 (default 0.02)
    """
    A = generate_random_matrix(d, r, std)  
    B = generate_zero_matrix(r, d, dtype=np.float32)
   
    return A, B

if __name__ == "__main__":
    A, B = generate_lora_low_rank_matrices(768, 8)  # 예시: dim=768, rank=8
    print("생성된 A(Gaussian) shape:", A.shape)
    print("생성된 B(zero) shape:", B.shape)