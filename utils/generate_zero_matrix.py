import numpy as np

# 전역변수 선언
NUM_SAMPLES = 32
DIM = 768

def generate_zero_matrix(num_samples=NUM_SAMPLES, dim=DIM, dtype=np.float32):
    """
    num_samples: 행 개수
    dim: 열 개수
    dtype: 데이터 타입 (기본 np.float32)
    0으로 채워진 NumPy 행렬 생성
    """
    return np.zeros((num_samples, dim), dtype=dtype)

if __name__ == "__main__":
    zero_mat = generate_zero_matrix()
    print("생성된 0행렬 shape:", zero_mat.shape)