import numpy as np

# B 행렬용

def generate_zero_matrix(rows, cols, dtype=np.float32):
    """
    num_samples: 행 개수
    dim: 열 개수
    dtype: 데이터 타입 (기본 np.float32)
    0으로 채워진 NumPy 행렬 생성
    """
    return np.zeros((rows, cols), dtype=dtype)

if __name__ == "__main__":
    zero_mat = generate_zero_matrix()
    print("생성된 0(Zero) 행렬 shape:", zero_mat.shape)