import numpy as np

# 전역변수 선언
NUM_SAMPLES = 32
DIM = 768

def generate_zero_matrix(num_samples=NUM_SAMPLES, dim=DIM):
    """
    num_samples: 생성할 샘플 개수, 기본값은 NUM_SAMPLES
    dim: 각 벡터의 차원, 기본값은 DIM
    0으로 채워진 NumPy 행렬 생성
    """
    return np.zeros((num_samples, dim), dtype=np.float32)

if __name__ == "__main__":
    zero_mat = generate_zero_matrix()
    print("생성된 0행렬 shape:", zero_mat.shape)