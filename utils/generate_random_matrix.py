import numpy as np

#A행렬용

# 전역변수 선언
NUM_SAMPLES = 32
DIM = 768

def generate_random_matrix(r, d, std=0.02):
    """
    num_samples: 생성할 샘플 개수, 기본값은 NUM_SAMPLES
    dim: 각 벡터의 차원, 기본값은 DIM
    0으로 채워진 NumPy 행렬 생성
    """
    return np.random.normal(0, std, (r, d)).astype(np.float32)

if __name__ == "__main__":
    zero_mat = generate_random_matrix()
    print("생성된 랜덤(Gaussian) 행렬 shape:", zero_mat.shape)