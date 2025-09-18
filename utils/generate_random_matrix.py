import numpy as np

# A 행렬용

def generate_random_matrix(rows, cols, std=0.02, dtype=np.float32):
    """
    num_samples: 생성할 샘플 개수, 기본값은 NUM_SAMPLES
    dim: 각 벡터의 차원, 기본값은 DIM
    0으로 채워진 NumPy 행렬 생성
    """
    return np.random.normal(0, std, (rows, cols)).astype(dtype)

if __name__ == "__main__":
    rand_mat = generate_random_matrix()
    print("생성된 랜덤(Gaussian) 행렬 shape:", rand_mat.shape)