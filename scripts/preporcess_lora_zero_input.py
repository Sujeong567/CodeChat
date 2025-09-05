import os
import numpy as np
from data.before.generate_zero_matrix import generate_zero_matrix, NUM_SAMPLES, DIM

def preprocess_lora_input(num_samples=NUM_SAMPLES, dim=DIM, flatten=False):
    zero_mat = generate_zero_matrix(num_samples, dim)
    print(f"원본 0행렬 shape: {zero_mat.shape}")

    if flatten:
        zero_mat = zero_mat.flatten()
        print(f"평탄화된 1D 배열 shape: {zero_mat.shape}")

    os.makedirs("data/after", exist_ok=True)
    np.save("data/after/lora_zero_input.npy", zero_mat)
    print("LoRA 입력용 0행렬 데이터 저장 완료: data/after/lora_input.npy")

if __name__ == "__main__":
    preprocess_lora_input(flatten=True)