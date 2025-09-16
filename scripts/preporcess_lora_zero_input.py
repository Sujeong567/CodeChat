import os
import numpy as np
from utils.generate_zero_matrix import generate_zero_matrix, NUM_SAMPLES, DIM

# 현재 파일(__file__) 기준 프로젝트 루트 잡기 (scripts 상위 폴더)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def preprocess_lora_input(num_samples=NUM_SAMPLES, dim=DIM, flatten=False):
    zero_mat = generate_zero_matrix(num_samples, dim)
    print(f"원본 0행렬 shape: {zero_mat.shape}")

    if flatten:
        zero_mat = zero_mat.flatten()
        print(f"평탄화된 1D 배열 shape: {zero_mat.shape}")

    # 프로젝트 루트 기준 경로에 저장
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, "lora_zero_input.npy")
    np.save(save_path, zero_mat)
    print(f"LoRA 입력용 0행렬 데이터 저장 완료: {save_path}")

if __name__ == "__main__":
    preprocess_lora_input(flatten=True)

