# tests/test_lora_he_layer.py
import numpy as np
from model.lora_he_layer import apply_he_lora
from utils.generate_lora_low_matrices import generate_lora_low_rank_matrices 

def test_apply_he_lora():
    plain_input = np.array([0.5, 1.0, -0.5, 2.0, 1.5])
    np.random.seed(42)
    lora_A, lora_B = generate_lora_low_rank_matrices(5, 3)
    print("LoRA 가중치 A:", lora_A)
    print("LoRA 가중치 B:", lora_B)

    #ckks_maumul -> 행렬 곱 차원 체크 
    lora_A = lora_A.reshape(5, 3)
    lora_B = lora_B.reshape(3, 5)

    alpha = 0.5

    result = apply_he_lora(plain_input, lora_A, lora_B, alpha=alpha)

    print("평문 LoRA 적용 결과:(enc_out)", result)

if __name__ == "__main__":
    test_apply_he_lora()

