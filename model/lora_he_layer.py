# model/lora_he_layer.py
# model/fhe_ckks_local.py에 정의된 범용 암호화 함수를 임포트합니다.

import tenseal as ts

from model.fhe_ckks_local import ckks_encrypt, ckks_matmul # <- Context 관련 함수는 임포트하지 않습니다.

def encrypt_lora_input(input_vector, encryption_context):
    """
    LoRA 입력 벡터를 CKKS 방식으로 암호화하는 함수.
    model/fhe_ckks_local.py의 ckks_encrypt 함수를 내부적으로 호출합니다.
    encryption_context를 인자로 받아서 ckks_encrypt에 전달합니다.
    """
    # encryption_context가 ckks_encrypt 함수의 Context 인자와 동일한 역할을 합니다.
    encrypted_vec = ckks_encrypt(input_vector, encryption_context)
    return encrypted_vec

# LoRA 가중치 적용, 추론 로직 등 이곳에 정의될 수 있습니다.
# (예: def apply_he_lora(encrypted_input, encrypted_lora_weights, encryption_context): ...)

import tenseal as ts
import numpy as np
from model.fhe_ckks_local import ckks_matmul

def apply_he_lora(enc_input, lora_A, lora_B, context):
    """
    암호화된 입력(enc_input)에 대해 LoRA 변환을 암호화 상태에서 수행.
    enc_input: ts.ckks_vector (길이 d)
    lora_A: numpy.ndarray (r x d)
    lora_B: numpy.ndarray (m x r)
    context: TenSEAL context
    반환: 길이 m의 리스트 [ts.ckks_vector, ...]
    """

    # 1차 곱셈: enc_input (1 x d) × A^T (d x r) = (1 x r)
    # -> tmp_vecs: 길이 r의 리스트, 각 요소는 암호화된 스칼라(ts.ckks_vector)
    tmp_vecs = ckks_matmul(enc_input, lora_A.T)

    # 2차 곱셈: tmp_vecs (1 x r) × B^T (r x m) = (1 x m)
    out = []
    for j in range(lora_B.shape[0]):  # 출력 차원 m
        # 암호화된 0으로 초기화
        result_sum = ts.ckks_vector(context, [0.0])

        for i, enc_val in enumerate(tmp_vecs):  # r 차원 합산
            # enc_val은 암호화된 스칼라, lora_B[j, i]는 평문 스칼라
            scaled = enc_val * float(lora_B[j, i])  # 암호화 상태에서 스칼라 곱
            result_sum = result_sum + scaled        # 암호화 상태에서 덧셈

        out.append(result_sum)  # 암호화된 스칼라 결과 저장

    return out


