# model/lora_he_layer.py
# model/fhe_ckks_local.py에 정의된 범용 암호화 함수를 임포트합니다.

import tenseal as ts
import numpy as np
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
def apply_he_lora(enc_input, lora_A, lora_B, context, alpha=1.0):
    """
    암호화된 입력(enc_input)에 대해 LoRA 변환을 암호화 상태에서 수행.
    enc_input: ts.ckks_vector (길이 d)
    lora_A: numpy.ndarray (d x r)
    lora_B: numpy.ndarray (r x d)
    context: TenSEAL context
    반환: 길이 d의 리스트 [ts.ckks_vector, ...]
    """

    # 1차 곱셈: enc_input (1 x d) × A (d x r) = (1 x r)
    # -> tmp_vecs: 길이 r의 리스트, 각 요소는 암호화된 스칼라(ts.ckks_vector)
    tmp_vecs = ckks_matmul(enc_input, lora_A)

    # 2차 곱셈: tmp_vecs (1 x r) × B (r x d) = (1 x d)
    enc_out = []
    for k in range(lora_B.shape[1]):  # 출력 차원 d
        acc = tmp_vecs[0] * float(lora_B[0, k])
        for j in range(1, len(tmp_vecs)):
            acc = acc + (tmp_vecs[j] * float(lora_B[j, k]))
        if alpha != 1.0:
            acc = acc * float(alpha)
        enc_out.append(acc)
    return enc_out

        # 암호화된 0으로 초기화
        #result_sum = ts.ckks_vector(context, [0.0])

        #for i, enc_val in enumerate(tmp_vecs):  # r 차원 합산
            # enc_val은 암호화된 스칼라, lora_B[j, i]는 평문 스칼라
            #scaled = enc_val * float(lora_B[j, i])  # 암호화 상태에서 스칼라 곱
            #result_sum = result_sum + scaled        # 암호화 상태에서 덧셈

        #out.append(result_sum)  # 암호화된 스칼라 결과 저장

    #return out


