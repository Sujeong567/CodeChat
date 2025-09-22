# model/lora_he_layer.py
# model/fhe_ckks_local.py에 정의된 범용 암호화 함수를 임포트합니다.

import tenseal as ts

from model.fhe_ckks_local import ckks_matmul # <- Context 관련 함수는 임포트하지 않습니다.

# LoRA 가중치 적용, 추론 로직 등 이곳에 정의될 수 있습니다.
def apply_he_lora(enc_input, lora_A, lora_B, alpha=1.0):
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
    context = enc_input.context()
    d = lora_B.shape[1]
    enc_out_list = []

    for k in range(d):
        acc = tmp_vecs[0] * float(lora_B[0, k])
        for j in range(1, len(tmp_vecs)):
            acc = acc + (tmp_vecs[j] * float(lora_B[j, k]))
        if alpha != 1.0:
            acc = acc * float(alpha)
        enc_out_list.append(acc)

    # 🔹 리스트를 단일 CKKS 벡터로 변환
    # TenSEAL은 CKKSVector로 바로 합치는 기능은 없지만,
    # 각 스칼라를 decrypt 없이 합치려면 평문 배열로 만드는 대신
    # 그대로 리스트를 이용해 새로운 CKKSVector 생성 (주의: 여기서는 테스트용, 실제 HE 환경에서는 concat 구현 필요)
    # 단순화 예제:
    out_vals = [v.decrypt()[0] for v in enc_out_list]  # ⚠ 실제 환경에서는 decrypt 하지 않음
    out_vec = ts.ckks_vector(context, out_vals)

    return out_vec

if __name__ == "__main__":
    print(apply_he_lora.__code__.co_varnames)



