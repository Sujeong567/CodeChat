# model/lora_he_layer.py
# model/fhe_ckks_local.py에 정의된 범용 암호화 함수를 임포트합니다.
from model.fhe_ckks_local import ckks_encrypt # <- Context 관련 함수는 임포트하지 않습니다.

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