# model/fhe_ckks_local.py
import tenseal as ts

def create_context():
    """
    TenSEAL CKKS Context를 생성하는 함수.
    이 컨텍스트는 비밀키를 포함하여 암호화 및 복호화에 사용될 수 있습니다.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    # 주의: make_context_public() 호출을 제거하여 비밀키가 컨텍스트에 유지되도록 합니다.
    # context.make_context_public()
    return context

def ckks_encrypt(input_vector, encryption_context):
    """
    주어진 encryption_context를 사용하여 input_vector를 CKKS 방식으로 암호화합니다.
    encryption_context는 비밀키를 포함한 Context 객체여야 합니다.
    """
    # ts.ckks_vector의 첫 번째 인자는 Context 객체입니다.
    enc_vec = ts.ckks_vector(encryption_context, input_vector)
    return enc_vec

def ckks_decrypt(encrypted_vector):
    """
    암호화된 벡터를 복호화합니다.
    복호화는 암호화 시 사용된 Context에 비밀키가 있어야만 가능합니다.
    """
    return encrypted_vector.decrypt()