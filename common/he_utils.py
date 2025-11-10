# common/he_utils.py

import tenseal as ts
import os

from common.config import (
    HE_POLY_MODULUS_DEGREE, HE_COEFF_MOD_BIT_SIZES, HE_GLOBAL_SCALE_BITS,
    CLIENT_SECRET_KEY_PATH, SERVER_PUBLIC_KEY_PATH, HE_KEY_DIR
)

def create_ckks_context() -> ts.Context:
    """
    CKKS Context 생성 (설정값은 config.py에서 가져옴)
    """
    print("CKKS Context 생성 중...")
    
    # 1. Context 생성
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=HE_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=HE_COEFF_MOD_BIT_SIZES
    )
    
    # 2. Global Scale 설정
    context.global_scale = 2 ** HE_GLOBAL_SCALE_BITS
    
    # 3. Galois Keys 생성
    context.generate_galois_keys()
    
    # 4. Relin Keys 생성
    context.generate_relin_keys()
    
    print("CKKS Context 생성 완료.")
    
    return context

def get_or_create_he_keys():
    """키 파일이 없으면 생성하고, 있으면 로드하여 클라이언트와 서버 컨텍스트 반환"""
    os.makedirs(HE_KEY_DIR, exist_ke=True)

    if os.path.exists(CLIENT_SECRET_KEY_PATH) and os.path.exists(SERVER_PUBLIC_KEY_PATH):
        # 키 파일이 이미 존재하는 경우
        print("[HEUtils] 기존 HE 키 로드 중...")
        with open(CLIENT_SECRET_KEY_PATH, "rb") as f:
            client_ctx = ts.context_from(f.read())
        with open(SERVER_PUBLIC_KEY_PATH, "rb") as f:
            server_ctx = ts.context_from(f.read())
        print("[HEUtils] HE 키 로드 완료.")
        return client_ctx, server_ctx
    
    # 키 파일이 없는 경우
    print("[HEUtils] HE 키 생성 중...")
    client_ctx = create_ckks_context()

    # 서버용 공개 컨텍스트 생성(비밀키 제외)
    server_ctx_bytes = client_ctx.serialize(save_secret_key=False)
    server_ctx = ts.context_from(server_ctx_bytes)

    # 클라이언트용 비밀 컨텍스트 저장
    with open(CLIENT_SECRET_KEY_PATH, "wb") as f:
        f.write(client_ctx.serialize(save_secret_key=True))

    # 서버용 공개 컨텍스트 저장
    with open(SERVER_PUBLIC_KEY_PATH, "wb") as f:
        f.write(server_ctx_bytes)

    print(f"[HEUtils] HE 키 생성 및 저장 완료: {HE_KEY_DIR}")
    return client_ctx, server_ctx

def load_client_context() -> ts.Context:
    """클라이언트가 사용할 컨텍스트 로드"""
    client_ctx, _ = get_or_create_he_keys()
    return client_ctx

def load_server_context() -> ts.Context:
    """서버가 사용할 컨텍스트 로드"""
    _, server_ctx = get_or_create_he_keys()
    return server_ctx

"""
def save_tenseal_context(context: ts.Context, filename: str, save_secret_key: bool):
    '''TenSEAL Context를 파일로 저장합니다.'''
    filepath = os.path.join(HE_KEY_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(context.serialize(save_secret_key=save_secret_key))
    print(f"[HEUtils] Context 저장 완료: {filepath} (Secret Key: {save_secret_key})")

def load_tenseal_context(filename: str) -> ts.Context:
    '''파일로부터 TenSEAL Context를 로드합니다.'''
    filepath = os.path.join(HE_KEY_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Context 파일이 없습니다: {filepath}")
    with open(filepath, "rb") as f:
        context = ts.context_from(f.read())
    print(f"[HEUtils] Context 로드 완료: {filepath}")
    return context
"""