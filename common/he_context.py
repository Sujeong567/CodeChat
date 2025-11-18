# common/he_context.py
import os
import tenseal as ts

from common.config import (
    HE_POLY_MODULUS_DEGREE,
    HE_COEFF_MOD_BIT_SIZES,
    HE_GLOBAL_SCALE_BITS,
    CLIENT_SECRET_KEY_PATH,
    SERVER_PUBLIC_KEY_PATH,
    HE_KEY_DIR,
)

def create_ckks_context() -> ts.Context:
    print("[HE] CKKS Context 생성 중...")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=HE_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=HE_COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = 2 ** HE_GLOBAL_SCALE_BITS
    context.generate_galois_keys()
    context.generate_relin_keys()

    print("[HE] CKKS Context 생성 완료.")
    return context


def get_or_create_he_keys():
    """키 파일이 없으면 생성하고, 있으면 로드하여 클라이언트/서버 컨텍스트 반환"""
    os.makedirs(HE_KEY_DIR, exist_ok=True)

    if os.path.exists(CLIENT_SECRET_KEY_PATH) and os.path.exists(SERVER_PUBLIC_KEY_PATH):
        print("[HE] 기존 HE 키 로드 중...")
        with open(CLIENT_SECRET_KEY_PATH, "rb") as f:
            client_ctx = ts.context_from(f.read())
        with open(SERVER_PUBLIC_KEY_PATH, "rb") as f:
            server_ctx = ts.context_from(f.read())
        print("[HE] HE 키 로드 완료.")
        return client_ctx, server_ctx

    print("[HE] HE 키 생성 중...")
    client_ctx = create_ckks_context()

    # 서버용 공개 컨텍스트
    server_ctx_bytes = client_ctx.serialize(save_secret_key=False)
    server_ctx = ts.context_from(server_ctx_bytes)

    # 파일 저장
    with open(CLIENT_SECRET_KEY_PATH, "wb") as f:
        f.write(client_ctx.serialize(save_secret_key=True))
    with open(SERVER_PUBLIC_KEY_PATH, "wb") as f:
        f.write(server_ctx_bytes)

    print(f"[HE] HE 키 생성 및 저장 완료: {HE_KEY_DIR}")
    return client_ctx, server_ctx


def load_client_context() -> ts.Context:
    client_ctx, _ = get_or_create_he_keys()
    return client_ctx


def load_server_context() -> ts.Context:
    _, server_ctx = get_or_create_he_keys()
    return server_ctx