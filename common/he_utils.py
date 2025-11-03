# common/he_utils.py

import tenseal as ts
import os
from common.config import HE_KEY_DIR

from common.config import (
    HE_SCHEME_TYPE,
    HE_POLY_MODULUS_DEGREE,
    HE_COEFF_MOD_BIT_SIZES,
    HE_GLOBAL_SCALE_BITS
)

def create_ckks_context() -> ts.Context:
    """
    CKKS Context 생성 (설정값은 config.py에서 가져옴)
    
    Returns:
        TenSEAL Context (비밀키 포함)
    """
    print(f"\n{'='*60}")
    print("🔐 CKKS Context 생성 중...")
    print(f"{'='*60}\n")
    
    print(f"   Scheme: {HE_SCHEME_TYPE}")
    print(f"   Poly Modulus Degree: {HE_POLY_MODULUS_DEGREE}")
    print(f"   Coeff Mod Bit Sizes: {HE_COEFF_MOD_BIT_SIZES}")
    print(f"   Global Scale: 2^{HE_GLOBAL_SCALE_BITS}")
    
    # 1. Context 생성
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=HE_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=HE_COEFF_MOD_BIT_SIZES
    )
    
    # 2. Global Scale 설정
    context.global_scale = 2 ** HE_GLOBAL_SCALE_BITS
    
    # 3. Galois Keys 생성
    print("\n   🔑 Galois Keys 생성 중...")
    context.generate_galois_keys()
    
    # 4. Relin Keys 생성
    print("   🔑 Relin Keys 생성 중...")
    context.generate_relin_keys()
    
    print(f"\n{'='*60}")
    print("✅ CKKS Context 생성 완료!")
    print(f"{'='*60}\n")
    
    return context

def save_tenseal_context(context: ts.Context, filename: str, save_secret_key: bool):
    """TenSEAL Context를 파일로 저장합니다."""
    filepath = os.path.join(HE_KEY_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(context.serialize(save_secret_key=save_secret_key))
    print(f"[HEUtils] Context 저장 완료: {filepath} (Secret Key: {save_secret_key})")

def load_tenseal_context(filename: str) -> ts.Context:
    """파일로부터 TenSEAL Context를 로드합니다."""
    filepath = os.path.join(HE_KEY_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Context 파일이 없습니다: {filepath}")
    with open(filepath, "rb") as f:
        context = ts.context_from(f.read())
    print(f"[HEUtils] Context 로드 완료: {filepath}")
    return context