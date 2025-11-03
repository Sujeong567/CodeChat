"""
클라이언트에서 암호화(중간값), 복호화(LoRA 연산 후) 담당
"""

import tenseal as ts
import numpy as np
from typing import Dict, List

def encrypt_vector(context: ts.Context, data: np.ndarray) -> Dict:
    """
    벡터를 CKKS로 암호화
    
    Args:
        context: CKKS 컨텍스트
        data: 암호화할 numpy 배열
    
    Returns:
        {
            'ciphertext': 직렬화된 암호문 (리스트),
            'size': 원본 크기,
            'shape': 원본 shape
        }
    """
    print(f"🔐 벡터 암호화 중... (크기: {data.shape})")
    
    # 1. 1D 배열로 변환
    flat_data = data.flatten()
    
    # 2. Python 리스트로 변환 (TenSEAL 입력 형식)
    data_list = flat_data.tolist()
    
    # 3. CKKS 암호화
    encrypted = ts.ckks_vector(context, data_list)
    
    # 4. 직렬화 (네트워크 전송 가능하게)
    serialized = encrypted.serialize()
    
    print(f"✅ 암호화 완료! (암호문 크기: {len(serialized)} bytes)")
    
    return {
        'ciphertext': list(serialized),  # bytes → list
        'size': len(data_list),
        'shape': list(data.shape)
    }

def decrypt_vector(context: ts.Context, encrypted_data: Dict) -> np.ndarray:
    """
    CKKS 암호문을 복호화
    
    Args:
        context: CKKS 컨텍스트
        encrypted_data: {
            'ciphertext': 직렬화된 암호문,
            'size': 크기,
            'shape': 원본 shape
        }
    
    Returns:
        복호화된 numpy 배열
    """
    print("🔓 벡터 복호화 중...")
    
    # 1. 직렬화된 데이터를 bytes로 변환
    serialized = bytes(encrypted_data['ciphertext'])
    
    # 2. TenSEAL 객체로 복원
    encrypted = ts.ckks_vector_from(context, serialized)
    
    # 3. 복호화
    decrypted_list = encrypted.decrypt()
    
    # 4. numpy 배열로 변환
    decrypted = np.array(decrypted_list)
    
    # 5. 원래 shape으로 복원
    if 'shape' in encrypted_data:
        original_shape = tuple(encrypted_data['shape'])
        decrypted = decrypted.reshape(original_shape)
    
    print(f"✅ 복호화 완료! (shape: {decrypted.shape})")
    
    return decrypted

# ============================================
# 컨텍스트 직렬화 (서버와 공유할 때)
# ============================================

def serialize_context(context: ts.Context) -> bytes:
    """
    CKKS 컨텍스트를 직렬화 (public key만 포함)
    서버와 공유할 때 사용
    
    Args:
        context: CKKS 컨텍스트
    
    Returns:
        직렬화된 컨텍스트 (bytes)
    """
    return context.serialize(save_secret_key=False)


def deserialize_context(serialized: bytes) -> ts.Context:
    """
    직렬화된 컨텍스트를 복원
    
    Args:
        serialized: 직렬화된 컨텍스트
    
    Returns:
        TenSEAL Context
    """
    return ts.context_from(serialized)