"""
CKKS 암호화/복호화 테스트 스크립트
"""

import sys
import os

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
from crypto.ckks_client import encrypt_vector, decrypt_vector
from common.he_utils import create_ckks_context


def test_basic_encryption():
    """기본 암호화/복호화 테스트"""
    
    print("\n" + "="*70)
    print("🧪 테스트 1: 기본 암호화/복호화")
    print("="*70 + "\n")
    
    # 1. Context 생성
    print("🔐 CKKS Context 생성 중...")
    context = create_ckks_context()
    
    # 2. 테스트 데이터 생성
    print("\n📊 테스트 데이터 생성")
    original_data = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    print(f"   원본 데이터: {original_data}")
    print(f"   Shape: {original_data.shape}")
    print(f"   dtype: {original_data.dtype}")
    
    # 3. 암호화
    print("\n🔐 암호화 시작...")
    encrypted = encrypt_vector(context, original_data)
    
    print(f"\n   암호문 크기: {len(encrypted['ciphertext'])} bytes")
    print(f"   원본 크기: {encrypted['size']}")
    print(f"   원본 Shape: {encrypted['shape']}")
    
    # 4. 복호화
    print("\n🔓 복호화 시작...")
    decrypted = decrypt_vector(context, encrypted)
    
    print(f"\n   복호화된 데이터: {decrypted}")
    print(f"   Shape: {decrypted.shape}")
    
    # 5. 정확도 검증
    print("\n✅ 정확도 검증")
    difference = np.abs(original_data - decrypted)
    max_error = np.max(difference)
    
    print(f"   최대 오차: {max_error:.10f}")
    print(f"   평균 오차: {np.mean(difference):.10f}")
    
    if max_error < 1e-5:
        print("   🎉 테스트 통과! (오차 < 0.00001)")
        return True
    else:
        print("   ⚠️ 오차가 큽니다!")
        return False


def test_2d_array():
    """2D 배열 암호화/복호화 테스트"""
    
    print("\n" + "="*70)
    print("🧪 테스트 2: 2D 배열 암호화/복호화")
    print("="*70 + "\n")
    
    # 1. Context 생성
    context = create_ckks_context()
    
    # 2. 2D 배열 생성
    print("📊 2D 배열 생성")
    original_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    print(f"   원본 데이터:\n{original_data}")
    print(f"   Shape: {original_data.shape}")
    
    # 3. 암호화
    print("\n🔐 암호화...")
    encrypted = encrypt_vector(context, original_data)
    
    # 4. 복호화
    print("\n🔓 복호화...")
    decrypted = decrypt_vector(context, encrypted)
    
    print(f"\n   복호화된 데이터:\n{decrypted}")
    print(f"   Shape: {decrypted.shape}")
    
    # 5. 검증
    max_error = np.max(np.abs(original_data - decrypted))
    print(f"\n   최대 오차: {max_error:.10f}")
    
    if max_error < 1e-5:
        print("   🎉 테스트 통과!")
        return True
    else:
        print("   ⚠️ 실패!")
        return False


def test_large_array():
    """큰 배열 테스트 (실제 hidden states 크기)"""
    
    print("\n" + "="*70)
    print("🧪 테스트 3: 대용량 배열 (실제 LLM hidden states 크기)")
    print("="*70 + "\n")
    
    # 1. Context 생성
    context = create_ckks_context()
    
    # 2. 실제 크기 배열 생성 (512 토큰 × 768 차원)
    print("📊 대용량 배열 생성")
    shape = (1, 512, 768)
    original_data = np.random.randn(*shape).astype(np.float32)
    
    print(f"   Shape: {shape}")
    print(f"   원소 개수: {original_data.size:,}")
    print(f"   메모리 크기: {original_data.nbytes / 1024 / 1024:.2f} MB")
    
    # 3. 암호화
    import time
    
    print("\n🔐 암호화 중... (시간 측정)")
    start_time = time.time()
    encrypted = encrypt_vector(context, original_data)
    encrypt_time = time.time() - start_time
    
    print(f"   암호화 시간: {encrypt_time:.2f}초")
    print(f"   암호문 크기: {len(encrypted['ciphertext']) / 1024 / 1024:.2f} MB")
    
    # 4. 복호화
    print("\n🔓 복호화 중... (시간 측정)")
    start_time = time.time()
    decrypted = decrypt_vector(context, encrypted)
    decrypt_time = time.time() - start_time
    
    print(f"   복호화 시간: {decrypt_time:.2f}초")
    
    # 5. 검증 (샘플링)
    sample_size = 1000
    sample_indices = np.random.choice(original_data.size, sample_size, replace=False)
    
    original_sample = original_data.flatten()[sample_indices]
    decrypted_sample = decrypted.flatten()[sample_indices]
    
    max_error = np.max(np.abs(original_sample - decrypted_sample))
    print(f"\n   최대 오차 (샘플 {sample_size}개): {max_error:.10f}")
    
    if max_error < 1e-5:
        print("   🎉 테스트 통과!")
        return True
    else:
        print("   ⚠️ 실패!")
        return False


def test_context_serialization():
    """Context 직렬화 테스트"""
    
    print("\n" + "="*70)
    print("🧪 테스트 4: Context 직렬화/역직렬화")
    print("="*70 + "\n")
    
    from crypto.ckks_client import serialize_context, deserialize_context
    
    # 1. Context 생성
    print("🔐 Context 생성")
    original_context = create_ckks_context()
    
    # 2. 데이터 암호화
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\n📊 원본 데이터: {data}")
    
    encrypted = encrypt_vector(original_context, data)
    
    # 3. Context 직렬화 (Public Key만)
    print("\n📦 Context 직렬화 (비밀키 제외)")
    serialized = serialize_context(original_context)
    print(f"   직렬화 크기: {len(serialized) / 1024:.2f} KB")
    
    # 4. Context 역직렬화
    print("\n📂 Context 역직렬화")
    deserialized_context = deserialize_context(serialized)
    print("   ✅ 역직렬화 완료")
    
    # 5. 비밀키가 없어서 복호화 불가능한지 테스트
    print("\n🔒 비밀키 없이 복호화 시도...")
    try:
        decrypted = decrypt_vector(deserialized_context, encrypted)
        print("   ⚠️ 경고: 복호화가 성공했습니다 (예상치 못함)")
        return False
    except Exception as e:
        print(f"   ✅ 예상대로 복호화 실패: {type(e).__name__}")
        print("   (서버는 복호화 불가능!)")
        return True


def run_all_tests():
    """모든 테스트 실행"""
    
    print("\n" + "🎯" * 30)
    print("CKKS 암호화/복호화 전체 테스트 시작")
    print("🎯" * 30 + "\n")
    
    results = []
    
    # 테스트 1
    try:
        results.append(("기본 암호화/복호화", test_basic_encryption()))
    except Exception as e:
        print(f"❌ 테스트 1 실패: {e}")
        results.append(("기본 암호화/복호화", False))
    
    # 테스트 2
    try:
        results.append(("2D 배열", test_2d_array()))
    except Exception as e:
        print(f"❌ 테스트 2 실패: {e}")
        results.append(("2D 배열", False))
    
    # 테스트 3
    try:
        results.append(("대용량 배열", test_large_array()))
    except Exception as e:
        print(f"❌ 테스트 3 실패: {e}")
        results.append(("대용량 배열", False))
    
    # 테스트 4
    try:
        results.append(("Context 직렬화", test_context_serialization()))
    except Exception as e:
        print(f"❌ 테스트 4 실패: {e}")
        results.append(("Context 직렬화", False))
    
    # 결과 요약
    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70 + "\n")
    
    for i, (name, passed) in enumerate(results, 1):
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{i}. {name:30s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "-"*70)
    print(f"총 {total}개 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    print("-"*70 + "\n")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
    else:
        print(f"⚠️ {total - passed}개 테스트 실패")


if __name__ == "__main__":
    run_all_tests()

