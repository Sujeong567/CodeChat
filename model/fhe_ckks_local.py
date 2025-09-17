# model/fhe_ckks_local.py
import tenseal as ts
import numpy as np 

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

def _get_ckks_vector_length(enc_vec):
    """
    TenSEAL CKKSVector의 길이를 안전하게 얻어오기 위한 헬퍼.
    가능한 여러 속성/메서드를 시도하고, 실패하면 None 반환.
    """
    # 1) 메서드 호출 시도 (있다면 호출)
    try:
        size = enc_vec.size()
        return int(size)
    except Exception:
        pass

    # 2) 속성 접근 시도
    try:
        size = enc_vec.size
        return int(size)
    except Exception:
        pass

    # 3) shape 속성 시도 (드물지만 호환되는 경우)
    try:
        shape = getattr(enc_vec, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    except Exception:
        pass

    # 4) 알 수 없으면 None 반환 (검사 생략)
    return None


def ckks_matmul(enc_vec, plain_mat):
    """
    암호화된 벡터(enc_vec)와 평문 행렬(plain_mat)의 곱셈 결과를 반환.
    enc_vec: TenSEAL CKKSVector (암호화된 1D 벡터)
    plain_mat: numpy.ndarray (n, d) 형태
    리턴: 암호화된 결과의 리스트 (길이 d), 각 요소는 암호화된 스칼라 (CKKSVector)
    """
    # 입력 검증
    if not isinstance(plain_mat, np.ndarray):
        # plain_mat이 1차원 벡터라면 리스트로 변환해 dot을 시도
        try:
            return enc_vec.dot(np.asarray(plain_mat).tolist())
        except Exception as e:
            raise ValueError("plain_mat must be a numpy array (2D) or a 1D-like iterable") from e

    if plain_mat.ndim == 2:
        n, d = plain_mat.shape

        # enc_vec 길이를 안전히 얻어보고, 가능하면 차원 일치 검사
        vec_len = _get_ckks_vector_length(enc_vec)
        if vec_len is not None and vec_len != n:
            raise ValueError(f"Dimension mismatch: enc_vec length {vec_len} vs plain_mat rows {n}")

        # 각 열(column)마다 enc_vec.dot(col) -> 암호화된 스칼라(보통 CKKSVector 형태) 반환
        result = []
        for i in range(d):
            col = plain_mat[:, i].tolist()
            result.append(enc_vec.dot(col))
        return result
    else:
        # plain_mat이 1차원일 경우 (벡터) 간단히 dot 수행
        return enc_vec.dot(plain_mat.tolist())
