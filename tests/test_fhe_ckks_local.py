from model.fhe_ckks_local import ckks_encrypt, create_context, ckks_decrypt

# 컨텍스트 생성 및 공개키 복사
context = create_context()  # 비밀키 포함 컨텍스트 생성
public_key = context.copy()  # 컨텍스트 복사
public_key.make_context_public()  # 공개키만 포함

# 평문 데이터 준비 및 암호화
data = [1.1, 2.2, 3.3]
print("[평문]", data)

ctxt = ckks_encrypt(data, public_key)
print("[암호문 객체]", ctxt)

#복호화 
decrypted_data = ckks_decrypt(ctxt, context)
print("[복호화된 데이터]", decrypted_data)