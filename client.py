import requests
import tenseal as ts
import base64

# 클라이언트 컨텍스트 생성 (비밀키 포함)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40,60]
)
context.global_scale = 2 ** 40
context.generate_galois_keys()

# 서버에 컨텍스트 업로드 (비밀키 제외)
ctx_b64 = base64.b64encode(context.serialize(save_secret_key=False)).decode("utf-8")
response = requests.post("http://127.0.0.1:8000/load_context", json={"ctx_b64": ctx_b64})
print("서버 컨텍스트 업로드:", response.json())

# 암호화할 데이터 (클라이언트 입력 데이터)
data = [1.1, 2.2, 3.3]
enc_x = ts.ckks_vector(context, data)                   # 암호화
enc_bytes = enc_x.serialize()                           # 직렬화
enc_b64 = base64.b64encode(enc_bytes).decode("utf-8")   # base64 인코딩

# 서버에 암호문 전송
response = requests.post("http://127.0.0.1:8000/lora_forward", json={"enc_data": enc_b64})
enc_result_b64 = response.json()["enc_result"]

# 서버 응답 복호화
enc_result_bytes = base64.b64decode(enc_result_b64.encode("utf-8"))
enc_result = ts.ckks_vector_from(context, enc_result_bytes)
result = enc_result.decrypt()

print("입력 데이터:", data)
print("서버 연산 결과:", result)