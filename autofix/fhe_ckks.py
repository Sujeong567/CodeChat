import tenseal as ts
import base64
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient

app = FastAPI()

# 서버 전역 컨텍스트 (비밀키 없음)
server_context = None

# 요청 모델
class CtxRequest(BaseModel):
    ctx_b64: str # 클라이언트가 보낸 컨텍스트 (비밀키 제외)

class EncRequest(BaseModel):
    enc_data: str # 클라이언트가 보낸 암호문 (base64 인코딩)

# Base64 변환
def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

# 클라이언트 컨텍스트 로드
@app.post("/load_context")
def load_context(req: CtxRequest):
    """
    클라이언트가 만든 CKKS context(비밀키 제외)를 서버에 업로드
    서버는 이 컨텍스트로만 연산 수행(복호화 불가능)
    """
    global server_context
    ctx_bytes = b64_to_bytes(req.ctx_b64)
    server_context = ts.context_from(ctx_bytes)
    return {"status": "context loaded"}

# 암호문 연산(LoRA)
@app.post("/lora_forward")
def lora_forward(req: EncRequest):
    """
    암호문 입력 받아 2배곱 반환
    여기서 LoRA 행렬 연산 수행
    """
    assert server_context is not None, "Context not loaded"
    enc_bytes = b64_to_bytes(req.enc_data)
    enc_x = ts.ckks_vector_from(server_context, enc_bytes)
    enc_result = enc_x * 2
    return {"enc_result": bytes_to_b64(enc_result.serialize())}

def test_client():
    client = TestClient(app)
    # 클라이언트 컨텍스트 생성 (비밀키 포함)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    # 서버에 컨텍스트 업로드 (비밀키 제외)
    ctx_b64 = base64.b64encode(context.serialize(save_secret_key=False)).decode("utf-8")
    '''
    기존 코드
    response = requests.post("http://127.0.0.1:8000/load_context", json={"ctx_b64": ctx_b64})
    print("서버 컨텍스트 업로드:", response.json())
    '''
    res1 = client.post("/load_context", json={"ctx_b64": ctx_b64})
    print("서버 컨텍스트 업로드:", res1.json())

    
    # 암호화할 데이터 (클라이언트 입력 데이터)
    data = [1.1, 2.2, 3.3]
    enc_x = ts.ckks_vector(context, data)
    enc_bytes = enc_x.serialize()
    enc_b64 = base64.b64encode(enc_bytes).decode("utf-8")
    
    # 서버에 암호문 전송
    ''''
    기존 코드
    response = requests.post("http://127.0.0.1:8000/lora_forward", json={"enc_data": enc_b64})
    enc_result_b64 = response.json()["enc_result"]
    '''
    res2 = client.post("/lora_forward", json={"enc_data": enc_b64})
    enc_result_b64 = res2.json()["enc_result"]
    

    # 서버 응답 복호화
    enc_result_bytes = base64.b64decode(enc_result_b64.encode("utf-8"))
    enc_result = ts.ckks_vector_from(context, enc_result_bytes)
    result = enc_result.decrypt()

    print("입력 데이터:", data)
    print("서버 연산 결과:", result)
