from fastapi import FastAPI
from pydantic import BaseModel
import tenseal as ts
import base64

app = FastAPI()

# 서버 전역 컨텍스트 (비밀키 없음)
server_context = None

# 요청 모델
class CtxRequest(BaseModel):
    ctx_b64: str    # 클라이언트가 보낸 컨텍스트 (비밀키 제외)
    
class EncRequest(BaseModel):
    enc_data: str   # 클라이언트가 보낸 암호문 (base64 인코딩)

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