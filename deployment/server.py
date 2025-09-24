import uvicorn
import tenseal as ts
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from model.lora_he_layer import apply_he_lora
from utils.generate_lora_low_matrices import generate_lora_low_rank_matrices

app = FastAPI()

# -------------------------
# Request / Response 모델
# -------------------------
class EncRequest(BaseModel):
    encrypted_vector: str   # 클라이언트에서 보내는 CKKS 벡터 (base64)

class EncResponse(BaseModel):
    encrypted_vector: str   # 서버에서 연산 후 직렬화된 결과 (base64)

# -------------------------
# CKKS 컨텍스트 로드
# -------------------------
CONTEXT_FILE = "data/ckks_enc/ckks_context.ctx"
with open(CONTEXT_FILE, "rb") as f:
    ctx_bytes = f.read()

try:
    context = ts.context_from(ctx_bytes)
except Exception:
    context = ts.context.load(ctx_bytes)

# -------------------------
# LoRA 준비
# -------------------------
d, r = 768, 8
lora_A, lora_B = generate_lora_low_rank_matrices(d, r, std=0.02)

# -------------------------
# 암호화 벡터 -> 처리 -> 반환
# -------------------------
@app.post("/process", response_model=EncResponse)
def process_vector(req: EncRequest):
    try:
        # base64 -> bytes -> CKKS vector
        enc_bytes = base64.b64decode(req.encrypted_vector.encode("utf-8"))
        enc_vec = ts.ckks_vector_from(context, enc_bytes)

        # LoRA 연산
        result_enc = apply_he_lora(enc_vec, lora_A, lora_B, alpha=1.0)

        # 결과 직렬화 -> base64 -> JSON
        result_b64 = base64.b64encode(result_enc.serialize()).decode("utf-8")
        return EncResponse(encrypted_vector=result_b64)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)