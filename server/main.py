# server/main.py
import os
import sys

import uvicorn
import tenseal as ts
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import __version__ as pydantic_version

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.config import SERVER_HOST, SERVER_PORT, HE_POLY_MODULUS_DEGREE, HIDDEN_SIZE
from common.he_context import load_server_context
from common.protocol import (
    EncryptedInferenceRequest,
    EncryptedInferenceResponse,
    encode_bytes_to_base64,
    decode_base64_to_bytes,
)
from server.lora.adapter import get_fhe_lora_tensors
from server.lora.inference import he_lora_inference

PYDANTIC_V2 = pydantic_version.startswith("2.")
def model_validate(model_cls, data):
    return model_cls.model_validate(data) if PYDANTIC_V2 else model_cls.parse_obj(data)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] 엔터프라이즈 서버 시작")

    ctx = load_server_context()
    app_state["he_context"] = ctx

    max_slots = HE_POLY_MODULUS_DEGREE // 2
    if HIDDEN_SIZE > max_slots:
        print(f"[Server] 경고: HiddenSize={HIDDEN_SIZE}, HE 슬롯={max_slots}")

    # 🔥 모든 proj(q,k,v,o) 로딩
    from server.lora.adapter import load_all_lora_tensors
    proj_tensors = load_all_lora_tensors()

    # 🔥 서버에서 쓸 dictionary로 저장
    app_state["proj_tensors"] = proj_tensors

    print("[Server] 준비 완료")
    yield
    print("[Server] 종료")

app = FastAPI(lifespan=lifespan)

@app.post("/compute_lora", response_model=EncryptedInferenceResponse)
async def compute_lora(request: EncryptedInferenceRequest):
    try:
        ctx: ts.Context = app_state["he_context"]
        proj_tensors = app_state["proj_tensors"]   # 🔥 여기서 오류 나던 부분

        # Base64 → bytes → CKKSVector
        enc_bytes = decode_base64_to_bytes(request.enc_hidden_state_bytes)
        enc_vec = ts.ckks_vector_from(ctx, enc_bytes)

        # 🔥 4개 proj 각각 계산
        from server.lora.inference import he_lora_inference_multi

        enc_deltas = he_lora_inference_multi(enc_vec, proj_tensors, ctx)

        resp = EncryptedInferenceResponse(
            enc_q_delta_bytes = encode_bytes_to_base64(enc_deltas["q_proj"]),
            enc_k_delta_bytes = encode_bytes_to_base64(enc_deltas["k_proj"]),
            enc_v_delta_bytes = encode_bytes_to_base64(enc_deltas["v_proj"]),
            enc_o_delta_bytes = encode_bytes_to_base64(enc_deltas["o_proj"]),
        )
        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )
