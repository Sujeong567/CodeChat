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

from common.config import SERVER_HOST, SERVER_PORT
from common.he_context import load_server_context
from common.protocol import (
    EncryptedInferenceRequest,
    EncryptedInferenceResponse,
    encode_bytes_to_base64,
    decode_base64_to_bytes,
)
from server.lora.adapter import get_multi_fhe_lora_tensors
from server.lora.inference import he_lora_inference

PYDANTIC_V2 = pydantic_version.startswith("2.")
def model_validate(model_cls, data):
    return model_cls.model_validate(data) if PYDANTIC_V2 else model_cls.parse_obj(data)

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] FHE-LoRA 서버 시작")

    ctx = load_server_context()
    app_state["ctx"] = ctx

    # (layer, module) → (W_A_pt, W_B_pt)
    app_state["lora_tensors"] = get_multi_fhe_lora_tensors()

    print("[Server] 준비 완료")
    yield
    print("[Server] 종료")


app = FastAPI(lifespan=lifespan)


@app.post("/compute_lora", response_model=EncryptedInferenceResponse)
async def compute_lora(request: EncryptedInferenceRequest):
    """
    단일 (layer_idx, module_name)에 대한 FHE-LoRA 연산
    """
    try:
        ctx: ts.Context = app_state["ctx"]
        tensors = app_state["lora_tensors"]

        key = (request.layer_idx, request.module_name)
        if key not in tensors:
            raise ValueError(f"Unknown (layer, module): {key}")

        W_A_pt, W_B_pt = tensors[key]

        enc_bytes = decode_base64_to_bytes(request.enc_hidden_state_bytes)
        enc_vec = ts.ckks_vector_from(ctx, enc_bytes)

        delta_bytes = he_lora_inference(enc_vec, W_A_pt, W_B_pt, ctx)
        resp_b64 = encode_bytes_to_base64(delta_bytes)

        return EncryptedInferenceResponse(enc_lora_delta_bytes=resp_b64)

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
