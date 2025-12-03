# server/main.py
import os
import sys

import uvicorn
import tenseal as ts
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.config import (
    SERVER_HOST,
    SERVER_PORT,
    HE_POLY_MODULUS_DEGREE,
    HIDDEN_SIZE,
)
from common.he_context import load_server_context
from common.protocol import (
    EncryptedInferenceRequest,
    EncryptedInferenceResponse,
    encode_bytes_to_base64,
    decode_base64_to_bytes,
)
from server.lora.adapter import get_multi_fhe_lora_tensors
from server.lora.inference import he_lora_inference

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] 엔터프라이즈 FHE-LoRA 서버 시작")

    # 1) CKKS 컨텍스트 로드
    ctx = load_server_context()
    app_state["ctx"] = ctx

    max_slots = HE_POLY_MODULUS_DEGREE // 2
    if HIDDEN_SIZE > max_slots:
        print(f"[Server] 경고: HIDDEN_SIZE={HIDDEN_SIZE}, CKKS slots={max_slots}")

    # 2) (layer, module) 전체에 대한 LoRA 텐서 준비
    lora_tensors = get_multi_fhe_lora_tensors()
    app_state["lora_tensors"] = lora_tensors

    print("[Server] 준비 완료")
    yield
    print("[Server] 종료")


app = FastAPI(lifespan=lifespan)


@app.post("/compute_lora", response_model=EncryptedInferenceResponse)
async def compute_lora(request: EncryptedInferenceRequest):
    """
    요청: EncryptedInferenceRequest
      - enc_dict: {
            "(15, 'q_proj')": "<base64-ckks>",
            "(15, 'o_proj')": "<base64-ckks>",
            ...
        }

    응답: EncryptedInferenceResponse
      - enc_delta_dict: {
            "(15, 'q_proj')": "<base64-ckks-delta>",
            ...
        }
    """
    try:
        ctx: ts.Context = app_state["ctx"]
        lora_tensors = app_state["lora_tensors"]

        enc_dict = request.enc_dict
        out_dict = {}

        for key_str, enc_b64 in enc_dict.items():
            # key_str: "(15, 'q_proj')" 형태
            layer_idx, mod = eval(key_str)  # tuple로 변환

            if (layer_idx, mod) not in lora_tensors:
                raise KeyError(f"[Server] LoRA tensor 없음: {key_str}")

            W_A_pt, W_B_pt = lora_tensors[(layer_idx, mod)]

            enc_bytes = decode_bytes = decode_base64_to_bytes(enc_b64)
            enc_vec = ts.ckks_vector_from(ctx, enc_bytes)

            delta_bytes = he_lora_inference(enc_vec, W_A_pt, W_B_pt, ctx)
            out_dict[key_str] = encode_bytes_to_base64(delta_bytes)

        return EncryptedInferenceResponse(enc_delta_dict=out_dict)

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
