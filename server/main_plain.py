#server/main_plain.py

import os
import sys
import torch
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import __version__ as pydantic_version

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.config import SERVER_HOST, SERVER_PORT, HIDDEN_SIZE

# ✅ Plaintext 프로토콜 임포트
from common.protocol import (
    PlaintextInferenceRequest,
    PlaintextInferenceResponse,
    
)

# ✅ Plaintext LoRA 관련 함수 임포트
from server.lora.adapter_plain import get_plaintext_lora_tensors
from server.lora.inference_plain import plaintext_lora_inference

PYDANTIC_V2 = pydantic_version.startswith("2.")
def model_validate(model_cls, data):
    return model_cls.model_validate(data) if PYDANTIC_V2 else model_cls.parse_obj(data)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] 평문 베이스라인 서버 시작")

 
    # ✅ q_proj 하나에 대한 Plaintext LoRA W_A, W_B만 준비
    # get_plaintext_lora_tensors 함수가 PyTorch 텐서를 반환한다고 가정
    W_A_pt, W_B_pt = get_plaintext_lora_tensors()
    app_state["W_A_pt"] = W_A_pt
    app_state["W_B_pt"] = W_B_pt

    print("[Server] 준비 완료")
    yield
    print("[Server] 종료")

app = FastAPI(lifespan=lifespan)

# ✅ Plaintext 프로토콜로 변경
@app.post("/compute_lora", response_model=PlaintextInferenceResponse)
async def compute_lora(request: PlaintextInferenceRequest):
    try:

        W_A_pt = app_state["W_A_pt"]
        W_B_pt = app_state["W_B_pt"]

        
        # ✅ 평문 입력 (List[float])을 PyTorch 텐서로 변환
        hidden_state_vec: List[float] = request.hidden_state_vec
        input_tensor = torch.tensor(hidden_state_vec, dtype=torch.float32)

        # ✅ 평문 LoRA delta 계산
        # result_tensor는 LoRA delta ($\Delta h = h \cdot W_A \cdot W_B$)
        result_tensor: torch.Tensor = plaintext_lora_inference(input_tensor, W_A_pt, W_B_pt)
        

        # ✅ 결과 텐서를 List[float]로 변환하여 응답
        result_vec: List[float] = result_tensor.tolist()

        return PlaintextInferenceResponse(lora_delta_vec=result_vec)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    
    uvicorn.run(
        "server.main_plain:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )