# client_backend/main.py
import os
import sys
import time
import gc
import requests

import torch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import __version__ as pydantic_version
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from common.config import (
    CLIENT_BACKEND_HOST,
    CLIENT_BACKEND_PORT,
    SERVER_HOST,
    SERVER_PORT,
    DEVICE,
)
from common.protocol import (
    ClientBackendRequest,
    ClientBackendResponse,
    EncryptedInferenceRequest,
    EncryptedInferenceResponse,
    encode_bytes_to_base64,
    decode_base64_to_bytes,
)
from client_backend.crypto.ckks_client import CKKSClientManager
from client_backend.model.base_llm import BaseLLMLoader
from client_backend.model.preprocessing import LLMPreProcessor
from client_backend.model.postprocessing import LLMPostProcessor

PYDANTIC_V2 = pydantic_version.startswith("2.")
def model_dump(model):
    return model.model_dump() if PYDANTIC_V2 else model.dict()
def model_validate(model_cls, data):
    return model_cls.model_validate(data) if PYDANTIC_V2 else model_cls.parse_obj(data)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[ClientBackend] 서버 시작 - 모델/키 로드")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = BaseLLMLoader()
    loader.load_model()

    app_state["llm_loader"] = loader
    app_state["preprocessor"] = LLMPreProcessor(loader)
    app_state["postprocessor"] = LLMPostProcessor(loader)
    app_state["ckks_manager"] = CKKSClientManager()

    app_state["http_session"] = requests.Session()
    app_state["server_url"] = f"http://{SERVER_HOST}:{SERVER_PORT}/compute_lora"

    print("[ClientBackend] 초기화 완료")
    yield

    app_state["http_session"].close()
    loader.clear_lora_hooks()
    print("[ClientBackend] 서버 종료")

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=ClientBackendResponse)
async def generate(request: ClientBackendRequest):
    start_time = time.time()
    try:
        print(f"[ClientBackend] 추론 요청: '{request.prompt[:80]}' ...")

        loader: BaseLLMLoader = app_state["llm_loader"]
        preproc: LLMPreProcessor = app_state["preprocessor"]
        postproc: LLMPostProcessor = app_state["postprocessor"]
        ckks: CKKSClientManager = app_state["ckks_manager"]
        session: requests.Session = app_state["http_session"]
        server_url: str = app_state["server_url"]

        # 매 요청마다 LoRA 가중치 0으로 초기화
        loader.reset_lora_weights()

        # 1) 초기 상태
        states = preproc.get_initial_states(request.prompt)
        generated_ids = states["generated_ids"][:]

        max_steps = request.max_new_tokens

        for step in range(max_steps):
            print(f"[ClientBackend] Token step {step + 1}/{max_steps}")

            # 2) 현재 xL (1, hidden) -> (hidden,) -> 암호화
            xL = states["lora_xL_input"]  # (1, H)
            xL_vec = xL.squeeze(0)        # (H,)
            enc_bytes = ckks.encrypt_tensor(xL_vec)

            # 3) 서버로 전송
            req_obj = EncryptedInferenceRequest(
                enc_hidden_state_bytes=encode_bytes_to_base64(enc_bytes)
            )
            res = session.post(server_url, json=model_dump(req_obj))
            res.raise_for_status()
            resp_obj = model_validate(EncryptedInferenceResponse, res.json())

            # 4) 서버에서 계산한 LoRA delta 복호화
            delta_q_bytes = decode_base64_to_bytes(resp_obj.enc_q_delta_bytes)
            delta_k_bytes = decode_base64_to_bytes(resp_obj.enc_k_delta_bytes)
            delta_v_bytes = decode_base64_to_bytes(resp_obj.enc_v_delta_bytes)
            delta_o_bytes = decode_base64_to_bytes(resp_obj.enc_o_delta_bytes)

            delta_q_vec = ckks.decrypt_tensor(delta_q_bytes)
            delta_k_vec = ckks.decrypt_tensor(delta_k_bytes)
            delta_v_vec = ckks.decrypt_tensor(delta_v_bytes)
            delta_o_vec = ckks.decrypt_tensor(delta_o_bytes)

            delta_tensors = {
                "q_proj": delta_q_vec.unsqueeze(0).to(DEVICE),
                "k_proj": delta_k_vec.unsqueeze(0).to(DEVICE),
                "v_proj": delta_v_vec.unsqueeze(0).to(DEVICE),
                "o_proj": delta_o_vec.unsqueeze(0).to(DEVICE),
            }

            # 5) 여러 delta를 hook에 전달
            loader.set_global_lora_output_deltas(delta_tensors)

            # 6) 현재 hidden state 기반으로 다음 토큰 argmax
            next_token_id, next_token_char = postproc.integrate_lora_delta_and_predict_token(
                states["current_llm_hidden_state"]
            )
            generated_ids.append(next_token_id)

            print(f"  -> 생성 토큰: {repr(next_token_char)}")

            if next_token_id == loader.eos_token_id:
                print("  EOS 토큰 감지, 종료.")
                break

            # 7) LLM 상태 업데이트 (이때 hook이 delta 주입하고 새 xL 캡처)
            states = preproc.get_next_token_states(next_token_id, states)

            # 8) 델타 주입 완료 후 전역 delta 초기화
            loader.clear_global_lora_output_deltas()

        final_text = postproc.decode_final_output(generated_ids)
        elapsed = time.time() - start_time

        print("[ClientBackend] 최종 결과:")
        print(final_text[:500])
        print(f"[ClientBackend] 소요 시간: {elapsed:.2f}초")

        return ClientBackendResponse(
            generated_text=final_text,
            status="success",
            message=f"LLM 추론 완료 ({elapsed:.2f}초)",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "client_backend.main:app",
        host=CLIENT_BACKEND_HOST,
        port=CLIENT_BACKEND_PORT,
        reload=True,
    )
