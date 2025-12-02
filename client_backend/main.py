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
)
from common.protocol import (
    ClientBackendRequest,
    ClientBackendResponse,
)
from client_backend.crypto.ckks_client import CKKSClientManager
from client_backend.model.base_llm import BaseLLMLoader
from client_backend.model.preprocessing import LLMPreProcessor
from client_backend.model.postprocessing import LLMPostProcessor

PYDANTIC_V2 = pydantic_version.startswith("2.")


app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[ClientBackend] 서버 시작 - 모델/키 로드")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = BaseLLMLoader()
    loader.load_model()

    ckks_manager = CKKSClientManager()
    http_session = requests.Session()
    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/compute_lora"

    # q_proj same-token FHE-LoRA를 위해 loader에 FHE 클라이언트 주입
    loader.attach_fhe_client(ckks_manager, http_session, server_url)

    app_state["llm_loader"] = loader
    app_state["preprocessor"] = LLMPreProcessor(loader)
    app_state["postprocessor"] = LLMPostProcessor(loader)
    app_state["ckks_manager"] = ckks_manager
    app_state["http_session"] = http_session
    app_state["server_url"] = server_url

    print("[ClientBackend] 초기화 완료")
    yield

    http_session.close()
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

        # 매 요청마다 LoRA 가중치 0으로 초기화 (FHE-LoRA만 쓰기 위해)
        loader.reset_lora_weights()

        # 1) 초기 상태
        states = preproc.get_initial_states(request.prompt)
        generated_ids = states["generated_ids"][:]

        max_steps = request.max_new_tokens

        for step in range(max_steps):
            print(f"[ClientBackend] Token step {step + 1}/{max_steps}")

            # 2) 현재 hidden state 기반으로 다음 토큰 argmax
            next_token_id, next_token_char = postproc.integrate_lora_delta_and_predict_token(
                states["current_llm_hidden_state"]
            )
            generated_ids.append(next_token_id)

            print(f"  -> 생성 토큰: {repr(next_token_char)}")

            if next_token_id == loader.eos_token_id:
                print("  EOS 토큰 감지, 종료.")
                break

            # 3) LLM 상태 업데이트
            #    (이 forward 과정에서 q_proj same-token hook이 FHE-LoRA를 수행함)
            states = preproc.get_next_token_states(next_token_id, states)

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
