# client_backend/main_plain.py (Plaintext Baseline Version)
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
from typing import List

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
    # 🚫 HE/암호화 관련 임포트 제거
    # EncryptedInferenceRequest,
    # EncryptedInferenceResponse,
    # encode_bytes_to_base64,
    # decode_base64_to_bytes,
    # ✅ Plaintext 프로토콜 임포트
    PlaintextInferenceRequest,
    PlaintextInferenceResponse,
)
# from client_backend.crypto.ckks_client import CKKSClientManager # 🚫 제거
from client_backend.model.base_llm import BaseLLMLoader
from client_backend.model.preprocessing import LLMPreProcessor
from client_backend.model.postprocessing import LLMPostProcessor

PYDANTIC_V2 = pydantic_version.startswith("2.")
def model_dump(model):
    return model.model_dump() if PYDANTIC_V2 else model.dict()
def model_validate(model_cls, data):
    # PlaintextInferenceResponse, PlaintextInferenceRequest 등을 위해 남겨둡니다.
    return model_cls.model_validate(data) if PYDANTIC_V2 else model_cls.parse_obj(data)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[ClientBackend] 평문 서버 시작 - 모델 로드")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = BaseLLMLoader()
    loader.load_model()

    app_state["llm_loader"] = loader
    app_state["preprocessor"] = LLMPreProcessor(loader)
    app_state["postprocessor"] = LLMPostProcessor(loader)
    # app_state["ckks_manager"] = CKKSClientManager() # 🚫 제거

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
        # ckks: CKKSClientManager = app_state["ckks_manager"] # 🚫 제거
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

            # 2) 현재 xL (1, hidden) -> (hidden,) -> 평문화
            xL = states["lora_xL_input"]  # (1, H)
            xL_vec = xL.squeeze(0)        # (H,)
            
            # 🚫 암호화 제거
            # enc_bytes = ckks.encrypt_tensor(xL_vec)
            # ✅ 평문 리스트로 변환
            hidden_state_vec: List[float] = xL_vec.tolist()


            # 3) 서버로 전송 (Plaintext 프로토콜 사용)
            req_obj = PlaintextInferenceRequest(
                hidden_state_vec=hidden_state_vec
            )
            res = session.post(server_url, json=model_dump(req_obj))
            res.raise_for_status()
            
            # ✅ 응답 모델 변경
            resp_obj = model_validate(PlaintextInferenceResponse, res.json())

            # 4) 서버에서 계산한 LoRA delta 처리 (복호화 제거)
            # 🚫 복호화 제거
            # delta_bytes = decode_base64_to_bytes(resp_obj.enc_lora_delta_bytes)
            # delta_vec = ckks.decrypt_tensor(delta_bytes)  # (H,)
            
            # ✅ Plaintext 응답 리스트를 PyTorch 텐서로 변환
            delta_vec_list: List[float] = resp_obj.lora_delta_vec
            delta_vec = torch.tensor(delta_vec_list, dtype=torch.float32).to(DEVICE)
            
            # (H,) -> (1, H)
            delta_tensor = delta_vec.unsqueeze(0) 

            # 5) 델타를 전역에 설정 (hook이 사용)
            loader.set_global_lora_output_delta(delta_tensor)


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
            loader.clear_global_lora_output_delta()

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
    # ⚠️ 실행 시 모듈 이름 변경 (main -> main_plain)
    uvicorn.run(
        "client_backend.main_plain:app",
        host=CLIENT_BACKEND_HOST,
        port=CLIENT_BACKEND_PORT,
        reload=True,
    )