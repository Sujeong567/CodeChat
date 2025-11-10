# codechat/client-backend/main.py
import torch
import gc
import sys
import os
import requests
import uvicorn
import time
from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from pydantic import __version__ as pydantic_version

# --- 1. 프로젝트 루트를 sys.path에 추가 ---
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# --- 2. 모듈 임포트 ---
from common.config import (
    CLIENT_BACKEND_HOST, CLIENT_BACKEND_PORT, SERVER_HOST, SERVER_PORT,
    MAX_GEN_LENGTH, DEVICE
)
from common.protocol import (
    ClientBackendRequest, ClientBackendResponse,
    EncryptedInferenceRequest, EncryptedInferenceResponse,
    encode_bytes_to_base64, decode_base64_to_bytes
)
from crypto.ckks_client import CKKSClientManager
from model.base_llm import BaseLLMLoader
from model.preprocessing import LLMPreProcessor
from model.postprocessing import LLMPostProcessor

# Pydantic V1/V2 호환성 처리
PYDANTIC_V2 = pydantic_version.startswith("2.")
def model_dump(model):
    return model.model_dump() if PYDANTIC_V2 else model.dict()
def model_validate(model_cls, data):
    return model_cls.model_validate(data) if PYDANTIC_V2 else model_cls.parse_obj(data)

# --- 3. 전역 애플리케이션 리소스 ---
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델/키 로드"""
    print("--- 클라이언트 백엔드 서버 시작 ---")
    gc.collect()
    torch.cuda.empty_cache()
    
    app_state["llm_loader"] = BaseLLMLoader()
    app_state["llm_loader"].load_model()    # LoRA Wrapped 모델 로드 및 훅 등록
    
    app_state["preprocessor"] = LLMPreProcessor(app_state["llm_loader"])
    app_state["postprocessor"] = LLMPostProcessor(app_state["llm_loader"])
    app_state["ckks_manager"] = CKKSClientManager()
    
    app_state["http_session"] = requests.Session()
    app_state["server_url"] = f"http://{SERVER_HOST}:{SERVER_PORT}/compute_lora"
    
    print("--- 클라이언트 백엔드 준비 완료 ---")
    yield
    # --- 서버 종료 시 ---
    app_state["http_session"].close()
    app_state["llm_loader"].clear_lora_hooks()  # 훅 제거
    print("--- 클라이언트 백엔드 서버 종료 ---")

app = FastAPI(lifespan=lifespan)

# --- 4. API 엔드포인트 ---
@app.post("/generate", response_model=ClientBackendResponse)
async def generate(request: ClientBackendRequest):
    """사용자로부터 GenerationRequest(프롬프트)를 받음"""
    start_time = time.time()

    try:
        print(f"추론 요청 수신: 프롬프트 '{request.prompt[:50]}...")

        # 매 추론 시작 시 LoRA 가중치 0으로 리셋
        app_state["llm_loader"].reset_lora_weights()

        # --- 1. 초기 상태 → hidden_state와 캐시를 받음 ---
        llm_states = app_state["preprocessor"].get_initial_states(request.prompt)
        generated_ids = llm_states["generated_ids"][:]

        current_llm_hidden_state = llm_states["current_llm_hidden_state"]
        xL_to_encrypt = llm_states["lora_xL_input"] # (Batch, Hidden)
        
        # --- 2. 토큰 생성 루프 ---
        for i in range(request.max_new_tokens):
            print(f"\n  Token {i+1}/{request.max_new_tokens}")

            # --- (3) 클라이언트: hidden_state 암호화 ---
            print("  (3) [Client] Hidden State 암호화 중...")
            enc_xL_bytes = app_state["ckks_manager"].encrypt_tensor(xL_to_encrypt)
            
            # --- (4) 클라이언트: 서버 전송 ---
            print("  (4) [Client] 서버로 암호문 전송 중...")
            req_data = EncryptedInferenceRequest(
                enc_hidden_state_bytes=encode_bytes_to_base64(enc_xL_bytes)
            )
            server_response = app_state["http_session"].post(
                app_state["server_url"], 
                json=model_dump(req_data)   # Pydantic V1/V2 호환
            )
            server_response.raise_for_status() # 오류 시 예외 발생
            
            res_data = model_validate(EncryptedInferenceResponse, server_response.json())
            
            # --- (6) 클라이언트: LoRA 델타 복호화 ---
            print("  (6) [Client] LoRA 델타 복호화 중...")
            enc_lora_output_delta_bytes = decode_base64_to_bytes(res_data.enc_lora_delta_bytes)
            dec_lora_output_delta = app_state["ckks_manager"].decrypt_tensor(enc_lora_output_delta_bytes)
            
            if dec_lora_output_delta.dim() == 1:
                dec_lora_output_delta_2D = dec_lora_output_delta.unsqueeze(0).to(DEVICE)
            else:
                dec_lora_output_delta_2D = dec_lora_output_delta.to(DEVICE)
            
            app.state["llm_loader"].set_global_lora_output_delta(dec_lora_output_delta_2D)
            
            # --- (6) 클라이언트: 다음 토큰 예측 ---
            print("  (6) [Client] 다음 토큰 예측 중...")
            next_token_id, next_token_char = app_state["postprocessor"].integrate_lora_delta_and_predict_token(
                current_llm_hidden_state=current_llm_hidden_state
            )
            
            generated_ids.append(next_token_id)
            print(f"  -> 생성: '{repr(next_token_char)}", end="")
            
            if next_token_id == app_state["llm_loader"].eos_token_id:
                print("\n  EOS 토큰 감지. 생성 종료.")
                break
            
            # --- (7) 클라이언트: 상태 업데이트 → hidden_state와 캐시 업데이트 ---
            llm_states = app_state["preprocessor"].get_next_token_states(next_token_id, llm_states)

            app_state["llm_loader"].clear_global_lora_output_delta()
            current_llm_hidden_state = llm_states["current_llm_hidden_state"]
            xL_to_encrypt = llm_states["lora_xL_input"] # 새 xL (Batch, Hidden)

        # --- 3. 최종 결과 반환 ---
        final_text = app_state["postprocessor"].decode_final_output(generated_ids)
        processing_time = time.time() - start_time

        return ClientBackendResponse(
            generated_text=final_text,
            status="success",
            message=f"LLM 추론 완료. 소요 시간: {processing_time:.2f}초"
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=CLIENT_BACKEND_HOST,
        port=CLIENT_BACKEND_PORT,
        reload=True
    )