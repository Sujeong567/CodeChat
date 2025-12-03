import os
import sys
import time
import gc
import torch
import requests
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
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


app_state = {}


@asynccontextmanager
async def lifespan(app):
    print("[ClientBackend] Starting — loading model and keys")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = BaseLLMLoader()
    loader.load_model()

    ckks = CKKSClientManager()
    session = requests.Session()
    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/compute_lora"

    loader.attach_fhe_client(ckks, session, server_url)

    app_state["loader"] = loader
    app_state["pre"] = LLMPreProcessor(loader)
    app_state["post"] = LLMPostProcessor(loader)
    app_state["ckks"] = ckks
    app_state["session"] = session
    app_state["server_url"] = server_url

    print("[ClientBackend] Ready.")
    yield

    session.close()
    loader.clear_hooks()
    print("[ClientBackend] Shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=ClientBackendResponse)
async def generate(req: ClientBackendRequest):
    t0 = time.time()

    try:
        loader = app_state["loader"]
        pre = app_state["pre"]
        post = app_state["post"]
        ckks = app_state["ckks"]
        session = app_state["session"]
        server_url = app_state["server_url"]

        loader.reset_lora_weights()

        states = pre.get_initial_states(req.prompt)
        gen_ids = states["generated_ids"][:]

        for step in range(req.max_new_tokens):
            print(f"[ClientBackend] Step {step+1}/{req.max_new_tokens}")

            xl_dict = states["xl_dict"]

            # 1) x_L 암호화
            enc_dict = {}
            for key, xl in xl_dict.items():
                vec = xl.squeeze(0).cpu().numpy()
                enc_bytes = ckks.encrypt_tensor(vec)
                enc_dict[key] = encode_bytes_to_base64(enc_bytes)

            # 2) 서버 요청
            enc_req = EncryptedInferenceRequest(enc_dict=enc_dict)
            http_res = session.post(server_url, json=enc_req.model_dump())
            http_res.raise_for_status()
            fhe_res = EncryptedInferenceResponse.model_validate(http_res.json())

            # 3) delta 복호화
            delta_dict = {}
            for key, enc_b64 in fhe_res.enc_delta_dict.items():
                b = decode_base64_to_bytes(enc_b64)
                v = ckks.decrypt_tensor(b)
                delta_dict[key] = v.unsqueeze(0).to(DEVICE)

            # 4) delta 주입 설정
            loader.set_delta(delta_dict)

            # 5) 다음 토큰 생성
            next_id, next_char = post.integrate_lora_delta_and_predict_token(
                states["current_hidden"]
            )
            print(f"  → token: {repr(next_char)}")
            gen_ids.append(next_id)

            if next_id == loader.eos_token_id:
                print("  EOS detected.")
                break

            # 6) forward → same-token hook → next x_L 얻기
            states = pre.get_next_token_states(next_id, states)

            # delta 초기화
            loader.reset_delta()

        txt = post.decode_final_output(gen_ids)
        elapsed = time.time() - t0
        print(f"[ClientBackend] DONE ({elapsed:.2f}s)")
        print(txt[:500])

        return ClientBackendResponse(
            generated_text=txt, status="success", message=f"OK ({elapsed:.2f}s)"
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
