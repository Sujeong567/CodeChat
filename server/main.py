"""
기업 서버 메인
- LoRA 가중치 보유
- 암호화된 hidden states 수신
- LoRA 연산 (암호문 상태)
- 결과 반환
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import List

# ============================================
# Python 경로 설정
# ============================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(f"📂 Current Dir: {current_dir}")
print(f"📂 Project Root: {project_root}\n")

# ============================================
# Import
# ============================================

# 공통 설정
from common.config import (
    SERVER_HOST,
    SERVER_PORT,
    DEVICE
)

# HE 관련
from common.he_utils import load_tenseal_context

# LoRA 관련 (TODO: 실제 구현 필요)
# from lora.adapter import load_lora_adapter
# from lora.inference import lora_inference_encrypted

# ============================================
# FastAPI 앱
# ============================================

app = FastAPI(
    title="CodeChat Enterprise Server",
    description="기업 서버 - LoRA 가중치 보유 및 암호화 연산",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 전역 변수
# ============================================

lora_adapter = None          # LoRA 가중치
public_context = None        # CKKS Public Context (비밀키 없음)
server_initialized = False

# ============================================
# 서버 시작 이벤트
# ============================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global lora_adapter, public_context, server_initialized
    
    print("\n" + "="*70)
    print("🏢 기업 서버 시작 중...")
    print("="*70 + "\n")
    
    # 1. Public CKKS Context 로드 (main.py에서!)
    try:
        public_context = load_tenseal_context("public_context.bin")
        print("✅ Public Context 로드 완료!\n")
    except Exception as e:
        print(f"❌ Public Context 로드 실패: {e}\n")
    
    # 2. LoRA 어댑터 로드 (adapter.py 사용)
    try:
        from lora.adapter import load_lora_adapter
        lora_adapter = load_lora_adapter("./models/lora_weights/checkpoint-final")
        print("✅ LoRA 어댑터 로드 완료!\n")
    except Exception as e:
        print(f"❌ LoRA 로드 실패: {e}\n")


# ============================================
# API - LoRA 추론
# ============================================

@app.post("/api/lora/inference")
async def lora_inference_endpoint(request: LoRAInferenceRequest):
    """
    암호화된 hidden states로 LoRA 연산
    """
    
    try:
        # 1. 암호문 복원
        import tenseal as ts
        serialized = bytes(request.encrypted_hidden_states)
        encrypted_vector = ts.ckks_vector_from(public_context, serialized)
        
        # 2. LoRA 연산 (inference.py 사용, public_context 전달!)
        from lora.inference import lora_inference_encrypted
        result_encrypted = lora_inference_encrypted(
            encrypted_vector,
            lora_adapter,
            public_context  # ← 여기서 전달!
        )
        
        # 3. 결과 직렬화
        result_serialized = result_encrypted.serialize()
        result_bytes = list(result_serialized)
        
        return LoRAInferenceResponse(
            status="success",
            ciphertext=result_bytes,
            size=request.size,
            shape=request.shape,
            message="LoRA 연산 완료"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))