"""
클라이언트 백엔드 메인 서버
- Base LLM 실행
- LoRA 직전까지 연산
- CKKS 암호화/복호화
- 기업 서버와 통신
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import requests
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 공통 설정 import
from common.config import (
    CLIENT_BACKEND_HOST,
    CLIENT_BACKEND_PORT,
    SERVER_HOST,
    SERVER_PORT,
    DEVICE,
    MAX_INPUT_LENGTH,
    MAX_GEN_LENGTH
)

# HE 관련 import
from common.he_utils import (
    create_ckks_context,
    save_tenseal_context,
    load_tenseal_context
)

# 클라이언트 암호화 import
from crypto.ckks_client import ( 
    encrypt_vector,
    decrypt_vector,
    serialize_context
)

# Base LLM 관련 import (이 파일들은 나중에 만들 예정)
# from app.model.base_llm import load_base_llm
# from app.model.preprocessing import preprocess_before_lora
# from app.model.postprocessing import postprocess_after_lora


# ============================================
# FastAPI 앱 생성
# ============================================

app = FastAPI(
    title="CodeChat Client Backend API",
    description="클라이언트 백엔드 - Base LLM 실행 및 암호화 처리",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 전역 변수
# ============================================

base_model = None        # Base LLM
ckks_context = None      # CKKS Context (비밀키 포함)
context_initialized = False


# ============================================
# 서버 시작 이벤트
# ============================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global base_model, ckks_context, context_initialized
    
    print("\n" + "="*70)
    print("🚀 클라이언트 백엔드 시작 중...")
    print("="*70 + "\n")
    
    # 1. CKKS Context 초기화
    print("🔐 Step 1: CKKS Context 초기화")
    print("-" * 70)
    
    try:
        # Context 파일이 있으면 로드, 없으면 생성
        context_file = "client_context_with_secret.bin"
        
        if os.path.exists(os.path.join("common", "he_keys", context_file)):
            print("   기존 Context 파일 발견! 로드 중...")
            ckks_context = load_tenseal_context(context_file)
        else:
            print("   Context 파일 없음. 새로 생성 중...")
            ckks_context = create_ckks_context()
            
            # Context 저장 (비밀키 포함)
            save_tenseal_context(ckks_context, context_file, save_secret_key=True)
            
            # Public Context도 저장 (서버 전송용, 비밀키 제외)
            public_context_file = "public_context.bin"
            save_tenseal_context(ckks_context, public_context_file, save_secret_key=False)
            print(f"   📤 Public Context 저장: {public_context_file}")
        
        context_initialized = True
        print("✅ CKKS Context 준비 완료!\n")
        
    except Exception as e:
        print(f"❌ CKKS 초기화 실패: {e}\n")
        raise
    
    # 2. Base LLM 로딩
    print("📦 Step 2: Base LLM 로딩")
    print("-" * 70)
    
    try:
        # TODO: Base LLM 로드 코드 작성 후 주석 해제
        # print(f"   모델명: {LLM_NAME}")
        # print(f"   장치: {DEVICE}")
        # base_model = load_base_llm()
        # print("✅ Base LLM 로딩 완료!\n")
        
        # 임시 (Base LLM 코드 작성 전)
        print("   ⚠️ Base LLM 로딩 코드 미구현 (TODO)")
        print("   현재는 더미 모델 사용\n")
        base_model = {"status": "dummy"}
        
    except Exception as e:
        print(f"❌ Base LLM 로딩 실패: {e}\n")
        raise
    
    print("="*70)
    print("✅ 클라이언트 백엔드 준비 완료!")
    print(f"🌐 포트: {CLIENT_BACKEND_PORT}")
    print(f"🔐 CKKS: 준비됨")
    print(f"🤖 Base LLM: {'준비됨' if base_model else '미준비'}")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시"""
    print("\n⏹️ 클라이언트 백엔드 종료 중...")
    print("✅ 종료 완료!\n")


# ============================================
# API 요청/응답 모델
# ============================================

class CodeReviewRequest(BaseModel):
    """코드 리뷰 요청"""
    code: str
    language: str = "python"


class CodeReviewResponse(BaseModel):
    """코드 리뷰 응답"""
    status: str
    review: str
    encryption_used: bool
    processing_time: float = 0.0


# ============================================
# API 엔드포인트 - 코드 리뷰
# ============================================

@app.post("/api/review", response_model=CodeReviewResponse)
async def review_code(request: CodeReviewRequest):
    """
    코드 리뷰 전체 프로세스
    
    1. Base LLM으로 LoRA 직전까지 연산 (평문)
    2. 중간 결과 CKKS 암호화
    3. 기업 서버로 전송 (LoRA 연산)
    4. 결과 복호화
    5. 나머지 추론 완료
    """
    
    import time
    start_time = time.time()
    
    try:
        print("\n" + "="*70)
        print("📥 코드 리뷰 요청 수신")
        print("="*70)
        print(f"언어: {request.language}")
        print(f"코드 길이: {len(request.code)} chars\n")
        
        # 1. 전처리 (LoRA 직전까지)
        print("🔄 Step 1/5: Base LLM으로 전처리 중...")
        print("-" * 70)
        
        # TODO: 실제 전처리 코드 작성 후 주석 해제
        # hidden_states = preprocess_before_lora(base_model, request.code)
        
        # 임시 더미 데이터
        import numpy as np
        hidden_states = np.random.randn(1, 512, 768).astype(np.float32)
        print(f"   중간 결과 shape: {hidden_states.shape}")
        print("✅ 전처리 완료!\n")
        
        # 2. 암호화
        print("🔐 Step 2/5: CKKS 암호화 중...")
        print("-" * 70)
        
        if not context_initialized:
            raise HTTPException(status_code=500, detail="CKKS Context가 초기화되지 않았습니다")
        
        encrypted_data = encrypt_vector(ckks_context, hidden_states)
        print("✅ 암호화 완료!\n")
        
        # 3. 기업 서버로 전송
        print("📤 Step 3/5: 기업 서버로 전송 중...")
        print("-" * 70)
        
        server_url = f"http://{SERVER_HOST}:{SERVER_PORT}/api/lora/inference"
        print(f"   서버 주소: {server_url}")
        
        try:
            response = requests.post(
                server_url,
                json={
                    "encrypted_hidden_states": encrypted_data["ciphertext"],
                    "size": encrypted_data["size"],
                    "shape": encrypted_data["shape"]
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"서버 응답 에러: {response.text}"
                )
            
            lora_result = response.json()
            print("✅ LoRA 연산 결과 수신!\n")
            
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail=f"기업 서버 연결 실패. 서버가 실행 중인지 확인하세요 ({server_url})"
            )
        
        # 4. 복호화
        print("🔓 Step 4/5: 결과 복호화 중...")
        print("-" * 70)
        
        decrypted_result = decrypt_vector(ckks_context, lora_result)
        print("✅ 복호화 완료!\n")
        
        # 5. 후처리
        print("✍️ Step 5/5: 최종 리뷰 생성 중...")
        print("-" * 70)
        
        # TODO: 실제 후처리 코드 작성 후 주석 해제
        # final_review = postprocess_after_lora(base_model, decrypted_result, hidden_states)
        
        # 임시 더미 결과
        final_review = f"""코드 리뷰 결과:

1. ✅ 변수명이 명확합니다
2. ⚠️ 함수 docstring 추가를 권장합니다
3. ✅ 코드 구조가 깔끔합니다

암호화 처리 완료: CKKS 사용됨
"""
        
        processing_time = time.time() - start_time
        print(f"✅ 리뷰 생성 완료! (소요 시간: {processing_time:.2f}초)\n")
        
        print("="*70 + "\n")
        
        return CodeReviewResponse(
            status="success",
            review=final_review,
            encryption_used=True,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# API 엔드포인트 - 상태 확인
# ============================================

@app.get("/api/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "server_type": "Client Backend",
        "base_model_loaded": base_model is not None,
        "ckks_ready": context_initialized,
        "device": DEVICE,
        "port": CLIENT_BACKEND_PORT
    }


@app.get("/api/context/info")
async def context_info():
    """CKKS Context 정보"""
    if not context_initialized:
        raise HTTPException(status_code=500, detail="Context가 초기화되지 않았습니다")
    
    return {
        "poly_modulus_degree": ckks_context.poly_modulus_degree,
        "global_scale": int(ckks_context.global_scale),
        "has_secret_key": True,
        "status": "initialized"
    }


# ============================================
# 루트 엔드포인트
# ============================================

@app.get("/")
async def root():
    """루트 경로"""
    return {
        "message": "CodeChat 클라이언트 백엔드",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "POST /api/review",
            "GET /api/health",
            "GET /api/context/info"
        ]
    }


# ============================================
# 서버 실행
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 클라이언트 백엔드 서버 시작 중...")
    print(f"📍 호스트: {CLIENT_BACKEND_HOST}")
    print(f"📍 포트: {CLIENT_BACKEND_PORT}")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host=CLIENT_BACKEND_HOST,
        port=CLIENT_BACKEND_PORT,
        log_level="info"
    )
