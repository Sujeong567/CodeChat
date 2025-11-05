# common/config.py
import os
import torch

# --- 프로젝트 루트 디렉토리 설정 ---
# 이 경로를 기준으로 상대 경로를 계산합니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 통신 설정 ---
CLIENT_BACKEND_HOST = "127.0.0.1" # 클라이언트 로컬 서버의 IP 주소 (React UI가 접속)
CLIENT_BACKEND_PORT = 5000        # 클라이언트 로컬 서버의 포트

SERVER_HOST = "127.0.0.1"        # 엔터프라이즈 서버의 IP 주소 (클라이언트 백엔드가 접속)
SERVER_PORT = 8000               # 엔터프라이즈 서버의 포트

# --- LLM 모델 설정 ---
LLM_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache") # Hugging Face 모델 캐시 경로

# --- BitsAndBytes (4비트 양자화) 설정 ---
# LLM 로딩 시 compute_dtype은 bfloat16이 최적이지만, FHE 통합 과정에서 float32 연산이 많아짐.
# 모델 로딩 자체는 bfloat16으로 하여 GPU 메모리 효율 유지.
BNB_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = False

# --- LoRA 설정 ---
R_RANK = 8                 # LoRA 랭크 (저랭크 행렬의 랭크)
LORA_ALPHA_FACTOR = R_RANK * 2 # LoRA alpha (스케일링 팩터)
LORA_DROPOUT = 0.0         # LoRA 드롭아웃 비율
LORA_TARGET_MODULES = ["q_proj"] # LoRA를 적용할 레이어 목록
LORA_BIAS = "none"         # LoRA 편향 설정 ("none", "all", "lora_only")
LORA_TASK_TYPE = "CAUSAL_LM" # LoRA 태스크 타입

# --- Homomorphic Encryption (HE) 설정 ---
HE_SCHEME_TYPE = "CKKS"
HE_POLY_MODULUS_DEGREE = 8192
HE_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60] # 두 번의 곱셈 연산 깊이 (X@A, (X@A)@B)
HE_GLOBAL_SCALE_BITS = 40                 # FHE 연산의 전역 스케일 팩터 (2**40)
HE_KEY_DIR = os.path.join(PROJECT_ROOT, "client-backend", "crypto", "keys") # HE 키 파일 저장 디렉토리
# --- LLM 추론 설정 ---
MAX_GEN_LENGTH = 50 # 각 요청에 대해 생성할 최대 토큰 수
MAX_INPUT_LENGTH = 512 # 입력 프롬프트의 최대 토큰 길이
SCALE_FACTOR_FOR_LM_INPUT_INJECTION = 1e-5 # FHE LoRA 델타를 LM Head 입력에 주입할 스케일 팩터 (매우 작게 설정하여 미세한 영향만 주도록)

# --- 통신 및 내부 유틸리티 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # PyTorch 연산 장치

# 필요한 디렉토리 생성
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(HE_KEY_DIR, exist_ok=True)