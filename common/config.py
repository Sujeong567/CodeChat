import os
import torch

# --- 프로젝트 루트 디렉토리 설정 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 통신 설정 ---
CLIENT_BACKEND_HOST = "127.0.0.1"
CLIENT_BACKEND_PORT = 5000

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# --- LLM 모델 설정 ---
LLM_NAME = "TechxGenus/starcoder2-7b-instruct"
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
HIDDEN_SIZE = 4608

# 입력 최대 길이 (프롬프트)
MAX_INPUT_LENGTH = 2048

# --- BitsAndBytes (4비트 양자화) 설정 ---
BNB_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = False

# --- LoRA 설정 ---
R_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# QLoRA 가중치 위치 (서버에서 사용)
LORA_WEIGHTS_DIR = os.path.join(
    PROJECT_ROOT, "server", "lora", "lora_weights_checkpoints_final"
)

# --- FHE 실험용 설정 ---
# same-token FHE-LoRA를 적용할 레이어들
FHE_LAYERS = [15]    # 예: [4, 10, 15, 20]

# same-token FHE-LoRA를 적용할 모듈들
# 항상 "q_proj"는 포함시키고, 원하는 모듈을 추가로 켜서 실험 가능
FHE_MODULES = ["q_proj"]  # 예: ["q_proj", "v_proj"], ["q_proj", "o_proj"]

# --- Homomorphic Encryption (HE) 설정 ---
HE_POLY_MODULUS_DEGREE = 16384
HE_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
HE_GLOBAL_SCALE_BITS = 40

HE_KEY_DIR = os.path.join(PROJECT_ROOT, ".cache", "he_keys")
CLIENT_SECRET_KEY_PATH = os.path.join(HE_KEY_DIR, "client_secret.bin")
SERVER_PUBLIC_KEY_PATH = os.path.join(HE_KEY_DIR, "server_public.bin")

# --- LLM 추론 설정 ---
MAX_GEN_LENGTH = 300
SCALE_INJECTION = LORA_ALPHA / R_RANK

# --- 디바이스 ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 디렉터리 생성 ---
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(HE_KEY_DIR, exist_ok=True)
os.makedirs(LORA_WEIGHTS_DIR, exist_ok=True)
