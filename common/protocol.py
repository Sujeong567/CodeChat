# common/protocol.py

from pydantic import BaseModel
import base64

# --- API Request/Response 모델 정의 ---

class EncryptedInferenceRequest(BaseModel):
    """
    클라이언트 백엔드에서 엔터프라이즈 서버로 전송하는 암호화된 추론 요청.
    """
    enc_hidden_state_bytes: str # Base64 인코딩된 암호화된 Hidden State (bytes)
    # (Optional) public_context_bytes: str # 클라이언트가 서버에 Public Context를 전달해야 할 경우

class EncryptedInferenceResponse(BaseModel):
    """
    엔터프라이즈 서버에서 클라이언트 백엔드로 반환하는 암호화된 추론 응답.
    """
    enc_lora_delta_bytes: str # Base64 인코딩된 암호화된 LoRA 델타 (bytes)

class ClientBackendRequest(BaseModel):
    """
    React UI에서 클라이언트 백엔드로 전송하는 LLM 추론 요청.
    """
    prompt: str

class ClientBackendResponse(BaseModel):
    """
    클라이언트 백엔드에서 React UI로 반환하는 LLM 추론 결과.
    """
    generated_text: str
    status: str = "success"
    message: str = ""

# --- 데이터 직렬화/역직렬화 유틸리티 ---
# Flask/FastAPI는 JSON을 기본으로 하므로, 바이너리 데이터는 Base64 인코딩/디코딩 필요.

def encode_bytes_to_base64(data_bytes: bytes) -> str:
    """바이트 데이터를 Base64 문자열로 인코딩합니다."""
    return base64.b64encode(data_bytes).decode('utf-8')

def decode_base64_to_bytes(data_b64_str: str) -> bytes:
    """Base64 문자열을 바이트 데이터로 디코딩합니다."""
    return base64.b64decode(data_b64_str.encode('utf-8'))