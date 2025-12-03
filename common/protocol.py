# common/protocol.py
from pydantic import BaseModel
import base64
from typing import List

class EncryptedInferenceRequest(BaseModel):
    enc_hidden_state_bytes: str  # Base64 인코딩된 bytes


class EncryptedInferenceResponse(BaseModel):
    enc_lora_delta_bytes: str


class ClientBackendRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


class ClientBackendResponse(BaseModel):
    generated_text: str
    status: str = "success"
    message: str = ""


def encode_bytes_to_base64(data_bytes: bytes) -> str:
    return base64.b64encode(data_bytes).decode("utf-8")


def decode_base64_to_bytes(data_b64_str: str) -> bytes:
    return base64.b64decode(data_b64_str.encode("utf-8"))

class PlaintextInferenceRequest(BaseModel):
    """
    평문 Hidden State 벡터를 실수 리스트 형태로 전송합니다.
    """
    hidden_state_vec: List[float] # 암호화 대신 평문 실수 리스트


class PlaintextInferenceResponse(BaseModel):
    """
    평문 LoRA Delta 벡터를 실수 리스트 형태로 반환합니다.
    """
    lora_delta_vec: List[float] # 암호화 대신 평문 실수 리스트