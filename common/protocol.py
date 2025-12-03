# common/protocol.py
from pydantic import BaseModel
import base64

class EncryptedInferenceRequest(BaseModel):
    layer_idx: int
    module_name: str      # "q_proj", "k_proj", "v_proj", "o_proj"
    enc_hidden_state_bytes: str  # Base64 인코딩 bytes


class EncryptedInferenceResponse(BaseModel):
    enc_lora_delta_bytes: str    # Base64 인코딩 bytes


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
