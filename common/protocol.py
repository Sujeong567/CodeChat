# common/protocol.py
from pydantic import BaseModel
import base64

class EncryptedInferenceRequest(BaseModel):
    """
    enc_dict:
        {
            "(15,'q_proj')" : "<base64 bytes>",
            "(15,'v_proj')" : "<base64 bytes>"
        }
    """
    enc_dict: dict[str, str]


class EncryptedInferenceResponse(BaseModel):
    """
    enc_delta_dict:
        {
            "(15,'q_proj')" : "<base64 bytes>",
            "(15,'v_proj')" : "<base64 bytes>"
        }
    """
    enc_delta_dict: dict[str, str]


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