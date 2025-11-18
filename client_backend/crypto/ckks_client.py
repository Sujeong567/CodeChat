# client_backend/crypto/ckks_client.py
import numpy as np
import torch
import tenseal as ts

from common.he_context import load_client_context


class CKKSClientManager:
    def __init__(self):
        self.context: ts.Context = load_client_context()

    def encrypt_tensor(self, tensor) -> bytes:
        """
        tensor: torch.Tensor 또는 numpy.ndarray 둘 다 지원
        return: serialized bytes
        """
        # --- numpy 혹은 torch 모두 지원하도록 개선 ---
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy().astype(np.float64)
        elif isinstance(tensor, np.ndarray):
            arr = tensor.astype(np.float64)
        else:
            raise TypeError("encrypt_tensor expects torch.Tensor or numpy.ndarray")

        flat = arr.reshape(-1)
        vec = ts.ckks_vector(self.context, flat.tolist())
        return vec.serialize()

    def decrypt_tensor(self, data: bytes) -> torch.Tensor:
        """
        serialized bytes → CKKS vector → float32 numpy → torch.Tensor
        """
        vec = ts.ckks_vector_from(self.context, data)
        arr = np.array(vec.decrypt(), dtype=np.float32)
        return torch.from_numpy(arr)
