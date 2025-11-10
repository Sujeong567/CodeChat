"""
클라이언트에서 암호화(중간값), 복호화(LoRA 연산 후) 담당
"""
import torch
import tenseal as ts

from common.config import(
    DEVICE, HE_GLOBAL_SCALE_BITS, HE_POLY_MODULUS_DEGREE
)

from common.he_utils import load_tenseal_context

class CKKSClientManager:
    def __init__(self):
        print("[CKKSClient] 클라이언트 HE Context 로드 중...")

        self.context = load_tenseal_context()

        self.scale = 2**HE_GLOBAL_SCALE_BITS
        self.max_slots = HE_POLY_MODULUS_DEGREE / 2

    def encrypt_tensor(self, data_tensor: torch.Tensor) -> bytes:
        """torch.Tensor(xL 벡터)를 암호화하여 직렬화된 bytes로 반환. 이때 xL은 [Batch, Hidden] 형태"""
        data_list = data_tensor.to(torch.float32).flatten().tolist()

        if len(data_list) > self.max_slots:
            raise ValueError(
                f"암호화하려는 텐서 크기({len(data_list)})가 FHE 최대 슬롯({self.max_slots})보다 큽니다."
            )
    
        print(f"벡터 암호화 중... (크기: {len(data_list)})")
        enc_vector = ts.ckks_vector(self.context, data_list, scale=self.scale)
        serialized_bytes = enc_vector.serialize()
        print("암호화 완료")
    
        return serialized_bytes

    def decrypt_tensor(self, enc_bytes: bytes) -> torch.Tensor:
        """서버에서 받은 bytes를 복호화하여 torch.Tensor로 반환"""
        print("벡터 복호화 중...")
        enc_vector = ts.ckks_vector_from(self.context, enc_bytes)
        dec_list = enc_vector.decrypt()

        # 복호화 결과는 [HiddenDim] 1D 텐서로 반환
        decrypted_tensor = torch.tensor(dec_list, dtype=torch.float32, device=DEVICE)
        print("복호화 완료.")