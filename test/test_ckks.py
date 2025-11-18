# test/test_ckks.py
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from client_backend.crypto.ckks_client import CKKSClientManager


def test_ckks():
    ckks = CKKSClientManager()

    # 랜덤 numpy vector 생성
    v = np.random.randn(4096).astype(np.float64)
    print("원본 일부:", v[:10])

    # 1) 암호화
    enc = ckks.encrypt_tensor(v)

    # 2) 복호화
    dec = ckks.decrypt_tensor(enc)     # torch.Tensor
    dec_np = dec.numpy()               # numpy로 변환

    print("복호화 일부:", dec_np[:10])

    # 3) 차이 계산
    diff = np.abs(v - dec_np)
    print("평균 절대 오차:", diff.mean())
    print("최대 오차:", diff.max())


if __name__ == "__main__":
    test_ckks()

"""
결과:
평균 절대 오차: 1.76e-08
최대 오차: 1.77e-07

>> ckks 암복호화 파이프라인 정상 동작
"""