# scripts/preprocess_ckks_enc.py
import os
import numpy as np
import tenseal as ts
from model.fhe_ckks_local import create_context

# 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDING_FILE = os.path.join(PROJECT_ROOT, "data", "embedding", "dummy_code_after_embedding.npy")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ckks_enc")
CONTEXT_FILE = os.path.join(OUTPUT_DIR, "ckks_context.ctx")

def preprocess_ckks_embeddings():
    # 🔹 1. 임베딩 로드
    embeddings = np.load(EMBEDDING_FILE)
    print(f"임베딩 로드 완료: {embeddings.shape}")

    # 🔹 2. CKKS 컨텍스트 생성
    context = create_context()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CONTEXT_FILE, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
    print(f"CKKS 컨텍스트 저장 완료: {CONTEXT_FILE}")

    # 🔹 3. 각 벡터를 암호화 후 저장
    for idx, emb in enumerate(embeddings):
        enc_vec = ts.ckks_vector(context, emb.tolist())
        out_path = os.path.join(OUTPUT_DIR, f"dummy_code_ckks_enc_{idx}.ckks")
        with open(out_path, "wb") as f:
            f.write(enc_vec.serialize())
        print(f"[{idx+1}/{len(embeddings)}] CKKS 암호화 벡터 저장: {out_path}")

    print("✅ 모든 임베딩 CKKS 변환 완료")

if __name__ == "__main__":
    preprocess_ckks_embeddings()
