import os
import sys
import json
import base64
import numpy as np
import tenseal as ts

# 상위 폴더를 import 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from deployment.client import Client

# 경로 설정
RAW_FILE = "data/raw/dummy_code_snippets.json"
EMBEDDING_FILE = "data/embedding/dummy_code_after_embedding.npy"

def test_client_pipeline():

    # 원본 임베딩 로드
    original_embeddings = np.load(EMBEDDING_FILE)

    # 클라이언트 초기화
    client = Client()

    # JSON 파일 읽기
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} snippets from '{RAW_FILE}'")

    # 각 스니펫 처리
    for idx, snippet in enumerate(data, start=1):
        text = snippet.get("text", "")  # JSON 구조에 맞춰 key 수정 가능
        if not text:
            continue

        # 임베딩
        try:
            emb = client.embed_text(text)
            print(f"Embedding done (first 5 values): {[round(x,6) for x in emb[:5]]}")
        except Exception as e:
            print(f"Embedding error: {e}")
            continue

        # CKKS 암호화
        enc_vec = client.encrypt_embedding(emb)

        # 서버 전송 
        resp = client.send_to_server(enc_vec)

        # 서버 응답(base64 -> bytes -> TenSEAL 복원)
        try:
            enc_bytes = base64.b64decode(resp["encrypted_vector"])
            result_vec = ts.ckks_vector_from(client.context, enc_bytes)

            # 복호화
            dec_vec = client.decrypt_embedding(result_vec)

            # 근사 임베딩 찾기
            approx_vec, approx_idx = client.approximate_token_embedding(dec_vec, original_embeddings)

            # 원본 인덱스와 비교
            print(f"[{idx}/{len(data)}] Original snippet: {text[:30]}...")
            print(f"Closest embedding index: {approx_idx}, first 5 values: {[round(x,6) for x in approx_vec[:5]]}")
        except Exception as e:
            print(f"[{idx}/{len(data)}] Error handling response: {e}")

if __name__ == "__main__":
    test_client_pipeline()