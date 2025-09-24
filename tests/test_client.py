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

def recall_at_k(true_idx, pred_indicies, k=5):
    """Recall@k 계산"""
    if true_idx is None:
        return None
    return int(true_idx in pred_indicies) / 1.0

def test_client_pipeline(k=5):

    # 원본 임베딩 로드
    original_embeddings = np.load(EMBEDDING_FILE)

    # 클라이언트 초기화
    client = Client()

    # JSON 파일 읽기
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} snippets from '{RAW_FILE}'")

    recalls_pre, recalls_post = [], []

    # 각 스니펫 처리
    for idx, snippet in enumerate(data, start=1):
        text = snippet.get("text", "")
        if not text:
            continue

        # HE 적용 전 Top-k
        emb = client.embed_text(text)
        topk_idx_pre, _ = client.topk_search(emb, original_embeddings, k=k)
        
        true_idx = idx - 1

        recall_pre = recall_at_k(true_idx, topk_idx_pre, k)
        recalls_pre.append(recall_pre)

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

            # HE 적용 후 Top-k
            topk_idx_post, _ = client.topk_search(dec_vec, original_embeddings, k=k)
            recall_post = recall_at_k(true_idx, topk_idx_post, k)
            recalls_post.append(recall_post)

            print(f"[{idx}/{len(data)}] Query: {text[:30]}...")
            print(f"HE 전 Top-{k}: {topk_idx_pre}, Recall@{k}: {recall_pre}")
            print(f"HE 후 Top-{k}: {topk_idx_post}, Recall@{k}: {recall_post}")
            print("-"*60)
    
        except Exception as e:
            print(f"[{idx}/{len(data)}] Error handling response: {e}")
    
    print(f"Average Recall@{k} HE 전: {np.mean(recalls_pre):.3f}")
    print(f"Average Recall@{k} HE 후: {np.mean(recalls_post):.3f}")
          
if __name__ == "__main__":
    test_client_pipeline(k=5)