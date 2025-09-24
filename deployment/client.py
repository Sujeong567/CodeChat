import os
import base64
import numpy as np
import tenseal as ts
import requests
from model.embedding_model import EmbeddingModel
from model.fhe_ckks_local import ckks_encrypt

# -------------------------
# 클라이언트 클래스
# -------------------------
class Client:

    def __init__(self, model_name="distilbert-base-uncased", 
                 context_file="data/ckks_enc/ckks_context.ctx", 
                 server_url="http://localhost:8000/process"):
        """
        param model_name : 임베딩 모델 이름
        param context_file : CKKS 컨텍스트 파일 경로
        param server_url : 서버 엔드포인트 URL
        """
        self.model = EmbeddingModel(model_name=model_name)
        self.context_file = context_file
        self.context = self._load_context(context_file)
        self.server_url = server_url
    
    def _load_context(self, path):
        """CKKS 컨텍스트 로드"""
        with open(path, "rb") as f:
            ctx_bytes = f.read()
        try:
            ctx = ts.context_from(ctx_bytes)
        except Exception:
            ctx = ts.Context.load(ctx_bytes)
        return ctx
    
    def embed_text(self, text: str) -> np.ndarray:
        """텍스트 -> 임베딩 벡터"""
        return self.model.get_sentence_embedding(text).numpy()

    def encrypt_embedding(self, embedding: np.ndarray):
        """임베딩 벡터 -> CKKS 암호화"""
        return ckks_encrypt(embedding.tolist(), self.context)
    
    def decrypt_embedding(self, enc_vec) -> np.ndarray:
        """CKKS 복호화 -> numpy 벡터"""
        return np.array(enc_vec.decrypt())
    
    def topk_search(self, query_vec, dataset_embeddings, k=5):
        """코사인 유사도로 Top-k 인덱스와 유사도 반환"""
        # 정규화
        norm_query = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        norm_dataset = dataset_embeddings / (np.linalg.norm(dataset_embeddings, axis=1, keepdims=True) + 1e-8)

        # 코사인 유사도 계산
        sims = norm_dataset @ norm_query
        topk_idx = np.argsort(sims)[::-1][:k]
        return topk_idx, sims[topk_idx]
    
    def send_to_server(self, enc_vec) -> dict:
        """
        서버로 CKKS 벡터 전송 후 결과 반환
        param enc_vec : CKKS 암호화 벡터
        return : 서버 응답(JSON)
        """
        # enc_vec 직렬화 -> bytes -> base64 문자열
        payload = {
            "encrypted_vector": base64.b64encode(enc_vec.serialize()).decode("utf-8")
        }

        try:
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[ERROR] 서버 전송 실패: {e}")
            return {}