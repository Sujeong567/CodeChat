import os
import json
import numpy as np
from model.embedding_model import EmbeddingModel

# 프로젝트 루트 기준 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dummy_code_snippets.json")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "embedding")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "dummy_code_after_embedding.npy")


def preprocess_embeddings(json_path=RAW_DATA_PATH, save_path=PROCESSED_FILE, model_name="distilbert-base-uncased"):
    # 1️⃣ JSON 파일 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2️⃣ Embedding 모델 초기화
    embedding_model = EmbeddingModel(model_name=model_name)

    # 3️⃣ 문장 단위 임베딩 추출
    embeddings_list = []
    for item in data:
        text = item["text"]
        embedding = embedding_model.get_sentence_embedding(text)
        embeddings_list.append(embedding.numpy())  # torch.Tensor → numpy

    embeddings_array = np.stack(embeddings_list)  # [num_samples, hidden_dim]
    print(f"임베딩 벡터 shape: {embeddings_array.shape}")

    # 4️⃣ 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings_array)
    print(f"임베딩 완료된 데이터 저장: {save_path}")


if __name__ == "__main__":
    preprocess_embeddings()
