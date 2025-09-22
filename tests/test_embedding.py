# tests/test_embedding.py

import tenseal as ts
from model.embedding_model import EmbeddingModel


def main():
    # ✅ 모델 선택 (쉽게 교체 가능)
    model_name = "distilbert-base-uncased"
    text = "Secure coding with DistilBERT tokenizer."

    # 모델 초기화
    embedding_model = EmbeddingModel(model_name=model_name)

    # 1. 토큰 단위 임베딩
    token_ids, token_embeddings = embedding_model.get_embeddings(text)
    print("📌 토큰 ID:", token_ids)
    print("📌 임베딩 벡터 shape:", token_embeddings.shape)

    # 2. 문장 단위 평균 pooling
    sentence_embedding = embedding_model.get_sentence_embedding(text)
    print("📌 문장 임베딩 shape:", sentence_embedding.shape)
    print("📌 문장 임베딩 (앞 5개):", sentence_embedding[:5])


if __name__ == "__main__":
    main()