# model/embedding_model.py

import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        """
        모델과 토크나이저를 초기화합니다.
        :param model_name: 허깅페이스 모델 이름
        :param device: 'cuda' 또는 'cpu'. None이면 자동 선택
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 토크나이저 & 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def get_embeddings(self, text: str):
        """
        입력 텍스트에 대한 토큰 ID와 임베딩 벡터 반환
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            token_embeddings = self.model.get_input_embeddings()(inputs["input_ids"])
        
        return inputs["input_ids"], token_embeddings

    def get_sentence_embedding(self, text: str):
        """
        문장 단위 평균 pooling → 하나의 벡터로 반환
        """
        _, token_embeddings = self.get_embeddings(text)
        # [batch, seq_len, hidden_dim] → [hidden_dim]
        sentence_embedding = token_embeddings.mean(dim=1).squeeze(0).cpu()
        return sentence_embedding

