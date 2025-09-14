def encode_texts(texts, method="ascii", tokenizer=None):
    """
    문자열 리스트를 숫자 벡터로 변환
    method = "ascii" 또는 "tokenizer"
    tokenizer는 HuggingFace Tokenizer 객체
    """
    if method == "ascii":
        return [[ord(c) for c in text] for text in texts]

    elif method == "tokenizer" and tokenizer is not None:
        return [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    
    else:
        raise ValueError("Invalid encoding method")
    
def decode_vectors(vectors, method="ascii", tokenizer=None):
    """
    숫자 벡터를 문자열로 복원
    """
    if method == "ascii":
        return ["".join([chr(i) for i in vec]) for vec in vectors]
    
    elif method == "tokenizer" and tokenizer is not None:
        return [tokenizer.decode(vec) for vec in vectors]
    
    else:
        raise ValueError("Invalid decoding method")