"""
데이터셋 불러오기 → 텍스트 벡터 변환 → CKKS 암호화 → 복호화 테스트
"""

import json
import base64
from model.fhe_ckks_local import create_context, ckks_encrypt, ckks_decrypt
from model.preprocess import encode_texts, decode_vectors

# JSON 데이터셋 로드
def load_json_dataset(path="data/raw/dummy_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    texts = [item["code"] for item in dataset]
    labels = [item["label"] for item in dataset]
    return texts, labels

def main():
    # 데이터셋 로드
    texts, labels = load_json_dataset()
    print(f"첫 번재 샘플 코드: {texts[0]}")

    # 텍스트 → 벡터 인코딩
    encoded_texts = encode_texts(texts, method="ascii")
    print(f"첫 번째 샘플 벡터화 (앞 10개): {encoded_texts[0][:10]}")

    # ckks 컨텍스트 생성
    context = create_context()

    # 첫 번째 샘플 암호화
    encrypted_vec = ckks_encrypt(encoded_texts[0], context)
    enc_bytes = encrypted_vec.serialize()
    enc_b64 = base64.b64encode(enc_bytes).decode("utf-8")
    print(f"암호화 결과 (앞 80자): {enc_b64[:80]}...")

    # 복호화
    decrypted_vec = ckks_decrypt(encrypted_vec)
    print(f"복호화 결과 (앞 10개): {decrypted_vec[:10]}")

    # float → int 반올림 후 복원
    rounded_vec = [round(x) for x in decrypted_vec]
    restored_text = decode_vectors([rounded_vec], method="ascii")[0]

    print(f"\n원문: {texts[0]}")
    print(f"복원 결과: {restored_text}")

    if texts[0] == restored_text:
        print("복원 성공")
    else:
        print("복원 결과가 원문과 다릅니다.")

if __name__ == "__main__":
    main()