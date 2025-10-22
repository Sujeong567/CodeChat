import os
import glob
import numpy as np

# 프로젝트 상대 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "dec")

# -----------------------
# 벡터를 텍스트로 변환하기 위한 (가상의) 코퍼스 및 유틸리티
# -----------------------

# 주의: 실제 임베딩 디코딩은 매우 복잡하며, 이 코드는 개념 증명용입니다.
# LoRA가 적용된 벡터는 원본 임베딩 공간에서 이동했기 때문에, 
# 기존 코퍼스만으로는 정확한 원본 텍스트를 찾을 수 없습니다.
# 여기서는 간단한 데모를 위해 더미 코퍼스를 사용합니다.

# DistilBERT의 hidden_size(768)에 맞춘 더미 임베딩 코퍼스
DUMMY_CORPUS = {
    "print('Hello, world!')": np.random.rand(768),
    "for i in range(5):\n    print(i)": np.random.rand(768),
    "x = 10\nif x > 5:\n    print('big')": np.random.rand(768),
    "def add(a, b):\n    return a + b": np.random.rand(768),
    "while True:\n    break": np.random.rand(768),
    "numbers = [1, 2, 3]\nfor n in numbers:\n    print(n)": np.random.rand(768),
    "try:\n    val = int('abc')\nexcept ValueError:\n    print('error')": np.random.rand(768),
    "with open('test.txt', 'w') as f:\n    f.write('data')": np.random.rand(768),
    "import math\nprint(math.sqrt(16))": np.random.rand(768),
    "class Dog:\n    def bark(self):\n        print('woof')": np.random.rand(768),
    "if x == 5: return True": np.random.rand(768),
    "def foo()\n    print('missing colon')": np.random.rand(768),
    "print(unknown_var)": np.random.rand(768),
    "for i in range(3): print(i)": np.random.rand(768),
    "x = [i*i for i in range(5)]": np.random.rand(768),
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)": np.random.rand(768),
    "lambda x: x**2": np.random.rand(768),
    "import os\nos.listdir('.')": np.random.rand(768),
    "def greet(name):\n    print(f'Hello {name}')": np.random.rand(768),
    "raise Exception('manual error')": np.random.rand(768),
}

def normalize_vector(vec):
    """L2 정규화."""
    # float32 또는 float64로 변환하여 NumPy 경고 방지 및 안정적인 계산 보장
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(a, b):
    """코사인 유사도 계산."""
    a = normalize_vector(a)
    b = normalize_vector(b)
    # NumPy dot product
    return np.dot(a, b)

def find_closest_text(query_vector, corpus_embeddings):
    """쿼리 벡터와 가장 유사한 코퍼스 텍스트를 찾습니다."""
    best_match_text = "Not Found"
    highest_similarity = -1.0
    
    # 쿼리 벡터를 정규화합니다.
    query_vector = normalize_vector(query_vector)

    for text, embedding in corpus_embeddings.items():
        # 임베딩이 쿼리 벡터와 같은 차원인지 확인
        if len(query_vector) != len(embedding):
             continue
             
        similarity = cosine_similarity(query_vector, embedding)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_text = text
            
    return best_match_text, highest_similarity


def main(input_dir=DEC_OUTPUT_DIR):
    # 실제 코퍼스 임베딩을 시뮬레이션하기 위해 더미 코퍼스 임베딩을 정규화합니다.
    corpus_embeddings = {
        text: normalize_vector(embedding)
        for text, embedding in DUMMY_CORPUS.items()
    }
    
    print("--- Vector-to-Text Decoding Demonstration ---")
    print("경고: 이것은 작고 가상의 코퍼스를 사용하는 개념 증명 데모입니다.")
    print("      HE-처리된 임베딩을 텍스트로 직접 역변환하는 것은 불가능합니다.")
    print(f"복호화된 벡터 로드 디렉토리: {input_dir}")

    # 복호화된 텍스트 파일 목록
    dec_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    
    if len(dec_files) == 0:
        print(f"오류: 복호화된 텍스트 파일 (*.txt)이 {input_dir}에서 발견되지 않았습니다. preprocess_decrypt.py를 먼저 실행하세요.")
        return

    print(f"디코딩할 {len(dec_files)}개의 복호화된 벡터 파일을 찾았습니다.")

    for idx, dec_path in enumerate(dec_files):
        filename = os.path.basename(dec_path)
        print(f"\n[{idx+1}/{len(dec_files)}] 디코딩 중 {filename}...")
        
        try:
            # 1. 복호화된 벡터 로드
            decrypted_vector = np.loadtxt(dec_path)
            
            # 2. 가장 가까운 텍스트 찾기
            match_text, similarity = find_closest_text(decrypted_vector, corpus_embeddings)
            
            # 3. 결과 출력
            print(f"  > 입력 벡터 길이: {len(decrypted_vector)}")
            print(f"  > 가장 가까운 텍스트 매칭: '{match_text}'")
            print(f"  > 코사인 유사도: {similarity:.4f}")
            
        except Exception as e:
            print(f"  > {filename} 디코딩 실패: {e}")
            continue

    print("\n--- 디코딩 시연 완료 ---")


if __name__ == "__main__":
    # DUMMY_CORPUS의 랜덤 벡터를 768 차원으로 통일
    # 실행 시 매번 새로운 랜덤 벡터를 생성하여 데모의 의미론적 정확도는 낮지만, 
    # 로직의 작동 방식은 시연할 수 있습니다.
    for key in DUMMY_CORPUS:
         DUMMY_CORPUS[key] = np.random.rand(768)

    main()
