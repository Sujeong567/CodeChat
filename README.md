# poppop-PLL-FHE: Secure Inference with LoRA & Homomorphic Encryption

본 프로젝트는 **Java + Spring** 기반의 팀 프로젝트 poppop의 구조를 참고해, 첨부 논문 "Practical Secure Inference Algorithm for Fine-tuned Large Language Model Based on Fully Homomorphic Encryption"을 실습 목표로 합니다. **VSCode** 환경을 기본으로 하며, LoRA 기반 모델 파인튜닝 후 Fully Homomorphic Encryption(FHE)를 적용한 Private Linear Layer(PLL)로 안전한 추론 시스템을 구현합니다.

***

## 🗂️ 프로젝트 구조 및 역할

```
/LoRA-AutoFix
│
├── /data                       # 학습 데이터 및 변환 페어 저장
│   ├── before/                 # 변환 전 코드 샘플 저장소
│   ├── after/                  # 변환 후 코드 샘플 저장소
│   └── pairs.json              # "Before → After" 페어 매핑 파일 (JSON, CSV 등)
│
├── /autofix                    # 규칙 기반 자동 변환 스크립트 및 도구
│   ├── generate_pairs.py       # 오픈소스 코드 + 사내 규칙 적용하여 페어 생성 스크립트
│   ├── rules/                  # 변환 규칙 스크립트 (ex. Semgrep 룰, AST 변환 모듈)
│   └── utils.py                # 코드 파싱/분석/수정 헬퍼 함수들
│
├── /model                      # Code Llama 모델 관리 및 LoRA 미세조정 관련 코드
│   ├── train_LoRA.py           # LoRA 미세조정 학습 스크립트
│   ├── infer_prompt.py         # 프롬프트 기반 테스트 코드 (1단계 테스트용)
│   ├── config.py               # 학습 및 추론 설정
│   └── utils.py                # 모델 입출력 전처리 및 후처리
│
├── /deployment                 # 내부 서버용 Code Llama 배포 및 실행 관련 파일
│   ├── serve_model.py          # 내부 서버에서 모델 실행용 스크립트
│   ├── requirements.txt        # 배포 환경 종속성
│   └── README.md               # 배포/운영 가이드
│
├── /tests                      # 단위 테스트 및 통합 테스트
│   ├── test_generate_pairs.py  # 변환 페어 생성 테스트
│   ├── test_rules.py           # 규칙 스크립트 테스트
│   ├── test_lora_training.py   # LoRA 학습 테스트
│   └── test_infer_prompt.py    # 프롬프트 테스트
│
├── README.md                   # 프로젝트 개요, 설치 및 실행 방법, 개발 가이드
├── LICENSE                    # 라이선스 파일
└── .gitignore                  # Git 무시 설정

```

***

## 🌱 개발/실행 환경

- **VSCode** (권장: Remote-WSL, Dev Containers)
- Python 3.10+ (모델, FHE 연동)
- Microsoft SEAL (C++ 라이브러리)
- seal-python (SEAL Python 바인딩)  

***

## 📦 초기 설정

### 1. Git, 줄바꿈(Windows/Mac)

**Windows:**
```bash
git config --global core.autocrlf true
```

**macOS/Linux:**
```bash
git config --global core.autocrlf input
```

`.gitattributes` 파일에:
```
* text=auto
```

### 2. `.gitignore` 예시

```
# OS/IDE
.DS_Store
.idea/
*.iml
out/

# Build & Logs
/build
/logs
*.log

# 환경설정
application-*.yml
.env
```

***

## 🌿 Git 브랜치 전략

- `main`: 최종(배포)
- `develop`: 통합(기능 안정화)
- `feat/*`: 기능 개발
- `fix/*`: 버그수정

예시:
- `feat/implement-fhe-api`
- `fix/fhe-serialization-issue`

***

## 🔄 Git Push/Pull 규칙

- **브랜치 전략**
  - `main`: 최종(배포) — **직접 푸시 금지**
  - `develop`: 기능 통합/안정화 — **직접 푸시 금지**
  - `feat/*`, `fix/*`, `refactor/*`: 기능·버그·리팩터링 작업 브랜치

- **작업 흐름**
  1. 최신 `develop`에서 새 브랜치 생성
     ```bash
     git checkout develop
     git pull origin develop
     git checkout -b feat/your-feature
     ```
  2. 작업 및 커밋
     ```bash
     git add .
     git commit -m "[FEAT] 기능 설명"
     git push origin feat/your-feature
     ```
  3. Pull Request 생성 → 리뷰/CI 통과 → `develop` 병합 (Squash Merge 권장)

- **Pull 규칙**
  - 작업 전·PR 전 **반드시 최신 develop 반영**
    ```bash
    git fetch origin
    git rebase origin/develop
    ```

- **Push 규칙**
  - `main`/`develop` 직접 푸시 금지
  - 강제 푸시(`--force`) 금지 (개인 브랜치에서 예외 시 팀 동의 필요)

- **커밋 규칙**
  - 형식: `[TAG] 요약` (예: `[FIX] SEAL 바인딩 오류 수정`)
  - 태그: FEAT, FIX, REFACTOR, DOCS, STYLE, TEST


## 📝 커밋 메시지 컨벤션

```
[태그] 작업 요약
```

| 태그        | 의미                     |
|-------------|--------------------------|
| [FEAT]      | 새로운 기능 추가         |
| [FIX]       | 버그 수정                |
| [REFACTOR]  | 리팩토링(기능 변화 없음) |
| [STYLE]     | 코드 스타일/포맷         |
| [DOCS]      | 문서 작업                |
| [TEST]      | 테스트 코드              |

예시:
```
git commit -m "[FEAT] LoRA 기반 PLL 암호화 레이어 구현"
git commit -m "[FIX] SEAL 바인딩 호환성 오류 수정"
```

***

## 🛠️ 주요 기술 및 구현 안내

### 1. LoRA 기반 fine-tuning
- open-source pre-trained LLM(예: ChatGLM2-6B, Llama 등) + LoRA 방식으로 파인튜닝.
- `lora/` 디렉토리 내 파인튜닝 스크립트 제공.

### 2. Fully Homomorphic Encryption(동형암호)
- `fhe/` 폴더에 SEAL, seal-python 활용 암호화 모듈 구현.
- 클라이언트: 평문 추출→암호화→서버 전송
- 서버: 암호문 상태 추론/선형변환(Private Linear Layer)
- 논문 방식의 PLL 변환 구현: 모델 추출 내성 활용.

### 3. 추론 및 프라이버시 보호
- Controller/service를 통해 REST API 등으로 암호화/복호화 기반 추론 서비스
- FHE 구현/병렬처리, 통신 최적화(Protocol Buffers 추천)
- 성능 정보: 논문 기준 약 1.61s/token(실측 환경 조건 상이)

***

## 🔄 개발 워크플로우 예시

1. **LoRA로 파인튜닝**
   - `/lora/finetune.py` 등 활용하여 베이스 모델+도메인 데이터로 파인튜닝
   - 파생된 LoRA 체크포인트는 서버에 업로드

2. **FHE(암호화) 환경 구축**
   - `/fhe/` 내 SEAL python 모듈 설치 및
     - 암호키 생성
     - 입력 데이터 암호화, 서버 전송
     - PLL 변환 + FHE 연산
     - 복호화 결과 반환

3. **API/Service 연동**
   - `/controller/`, `/service/`에 Spring REST API, 추론/암호화 서비스 클래스로 연결

4. **(선택) DB 연동**
   - 주요 입력/결과, 모델 메타데이터 저장: `/repository/`, `/domain/`

***

## 📝 참고/실행 예시

### Spring Controller 예시
```java
@PostMapping("/predict")
public PredictionResult predict(@RequestBody InputDto input) {
    // 1. 입력 평문→암호문 변환
    // 2. 서버 측 PLL-FHE 처리 요청
    // 3. 결과 복호화 및 반환
}
```

### Python (seal-python) 추론 암호화 예시
```python
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, Ciphertext

# 파라미터 설정, 키 생성...
# 입력 벡터 암호화 및 서버 연동...
```

***

## 📜 논문 구현 핵심 참고사항

- **공개/비공개 모델 분리:** 공개(base) 모델은 로컬, Private-LoRA는 서버에서 관리.
- **Private Linear Layer 적용:** 서버 LoRA 레이어를 안전하게 보호, LWE/CLWE 기반의 모델 추출 저항.
- **FHE-CKKS 사용:** 신뢰성 있는 수치 계산, 벡터화, 병렬처리 지원.
- **성능 최적화:** 통신 최소화, 병렬처리(실제 논문은 C++-thread pool+protobuf 추천)

***

## 📚 참고 논문

Zhang Ruoyan, Zheng Zhongxiang, Bao Wankang, “Practical Secure Inference Algorithm for Fine-tuned Large Language Model Based on Fully Homomorphic Encryption,” arXiv:2501.01672v2 [cs.CR], 2025.[1]

***

