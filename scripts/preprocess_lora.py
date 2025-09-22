# scripts/apply_he_lora_batch.py
import os
import glob
import numpy as np
import tenseal as ts
from model.lora_he_layer import apply_he_lora
from utils.generate_lora_low_matrices import generate_lora_low_rank_matrices 


# 프로젝트 상대 경로 설정 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKKS_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ckks_enc")
CONTEXT_FILE = os.path.join(CKKS_INPUT_DIR, "ckks_context.ctx")
LORA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "lora")

os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

# -----------------------
# 컨텍스트 / ckks vector 로딩 헬퍼 (TenSEAL 버전 차이를 감안한 예외 처리)
# -----------------------
def load_context_from_file(path):
    with open(path, "rb") as f:
        ctx_bytes = f.read()

    # 여러 TenSEAL 버전에서의 로딩 함수명 차이를 처리
    try:
        ctx = ts.context_from(ctx_bytes)
    except Exception:
        # 대안: ts.Context.load 혹은 ts.context_from_serialized
        try:
            ctx = ts.Context.load(ctx_bytes)
        except Exception as e:
            raise RuntimeError(
                "TenSEAL context load failed. Check your TenSEAL version and API. "
                "Tried ts.context_from and ts.Context.load."
            ) from e
    return ctx


def load_ckks_vector_from_file(ctx, path):
    with open(path, "rb") as f:
        vec_bytes = f.read()
    try:
        enc_vec = ts.ckks_vector_from(ctx, vec_bytes)
    except Exception:
        # 대안 시도: ts.CKKSVector.load 등 (버전에 따라)
        try:
            enc_vec = ts.CKKSVector.load(ctx, vec_bytes)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ckks vector from {path}. "
                "Check TenSEAL API for loading serialized ckks vectors."
            ) from e
    return enc_vec

def main(
    *,
    distilbert_hidden_size=768,  # distilbert-base-uncased hidden dim
    rank=8,
    alpha=1.0,
    a_std=0.02,
    seed=None,
    input_dir=CKKS_INPUT_DIR,
    context_file=CONTEXT_FILE,
    output_dir=LORA_OUTPUT_DIR,
):
    # 1) 컨텍스트 로드 (비밀키 포함된 상태여야 암호화 출력 생성/저장 가능)
    print("Loading TenSEAL context from:", context_file)
    context = load_context_from_file(context_file)

    # 2) LoRA 행렬 생성 (A: d x r, B: r x d)
    d = distilbert_hidden_size
    r = rank
    lora_A, lora_B = generate_lora_low_rank_matrices(d, r, std=a_std)



    print(f"LoRA matrices created: A {lora_A.shape}, B {lora_B.shape}")

    # 3) 입력 암호화 벡터 파일 목록
    ckks_files = sorted(glob.glob(os.path.join(input_dir, "*.ckks")))
    if len(ckks_files) == 0:
        print("No .ckks files found in", input_dir)
        return

    print(f"Found {len(ckks_files)} CKKS files. Processing...")

    for idx, ckks_path in enumerate(ckks_files):
        print(f"[{idx+1}/{len(ckks_files)}] Loading {ckks_path}")
        enc_vec = load_ckks_vector_from_file(context, ckks_path)

        # 안전성 검사: 입력 길이 확인 (가능하면)
        try:
            vec_len = enc_vec.size()
        except Exception:
            try:
                vec_len = getattr(enc_vec, "size", None)
            except Exception:
                vec_len = None

        if vec_len is not None and vec_len < d:
            print(f"Warning: encrypted vector slots ({vec_len}) < required dim ({d}).")

        # 4) LoRA 적용
        enc_out_vec = apply_he_lora(enc_vec, lora_A, lora_B, alpha=alpha)

        # 5) 결과 저장: 각 출력 차원별로 개별 ckks 파일로 저장
        # LoRA 적용


        # 결과 저장: 단일 ckks 파일
        out_name = os.path.basename(ckks_path).replace(".ckks", f"_lora_out_{idx}.ckks")
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(enc_out_vec.serialize())
        print(f"Saved LoRA output vector to {out_path}")


    print("All files processed. LoRA outputs saved.")


if __name__ == "__main__":
    # 파라미터를 여기에 직접 설정하거나 CLI 인자로 확장 가능
    main(
        distilbert_hidden_size=768,
        rank=8,
        alpha=1.0,
        a_std=0.02,
        seed=42,
    )
