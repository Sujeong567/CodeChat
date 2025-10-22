import os
import glob
import tenseal as ts
import numpy as np

# 프로젝트 상대 경로 설정 (apply_he_lora_batch.py와 동일)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKKS_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "ckks_enc")
CONTEXT_FILE = os.path.join(CKKS_INPUT_DIR, "ckks_context.ctx")
LORA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "lora")
# 새 복호화 결과 저장 디렉토리 추가
DEC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "dec") 

# -----------------------
# 컨텍스트 / ckks vector 로딩 헬퍼 (apply_he_lora_batch.py에서 재사용)
# -----------------------
def load_context_from_file(path):
    """지정된 경로에서 TenSEAL Context를 로드합니다."""
    with open(path, "rb") as f:
        ctx_bytes = f.read()
    
    # TenSEAL 버전 차이를 감안한 로딩 시도
    try:
        ctx = ts.context_from(ctx_bytes)
    except Exception:
        try:
            ctx = ts.Context.load(ctx_bytes)
        except Exception as e:
            raise RuntimeError(
                "TenSEAL context load failed. Check your TenSEAL version and API."
            ) from e
    return ctx


def load_ckks_vector_from_file(ctx, path):
    """지정된 경로에서 암호화된 CKKSVector를 로드합니다."""
    with open(path, "rb") as f:
        vec_bytes = f.read()
    try:
        enc_vec = ts.ckks_vector_from(ctx, vec_bytes)
    except Exception:
        try:
            enc_vec = ts.CKKSVector.load(ctx, vec_bytes)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ckks vector from {path}."
            ) from e
    return enc_vec


def ckks_decrypt(encrypted_vector, decryption_context):
    """
    암호화된 벡터를 복호화합니다.
    decryption_context는 반드시 비밀키를 포함해야 합니다.
    """
    # 암호문 객체(encrypted_vector)는 암호화에 사용된 Context를 참조하지만,
    # 그 Context에 비밀키가 제거되었을 수 있으므로, 복호화 시 
    # 비밀키를 가진 Context의 secret_key를 명시적으로 전달합니다.
    # 이 방식은 가장 안전하고 호환성이 높습니다.
    if decryption_context.secret_key() is None:
         raise ValueError("Decryption context does not contain a secret key. Cannot decrypt.")
         
    return encrypted_vector.decrypt(secret_key=decryption_context.secret_key())


# -----------------------
# 메인 복호화 로직
# -----------------------
def main(context_file=CONTEXT_FILE, input_dir=LORA_OUTPUT_DIR, dec_output_dir=DEC_OUTPUT_DIR):
    # 1) 컨텍스트 로드 (비밀키 포함되어 있어야 복호화 가능)
    print("--- 1. Context Loading ---")
    if not os.path.exists(context_file):
        print(f"Error: Context file not found at {context_file}. Please run encryption/setup first.")
        return
        
    context = load_context_from_file(context_file)
    print("TenSEAL Context loaded successfully (assumed to contain Secret Key).")

    # 복호화 결과 저장 디렉토리 생성
    os.makedirs(dec_output_dir, exist_ok=True)
    print(f"Decrypted output directory created at: {dec_output_dir}")

    # 2) 암호화된 LoRA 출력 파일 목록
    print("\n--- 2. Finding Encrypted Outputs ---")
    # 파일 이름을 확인하여 검색 패턴을 수정합니다.
    # 수정 전: "dummy_code_lora_*.ckks"
    # 수정 후: "dummy_code_ckks_enc_*_lora_out_*.ckks"
    ckks_files = sorted(glob.glob(os.path.join(input_dir, "dummy_code_ckks_enc_*_lora_out_*.ckks")))
    
    if len(ckks_files) == 0:
        print(f"No LoRA output files (*.ckks) found in {input_dir} matching pattern 'dummy_code_ckks_enc_*_lora_out_*.ckks'. Please run apply_he_lora_batch.py first.")
        return

    print(f"Found {len(ckks_files)} CKKS files to decrypt.")

    # 3) 파일 순회 및 복호화
    print("\n--- 3. Decrypting and Saving Results ---")
    
    for idx, ckks_path in enumerate(ckks_files):
        filename = os.path.basename(ckks_path)
        print(f"\n[{idx+1}/{len(ckks_files)}] Processing {filename}...")
        
        try:
            # 암호문 로드
            enc_vec = load_ckks_vector_from_file(context, ckks_path)
            
            # 복호화 수행 (비밀키가 있는 context 전달)
            decrypted_data = ckks_decrypt(enc_vec, context)
            
            # 결과 저장 로직: 전체 벡터를 data/dec에 .txt 파일로 저장
            base_name = os.path.splitext(filename)[0]
            out_name = f"{base_name}.txt"
            out_path = os.path.join(dec_output_dir, out_name)

            # NumPy 배열로 변환 후 텍스트 파일로 저장 (전체 벡터)
            # fmt='%.8f'로 소수점 8자리까지 저장
            np.savetxt(out_path, np.array(decrypted_data), fmt='%.8f') 
            
            # 결과 출력 업데이트
            print(f"  > Decryption successful.")
            print(f"  > Saved decrypted vector (length: {len(decrypted_data)}) to {out_path}")
            
        except Exception as e:
            print(f"  > FAILED to decrypt or save {filename}: {e}")
            continue

    print("\n--- Decryption Process Complete ---")


if __name__ == "__main__":
    main()


