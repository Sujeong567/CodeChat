import torch
import json
import os
from pathlib import Path
import sys
from adapter import load_lora_adapter, extract_lora_matrices

# load_lora_adapter, extract_lora_matrices 함수는 위에 정의된 원본 코드를 그대로 사용합니다.
# 편의를 위해 여기에 다시 포함하지 않고, 같은 파일 내에 함수가 정의되어 있다고 가정합니다.

# --- 함수 원본 (재확인용) ---
# def load_lora_adapter(lora_path: str = None): ...
# def extract_lora_matrices(weights: dict, layer_name: str): ...
# -----------------------------

TEST_LORA_PATH = Path("./server/lora/lora_weights_checkpoints_final")
TEST_LAYER_NAME = "base_model.model.model.layers.0.self_attn.q_proj" 
# 또는 "layers.0.self_attn.q_proj" (가중치 파일의 실제 키 구조에 따라 다름)

def run_actual_lora_test():
    """실제 경로에서 LoRA 파일을 로드하는 테스트를 실행합니다."""
    print("============================================================")
    print("🚀 실제 LoRA 파일 로딩 테스트 시작")
    print(f"✅ 대상 경로: {TEST_LORA_PATH.resolve()}")
    print("============================================================\n")

    try:
        # 1. load_lora_adapter 함수 테스트
        # 실제 파일이 존재하지 않으면 여기서 FileNotFoundError 발생
        lora_data = load_lora_adapter(lora_path=str(TEST_LORA_PATH))
        
        weights = lora_data['weights']
        rank = lora_data['rank']
        alpha = lora_data['alpha']
        
        print("\n--- 로딩 결과 확인 ---")
        print(f"가져온 LoRA Rank: {rank}")
        print(f"가져온 LoRA Alpha: {alpha}")
        print(f"Weights Key 개수: {len(weights)}")

        # 2. extract_lora_matrices 함수 테스트
        # 가중치 딕셔너리의 실제 키 구조에 맞게 TEST_LAYER_NAME을 조정해야 할 수 있습니다.
        # 예: PEFT 라이브러리는 'base_model.model.layers.0.self_attn.q_proj' 같은 긴 키를 사용합니다.

        # 가장 흔한 LoRA 키 패턴을 사용하여 테스트할 레이어 이름을 찾습니다.
        # 가중치 딕셔너리에 실제로 포함된 키를 기반으로 테스트 레이어 이름을 선택해야 합니다.
        potential_lora_keys = [k for k in weights.keys() if "lora_A" in k]
        
        if not potential_lora_keys:
             print("\n❌ 오류: 가중치 딕셔너리에서 LoRA A/B 행렬 키를 찾을 수 없습니다. (lora_A 키 없음)")
             return
             
        # 첫 번째 LoRA A 키에서 레이어 이름(prefix) 추출
        # 예: 'base_model.model.layers.0.self_attn.q_proj.lora_A.default' -> 'base_model.model.layers.0.self_attn.q_proj'
        first_lora_key = potential_lora_keys[0]
        # lora_A.default 또는 lora_A 부분을 제거
        test_layer_name_actual = first_lora_key.split(".lora_A")[0]
        
        print(f"\n✨ 테스트할 실제 레이어 이름: **{test_layer_name_actual}**")

        lora_A, lora_B = extract_lora_matrices(weights, test_layer_name_actual)
        
        # 추출된 행렬 Shape 확인 (rank 값으로 검증)
        
        print("\n📏 최종 Shape 검증:")
        print(f"  - LoRA A Shape: {lora_A.shape}")
        print(f"  - LoRA B Shape: {lora_B.shape}")
        
        # LoRA A 행렬의 첫 번째 차원과 LoRA B 행렬의 두 번째 차원이 rank와 일치해야 합니다.
        if lora_A.shape[0] == rank and lora_B.shape[1] == rank:
            print("🎉 테스트 성공! LoRA 행렬이 rank와 함께 성공적으로 추출되었습니다.")
        else:
            print("⚠️ 경고: 추출된 행렬의 Shape이 예상(rank)과 다릅니다.")

    except FileNotFoundError as e:
        print("\n============================================================")
        print("🚨 테스트 실패: 파일이 지정된 경로에 없습니다.")
        print(f"    {e}")
        print(f"    경로를 다시 확인해 주세요: {TEST_LORA_PATH.resolve()}")
        print("============================================================")
        sys.exit(1)
    except RuntimeError as e:
        print("\n============================================================")
        print("🚨 테스트 실패: 가중치 파일 로딩 오류")
        print(f"    PyTorch 로딩 오류 발생: {e}")
        print("    **safetensors 파일을 torch.load로 로드할 때 형식이 맞지 않아 발생할 수 있습니다.**")
        print("    adapter_model.safetensors 파일이 실제로 PyTorch .bin 형식으로 저장되어 있는지 확인해 주세요.")
        print("============================================================")
        sys.exit(1)
    except ValueError as e:
        print(f"\n🚨 테스트 실패: 추출 오류 - {e}")
        print("    추출하려는 레이어 이름이 가중치 딕셔너리 키에 포함되어 있는지 확인해 주세요.")
        sys.exit(1)
    except Exception as e:
        print(f"\n🚨 예상치 못한 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_actual_lora_test()