# server/lora/adapter.py
import torch
import json
from pathlib import Path
from safetensors.torch import load_file


from common.config import (
    LORA_WEIGHTS_DIR, R_RANK, HIDDEN_SIZE,
    TARGET_LAYER_INDEX, REPRESENTATIVE_LORA_TARGET_MODULE,
)

def load_lora_adapter(lora_path: str = None):
    if lora_path is None:
        lora_path = LORA_WEIGHTS_DIR
    lora_path = Path(lora_path)

    config_file = lora_path / "adapter_config.json"
    adapter_file = lora_path / "adapter_model.safetensors"

    with open(config_file, "r") as f:
        config = json.load(f)

    weights = load_file(str(adapter_file))
    rank = config.get("r", R_RANK)
    alpha = config.get("lora_alpha", 32)

    return {
        "weights": weights,
        "config": config,
        "rank": rank,
        "alpha": alpha,
    }

def extract_lora_matrices(weights: dict, layer_name: str):
    lora_A_key = None
    lora_B_key = None

    for key in weights.keys():
        if layer_name in key:
            if "lora_A" in key:
                lora_A_key = key
            elif "lora_B" in key:
                lora_B_key = key

    if lora_A_key is None or lora_B_key is None:
        raise ValueError(f"Layer '{layer_name}'의 LoRA 행렬을 찾을 수 없습니다.")

    # (r, hidden), (hidden, r)
    W_A = weights[lora_A_key].float()
    W_B = weights[lora_B_key].float()

    print(f"[Adapter] 레이어: {layer_name}")
    print(f"         W_A: {W_A.shape}, W_B: {W_B.shape}")

    return W_A, W_B


def get_plaintext_lora_tensors(lora_path: str = None):
    """
    q_proj 하나에 대해서만 PyTorch 텐서 (r, hidden), (hidden, r)를 반환
    (주의: 추론 함수에 맞춰 전치(T) 여부는 필요에 따라 조정)
    """
    try:
        lora_data = load_lora_adapter(lora_path)
        weights = lora_data["weights"]

        # ex) model.layers.0.self_attn.q_proj
        layer_name = f"base_model.model.model.layers.{TARGET_LAYER_INDEX}.self_attn.{REPRESENTATIVE_LORA_TARGET_MODULE}"
        
        # W_A: (r, hidden), W_B: (hidden, r)
        W_A, W_B = extract_lora_matrices(weights, layer_name)

        # ⚠️ 동형암호 코드 제거
        # HE 버전에서는 W_A.T와 W_B.T를 plain_tensor로 변환하여 반환했지만,
        # PyTorch 기반의 plaintext_lora_inference 함수가 W_A(r, H), W_B(H, R) 그대로를
        # 입력으로 받는다고 가정하고 원본 텐서를 그대로 반환합니다.
        # (혹은 추론 함수에 맞춰 전치된 형태를 반환할 수도 있습니다.)
        
        # 이전 코드: W_A: (r, hidden) -> W_A.T: (hidden, r)
        #             W_B: (hidden, r) -> W_B.T: (r, hidden)
        
        # ✅ 평문 추론 함수(plaintext_lora_inference)가 W_A(H, R), W_B(R, H) 형태를 사용한다면:
        # (W_A: (r, H) -> W_A_pt: (H, r)), (W_B: (H, r) -> W_B_pt: (r, H))
        W_A_pt = W_A.T.float()
        W_B_pt = W_B.T.float()
        
        # 💡 참고: W_A와 W_B의 차원은 프로젝트 설정 및 추론 코드에 따라 다를 수 있습니다.
        # 이전 HE 코드의 전치 로직을 따라 그대로 전치하여 반환합니다.
        
        print("[Adapter] q_proj에 대한 PyTorch 텐서 준비 완료.")
        # W_A_pt: (hidden, r), W_B_pt: (r, hidden)
        return W_A_pt, W_B_pt

    except Exception as e:
        print(f"[Adapter] 가중치 준비 실패: {e}")
        print("[Adapter] 0 텐서로 대체합니다.")

        # PyTorch 텐서로 대체 (TenSEAL 관련 코드 제거)
        W_A_pt = torch.zeros(HIDDEN_SIZE, R_RANK).float()
        W_B_pt = torch.zeros(R_RANK, HIDDEN_SIZE).float()
        
        return W_A_pt, W_B_pt