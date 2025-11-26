# server/lora/adapter.py
import torch
import json
from pathlib import Path
from safetensors.torch import load_file
import tenseal as ts

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

N_LAYERS = 32

def get_multi_layer_qproj_tensors(lora_path: str = None):
    """
    0~31 모든 layer의 q_proj에 대해
    (hidden, r), (r, hidden) plain_tensor를 dict로 반환
    return:
        {
          layer_idx: (W_A_pt, W_B_pt),
          ...
        }
    """
    try:
        lora_data = load_lora_adapter(lora_path)
        weights = lora_data["weights"]

        layer_tensors = {}

        for layer_idx in range(N_LAYERS):
            layer_name = (
                f"base_model.model.model.layers.{layer_idx}.self_attn."
                f"{REPRESENTATIVE_LORA_TARGET_MODULE}"  # "q_proj"
            )
            W_A, W_B = extract_lora_matrices(weights, layer_name)

            # (r, hidden) -> (hidden, r)
            W_A_pt = ts.plain_tensor(W_A.T.float().tolist())
            W_B_pt = ts.plain_tensor(W_B.T.float().tolist())

            layer_tensors[layer_idx] = (W_A_pt, W_B_pt)

        print(
            f"[Adapter] 모든 layer q_proj에 대한 TenSEAL PlainTensor 변환 완료. "
            f"총 개수 = {len(layer_tensors)}"
        )
        return layer_tensors

    except Exception as e:
        print(f"[Adapter] 가중치 준비 실패: {e}")
        print("[Adapter] 0 텐서로 대체합니다.")

        layer_tensors = {}
        for layer_idx in range(N_LAYERS):
            W_A = torch.zeros(R_RANK, HIDDEN_SIZE).float()
            W_B = torch.zeros(HIDDEN_SIZE, R_RANK).float()
            W_A_pt = ts.plain_tensor(W_A.T.tolist())
            W_B_pt = ts.plain_tensor(W_B.T.tolist())
            layer_tensors[layer_idx] = (W_A_pt, W_B_pt)

        return layer_tensors