# server/lora/adapter.py
import torch
import json
from pathlib import Path
from safetensors.torch import load_file
import tenseal as ts

from common.config import (
    LORA_WEIGHTS_DIR, R_RANK, HIDDEN_SIZE,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


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


def _infer_num_layers_from_weights(weights: dict) -> int:
    """
    base_model.model.model.layers.{L}.self_attn.q_proj.lora_A.weight
    이런 키를 이용해서 L의 최대값을 찾아 레이어 수를 추론
    """
    max_idx = -1
    for key in weights.keys():
        if ".self_attn.q_proj.lora_A.weight" in key:
            # 'base_model.model.model.layers.10.self_attn.q_proj.lora_A.weight'
            try:
                after_layers = key.split("layers.")[1]
                layer_idx_str = after_layers.split(".")[0]
                layer_idx = int(layer_idx_str)
                max_idx = max(max_idx, layer_idx)
            except Exception:
                continue

    if max_idx < 0:
        raise ValueError("self_attn.q_proj.lora_A.weight 키를 찾지 못해 레이어 수를 추론할 수 없습니다.")

    num_layers = max_idx + 1
    print(f"[Adapter] LoRA 레이어 수 추론: num_layers = {num_layers}")
    return num_layers


def extract_lora_matrices(weights: dict, layer_name: str):
    """
    layer_name 예:
      'base_model.model.model.layers.0.self_attn.q_proj'
    에 대해
      *.lora_A.weight, *.lora_B.weight
    를 찾아서 (r, hidden), (hidden, r) shape로 반환
    """
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

    W_A = weights[lora_A_key].float()   # 보통 (r, hidden)
    W_B = weights[lora_B_key].float()   # 보통 (hidden, r)

    print(f"[Adapter] 레이어: {layer_name}")
    print(f"         W_A: {W_A.shape}, W_B: {W_B.shape}")

    return W_A, W_B


def get_all_fhe_lora_tensors(lora_path: str = None):
    """
    모든 layer(0..N-1) × 모든 proj(q,k,v,o)에 대해
    TenSEAL plain_tensor로 변환한 후 dict로 반환.

    return:
      {
        (layer_idx, "q_proj"): (W_A_pt, W_B_pt),
        (layer_idx, "k_proj"): (W_A_pt, W_B_pt),
        ...
      }
    """
    try:
        lora_data = load_lora_adapter(lora_path)
        weights = lora_data["weights"]

        num_layers = _infer_num_layers_from_weights(weights)

        all_tensors = {}

        for layer_idx in range(num_layers):
            for proj in TARGET_MODULES:
                layer_prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}"
                W_A, W_B = extract_lora_matrices(weights, layer_prefix)

                # (r, hidden) → transpose → (hidden, r)
                W_A_pt = ts.plain_tensor(W_A.T.float().tolist())
                W_B_pt = ts.plain_tensor(W_B.T.float().tolist())

                all_tensors[(layer_idx, proj)] = (W_A_pt, W_B_pt)

        print(f"[Adapter] 모든 layer × proj에 대한 TenSEAL PlainTensor 변환 완료. "
              f"총 개수 = {len(all_tensors)}")
        return all_tensors

    except Exception as e:
        print(f"[Adapter] 가중치 준비 실패: {e}")
        print("[Adapter] 0 텐서로 전체를 대체합니다.")

        # fallback: 모든 layer/proj에 대해 zero tensor 사용
        # (레이어 수 추론 실패 시, 1 layer만 zero로 사용)
        try:
            num_layers = _infer_num_layers_from_weights(weights)
        except Exception:
            num_layers = 1

        all_tensors = {}
        for layer_idx in range(num_layers):
            for proj in TARGET_MODULES:
                W_A = torch.zeros(R_RANK, HIDDEN_SIZE).float()
                W_B = torch.zeros(HIDDEN_SIZE, R_RANK).float()
                W_A_pt = ts.plain_tensor(W_A.T.tolist())
                W_B_pt = ts.plain_tensor(W_B.T.tolist())
                all_tensors[(layer_idx, proj)] = (W_A_pt, W_B_pt)

        return all_tensors
