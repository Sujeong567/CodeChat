# server/lora/adapter.py
import torch
import json
from pathlib import Path
from safetensors.torch import load_file
import tenseal as ts

from common.config import (
    LORA_WEIGHTS_DIR,
    R_RANK,
    HIDDEN_SIZE,
    FHE_LAYERS,
    FHE_MODULES,
)


def load_lora_weights(lora_path: str = None):
    """
    adapter_config.json + adapter_model.safetensors 로드
    """
    if lora_path is None:
        lora_path = LORA_WEIGHTS_DIR
    p = Path(lora_path)

    config_file = p / "adapter_config.json"
    adapter_file = p / "adapter_model.safetensors"

    with open(config_file, "r") as f:
        config = json.load(f)

    weights = load_file(str(adapter_file))
    return weights, config


def extract_lora_matrices(weights: dict, key_prefix: str):
    """
    key_prefix:
      "base_model.model.model.layers.15.self_attn.q_proj"
    에 대해
      "...lora_A.weight", "...lora_B.weight" 찾아서 반환
    """
    A_key = None
    B_key = None

    for k in weights.keys():
        if key_prefix in k:
            if "lora_A" in k:
                A_key = k
            elif "lora_B" in k:
                B_key = k

    if A_key is None or B_key is None:
        raise ValueError(f"[Adapter] LoRA 행렬 없음: {key_prefix}")

    W_A = weights[A_key].float()  # (r, hidden)
    W_B = weights[B_key].float()  # (hidden, r)
    print(f"[Adapter] {key_prefix}")
    print(f"         W_A: {W_A.shape}, W_B: {W_B.shape}")
    return W_A, W_B


def get_multi_fhe_lora_tensors():
    """
    (layer_idx, module_name) → (W_A_pt, W_B_pt) 의 dict를 반환
    - W_A_pt: (H, r)  plain_tensor
    - W_B_pt: (r, H)  plain_tensor
    """
    weights, _ = load_lora_weights()
    fhe_dict = {}

    for layer_idx in FHE_LAYERS:
        for mod in FHE_MODULES:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
            try:
                W_A, W_B = extract_lora_matrices(weights, prefix)

                # (r, H) -> (H, r), (H, r)^T -> (r, H)
                W_A_pt = ts.plain_tensor(W_A.T.float().tolist())
                W_B_pt = ts.plain_tensor(W_B.T.float().tolist())

                fhe_dict[(layer_idx, mod)] = (W_A_pt, W_B_pt)
                print(f"[Adapter] LoRA FHE tensor ready: (layer={layer_idx}, mod={mod})")

            except Exception as e:
                print(f"[Adapter] FAILED for {prefix}: {e}")
                # zero fallback
                W_Az = torch.zeros(R_RANK, HIDDEN_SIZE).float()
                W_Bz = torch.zeros(HIDDEN_SIZE, R_RANK).float()
                fhe_dict[(layer_idx, mod)] = (
                    ts.plain_tensor(W_Az.T.tolist()),
                    ts.plain_tensor(W_Bz.T.tolist()),
                )
                print(f"[Adapter] ZERO tensor inserted for (layer={layer_idx}, mod={mod})")

    print(f"[Adapter] 총 {len(fhe_dict)}개 (layer, module) LoRA 텐서 준비 완료")
    return fhe_dict
