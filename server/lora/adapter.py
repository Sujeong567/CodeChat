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


def load_lora_adapter(lora_path: str = None):
    """
    LoRA 어댑터 전체를 로드해서 (weights, config) 반환
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


def extract_lora_matrices(weights: dict, layer_name: str):
    """
    특정 layer_name (예: "base_model.model.model.layers.15.self_attn.q_proj")에 대해
    lora_A, lora_B 행렬을 추출하여 (W_A, W_B)로 반환

    - W_A: (r, H)
    - W_B: (H, r)
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
        raise ValueError(f"[Adapter] LoRA 행렬 없음: {layer_name}")

    W_A = weights[lora_A_key].float()
    W_B = weights[lora_B_key].float()

    print(f"[Adapter] Layer: {layer_name}")
    print(f"          W_A: {tuple(W_A.shape)}, W_B: {tuple(W_B.shape)}")

    return W_A, W_B


def get_multi_fhe_lora_tensors():
    """
    (layer, module) → (W_A_pt, W_B_pt)
    형태의 dict를 반환

    - FHE_LAYERS × FHE_MODULES 조합 전체에 대해
      TenSEAL plain_tensor로 변환해서 미리 메모리에 올려둔다.
    """
    weights, _ = load_lora_adapter()
    fhe_dict = {}

    for layer_idx in FHE_LAYERS:
        for mod in FHE_MODULES:
            layer_name = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"

            try:
                W_A, W_B = extract_lora_matrices(weights, layer_name)

                # (r, H) → (H, r)
                W_A_pt = ts.plain_tensor(W_A.T.float().tolist())
                W_B_pt = ts.plain_tensor(W_B.T.float().tolist())

                fhe_dict[(layer_idx, mod)] = (W_A_pt, W_B_pt)
                print(f"[Adapter] Loaded LoRA for ({layer_idx}, {mod})")

            except Exception as e:
                print(f"[Adapter] FAILED loading {layer_name}: {e}")
                print("[Adapter]  → ZERO tensor fallback 사용")

                W_A_zero = torch.zeros(R_RANK, HIDDEN_SIZE).float()
                W_B_zero = torch.zeros(HIDDEN_SIZE, R_RANK).float()

                W_A_pt = ts.plain_tensor(W_A_zero.T.tolist())
                W_B_pt = ts.plain_tensor(W_B_zero.T.tolist())
                fhe_dict[(layer_idx, mod)] = (W_A_pt, W_B_pt)

    print(f"[Adapter] 총 {len(fhe_dict)}개 (layer, module) LoRA 준비 완료")
    return fhe_dict
