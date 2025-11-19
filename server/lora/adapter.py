# server/lora/adapter.py
import torch
import json
from pathlib import Path
from safetensors.torch import load_file
import tenseal as ts

from common.config import (
    LORA_WEIGHTS_DIR, R_RANK, HIDDEN_SIZE,
    TARGET_LAYER_INDEX
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
    return weights, config


def get_layer_prefix(layer_idx: int):
    # 실제 key prefix
    # base_model.model.model.layers.{L}.self_attn.
    return f"base_model.model.model.layers.{layer_idx}.self_attn."


def extract_lora_for_proj(weights, layer_idx, proj_name):
    """
    proj_name: q_proj / k_proj / v_proj / o_proj
    """

    prefix = get_layer_prefix(layer_idx)
    key_A = f"{prefix}{proj_name}.lora_A.weight"
    key_B = f"{prefix}{proj_name}.lora_B.weight"

    if key_A not in weights or key_B not in weights:
        raise KeyError(f"Missing LoRA keys: {key_A} or {key_B}")

    W_A = weights[key_A].float()    # (r, hidden)
    W_B = weights[key_B].float()    # (hidden, r)

    print(f"[Adapter] Loaded {proj_name}: W_A={W_A.shape}, W_B={W_B.shape}")
    return W_A, W_B


def load_all_lora_tensors(lora_path: str = None):
    weights, config = load_lora_adapter(lora_path)

    layer_idx = TARGET_LAYER_INDEX
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

    proj_dict = {}

    for proj in proj_names:
        W_A, W_B = extract_lora_for_proj(weights, layer_idx, proj)

        # transpose: (r, hidden) → (hidden, r)
        W_A_pt = ts.plain_tensor(W_A.T.tolist())
        W_B_pt = ts.plain_tensor(W_B.T.tolist())

        proj_dict[proj] = (W_A_pt, W_B_pt)

    print("[Adapter] 모든 proj에 대한 TenSEAL PlainTensor 변환 완료.")
    return proj_dict
