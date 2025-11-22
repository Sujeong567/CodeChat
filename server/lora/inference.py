# server/lora/inference.py
import tenseal as ts
from common.config import HIDDEN_SIZE


def he_lora_inference(
    enc_input: ts.CKKSVector,
    W_A_pt,
    W_B_pt,
    ctx: ts.Context
) -> ts.CKKSVector:
    """
    단일 (W_A, W_B)에 대한 LoRA delta 계산 (encrypted)
    enc_input: CKKSVector (길이 = HIDDEN_SIZE)
    W_A_pt: plain_tensor (H, R)
    W_B_pt: plain_tensor (R, H)
    """
    enc_intermediate = enc_input.matmul(W_A_pt)       # (R)
    enc_delta = enc_intermediate.matmul(W_B_pt)       # (H)

    try:
        enc_delta.rescale_()
    except Exception:
        pass

    return enc_delta


def he_lora_inference_multi(
    enc_input: ts.CKKSVector,
    all_tensors: dict,
    ctx: ts.Context
) -> bytes:
    """
    모든 layer × proj의 LoRA를 적용한 delta를 합산해서 하나의 CKKSVector로 반환 후 serialize.

    all_tensors: dict[(layer_idx, proj_name)] = (W_A_pt, W_B_pt)
    """
    print("[Server] FHE LoRA multi-layer 연산 시작")

    # 0 벡터로 초기화
    enc_acc = ts.ckks_vector(ctx, [0.0] * HIDDEN_SIZE)

    for (layer_idx, proj_name), (W_A_pt, W_B_pt) in all_tensors.items():
        enc_delta = he_lora_inference(enc_input, W_A_pt, W_B_pt, ctx)
        try:
            enc_acc = enc_acc + enc_delta
        except Exception:
            # scale mismatch 등 발생 시, 일단 해당 delta는 스킵
            print(f"[Server] WARN: 델타 합산 실패 (layer={layer_idx}, proj={proj_name}) -> 스킵")
            continue

    try:
        enc_acc.rescale_()
    except Exception:
        pass

    print("[Server] FHE LoRA multi-layer 연산 완료")
    return enc_acc.serialize()
