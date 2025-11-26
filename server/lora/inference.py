# server/lora/inference.py
import tenseal as ts

def he_lora_inference(enc_input: ts.CKKSVector, W_A_pt, W_B_pt, ctx: ts.Context) -> bytes:
    """
    enc_input: CKKSVector (길이 = HIDDEN_SIZE)
    W_A_pt: plain_tensor (H, R)
    W_B_pt: plain_tensor (R, H)
    """
    print("[Server] FHE LoRA 연산 시작")

    # enc_input: (H,)
    enc_intermediate = enc_input.matmul(W_A_pt)       # (R)
    enc_logits_lora = enc_intermediate.matmul(W_B_pt) # (H)

    try:
        enc_logits_lora.rescale_()
    except Exception:
        pass

    print("[Server] FHE LoRA 연산 완료")
    return enc_logits_lora.serialize()

def he_lora_inference_multi_qproj(
    enc_input: ts.CKKSVector,
    layer_tensors: dict,   # {layer_idx: (W_A_pt, W_B_pt)}
    ctx: ts.Context
) -> bytes:
    """
    enc_input: CKKSVector (H,)
    layer_tensors: { layer_idx: (W_A_pt, W_B_pt) }
        - 각 W_A_pt: (H, R), W_B_pt: (R, H)
    """
    print("[Server] FHE LoRA multi-layer(q_proj) 연산 시작")

    enc_total = None

    for layer_idx, (W_A_pt, W_B_pt) in layer_tensors.items():
        # (H,) -> (R,) -> (H,)
        enc_intermediate = enc_input.matmul(W_A_pt)
        enc_logits_lora = enc_intermediate.matmul(W_B_pt)

        try:
            enc_logits_lora.rescale_()
        except Exception:
            pass

        if enc_total is None:
            enc_total = enc_logits_lora
        else:
            enc_total = enc_total + enc_logits_lora

    print("[Server] FHE LoRA multi-layer(q_proj) 연산 완료")
    return enc_total.serialize()

