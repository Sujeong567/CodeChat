# server/lora/inference.py
import tenseal as ts

def he_lora_inference(enc_input: ts.CKKSVector, W_A_pt, W_B_pt, ctx: ts.Context) -> bytes:
    """
    enc_input: CKKSVector (H,)
    W_A_pt: plain_tensor (H, r)
    W_B_pt: plain_tensor (r, H)
    """
    print("[Server] FHE LoRA 연산 시작")

    enc_intermediate = enc_input.matmul(W_A_pt)       # (r,)
    enc_logits_lora = enc_intermediate.matmul(W_B_pt) # (H,)

    try:
        enc_logits_lora.rescale_()
    except Exception:
        pass

    print("[Server] FHE LoRA 연산 완료")
    return enc_logits_lora.serialize()
