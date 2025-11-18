# server/lora/inference.py
import tenseal as ts

def he_lora_inference(enc_input: ts.CKKSVector, lora_tensors: dict, ctx: ts.Context) -> dict:
    """
    enc_input: CKKSVector (길이 = HIDDEN_SIZE)
    lora_tensors: {
        "q_proj": (W_A_pt, W_B_pt),
        "k_proj": (W_A_pt, W_B_pt),
        "v_proj": (W_A_pt, W_B_pt),
        "o_proj": (W_A_pt, W_B_pt)
    }

    return:
    {
        "q_proj": serialized_bytes,
        "k_proj": serialized_bytes,
        ...
    }
    """
    print("[Server] FHE LoRA 연산 시작 (multi-proj)")
    results = {}

    for name, (W_A_pt, W_B_pt) in lora_tensors.items():
        # enc_input: (H)
        enc_intermediate = enc_input.matmul(W_A_pt)   # (r)
        enc_delta = enc_intermediate.matmul(W_B_pt)   # (H)

        try:
            enc_delta.rescale_()
        except Exception:
            pass

        results[name] = enc_delta.serialize()

    print("[Server] FHE LoRA 연산 완료 (multi-proj)")
    return results
