# server/lora/inference.py
import tenseal as ts
import torch

def he_lora_inference(enc_input: ts.CKKSVector, W_A_pt, W_B_pt, ctx: ts.Context) -> bytes:
    """
    enc_input: CKKSVector       (encrypted vector of size HIDDEN_SIZE)
    W_A_pt: plain tensor shape  (HIDDEN_SIZE, R)
    W_B_pt: plain tensor shape  (R, HIDDEN_SIZE)
    """

    print("[Server] FHE LoRA 연산 시작")

    # ----------------------------------------------------------------------
    # (1) 기존 방식 그대로: 암호문 상태로 LoRA 계산
    # ----------------------------------------------------------------------
    enc_intermediate = enc_input.matmul(W_A_pt)             # (R)
    enc_logits_lora = enc_intermediate.matmul(W_B_pt)       # (H)

    try:
        enc_logits_lora.rescale_()
    except Exception:
        pass

    print("[Server] FHE LoRA 연산 완료")

    # ----------------------------------------------------------------------
    # (2) 평문 delta 계산 → 서버 로깅 (디버깅에 필수)
    # ----------------------------------------------------------------------
    try:
        # enc_input decrypt() → CKKSVector 이면 decrypt() 가능
        plain_input = torch.tensor(enc_input.decrypt(), dtype=torch.float32)

        W_A = torch.tensor(W_A_pt, dtype=torch.float32)
        W_B = torch.tensor(W_B_pt, dtype=torch.float32)

        # A*x
        Ax = plain_input @ W_A          # shape (R,)

        # delta = (A*x) @ B
        delta = Ax @ W_B                # shape (H,)

        # ---- delta 검사 로그 ----
        print("[Server] ---- Plain delta debug ----")
        print("  delta.shape:", tuple(delta.shape))
        print("  delta.min:", float(delta.min()))
        print("  delta.max:", float(delta.max()))
        print("  delta.mean:", float(delta.mean()))
        print("  delta.std:", float(delta.std()))
        print("  delta.norm:", float(delta.norm()))
        print("  top-5 abs(delta):", torch.topk(delta.abs(), 5).values.tolist())
        print("[Server] --------------------------------")

    except Exception as e:
        print("[WARN] 평문 delta 계산 실패:", e)

    # ----------------------------------------------------------------------
    # (3) 암호문 delta만 클라이언트로 반환 (기능 유지)
    # ----------------------------------------------------------------------
    return enc_logits_lora.serialize()
