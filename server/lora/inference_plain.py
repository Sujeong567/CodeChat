# server/lora/inference_plain.py
import torch
from common.config import R_RANK, LORA_ALPHA


def plaintext_lora_inference(
    hidden_state_tensor: torch.Tensor,
    W_A_pt: torch.Tensor,
    W_B_pt: torch.Tensor,
) -> torch.Tensor:
    """
    평문 은닉 상태 벡터에 대해 LoRA 행렬 곱셈을 수행합니다.
    
    Args:
        hidden_state_tensor: 입력 은닉 상태 벡터 h (e.g., [1, H] 또는 [H])
        W_A_pt: LoRA W_A 가중치 텐서 (H, R)
        W_B_pt: LoRA W_B 가중치 텐서 (R, H)
        
    Returns:
        LoRA 델타 텐서 $\Delta h$ (H)
    """
    print("[Server] Plaintext LoRA 연산 시작")

    # 1. 입력 텐서 차원 조정 (batch 차원이 없는 경우 추가)
    if hidden_state_tensor.dim() == 1:
        # [H] -> [1, H]
        h_in = hidden_state_tensor.unsqueeze(0)
    else:
        h_in = hidden_state_tensor

    # 2. h * W_A 계산 (다운 프로젝션)
    # W_A_pt는 (H, R) 형태로 가정하고, 행렬 곱을 위해 전치(Transpose)가 필요합니다.
    # [1, H] @ [H, R] -> [1, R] 
    intermediate_h = torch.matmul(h_in, W_A_pt) 

    # 3. (h * W_A) * W_B 계산 (업 프로젝션)
    # W_B_pt는 (R, H) 형태로 가정합니다.
    # [1, R] @ [R, H] -> [1, H]
    delta_h = torch.matmul(intermediate_h, W_B_pt) 

   
    # 4. LoRA 스케일링 적용 (핵심 수정)
    SCALE_INJECTION = LORA_ALPHA / R_RANK
    delta_h = delta_h * SCALE_INJECTION
    print(f"[Server] LoRA 스케일: {SCALE_INJECTION:.2f} 적용 완료.")

    print("[Server] Plaintext LoRA 연산 완료")
    return delta_h