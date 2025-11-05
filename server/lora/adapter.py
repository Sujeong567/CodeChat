"""
가중치 파일 로드 
LoRA A, B 행렬을 메모리에 올림
설정 정보(rank, alpha 등 파싱)
"""

"""
LoRA 어댑터 로딩
- adapter_model.bin, adapter_config.json 읽기
- LoRA A, B 행렬 메모리에 로드
"""

import torch
import json
import os
from pathlib import Path


def load_lora_adapter(lora_path: str = None):
    """
    데이터팀이 학습한 LoRA 가중치 로딩
    
    Args:
        lora_path: LoRA 폴더 경로
    
    Returns:
        {
            'weights': LoRA 가중치 딕셔너리,
            'config': 설정 정보,
            'rank': LoRA rank,
            'alpha': LoRA alpha
        }
    """
    
    if lora_path is None:
        lora_path = "./models/lora_weights/checkpoint-final"
    
    lora_path = Path(lora_path)
    
    print(f"\n{'='*60}")
    print(f"📂 LoRA 경로: {lora_path}")
    print(f"{'='*60}\n")
    
    # 파일 존재 확인
    adapter_file = lora_path / "adapter_model.bin"
    config_file = lora_path / "adapter_config.json"
    
    if not adapter_file.exists():
        raise FileNotFoundError(f"adapter_model.bin 없음: {adapter_file}")
    
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json 없음: {config_file}")
    
    print("✅ 파일 발견!")
    print(f"   - adapter_model.bin: {adapter_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   - adapter_config.json: {config_file.stat().st_size / 1024:.2f} KB\n")
    
    # config 로드
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    rank = config.get("r", 4)
    alpha = config.get("lora_alpha", 32)
    
    print(f"   LoRA Rank: {rank}")
    print(f"   LoRA Alpha: {alpha}\n")
    
    # 가중치 로드
    print("📦 LoRA 가중치 로딩 중...")
    weights = torch.load(adapter_file, map_location="cpu")
    
    num_params = sum(p.numel() for p in weights.values())
    print(f"   전체 파라미터: {num_params:,}\n")
    
    print(f"{'='*60}")
    print("✅ LoRA 어댑터 로딩 완료!")
    print(f"{'='*60}\n")

        # 5. 가중치 분석
    print("📊 LoRA 가중치 분석:")
    
    num_params = sum(p.numel() for p in weights.values())
    print(f"   전체 파라미터: {num_params:,}")
    
    # A, B 행렬 개수 세기
    lora_A_count = sum(1 for k in weights.keys() if "lora_A" in k)
    lora_B_count = sum(1 for k in weights.keys() if "lora_B" in k)
    print(f"   LoRA A 행렬 개수: {lora_A_count}")
    print(f"   LoRA B 행렬 개수: {lora_B_count}")
    
    # 샘플 확인 (첫 번째 레이어)
    first_key = list(weights.keys())[0]
    print(f"\n   샘플 키: {first_key}")
    print(f"   샘플 Shape: {weights[first_key].shape}")
    
    print(f"\n{'='*60}")
    print("✅ LoRA 어댑터 로딩 완료!")
    print(f"{'='*60}\n")
    
    return {
        'weights': weights,
        'config': config,
        'rank': rank,
        'alpha': alpha
    }


def extract_lora_matrices(weights: dict, layer_name: str):
    """
    특정 레이어의 LoRA A, B 행렬 추출
    
    Args:
        weights: load_lora_adapter()에서 받은 가중치
        layer_name: 레이어 이름 (예: "layers.0.self_attn.q_proj")
    
    Returns:
        (lora_A, lora_B) 튜플
    """
    # 키 패턴 찾기
    lora_A_key = None
    lora_B_key = None
    
    for key in weights.keys():
        if layer_name in key:
            if "lora_A" in key:
                lora_A_key = key
            elif "lora_B" in key:
                lora_B_key = key
    
    if lora_A_key is None or lora_B_key is None:
        raise ValueError(f"Layer {layer_name}의 LoRA 행렬을 찾을 수 없습니다")
    
    lora_A = weights[lora_A_key]
    lora_B = weights[lora_B_key]
    
    print(f"📐 {layer_name}")
    print(f"   LoRA A shape: {lora_A.shape}")
    print(f"   LoRA B shape: {lora_B.shape}")
    
    return lora_A, lora_B
