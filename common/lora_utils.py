#예시 

from transformers import BitsAndBytesConfig

def get_bnb_config():
    """BitsAndBytes 4비트 양자화 설정"""
    from common.config import (
        BNB_COMPUTE_DTYPE,
        BNB_4BIT_QUANT_TYPE,
        BNB_4BIT_USE_DOUBLE_QUANT
    )
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT
    )


def apply_lora_scaling(lora_output, alpha, rank):
    """LoRA 스케일링 적용"""
    scaling = alpha / rank
    return lora_output * scaling


def merge_lora_weights(base_weight, lora_A, lora_B, alpha, rank):
    """LoRA 가중치를 Base 가중치와 병합"""
    scaling = alpha / rank
    lora_delta = scaling * (lora_B @ lora_A)
    return base_weight + lora_delta
