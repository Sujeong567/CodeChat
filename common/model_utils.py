# common/model_utils.py
from transformers import BitsAndBytesConfig
from peft import LoraConfig

from common.config import (
    BNB_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    R_RANK,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    LORA_BIAS,
    LORA_TASK_TYPE,
)

def get_bnb_config() -> BitsAndBytesConfig:
    """4bit 양자화 설정"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


def get_lora_config() -> LoraConfig:
    """LoRAConfig 생성"""
    return LoraConfig(
        r=R_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
    )