import os, math
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTConfig
from weighted_trainer import WeightedSFTTrainer

# ===== 경로/설정 =====
# FOLDER = 폴더 경로
MODEL_NAME = "TechxGenus/starcoder2-7b-instruct"
DATA_FILE = os.path.join(FOLDER, "data.jsonl")
OUT_DIR = os.path.join(FOLDER, "qlora_checkpoints")
LORA_OUT_DIR = os.path.join(OUT_DIR, "qlora_only")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LORA_OUT_DIR, exist_ok=True)

# ===== 토크나이저 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # PAD=EOS alias(학습은 attention_mask로 안전)
tokenizer.model_max_length = 2048

# 리뷰 태그(스페셜 토큰) 추가
SPECIAL_TOKENS = {"additional_special_tokens": ["<REVIEW_START>", "<REVIEW_END>"]}
tokenizer.add_special_tokens(SPECIAL_TOKENS)

# ===== 4bit/LoRA 설정 =====
compute_dtype = (
    torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
    else torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

# ===== 베이스 모델 =====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
# 스페셜 토큰 반영
model.resize_token_embeddings(len(tokenizer))

# ===== 데이터셋 빌드 =====
# JSONL 예시 필드:
# { "input": "<원본 코드 문자열>",
#   "output": "<수정된 코드 문자열 (```python ... ``` 포함 or 미포함 상관없음)>",
#   "review": "<REVIEW_START>\n# Review: ...\n...\n<REVIEW_END>" }
#
# 프롬프트 템플릿: (assistant에 코드+리뷰를 붙여서 하나의 답변으로)
SYSTEM_GUIDE = """You are an all-in-one Python code refactoring bot.
Your goal is to fix any violations of the rules below in the user's code.

[Company Rules]
rule1) 변수명 snake_case
rule2) 함수명 camelCase
rule3) 클래스명 PascalCase
rule4) magic number 상수화
"""

def to_chat(example):
    user_txt = f"다음 코드를 고쳐라.\n\n코드:\n{example['input'].rstrip()}"
    # 코드 블록 보장: output에 없으면 감싸기
    out = example["output"]
    if "```" not in out:
        out = f"```python\n{out.rstrip()}\n```"
    # 리뷰 태그는 데이터에 이미 포함되어 있어야 함
    review = example["review"].rstrip()
    assistant_txt = f"{out}\n\n{review}\n<|endoftext|>"

    text = (
        "<|system|>\n" + SYSTEM_GUIDE + "<|endoftext|>\n"
        "<|user|>\n"   + user_txt        + "<|endoftext|>\n"
        "<|assistant|>\n" + assistant_txt + "\n"
    )
    return {"text": text}

raw = load_dataset("json", data_files=DATA_FILE, split="train")
ds  = raw.map(to_chat, remove_columns=raw.column_names)

# ===== Train/Eval split (5%) =====
eval_ratio = 0.05 if len(ds) > 100 else 0.1
split = ds.train_test_split(test_size=eval_ratio, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"Train/Eval sizes: {len(train_ds)} / {len(eval_ds)}")

# ===== 트레이닝 설정 =====
# 리뷰 가중치 3→1 선형감소
def review_weight_fn(step, base=3.0):
    return 1.0 + (base - 1.0) * max(0.0, 1.0 - step / 1322.0)

train_cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=300,
    save_steps=300,
    save_total_limit=5,
    bf16=(compute_dtype == torch.bfloat16),
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="text",
    optim="paged_adamw_8bit",
    adam_beta2=0.95,
    weight_decay=0.1,
    packing=False,
    report_to="none",
)

trainer = WeightedSFTTrainer(
    model=model,
    processing_class=tokenizer,
    peft_config=lora_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=train_cfg,
    review_weight=3.0,
    review_weight_fn=review_weight_fn,
    log_review_weight_steps=300,
)

if __name__ == "__main__":
    train_result = trainer.train()
    print("Final training loss:", train_result.training_loss)

    # 풀 모델(베이스 + 스패셜토큰 포함 토크나이저) 저장
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Full model saved to:", OUT_DIR)

    # LoRA 어댑터만 저장
    trainer.model.save_pretrained(LORA_OUT_DIR)
    print("LoRA adapter saved to:", LORA_OUT_DIR)
    
    if eval_ds is not None:
        eval_results = trainer.evaluate()
        print("Evaluation loss:", eval_results["eval_loss"])
