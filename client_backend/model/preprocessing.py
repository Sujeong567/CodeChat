# client_backend/model/preprocessing.py
import torch

from common.config import DEVICE, MAX_INPUT_LENGTH
from .base_llm import BaseLLMLoader

SYSTEM_GUIDE = """You are an all-in-one Python code refactoring bot.
Your goal is to fix any violations of the rules below in the user's code.

[Company Rules]
rule1) 변수명 snake_case
rule2) 함수명 camelCase
rule3) 클래스명 PascalCase
rul34) magic number 상수화
"""

class LLMPreProcessor:
    def __init__(self, llm_loader: BaseLLMLoader):
        self.llm_loader = llm_loader
        self.max_input_length = MAX_INPUT_LENGTH
        self.device = DEVICE

    def get_initial_states(self, user_code: str) -> dict:
        tokenizer = self.llm_loader.tokenizer
        peft_model = self.llm_loader.peft_model

        user_txt = f"다음 코드를 고쳐라.\n\n코드:\n{user_code.rstrip()}"
        full_prompt = (
            "<|system|>\n" + SYSTEM_GUIDE + "<|endoftext|>\n"
            "<|user|>\n" + user_txt + "<|endoftext|>\n"
            "<|assistant|>\n"
        )

        print(f"[PreProcessor] full_prompt 앞부분:\n{full_prompt[:200]}...")

        input_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        ).input_ids.to(self.device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
            )

        lora_xL_input = self.llm_loader.get_lora_xL_input()  # (1, hidden)
        current_llm_hidden_state = outputs.hidden_states[-1][0, -1, :].to(torch.bfloat16)

        return {
            "attention_mask": attention_mask,
            "past_key_values": outputs.past_key_values,
            "current_llm_hidden_state": current_llm_hidden_state,
            "lora_xL_input": lora_xL_input,
            "generated_ids": input_ids.tolist()[0],
        }

    def get_next_token_states(self, next_token_id: int, prev_state: dict) -> dict:
        peft_model = self.llm_loader.peft_model
        next_input = torch.tensor([[next_token_id]], device=self.device)

        new_mask = torch.cat(
            [prev_state["attention_mask"], torch.ones((1, 1), dtype=torch.bool, device=self.device)],
            dim=1,
        )

        with torch.no_grad():
            outputs = peft_model(
                input_ids=next_input,
                attention_mask=new_mask,
                past_key_values=prev_state["past_key_values"],
                output_hidden_states=True,
                use_cache=True,
            )

        lora_xL_input = self.llm_loader.get_lora_xL_input()
        current_llm_hidden_state = outputs.hidden_states[-1][0, -1, :].to(torch.bfloat16)

        prev_state.update(
            {
                "attention_mask": new_mask,
                "past_key_values": outputs.past_key_values,
                "current_llm_hidden_state": current_llm_hidden_state,
                "lora_xL_input": lora_xL_input,
            }
        )
        return prev_state
