import torch
from common.config import DEVICE, MAX_INPUT_LENGTH
from .base_llm import BaseLLMLoader

SYSTEM_GUIDE = """You are an all-in-one Python code refactoring bot.
Your goal is to fix any violations of the rules below in the user's code.

[Company Rules]
rule1) 변수명 snake_case
rule2) 함수명 camelCase
rule3) 클래스명 PascalCase
rule34) magic number 상수화
"""


class LLMPreProcessor:
    def __init__(self, llm_loader: BaseLLMLoader):
        self.llm_loader = llm_loader
        self.max_input_length = MAX_INPUT_LENGTH

    def get_initial_states(self, user_code: str) -> dict:
        tokenizer = self.llm_loader.tokenizer
        model = self.llm_loader.peft_model

        user_txt = f"다음 코드를 고쳐라.\n\n코드:\n{user_code.rstrip()}"
        full_prompt = (
            "<|system|>\n"
            + SYSTEM_GUIDE
            + "<|endoftext|>\n"
            "<|user|>\n"
            + user_txt
            + "<|endoftext|>\n"
            "<|assistant|>\n"
        )

        input_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).input_ids.to(DEVICE)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
            )

        # same-token hook에서 x_L가 queue에 저장됨
        xl_dict = self.llm_loader.get_xl_batch()

        hidden = outputs.hidden_states[-1][0, -1, :].to(torch.bfloat16)

        return {
            "past_key_values": outputs.past_key_values,
            "current_hidden": hidden,
            "attention_mask": attention_mask,
            "xl_dict": xl_dict,
            "generated_ids": input_ids.tolist()[0],
        }

    def get_next_token_states(self, next_id: int, prev_state: dict) -> dict:
        model = self.llm_loader.peft_model

        next_input = torch.tensor([[next_id]], device=DEVICE)
        new_mask = torch.cat(
            [prev_state["attention_mask"], torch.ones((1, 1), dtype=torch.bool, device=DEVICE)],
            dim=1,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=next_input,
                attention_mask=new_mask,
                past_key_values=prev_state["past_key_values"],
                use_cache=True,
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1][0, -1, :].to(torch.bfloat16)

        prev_state["past_key_values"] = outputs.past_key_values
        prev_state["attention_mask"] = new_mask
        prev_state["current_hidden"] = hidden
        prev_state["xl_dict"] = self.llm_loader.get_xl_batch()

        return prev_state
