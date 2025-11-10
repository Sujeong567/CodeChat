# codechat/client-backend/model/preprocessing.py
import torch

from .base_llm import BaseLLMLoader
from common.config import DEVICE, MAX_GEN_LENGTH

class LLMPreProcessor:
    """
    Forward pass하여 훅(Hook) 트리거 → xL과 final_hidden_state 추출
    """
    def __init__(self, llm_loader: BaseLLMLoader):
        self.llm_loader = llm_loader
        self.max_input_length = MAX_GEN_LENGTH
        self.device = DEVICE

    def get_initial_states(self, prompt: str) -> dict:
        """초기 프롬프트 처리"""
        tokenizer = self.llm_loader.tokenizer
        peft_model = self.llm_loader.peft_model

        messages = [{"role": "user", "content": prompt}]
        chat_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_ids = tokenizer(
            chat_prompt_text, return_tensors="pt",
            max_length=self.max_input_length, truncation=True
            ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = peft_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True
            )
        
        lora_xL_input = self.llm_loader.get_lora_xL_input() # 훅이 캡처한 xL (Batch, Hidden) 가져옴
        # bfloat16으로 상태 반환 → float16과 메모리 크기 같으나 수치 범위가 float32만큼 넓어 100번의 루프 동안 값이 오염되는 것 방지
        current_llm_hidden_state = outputs.hidden_states[-1][0, -1, :].to(torch.bfloat16)

        return {
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool, device=self.device),
            "past_key_values": outputs.past_key_values, # 이전 토큰 정보 기억
            "current_llm_hidden_state": current_llm_hidden_state,
            "lora_xL_input": lora_xL_input,
            "generated_ids": input_ids.tolist()[0],
        }

    def get_next_token_states(self, next_token_id: int, prev_state: dict) -> dict:
        """다음 토큰 상태 업데이트"""
        peft_model = self.llm_loader.peft_model
        next_input_token = torch.tensor([[next_token_id]], device=self.device)

        new_mask = torch.cat(
            [prev_state["attention_mask"], torch.ones((1, 1), dtype=torch.bool, device=self.device)],
            dim=1
        )

        with torch.no_grad():
            outputs = peft_model(
                input_ids=next_input_token,
                attention_mask=new_mask,
                past_key_values=prev_state["past_key_values"],
                output_hidden_states=True,
                use_cache=True
            )
        
        lora_xL_input = self.llm_loader.get_lora_xL_input() # 새 xL
        current_llm_hidden_state = outputs.hidden_states[-1][0, -1, :].to(torch.bfloat16) # 델타가 반영된 최종 hidden state를 bfloat16으로 반환

        prev_state.update({
            "attention_mask": new_mask,
            "past_key_values": outputs.past_key_values,
            "current_llm_hidden_state": current_llm_hidden_state,
            "lora_xL_input": lora_xL_input
        })
        return prev_state