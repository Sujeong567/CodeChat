import torch

from common.config import MAX_INPUT_LENGTH, BNB_COMPUTE_DTYPE, DEVICE
from model.base_llm import BaseLLMLoader

class LLMPreprocessor:
    def __init__(self, llm_loader: BaseLLMLoader):
        self.llm_loader = llm_loader
        self.max_input_length = MAX_INPUT_LENGTH
        self.device = DEVICE
        self.dtype = BNB_COMPUTE_DTYPE
    
    def get_initial_states(self, prompt: str) -> dict:
        """
        초기 프롬프트 토큰화 후 첫 forward pass 결과로 LLM 초기 상태 반환
        current_llm_hidden_state ← 암호화
        """
        tokenizer = self.llm_loader.tokenizer
        base_model = self.llm_loader.base_model

        # Instruction-tuned 모델용 채팅 프롬프트 적용
        messages = [{"role": "user", "content": prompt}]
        chat_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_ids = tokenizer(
            chat_prompt_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).input_ids.to(self.device)

        # 첫 번째 포워드 패스 수행
        # outputs.hidden_states[-1]은 마지막 레이어의 hidden state (LM Head 입력)
        # use_cache=True를 통해 past_key_values 생성하여 다음 토큰 예측에 사용
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                output_hidden_states=True,  # Hidden states 추출 활성화
                use_cache=True              # past_key_values 사용 활성화
            )
        
        # current_llm_hidden_states ← FHE 연산을 위해 서버로 전송될 Hidden State
        current_llm_hidden_state = outputs.hidden_states[-1][0, -1, :].to(self.dtype)

        # LLM의 현재 상태 초기화 및 반환
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool, device=self.device),
            "past_key_values": outputs.past_key_values,
            "current_llm_hidden_state": current_llm_hidden_state,
            "generated_ids": input_ids.tolist()[0],  # 생성된 토큰 ID 리스트 (초기 프롬프트 포함)
            "prompt_text": chat_prompt_text
        }
    
    def get_next_token_states(self, next_token_id: int, prev_state: dict) -> dict:
        """
        이전 LLM 상태와 새로 예측된 다음 토큰 사용 → LLM 상태 업데이트 및 current_llm_hidden_state 반환
        """
        base_model = self.llm_loader.base_model
        # 새로 예측된 토큰 하나만 입력으로 용
        next_input_token = torch.tensor([[next_token_id]], device=self.device)

        # attention_mask 업데이트: 새로운 토큰 추가
        new_mask = torch.cat(
            [prev_state["attention_mask"], torch.ones((1, 1), dtype=torch.bool, device=self.device)],
            dim=1
        )

        # 다음 토큰에 대한 포워드 패스 수행
        with torch.no_grad():
            outputs = base_model(
                input_ids=next_input_token,
                attention_mask=new_mask,
                past_key_values=prev_state["past_key_values"],
                output_hidden_states=True,
                use_cache=True
            )

        # 업데이트된 상태 반환
        prev_state.update({
            "input_ids": next_input_token,
            "attention_mask": new_mask,
            "past_key_values": outputs.past_key_values,
            "current_llm_hidden_state": outputs.hidden_states[-1][0, -1, :].to(self.dtype)
        })
        prev_state["generated_ids"].append(next_token_id)
        return prev_state