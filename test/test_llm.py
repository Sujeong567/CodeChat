# test/test_llm.py
import torch
import sys
import os

# 프로젝트 루트 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from client_backend.model.base_llm import BaseLLMLoader
from client_backend.model.preprocessing import LLMPreProcessor
from client_backend.model.postprocessing import LLMPostProcessor


def disable_all_lora(loader: BaseLLMLoader):
    """모든 LoRA hook + LoRA weight 제거"""
    loader.clear_lora_hooks()
    for name, param in loader.peft_model.named_parameters():
        if "lora_" in name:
            param.data.zero_()
    print("[TEST] LoRA 완전 비활성화 완료.")


def generate_no_hook(prompt, max_new_tokens=100):
    print("\n===== CLIENT LLM ONLY (NO HOOK, NO LORA) =====")

    # 1) 로더
    llm = BaseLLMLoader()
    llm.load_model()

    # 2) LoRA 완전 제거
    disable_all_lora(llm)

    # 3) 프리/포스트 초기화
    pre = LLMPreProcessor(llm)
    post = LLMPostProcessor(llm)

    # ⛔ Hook이 없으므로 get_initial_states()를 그대로 쓰면 안 됨
    #    → 내부에서 get_lora_xL_input() 호출하기 때문
    # 그래서 여기서 직접 forward pass 수행해야 한다.

    tokenizer = llm.tokenizer
    peft_model = llm.peft_model

    # 프롬프트 구성
    user_txt = f"다음 코드를 고쳐라.\n\n코드:\n{prompt.rstrip()}"
    full_prompt = (
        "<|system|>\n" +
        """You are an all-in-one Python code refactoring bot.
Your goal is to fix violations of rules below.""" +
        "<|endoftext|>\n"
        "<|user|>\n" + user_txt + "<|endoftext|>\n"
        "<|assistant|>\n"
    )

    print("[TEST] Prompt prepared.")

    # 모델 입력
    ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(llm.peft_model.device)

    # 첫 forward
    with torch.no_grad():
        out = peft_model(ids, output_hidden_states=True, use_cache=True)

    # 상태 값 수동 구성 (LoRA는 없음)
    hidden = out.hidden_states[-1][0, -1, :].to(torch.bfloat16)
    past = out.past_key_values
    attention_mask = torch.ones_like(ids, dtype=torch.bool)

    generated_ids = ids[0].tolist()

    # ----------------------
    # 토큰 생성 루프
    # ----------------------
    for i in range(max_new_tokens):

        # 1-step next token 생성 (LoRA delta 없음)
        token_id, token_str = post.integrate_lora_delta_and_predict_token(hidden)
        print(f"[{i}] {repr(token_str)}")

        generated_ids.append(token_id)

        if token_id == llm.eos_token_id:
            break

        next_input = torch.tensor([[token_id]], device=llm.peft_model.device)

        # attention mask 확장
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.bool, device=attention_mask.device)],
            dim=1
        )

        with torch.no_grad():
            out = peft_model(
                input_ids=next_input,
                past_key_values=past,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )

        past = out.past_key_values
        hidden = out.hidden_states[-1][0, -1, :].to(torch.bfloat16)

    # 최종 decode
    result = post.decode_final_output(generated_ids)
    print("\n===== FINAL OUTPUT =====")
    print(result)
    print("=======================================")

    return result


if __name__ == "__main__":
    prompt = """class shopping_cart:
    def CALC_TOTAL(self, Price_List):
        Total = 0
        for p in Price_List:
            if p > 10000:
                Total += p * 0.9
            else:
                Total += p
        return Total
    """

    generate_no_hook(prompt, max_new_tokens=100)

"""
결과:
StarCoder2-7B + PreProcessor → PostProcessor → Generation loop 정상 동작
- 모델 로딩 정상
- LayerNorm float32 conversion 정상
- LM head matmul → logits → argmax 정상
- attention_mask / past_key_values 연결 정상
- EOS ID = 0 정상 (GPT 계열 instruct 모델의 EOS는 보통 <|endoftext|> = 0)

>> 현재 LoRA가 적용되지 않았기 때문에 코드 설명 출력
"""