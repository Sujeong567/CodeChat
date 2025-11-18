# test/test_prompt.py
import requests

def call_backend(prompt):
    url = "http://127.0.0.1:5000/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": 300
    }
    resp = requests.post(url, json=payload)
    print("\n=== Response ===")
    print(resp.json()["generated_text"])
    print("================\n")

if __name__ == "__main__":
    prompt = input("프롬프트 입력: ")
    call_backend(prompt)
