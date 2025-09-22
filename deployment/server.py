from model.loader import load_model

model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0"
tokenizer, model = load_model(model_name, "cuda")  # 또는 cpu