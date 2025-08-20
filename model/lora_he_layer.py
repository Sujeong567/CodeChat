#LoRA 동형암호 레이어(연산/클래스)

from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch

class LoRAHELayer: 
    pass

model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.to("cpu")  # GPU 없으면 "cpu" 사용

def load_model (model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input = input.to(device)

from generate_dummy_encrypted import load_data

texts, labels = load_data()

dataset = MyDataset(texts, labels, tokenizer, max_length=512)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

