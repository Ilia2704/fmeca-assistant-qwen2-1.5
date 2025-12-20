from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

print("🔻 Загружаю модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32
)
print("✅ Модель загружена!")

app = FastAPI()

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 150

@app.post("/generate")
def generate_text(q: Query):
    inputs = tokenizer(q.prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=q.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": answer}
