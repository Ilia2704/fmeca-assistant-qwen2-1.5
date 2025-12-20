from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Qwen/Qwen2-1.5B-Instruct"

print("🔻 Downloading model... (только первый запуск)")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("✅ Model loaded!")

prompt = "Объясни простыми словами, что такое машинное обучение."

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

print("\n🟦 Ответ модели:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
