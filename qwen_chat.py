from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Qwen/Qwen2-1.5B-Instruct"

print("🔻 Загружаю модель (первый запуск может занять время)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("✅ Модель готова!")

while True:
    prompt = input("\n🧑‍💻 Ты: ")
    if prompt.strip().lower() in ["exit", "quit", "stop"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n🤖 Qwen:", answer)
