import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Answer briefly and clearly.\n"
    "ALWAYS output the final answer wrapped in curly braces.\n"
    "Example: {hello}\n"
    "Do not output anything outside the braces."
)

def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model

def init_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def wrap_braces(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "{}"
    if not (text.startswith("{") and text.endswith("}")):
        text = "{" + text.strip("{} \n\t") + "}"
    return text

def generate(tokenizer, model, history, user_text: str, *,
             max_new_tokens: int = 128,
             do_sample: bool = False,
             temperature: float = 0.7,
             top_p: float = 0.9,
             top_k: int = 50) -> str:

    history.append({"role": "user", "content": user_text})

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p, top_k=top_k))

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    gen_ids = output[0][inputs["input_ids"].shape[-1]:]
    answer = "" if gen_ids.numel() == 0 else tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    answer = wrap_braces(answer)

    history.append({"role": "assistant", "content": answer})
    return answer
