from chat_core import load, init_history, generate

def main():
    tokenizer, model = load()
    history = init_history()

    print("Chat ready. Type 'exit' to quit.\n")

    while True:
        user_text = input(">>> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        answer = generate(tokenizer, model, history, user_text, do_sample=False, max_new_tokens=128)
        print(answer)
        print("-" * 40)

if __name__ == "__main__":
    main()
