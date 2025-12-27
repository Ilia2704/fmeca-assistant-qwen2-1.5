import streamlit as st
from chat_core import load, init_history, generate
import json
from pathlib import Path


CHAT_HISTORY_FILE = Path(".chat_history.json")

def load_chat():
    if CHAT_HISTORY_FILE.exists():
        try:
            return json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def save_chat(ui):
    CHAT_HISTORY_FILE.write_text(
        json.dumps(ui, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

@st.cache_resource
def get_model():
    return load()

def main():
    st.set_page_config(page_title="assistant-qwen2-1.5 chat")
    st.title("assistant-bot chat")

    tokenizer, model = get_model()

    if "history" not in st.session_state:
        st.session_state.history = init_history()
    if "ui" not in st.session_state:
        st.session_state.ui = load_chat()

        for m in st.session_state.ui:
            if m["role"] in ("user", "assistant"):
                st.session_state.history.append(
                    {"role": m["role"], "content": m["content"]}
                )

    with st.sidebar:
        do_sample = st.toggle("Sampling", value=False)
        max_new_tokens = st.slider("max_new_tokens", 16, 512, 128, 16)
        temperature = st.slider("temperature", 0.1, 1.5, 0.7, 0.1)
        top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)
        top_k = st.slider("top_k", 0, 200, 50, 10)

        if st.button("Clear"):
            st.session_state.history = init_history()
            st.session_state.ui = []
            st.rerun()

    for m in st.session_state.ui:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Type a message…")
    if user_text:
        st.session_state.ui.append({"role": "user", "content": user_text})
        save_chat(st.session_state.ui)
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate(
                    tokenizer, model, st.session_state.history, user_text,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                st.markdown(answer)

        st.session_state.ui.append({"role": "assistant", "content": answer})
        save_chat(st.session_state.ui)

if __name__ == "__main__":
    main()
