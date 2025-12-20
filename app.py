import streamlit as st
from chat_core import load, init_history, generate

@st.cache_resource
def get_model():
    return load()

def main():
    st.set_page_config(page_title="lawer-qwen2-1.5 chat")
    st.title("lawer-bot chat")

    tokenizer, model = get_model()

    if "history" not in st.session_state:
        st.session_state.history = init_history()
    if "ui" not in st.session_state:
        st.session_state.ui = []

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

if __name__ == "__main__":
    main()
