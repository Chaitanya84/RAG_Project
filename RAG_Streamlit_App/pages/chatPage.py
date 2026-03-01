import sys
import streamlit as st

# Ensure we can import AnswerAll from the app root
sys.path.append("/home/prichai/AI_ML/RAG_Project/RAG_Streamlit_App")

from AnswerAll import (
    load_rag_resources,
    answer_with_rag_cached,
)

st.set_page_config(page_title="RAG Chat - Mahabharata", layout="centered")

# Require login
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

st.title("Mahabharata RAG Chat")
st.write(f"Logged in as: {st.session_state.user_email}")

@st.cache_resource
def get_rag_resources():
    return load_rag_resources()

# Dummy LLM resources – kept only for signature compatibility
@st.cache_resource
def get_llm_resources():
    return None, None

embedding_model, pages_and_chunks, embeddings = get_rag_resources()
tokenizer, llm_model = get_llm_resources()  # both None, but required by signature

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# Input area
user_input = st.text_input("Ask a question about the Mahabharata:", "")

if st.button("Send") and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            answer = answer_with_rag_cached(
                user_query=question,
                embedding_model=embedding_model,
                pages_and_chunks=pages_and_chunks,
                embeddings=embeddings,
                tokenizer=tokenizer,
                llm_model=llm_model,
            )
        except Exception as e:
            answer = f"Error while generating answer: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

if st.button("Clear chat"):
    st.session_state.chat_history = []
    st.rerun()