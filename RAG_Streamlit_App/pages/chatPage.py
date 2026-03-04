import os
import sys

# Portable path: resolve relative to this file's location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from AnswerAll import (
    load_rag_resources,
    answer_with_rag_cached,
)
from config import CHUNKS_CSV, EMBEDDINGS_PATH, logger

# NOTE: st.set_page_config() is only called in the main app.py entry point.

# Require login
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

st.title("📚 RAG Chat")
st.write(f"Logged in as: {st.session_state.user_email}")

# ---------------------------------------------------------------------------
# Determine which chunks / embeddings to load.
# If the user processed a PDF directory on mainPage, use those paths;
# otherwise fall back to the default data.
# ---------------------------------------------------------------------------
chunks_csv = st.session_state.get("active_chunks_csv") or CHUNKS_CSV
embeddings_path = st.session_state.get("active_embeddings_path") or EMBEDDINGS_PATH

if not os.path.exists(chunks_csv) or not os.path.exists(embeddings_path):
    st.warning(
        "No processed PDF data found. Please process a PDF directory on the **MainPage** first."
    )
    st.stop()

# Show which knowledge base is active
processed_dir = st.session_state.get("last_processed_dir")
processed_files = st.session_state.get("processed_files", [])
if processed_dir:
    with st.sidebar:
        st.markdown("### 📂 Active Knowledge Base")
        st.markdown(f"**Directory:** `{processed_dir}`")
        if processed_files:
            st.markdown(f"**PDFs loaded:** {len(processed_files)}")
            with st.expander("Show files"):
                for pf in processed_files:
                    st.markdown(f"- `{pf}`")


@st.cache_resource
def get_rag_resources(_chunks_csv: str, _embeddings_path: str):
    """Load RAG resources; cached so they are only read once per unique path pair."""
    return load_rag_resources(chunks_csv=_chunks_csv, embeddings_path=_embeddings_path)


pages_and_chunks, embeddings = get_rag_resources(chunks_csv, embeddings_path)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages using native chat bubbles
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Native chat input (pinned to the bottom of the page)
user_input = st.chat_input("Ask a question about the document…")

if user_input:
    question = user_input.strip()
    if not question:
        st.stop()

    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = answer_with_rag_cached(
                    user_query=question,
                    pages_and_chunks=pages_and_chunks,
                    embeddings=embeddings,
                )
            except Exception as e:
                logger.error("RAG answer failed: %s", e)
                answer = f"⚠️ Error while generating answer: {e}"
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Sidebar actions
with st.sidebar:
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()