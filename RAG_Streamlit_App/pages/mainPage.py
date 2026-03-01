import os
import sys

import streamlit as st

# Make sure we can import embedandchunk
sys.path.append("/home/prichai/AI_ML/RAG_Project/RAG_Streamlit_App")

from embedandchunk import process_pdf_for_rag

st.set_page_config(page_title="RAG Streamlit App - Main", layout="centered")

# Require login
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

# Init state flags
if "last_uploaded_path" not in st.session_state:
    st.session_state.last_uploaded_path = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

st.title("Upload a PDF File")
st.write(f"Logged in as: {st.session_state.user_email}")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Handle new upload + processing in one run (no st.rerun)
if uploaded_file is not None and not st.session_state.processing_done:
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"File uploaded successfully and saved as: {file_path}")

    st.session_state.last_uploaded_path = file_path

    # If this is Mahabharata.pdf, map to the known filenames
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if base_name.lower() == "mahabharata":
        pages_csv = os.path.join(data_folder, "Mahabharata_pages.csv")
        chunks_csv = os.path.join(data_folder, "Mahabharata_chunks.csv")
        embeddings_pt = os.path.join(data_folder, "MahaBharata_embeddings.pt")
    else:
        # For other PDFs, use per-file names
        pages_csv = os.path.join(data_folder, f"{base_name}_pages.csv")
        chunks_csv = os.path.join(data_folder, f"{base_name}_chunks.csv")
        embeddings_pt = os.path.join(data_folder, f"{base_name}_embeddings.pt")

    with st.spinner("Processing PDF for RAG (this may take a while)..."):
        process_pdf_for_rag(
            pdf_path=file_path,
            pages_csv_path=pages_csv,
            chunks_csv_path=chunks_csv,
            embeddings_tensor_path=embeddings_pt,
        )

    st.session_state.processing_done = True

    st.success("PDF processed successfully for RAG!")
    st.info(f"Pages CSV: {pages_csv}")
    st.info(f"Chunks CSV: {chunks_csv}")
    st.info(f"Embeddings tensor: {embeddings_pt}")

# Show status if already processed in this session
elif st.session_state.processing_done and st.session_state.last_uploaded_path:
    st.info(f"Last processed file: {st.session_state.last_uploaded_path}")

# Logout button – only changes state, does NOT trigger processing
if st.button("Logout"):
    st.session_state.user_logged_in = False
    st.session_state.user_email = None
    st.session_state.last_uploaded_path = None
    st.session_state.processing_done = False
    st.switch_page("app.py")