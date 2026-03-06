import os
import sys
import glob
import tempfile
import shutil

# Portable path: resolve relative to this file's location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from embedandchunk import process_pdf_directory_for_rag
from config import DATA_DIR, PDF_DIR, logger

# NOTE: st.set_page_config() is only called in the main app.py entry point.

# Require login
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

# Init state flags
if "last_processed_dir" not in st.session_state:
    st.session_state.last_processed_dir = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "active_chunks_csv" not in st.session_state:
    st.session_state.active_chunks_csv = None
if "active_embeddings_path" not in st.session_state:
    st.session_state.active_embeddings_path = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

st.title("📂 Upload PDF Files")
st.write(f"Logged in as: {st.session_state.user_email}")

st.markdown(
    "Use the **file uploader** below to select one or more PDF files from your computer. "
    "All uploaded PDFs will be combined into a **single knowledge base** for RAG."
)

# ---------------------------------------------------------------------------
# File uploader UI
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to upload and process.",
)

if uploaded_files:
    st.success(f"📄 **{len(uploaded_files)}** PDF file(s) selected:")
    for uf in uploaded_files:
        st.markdown(f"- `{uf.name}` ({uf.size / 1024:.1f} KB)")
else:
    st.info("👆 Click **Browse files** or drag and drop PDF files above to get started.")

# ---------------------------------------------------------
# Handle processing
# ---------------------------------------------------------
force_reprocess = st.checkbox("🔄 Force re-process (delete existing outputs)", value=False)

if st.button("🚀 Process All PDFs", disabled=(not uploaded_files)):
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
        st.stop()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Save uploaded files to a temporary directory for processing
    upload_dir = os.path.join(DATA_DIR, "uploaded_pdfs")
    # Clean previous uploads
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir, exist_ok=True)

    for uf in uploaded_files:
        file_path = os.path.join(upload_dir, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        logger.info("Saved uploaded file: %s", file_path)

    # Derive output filenames from a combined name
    dir_name = "uploaded_pdfs"
    pages_csv = os.path.join(DATA_DIR, f"{dir_name}_pages.csv")
    chunks_csv = os.path.join(DATA_DIR, f"{dir_name}_chunks.csv")
    embeddings_pkl = os.path.join(DATA_DIR, f"{dir_name}_embeddings.pkl")

    # Delete existing outputs if force-reprocess is checked
    if force_reprocess:
        for path in [pages_csv, chunks_csv, embeddings_pkl]:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Removed existing output: %s", path)

    with st.spinner(f"Processing {len(uploaded_files)} PDF(s) — this may take a while…"):
        try:
            processed_files = process_pdf_directory_for_rag(
                pdf_dir=upload_dir,
                pages_csv_path=pages_csv,
                chunks_csv_path=chunks_csv,
                embeddings_tensor_path=embeddings_pkl,
            )
        except Exception as e:
            logger.error("PDF processing failed: %s", e)
            st.error(f"Processing failed: {e}")
            st.stop()

    st.session_state.processing_done = True
    st.session_state.last_processed_dir = upload_dir
    st.session_state.active_chunks_csv = chunks_csv
    st.session_state.active_embeddings_path = embeddings_pkl
    st.session_state.processed_files = processed_files

    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s) into a unified knowledge base!")
    st.info(f"Pages CSV: `{pages_csv}`")
    st.info(f"Chunks CSV: `{chunks_csv}`")
    st.info(f"Embeddings: `{embeddings_pkl}`")
    if processed_files:
        st.markdown("**Processed files:**")
        for pf in processed_files:
            st.markdown(f"- `{pf}`")

# Show status if already processed in this session
if st.session_state.processing_done and st.session_state.last_processed_dir:
    st.divider()
    st.markdown("**✅ Knowledge base is ready!**")
    if st.session_state.processed_files:
        st.markdown(f"**PDFs in knowledge base:** {len(st.session_state.processed_files)}")
        for pf in st.session_state.processed_files:
            st.markdown(f"- `{pf}`")

# Logout button
if st.button("Logout"):
    st.session_state.user_logged_in = False
    st.session_state.user_email = None
    st.session_state.last_processed_dir = None
    st.session_state.processing_done = False
    st.session_state.active_chunks_csv = None
    st.session_state.active_embeddings_path = None
    st.session_state.processed_files = []
    st.switch_page("app.py")