import os
import sys
import glob

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

st.title("📂 Upload a PDF Directory")
st.write(f"Logged in as: {st.session_state.user_email}")

st.markdown(
    "Enter the **absolute path** to a directory containing one or more PDF files. "
    "All PDFs in that folder will be combined into a **single knowledge base** for RAG."
)

pdf_dir_input = st.text_input(
    "PDF Directory Path",
    value=PDF_DIR,
    help="e.g. /home/user/documents/pdfs",
)

# Show a preview of PDF files found in the directory
if pdf_dir_input and os.path.isdir(pdf_dir_input):
    found_pdfs = sorted(glob.glob(os.path.join(pdf_dir_input, "*.pdf")))
    if found_pdfs:
        st.success(f"Found **{len(found_pdfs)}** PDF file(s) in `{pdf_dir_input}`:")
        for fp in found_pdfs:
            st.markdown(f"- `{os.path.basename(fp)}`")
    else:
        st.warning(f"No `.pdf` files found in `{pdf_dir_input}`.")
elif pdf_dir_input:
    st.error(f"Directory does not exist: `{pdf_dir_input}`")

# ---------------------------------------------------------
# Handle processing
# ---------------------------------------------------------
force_reprocess = st.checkbox("🔄 Force re-process (delete existing outputs)", value=False)

if st.button("🚀 Process All PDFs"):
    if not pdf_dir_input or not os.path.isdir(pdf_dir_input):
        st.error("Please enter a valid directory path.")
        st.stop()

    found_pdfs = sorted(glob.glob(os.path.join(pdf_dir_input, "*.pdf")))
    if not found_pdfs:
        st.error(f"No PDF files found in `{pdf_dir_input}`.")
        st.stop()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Derive output filenames from the directory name
    dir_name = os.path.basename(os.path.normpath(pdf_dir_input))
    pages_csv = os.path.join(DATA_DIR, f"{dir_name}_pages.csv")
    chunks_csv = os.path.join(DATA_DIR, f"{dir_name}_chunks.csv")
    embeddings_pkl = os.path.join(DATA_DIR, f"{dir_name}_embeddings.pkl")

    # Delete existing outputs if force-reprocess is checked
    if force_reprocess:
        for path in [pages_csv, chunks_csv, embeddings_pkl]:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Removed existing output: %s", path)

    with st.spinner(f"Processing {len(found_pdfs)} PDF(s) — this may take a while…"):
        try:
            processed_files = process_pdf_directory_for_rag(
                pdf_dir=pdf_dir_input,
                pages_csv_path=pages_csv,
                chunks_csv_path=chunks_csv,
                embeddings_tensor_path=embeddings_pkl,
            )
        except Exception as e:
            logger.error("PDF directory processing failed: %s", e)
            st.error(f"Processing failed: {e}")
            st.stop()

    st.session_state.processing_done = True
    st.session_state.last_processed_dir = pdf_dir_input
    st.session_state.active_chunks_csv = chunks_csv
    st.session_state.active_embeddings_path = embeddings_pkl
    st.session_state.processed_files = processed_files

    st.success(f"✅ Successfully processed {len(found_pdfs)} PDF(s) into a unified knowledge base!")
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
    st.markdown(f"**Last processed directory:** `{st.session_state.last_processed_dir}`")
    if st.session_state.processed_files:
        st.markdown(f"**PDFs in knowledge base:** {len(st.session_state.processed_files)}")

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