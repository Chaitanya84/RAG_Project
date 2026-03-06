import os
import sys
import glob
import shutil
import threading
import time

# Portable path: resolve relative to this file's location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from config import DATA_DIR, PDF_DIR, logger

# NOTE: st.set_page_config() is only called in the main app.py entry point.

# Require login
if "user_logged_in" not in st.session_state or not st.session_state.user_logged_in:
    st.error("You are not logged in. Please go to the main page and log in first.")
    if st.button("Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_pdfs")

# ---------------------------------------------------------------------------
# Session state initialisation (done ONCE per session)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "last_processed_dir": None,
    "processing_done": False,
    "active_chunks_csv": None,
    "active_embeddings_path": None,
    "processed_files": [],
    "upload_saved": False,          # True after files are persisted to disk
    "bg_processing": False,         # True while background thread is running
    "bg_error": None,               # Error message from background thread
    "bg_progress": "",              # Progress message from background thread
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Background processing function (runs in a thread)
# ---------------------------------------------------------------------------
def _run_processing_in_background(
    upload_dir: str,
    pages_csv: str,
    chunks_csv: str,
    embeddings_pkl: str,
):
    """Run the heavy PDF processing in a background thread."""
    try:
        # Import here to avoid loading spaCy/heavy libs on every Streamlit rerun
        from embedandchunk import process_pdf_directory_for_rag

        st.session_state.bg_progress = "📖 Reading PDFs and extracting text…"
        processed_files = process_pdf_directory_for_rag(
            pdf_dir=upload_dir,
            pages_csv_path=pages_csv,
            chunks_csv_path=chunks_csv,
            embeddings_tensor_path=embeddings_pkl,
        )

        st.session_state.processing_done = True
        st.session_state.last_processed_dir = upload_dir
        st.session_state.active_chunks_csv = chunks_csv
        st.session_state.active_embeddings_path = embeddings_pkl
        st.session_state.processed_files = processed_files
        st.session_state.bg_progress = "✅ Done!"
        logger.info("Background processing completed successfully.")

    except Exception as e:
        logger.error("Background PDF processing failed: %s", e)
        st.session_state.bg_error = str(e)
        st.session_state.bg_progress = f"❌ Failed: {e}"
    finally:
        st.session_state.bg_processing = False


# ---------------------------------------------------------------------------
# Page UI
# ---------------------------------------------------------------------------
st.title("📂 Upload PDF Files")
st.write(f"Logged in as: {st.session_state.user_email}")

st.markdown(
    "Use the **file uploader** below to select one or more PDF files from your computer. "
    "All uploaded PDFs will be combined into a **single knowledge base** for RAG."
)

# ---------------------------------------------------------------------------
# If background processing is running, show progress and auto-refresh
# ---------------------------------------------------------------------------
if st.session_state.bg_processing:
    st.warning("⏳ **Processing is in progress — please wait…**")
    st.info(st.session_state.bg_progress)
    st.caption("This page auto-refreshes every 5 seconds.")
    time.sleep(5)
    st.rerun()

# Show any error from the background thread
if st.session_state.bg_error:
    st.error(f"Processing failed: {st.session_state.bg_error}")
    if st.button("🔄 Clear error and retry"):
        st.session_state.bg_error = None
        st.session_state.bg_progress = ""
        st.rerun()

# ---------------------------------------------------------------------------
# File uploader (only shown when NOT processing)
# ---------------------------------------------------------------------------
if not st.session_state.bg_processing:
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

        # Clean previous uploads and save new ones to disk IMMEDIATELY
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        for uf in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
            logger.info("Saved uploaded file: %s", file_path)

        st.session_state.upload_saved = True

        # Derive output filenames
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

        # Reset state and launch background thread
        st.session_state.bg_processing = True
        st.session_state.bg_error = None
        st.session_state.bg_progress = "🚀 Starting processing…"
        st.session_state.processing_done = False
        st.session_state.processed_files = []

        thread = threading.Thread(
            target=_run_processing_in_background,
            args=(UPLOAD_DIR, pages_csv, chunks_csv, embeddings_pkl),
            daemon=True,
        )
        thread.start()
        logger.info("Background processing thread started.")

        st.rerun()

# ---------------------------------------------------------------------------
# Show status if already processed in this session
# ---------------------------------------------------------------------------
if st.session_state.processing_done and not st.session_state.bg_processing:
    st.divider()
    st.markdown("### ✅ Knowledge base is ready!")
    if st.session_state.processed_files:
        st.markdown(f"**PDFs in knowledge base:** {len(st.session_state.processed_files)}")
        for pf in st.session_state.processed_files:
            st.markdown(f"- `{pf}`")
    st.markdown("Navigate to **Chat** or **Quiz** pages to use the knowledge base.")

# Logout button
if st.button("Logout"):
    st.session_state.user_logged_in = False
    st.session_state.user_email = None
    for key in _DEFAULTS:
        st.session_state.pop(key, None)
    st.switch_page("app.py")