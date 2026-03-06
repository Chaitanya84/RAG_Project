import os
import sys
import glob
import json
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
# File-based signaling for the background thread (avoids ScriptRunContext issues)
_STATUS_FILE = os.path.join(DATA_DIR, "_processing_status.json")

# ---------------------------------------------------------------------------
# Session state initialisation (done ONCE per session)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "last_processed_dir": None,
    "processing_done": False,
    "active_chunks_csv": None,
    "active_embeddings_path": None,
    "processed_files": [],
    "upload_saved": False,
    "bg_processing": False,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# File-based status helpers (thread-safe — no st.session_state from thread)
# ---------------------------------------------------------------------------
def _write_status(status: str, error: str = "", processed_files: list | None = None):
    """Write processing status to a JSON file on disk (called from background thread)."""
    payload = {
        "status": status,           # "running", "done", "error"
        "error": error,
        "processed_files": processed_files or [],
        "timestamp": time.time(),
    }
    try:
        with open(_STATUS_FILE, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass  # best-effort — don't crash the processing thread


def _read_status() -> dict:
    """Read processing status from disk (called from Streamlit main thread)."""
    try:
        if os.path.exists(_STATUS_FILE):
            with open(_STATUS_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {"status": "idle", "error": "", "processed_files": []}


def _clear_status():
    """Remove the status file."""
    try:
        if os.path.exists(_STATUS_FILE):
            os.remove(_STATUS_FILE)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Background processing function (runs in a thread — NO st.session_state!)
# ---------------------------------------------------------------------------
def _run_processing_in_background(
    upload_dir: str,
    pages_csv: str,
    chunks_csv: str,
    embeddings_pkl: str,
):
    """Run the heavy PDF processing in a background thread."""
    try:
        _write_status("running")

        # Import here to avoid loading heavy libs on every Streamlit rerun
        from embedandchunk import process_pdf_directory_for_rag

        processed_files = process_pdf_directory_for_rag(
            pdf_dir=upload_dir,
            pages_csv_path=pages_csv,
            chunks_csv_path=chunks_csv,
            embeddings_tensor_path=embeddings_pkl,
        )

        _write_status("done", processed_files=processed_files)
        logger.info("Background processing completed successfully.")

    except Exception as e:
        logger.error("Background PDF processing failed: %s", e)
        _write_status("error", error=str(e))


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
# Check background processing status from file (thread-safe)
# ---------------------------------------------------------------------------
disk_status = _read_status()

# Sync file-based status → session state
if disk_status["status"] == "done" and st.session_state.bg_processing:
    # Processing finished! Update session state from disk status.
    dir_name = "uploaded_pdfs"
    st.session_state.bg_processing = False
    st.session_state.processing_done = True
    st.session_state.last_processed_dir = UPLOAD_DIR
    st.session_state.active_chunks_csv = os.path.join(DATA_DIR, f"{dir_name}_chunks.csv")
    st.session_state.active_embeddings_path = os.path.join(DATA_DIR, f"{dir_name}_embeddings.pkl")
    st.session_state.processed_files = disk_status.get("processed_files", [])
    _clear_status()
    st.rerun()

if disk_status["status"] == "error" and st.session_state.bg_processing:
    st.session_state.bg_processing = False
    _clear_status()
    st.error(f"❌ Processing failed: {disk_status['error']}")
    if st.button("🔄 Clear error and retry"):
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# If background processing is running, show progress and auto-refresh
# ---------------------------------------------------------------------------
if st.session_state.bg_processing:
    st.warning("⏳ **Processing is in progress — please wait…**")
    st.info("📖 Reading PDFs, chunking text, and generating embeddings via OpenAI…")
    st.caption("This page auto-refreshes every 3 seconds. Do not navigate away.")
    time.sleep(3)
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

        # Clear any stale status and launch background thread
        _clear_status()
        st.session_state.bg_processing = True
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
    _clear_status()
    for key in _DEFAULTS:
        st.session_state.pop(key, None)
    st.switch_page("app.py")