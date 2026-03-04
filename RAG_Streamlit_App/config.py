"""
Centralized configuration for the RAG Streamlit App.
Loads environment variables, validates required keys, and resolves paths.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Resolve the app root directory (where this file lives) so all paths are
# relative to it regardless of the working directory.
# ---------------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load .env from app root
load_dotenv(os.path.join(APP_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Logging – replaces scattered print() calls
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("rag_app")

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.warning(
        "OPENAI_API_KEY is not set. Export it or add it to .env before using the app."
    )

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Data paths (all relative to APP_ROOT)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(APP_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PDF_DIR = os.path.join(DATA_DIR, "pdfs")  # Directory containing PDF files for batch processing
os.makedirs(PDF_DIR, exist_ok=True)

PDF_PATH = os.path.join(DATA_DIR, "Mahabharata.pdf")
PAGES_CSV = os.path.join(DATA_DIR, "Mahabharata_pages.csv")
CHUNKS_CSV = os.path.join(DATA_DIR, "Mahabharata_chunks.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "MahaBharata_embeddings.pkl")

# ---------------------------------------------------------------------------
# Chunking / embedding defaults
# ---------------------------------------------------------------------------
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30

# Page‑number offset applied when reading the Mahabharata PDF.
# The first 20 pages are front‑matter, so we subtract 20 so that
# "page 1" in the CSV corresponds to the first real page of content.
PAGE_NUMBER_OFFSET = 0

# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
if DEBUG_MODE:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501/")
CLIENT_SECRET_PATH = os.path.join(APP_ROOT, "client_secret.json")

# ---------------------------------------------------------------------------
# Query length limit (basic prompt‑injection mitigation)
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH = 1000

# ---------------------------------------------------------------------------
# Shared OpenAI client (lazy singleton)
# ---------------------------------------------------------------------------
_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return a shared OpenAI client, created on first use."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or add it to .env / .bashrc."
            )
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client
