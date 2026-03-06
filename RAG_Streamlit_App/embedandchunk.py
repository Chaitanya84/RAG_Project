# Improved PDF -> sentence chunks -> embeddings pipeline
# Uses OpenAI Embeddings API instead of local SentenceTransformer.
# Memory-optimised for deployment on constrained environments (e.g. Render 512 MB).
import fitz
import os
import glob
import re
import csv
import gc
import pickle
import numpy as np
from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError

from config import (
    APP_ROOT,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    DATA_DIR,
    NUM_SENTENCE_CHUNK_SIZE,
    MIN_TOKEN_LENGTH,
    PAGE_NUMBER_OFFSET,
    PAGES_CSV,
    CHUNKS_CSV as CHUNKS_EMBEDDINGS_CSV,
    EMBEDDINGS_PATH as EMBEDDINGS_FILE,
    PDF_PATH,
    get_openai_client,
    logger,
)

_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)

# ---------------------------------------------------------------------------
# Lightweight regex-based sentence splitter (replaces spaCy — saves ~150 MB)
# ---------------------------------------------------------------------------
# Splits on period/question-mark/exclamation followed by whitespace and an
# uppercase letter, or on newline boundaries that look like sentence ends.
_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'   # standard sentence boundary
    r'|(?<=[.!?])\s*\n\s*'      # sentence end followed by newline
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight regex."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def text_formatter(text: str) -> str:
    """Minor, document-specific text normalization."""
    return text.replace("\n", " ").strip()


def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """Split a list into sublists of size slice_size (last one may be smaller)."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


# ---------------------------------------------------------------------------
# Streaming page reader — yields one page dict at a time to avoid holding
# the entire PDF in memory.
# ---------------------------------------------------------------------------
def _iter_pdf_pages(pdf_path: str, page_number_offset: int = 0):
    """Yield one page dict at a time from a PDF, closing it when done."""
    source_file = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        for page_number in range(len(doc)):
            page = doc[page_number]
            text = text_formatter(page.get_text())
            sentences = _split_sentences(text)
            yield {
                "source_file": source_file,
                "page_number": page_number - page_number_offset,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
                "sentences": sentences,
                "page_sentence_count_spacy": len(sentences),
            }
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Build chunks from a single page dict (avoids keeping all pages in memory)
# ---------------------------------------------------------------------------
def _chunks_from_page(page_dict: dict, chunk_size: int) -> list[dict]:
    """Return chunk dicts for one page."""
    sentence_groups = split_list(page_dict["sentences"], slice_size=chunk_size)
    chunks = []
    for group in sentence_groups:
        joined = " ".join(group).replace("  ", " ").strip()
        joined = re.sub(r'\.([A-Z])', r'. \1', joined)
        token_count = len(joined) / 4
        if token_count <= MIN_TOKEN_LENGTH:
            continue
        chunks.append({
            "source_file": page_dict["source_file"],
            "page_number": page_dict["page_number"],
            "sentence_chunk": joined,
            "chunk_char_count": len(joined),
            "chunk_word_count": len(joined.split(" ")),
            "chunk_token_count": token_count,
        })
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=lambda rs: logger.warning("Retrying OpenAI embed batch (attempt %d)…", rs.attempt_number),
)
def _embed_batch(client, batch: list[str], model: str) -> list:
    """Embed a single batch with retry logic."""
    response = client.embeddings.create(model=model, input=batch)
    return [item.embedding for item in response.data]


def _embed_and_save_streaming(
    chunks: list[dict],
    embeddings_path: str,
    batch_size: int = 50,
) -> None:
    """
    Embed chunk texts via OpenAI in small batches and save the embedding
    matrix to disk.  Keeps only one batch of raw API responses in memory
    at a time.
    """
    client = get_openai_client()
    texts = [c["sentence_chunk"] for c in chunks]
    total = len(texts)

    # Pre-allocate: text-embedding-3-small = 1536 dims
    embeddings = np.empty((total, 1536), dtype=np.float32)

    for start in tqdm(range(0, total, batch_size), desc="Embedding chunks via OpenAI"):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        try:
            batch_embs = _embed_batch(client, batch, OPENAI_EMBEDDING_MODEL)
            for j, emb in enumerate(batch_embs):
                embeddings[start + j] = emb
            del batch_embs
        except Exception as e:
            logger.error("Embedding API failed on batch %d: %s", start, e)
            raise

    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("Saved embeddings (%d × %d) to: %s", *embeddings.shape, embeddings_path)


# ---------------------------------------------------------------------------
# Legacy wrappers (kept for backward compat with AnswerAll.py / CLI demo)
# ---------------------------------------------------------------------------
def open_and_read_pdf(pdf_path: str, page_number_offset: int | None = None) -> list[dict]:
    offset = page_number_offset if page_number_offset is not None else PAGE_NUMBER_OFFSET
    return list(_iter_pdf_pages(pdf_path, page_number_offset=offset))


def open_and_read_pdf_directory(pdf_dir: str) -> list[dict]:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")
    all_pages: list[dict] = []
    for pdf_path in pdf_paths:
        logger.info("Reading PDF: %s", pdf_path)
        pages = list(_iter_pdf_pages(pdf_path, page_number_offset=0))
        all_pages.extend(pages)
        logger.info("  ✅ %d pages from %s", len(pages), os.path.basename(pdf_path))
    logger.info("📚 Total pages loaded from %d PDF(s): %d", len(pdf_paths), len(all_pages))
    return all_pages


def add_sentences_with_spacy(pages_and_texts: list[dict], nlp=None) -> None:
    """Backward-compat shim — now uses regex splitter, nlp arg is ignored."""
    for item in pages_and_texts:
        if "sentences" not in item:
            item["sentences"] = _split_sentences(item["text"])
            item["page_sentence_count_spacy"] = len(item["sentences"])


def build_sentence_chunks(pages_and_texts: list[dict], chunk_size: int) -> list[dict]:
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Chunking sentences"):
        pages_and_chunks.extend(_chunks_from_page(item, chunk_size))
    return pages_and_chunks


def embed_texts_openai(texts: list[str], batch_size: int = 50) -> np.ndarray:
    client = get_openai_client()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks via OpenAI"):
        batch = texts[i:i + batch_size]
        batch_embeddings = _embed_batch(client, batch, OPENAI_EMBEDDING_MODEL)
        all_embeddings.extend(batch_embeddings)
        del batch_embeddings
    return np.array(all_embeddings, dtype=np.float32)


def filter_and_embed_chunks(
    pages_and_chunks: list[dict],
    embeddings_file_path: str | None = None,
) -> list[dict]:
    pages_and_chunks_over_min = [
        c for c in pages_and_chunks if c["chunk_token_count"] > MIN_TOKEN_LENGTH
    ]
    if not pages_and_chunks_over_min:
        logger.warning("No chunks above MIN_TOKEN_LENGTH; skipping embedding.")
        return []
    all_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min]
    all_chunk_embeddings = embed_texts_openai(all_chunks)
    save_path = embeddings_file_path or EMBEDDINGS_FILE
    with open(save_path, "wb") as f:
        pickle.dump(all_chunk_embeddings, f)
    logger.info("Saved embeddings to: %s", save_path)
    for i, item in enumerate(pages_and_chunks_over_min):
        item["embedding"] = all_chunk_embeddings[i].tolist()
    return pages_and_chunks_over_min


# ---------------------------------------------------------------------------
# Main entry-point: MEMORY-OPTIMISED directory processing
# ---------------------------------------------------------------------------
def process_pdf_directory_for_rag(
    pdf_dir: str,
    pages_csv_path: str,
    chunks_csv_path: str,
    embeddings_tensor_path: str,
) -> list[str]:
    """
    Process ALL PDFs in *pdf_dir* into a single unified knowledge base.
    Memory-optimised: streams pages one at a time, writes CSVs row-by-row,
    and only keeps the final chunk list for embedding.

    Returns the list of PDF filenames that were processed.
    """
    logger.info("process_pdf_directory_for_rag (memory-optimised) called with:")
    logger.info("  pdf_dir             = %s", pdf_dir)
    logger.info("  pages_csv_path      = %s", pages_csv_path)
    logger.info("  chunks_csv_path     = %s", chunks_csv_path)
    logger.info("  embeddings_file     = %s", embeddings_tensor_path)

    if (
        os.path.exists(pages_csv_path)
        and os.path.exists(chunks_csv_path)
        and os.path.exists(embeddings_tensor_path)
    ):
        logger.info("Outputs already exist — skipping processing.")
        return []

    # Clean partial outputs
    for path in [pages_csv_path, chunks_csv_path, embeddings_tensor_path]:
        if os.path.exists(path):
            os.remove(path)
            logger.warning("Removed partial output: %s", path)

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")

    processed_files: set[str] = set()
    all_chunks: list[dict] = []

    # ---- Phase 1: Read PDFs page-by-page, write pages CSV, collect chunks ----
    PAGE_FIELDS = [
        "source_file", "page_number", "page_char_count", "page_word_count",
        "page_sentence_count_raw", "page_token_count", "text",
        "sentences", "page_sentence_count_spacy",
    ]
    CHUNK_FIELDS = [
        "source_file", "page_number", "sentence_chunk",
        "chunk_char_count", "chunk_word_count", "chunk_token_count",
    ]

    with open(pages_csv_path, "w", newline="", encoding="utf-8") as pf:
        page_writer = csv.DictWriter(pf, fieldnames=PAGE_FIELDS)
        page_writer.writeheader()

        for pdf_path in pdf_paths:
            source = os.path.basename(pdf_path)
            logger.info("Reading PDF: %s", pdf_path)
            page_count = 0
            for page_dict in _iter_pdf_pages(pdf_path, page_number_offset=0):
                # Write page row to CSV immediately
                page_writer.writerow(page_dict)
                # Build chunks for this page
                page_chunks = _chunks_from_page(page_dict, NUM_SENTENCE_CHUNK_SIZE)
                all_chunks.extend(page_chunks)
                page_count += 1
            processed_files.add(source)
            logger.info("  ✅ %d pages from %s", page_count, source)

    logger.info("📚 Total chunks collected: %d", len(all_chunks))
    gc.collect()

    # ---- Phase 2: Write chunks CSV (without embeddings column) ----
    with open(chunks_csv_path, "w", newline="", encoding="utf-8") as cf:
        chunk_writer = csv.DictWriter(cf, fieldnames=CHUNK_FIELDS)
        chunk_writer.writeheader()
        for chunk in all_chunks:
            chunk_writer.writerow({k: chunk[k] for k in CHUNK_FIELDS})
    logger.info("Saved chunks CSV to: %s", chunks_csv_path)

    # ---- Phase 3: Embed in small batches and save ----
    _embed_and_save_streaming(all_chunks, embeddings_tensor_path, batch_size=50)

    # Free the big list
    del all_chunks
    gc.collect()

    logger.info("Done processing %d PDFs.", len(processed_files))
    return sorted(processed_files)


# ---------------------------------------------------------------------------
# Single-PDF wrapper (kept for backward compat)
# ---------------------------------------------------------------------------
def process_pdf_for_rag(
    pdf_path: str,
    pages_csv_path: str,
    chunks_csv_path: str,
    embeddings_tensor_path: str,
) -> None:
    """Process a single PDF — delegates to directory processor."""
    pdf_dir = os.path.dirname(pdf_path)
    # Temporarily copy the single PDF to a temp dir so the directory processor works
    import shutil, tempfile
    tmp_dir = tempfile.mkdtemp(prefix="rag_single_")
    try:
        shutil.copy2(pdf_path, tmp_dir)
        process_pdf_directory_for_rag(
            pdf_dir=tmp_dir,
            pages_csv_path=pages_csv_path,
            chunks_csv_path=chunks_csv_path,
            embeddings_tensor_path=embeddings_tensor_path,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)