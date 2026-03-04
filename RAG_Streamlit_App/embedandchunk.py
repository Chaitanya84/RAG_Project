# Improved PDF -> sentence chunks -> embeddings pipeline
# Uses OpenAI Embeddings API instead of local SentenceTransformer.
import fitz
import os
import glob
import pandas as pd
import numpy as np
import re
import pickle
from tqdm.auto import tqdm
import spacy
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


def text_formatter(text: str) -> str:
    """Minor, document-specific text normalization."""
    return text.replace("\n", " ").strip()


def open_and_read_pdf(pdf_path: str, page_number_offset: int | None = None) -> list[dict]:
    """
    Open PDF and return a list of dicts with page metadata and extracted text.
    Each dict includes a 'source_file' key with the PDF filename.
    """
    offset = page_number_offset if page_number_offset is not None else PAGE_NUMBER_OFFSET
    source_file = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), desc=f"Reading pages [{source_file}]"):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "source_file": source_file,
            "page_number": page_number - offset,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    return pages_and_texts


def open_and_read_pdf_directory(pdf_dir: str) -> list[dict]:
    """
    Read ALL PDF files in the given directory and return a combined list of
    page dicts. Each dict carries a 'source_file' key so the origin PDF can
    be traced later.
    """
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")

    all_pages: list[dict] = []
    for pdf_path in pdf_paths:
        logger.info("Reading PDF: %s", pdf_path)
        pages = open_and_read_pdf(pdf_path, page_number_offset=0)
        all_pages.extend(pages)
        logger.info("  ✅ %d pages from %s", len(pages), os.path.basename(pdf_path))

    logger.info("📚 Total pages loaded from %d PDF(s): %d", len(pdf_paths), len(all_pages))
    return all_pages


def add_sentences_with_spacy(pages_and_texts: list[dict], nlp) -> None:
    """
    Use nlp.pipe to process pages in batches and add 'sentences' and
    'page_sentence_count_spacy' to each page dict in place.
    """
    texts = [item["text"] for item in pages_and_texts]
    for doc, item in tqdm(zip(nlp.pipe(texts, batch_size=32), pages_and_texts),
                         desc="Sentence segmentation", total=len(texts)):
        sents = [str(s) for s in doc.sents]
        item["sentences"] = sents
        item["page_sentence_count_spacy"] = len(sents)


def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """Split a list into sublists of size slice_size (last one may be smaller)."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


def build_sentence_chunks(pages_and_texts: list[dict], chunk_size: int) -> list[dict]:
    """
    Build sentence chunks from pages_and_texts and return list of chunk dicts.
    """
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Chunking sentences"):
        item["sentence_chunks"] = split_list(item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
        for sentence_chunk in item["sentence_chunks"]:
            joined = "".join(sentence_chunk).replace("  ", " ").strip()
            joined = re.sub(r'\.([A-Z])', r'. \1', joined)
            chunk_dict = {
                "source_file": item.get("source_file", ""),
                "page_number": item["page_number"],
                "sentence_chunk": joined,
                "chunk_char_count": len(joined),
                "chunk_word_count": len(joined.split(" ")),
                "chunk_token_count": len(joined) / 4
            }
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks


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


def embed_texts_openai(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a list of texts using the OpenAI Embeddings API.
    Processes in batches to stay within API limits.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    client = get_openai_client()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks via OpenAI"):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = _embed_batch(client, batch, OPENAI_EMBEDDING_MODEL)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("Embedding API failed on batch %d: %s", i, e)
            raise
    return np.array(all_embeddings, dtype=np.float32)


def filter_and_embed_chunks(
    pages_and_chunks: list[dict],
    embeddings_file_path: str | None = None,
) -> list[dict]:
    """
    Filter chunks by MIN_TOKEN_LENGTH and compute embeddings via OpenAI API.
    Returns list of chunk dicts (only those above the token length threshold)
    with an added 'embedding' key. Also persists embeddings to a pickle file.
    """
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min = df[df["chunk_token_count"] > MIN_TOKEN_LENGTH].to_dict(
        orient="records"
    )

    if not pages_and_chunks_over_min:
        logger.warning("No chunks above MIN_TOKEN_LENGTH; skipping embedding.")
        return []

    all_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min]

    # Use OpenAI API for embeddings
    all_chunk_embeddings = embed_texts_openai(all_chunks)

    # Decide where to save embeddings
    save_path = embeddings_file_path or EMBEDDINGS_FILE
    with open(save_path, "wb") as f:
        pickle.dump(all_chunk_embeddings, f)
    logger.info("Saved embeddings to: %s", save_path)

    # Attach embeddings back to dicts
    for i, item in enumerate(pages_and_chunks_over_min):
        item["embedding"] = all_chunk_embeddings[i].tolist()
    return pages_and_chunks_over_min


def process_pdf_for_rag(
    pdf_path: str,
    pages_csv_path: str,
    chunks_csv_path: str,
    embeddings_tensor_path: str,
) -> None:
    logger.info("process_pdf_for_rag called with:")
    logger.info("  pdf_path          = %s", pdf_path)
    logger.info("  pages_csv_path    = %s (exists: %s)", pages_csv_path, os.path.exists(pages_csv_path))
    logger.info("  chunks_csv_path   = %s (exists: %s)", chunks_csv_path, os.path.exists(chunks_csv_path))
    logger.info("  embeddings_file   = %s (exists: %s)", embeddings_tensor_path, os.path.exists(embeddings_tensor_path))

    if (
        os.path.exists(pages_csv_path)
        and os.path.exists(chunks_csv_path)
        and os.path.exists(embeddings_tensor_path)
    ):
        logger.info(
            "Outputs already exist, skipping processing:\n"
            "  Pages CSV: %s\n  Chunks CSV: %s\n  Embeddings: %s",
            pages_csv_path, chunks_csv_path, embeddings_tensor_path,
        )
        return

    # If only some outputs exist, warn and re-process everything
    for path, label in [
        (pages_csv_path, "Pages CSV"),
        (chunks_csv_path, "Chunks CSV"),
        (embeddings_tensor_path, "Embeddings"),
    ]:
        if os.path.exists(path):
            logger.warning("Partial output found (%s: %s). Will re-process from scratch.", label, path)

    logger.info("Processing PDF: %s", pdf_path)

    # 1) Read pages
    pages_and_texts = open_and_read_pdf(pdf_path)

    # 2) Add sentences with spaCy
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner"])
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Downloading now...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner"])
    nlp.add_pipe("sentencizer")
    add_sentences_with_spacy(pages_and_texts, nlp)

    # 3) Save page-level CSV
    df_pages = pd.DataFrame(pages_and_texts)
    if "page_sentence_count_spacy" not in df_pages.columns:
        df_pages["page_sentence_count_spacy"] = df_pages["sentences"].apply(len)
    # Ensure source_file column is present
    if "source_file" not in df_pages.columns:
        df_pages["source_file"] = os.path.basename(pdf_path)
    df_pages.to_csv(pages_csv_path, index=False)
    logger.info("Saved pages CSV to: %s", pages_csv_path)

    # 4) Build chunks
    pages_and_chunks = build_sentence_chunks(
        pages_and_texts, chunk_size=NUM_SENTENCE_CHUNK_SIZE
    )

    # 5) Filter + embed via OpenAI API
    pages_and_chunks_over_min = filter_and_embed_chunks(
        pages_and_chunks,
        embeddings_file_path=embeddings_tensor_path,
    )

    # 6) Save chunk-level CSV
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min)
    text_chunks_and_embeddings_df.to_csv(chunks_csv_path, index=False)
    logger.info("Saved chunks CSV to: %s", chunks_csv_path)

    logger.info("Done. Pages CSV: %s", pages_csv_path)
    logger.info("      Chunks CSV: %s", chunks_csv_path)
    logger.info("      Embeddings: %s", embeddings_tensor_path)


def process_pdf_directory_for_rag(
    pdf_dir: str,
    pages_csv_path: str,
    chunks_csv_path: str,
    embeddings_tensor_path: str,
) -> list[str]:
    """
    Process ALL PDFs in *pdf_dir* into a single unified knowledge base.

    Returns the list of PDF filenames that were processed.
    """
    logger.info("process_pdf_directory_for_rag called with:")
    logger.info("  pdf_dir             = %s", pdf_dir)
    logger.info("  pages_csv_path      = %s", pages_csv_path)
    logger.info("  chunks_csv_path     = %s", chunks_csv_path)
    logger.info("  embeddings_file     = %s", embeddings_tensor_path)

    if (
        os.path.exists(pages_csv_path)
        and os.path.exists(chunks_csv_path)
        and os.path.exists(embeddings_tensor_path)
    ):
        logger.info(
            "Outputs already exist, skipping processing:\n"
            "  Pages CSV: %s\n  Chunks CSV: %s\n  Embeddings: %s",
            pages_csv_path, chunks_csv_path, embeddings_tensor_path,
        )
        return []

    # Warn about partial outputs
    for path, label in [
        (pages_csv_path, "Pages CSV"),
        (chunks_csv_path, "Chunks CSV"),
        (embeddings_tensor_path, "Embeddings"),
    ]:
        if os.path.exists(path):
            logger.warning("Partial output found (%s: %s). Will re-process from scratch.", label, path)

    # 1) Read all PDFs in directory
    pages_and_texts = open_and_read_pdf_directory(pdf_dir)
    processed_files = list({p["source_file"] for p in pages_and_texts})

    # 2) Add sentences with spaCy
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner"])
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Downloading now...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner"])
    nlp.add_pipe("sentencizer")
    add_sentences_with_spacy(pages_and_texts, nlp)

    # 3) Save page-level CSV
    df_pages = pd.DataFrame(pages_and_texts)
    if "page_sentence_count_spacy" not in df_pages.columns:
        df_pages["page_sentence_count_spacy"] = df_pages["sentences"].apply(len)
    df_pages.to_csv(pages_csv_path, index=False)
    logger.info("Saved pages CSV to: %s", pages_csv_path)

    # 4) Build chunks
    pages_and_chunks = build_sentence_chunks(
        pages_and_texts, chunk_size=NUM_SENTENCE_CHUNK_SIZE
    )

    # 5) Filter + embed via OpenAI API
    pages_and_chunks_over_min = filter_and_embed_chunks(
        pages_and_chunks,
        embeddings_file_path=embeddings_tensor_path,
    )

    # 6) Save chunk-level CSV
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min)
    text_chunks_and_embeddings_df.to_csv(chunks_csv_path, index=False)
    logger.info("Saved chunks CSV to: %s", chunks_csv_path)

    logger.info("Done processing %d PDFs.", len(processed_files))
    logger.info("  Pages CSV:   %s", pages_csv_path)
    logger.info("  Chunks CSV:  %s", chunks_csv_path)
    logger.info("  Embeddings:  %s", embeddings_tensor_path)

    return sorted(processed_files)