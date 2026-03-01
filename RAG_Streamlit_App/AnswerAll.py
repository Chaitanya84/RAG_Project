# Python 3.8+
# Use a pre-saved embeddings tensor (.pt) for fast startup.
# All CSV and image output filenames and printed output formats are preserved.

import fitz  # pymupdf
import random
import torch
import pandas as pd
import numpy as np
import re
import textwrap
from time import perf_counter as timer
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer
import os
from openai import OpenAI  # NEW

# -----------------------
# Config / constants
# -----------------------
DATA_DIR = "data"

PDF_PATH = os.path.join(DATA_DIR, "Mahabharata.pdf")

PAGES_CSV = os.path.join(DATA_DIR, "Mahabharata_pages.csv")
CHUNKS_CSV = os.path.join(DATA_DIR, "Mahabharata_chunks.csv")
EMBEDDINGS_TENSOR_PATH = os.path.join(DATA_DIR, "MahaBharata_embeddings.pt")

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OpenAI chat model name – adjust to what you have access to
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4o", etc.

# Initialize OpenAI client (expects OPENAI_API_KEY in env)
openai_client = OpenAI()

# -----------------------
# Utility functions
# -----------------------
def print_wrapped(text: str, wrap_length: int = 80) -> None:
    print(textwrap.fill(text, wrap_length))

# -----------------------
# Embedding & metadata loading (uses binary tensor)
# -----------------------
def load_embedding_model(device: str = DEVICE) -> SentenceTransformer:
    """Load the sentence-transformers model on the selected device."""
    model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=device)
    return model

def load_metadata_and_tensor(csv_path: str, tensor_path: str, device: str = DEVICE):
    """
    Load metadata CSV and embeddings tensor (.pt). Ensure shapes align.
    Returns pages_and_chunks (list of dicts) and embeddings tensor on `device`.
    """
    df = pd.read_csv(csv_path)
    embeddings_cpu = torch.load(tensor_path, map_location="cpu")
    if not isinstance(embeddings_cpu, torch.Tensor):
        raise ValueError(f"Loaded object from {tensor_path} is not a torch.Tensor")

    embeddings = embeddings_cpu.to(device)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings tensor must be 2D, got shape: {embeddings.shape}")

    if df.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Metadata rows ({df.shape[0]}) and embeddings count ({embeddings.shape[0]}) do not match. "
            "Ensure the CSV and tensor were saved from the same data and order."
        )

    pages_and_chunks = df.to_dict(orient="records")
    print(f"Loaded metadata rows: {len(pages_and_chunks)}")
    print(f"Shape of loaded embeddings tensor: {embeddings.shape}")
    print(df.head())
    return pages_and_chunks, embeddings

def load_rag_resources(device: str = DEVICE):
    """
    Load and return (embedding_model, pages_and_chunks, embeddings).
    Used by chatPage.py and cached with st.cache_resource.
    """
    embedding_model = load_embedding_model(device=device)
    pages_and_chunks, embeddings = load_metadata_and_tensor(
        CHUNKS_CSV,
        EMBEDDINGS_TENSOR_PATH,
        device=device,
    )
    return embedding_model, pages_and_chunks, embeddings

# -----------------------
# Retrieval
# -----------------------
def retrieve_relevant_resources(query: str,
                                embeddings: torch.Tensor,
                                model: SentenceTransformer,
                                top_k: int = 3,
                                print_time: bool = True):
    """Encode the query then compute dot-product vs embeddings, returning top-k scores and indices."""
    with torch.no_grad():
        query_emb = model.encode(query, convert_to_tensor=True)

    start = timer()
    dot_scores = util.dot_score(query_emb, embeddings)[0]  # shape: (N,)
    elapsed = timer() - start
    if print_time:
        print(f"[INFO] Time taken to get scores on {embeddings.shape[0]} embeddings: {elapsed:.5f} seconds.")
    scores, indices = torch.topk(dot_scores, k=top_k)
    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.Tensor,
                                 pages_and_chunks: list[dict],
                                 model: SentenceTransformer,
                                 top_k: int = 5):
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, model=model, top_k=top_k)
    print(f"Query: {query}\n")
    print("Results:")
    for score, idx in zip(scores, indices):
        idx = idx.item()
        print(f"Score: {score:.4f}")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
    return scores, indices

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """Build concise instruction + context prompt for OpenAI chat."""
    context = "\n\n".join([item["sentence_chunk"] for item in context_items])
    prompt = f"""You are an expert on the Mahabharata and related dharma shastras.

Based ONLY on the context passages below, answer the user's question in 2–4 sentences.
First clearly say "Yes" or "No" if the question allows, then briefly explain why
using the exact ideas in the context. Do NOT add information that is not in the context.
If the answer is not in the context, say you cannot answer from the given passages.

Context:
{context}

User question: {query}

Answer:"""

    print("=== RAG QUERY ===", query)
    print("=== CONTEXT ===", context)
    print("=== PROMPT ===", prompt[:1000])
    return prompt

def answer_with_rag(user_query: str) -> str:
    """
    Single-call RAG answer function for Streamlit:
    - Loads metadata + embeddings + embedding model.
    - Retrieves top-k chunks.
    - Builds prompt and returns answer text via OpenAI API.
    """
    embedding_model = load_embedding_model(device=DEVICE)
    pages_and_chunks, embeddings = load_metadata_and_tensor(
        CHUNKS_CSV, EMBEDDINGS_TENSOR_PATH, device=DEVICE
    )

    scores, indices = retrieve_relevant_resources(
        query=user_query, embeddings=embeddings, model=embedding_model
    )
    context_items = [pages_and_chunks[i.item()] for i in indices]

    prompt = prompt_formatter(query=user_query, context_items=context_items)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in the Mahabharata."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    answer = response.choices[0].message.content.strip()
    return answer

def answer_with_rag_cached(
    user_query: str,
    embedding_model,
    pages_and_chunks,
    embeddings,
    tokenizer,     # kept for signature compatibility, but unused
    llm_model,     # kept for signature compatibility, but unused
) -> str:
    """
    RAG answer that reuses preloaded embeddings / metadata and calls OpenAI.
    """
    scores, indices = retrieve_relevant_resources(
        query=user_query,
        embeddings=embeddings,
        model=embedding_model,
        top_k=3,
    )
    scores, indices = scores.cpu(), indices.cpu()
    sorted_pairs = sorted(zip(scores.tolist(), indices.tolist()), reverse=True)
    context_items = [pages_and_chunks[i] for _, i in sorted_pairs]

    prompt = prompt_formatter(query=user_query, context_items=context_items)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in the Mahabharata."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    answer = response.choices[0].message.content.strip()
    return answer

def load_llm_resources():
    """
    Placeholder for API-based LLM; kept for compatibility with chatPage.
    Returns (None, None).
    """
    return None, None

# -----------------------
# Main execution
# -----------------------
def main():
    # 1) Open PDF document so we can extract matched page image later
    document = fitz.open(PDF_PATH)

    # 2) Load embedding model and metadata + embeddings
    embedding_model = load_embedding_model(device=DEVICE)
    pages_and_chunks, embeddings = load_metadata_and_tensor(
        CHUNKS_CSV, EMBEDDINGS_TENSOR_PATH, device=DEVICE
    )

    # 3) Demo retrieval
    query = "Is Krishna talking to arjuna on the battlefeild?"
    print(f"Query: {query}")

    scores, indices = retrieve_relevant_resources(
        query=query, embeddings=embeddings, model=embedding_model
    )
    print(scores, indices)

    scores, indices = print_top_results_and_scores(
        query=query,
        embeddings=embeddings,
        pages_and_chunks=pages_and_chunks,
        model=embedding_model,
    )

    print(f"Highest score: {scores[0]:.4f}")
    print(f"Corresponding page number: {pages_and_chunks[indices[0].item()]['page_number']}")
    best_match_page_number = pages_and_chunks[indices[0].item()]["page_number"]
    print(f"Best match is on page number: {best_match_page_number}")

    # Save matched page as image (same filename)
    page = document[best_match_page_number + 41]  # original offset
    img = page.get_pixmap(dpi=300)
    img.save("matched_page.png")
    print("Saved matched page as 'matched_page.png'")

    # 4) RAG-style query via OpenAI
    query_list = gpt4_questions + manual_questions
    query = random.choice(query_list)
    print(f"RAG demo query: {query}")

    scores, indices = retrieve_relevant_resources(
        query=query, embeddings=embeddings, model=embedding_model
    )
    context_items = [pages_and_chunks[i.item()] for i in indices]

    prompt = prompt_formatter(query=query, context_items=context_items)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in the Mahabharata."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    answer = response.choices[0].message.content.strip()

    print(f"Query: {query}")
    print(f"RAG answer:\n{answer}")

    document.close()

if __name__ == "__main__":
    main()