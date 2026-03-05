# Python 3.10+
# Use a pre-saved embeddings file (.pkl) for fast startup.
# All retrieval and answering is done via OpenAI APIs.

import numpy as np
import pandas as pd
import os
import pickle
import textwrap
from time import perf_counter as timer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError

from config import (
    APP_ROOT,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    CHUNKS_CSV,
    EMBEDDINGS_PATH,
    MAX_QUERY_LENGTH,
    get_openai_client,
    logger,
)

# -----------------------
# Utility functions
# -----------------------
def print_wrapped(text: str, wrap_length: int = 80) -> None:
    print(textwrap.fill(text, wrap_length))


_RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)

# -----------------------
# Embedding & metadata loading
# -----------------------
@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=lambda rs: logger.warning("Retrying OpenAI embed call (attempt %d)…", rs.attempt_number),
)
def embed_query_openai(query: str) -> np.ndarray:
    """Embed a single query string using the OpenAI Embeddings API."""
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=[query],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error("Failed to embed query: %s", e)
        raise


def load_metadata_and_embeddings(csv_path: str, embeddings_path: str):
    """
    Load metadata CSV and embeddings file (.pkl).
    Returns pages_and_chunks (list of dicts) and embeddings (numpy array).
    """
    df = pd.read_csv(csv_path)

    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape: {embeddings.shape}")

    if df.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Metadata rows ({df.shape[0]}) and embeddings count ({embeddings.shape[0]}) do not match. "
            "Ensure the CSV and embeddings were saved from the same data and order."
        )

    pages_and_chunks = df.to_dict(orient="records")
    logger.info("Loaded metadata rows: %d", len(pages_and_chunks))
    logger.info("Shape of loaded embeddings: %s", embeddings.shape)
    logger.debug("First rows:\n%s", df.head())
    return pages_and_chunks, embeddings


def load_rag_resources(
    chunks_csv: str | None = None,
    embeddings_path: str | None = None,
):
    """
    Load and return (pages_and_chunks, embeddings).
    Accepts optional paths so the chat page can load any processed PDF,
    falling back to the default Mahabharata paths.
    """
    csv = chunks_csv or CHUNKS_CSV
    emb = embeddings_path or EMBEDDINGS_PATH
    return load_metadata_and_embeddings(csv, emb)


# -----------------------
# Retrieval (cosine similarity via numpy)
# -----------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def retrieve_relevant_resources(
    query: str,
    embeddings: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the query via OpenAI and compute cosine similarity, returning top-k scores and indices."""
    query_emb = embed_query_openai(query)

    start = timer()
    scores = cosine_similarity(query_emb, embeddings)
    elapsed = timer() - start
    logger.debug(
        "Similarity search over %d embeddings took %.5f s",
        embeddings.shape[0],
        elapsed,
    )

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    return top_scores, top_indices


def print_top_results_and_scores(
    query: str,
    embeddings: np.ndarray,
    pages_and_chunks: list[dict],
    top_k: int = 5,
):
    scores, indices = retrieve_relevant_resources(
        query=query, embeddings=embeddings, top_k=top_k,
    )
    print(f"Query: {query}\n")
    print("Results:")
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
    return scores, indices


# -----------------------
# Prompt + Answer
# -----------------------
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """Build concise instruction + context prompt for OpenAI chat."""
    context_parts = []
    for item in context_items:
        source = item.get("source_file", "unknown")
        page = item.get("page_number", "?")
        chunk_text = item["sentence_chunk"]
        context_parts.append(f"[Source: {source}, Page: {page}]\n{chunk_text}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a knowledgeable assistant that answers questions based on provided document passages.

Based ONLY on the context passages below, answer the user's question in 2–4 sentences.
First clearly say "Yes" or "No" if the question allows, then briefly explain why
using the exact ideas in the context. Do NOT add information that is not in the context.
If the answer is not in the context, say you cannot answer from the given passages.
When citing information, mention the source file name and page number.

Context:
{context}

User question: {query}

Answer:"""

    logger.debug("RAG QUERY: %s", query)
    logger.debug("CONTEXT (first 500 chars): %s", context[:500])
    return prompt


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=lambda rs: logger.warning("Retrying OpenAI chat call (attempt %d)…", rs.attempt_number),
)
def _call_chat_api(prompt: str) -> str:
    """Send a prompt to the OpenAI Chat API and return the answer text."""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions strictly based on provided document context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI Chat API call failed: %s", e)
        raise


def answer_with_rag(user_query: str) -> str:
    """
    Single-call RAG answer:
    - Loads metadata + embeddings.
    - Retrieves top-k chunks.
    - Calls OpenAI Chat API.
    """
    pages_and_chunks, embeddings = load_metadata_and_embeddings(
        CHUNKS_CSV, EMBEDDINGS_PATH
    )

    scores, indices = retrieve_relevant_resources(
        query=user_query, embeddings=embeddings
    )
    context_items = [pages_and_chunks[i] for i in indices]

    logger.debug("=" * 60)
    logger.debug("QUERY: %s", user_query)
    for rank, (score, idx) in enumerate(zip(scores, indices), 1):
        logger.debug(
            "  [%d] score=%.4f | page=%s | chunk: %.200s…",
            rank, score,
            pages_and_chunks[idx].get("page_number", "?"),
            pages_and_chunks[idx].get("sentence_chunk", "")[:200],
        )
    logger.debug("=" * 60)

    prompt = prompt_formatter(query=user_query, context_items=context_items)
    return _call_chat_api(prompt)


def answer_with_rag_cached(
    user_query: str,
    pages_and_chunks: list[dict],
    embeddings: np.ndarray,
) -> str:
    """
    RAG answer that reuses preloaded metadata + embeddings.
    Used by chatPage.py.
    """
    # Basic input validation
    if not user_query or not user_query.strip():
        return "Please enter a question."
    if len(user_query) > MAX_QUERY_LENGTH:
        return (
            f"Your question is too long ({len(user_query)} chars). "
            f"Please keep it under {MAX_QUERY_LENGTH} characters."
        )

    scores, indices = retrieve_relevant_resources(
        query=user_query,
        embeddings=embeddings,
        top_k=3,
    )
    sorted_pairs = sorted(zip(scores.tolist(), indices.tolist()), reverse=True)
    context_items = [pages_and_chunks[i] for _, i in sorted_pairs]

    logger.debug("=" * 60)
    logger.debug("CACHED QUERY: %s", user_query)
    for rank, (score, idx) in enumerate(sorted_pairs, 1):
        logger.debug(
            "  [%d] score=%.4f | page=%s | chunk: %.200s…",
            rank, score,
            pages_and_chunks[idx].get("page_number", "?"),
            pages_and_chunks[idx].get("sentence_chunk", "")[:200],
        )
    logger.debug("=" * 60)

    prompt = prompt_formatter(query=user_query, context_items=context_items)
    return _call_chat_api(prompt)


# -----------------------
# Quiz Generation (RAG-based)
# -----------------------
def _build_quiz_prompt(
    context_items: list[dict],
    num_questions: int,
    difficulty: str,
    topic_hint: str,
) -> str:
    """Build a prompt that asks OpenAI to generate MCQ quiz questions from context."""
    context_parts = []
    for item in context_items:
        source = item.get("source_file", "unknown")
        page = item.get("page_number", "?")
        chunk_text = item["sentence_chunk"]
        context_parts.append(f"[Source: {source}, Page: {page}]\n{chunk_text}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a quiz creator that generates questions EXCLUSIVELY from the provided document passages.

STRICT RULES — VIOLATION OF ANY RULE MAKES THE QUIZ INVALID:
1. Every question MUST be directly answerable from the context passages below.
2. Every correct answer MUST be a fact explicitly stated in the context.
3. Every wrong option MUST be plausible but clearly contradicted by or not present in the context.
4. Every explanation MUST quote or closely paraphrase the context and cite the source file and page number.
5. Do NOT use any outside knowledge, general knowledge, or information not in the context.
6. If you cannot generate {num_questions} questions from the context alone, generate as many as possible.
7. Difficulty level: {difficulty}
   - Easy: straightforward factual recall from the text
   - Medium: requires understanding relationships between facts in the text
   - Hard: requires inference or synthesis across multiple passages in the text
8. Topic focus: {topic_hint}

Return the quiz in this EXACT JSON format (no markdown, no code fences, just raw JSON array):
[
  {{
    "question": "According to the document, what is ...?",
    "options": {{
      "A": "Option from context",
      "B": "Option from context",
      "C": "Option from context",
      "D": "Option from context"
    }},
    "correct": "B",
    "explanation": "The text states that '...' (Source: filename.pdf, Page: 5)"
  }}
]

CONTEXT PASSAGES (this is your ONLY source of information):
{context}
"""
    return prompt


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=lambda rs: logger.warning("Retrying OpenAI quiz call (attempt %d)…", rs.attempt_number),
)
def _call_quiz_api(prompt: str) -> str:
    """Send a quiz generation prompt to OpenAI and return the raw response."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate multiple-choice quiz questions using ONLY the document "
                    "passages provided by the user. You NEVER use outside knowledge. "
                    "Every question, answer option, and explanation must come directly "
                    "from the provided text. If a fact is not in the passages, do not "
                    "reference it. Always respond with valid JSON only — no markdown "
                    "fences, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def generate_quiz(
    topic: str,
    pages_and_chunks: list[dict],
    embeddings: np.ndarray,
    num_questions: int = 5,
    difficulty: str = "Medium",
    context_override: list[dict] | None = None,
) -> list[dict]:
    """
    Generate a quiz from the knowledge base.

    If context_override is provided, use those chunks directly (no retrieval).
    Otherwise, use the topic as a query to retrieve relevant chunks.
    """
    import json as _json

    if not topic or not topic.strip():
        raise ValueError("Please provide a topic for the quiz.")

    if context_override:
        # Use the pre-selected chunks directly — no semantic search needed
        context_items = context_override
    else:
        # Fallback: retrieve chunks via semantic search
        top_k = min(max(num_questions * 2, 5), 15)
        scores, indices = retrieve_relevant_resources(
            query=topic,
            embeddings=embeddings,
            top_k=top_k,
        )
        sorted_pairs = sorted(zip(scores.tolist(), indices.tolist()), reverse=True)
        context_items = [pages_and_chunks[i] for _, i in sorted_pairs]

    logger.info(
        "Quiz generation — topic: %r, difficulty: %s, num_questions: %d, context chunks: %d",
        topic, difficulty, num_questions, len(context_items),
    )

    prompt = _build_quiz_prompt(context_items, num_questions, difficulty, topic)
    raw = _call_quiz_api(prompt)

    # Strip markdown code fences if the model wraps them anyway
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        questions = _json.loads(raw)
    except _json.JSONDecodeError as e:
        logger.error("Failed to parse quiz JSON: %s\nRaw response:\n%s", e, raw[:500])
        raise ValueError(
            "The AI returned an invalid quiz format. Please try again."
        ) from e

    # Basic validation
    validated = []
    for q in questions:
        if all(k in q for k in ("question", "options", "correct", "explanation")):
            if q["correct"] in q["options"]:
                validated.append(q)
    
    if not validated:
        raise ValueError("No valid questions could be generated. Try a different topic.")

    logger.info("Generated %d valid quiz questions.", len(validated))
    return validated


# -----------------------
# Main execution (CLI demo)
# -----------------------
if __name__ == "__main__":
    import fitz
    import random

    from config import PDF_PATH, PAGE_NUMBER_OFFSET

    # Demo questions
    gpt4_questions = [
        "Who are the five Pandava brothers?",
        "Who is the eldest of the Kauravas?",
        "Who delivers the Bhagavad Gita to Arjuna?",
        "Which kingdom do the Pandavas rule after the war?",
        "Who killed Bhishma and how?",
    ]
    manual_questions = [
        "Who is Draupadi married to?",
        "Which warrior killed Karna?",
        "Who is known as the charioteer and guide of Arjuna?",
        "Which son of Arjuna succeeds him as a notable warrior?",
        "Which sage is credited with composing or compiling large portions of the epic in tradition?",
    ]

    # 1) Open PDF document
    document = fitz.open(PDF_PATH)

    # 2) Load metadata + embeddings
    pages_and_chunks, embeddings = load_metadata_and_embeddings(
        CHUNKS_CSV, EMBEDDINGS_PATH
    )

    # 3) Demo retrieval
    query = "Is Krishna talking to Arjuna on the battlefield?"
    print(f"Query: {query}")

    scores, indices = retrieve_relevant_resources(
        query=query, embeddings=embeddings
    )
    print(scores, indices)

    scores, indices = print_top_results_and_scores(
        query=query,
        embeddings=embeddings,
        pages_and_chunks=pages_and_chunks,
    )

    print(f"Highest score: {scores[0]:.4f}")
    best_match_page_number = pages_and_chunks[indices[0]]["page_number"]
    print(f"Best match is on page number: {best_match_page_number}")

    # Save matched page as image (offset accounts for front-matter)
    # The page_number in CSV already has PAGE_NUMBER_OFFSET subtracted,
    # so we add it back to get the real PDF page index.
    real_page_idx = best_match_page_number + PAGE_NUMBER_OFFSET
    if 0 <= real_page_idx < len(document):
        page = document[real_page_idx]
        img = page.get_pixmap(dpi=300)
        img.save("matched_page.png")
        print(f"Saved matched page (PDF index {real_page_idx}) as 'matched_page.png'")
    else:
        print(f"Page index {real_page_idx} is out of range (0–{len(document) - 1}), skipping image save.")

    # 4) RAG-style query
    query_list = gpt4_questions + manual_questions
    query = random.choice(query_list)
    print(f"RAG demo query: {query}")

    answer = answer_with_rag(query)
    print(f"Query: {query}")
    print(f"RAG answer:\n{answer}")

    document.close()