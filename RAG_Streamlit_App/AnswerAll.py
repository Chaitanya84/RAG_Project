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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    #is_flash_attn_2_available,
)
from transformers.utils import is_flash_attn_2_available 
import os

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

# LLM quantization config (preserved behavior)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# attention implementation selection (preserved)
if (is_flash_attn_2_available()) and (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")

# Question lists preserved for sampling later
gpt4_questions = [
    "Who are the five Pandava brothers?",
    "Who is the eldest of the Kauravas?",
    "Who delivers the Bhagavad Gita to Arjuna?",
    "Which kingdom do the Pandavas rule after the war?",
    "Who killed Bhishma and how?"
]
manual_questions = [
                "Who is Draupadi married to?",
                "Which warrior killed Karna?",
                "Who is known as the charioteer and guide of Arjuna?",
                "Which son of Arjuna succeeds him as a notable warrior?",
                "Which sage is credited with composing or compiling large portions of the epic in tradition?",
]

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
    # Load metadata CSV (contains sentence_chunk and page_number columns etc.)
    df = pd.read_csv(csv_path)
    # Load the tensor safely (map to CPU, then move to device)
    embeddings_cpu = torch.load(tensor_path, map_location="cpu")
    if not isinstance(embeddings_cpu, torch.Tensor):
        raise ValueError(f"Loaded object from {tensor_path} is not a torch.Tensor")

    # Move to target device
    embeddings = embeddings_cpu.to(device)

    # Quick checks: number of rows must match embeddings first dimension
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

# -----------------------
# LLM helpers (keep behaviour consistent)
# -----------------------
def choose_model_by_gpu_memory():
    """Preserve original selection logic for model_id and quantization usage."""
    if not torch.cuda.is_available():
        gpu_memory_gb = 0
    else:
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2 ** 30))

    use_quantization_config = False
    model_id = "google/gemma-2b-it"

    if gpu_memory_gb < 5.1:
        print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        use_quantization_config = True
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb < 8.1:
        print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
        use_quantization_config = True
        model_id = "google/gemma-2b-it"
    elif gpu_memory_gb < 19.0:
        print(f"GPU memory: {gpu_memory_gb} | Recommended: Gemma 2B float16 or Gemma 7B in 4-bit.")
        use_quantization_config = False
        model_id = "google/gemma-2b-it"
    else:
        print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
        use_quantization_config = False
        model_id = "google/gemma-7b-it"

    print(f"use_quantization_config set to: {use_quantization_config}")
    print(f"model_id set to: {model_id}")
    return use_quantization_config, model_id

def load_llm(model_id: str, use_quantization_config: bool, attn_impl: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if use_quantization_config else None,
        low_cpu_mem_usage=False,
        attn_implementation=attn_impl
    )
    if (not use_quantization_config) and torch.cuda.is_available():
        llm_model.to("cuda")
    return tokenizer, llm_model

def prompt_formatter(query: str, context_items: list[dict], tokenizer) -> str:
    """Build concise instruction + context prompt."""
    context = "\n\n".join([item["sentence_chunk"] for item in context_items])
    base_prompt = f"""You are an expert on the Mahabharata and related dharma shastras.

Based ONLY on the context passages below, answer the user's question in 2–4 sentences.
Briefly explain why?

Context:
{context}

User question: {query}

Answer:"""

    dialogue_template = [{"role": "user", "content": base_prompt}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("=== RAG QUERY ===", query)
    print("=== CONTEXT ===", context)
    print("=== PROMPT ===", prompt[:1000])
    return prompt

# -----------------------
# Main execution
# -----------------------
def answer_with_rag(user_query: str) -> str:
    """
    Single-call RAG answer function for Streamlit:
    - Loads metadata + embeddings + embedding model + LLM (once per call).
    - Retrieves top-k chunks.
    - Builds prompt and returns answer text.
    """
    # 1) Load embedding model and metadata+embeddings
    embedding_model = load_embedding_model(device=DEVICE)
    pages_and_chunks, embeddings = load_metadata_and_tensor(
        CHUNKS_CSV, EMBEDDINGS_TENSOR_PATH, device=DEVICE
    )

    # 2) Retrieve context
    scores, indices = retrieve_relevant_resources(
        query=user_query, embeddings=embeddings, model=embedding_model
    )
    context_items = [pages_and_chunks[i.item()] for i in indices]

    # 3) Load LLM (Gemma)
    use_quantization_config, model_id = choose_model_by_gpu_memory()
    tokenizer, llm_model = load_llm(
        model_id=model_id,
        use_quantization_config=use_quantization_config,
        attn_impl=attn_implementation,
    )

    # 4) Build prompt and generate
    prompt = prompt_formatter(
        query=user_query, context_items=context_items, tokenizer=tokenizer
    )

    input_ids = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    with torch.no_grad():
        outputs = llm_model.generate(
            **input_ids,
            temperature=0.3,      # lower temperature
            do_sample=False,      # deterministic
            max_new_tokens=256,
        )
    output_ids = outputs[0][input_ids["input_ids"].shape[-1]:]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True)

    return answer.strip()

def load_rag_resources(device: str = DEVICE):
    """
    Load and return (embedding_model, pages_and_chunks, embeddings).
    Intended to be called once and cached by Streamlit.
    """
    embedding_model = load_embedding_model(device=device)
    pages_and_chunks, embeddings = load_metadata_and_tensor(
        CHUNKS_CSV,
        EMBEDDINGS_TENSOR_PATH,
        device=device,
    )
    return embedding_model, pages_and_chunks, embeddings


def load_llm_resources():
    """
    Load and return (tokenizer, llm_model).
    Intended to be called once and cached by Streamlit.
    """
    use_quantization_config, model_id = choose_model_by_gpu_memory()
    tokenizer, llm_model = load_llm(
        model_id=model_id,
        use_quantization_config=use_quantization_config,
        attn_impl=attn_implementation,
    )
    return tokenizer, llm_model


def answer_with_rag_cached(
    user_query: str,
    embedding_model,
    pages_and_chunks,
    embeddings,
    tokenizer,
    llm_model,
) -> str:
    """
    RAG answer that reuses preloaded models/resources.
    Used by chatPage.py.
    """
    # 1) Retrieve context
    scores, indices = retrieve_relevant_resources(
        query=user_query,
        embeddings=embeddings,
        model=embedding_model,
        top_k=3,
    )
    # sort by score descending explicitly (optional)
    scores, indices = scores.cpu(), indices.cpu()
    sorted_pairs = sorted(zip(scores.tolist(), indices.tolist()), reverse=True)
    context_items = [pages_and_chunks[i] for _, i in sorted_pairs]

    # 2) Build prompt
    prompt = prompt_formatter(
        query=user_query,
        context_items=context_items,
        tokenizer=tokenizer,
    )

    # 3) Generate
    input_ids = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **input_ids,
            temperature=0.3,          # lower temperature for factual Q&A
            do_sample=False,          # disable sampling for more deterministic answers
            max_new_tokens=256,
        )

    # Only decode newly generated tokens
    generated_ids = outputs[0][input_ids["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()

def main():
    # 1) Open PDF document so we can extract matched page image later
    document = fitz.open(PDF_PATH)

    # 2) Load embedding model (to encode queries) and then metadata + embeddings tensor
    embedding_model = load_embedding_model(device=DEVICE)

    pages_and_chunks, embeddings = load_metadata_and_tensor(CHUNKS_CSV, EMBEDDINGS_TENSOR_PATH, device=DEVICE)

    # 3) Demo retrieval (preserve original prints)
    query = "Is Krishna talking to arjuna on the battlefeild?"
    print(f"Query: {query}")

    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, model=embedding_model)
    print(scores, indices)

    scores, indices = print_top_results_and_scores(query=query, embeddings=embeddings, pages_and_chunks=pages_and_chunks, model=embedding_model)

    # Report top score and page like original script
    print(f"Highest score: {scores[0]:.4f}")
    print(f"Corresponding page number: {pages_and_chunks[indices[0].item()]['page_number']}")
    best_match_page_number = pages_and_chunks[indices[0].item()]["page_number"]
    print(f"Best match is on page number: {best_match_page_number}")

    # Save the matched page as image (same filename)
    page = document[best_match_page_number + 41]  # original offset
    img = page.get_pixmap(dpi=300)
    img.save("matched_page.png")
    print("Saved matched page as 'matched_page.png'")

    # 4) LLM selection / load (preserved)
    if torch.cuda.is_available():
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2 ** 30))
    else:
        gpu_memory_gb = 0
    print(f"Available GPU memory: {gpu_memory_gb} GB")

    use_quantization_config, model_id = choose_model_by_gpu_memory()
    print(f"[INFO] Using model_id: {model_id}")

    tokenizer, llm_model = load_llm(model_id=model_id, use_quantization_config=use_quantization_config, attn_impl=attn_implementation)

    # 5) Example prompt/generation (preserved)
    input_text = "Which characters in mahabharata did not fight in the war?"
    print(f"Input text:\n{input_text}")
    dialogue_template = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
    print(f"\nPrompt (formatted):\n{prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    with torch.no_grad():
        outputs = llm_model.generate(**input_ids, max_new_tokens=256)
    print(f"Model output (tokens):\n{outputs[0]}\n")
    outputs_decoded = tokenizer.decode(outputs[0])
    print(f"Model output (decoded):\n{outputs_decoded}\n")

    # 6) RAG-style query: pick random query, retrieve context, build prompt and generate
    query_list = gpt4_questions + manual_questions
    query = random.choice(query_list)
    print(f"Query: {query}")

    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, model=embedding_model)
    context_items = [pages_and_chunks[i.item()] for i in indices]

    prompt = prompt_formatter(query=query, context_items=context_items, tokenizer=tokenizer)
    print(f"My Name Is Chaitanya \n {prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    with torch.no_grad():
        outputs = llm_model.generate(**input_ids, temperature=0.7, do_sample=True, max_new_tokens=256)
    output_text = tokenizer.decode(outputs[0])
    answer = output_text.replace(prompt, "")

    print(f"Query: {query}")
    print(f"RAG answer:\n{answer}")

    # close document
    document.close()

if __name__ == "__main__":
    main()