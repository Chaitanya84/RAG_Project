# RAG Streamlit App

A Retrieval-Augmented Generation (RAG) application built with Streamlit, OpenAI APIs, and Google OAuth.

## Project Structure

```
RAG_Project/
├── requirements.txt          # Python dependencies
├── README.md
├── RAG_Streamlit_App/
│   ├── .env                  # Environment variables (not committed)
│   ├── .env.example          # Template for .env (safe to commit)
│   ├── config.py             # Centralized configuration
│   ├── app.py                # Streamlit entry point (Google OAuth login)
│   ├── AnswerAll.py          # RAG retrieval & answering logic
│   ├── embedandchunk.py      # PDF → sentence chunks → embeddings pipeline
│   ├── client_secret.json    # Google OAuth client secret (not committed)
│   ├── data/                 # Generated CSVs, embeddings, and PDFs (not committed)
│   └── pages/
│       ├── mainPage.py       # PDF upload & processing page
│       └── chatPage.py       # Chat interface for RAG Q&A
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# .\venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

**Option A – Set in `.bashrc` (recommended, keeps secrets out of files):**

```bash
echo 'export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"' >> ~/.bashrc
source ~/.bashrc
```

Then copy the template to create your `.env` (non-secret settings only):

```bash
cd RAG_Streamlit_App
cp .env.example .env
```

**Option B – Set directly in `.env`:**

```bash
cd RAG_Streamlit_App
cp .env.example .env
# Edit .env and uncomment/set OPENAI_API_KEY
```

Required variables in `RAG_Streamlit_App/.env`:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key | *(required)* |
| `OPENAI_EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Chat completion model | `gpt-4o-mini` |
| `DEBUG` | Enable debug logging | `true` |
| `OAUTHLIB_INSECURE_TRANSPORT` | Allow HTTP OAuth (local dev only) | `1` |

### 4. Run the app

```bash
cd RAG_Streamlit_App
streamlit run app.py
```

### 5. (Optional) Download spaCy model

If not auto-installed via `requirements.txt`:

```bash
python -m spacy download en_core_web_sm
```

> **Note:** The app will auto-download the spaCy model if it's missing on first PDF processing.

## Usage

1. **Login** – Authenticate with Google on the Welcome page.
2. **Upload** – Go to **MainPage** and upload a PDF. The app will chunk it, embed it via OpenAI, and save the artifacts.
3. **Chat** – Go to **ChatPage** and ask questions. The app retrieves the most relevant chunks and generates answers using OpenAI.

> **Note:** The `data/` directory is gitignored. If you clone fresh, you'll need to upload and process a PDF before chatting.

## Deactivating the Virtual Environment

```bash
deactivate
```