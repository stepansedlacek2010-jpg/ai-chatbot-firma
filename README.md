# AI Chatbot for Business — RAG chatbot over your own documents

A Streamlit app that answers questions based solely on uploaded documents
(PDF, TXT, DOCX, MD). Uses LangChain + FAISS for indexing and retrieval, and
one of 4 supported LLM providers for generating answers:

| Provider | Tier | Model(s) |
|---|---|---|
| **Google Gemini** | free | Gemini 3.7 Flash, Gemini 3.1 Pro |
| **Groq** | free | GPT-OSS 120B, GPT-OSS 20B |
| Anthropic Claude | paid | Claude Sonnet 5, Opus 5, Haiku 4.5 |
| OpenAI | paid | GPT-4o, GPT-4o mini |

The sidebar only shows **providers for which an API key is set in secrets**
— you only need to fill in one key (even just Gemini or just Groq) and the
app works fully.

## 1. Installation

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

> Note: `sentence-transformers` and `faiss-cpu` have fairly large install
> footprints (the embedding model weights get downloaded) — the first
> install and first run may take a few minutes.

## 2. Adding an API key

The app needs **at least one** of the 4 API keys. We recommend starting with
Gemini or Groq — both have a generous free tier and you can get a key in a
minute, no card required.

1. Copy the example file:

   ```bash
   copy .streamlit\secrets.toml.example .streamlit\secrets.toml    # Windows
   # cp .streamlit/secrets.toml.example .streamlit/secrets.toml    # macOS / Linux
   ```

2. Open `.streamlit/secrets.toml` and fill in at least one key:

   ```toml
   GEMINI_API_KEY = "AIza..."
   GROQ_API_KEY = "gsk_..."
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   OPENAI_API_KEY = "sk-proj-..."
   ```

   - **Gemini** (free): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
   - **Groq** (free): [console.groq.com/keys](https://console.groq.com/keys)
   - Anthropic: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
   - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

3. **Never commit `.streamlit/secrets.toml` to git** — it contains sensitive
   data. Add it to `.gitignore`:

   ```
   .streamlit/secrets.toml
   vectorstore_data/
   ```

If no key is set, the app shows a clear error message with instructions on
startup instead of crashing with a confusing error.

## 3. Running locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Usage:**

1. In the left panel, pick a provider and model (only ones you have a key for are shown).
2. Upload one or more documents (PDF/TXT/DOCX/MD).
3. Click "📥 Process and add to knowledge base" — the documents get split
   into chunks, indexed, and saved to a local FAISS database (`vectorstore_data/`).
4. Ask questions in the chat — answers will be based only on the uploaded
   documents, and each answer shows its source documents (the "📄 Sources" expander).
5. The sidebar lets you edit the system prompt, model, temperature and max
   tokens at any time, or clear the chat history / entire knowledge base.

## 4. Deploying to Streamlit Cloud

1. Push the project (without `.streamlit/secrets.toml`!) to a GitHub repository.
2. On [share.streamlit.io](https://share.streamlit.io), click **New app**
   and select the repository, branch, and `app.py` as the main file.
3. Before (or after) the first run, open **App settings → Secrets** and paste
   the same content as your local `secrets.toml` (one key is enough):

   ```toml
   GEMINI_API_KEY = "AIza..."
   GROQ_API_KEY = "gsk_..."
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   OPENAI_API_KEY = "sk-proj-..."
   ```

4. Save — the app restarts and the keys are loaded automatically via
   `st.secrets`.

> **Knowledge base persistence on Streamlit Cloud:** the app saves the FAISS
> index to a local `vectorstore_data/` folder. On Streamlit Cloud the
> filesystem is ephemeral — after a restart/redeploy the index is lost and
> documents need to be re-uploaded. For persistent storage in production,
> you'd save the index to external storage (e.g. S3), which is outside the
> scope of this template.

## Technical stack

| Layer | Technology |
|---|---|
| UI | Streamlit (`st.chat_message`, `st.chat_input`) |
| Orchestration / RAG | LangChain (`langchain`, `langchain-community`) |
| Vector database | FAISS (`faiss-cpu`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, no API key needed) |
| File loading | `pypdf` (PDF), `python-docx` (DOCX), plain read (TXT/MD) |
| LLM — Gemini | official `google-generativeai` SDK |
| LLM — Groq | `openai` SDK with `base_url="https://api.groq.com/openai/v1"` (Groq is OpenAI-compatible) |
| LLM — Claude | official `anthropic` SDK |
| LLM — OpenAI | official `openai` SDK |

**Implementation notes:**

- **Claude Opus 5 / Sonnet 5** have adaptive "thinking" enabled by default,
  which isn't compatible with the `temperature` parameter (the API would
  return an error). The app automatically disables it for these two models
  so the temperature slider works consistently across all providers.
- PDF/DOCX/TXT/MD loading is handled directly by `pypdf` and `python-docx`
  — this is more reliable and needs no extra system dependencies (unlike the
  `unstructured` package), so the app "works right out of the box" even on Windows.
- The provider choice in the sidebar is generated dynamically based on which
  API keys are set in `st.secrets` — no provider without a key is ever offered in the UI.

## Default system prompt

```
You are a professional AI assistant for small businesses. You answer only
based on the documents provided. Be concise, friendly and professional. If
you don't have the information, say so openly and offer to put the user in
touch with a member of staff.
```

Editable directly in the app's sidebar.

## Troubleshooting

| Problem | Solution |
|---|---|
| "Missing API key" right on startup | Check that `.streamlit/secrets.toml` exists with at least one key filled in (see step 2). |
| "Invalid API key" | The key was copied incorrectly or has expired — generate a new one in the provider's console. |
| "Rate limit exceeded" on Gemini/Groq | Free tiers have request-per-minute limits — wait a moment, or switch to a different provider in the sidebar. |
| "Model does not exist" / 404 error on Gemini or Groq | Both providers periodically (every few months) retire older models from the free tier. Check the current list at [console.groq.com/docs/models](https://console.groq.com/docs/models) or [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) and update `model_id` in `PROVIDERS` in `app.py`. |
| "Response was blocked by Gemini's safety filters" | Gemini has its own content safety filters — rephrase the question or switch to a different provider. |
| Chatbot answers "I don't have this information in the documents" to everything | Documents aren't indexed yet — click "📥 Process and add to knowledge base", or check the indexed chunk count in the sidebar. |
| File upload fails / file too large | The limit is 20 MB per file (configurable via the `MAX_FILE_SIZE_MB` constant in `app.py`). |
| Slow first run | The embedding model (`sentence-transformers`) downloads and initialises only once (the result is cached via `st.cache_resource`). |
