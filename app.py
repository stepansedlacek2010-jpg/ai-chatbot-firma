"""
AI Chatbot pro firmy — RAG chatbot nad nahranými dokumenty (Streamlit + LangChain + FAISS).

Odpovědi generuje LLM výhradně na základě obsahu nahraných dokumentů (PDF, TXT, DOCX, MD).
Podporovaní poskytovatelé (zobrazí se jen ti, pro které je v secrets vyplněný API klíč):
  - Google Gemini (free)
  - Groq (free, OpenAI-kompatibilní API)
  - Anthropic Claude
  - OpenAI
"""

import os
import shutil
import sys
from pathlib import Path

# Některá cloudová prostředí (např. Streamlit Community Cloud) nemají nastavenou
# UTF-8 locale, takže výchozí kódování stdout/stderr je ASCII — jakýkoliv pokus
# knihovny (httpx/openai/logging) zapsat nestandardní text pak spadne na
# UnicodeEncodeError. Vynutíme UTF-8 hned na startu, nezávisle na locale hostu.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import anthropic
import google.generativeai as genai
import openai
import streamlit as st
from docx import Document as DocxReader
from google.api_core import exceptions as google_exceptions
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Konfigurace poskytovatelů ---
# Pořadí = pořadí v sidebaru. Free možnosti (Gemini, Groq) jsou první.

PROVIDERS = {
    "gemini": {
        "label": "🟢 Google Gemini (free)",
        "secret_key": "GEMINI_API_KEY",
        "models": {
            # Pozn.: Gemini modely mají pravidelný deprecation cyklus (řádově měsíce) —
            # pokud model přestane fungovat, aktuální seznam je na ai.google.dev/gemini-api/docs/models.
            "Gemini 3.7 Flash (fast, recommended)": "gemini-3.7-flash",
            "Gemini 3.1 Pro (highest quality)": "gemini-3.1-pro-preview",
        },
    },
    "groq": {
        "label": "🟢 Groq (free, very fast)",
        "secret_key": "GROQ_API_KEY",
        "models": {
            # Pozn.: Groq modely mají pravidelný deprecation cyklus (řádově měsíce) —
            # pokud model přestane fungovat, aktuální seznam je na console.groq.com/docs/models.
            "GPT-OSS 120B (recommended)": "openai/gpt-oss-120b",
            "GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
        },
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "secret_key": "ANTHROPIC_API_KEY",
        "models": {
            "Claude Sonnet 5 (recommended)": "claude-sonnet-5",
            "Claude Opus 5 (highest quality)": "claude-opus-5",
            "Claude Haiku 4.5 (fastest)": "claude-haiku-4-5",
        },
    },
    "openai": {
        "label": "OpenAI",
        "secret_key": "OPENAI_API_KEY",
        "models": {
            "GPT-4o": "gpt-4o",
            "GPT-4o mini": "gpt-4o-mini",
        },
    },
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional AI assistant for small businesses. You answer only based on "
    "the documents provided. Be concise, friendly and professional. If you don't have the "
    "information, say so openly and offer to put the user in touch with a member of staff."
)

VECTORSTORE_DIR = Path("vectorstore_data")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Modely s adaptivním "thinking" zapnutým ve výchozím stavu — pro použití
# nastavení teploty je nutné thinking explicitně vypnout.
ANTHROPIC_THINKING_MODELS = {"claude-opus-5", "claude-sonnet-5"}


# --- Načtení API klíčů ---

def get_secret(key: str):
    """Bezpečně načte hodnotu z st.secrets, i pokud soubor secrets.toml neexistuje."""
    try:
        return st.secrets.get(key)
    except Exception:
        return None


API_KEYS = {key: get_secret(cfg["secret_key"]) for key, cfg in PROVIDERS.items()}
AVAILABLE_PROVIDERS = [key for key in PROVIDERS if API_KEYS[key]]


# --- Zpracování dokumentů ---

def load_pdf(file, filename: str) -> list[Document]:
    reader = PdfReader(file)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "page": i}))
    return docs


def load_docx(file, filename: str) -> list[Document]:
    doc = DocxReader(file)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [Document(page_content=text, metadata={"source": filename})] if text.strip() else []


def load_text(file, filename: str) -> list[Document]:
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": filename})] if text.strip() else []


def process_uploaded_files(uploaded_files) -> tuple[list[Document], list[str]]:
    """Načte nahrané soubory do LangChain Document objektů. Vrací (dokumenty, chybové hlášky)."""
    all_docs: list[Document] = []
    errors: list[str] = []

    for f in uploaded_files:
        if f.size > MAX_FILE_SIZE_BYTES:
            errors.append(f'"{f.name}": file is too large (max. {MAX_FILE_SIZE_MB} MB).')
            continue

        ext = Path(f.name).suffix.lower()
        try:
            if ext == ".pdf":
                docs = load_pdf(f, f.name)
            elif ext == ".docx":
                docs = load_docx(f, f.name)
            elif ext in (".txt", ".md"):
                docs = load_text(f, f.name)
            else:
                errors.append(f'"{f.name}": unsupported file format.')
                continue

            if not docs:
                errors.append(f'"{f.name}": couldn\'t extract any text from this document.')
                continue

            all_docs.extend(docs)
        except Exception as e:
            errors.append(f'"{f.name}": error processing file ({e}).')

    return all_docs, errors


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Lokální embedding model (běží bez API klíče, výsledek se cachuje mezi requesty)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vectorstore_from_disk(embeddings: HuggingFaceEmbeddings):
    """Načte dříve uloženou znalostní bázi z disku, pokud existuje."""
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        try:
            return FAISS.load_local(
                str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            return None
    return None


def index_documents(uploaded_files) -> tuple[int, list[str]]:
    """Zpracuje a naindexuje nahrané soubory do znalostní báze (FAISS). Vrací (počet úseků, chyby)."""
    docs, errors = process_uploaded_files(uploaded_files)
    if not docs:
        return 0, errors

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.vectorstore.add_documents(chunks)

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    st.session_state.vectorstore.save_local(str(VECTORSTORE_DIR))

    return len(chunks), errors


def delete_knowledge_base():
    st.session_state.vectorstore = None
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)


# --- Vyhledávání relevantního kontextu ---

def retrieve_context(query: str) -> tuple[str, list[dict]]:
    vectorstore = st.session_state.vectorstore
    if vectorstore is None:
        return "", []

    results = vectorstore.similarity_search(query, k=RETRIEVAL_K)
    context_parts = []
    sources = []
    for doc in results:
        source = doc.metadata.get("source", "unknown document")
        page = doc.metadata.get("page")
        label = f"{source}" + (f" (page {page})" if page else "")
        context_parts.append(f"[{label}]\n{doc.page_content}")
        snippet = doc.page_content[:300].strip()
        sources.append({"label": label, "snippet": snippet + ("…" if len(doc.page_content) > 300 else "")})

    return "\n\n---\n\n".join(context_parts), sources


def build_system_prompt(base_prompt: str, context: str) -> str:
    if context.strip():
        return (
            f"{base_prompt}\n\n"
            "You have the following excerpts from the uploaded documents available. Answer "
            "STRICTLY based on this context. If the answer isn't in the context, say so openly: "
            "\"I don't have this information in the documents\" and offer to put the user in "
            "touch with a member of staff. Don't include any information that isn't supported "
            "by the context. Don't include any internal or system tags in your reply "
            "(e.g. <thinking>).\n\n"
            f"CONTEXT FROM DOCUMENTS:\n{context}"
        )
    return (
        f"{base_prompt}\n\n"
        "There are no documents uploaded to the knowledge base yet. Let the user know and ask "
        "them to upload files via the sidebar first."
    )


# --- Volání LLM ---

def call_claude(model_id: str, system_prompt: str, history: list[dict], temperature: float, max_tokens: int) -> str:
    client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
    kwargs = dict(
        model=model_id,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=history,
        temperature=temperature,
    )
    # Opus 5 / Sonnet 5 mají ve výchozím stavu zapnuté adaptivní "thinking", které
    # není kompatibilní s parametrem temperature — pro tento use-case ho vypínáme.
    if model_id in ANTHROPIC_THINKING_MODELS:
        kwargs["thinking"] = {"type": "disabled"}

    response = client.messages.create(**kwargs)

    if response.stop_reason == "refusal":
        detail = response.stop_details.explanation if response.stop_details else ""
        return f"The response could not be generated for safety reasons. {detail}".strip()

    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts).strip() or "Sorry, I couldn't generate a response."


def call_openai_compatible(
    api_key: str,
    base_url: str | None,
    model_id: str,
    system_prompt: str,
    history: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Společná implementace pro OpenAI a pro Groq (OpenAI-kompatibilní API)."""
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": system_prompt}] + history
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except UnicodeEncodeError as e:
        # API klíč obsahuje neascii znak — typicky proto, že v secrets zůstal
        # placeholder text (např. "your-key") místo skutečné hodnoty klíče.
        raise ValueError(
            "The API key contains invalid characters. Please check that you've entered your "
            "actual API key in secrets, not the example placeholder text."
        ) from e
    return (response.choices[0].message.content or "").strip()


def call_gemini(model_id: str, system_prompt: str, history: list[dict], temperature: float, max_tokens: int) -> str:
    genai.configure(api_key=API_KEYS["gemini"])
    model = genai.GenerativeModel(model_name=model_id, system_instruction=system_prompt)

    # Gemini používá role "user" / "model" (ne "assistant") a obsah v poli "parts".
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
        for m in history
    ]

    response = model.generate_content(
        contents,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    try:
        text = response.text
    except Exception:
        reason = None
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason
        return (
            "The response can't be displayed — it was likely blocked by Gemini's safety filters"
            f"{f' ({reason})' if reason else ''}."
        )

    return text.strip() or "Sorry, I couldn't generate a response."


def generate_answer(
    provider: str, model_id: str, system_prompt: str, history: list[dict], temperature: float, max_tokens: int
) -> tuple[str | None, str | None]:
    """Vrací (odpověď, chybová hláška) — právě jedno z toho je None."""
    try:
        if provider == "anthropic":
            return call_claude(model_id, system_prompt, history, temperature, max_tokens), None
        if provider == "openai":
            return call_openai_compatible(
                API_KEYS["openai"], None, model_id, system_prompt, history, temperature, max_tokens
            ), None
        if provider == "groq":
            return call_openai_compatible(
                API_KEYS["groq"], GROQ_BASE_URL, model_id, system_prompt, history, temperature, max_tokens
            ), None
        if provider == "gemini":
            return call_gemini(model_id, system_prompt, history, temperature, max_tokens), None
        return None, f"Unknown provider: {provider}"

    # --- Anthropic chyby ---
    except anthropic.AuthenticationError:
        return None, "Invalid Anthropic API key. Check the ANTHROPIC_API_KEY value in secrets."
    except anthropic.PermissionDeniedError:
        return None, "Your Anthropic API key doesn't have permission to use this model."
    except anthropic.NotFoundError:
        return None, "The specified Claude model was not found."
    except anthropic.RateLimitError:
        return None, "The Anthropic API rate limit has been exceeded. Please try again shortly."
    except anthropic.APIConnectionError:
        return None, "Couldn't connect to the Anthropic API. Check your internet connection."
    except anthropic.APIStatusError as e:
        return None, f"Anthropic API error ({e.status_code}): {e.message}"

    # --- OpenAI / Groq chyby (Groq používá stejný SDK, tedy i stejné výjimky) ---
    except openai.AuthenticationError:
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        return None, f"Invalid API key. Check the {key_name} value in secrets."
    except openai.RateLimitError:
        return None, "The API rate limit has been exceeded. Please try again shortly."
    except openai.APIConnectionError:
        return None, "Couldn't connect to the API. Check your internet connection."
    except openai.APIStatusError as e:
        return None, f"API error: {e}"
    except ValueError as e:
        return None, str(e)

    # --- Gemini chyby ---
    except google_exceptions.PermissionDenied:
        return None, "Invalid Gemini API key or insufficient permissions. Check the GEMINI_API_KEY value in secrets."
    except google_exceptions.Unauthenticated:
        return None, "Invalid Gemini API key. Check the GEMINI_API_KEY value in secrets."
    except google_exceptions.ResourceExhausted:
        return None, "The Gemini API rate limit has been exceeded (free tier). Please try again shortly."
    except google_exceptions.InvalidArgument as e:
        return None, f"Invalid Gemini API request: {e.message}"
    except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable):
        return None, "Couldn't connect to the Gemini API. Check your internet connection."
    except google_exceptions.GoogleAPICallError as e:
        return None, f"Gemini API error: {e.message}"

    except Exception as e:
        return None, f"Unexpected error while generating the response: {e}"


# --- Streamlit UI ---

st.set_page_config(page_title="AI Chatbot for Business", page_icon="💬", layout="wide")

if not AVAILABLE_PROVIDERS:
    key_list = "\n".join(f'{cfg["secret_key"]} = "..."' for cfg in PROVIDERS.values())
    st.error(
        "**Missing API key.**\n\n"
        "The app needs at least one API key. We recommend starting with Google Gemini or "
        "Groq — both have a free tier. Add your key to the `.streamlit/secrets.toml` file "
        "at the project root:\n\n"
        f"```toml\n{key_list}\n```\n\n"
        "You only need to fill in one line. On Streamlit Cloud, add the key in the app "
        "settings under **Secrets**. See README.md for detailed instructions."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = load_vectorstore_from_disk(get_embeddings())

# --- Sidebar: nastavení ---

with st.sidebar:
    st.header("⚙️ Settings")

    provider_labels = {key: PROVIDERS[key]["label"] for key in AVAILABLE_PROVIDERS}
    provider_choice_label = st.selectbox("Provider", list(provider_labels.values()))
    provider = next(key for key, label in provider_labels.items() if label == provider_choice_label)

    model_options = PROVIDERS[provider]["models"]
    model_label = st.selectbox("Model", list(model_options.keys()))
    model_id = model_options[model_label]

    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max response tokens", 256, 4096, 1024, 128)

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=170)

    st.divider()
    st.subheader("📚 Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX, MD)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 Process and add to knowledge base", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Processing and indexing documents..."):
            n_chunks, errors = index_documents(uploaded_files)
        for err in errors:
            st.warning(err)
        if n_chunks:
            st.success(f"Added {n_chunks} text chunks from {len(uploaded_files)} file(s).")

    n_indexed = st.session_state.vectorstore.index.ntotal if st.session_state.vectorstore else 0
    st.caption(f"Currently indexed: **{n_indexed}** text chunks")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear KB", use_container_width=True):
            delete_knowledge_base()
            st.rerun()

# --- Hlavní chat ---

st.title("💬 AI Chatbot for Business")
st.caption("Answers are based solely on the uploaded documents.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['label']}**")
                    st.caption(s["snippet"])

user_input = st.chat_input("Ask anything about the uploaded documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context, sources = retrieve_context(user_input)
            full_system_prompt = build_system_prompt(system_prompt, context)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            answer, error = generate_answer(
                provider, model_id, full_system_prompt, history, temperature, max_tokens
            )

        if error:
            st.error(error)
        else:
            st.markdown(answer)
            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.markdown(f"**{s['label']}**")
                        st.caption(s["snippet"])
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
