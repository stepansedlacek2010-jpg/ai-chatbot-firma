# AI Chatbot pro firmy — RAG chatbot nad vlastními dokumenty

Streamlit aplikace, která odpovídá na dotazy výhradně na základě nahraných dokumentů
(PDF, TXT, DOCX, MD). Používá LangChain + FAISS pro indexaci a vyhledávání a jeden
ze 4 podporovaných LLM poskytovatelů pro generování odpovědí:

| Poskytovatel | Tarif | Model(y) |
|---|---|---|
| **Google Gemini** | zdarma | Gemini 3.7 Flash, Gemini 3.1 Pro |
| **Groq** | zdarma | GPT-OSS 120B, GPT-OSS 20B |
| Anthropic Claude | placené | Claude Sonnet 5, Opus 5, Haiku 4.5 |
| OpenAI | placené | GPT-4o, GPT-4o mini |

V sidebaru se zobrazí **jen ti poskytovatelé, pro které je v secrets vyplněný API
klíč** — stačí tedy vyplnit jeden klíč (klidně jen Gemini nebo jen Groq) a aplikace
plně funguje.

## 1. Instalace

Vyžaduje Python 3.10+.

```bash
# 1. Vytvořte a aktivujte virtuální prostředí
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Nainstalujte závislosti
pip install -r requirements.txt
```

> Poznámka: `sentence-transformers` a `faiss-cpu` mají větší instalační balíčky
> (stahují se váhy embedding modelu) — první instalace i první spuštění proto
> mohou trvat několik minut.

## 2. Přidání API klíče

Aplikace potřebuje **alespoň jeden** ze 4 API klíčů. Doporučujeme začít s Gemini
nebo Groq — oba mají štědrý free tarif a klíč získáte za minutu bez zadávání karty.

1. Zkopírujte ukázkový soubor:

   ```bash
   copy .streamlit\secrets.toml.example .streamlit\secrets.toml    # Windows
   # cp .streamlit/secrets.toml.example .streamlit/secrets.toml    # macOS / Linux
   ```

2. Otevřete `.streamlit/secrets.toml` a vyplňte alespoň jeden klíč:

   ```toml
   GEMINI_API_KEY = "AIza..."
   GROQ_API_KEY = "gsk_..."
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   OPENAI_API_KEY = "sk-proj-..."
   ```

   - **Gemini** (zdarma): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
   - **Groq** (zdarma): [console.groq.com/keys](https://console.groq.com/keys)
   - Anthropic: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
   - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

3. **Soubor `.streamlit/secrets.toml` nikdy nenahrávejte do gitu** — obsahuje
   citlivé údaje. Přidejte ho do `.gitignore`:

   ```
   .streamlit/secrets.toml
   vectorstore_data/
   ```

Pokud nechybí ani jeden klíč, aplikace po spuštění zobrazí jasnou chybovou hlášku
s návodem místo pádu s nesrozumitelnou chybou.

## 3. Spuštění lokálně

```bash
streamlit run app.py
```

Aplikace se otevře na `http://localhost:8501`.

**Použití:**

1. V levém panelu vyberte poskytovatele a model (nabízí se jen ti, pro které máte klíč).
2. Nahrajte jeden nebo více dokumentů (PDF/TXT/DOCX/MD).
3. Klikněte na „📥 Zpracovat a přidat do znalostní báze“ — dokumenty se rozdělí
   na úseky, zaindexují a uloží do lokální FAISS databáze (`vectorstore_data/`).
4. Ptejte se v chatu — odpovědi budou vycházet pouze z nahraných dokumentů a
   u každé odpovědi uvidíte zdrojové dokumenty (expander „📄 Zdroje“).
5. V panelu lze kdykoliv upravit systémový prompt, model, teplotu a max. počet
   tokenů, nebo smazat historii chatu / celou znalostní bázi.

## 4. Nasazení na Streamlit Cloud

1. Nahrajte projekt (bez `.streamlit/secrets.toml`!) do GitHub repozitáře.
2. Na [share.streamlit.io](https://share.streamlit.io) klikněte na **New app**
   a vyberte repozitář, branch a `app.py` jako hlavní soubor.
3. Před (nebo po) prvním spuštěním otevřete **App settings → Secrets** a vložte
   stejný obsah jako do lokálního `secrets.toml` (stačí jeden klíč):

   ```toml
   GEMINI_API_KEY = "AIza..."
   GROQ_API_KEY = "gsk_..."
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   OPENAI_API_KEY = "sk-proj-..."
   ```

4. Uložte — aplikace se restartuje a klíče se načtou automaticky přes
   `st.secrets`.

> **Persistence znalostní báze na Streamlit Cloud:** aplikace ukládá FAISS index
> do lokální složky `vectorstore_data/`. Na Streamlit Cloud je souborový systém
> efemérní — po restartu/redeploy kontejneru se index ztratí a dokumenty je nutné
> nahrát znovu. Pro trvalé úložiště v produkci byste index ukládali do externího
> storage (např. S3) mimo rozsah tohoto template.

## Technický stack

| Vrstva | Technologie |
|---|---|
| UI | Streamlit (`st.chat_message`, `st.chat_input`) |
| Orchestrace / RAG | LangChain (`langchain`, `langchain-community`) |
| Vektorová databáze | FAISS (`faiss-cpu`) |
| Embeddingy | `sentence-transformers/all-MiniLM-L6-v2` (lokálně, zdarma, bez API klíče) |
| Načítání souborů | `pypdf` (PDF), `python-docx` (DOCX), přímé čtení (TXT/MD) |
| LLM — Gemini | oficiální `google-generativeai` SDK |
| LLM — Groq | `openai` SDK s `base_url="https://api.groq.com/openai/v1"` (Groq je OpenAI-kompatibilní) |
| LLM — Claude | oficiální `anthropic` SDK |
| LLM — OpenAI | oficiální `openai` SDK |

**Poznámky k implementaci:**

- U modelů **Claude Opus 5 / Sonnet 5** je ve výchozím stavu zapnuté adaptivní
  „thinking“, které není kompatibilní s parametrem `temperature` (API by vracelo
  chybu). Aplikace ho pro tyto dva modely automaticky vypíná, aby posuvník teploty
  fungoval konzistentně napříč všemi poskytovateli.
- Načítání PDF/DOCX/TXT/MD řeší přímo `pypdf` a `python-docx` — je to spolehlivější
  a nevyžaduje žádné systémové závislosti navíc (na rozdíl od balíčku `unstructured`),
  takže aplikace „funguje hned po spuštění“ i na Windows.
- Volba poskytovatele v sidebaru se generuje dynamicky podle toho, které API klíče
  jsou vyplněné v `st.secrets` — žádný poskytovatel bez klíče se v UI nenabízí.

## Výchozí systémový prompt

```
Jsi profesionální AI asistent pro malé firmy. Odpovídáš pouze na základě
poskytnutých dokumentů. Buď stručný, přátelský a profesionální. Pokud
informace nemáš, otevřeně to řekni a nabídni kontakt na člověka.
```

Editovatelné přímo v postranním panelu aplikace.

## Řešení problémů

| Problém | Řešení |
|---|---|
| „Chybí API klíč“ hned po spuštění | Zkontrolujte, že existuje `.streamlit/secrets.toml` s alespoň jedním vyplněným klíčem (viz krok 2). |
| „Neplatný API klíč“ | Klíč je špatně zkopírovaný nebo expirovaný — vygenerujte nový v konzoli daného poskytovatele. |
| „Byl překročen limit požadavků“ u Gemini/Groq | Free tarify mají limity na počet požadavků za minutu — počkejte chvíli, nebo přepněte na jiného poskytovatele v sidebaru. |
| Chyba „model does not exist“ / 404 u Gemini nebo Groq | Oba poskytovatelé pravidelně (řádově co pár měsíců) vyřazují starší modely z free tarifu. Zkontrolujte aktuální seznam na [console.groq.com/docs/models](https://console.groq.com/docs/models) resp. [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) a upravte `model_id` v `PROVIDERS` v `app.py`. |
| „Odpověď byla zablokována bezpečnostními filtry Gemini“ | Gemini má vlastní bezpečnostní filtry na obsah — přeformulujte dotaz nebo přepněte na jiného poskytovatele. |
| Chatbot odpovídá „Nemám tuto informaci v dokumentech“ na vše | Dokumenty nejsou naindexované — klikněte na „📥 Zpracovat a přidat do znalostní báze“, případně zkontrolujte počet indexovaných úseků v sidebaru. |
| Nahrání souboru selže / soubor je příliš velký | Limit je 20 MB na soubor (nastavitelné v `app.py` konstantou `MAX_FILE_SIZE_MB`). |
| Pomalé první spuštění | Embedding model (`sentence-transformers`) se stahuje a inicializuje jen jednou (výsledek se cachuje přes `st.cache_resource`). |
