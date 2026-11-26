# 🛠️ DevResolve – Experience-Driven Technical Knowledge Platform

**Share the error. Explain the fix. Help the next developer.**

A Streamlit community platform where developers document real troubleshooting experiences
(error → root cause → fix) with structured metadata, search them semantically, and ask an
AI assistant that answers **only from retrieved community experiences and cites them** (RAG).

Stack: Python · Streamlit · SQLite + SQLAlchemy · Sentence Transformers · FAISS · Ollama (Qwen2.5 0.5B)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch the app creates `data/devresolve.db`, loads **47 sample experiences** (flagged 🧪,
not real community posts) and builds the FAISS index in `vector_store/`. Register an account in the
sidebar to post experiences and solutions; browsing and searching need no login.

### Enable AI answers (optional)
Retrieval, search and source display work without an LLM. For generated answers:
```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:0.5b
```
The sidebar shows 🟢/🟠 for Ollama status. Without it, the AI Troubleshooter returns the most
relevant community experiences with a notice that generation is unavailable.

## Features (PRD mapping)
| Page | What it does |
|---|---|
| Home / Community Feed | Cards with tech, version, OS, category, solution count, accepted badge; filters + text search; full detail view with community solutions and **Accept** (author only) |
| Share Experience | Validated structured form → SQLite → knowledge record → embedding → FAISS. If embedding fails the record is kept and shown as *pending* (sidebar **Retry indexing**) |
| Knowledge Search | Semantic (FAISS) or keyword search, filtered by technology / OS / category, with relevance scores |
| AI Troubleshooter | Chat UI; retrieve → build context → Ollama → **citation verification** → sources shown. Weak evidence ⇒ the insufficient-evidence message (the LLM is not called) |
| My Contributions | Your experiences and stats |

### RAG safeguards
* Citations use `[#ID]`; any ID that wasn't actually retrieved is stripped and reported.
* Retrieved content is treated as **untrusted data**: injection phrases are filtered, delimiter tags can't be forged, text is length-capped.
* Community solutions are labelled *unverified* unless the author accepted one; the prompt forbids calling them verified otherwise.
* The FAISS index is rebuilt from SQLite automatically if missing, corrupt, from a different embedder, or out of sync.
* Submissions containing high-confidence secrets (API keys, GitHub/AWS/Slack tokens, private keys) are rejected; weaker hints (`password=...`) warn.
* Passwords are bcrypt-hashed; user text is control-char-stripped on input and rendered with escaping/`st.text`/`st.code`.

## Embeddings: important note
`DEVRESOLVE_EMBEDDER=auto` (default) uses **Sentence Transformers `all-MiniLM-L6-v2`** when installed and
downloadable, otherwise falls back to an **offline hashing embedder** (term-based, no downloads).

* The hashing fallback matches on shared vocabulary – it is **not truly semantic**. In testing, queries sharing
  words with a record ranked it first; heavily reworded queries often scored below the evidence threshold.
* For real semantic search use the default `sentence-transformers` path (first run downloads the model).
* The score thresholds in `config.MIN_SCORE` (`hashing: 0.15`, `sbert: 0.35`) decide "insufficient evidence".
  The hashing value was calibrated on this dataset; **the sbert value is an untuned starting point** –
  run `python evaluation/evaluate.py` with sbert and adjust it.

## Configuration (environment variables)
`DEVRESOLVE_DB`, `DEVRESOLVE_VECTOR_DIR`, `DEVRESOLVE_EMBEDDER` (auto|sbert|hashing), `DEVRESOLVE_SBERT_MODEL`,
`OLLAMA_URL` (default http://localhost:11434), `OLLAMA_MODEL` (default qwen2.5:0.5b), `OLLAMA_TIMEOUT`.

## Tests
```bash
pytest -q          # 33 tests: validation, persistence, auth, search, index recovery, RAG safeguards
```
Tests use temporary databases/indexes, the offline embedder and a fake LLM, so they need no network or Ollama.

## Evaluation
```bash
python evaluation/evaluate.py
```
Computes **Recall@5**, **MRR** and the insufficient-evidence rate over `evaluation/questions.json`
(20 queries with expected records + 4 queries with no relevant record; expected records are matched by title).

Result with the offline hashing embedder on the 47 seed records:

| Metric | Value |
|---|---|
| Recall@5 (20 queries) | 1.00 |
| MRR | 1.00 |
| No-match queries correctly rejected (4) | 4/4 |

**Read this carefully:** the questions were written by the same author as the seed records, so they share vocabulary
and this result is optimistic. A separate set of 6 heavily reworded queries gave Recall@5 = 5/6, but their top
scores fell *below* the 0.15 threshold, meaning the AI page would have answered "insufficient evidence" for them.
That is the lexical limitation described above. **Faithfulness and citation correctness are not measured
automatically** – inspect AI answers manually (cited IDs are highlighted in the UI).

## Project structure
```
app.py                 entry point: navigation + sidebar (login, index status)
pages/                 1_Home · 2_Share_Experience · 3_Knowledge_Search · 4_AI_Troubleshooter · 5_My_Contributions
database.py  models.py SQLite access (returns plain dicts) and SQLAlchemy models
rag_engine.py          embedders, FAISS index, retrieval, prompt building, Ollama, citation checks
validators.py auth.py  validation/sanitising/secret detection; bcrypt auth
seed_data.py           47 sample experiences (10 Python, 8 Java, 7 Git, 7 React/JS/Node, 5 Docker, 5 SQL, 5 VS Code)
ui_helpers.py config.py
evaluation/            questions.json, evaluate.py
tests/                 test_database.py, test_search.py, test_rag.py
```

## Known limitations
* Local prototype: SQLite, no rate limiting, no moderation. Do not expose publicly as a multi-user service.
* Qwen2.5 0.5B is small; answers can be weak. Use a larger Ollama model via `OLLAMA_MODEL` for better quality.
* A retrieved solution is community-reported, not technically verified. Never run suggested commands blindly.
* Sentence-Transformers and live Ollama generation were not exercised in the build environment (no model downloads there); the
  code paths are implemented and the LLM layer is covered by tests with a fake generator.
