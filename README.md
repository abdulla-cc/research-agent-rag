# Research Agent — RAG over arXiv papers

A retrieval-augmented generation (RAG) system that answers questions about a corpus of arXiv papers on RAG itself, returning cited answers grounded in the source papers. Built end-to-end: data pipeline, retrieval, LLM generation, an *agentic* layer, a production API, a web frontend, unit tests, and cloud deployment.

**Live demo:** https://research-agent-rag-ko8d.onrender.com

## What it does

Ask a natural-language question → the system embeds it, retrieves the most relevant chunks from a vector database of ~900 paper chunks, and uses an LLM to generate an answer grounded *only* in those chunks, with inline `[Source N]` citations.

It offers two pipelines:
- **`/ask`** — fast single-shot RAG (retrieve once, answer)
- **`/ask-agent`** — agentic RAG that decomposes the question, retrieves per sub-question, generates an answer, and runs a critic pass to check the answer against its sources

## How the core pipeline works

1. **Ingest** — pulls papers from the arXiv API and extracts full text from the PDFs
2. **Chunk** — splits papers into semantic chunks (sentence-boundary aware), then filters out low-value content: reference lists, mangled tables/figures, and title-page headers
3. **Embed & store** — embeds chunks with `all-MiniLM-L6-v2` and stores them in a Chroma vector database
4. **Retrieve** — embeds the question and finds the nearest chunks by cosine distance
5. **Generate** — passes retrieved chunks to Google Gemini with a grounding prompt that forbids answering beyond the sources

## Agentic pipeline (`/ask-agent`)

Wraps the core pipeline with two extra LLM-powered stages, addressing two real RAG weaknesses:

- **Query decomposition** — a complex question ("compare X and Y and say which is better for Z") is broken into focused sub-questions, each retrieved separately for better coverage than a single blurry query
- **Critic / verifier** — after the answer is drafted, a second LLM pass checks each claim against the retrieved sources and reports whether the answer is fully supported, flagging any unsupported claims

Both stages degrade gracefully: if the LLM returns malformed output, the pipeline falls back to sensible defaults rather than failing.

## Retrieval-quality engineering

Naive RAG retrieved a lot of junk (author names, emails, bibliography entries) that matched topically but carried no explanatory value. I built a conservative junk filter distinguishing three failure modes (reference lists via citation *density*, tables/figures via low alphabetic ratio, title pages via email detection), tuned to avoid dropping real content. Measured result: on a benchmark query, the top-match distance dropped from 0.71 to 0.44 after filtering.

## Production features

- **Input validation** (pydantic) — malformed requests rejected with 422 before hitting the pipeline
- **Structured logging** — every request logged with timing, to console and a rotating file
- **Retry with exponential backoff** — transient LLM failures retried automatically (tenacity)
- **Rate limiting** — a custom per-IP sliding-window limiter caps requests to control cost/abuse
- **Graceful error handling** — distinguishes quota exhaustion (429) from transient unavailability (503)
- **Unit tests** — 16 tests covering chunking, junk-filtering, and the agent's parsing/fallback logic (run with `pytest`)

## Stack

FastAPI · Chroma · sentence-transformers · Google Gemini · Docker · Render

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/ask` | POST | Single-shot RAG: ask a question, get a cited answer |
| `/ask-agent` | POST | Agentic RAG: decomposition + multi-retrieval + critique |
| `/health` | GET | Liveness check |
| `/docs` | GET | Interactive API documentation |

## Running locally

```bash
pip install -r requirements.txt
# set GEMINI_API_KEY in a .env file
cd src
uvicorn api:app --reload
```

Then open http://localhost:8000

## Running the tests

```bash
python -m pytest tests/ -v
```

## Running with Docker

```bash
docker build -t research-agent-rag .
docker run --rm -p 8000:7860 --env-file .env research-agent-rag
```
