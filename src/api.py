import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag import answer_question, LLMUnavailableError
from logging_config import logger

app = FastAPI(
    title="Research Agent RAG API",
    description="Ask questions about a corpus of arXiv papers on RAG.",
    version="0.1.0",
)


# --- simple in-memory rate limiter (sliding window) ---
# for each client IP, we keep a queue of recent request timestamps.
# a request is allowed only if fewer than LIMIT requests happened in
# the last WINDOW seconds.
RATE_LIMIT = 10          # max requests
RATE_WINDOW = 60         # per this many seconds
_request_log = defaultdict(deque)  # ip -> deque of timestamps


def check_rate_limit(client_ip: str):
    """Return True if allowed, False if the client is over the limit."""
    now = time.time()
    q = _request_log[client_ip]

    # drop timestamps older than the window
    while q and q[0] <= now - RATE_WINDOW:
        q.popleft()

    if len(q) >= RATE_LIMIT:
        return False

    q.append(now)
    return True


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          description="The question to ask the research assistant.")


class Source(BaseModel):
    title: str
    distance: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    """Simple liveness check — real services always have one of these."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: Request, body: AskRequest):
    """Take a question, run the full RAG pipeline, return a cited answer.
    Rate limited to RATE_LIMIT requests per RATE_WINDOW seconds per IP."""
    client_ip = request.client.host

    if not check_rate_limit(client_ip):
        logger.warning(f"RATE LIMITED: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT} requests per {RATE_WINDOW}s.",
        )

    start = time.time()
    logger.info(f"ASK received from {client_ip}: {body.question!r}")
    try:
        result = answer_question(body.question)
        elapsed = time.time() - start
        logger.info(f"ASK ok in {elapsed:.2f}s: {body.question!r}")
        return result
    except LLMUnavailableError as e:
        elapsed = time.time() - start
        logger.warning(f"ASK llm-unavailable in {elapsed:.2f}s: {e}")
        raise HTTPException(
            status_code=503,
            detail="The language model is temporarily busy. Please try again in a moment.",
        )
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"ASK failed in {elapsed:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
