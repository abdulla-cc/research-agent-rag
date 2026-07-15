from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import answer_question, LLMUnavailableError

app = FastAPI(
    title="Research Agent RAG API",
    description="Ask questions about a corpus of arXiv papers on RAG.",
    version="0.1.0",
)


# --- request/response shapes (pydantic validates these automatically) ---

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


# --- endpoints ---

@app.get("/health")
def health():
    """Simple liveness check — real services always have one of these."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Take a question, run the full RAG pipeline, return a cited answer."""
    try:
        result = answer_question(request.question)
        return result
    except LLMUnavailableError:
        # 503 = "service temporarily unavailable" — the correct HTTP code
        # for "the upstream LLM is busy, try again shortly"
        raise HTTPException(
            status_code=503,
            detail="The language model is temporarily busy. Please try again in a moment.",
        )
    except Exception as e:
        # any other unexpected failure -> clean 500 instead of a raw crash
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
