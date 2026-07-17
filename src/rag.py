import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()  # reads .env, makes GEMINI_API_KEY available

# anchor paths to the project root (parent of this src/ folder) so
# the code works no matter which directory you launch it from
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
DB_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma")
COLLECTION_NAME = "papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.0-flash"
TOP_K = 5


class LLMUnavailableError(Exception):
    """Raised when the LLM provider is temporarily unavailable (e.g. 503)."""
    pass


class QuotaExceededError(Exception):
    """Raised when the API quota is exhausted (429 RESOURCE_EXHAUSTED)."""
    pass

# load these ONCE at import, not per-request (expensive to load)
_embedder = SentenceTransformer(EMBED_MODEL)
_chroma = chromadb.PersistentClient(path=DB_DIR)
_collection = _chroma.get_collection(COLLECTION_NAME)
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def retrieve(query, k=TOP_K):
    """Embed the query and return the k most relevant chunks + metadata."""
    query_embedding = _embedder.encode([query]).tolist()
    results = _collection.query(query_embeddings=query_embedding, n_results=k)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "distance": results["distances"][0][i],
        })
    return chunks


def build_prompt(query, chunks):
    """Assemble the retrieved chunks into a grounded prompt."""
    context_blocks = []
    for i, c in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1}] (from: {c['title']})\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the sources below. If the sources don't contain enough information to answer, say so honestly rather than guessing.

Cite your sources inline using [Source N] notation.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm(prompt):
    """Call Gemini with automatic retries on transient failures.
    Retries up to 3 times with exponential backoff (2s, 4s, ...)."""
    response = _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


def answer_question(query, k=TOP_K):
    """Full RAG pipeline: retrieve -> build prompt -> call LLM -> return answer.
    This is the single entry point FastAPI will call next week."""
    chunks = retrieve(query, k)
    prompt = build_prompt(query, chunks)

    try:
        answer = _call_llm(prompt)  # retries transient failures automatically
    except Exception as e:
        msg = str(e)
        # 429 = quota exhausted (needs to wait for reset), distinct from
        # a transient 503 (busy, retry shortly)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise QuotaExceededError(msg)
        raise LLMUnavailableError(msg)

    return {
        "query": query,
        "answer": answer,
        "sources": [{"title": c["title"], "distance": c["distance"]} for c in chunks],
    }


# lets you test it from the terminal: python src/rag.py your question
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the main challenges in retrieval augmented generation?"
    result = answer_question(q)
    print(f"\nQ: {result['query']}\n")
    print(result["answer"])
    print("\n--- Sources used ---")
    for s in result["sources"]:
        print(f"  ({s['distance']:.3f}) {s['title'][:70]}")
