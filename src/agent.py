"""Agentic RAG pipeline: query decomposition + multi-retrieval + critique.

Wraps the simple pipeline in rag.py with two extra LLM-powered steps:
  1. DECOMPOSE  — break a complex question into focused sub-questions
  2. RETRIEVE   — gather chunks for each sub-question
  3. GENERATE   — write a cited answer from the combined evidence
  4. CRITIQUE   — check the answer against the sources for unsupported claims

Reuses retrieve(), build_prompt(), and _call_llm() from rag.py unchanged.
"""

import json
import re

from rag import retrieve, _call_llm, LLMUnavailableError, QuotaExceededError, TOP_K


# ---------- stage 1: decompose ----------

def decompose_question(question):
    """Ask the LLM to break a question into 1-3 focused sub-questions.
    Falls back to [question] if the question is simple or parsing fails."""
    prompt = f"""Break the following question into 1 to 3 focused sub-questions that,
answered together, would fully address it. If the question is already simple and
atomic, return it unchanged as a single item.

Return ONLY a JSON array of strings, nothing else. Example: ["sub q 1", "sub q 2"]

QUESTION: {question}

JSON:"""
    try:
        raw = _call_llm(prompt)
        # pull the JSON array out of the response (LLMs sometimes add prose)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return [question]
        subs = json.loads(match.group(0))
        # sanity: must be a non-empty list of strings
        subs = [s for s in subs if isinstance(s, str) and s.strip()]
        return subs if subs else [question]
    except (LLMUnavailableError, QuotaExceededError):
        raise  # let quota/availability errors bubble up
    except Exception:
        # any parsing weirdness -> just use the original question
        return [question]


# ---------- stage 2: retrieve for each sub-question ----------

def multi_retrieve(sub_questions, k=TOP_K):
    """Retrieve chunks for each sub-question, dedup by chunk text."""
    seen = set()
    combined = []
    for sq in sub_questions:
        for chunk in retrieve(sq, k=k):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                combined.append(chunk)
    return combined


# ---------- stage 3: generate the answer ----------

def generate_answer(question, chunks):
    """Write a cited answer grounded in the combined chunks."""
    context_blocks = []
    for i, c in enumerate(chunks):
        context_blocks.append(f"[Source {i+1}] (from: {c['title']})\n{c['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the sources
below. If the sources don't contain enough information, say so honestly.
Cite sources inline using [Source N].

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""
    return _call_llm(prompt)


# ---------- stage 4: critique ----------

def critique_answer(answer, chunks):
    """Second LLM pass: check the answer's claims against the sources.
    Returns a dict with 'supported' (bool) and 'notes' (str)."""
    context_blocks = []
    for i, c in enumerate(chunks):
        context_blocks.append(f"[Source {i+1}]\n{c['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a fact-checker. Below is an ANSWER and the SOURCES it was
based on. Check whether every claim in the answer is actually supported by the
sources. 

Return ONLY a JSON object like:
{{"supported": true/false, "notes": "brief explanation of any unsupported claims, or 'All claims supported.'"}}

SOURCES:
{context}

ANSWER:
{answer}

JSON:"""
    try:
        raw = _call_llm(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"supported": None, "notes": "Could not parse critique."}
        result = json.loads(match.group(0))
        return {
            "supported": result.get("supported"),
            "notes": result.get("notes", ""),
        }
    except (LLMUnavailableError, QuotaExceededError):
        raise
    except Exception:
        return {"supported": None, "notes": "Critique unavailable."}


# ---------- the full agentic pipeline ----------

def answer_question_agentic(question, k=TOP_K):
    """Full agentic flow: decompose -> multi-retrieve -> generate -> critique."""
    sub_questions = decompose_question(question)
    chunks = multi_retrieve(sub_questions, k=k)
    answer = generate_answer(question, chunks)
    critique = critique_answer(answer, chunks)

    return {
        "query": question,
        "sub_questions": sub_questions,
        "answer": answer,
        "verification": critique,
        "sources": [
            {"title": c["title"], "distance": c["distance"]} for c in chunks
        ],
    }


# quick terminal test: python src/agent.py your question here
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Compare fixed and semantic chunking for retrieval quality."
    result = answer_question_agentic(q)
    print(f"\nQ: {result['query']}\n")
    print("Sub-questions:")
    for sq in result["sub_questions"]:
        print(f"  - {sq}")
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Verification: supported={result['verification']['supported']}")
    print(f"  {result['verification']['notes']}")
    print(f"\nSources: {len(result['sources'])}")
