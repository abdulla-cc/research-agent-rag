"""Unit tests for the chunking and junk-filtering logic.
These test pure functions — no network, no LLM, no Gemini quota needed."""

import sys
import os

# make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunk_papers import (
    fixed_size_chunks,
    semantic_chunks,
    looks_like_junk,
    extract_title_from_header,
)


# ---------- fixed_size_chunks ----------

def test_fixed_size_chunks_splits_long_text():
    text = " ".join(["word"] * 1200)  # 1200 words
    chunks = fixed_size_chunks(text, chunk_words=500, overlap_words=50)
    # 1200 words at 500/chunk with overlap should give ~3 chunks
    assert len(chunks) >= 2
    # each chunk (except maybe last) should be around 500 words
    assert len(chunks[0].split()) == 500


def test_fixed_size_chunks_short_text_single_chunk():
    text = "just a few words here"
    chunks = fixed_size_chunks(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_fixed_size_chunks_overlap():
    text = " ".join(str(i) for i in range(1000))
    chunks = fixed_size_chunks(text, chunk_words=500, overlap_words=50)
    # the last 50 words of chunk 0 should reappear at the start of chunk 1
    tail = chunks[0].split()[-50:]
    head = chunks[1].split()[:50]
    assert tail == head


# ---------- semantic_chunks ----------

def test_semantic_chunks_keeps_sentences_whole():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = semantic_chunks(text, max_words=100)
    # short text fits in one chunk, and it should contain full sentences
    assert len(chunks) == 1
    assert "sentence one" in chunks[0]
    assert "sentence three" in chunks[0]


def test_semantic_chunks_handles_pdf_style_linebreaks():
    # PDF text has newlines mid-sentence — the chunker must not choke on them
    text = "This is a\nsentence split across\nlines. And another\none here."
    chunks = semantic_chunks(text, max_words=100)
    assert len(chunks) >= 1
    # newlines should be collapsed, not preserved as chunk boundaries
    assert "\n" not in chunks[0]


def test_semantic_chunks_empty_input():
    chunks = semantic_chunks("", max_words=100)
    # empty text should produce no chunks (or one empty), not crash
    assert chunks == [] or chunks == [""]


# ---------- looks_like_junk (the important one) ----------

def test_junk_filter_catches_reference_list():
    # a dense reference list: many citation markers per word
    ref = ("[12] Smith et al. Proceedings 2021. [13] Jones et al. "
           "Journal 2020. [14] Lee et al. arXiv:2101.00001 2019. "
           "[15] Kim et al. pages 10-20, 2022.")
    action, reason = looks_like_junk(ref)
    assert action == "drop", f"should drop reference list, got {action}"


def test_junk_filter_keeps_content_with_few_citations():
    # a real content paragraph that happens to cite a source or two
    # should be KEPT (this is the false-positive guard we tuned for)
    content = (
        "Retrieval-augmented generation improves factual accuracy by "
        "grounding responses in retrieved documents [1]. This approach "
        "has been shown to reduce hallucination substantially. The method "
        "works by first retrieving relevant passages, then conditioning "
        "the language model's generation on those passages. Unlike "
        "fine-tuning, it requires no model retraining and adapts to new "
        "information simply by updating the document store."
    )
    action, reason = looks_like_junk(content)
    assert action == "keep", f"should keep real content, got {action} ({reason})"


def test_junk_filter_catches_header_with_emails():
    header = ("Deep Learning for Science John Smith1 Jane Doe2 "
              "1MIT 2Stanford john@mit.edu jane@stanford.edu")
    action, reason = looks_like_junk(header)
    assert action == "trim_header", f"should trim header, got {action}"


def test_extract_title_keeps_beginning():
    header = "A Great Paper Title Author One author@uni.edu Author Two"
    title = extract_title_from_header(header)
    # should keep the title part, cut at the email
    assert "A Great Paper Title" in title
    assert "author@uni.edu" not in title
