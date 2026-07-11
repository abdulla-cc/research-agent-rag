import json
import os
import re

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

def fixed_size_chunks(text, chunk_words=500, overlap_words=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_words - overlap_words
    return chunks

def semantic_chunks(text, max_words=500):
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?]) +(?=[A-Z])", clean)
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip()
        if len(candidate.split()) <= max_words:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def looks_like_junk(chunk):
    """Conservative junk detector. Returns (action, reason) where
    action is one of: 'keep', 'drop', or 'trim_header'.
    - 'drop': reference lists and mangled tables/figures
    - 'trim_header': title-page chunks (keep title, cut author/email soup)
    - 'keep': everything else"""

    words = chunk.split()
    n_words = len(words) if words else 1

    # --- Signal: title-page / header (email density) ---
    # body text essentially never contains email addresses; title
    # pages contain several. this is a very clean separator.
    emails = len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", chunk))
    if emails >= 2:
        return "trim_header", f"title-page ({emails} emails)"

    # --- Signal 1: reference-list density ---
    citation_brackets = len(re.findall(r"\[\d+\]", chunk))
    year_refs = len(re.findall(r",\s*(19|20)\d{2}[.\s]", chunk))
    ref_keywords = len(re.findall(r"\b(et al\.|pages|arXiv:|Proceedings|Journal|Conference)\b", chunk))
    ref_score = citation_brackets + year_refs + ref_keywords
    ref_density = ref_score / n_words

    if ref_score >= 8 and ref_density >= 0.04:
        return "drop", f"reference-list (score={ref_score}, density={ref_density:.3f})"

    # --- Signal 2: low alphabetic ratio (table/figure soup) ---
    letters = sum(c.isalpha() for c in chunk)
    total = len(chunk)
    alpha_ratio = letters / total if total else 0

    if alpha_ratio < 0.55:
        return "drop", f"table/figure soup (alpha_ratio={alpha_ratio:.2f})"

    return "keep", ""


def extract_title_from_header(chunk):
    """Given a title-page chunk, keep just the leading title text
    (up to the first email/affiliation clutter). Best-effort: we
    take everything before the first email address, capped to a
    reasonable length so we don't keep the whole author block."""
    # cut at the first email address
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", chunk)
    head = chunk[:m.start()] if m else chunk
    # also cap to first ~40 words as a safety net
    head = " ".join(head.split()[:40]).strip()
    return head

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))

    total_kept = 0
    total_dropped = 0
    total_trimmed = 0

    for filename in files:
        path = os.path.join(RAW_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        text = paper.get("full_text")
        if not text:
            continue

        fixed = fixed_size_chunks(text)
        raw_semantic = semantic_chunks(text)

        # apply the conservative junk filter to semantic chunks
        semantic = []
        for c in raw_semantic:
            action, reason = looks_like_junk(c)
            if action == "drop":
                total_dropped += 1
            elif action == "trim_header":
                title_text = extract_title_from_header(c)
                if title_text:
                    semantic.append(title_text)
                    total_kept += 1
                total_trimmed += 1
            else:
                semantic.append(c)
                total_kept += 1

        out = {
            "id": paper["id"],
            "title": paper["title"],
            "fixed_chunks": fixed,
            "semantic_chunks": semantic,
        }

        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(f"Chunked {len(files)} papers.")
    print(f"Semantic chunks kept: {total_kept}, dropped: {total_dropped}, headers trimmed: {total_trimmed}")

if __name__ == "__main__":
    main()
