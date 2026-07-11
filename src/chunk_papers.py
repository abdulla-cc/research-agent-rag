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
    # collapse all whitespace/newlines first — PDF text has line breaks
    # in the middle of sentences, not just at paragraph ends, so we
    # can't trust blank-line detection here
    clean = re.sub(r"\s+", " ", text).strip()

    # split into sentences (naive but good enough): break after
    # ., !, or ? followed by a space and a capital letter
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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))

    for filename in files:
        path = os.path.join(RAW_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            paper = json.load(f)

        text = paper.get("full_text")
        if not text:
            continue

        fixed = fixed_size_chunks(text)
        semantic = semantic_chunks(text)

        out = {
            "id": paper["id"],
            "title": paper["title"],
            "fixed_chunks": fixed,
            "semantic_chunks": semantic,
        }

        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    print(f"Chunked {len(files)} papers, saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
