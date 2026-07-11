import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "data/processed"
DB_DIR = "data/chroma"
COLLECTION_NAME = "papers"
CHUNK_TYPE = "semantic_chunks"  # using semantic chunks (cleaner, from Week 2)

def main():
    # 1. load the embedding model (downloads ~90MB the first time, then cached)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. set up a persistent Chroma database on disk
    client = chromadb.PersistentClient(path=DB_DIR)

    # start fresh each run so re-running doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # 3. gather all chunks from all papers
    documents = []
    metadatas = []
    ids = []

    files = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".json"))
    for filename in files:
        with open(os.path.join(PROCESSED_DIR, filename), "r", encoding="utf-8") as f:
            paper = json.load(f)

        chunks = paper.get(CHUNK_TYPE, [])
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "paper_id": paper["id"],
                "title": paper["title"],
                "chunk_index": i,
            })
            ids.append(f"{filename}_{i}")

    print(f"Total chunks to embed: {len(documents)}")

    # 4. embed all chunks (this is the heavy compute step)
    print("Embedding chunks... (this may take a minute)")
    embeddings = model.encode(documents, show_progress_bar=True)

    # 5. store everything in Chroma
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Done. Stored {collection.count()} chunks in {DB_DIR}/")

if __name__ == "__main__":
    main()
