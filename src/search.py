import sys
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma"
COLLECTION_NAME = "papers"

def main():
    query = " ".join(sys.argv[1:])
    if not query:
        print("Usage: python src/search.py your question here")
        return

    # load the SAME model used for embedding — this is essential
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    # embed the query the same way the chunks were embedded
    query_embedding = model.encode([query]).tolist()

    # find the 3 closest chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    print(f"\nQuery: {query}\n")
    print("=" * 60)
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        print(f"\n[Result {i+1}] distance={distance:.3f}")
        print(f"From: {meta['title'][:70]}")
        print(f"Chunk: {doc[:350]}...")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
