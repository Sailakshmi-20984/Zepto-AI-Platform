import os
import chromadb
from chromadb.utils import embedding_functions

MODULE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(MODULE_DIR, "docs")
DB_DIR = os.path.join(MODULE_DIR, "chroma_db")

def build_vector_store():
    print("[1/2] Reading document corpus...")
    docs, ids = [], []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCS_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read().strip())
                ids.append(filename)

    print(f"[2/2] Generating embeddings for {len(docs)} documents...")
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="zepto_policies", embedding_function=embed_fn)

    collection.upsert(documents=docs, ids=ids)
    print(f"Vector store created in {DB_DIR} with IDs: {ids}")

if __name__ == "__main__":
    build_vector_store()