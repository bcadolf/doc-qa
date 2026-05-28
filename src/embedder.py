# Embeds chunks and stores in chromaDb

import chromadb
from chromadb.utils import embedding_functions


chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_or_create_collection(name: str = "documents"):
    return chroma_client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn
    )


def store_chunks(chunks: list[str], doc_name: str):
    """Embed and store chunks in ChromaDB"""
    collection = get_or_create_collection()

    
    collection.add(
        ids=[f"{doc_name}_chunk_{i}" for i, _ in enumerate(chunks)],
        documents=chunks,
        metadatas=[{"source": doc_name, "chunk_index": i} for i, _ in enumerate(chunks)]
    )

    print(f"Stored {len(chunks)} chunks from document '{doc_name}'")