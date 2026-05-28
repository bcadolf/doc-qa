# Queries ChromaDb for relavent chunks

from src.embedder import get_or_create_collection

def retrieve(query: str, n_results: int = 4) -> list[str]:
    """
    Embed the query and return the most relevant chunks from the db.
    n_results: number of chunks to return
    """
    collection = get_or_create_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results["documents"][0]