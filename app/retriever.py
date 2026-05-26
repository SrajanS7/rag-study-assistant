"""
retriever.py — ChromaDB Vector Store
--------------------------------------
Responsible for:
  1. Taking chunks from ingestor.py and storing them as embeddings in ChromaDB
  2. Persisting the vector store to disk (chroma_db/) so it survives restarts
  3. Loading an existing vector store from disk
  4. Retrieving the top-k most relevant chunks for a given query

ChromaDB stores vectors locally — no external services, no API keys.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Directory where ChromaDB persists its data
CHROMA_DB_DIR = "chroma_db"

# Collection name inside ChromaDB (like a table name)
COLLECTION_NAME = "study_assistant"


def build_vector_store(chunks: list,
                       embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed chunks and store them in ChromaDB (persisted to disk).
    Clears any existing collection first to avoid duplicate chunks
    when the same PDF is re-ingested.

    Args:
        chunks (list): List of Document objects from ingestor.py
        embedding_model: HuggingFace embedding model from embedder.py

    Returns:
        Chroma: The populated vector store instance
    """

    print(f"[retriever] Building vector store with {len(chunks)} chunks...")

    # Delete existing collection first to prevent duplicate chunks
    # on re-ingestion of the same document
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[retriever] Cleared existing collection '{COLLECTION_NAME}'")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"[retriever] Vector store built and persisted to '{CHROMA_DB_DIR}'")
    return vector_store


def load_vector_store(embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        embedding_model: Must be the same model used when building the store.

    Returns:
        Chroma: The loaded vector store instance ready for querying.
    """

    print(f"[retriever] Loading vector store from '{CHROMA_DB_DIR}'...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    print("[retriever] Vector store loaded successfully.")
    return vector_store


def get_retriever(vector_store: Chroma, top_k: int = 4):
    """
    Return a LangChain retriever that fetches the top-k relevant chunks.

    Args:
        vector_store (Chroma): The populated ChromaDB instance
        top_k (int): Number of chunks to retrieve per query (default: 4)

    Returns:
        VectorStoreRetriever: LangChain-compatible retriever object
    """

    # search_type="similarity" uses cosine similarity to rank chunks
    # search_kwargs={"k": top_k} returns the top_k most similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    print(f"[retriever] Retriever ready (top_k={top_k})")
    return retriever