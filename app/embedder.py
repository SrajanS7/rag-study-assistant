"""
embedder.py — HuggingFace Embeddings
--------------------------------------
Responsible for:
  1. Loading the sentence-transformers model (all-MiniLM-L6-v2)
  2. Wrapping it in a LangChain-compatible embedding object
  3. Returning it for use in the vector store

Why all-MiniLM-L6-v2?
  - Small (90MB), fast, and runs entirely on CPU
  - Strong performance on semantic similarity tasks
  - No API key needed — downloads once, cached locally
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: LangChain-compatible embedding model
                               ready to embed text chunks into vectors.
    """

    print("[embedder] Loading HuggingFace embedding model "
          "(all-MiniLM-L6-v2)...")

    # model_name: the sentence-transformers model to use
    # model_kwargs: device="cpu" ensures it runs locally without GPU
    # encode_kwargs: normalize_embeddings=True improves cosine similarity
    #                accuracy when comparing vectors
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("[embedder] Model loaded successfully.")
    return embedding_model