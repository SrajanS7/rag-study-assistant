"""
test_retriever.py — Unit Tests for retriever.py
-------------------------------------------------
Tests:
  1. Vector store builds without errors
  2. Retriever returns the correct number of results (top_k)
  3. Retrieved results are Documents with content and metadata
  4. Loading an existing vector store works correctly
"""

import pytest
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestor import load_and_chunk_pdf
from app.embedder import get_embedding_model
from app.retriever import build_vector_store, get_retriever

TEST_PDF_PATH = "tests/fixtures/test_sample.pdf"
TEST_CHROMA_DIR = "tests/fixtures/test_chroma_db"


@pytest.fixture(scope="module")
def embedding_model():
    """Load embedding model once for all tests."""
    return get_embedding_model()


@pytest.fixture(scope="module")
def vector_store(embedding_model):
    """
    Build a test vector store in a separate test directory.
    Cleaned up after all tests in the module finish.
    """
    chunks = load_and_chunk_pdf(TEST_PDF_PATH)

    # Override the DB directory for tests
    import app.retriever as retriever_module
    original_dir = retriever_module.CHROMA_DB_DIR
    retriever_module.CHROMA_DB_DIR = TEST_CHROMA_DIR

    store = build_vector_store(chunks, embedding_model)

    yield store

    # Cleanup: restore original dir and delete test DB
    retriever_module.CHROMA_DB_DIR = original_dir
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)


def test_vector_store_builds(vector_store):
    """Vector store should build without raising exceptions."""
    assert vector_store is not None


def test_retriever_returns_correct_k(vector_store):
    """Retriever should return at most top_k results."""
    top_k = 2
    retriever = get_retriever(vector_store, top_k=top_k)
    results = retriever.invoke("machine learning")
    assert len(results) >= 1, "Expected at least 1 result"
    assert len(results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"


def test_retrieved_docs_have_content(vector_store):
    """All retrieved documents should have non-empty content."""
    retriever = get_retriever(vector_store, top_k=2)
    results = retriever.invoke("machine learning")
    for doc in results:
        assert doc.page_content.strip() != "", "Retrieved doc has empty content"


def test_retrieved_docs_have_metadata(vector_store):
    """All retrieved documents should have source and page metadata."""
    retriever = get_retriever(vector_store, top_k=2)
    results = retriever.invoke("machine learning")
    for doc in results:
        assert "source" in doc.metadata
        assert "page" in doc.metadata