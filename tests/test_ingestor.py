"""
test_ingestor.py — Unit Tests for ingestor.py
-----------------------------------------------
Tests:
  1. load_and_chunk_pdf returns a non-empty list
  2. Each chunk is a LangChain Document with page_content and metadata
  3. Metadata contains 'source' and 'page' keys
  4. Chunk size does not exceed the configured maximum
  5. Function raises an error on invalid file path
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestor import load_and_chunk_pdf

# Path to a small test PDF we'll create for testing
TEST_PDF_PATH = "tests/fixtures/test_sample.pdf"


@pytest.fixture(scope="module")
def sample_chunks():
    """
    Fixture: load and chunk the test PDF once,
    reuse across all tests in this module.
    """
    return load_and_chunk_pdf(TEST_PDF_PATH)


def test_chunks_not_empty(sample_chunks):
    """Ingestor should return at least one chunk from a valid PDF."""
    assert len(sample_chunks) > 0, "Expected at least one chunk from PDF"


def test_chunks_are_documents(sample_chunks):
    """Every item returned should be a LangChain Document."""
    from langchain_core.documents import Document
    for chunk in sample_chunks:
        assert isinstance(chunk, Document), (
            f"Expected Document, got {type(chunk)}"
        )


def test_chunks_have_page_content(sample_chunks):
    """Every chunk should have non-empty page_content."""
    for chunk in sample_chunks:
        assert chunk.page_content.strip() != "", (
            "Found a chunk with empty page_content"
        )


def test_chunks_have_metadata(sample_chunks):
    """Every chunk should have 'source' and 'page' in metadata."""
    for chunk in sample_chunks:
        assert "source" in chunk.metadata, "Missing 'source' in metadata"
        assert "page" in chunk.metadata, "Missing 'page' in metadata"


def test_chunk_size_within_limit(sample_chunks):
    """No chunk should exceed chunk_size + overlap (1200 + 200 = 1400 chars)."""
    MAX_CHARS = 1400
    for chunk in sample_chunks:
        assert len(chunk.page_content) <= MAX_CHARS, (
            f"Chunk exceeds max size: {len(chunk.page_content)} chars"
        )


def test_invalid_path_raises_error():
    """Passing a non-existent path should raise an exception."""
    with pytest.raises(Exception):
        load_and_chunk_pdf("non_existent_file.pdf")