"""
ingestor.py — PDF Loading & Chunking
-------------------------------------
Responsible for:
  1. Loading a PDF from disk using LangChain's PyPDFLoader
  2. Splitting pages into overlapping chunks using RecursiveCharacterTextSplitter
  3. Returning a list of Document objects with metadata (source, page number)

This is the entry point of the RAG pipeline — everything downstream
(embedder, retriever, chain) consumes the output of this module.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(file_path: str) -> list:
    """
    Load a PDF and split it into chunks.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        list[Document]: List of LangChain Document objects, each containing:
                        - page_content: the chunk text
                        - metadata: {"source": filename, "page": page_number}
    """

    # --- Step 1: Load the PDF ---
    # PyPDFLoader reads the PDF page by page and returns one Document per page.
    # Each Document has page_content (text) and metadata (source path, page number).
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    print(f"[ingestor] Loaded {len(pages)} pages from '{file_path}'")

    # --- Step 2: Split pages into chunks ---
    # RecursiveCharacterTextSplitter splits text by trying "\n\n", "\n", " ", ""
    # in order — preserving paragraph/sentence structure as much as possible.
    #
    # chunk_size=500   → each chunk is at most 500 characters
    # chunk_overlap=50 → 50 characters overlap between consecutive chunks
    #                    (prevents answers from being cut at chunk boundaries)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    print(f"[ingestor] Split into {len(chunks)} chunks "
          f"(chunk_size=500, overlap=50)")

    # --- Step 3: Tag each chunk with clean metadata ---
    # We extract just the filename (not the full path) for cleaner citations
    # in the UI later (e.g. "Source: lecture_notes.pdf, Page 3")
    import os
    filename = os.path.basename(file_path)

    for chunk in chunks:
        chunk.metadata["source"] = filename

    return chunks