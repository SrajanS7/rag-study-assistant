"""
chain.py — RAG Chain with Ollama LLaMA3
-----------------------------------------
Responsible for:
  1. Building a prompt template that instructs LLaMA3 to answer
     using only the retrieved context chunks
  2. Wiring retriever → prompt → LLaMA3 into a LangChain RAG chain
  3. Returning the answer along with source page citations

Pipeline:
  question → retriever → relevant chunks → prompt template
  → Ollama LLaMA3 → answer + citations
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# --- Prompt Template ---
# This is the instruction we give LLaMA3 for every question.
# Key design decisions:
#   - "ONLY use the context below" → prevents hallucination
#   - "If you don't know, say so" → honest about missing info
#   - "cite the page number" → gives users source references
PROMPT_TEMPLATE = """
You are a helpful study assistant. Use ONLY the context below to answer 
the question. If the answer is not in the context, say 
"I don't have enough information to answer that from the provided document."

Always cite the page number(s) where you found the information, 
like this: (Source: filename.pdf, Page X)

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved Document chunks into a single context string
    for the prompt, including source metadata for citations.

    Args:
        docs: List of Document objects from the retriever

    Returns:
        str: Formatted context string with page references
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        # Add page number as a hint so LLaMA3 can cite it
        formatted.append(
            f"[Source: {source}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def build_rag_chain(retriever):
    """
    Build and return the full RAG chain.

    Chain flow:
      {"context": retriever | format_docs, "question": passthrough}
      → prompt template
      → Ollama LLaMA3
      → string output parser

    Args:
        retriever: LangChain retriever from retriever.py

    Returns:
        Runnable: A LangChain chain that accepts a question string
                  and returns an answer string with citations.
    """

    # --- LLM: Ollama running LLaMA3 locally ---
    # base_url: where Ollama is serving (default port 11434)
    # temperature=0.1: low randomness → more factual, consistent answers
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.1
    )

    # --- Prompt ---
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # --- RAG Chain ---
    # RunnablePassthrough passes the question through unchanged
    # retriever fetches relevant chunks, format_docs formats them
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[chain] RAG chain built successfully.")
    return rag_chain


def ask(rag_chain, question: str) -> dict:
    """
    Ask a question and return the answer with source chunks.

    Args:
        rag_chain: The built RAG chain from build_rag_chain()
        question (str): The user's question

    Returns:
        dict: {
            "answer": str,        # LLaMA3's answer with citations
            "question": str       # The original question
        }
    """

    print(f"[chain] Processing question: '{question}'")
    answer = rag_chain.invoke(question)

    return {
        "question": question,
        "answer": answer
    }