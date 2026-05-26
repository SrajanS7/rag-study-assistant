"""
ui.py — Streamlit Chat Interface
----------------------------------
Responsible for:
  1. PDF upload via sidebar — triggers ingestion + embedding pipeline
  2. Chat interface — user asks questions, gets cited answers
  3. Session state management — conversation history persists during session
  4. Source citations — displayed below each answer with page numbers

Run with:
  streamlit run app/ui.py
"""

import streamlit as st
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestor import load_and_chunk_pdf
from app.embedder import get_embedding_model
from app.retriever import build_vector_store, get_retriever
from app.chain import build_rag_chain, ask

# --- Page Config ---
st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide"
)


# --- Session State Initialisation ---
# Streamlit reruns the entire script on every interaction,
# so we use session_state to persist objects across reruns.
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {"role", "content", "sources"}
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None    # built after PDF upload
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "embedding_model" not in st.session_state:
        # Load once and reuse — expensive to reload on every rerun
        st.session_state.embedding_model = None


init_session_state()


# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.title("📚 RAG Study Assistant")
    st.markdown("Upload a PDF and chat with it using a local LLM.")
    st.caption("💡 This assistant answers from your document only — not a general chatbot.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Lecture notes, research papers, textbooks — any PDF works."
    )

    if uploaded_file is not None:
        # Only re-process if a new PDF is uploaded
        if uploaded_file.name != st.session_state.pdf_name:

            with st.spinner("Processing PDF... this may take a moment."):

                # Save uploaded file to data/uploads/
                upload_dir = "data/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load embedding model once
                if st.session_state.embedding_model is None:
                    st.session_state.embedding_model = get_embedding_model()

                # Run ingestion pipeline
                chunks = load_and_chunk_pdf(file_path)
                vector_store = build_vector_store(
                    chunks,
                    st.session_state.embedding_model
                )
                retriever = get_retriever(vector_store, top_k=4)
                rag_chain = build_rag_chain(retriever)

                # Store in session
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []  # reset chat on new PDF

            st.success(f"✅ Ready! '{uploaded_file.name}' loaded.")

    if st.session_state.pdf_loaded:
        st.info(f"📄 Active: {st.session_state.pdf_name}")
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption("Running locally · No API costs · Powered by LLaMA3")


# --- Main Area: Chat Interface ---
st.title("💬 Chat with your PDF")

if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources below assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 View Sources"):
                for i, source in enumerate(message["sources"]):
                    page = source.metadata.get("page", 0)
                    filename = source.metadata.get("source", "unknown")
                    st.markdown(
                        f"**Chunk {i+1}** · "
                        f"{filename} · "
                        f"Page {page + 1}"  # +1 for human-readable page numbers
                    )
                    st.caption(source.page_content[:300] + "...")
                    st.divider()

# Chat input
if question := st.chat_input("Ask a question about your PDF..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "sources": []
    })

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Get answer from RAG chain
            result = ask(st.session_state.rag_chain, question)

            # Also fetch source chunks for the expander
            source_docs = st.session_state.retriever.invoke(question)

            st.markdown(result["answer"])

            # Show sources in expander
            if source_docs:
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(source_docs):
                        page = source.metadata.get("page", 0)
                        filename = source.metadata.get("source", "unknown")
                        st.markdown(
                            f"**Chunk {i+1}** · "
                            f"{filename} · "
                            f"Page {page + 1}"
                        )
                        st.caption(source.page_content[:300] + "...")
                        st.divider()

    # Save assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": source_docs
    })