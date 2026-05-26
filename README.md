# 📚 RAG Study Assistant

> Upload any PDF and chat with it using a fully local LLM — no API costs, no data leaves your machine.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-orange)
![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-purple)
![CI](https://github.com/SrajanS7/rag-study-assistant/actions/workflows/ci.yml/badge.svg)

---

## 🧠 What It Does

RAG Study Assistant lets you upload lecture notes, research papers, or any PDF and ask questions about it in natural language. The app retrieves the most relevant sections and uses a local LLaMA3 model to generate cited answers — completely offline.

**Example:**
> *"What certifications does the candidate have?"*
> → *"According to the document, the candidate holds AWS Certified AI Practitioner and AWS Solutions Architect – Associate, both valid through 2028. (Source: resume.pdf, Page 1)"*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User (Browser)                      │
└─────────────────────┬───────────────────────────────────┘
                      │ Streamlit UI (ui.py)
                      │
          ┌───────────▼───────────┐
          │   PDF Upload sidebar  │
          │   Chat interface      │
          └───────────┬───────────┘
                      │
        ┌─────────────▼──────────────┐
        │        RAG Pipeline        │
        │                            │
        │  ingestor.py               │
        │  └─ PyPDFLoader            │
        │  └─ RecursiveTextSplitter  │
        │                            │
        │  embedder.py               │
        │  └─ all-MiniLM-L6-v2      │
        │     (HuggingFace, local)   │
        │                            │
        │  retriever.py              │
        │  └─ ChromaDB (on disk)     │
        │  └─ Cosine similarity      │
        │                            │
        │  chain.py                  │
        │  └─ LangChain LCEL chain   │
        │  └─ Ollama LLaMA3 (local)  │
        └────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | LangChain 0.2.16 | PDF loading, chunking, RAG chain |
| Vector Store | ChromaDB 0.5.5 | Local embedding storage + retrieval |
| Embeddings | all-MiniLM-L6-v2 | Semantic chunk embeddings (free, local) |
| LLM | LLaMA3 8B via Ollama | Answer generation (free, local) |
| UI | Streamlit 1.38.0 | Chat interface |
| CI | GitHub Actions | Lint + test on every push |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed

### 1. Clone the repo
```bash
git clone https://github.com/SrajanS7/rag-study-assistant.git
cd rag-study-assistant
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull LLaMA3
```bash
ollama pull llama3
brew services start ollama   # Mac
```

### 5. Run the app
```bash
streamlit run app/ui.py
```

Open [http://localhost:8501](http://localhost:8501), upload a PDF, and start chatting.

---

## 📁 Project Structure

```
rag-study-assistant/
├── app/
│   ├── ingestor.py      # PDF loading + chunking
│   ├── embedder.py      # HuggingFace embeddings
│   ├── retriever.py     # ChromaDB vector store
│   ├── chain.py         # LangChain RAG chain + Ollama
│   └── ui.py            # Streamlit chat interface
├── data/uploads/        # User-uploaded PDFs (gitignored)
├── chroma_db/           # Persisted vector DB (gitignored)
├── tests/
│   ├── test_ingestor.py
│   └── test_retriever.py
├── .github/workflows/
│   └── ci.yml           # GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

All 10 tests pass locally:
```
tests/test_ingestor.py::test_chunks_not_empty              PASSED
tests/test_ingestor.py::test_chunks_are_documents          PASSED
tests/test_ingestor.py::test_chunks_have_page_content      PASSED
tests/test_ingestor.py::test_chunks_have_metadata          PASSED
tests/test_ingestor.py::test_chunk_size_within_limit       PASSED
tests/test_ingestor.py::test_invalid_path_raises_error     PASSED
tests/test_retriever.py::test_vector_store_builds          PASSED
tests/test_retriever.py::test_retriever_returns_correct_k  PASSED
tests/test_retriever.py::test_retrieved_docs_have_content  PASSED
tests/test_retriever.py::test_retrieved_docs_have_metadata PASSED
```

---

## ⚠️ Known Limitations

- **Table extraction:** Complex multi-column tables in PDFs may yield incomplete results due to PyPDF's text parsing limitations.
- **Page indexing:** PyPDF uses 0-based page indexing; displayed page numbers are adjusted by +1 for readability.
- **LLaMA3 response time:** First response may take 10–20 seconds depending on hardware.
- **Ollama required:** The app requires Ollama running locally — it is not deployable to cloud without modification.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 👤 Author

**Srajan Sharma**  
M.Sc. Computer Science — Paderborn University  
[LinkedIn](https://www.linkedin.com/in/srajan-sharma-671626222/) · [GitHub](https://github.com/SrajanS7)
