# RAG-Based QA System

## Overview
This is a Retrieval-Augmented Generation (RAG) Question-Answering system designed to query a single PDF book as its source of truth. The system runs locally or on a private server, ensuring compliance with privacy constraints and preventing external model knowledge leakage. It leverages **FAISS** for efficient vector storage, **Sentence Transformers** for text embeddings, and supports multiple language models** (Groq, Ollama, Google Gemini) via **LangChain** for answer generation. The frontend is built with **Streamlit**, providing an interactive interface for PDF uploads and querying.

## Features
- **PDF Processing**: Extracts text and metadata (page numbers) from uploaded PDFs.
- **Chunking Methods**: Supports three chunking strategies: Word-based, Semantic, and Recursive, with configurable chunk sizes.
- **Embedding & Retrieval**: Uses SentenceTransformer (`all-MiniLM-L6-v2`) for embeddings and FAISS for vector search with cosine or L2 distance metrics.
- **Reranking**: Optionally applies a `FlagRer` model to improve chunk relevance.
- **LLM Backends**: Supports Groq** (`llama-3.3-70b-versatile`), **Ollama** (`llama3.1`), and **Google Gemini** (`gemini-2.0-flash`) for answer generation.
- **Interactive UI**: Configurable settings for chunking, similarity metrics, top-k retrieval, and reranking via Streamlit.

## Setup & Installation
1. **Prerequisites**:
   - Python 3.8+
   - Install Ollama and pull the `llama3.1` model: `ollama pull llama3.1` (if using Ollama backend)
   - Ensure the Ollama server is running at `http://192.168.11.204:11434` (if using Ollama)
   - Set environment variables for API keys:
     - `GROQ_API_KEY` for Groq (optional, defaults to provided key)
     - `GOOGLE_API_KEY` for Gemini
   - A PDF book file for querying

2. **Create Virtual Environment**:
   ```bash
   python -m venv my_env
   source my_env/bin/activate  # On Windows: my_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Deployment Steps
1. **Start the Ollama Server** (if using Ollama):
   - Ensure the Ollama server is running at `http://192.168.11.204:11434` with the `llama3.1` model loaded.

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   - Access the UI at `http://localhost:8501`.

3. **Preprocessing**:
   - Upload a PDF via the Streamlit interface.
   - The system extracts text, chunks it based on selected settings, and builds a FAISS index stored in memory (persisted via Streamlit session state).
   - Processing occurs once per PDF upload unless settings change.

## Usage
- Open the Streamlit app in your browser.
- Configure settings in the sidebar:
  - Select LLM backend (`Groq`, `Ollama`, `Gemini`)
  - Choose chunking method (`Word-based`, `Semantic`, `Recursive`)
  - Adjust chunk size (300–1000 words)
  - Select similarity metric (`Cosine`, `L2`)
  - Set top-k chunks (3–10)
  - Enable/disable reranker
- Upload a PDF file.
- Enter a question in the input field.
- View the generated answer, retrieved chunks, and metadata (page numbers) in the UI.

## Limitations
- The system is restricted to the content of the uploaded PDF.
- Answer accuracy depends on PDF text extraction quality (via `PyPDF2`).
- LLM performance varies by backend and hardware (e.g., Ollama requires sufficient GPU/CPU resources).
- Requires a running Ollama server at `http://192.168.11.204:11434` for Ollama backend.
- Gemini and Groq backends require valid API keys.

## Security & Privacy
- All processing (text extraction, embedding, inference) occurs locally or on the private server, except for API calls to Groq or Gemini (if selected).
- No external API calls are made at runtime for Ollama backend beyond the specified server.
- Data is stored in memory via Streamlit session state; no persistent local files are created unless explicitly saved.

## Dependencies
Key libraries (see `requirements.txt` for full list):
- `streamlit`
- `faiss-cpu`
- `sentence-transformers`
- `langchain`, `langchain-ollama`, `langchain-groq`, `langchain-google-genai`
- `PyPDF2`
- `FlagEmbedding`