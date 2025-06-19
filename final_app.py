import json
import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
BOOK_PATH = "applsci-10-03214.pdf"  # Replace with your book file
CHUNK_SIZE = 600  # Approximate words per chunk (for word-based chunking)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "thenlper/gte-large"
OLLAMA_MODEL = "llama3.1"  # Ensure this model is available on the Ollama server, using meta llama-3.1 model
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "book_chunks.json"
# OLLAMA_BASE_URL = "http://192.168.11.204:11434"

# Initialize Ollama LLM
# llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="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")


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            metadata = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
                metadata.append({"page": page_num + 1})
            return text, metadata
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "", []

def clean_text(text):
    """Clean and normalize text by removing headers, footers, and noise."""
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip() and not line.isspace()]
    return " ".join(cleaned_lines)

# def chunk_text_semantic(text, metadata, model_name=EMBEDDING_MODEL):
#     """Split text into semantic chunks using SentenceTransformer embeddings."""
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)
#     text_splitter = SemanticChunker(embeddings)
    
#     chunks = text_splitter.split_text(text)
    
#     chunk_metadata = []
#     page_index = 0
#     words_processed = 0
#     words_per_page = len(text.split()) // len(metadata) if metadata else 1

#     for chunk in chunks:
#         words_in_chunk = len(chunk.split())
#         words_processed += words_in_chunk
#         while words_processed > words_per_page * (page_index + 1) and page_index < len(metadata) - 1:
#             page_index += 1
#         chunk_metadata.append({"page": metadata[page_index]["page"] if metadata else 1})
    
#     return chunks, chunk_metadata

def chunk_text_word_based(text, metadata, chunk_size=600):
    """Split text into chunks based on word count with metadata."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_metadata = []
    page_index = 0

    for word in words:
        current_chunk.append(word)
        current_word_count += 1
        if current_word_count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            chunk_metadata.append({"page": metadata[page_index]["page"]})
            current_chunk = []
            current_word_count = 0
            page_index = min(page_index + 1, len(metadata) - 1)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        chunk_metadata.append({"page": metadata[page_index]["page"]})
    
    return chunks, chunk_metadata

def save_chunks(chunks, metadata, output_path):
    """Save chunks and metadata to a JSON file."""
    data = [{"text": chunk, "metadata": meta} for chunk, meta in zip(chunks, metadata)]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def load_chunks(input_path):
    """Load chunks and metadata from a JSON file."""
    with open(input_path, "r") as f:
        data = json.load(f)
    return [item["text"] for item in data], [item["metadata"] for item in data]

def create_vector_store(chunks, model_name=EMBEDDING_MODEL, index_path=INDEX_PATH, use_cosine=True):
    """Generate embeddings and store in FAISS index, supporting L2 or cosine similarity."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity if requested
    if use_cosine:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    dimension = embeddings.shape[1]
    # Use IndexFlatIP for cosine similarity, IndexFlatL2 for Euclidean distance
    index = faiss.IndexFlatIP(dimension) if use_cosine else faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    return model, embeddings

def load_vector_store(index_path=INDEX_PATH):
    """Load FAISS index from file."""
    return faiss.read_index(index_path)

def retrieve_chunks_l2(query, model, index, chunks, k=6):
    """Retrieve top-k relevant chunks using L2 distance (original method)."""
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k)
    return [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

def retrieve_chunks_cosine(query, model, index, chunks, k=4):
    """Retrieve top-k relevant chunks using cosine similarity."""
    query_embedding = model.encode([query])[0]
    # Normalize query embedding for cosine similarity
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k)
    # Convert inner product (FAISS IP) to cosine similarity (1 - cosine distance)
    similarities = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return similarities

def generate_answer(query, chunks, metadata):
    """Generate answer using ChatOllama with retrieved chunks."""
    context = "\n".join([chunk for chunk, _ in chunks])
    prompt_template = ChatPromptTemplate.from_template("""
Answer the following question using only the provided text. If the answer is not found, respond with: "Answer not found in the book."
Context: 
{context}
Question: 
{query}
""")
    try:
        chain = prompt_template | llm
        response = chain.invoke({"context": context, "query": query})
        return response.content, context, [meta for _, meta in chunks]
    except Exception as e:
        return f"Error generating answer: {e}", context, []

def main():
    """Main function to set up and run the Streamlit app with a polished UI."""
    # Set page config
    st.set_page_config(page_title="RAG Q&A Assistant", page_icon="📘", layout="wide")

    # Add custom styling
    st.markdown("""
        <style>
            body {
                background-color: #f5f7fa;
            }
            .main {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .title {
                font-size: 2.5rem;
                color: #003566;
                font-weight: bold;
            }
            .subtitle {
                font-size: 1.1rem;
                color: #555;
                margin-bottom: 1.5rem;
            }
            .question-box input {
                font-size: 1.1rem;
                padding: 10px;
            }
            .chunk-area textarea {
                font-family: 'Courier New', Courier, monospace;
                background-color: #f1f3f5;
                color: #333;
                border-radius: 6px;
                border: 1px solid #ccc;
            }
        </style>
    """, unsafe_allow_html=True)

    # st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">📘 RAG-Based QA System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask questions based on the content of the uploaded book PDF.</div>', unsafe_allow_html=True)

    # Check if preprocessed data exists
    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(INDEX_PATH):
        with st.spinner("⏳ Preprocessing book and creating vector store..."):
            text, metadata = extract_text_from_pdf(BOOK_PATH)
            if text:
                cleaned_text = clean_text(text)
                chunks, chunk_metadata = chunk_text_word_based(cleaned_text, metadata, CHUNK_SIZE)
                save_chunks(chunks, chunk_metadata, CHUNKS_PATH)
                model, _ = create_vector_store(chunks, use_cosine=True)
                st.success("✅ Preprocessing complete!")
            else:
                st.error("❌ Failed to preprocess the book.")
                return
    else:
        chunks, chunk_metadata = load_chunks(CHUNKS_PATH)
        model = SentenceTransformer(EMBEDDING_MODEL)
        index = load_vector_store()

    # User Input Section
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    query = st.text_input("🔍 Enter your question:", placeholder="What is the main contribution of this paper?")
    st.markdown('</div>', unsafe_allow_html=True)

    if query:
        with st.spinner("🔎 Searching and generating response..."):
            relevant_chunks = retrieve_chunks_cosine(query, model, index, chunks)
            answer, context, retrieved_metadata = generate_answer(query, relevant_chunks, chunk_metadata)

        st.markdown("### ✅ Answer")
        st.markdown(f"> {answer}")

        with st.expander("📚 Retrieved Context Chunks", expanded=True):
            for i, (chunk, score) in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}** — *(Cosine Similarity: {score:.4f})*")
                st.text_area(
                    label="Context Chunk",
                    value=chunk,
                    height=150,
                    key=f"chunk_display_{i}",
                    disabled=True,
                    label_visibility="collapsed"
                )
                st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()