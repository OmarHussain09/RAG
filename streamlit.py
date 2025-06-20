# app.py

import streamlit as st
from rag_engine import (
    extract_text_from_pdf, clean_text, chunk_text,
    build_vector_store, retrieve_chunks, rerank_chunks,
    load_llm, generate_answer
)

# App layout
st.set_page_config(page_title="📘 RAG PDF QA", layout="wide")
st.title("📘 RAG-Based PDF Question Answering")
st.markdown("Upload a PDF and ask questions based on its content.")

# Sidebar settings
st.sidebar.title("Settings")
# llm_choice = st.sidebar.radio("LLM Backend", ["Groq", "Ollama"])
llm_choice = st.sidebar.radio("LLM Backend", ["Groq", "Ollama", "Gemini"])
chunk_method = st.sidebar.radio("Chunking Method", ["Word-based", "Semantic", "Recursive"])
chunk_size = st.sidebar.slider("Chunk Size (words)", 300, 1000, 600, 100)
similarity = st.sidebar.radio("Similarity Metric", ["Cosine", "L2"])
top_k = st.sidebar.slider("Top-k Chunks", 3, 10, 5)
use_reranker = st.sidebar.checkbox("Use FlashRank Reranker", value=True)
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# Optional: clear cached state if a new file is uploaded
if uploaded_file and st.session_state.get("pdf_id") != uploaded_file.name:
    for key in ["pdf_id", "chunks", "chunk_meta", "embedder", "index"]:
        st.session_state.pop(key, None)

# Persistent state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.chunk_meta = []
    st.session_state.embedder = None
    st.session_state.index = None

# # Process PDF
# if uploaded_file:
#     with st.spinner("Processing PDF..."):
#         raw_text, meta = extract_text_from_pdf(uploaded_file)
#         cleaned = clean_text(raw_text)
#         chunks, chunk_meta = chunk_text(cleaned, meta, method=chunk_method, chunk_size=chunk_size)
#         embedder, index = build_vector_store(chunks, use_cosine=(similarity == "Cosine"))

#         st.session_state.chunks = chunks
#         st.session_state.chunk_meta = chunk_meta
#         st.session_state.embedder = embedder
#         st.session_state.index = index

#         st.success("PDF processed and indexed!")
if uploaded_file and "pdf_id" not in st.session_state:
    with st.spinner("Processing PDF..."):
        raw_text, meta = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(raw_text)
        chunks, chunk_meta = chunk_text(cleaned, meta, method=chunk_method, chunk_size=chunk_size)
        embedder, index = build_vector_store(chunks, use_cosine=(similarity == "Cosine"))

        # Store everything in session_state
        st.session_state.chunks = chunks
        st.session_state.chunk_meta = chunk_meta
        st.session_state.embedder = embedder
        st.session_state.index = index
        st.session_state.pdf_id = str(uploaded_file.name)  # mark this file as processed

        st.success("PDF processed and indexed!")


# Ask questions
if uploaded_file and st.session_state.embedder:
    query = st.text_input("🔎 Ask a question:")
    if query:
        with st.spinner("Searching..."):
            results = retrieve_chunks(query, st.session_state.embedder, st.session_state.index, st.session_state.chunks, k=top_k, use_cosine=(similarity == "Cosine"))
            if use_reranker:
                results = rerank_chunks(query, results)

            llm = load_llm(llm_choice)
            answer, _ = generate_answer(query, results, llm)

            st.markdown("### ✅ Answer")
            st.write(answer)

            with st.expander("📚 Retrieved Chunks"):
                for i, (chunk, score) in enumerate(results):
                    st.markdown(f"**Chunk {i+1}** *(Score: {score:.4f})*")
                    st.text_area("Context", chunk, height=150, disabled=True, key=f"chunk_{i}", label_visibility="collapsed")
