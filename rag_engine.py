# rag_engine.py

import os
import json
import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------- PDF Processing ---------------- #

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text, metadata = "", []
    for i, page in enumerate(reader.pages):
        text += page.extract_text() + "\n"
        metadata.append({"page": i + 1})
    return text, metadata


def clean_text(text):
    lines = text.split("\n")
    return " ".join([line.strip() for line in lines if line.strip()])


# ---------------- Chunking ---------------- #

# def chunk_text(text, metadata, method="Word-based", chunk_size=600):
#     if method == "Semantic":
#         embedder = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-base-en-v1.5",
#             encode_kwargs={"normalize_embeddings": True}
#         )

#         def add_prefix(chunk):
#             return "Represent this sentence for searching relevant passages: " + chunk

#         splitter = SemanticChunker(
#             embedder,
#             embedding_function=lambda x: embedder.embed_documents([add_prefix(xi) for xi in x])
#         )
#         chunks = splitter.split_text(text)
#         chunk_metadata = [{"page": metadata[min(i, len(metadata)-1)]["page"]} for i in range(len(chunks))]
#     else:
#         words = text.split()
#         chunks, chunk_metadata = [], []
#         current_chunk, word_count, page_index = [], 0, 0
#         for word in words:
#             current_chunk.append(word)
#             word_count += 1
#             if word_count >= chunk_size:
#                 chunks.append(" ".join(current_chunk))
#                 chunk_metadata.append({"page": metadata[page_index]["page"]})
#                 current_chunk, word_count = [], 0
#                 page_index = min(page_index + 1, len(metadata) - 1)
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
#             chunk_metadata.append({"page": metadata[page_index]["page"]})
#     return chunks, chunk_metadata


def chunk_text(text, metadata, method="Word-based", chunk_size=600):
    if method == "Semantic":
        embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )
        splitter = SemanticChunker(embedder)
        chunks = splitter.split_text(text)
        chunk_metadata = [{"page": metadata[min(i, len(metadata)-1)]["page"]} for i in range(len(chunks))]

    elif method == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        chunk_metadata = [{"page": metadata[min(i, len(metadata)-1)]["page"]} for i in range(len(chunks))]

    else:  # Default: word-based
        words = text.split()
        chunks, chunk_metadata = [], []
        current_chunk, word_count, page_index = [], 0, 0
        for word in words:
            current_chunk.append(word)
            word_count += 1
            if word_count >= chunk_size:
                chunks.append(" ".join(current_chunk))
                chunk_metadata.append({"page": metadata[page_index]["page"]})
                current_chunk, word_count = [], 0
                page_index = min(page_index + 1, len(metadata) - 1)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            chunk_metadata.append({"page": metadata[page_index]["page"]})

    return chunks, chunk_metadata



# ---------------- Embeddings + Retrieval ---------------- #

def build_vector_store(chunks, model_name="all-MiniLM-L6-v2", use_cosine=True):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    if use_cosine:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) if use_cosine else faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return model, index


def retrieve_chunks(query, model, index, chunks, k=5, use_cosine=True):
    q_emb = model.encode([query])[0]
    if use_cosine:
        q_emb = q_emb / np.linalg.norm(q_emb)
    D, I = index.search(np.array([q_emb]).astype(np.float32), k)
    return [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]


# ---------------- Reranker ---------------- #

_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
    return _reranker


def rerank_chunks(query, retrieved_chunks):
    reranker = get_reranker()
    passages = [chunk for chunk, _ in retrieved_chunks]
    pairs = [(query, passage) for passage in passages]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, score) for chunk, score in reranked]


# ---------------- Answer Generator ---------------- #

# def load_llm(backend="Groq"):
#     if backend == "Ollama":
#         return ChatOllama(model="llama3.1", base_url="http://localhost:11434")
#     else:
#         return ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

def load_llm(backend="Groq"):
    if backend == "Ollama":
        return ChatOllama(model="llama3.1", base_url="192.168.11.204:11434")
    elif backend == "Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # or gemini-1.5-pro
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyC6ntt0B16X3KRgTJFp2vGXq0QKyx8Cw-c")
        )
    else:
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY", "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"))

def generate_answer(query, context_chunks, llm):
    context = "\n".join([chunk for chunk, _ in context_chunks])
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context. If not answerable, say: "Answer not found in the book."

Context:
{context}

Question:
{query}
""")
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return response.content, context
