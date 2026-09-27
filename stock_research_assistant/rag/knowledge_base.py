"""
rag/knowledge_base.py
─────────────────────
RAG pipeline for the Stock Research Assistant.

Builds (or loads from disk) a FAISS vector store from the documents in
rag/documents/, then exposes:
  - build_vectorstore()   – (re)build and persist the index
  - get_retriever()       – LangChain retriever ready to use as a tool
  - get_rag_tool()        – LangChain Tool wrapper with citation support

Dependencies: langchain, langchain-openai, faiss-cpu, sentence-transformers
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Embeddings ─────────────────────────────────────────────────────────────────

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=config.GEMINI_API_KEY,
    )


# ── Build / Load vector store ──────────────────────────────────────────────────

def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    Load documents from RAG_DOCS_PATH, chunk them, embed and save to FAISS.
    If a persisted index already exists and force_rebuild=False, load it.
    """
    store_path = Path(config.VECTOR_STORE_PATH)
    embeddings = _get_embeddings()

    if store_path.exists() and not force_rebuild:
        print(f"[RAG] Loading existing vector store from {store_path}")
        return FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print(f"[RAG] Building vector store from {config.RAG_DOCS_PATH} …")
    docs_path = Path(config.RAG_DOCS_PATH)

    # Load all .txt files from the documents directory
    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    raw_docs: List[Document] = loader.load()

    if not raw_docs:
        raise RuntimeError(
            f"No documents found in {docs_path}. "
            "Please add .txt files to rag/documents/."
        )

    # Enrich metadata with source filename
    for doc in raw_docs:
        src = doc.metadata.get("source", "")
        doc.metadata["source"] = Path(src).name if src else "unknown"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks: List[Document] = splitter.split_documents(raw_docs)
    print(f"[RAG] Split into {len(chunks)} chunks from {len(raw_docs)} documents.")

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist
    store_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(store_path))
    print(f"[RAG] Vector store saved to {store_path}")

    return vectorstore


# ── Retriever ──────────────────────────────────────────────────────────────────

_vectorstore: FAISS | None = None


def get_retriever():
    """Return a LangChain similarity retriever (top-K chunks)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RAG_TOP_K},
    )


# ── LangChain Tool ─────────────────────────────────────────────────────────────

def _rag_query(question: str) -> str:
    """
    Retrieve relevant knowledge-base passages for the given question and
    return them with source citations.
    """
    retriever = get_retriever()
    docs: List[Document] = retriever.invoke(question)

    if not docs:
        return "No relevant information found in the knowledge base for this question."

    parts = [
        f"**[Source: {doc.metadata.get('source', 'knowledge base')}]**\n{doc.page_content}"
        for doc in docs
    ]
    return "\n\n---\n\n".join(parts)


def get_rag_tool() -> Tool:
    """
    Return a LangChain Tool that wraps the RAG retriever.
    This is one of the mandatory 'additional LangChain tools'.
    """
    return Tool(
        name="financial_knowledge_base",
        func=_rag_query,
        description=(
            "Search the financial knowledge base for definitions, explanations "
            "and educational content about financial ratios, concepts, SEBI regulations, "
            "portfolio risk and Indian market mechanics. "
            "Input should be a clear question about a financial concept. "
            "Returns relevant passages WITH source citations."
        ),
    )
