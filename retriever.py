"""
retriever.py — Dense Retrieval Engine
=======================================
Responsibilities:
  1. Load the persisted FAISS vectorstore from disk (built by ingest.py).
     This is done once at module import time so the index stays in memory
     across multiple API requests (no repeated disk I/O per query).
  2. Expose `retrieve_context(query, k)` — the single public function used
     by llm_engine.py and main.py.

Flow:
    User Query → Embed with all-MiniLM-L6-v2 → FAISS similarity search
    → Top-k document chunks → Formatted context string
"""

import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Configuration — must match the values used in ingest.py
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default number of chunks returned per query
DEFAULT_TOP_K = 4


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused across all requests
# ---------------------------------------------------------------------------
_vectorstore = None
_embeddings = None


def _load_vectorstore():
    """
    Load the FAISS index from disk into memory.
    Called lazily on the first retrieval request.
    Raises a descriptive RuntimeError if ingest.py has not been run yet.
    """
    global _vectorstore, _embeddings

    if not os.path.isdir(VECTORSTORE_DIR):
        raise RuntimeError(
            f"Vectorstore not found at '{VECTORSTORE_DIR}'. "
            "Please run `python ingest.py` first to build the index."
        )

    if _embeddings is None:
        print(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if _vectorstore is None:
        print(f"[Retriever] Loading FAISS index from: {VECTORSTORE_DIR}")
        # allow_dangerous_deserialization is required for FAISS pickle loading
        _vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[Retriever] Vectorstore loaded successfully.")

    return _vectorstore


def retrieve_context(query: str, k: int = DEFAULT_TOP_K) -> str:
    """
    Embed the user query, run a similarity search against the FAISS index,
    and return the top-k matching document chunks as a single formatted string.

    Args:
        query: The user's natural-language question.
        k:     Number of document chunks to retrieve (default: 4).

    Returns:
        A formatted string with each retrieved chunk separated by a divider.
        Returns an empty string if no documents are found.

    Raises:
        RuntimeError: If the vectorstore has not been initialised (ingest.py
                      not yet run).
    """
    store = _load_vectorstore()

    # similarity_search returns a list of Document objects ranked by cosine sim
    results = store.similarity_search(query, k=k)

    if not results:
        return ""

    # Format each chunk with its source file for transparency
    context_parts = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        source_name = os.path.basename(source)
        context_parts.append(
            f"--- Chunk {i} (source: {source_name}) ---\n{doc.page_content.strip()}"
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Quick smoke test — run: python retriever.py "your question here"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about Ayush's projects."
    print(f"\n[TEST] Query: {query}\n")
    try:
        context = retrieve_context(query)
        if context:
            print("[TEST] Retrieved context:\n")
            print(context)
        else:
            print("[TEST] No matching chunks found.")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
