"""
ingest.py — Data Ingestion Pipeline
====================================
Responsibilities:
  1. Load all Markdown (.md) files from the `data/` directory.
  2. Split them into manageable chunks using RecursiveCharacterTextSplitter.
  3. Generate dense embeddings for every chunk using the local
     `sentence-transformers/all-MiniLM-L6-v2` model (no API key required).
  4. Store the embedding vectors in a FAISS index and persist it to disk
     at `vectorstore/` so retriever.py can load it without rebuilding.

Run this script once (or whenever your data/ changes):
    python ingest.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

# Local embedding model — fast, runs fully offline, no API key needed.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings — smaller chunks keep individual entries (e.g. one internship) intact
# and prevent unrelated experiences from merging into the same retrieval unit.
CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 100     # overlap to preserve context across chunk boundaries


def load_documents():
    """Load all Markdown files from the data/ directory."""
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        print("  Create the `data/` folder and add your Markdown portfolio files.")
        sys.exit(1)

    md_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".md")]
    if not md_files:
        print(f"[ERROR] No Markdown (.md) files found in: {DATA_DIR}")
        print("  Add your portfolio content as .md files inside the `data/` directory.")
        sys.exit(1)

    print(f"[INFO] Found {len(md_files)} Markdown file(s) in {DATA_DIR}.")

    # DirectoryLoader globs *.md and uses TextLoader for each file (UTF-8 safe).
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} document(s) total.")
    return docs


def split_documents(docs):
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    This splitter tries to break on double-newlines first (paragraphs),
    then single newlines, then spaces — keeping semantic coherence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # hierarchy of split points
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunk(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_vectorstore(chunks):
    """
    Embed each chunk with all-MiniLM-L6-v2 and store in a FAISS index.
    The index is persisted to disk so retriever.py can load it instantly.
    """
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    token = os.getenv("HF_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_ACCESS_TOKEN is not set in the environment or .env file."
        )
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=token,
    )

    print("[INFO] Generating embeddings and building FAISS index…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist the FAISS index to disk
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"[SUCCESS] Vectorstore saved to: {VECTORSTORE_DIR}")
    return vectorstore


def main():
    print("=" * 60)
    print("  RAG Ingestion Pipeline — Ayush Kumar's Portfolio Twin")
    print("=" * 60)

    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)

    print("\n[DONE] Ingestion complete. Run `uvicorn main:app` to start the API.")


if __name__ == "__main__":
    main()
