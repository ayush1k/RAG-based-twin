# Gemini Developer Context

This document provides system context, architectural constraints, and developer guidelines for AI assistants working on the **"talk to Ayush" portfolio chatbot clone**.

---

## 1. Project Overview

The project is a lightweight, full-stack, RAG-powered chatbot designed to act as a clone of **Ayush** (an CSE graduate pursuing an M.Tech in AI & Data Science). It answers professional questions about his background, experience, projects, and skills.

### Tech Stack
- **Backend**: FastAPI, Uvicorn, python-dotenv
- **RAG & Retrieval**: Scikit-Learn (`TfidfVectorizer` + `cosine_similarity`)
- **LLM API**: Swappable LLM Providers (Anthropic Claude, Hugging Face Serverless Inference, or NVIDIA NIM API)
- **Frontend**: React (single standalone page component with inline styling)

---

## 2. File Directory Map

- `data/`
  - `about.md`, `experience.md`, `faq.md`, `projects.md`: Source documents containing raw professional details.
  - `knowledge_base.json`: Auto-compiled database file containing the processed knowledge chunks and metadata.
- `main.py`: FastAPI application entry point. Implements CORS, the health check `/health`, and the `/chat` route. Contains the persona system prompt and chat history parameters.
- `rag.py`: Handles TF-IDF vectorization, document parsing/chunking, document modification-time comparison, and similarity search queries.
- `verify.py`: Console testing script to debug retrieval logic and print Cosine Similarity scores.
- `ChatbotPage.jsx`: The frontend page containing interactive chat styling (supports dark/light modes), chips, scroll locking, and link formatting.
- `requirements.txt`: Python package dependencies.
- `.env.example`: Template for environment setup.

---

## 3. Important Implementation Constraints

### RAG & Database Logic
- **JSON & Markdown Hybrid**: The source of truth is either the markdown files or `knowledge_base.json`. On startup, `rag.py` checks if any `.md` files are newer than `knowledge_base.json`. If so, it compiles them to JSON automatically. If no `.md` files exist, it loads from the JSON file directly.
- **Lightweight Architecture**: Do **not** replace the TF-IDF search with heavy ML frameworks (like PyTorch, transformers, sentence-transformers, or ChromaDB) unless explicitly asked. The backend is designed to run efficiently on CPU-only, free-tier hosting (e.g. Render, Railway).

### Conversational Capping & Grounding
- **Capped History**: To maintain low latency and prevent excessive API costs, context history is capped to `MAX_HISTORY_TURNS = 6` (3 back-and-forth conversation turns).
- **Follow-up Contextualization**: The current user query is merged with recent turns to build the retrieval query, resolving reference words (like "this", "that", "it") in subsequent queries.
- **Strict Grounding**: The LLM must not invent details. If the context does not supply the answer, the LLM is instructed to state that it doesn't know and invite direct contact.

### Frontend Integration
- **Zero-Dependency Markdown rendering**: [ChatbotPage.jsx](file:///workspaces/RAG-based-twin/ChatbotPage.jsx) uses a built-in regular-expression parser to convert bold formatting, code blocks, and URLs into HTML links. This keeps the React file fully copy-pasteable without installing npm markdown packages.

---

## 4. Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Backend Server (Reload Mode)
```bash
uvicorn main:app --reload
```

### Validate Retrieval Chunks
```bash
python verify.py "query text"
```
