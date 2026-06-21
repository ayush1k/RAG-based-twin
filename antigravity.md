# Project Status: RAG-based Twin

This file documents the current state, recent diagnoses, fixes, and architecture of the RAG-based Twin project.

---

## 1. Project Overview
The goal of this project is to build a Retrieval-Augmented Generation (RAG) system using **LangChain** and **Hugging Face** serverless inference API.

---

## 2. Directory Structure
Currently, the repository is structured as follows:

```text
RAG-based-twin/
├── .env                # Environment variables containing HF API keys
├── .gitignore          # Git exclusion rules
├── LICENSE             # Project license
├── README.md           # Instructions on project usage
├── requirements.txt    # Python package dependencies
├── rag.py              # Main entry point for RAG execution (fixed & working)
├── data/               # Target directory for vector database source files
└── .venv/              # Project local Python virtual environment
```

---

## 3. Core Diagnoses & Working Solution

### The Issue
Using a standard `HuggingFaceEndpoint` with instruction-following models (e.g. `google/gemma-2-9b-it`, `mistralai/Mistral-7B-Instruct-v0.3`) failed with a `StopIteration` error inside the `huggingface_hub` package router.

- **Reason**: Hugging Face Serverless API routes modern instruction models to third-party providers (like Together, Novita, or Featherless-AI). These providers only support the `conversational` (chat completions) task, whereas `HuggingFaceEndpoint`'s internal invocation defaults to the `text-generation` task.

### The Fix
We updated [rag.py](file:///workspaces/RAG-based-twin/rag.py) to:
1. Use `Qwen/Qwen2.5-7B-Instruct` (which is highly capable and fully supported via the HF serverless router).
2. Initialize `HuggingFaceEndpoint` with `task="conversational"`.
3. Wrap the endpoint inside `ChatHuggingFace`, which redirects invocation requests to the `chat_completion` API of `huggingface_hub`.

---

## 4. Current File Contents

### `rag.py`
```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Setup the model
# Qwen/Qwen2.5-7B-Instruct is supported via the Hugging Face Serverless API's conversational task.
model_id = "Qwen/Qwen2.5-7B-Instruct"
token = os.getenv("HF_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id=model_id,
    huggingfacehub_api_token=token,
    task="conversational"
)

chat = ChatHuggingFace(llm=llm)

# Execute and print results
messages = [HumanMessage(content="What is the capital of India?")]
result = chat.invoke(messages)
print(result.content)
```

---

## 5. Next Steps
1. **RAG Pipeline Implementation**:
   - Load documents from the `data/` directory (e.g. using `DirectoryLoader` or `PyPDFLoader`).
   - Chunk documents using `RecursiveCharacterTextSplitter`.
   - Embed chunks using a compatible HF embeddings endpoint (like `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`).
   - Store vectors in a lightweight vector DB (such as Chroma or FAISS).
2. **FastAPI Integration**:
   - Create a REST endpoint using FastAPI to query the RAG pipeline.
