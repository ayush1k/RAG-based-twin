# Ayush Kumar — RAG-based Digital Twin Portfolio Chatbot

This is a production-grade RAG (Retrieval-Augmented Generation) chatbot designed to act as Ayush Kumar's digital twin clone. It retrieves context from a local knowledge base of Markdown files and generates precise, grounded, first-person answers.

The architecture is built using **FastAPI**, **LangChain**, **FAISS**, **Streamlit**, and the **Hugging Face Serverless Inference API** (for both sentence embeddings and text generation).

---

## 🚀 Key Features
* **Hosted Embeddings**: Leverages `sentence-transformers/all-MiniLM-L6-v2` through the Hugging Face Serverless Inference API (fast, lightweight, no heavy local model weight downloads).
* **Semantic Similarity Search**: Optimized retrieval with `top_k=10` using FAISS similarity search. We deliberately transitioned from MMR to plain similarity search to ensure multiple relevant experience entries from the same file aren't suppressed.
* **Strict First-Person Grounding**: The LLM acts strictly as Ayush's digital twin clone, answering in natural, flowing conversational sentences (no lists/markdown headers) strictly from the retrieved context. If information is missing, it returns a friendly fallback message.
* **Multi-Turn Conversational Chatbot**: Built with Streamlit (`st.chat_input`, `st.chat_message`, and `st.session_state`), enabling dynamic conversations that remember previous questions and answers.
* **FastAPI Backend Server**: Exposes standard, structured Pydantic REST endpoints (`POST /chat` and `GET /health`) with built-in CORS configurations.

---

## 📁 Directory Structure
```text
RAG-based-twin/
├── .streamlit/
│   └── config.toml      # Disables deep library watchers to suppress torchvision warning spams
├── data/                # Source of truth — add portfolio details as Markdown (.md) here
│   ├── about.md
│   ├── experience.md
│   ├── faq.md
│   └── projects.md
├── vectorstore/         # Persisted local FAISS index (built by ingest.py)
│
├── ingest.py            # Chunks Markdown files (600 char size, 100 overlap) and builds FAISS index
├── retriever.py         # Loads FAISS index and performs semantic similarity search (k=10)
├── llm_engine.py        # Prompts Qwen/Qwen2.5-7B-Instruct with conversational memory and grounding
├── main.py              # FastAPI application server exposing REST endpoints
├── dashboard.py         # Streamlit conversational chatbot and analytics interface
│
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation (this file)
└── antigravity.md       # Project status and implementation tracker
```

---

## 🛠️ Setup & Installation

### 1. Clone & Set Up Environment
Navigate into the workspace and set up a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API Keys
Create a `.env` file in the root directory:
```env
HF_ACCESS_TOKEN=your_huggingface_access_token_here
```

---

## 💻 Running the Application

### Step 1: Populate Data & Build the Vectorstore
Make sure your profile files are in the `data/` directory, then compile the FAISS index:
```bash
python ingest.py
```
This builds and saves your embeddings database locally to `vectorstore/`.

### Step 2: Run the FastAPI Server
Start the backend endpoint:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
* Interactive API documentation is available at `http://localhost:8000/docs`.

### Step 3: Run the Streamlit Dashboard
Open a separate terminal and start the interactive front-end:
```bash
streamlit run dashboard.py
```

---

## 🔌 API Documentation

### **POST** `/chat`
Generates a grounded response based on the query.

* **Request Body:**
```json
{
  "query": "What machine learning projects has Ayush worked on?",
  "top_k": 4
}
```

* **Response:**
```json
{
  "query": "What machine learning projects has Ayush worked on?",
  "answer": "I have worked on a Multi-Modal Agentic OCR & Document Intelligence Hub...",
  "num_chunks_retrieved": 4
}
```

### **GET** `/health`
Returns a simple liveness status of the API server.
```json
{
  "status": "ok",
  "message": "API is live and ready."
}
```
