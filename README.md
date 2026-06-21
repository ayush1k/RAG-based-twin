# Ayush Kumar — RAG-based Digital Twin Portfolio Chatbot

This is a production-grade RAG (Retrieval-Augmented Generation) chatbot designed to act as Ayush Kumar's digital twin clone. It retrieves context from a local knowledge base of Markdown files and generates precise, grounded, first-person answers.

The architecture is built using **FastAPI**, **LangChain**, **FAISS**, **Streamlit**, and the **Hugging Face Serverless Inference API** (for both sentence embeddings and text generation).

---

## 🚀 Key Features
* **Hosted Embeddings**: Leverages `sentence-transformers/all-MiniLM-L6-v2` through the Hugging Face Inference API (no heavy local model downloads required).
* **MMR Retrieval**: Uses Maximal Marginal Relevance (MMR) inside FAISS to balance relevant context retrieval with topic diversity.
* **Strict Grounding**: The LLM persona is strictly instructed to answer in the first person using **only** the retrieved context, falling back to a custom message instead of hallucinating.
* **Streamlit Testing Dashboard**: A sleek, interactive UI to ask questions, tweak RAG retrieval metrics (`top_k`), view raw context chunks, and explore the knowledge source files.
* **FastAPI Server**: Modular REST API with structured Pydantic requests/responses and `/health` checks.

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
├── ingest.py            # Chunks Markdown files and generates/saves the FAISS vector index
├── retriever.py         # Loads FAISS index and runs MMR context retrieval
├── llm_engine.py        # Connects to Qwen/Qwen2.5-7B-Instruct using LCEL prompt chains
├── main.py              # FastAPI application server exposing REST endpoints
├── dashboard.py         # Streamlit visual dashboard for testing and analytics
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
