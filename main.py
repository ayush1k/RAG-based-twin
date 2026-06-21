"""
main.py — FastAPI Application Entrypoint
=========================================
This is the orchestration layer that ties retriever.py and llm_engine.py
together into a production-ready REST API.

Endpoints:
  POST /chat    — Main RAG chat endpoint (query → retrieve → generate → respond)
  GET  /health  — Liveness check for deployment health monitoring
  GET  /        — Welcome message with usage instructions

Run the server:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from retriever import retrieve_context
from llm_engine import generate_answer

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ayush Kumar — RAG Portfolio Twin",
    description=(
        "A production-grade RAG chatbot powered by Qwen/Qwen2.5-7B-Instruct "
        "and a FAISS dense-embedding vectorstore. Ask anything about Ayush."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local development.
# Restrict in production to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question for Ayush's digital twin.",
        example="What projects has Ayush worked on?",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve (default: 4).",
    )


class ChatResponse(BaseModel):
    query: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The model's grounded response.")
    num_chunks_retrieved: int = Field(
        ..., description="Number of context chunks used for generation."
    )


class HealthResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    """Welcome endpoint with usage instructions."""
    return {
        "name": "Ayush Kumar — RAG Portfolio Twin",
        "description": "Ask me anything about Ayush's experience, projects, and skills.",
        "usage": "Send a POST request to /chat with a JSON body: { \"query\": \"your question\" }",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Liveness probe — confirms the API server is running.
    Does NOT verify the vectorstore or LLM connectivity (use /docs to test those).
    """
    return HealthResponse(status="ok", message="API is live and ready.")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main RAG chat endpoint.

    Pipeline:
      1. Receive user query.
      2. Embed the query and retrieve the top-k most relevant chunks
         from the FAISS vectorstore (built by ingest.py).
      3. Inject the retrieved context + query into Qwen's prompt via llm_engine.py.
      4. Return Qwen's grounded, context-constrained answer.
    """
    query = request.query.strip()
    top_k = request.top_k

    # --- Step 1: Retrieve relevant context chunks from FAISS ---
    try:
        context = retrieve_context(query, k=top_k)
    except RuntimeError as e:
        # Vectorstore not built yet — give the user actionable guidance.
        raise HTTPException(
            status_code=503,
            detail=(
                f"Vectorstore is not initialised: {str(e)}. "
                "Please run `python ingest.py` before starting the server."
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed unexpectedly: {str(e)}",
        )

    # Count how many chunks were actually retrieved
    num_chunks = len([c for c in context.split("--- Chunk") if c.strip()])

    # --- Step 2: Generate a grounded answer using Qwen ---
    try:
        answer = generate_answer(query, context)
    except EnvironmentError as e:
        # Missing HF token — configuration error
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed unexpectedly: {str(e)}",
        )

    return ChatResponse(
        query=query,
        answer=answer,
        num_chunks_retrieved=num_chunks,
    )


# ---------------------------------------------------------------------------
# Development entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
