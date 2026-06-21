"""
FastAPI backend for the "talk to Ayush" portfolio chatbot.

Run locally:
    uvicorn main:app --reload

Deploy: any host that runs a Python web service (Render, Railway,
Fly.io, etc). Set the appropriate API keys in the environment.
"""

import json
import os
import urllib.request
import urllib.error

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

from rag import KnowledgeBase

load_dotenv()

app = FastAPI(title="Ayush's portfolio chatbot")

# --- CORS -------------------------------------------------------------
# Add your real portfolio domain(s) here. Keep localhost for local dev.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://your-portfolio-domain.com",  # TODO: replace with your real domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# --- Knowledge base + LLM Client Setup ---------------------------------
kb = KnowledgeBase()

PROVIDER = os.environ.get("MODEL_PROVIDER", "huggingface").lower()

# Read appropriate API keys
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
hf_key = os.environ.get("HF_API_KEY")
nvidia_key = os.environ.get("NVIDIA_API_KEY")

# Set default model names based on provider
if PROVIDER == "anthropic":
    MODEL = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")
    if not anthropic_key:
        print("\n" + "="*80)
        print("WARNING: ANTHROPIC_API_KEY environment variable is not set.")
        print("Create a .env file containing: ANTHROPIC_API_KEY=your_key")
        print("="*80 + "\n")
    client = anthropic.Anthropic(api_key=anthropic_key or "DUMMY_KEY_FOR_STARTUP")
elif PROVIDER == "nvidia":
    MODEL = os.environ.get("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
    if not nvidia_key:
        print("\n" + "="*80)
        print("WARNING: NVIDIA_API_KEY environment variable is not set.")
        print("Create a .env file containing: NVIDIA_API_KEY=your_nvidia_nim_key")
        print("="*80 + "\n")
else:  # default: huggingface
    PROVIDER = "huggingface"
    MODEL = os.environ.get("MODEL_NAME", "google/gemma-2-9b-it")
    if not hf_key:
        print("\n" + "="*80)
        print("WARNING: HF_API_KEY environment variable is not set.")
        print("Create a .env file containing: HF_API_KEY=your_huggingface_token")
        print("="*80 + "\n")


MAX_HISTORY_TURNS = 6  # last 3 exchanges (user+assistant each count as one turn)

PERSONA_PROMPT = """You are answering questions as Ayush, a Computer Science \
graduate currently pursuing an M.Tech in Artificial Intelligence and Data \
Science. You're speaking directly to a visitor on Ayush's portfolio site.

Rules:
- Always answer in first person, as Ayush — never refer to "Ayush" in the \
third person.
- Base every factual claim ONLY on the CONTEXT below. If something isn't \
covered there, say you're not sure or suggest the visitor ask Ayush directly \
— never invent details, dates, or numbers.
- Keep answers conversational and concise (2-4 sentences) unless the \
visitor explicitly asks for more detail.
- If asked something unrelated to Ayush's background, work, or skills, \
politely redirect to what you can help with.

CONTEXT:
{context}"""


# --- API Helper Functions ------------------------------------------------
def call_huggingface_api(system_prompt: str, messages: list, model: str, api_key: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 400,
        "temperature": 0.4
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            response_data = json.loads(res.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise Exception(f"Hugging Face API returned HTTP {e.code}: {error_body}")
    except Exception as e:
        raise Exception(f"Failed to reach Hugging Face API: {e}")


def call_nvidia_nim_api(system_prompt: str, messages: list, model: str, api_key: str) -> str:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 400,
        "temperature": 0.4
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            response_data = json.loads(res.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise Exception(f"NVIDIA NIM API returned HTTP {e.code}: {error_body}")
    except Exception as e:
        raise Exception(f"Failed to reach NVIDIA NIM API: {e}")


# --- Schemas -------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str


# --- Routes -------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(kb.chunks), "provider": PROVIDER, "model": MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    recent_history = req.history[-MAX_HISTORY_TURNS:]

    # Contextualize: fold recent turns into the retrieval query so
    # follow-up questions ("what stack did you use there?") still
    # retrieve the right chunks.
    retrieval_query = " ".join(t.content for t in recent_history) + " " + req.message
    chunks = kb.search(retrieval_query, k=4)
    context_block = "\n\n".join(chunks) if chunks else "No directly relevant info found."

    system_prompt = PERSONA_PROMPT.format(context=context_block)
    messages = [{"role": t.role, "content": t.content} for t in recent_history]
    messages.append({"role": "user", "content": req.message})

    try:
        if PROVIDER == "huggingface":
            if not hf_key:
                raise HTTPException(status_code=401, detail="Hugging Face API key (HF_API_KEY) is not set.")
            answer = call_huggingface_api(system_prompt, messages, MODEL, hf_key)
        elif PROVIDER == "nvidia":
            if not nvidia_key:
                raise HTTPException(status_code=401, detail="NVIDIA NIM API key (NVIDIA_API_KEY) is not set.")
            answer = call_nvidia_nim_api(system_prompt, messages, MODEL, nvidia_key)
        else:  # anthropic
            if not anthropic_key:
                raise HTTPException(status_code=401, detail="Anthropic API key (ANTHROPIC_API_KEY) is not set.")
            response = client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=system_prompt,
                messages=messages,
            )
            answer = "".join(block.text for block in response.content if block.type == "text")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed (Anthropic): {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed ({PROVIDER}): {e}")

    return ChatResponse(answer=answer)

