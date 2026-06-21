"""
llm_engine.py — LLM Generation Engine
=======================================
Responsibilities:
  1. Connect to `Qwen/Qwen2.5-7B-Instruct` via the Hugging Face Serverless
     API using an OpenAI-compatible chat completions interface.
  2. Enforce a strict system prompt so that the model acts as Ayush Kumar's
     digital twin and answers ONLY from the retrieved context — no hallucination.
  3. Expose `generate_answer(query, context)` — the single public function
     called by main.py after the retriever has fetched the relevant chunks.

Architecture note:
  We use `langchain_huggingface.ChatHuggingFace` wrapping `HuggingFaceEndpoint`
  (task="conversational"). This routes requests through the HF router's
  chat/completions API, which is the only task supported by the third-party
  providers (Together, Novita, Featherless-AI) that HF Serverless maps
  modern instruction models to.
"""

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# Strict system prompt — grounds the model as Ayush's digital twin.
# The model is explicitly forbidden from hallucinating or going off-context.
SYSTEM_PROMPT = (
    "You are the digital twin clone of Ayush Kumar, a software engineer and AI practitioner. "
    "Your sole purpose is to answer questions about Ayush's background, projects, skills, "
    "and experiences using ONLY the retrieved context provided in each message. "
    "If the answer cannot be found in the context, say: "
    "'I don't have that information in my knowledge base.' "
    "Do NOT hallucinate, speculate, or use any outside knowledge. "
    "Speak in the first person, as if you are Ayush himself. "
    "Be concise, professional, and helpful."
)

# ---------------------------------------------------------------------------
# Module-level singleton — initialised once per process
# ---------------------------------------------------------------------------
_chat_model = None


def _get_chat_model() -> ChatHuggingFace:
    """
    Lazily initialise the ChatHuggingFace model singleton.
    This avoids re-creating the connection object on every request.
    """
    global _chat_model

    if _chat_model is None:
        if not HF_TOKEN:
            raise EnvironmentError(
                "HF_ACCESS_TOKEN is not set. "
                "Add it to your .env file or export it as an environment variable."
            )

        print(f"[LLM Engine] Initialising model: {MODEL_ID}")

        endpoint = HuggingFaceEndpoint(
            repo_id=MODEL_ID,
            huggingfacehub_api_token=HF_TOKEN,
            task="conversational",        # required for HF router partner providers
            max_new_tokens=512,
            temperature=0.2,              # low temp for factual, grounded answers
            repetition_penalty=1.1,
        )

        _chat_model = ChatHuggingFace(llm=endpoint)
        print("[LLM Engine] Model ready.")

    return _chat_model


def generate_answer(query: str, context: str) -> str:
    """
    Build a RAG prompt, call Qwen, and return the grounded answer.

    The prompt structure injects retrieved context into the human turn so that
    the model has all necessary information before seeing the user's question.

    Args:
        query:   The original user question.
        context: Formatted string of retrieved document chunks from retriever.py.

    Returns:
        The model's text response as a string.
        Returns a fallback message if the model call fails.
    """
    model = _get_chat_model()

    # If retrieval returned nothing, tell the model explicitly.
    if not context or context.strip() == "":
        context_block = "No relevant context was found in the knowledge base."
    else:
        context_block = context

    # The human message bundles context + question in a clear, structured format.
    human_message_content = (
        f"## Retrieved Context\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"## Question\n\n"
        f"{query}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_message_content),
    ]

    try:
        response = model.invoke(messages)
        return response.content.strip()
    except Exception as e:
        error_msg = f"[LLM Engine ERROR] Failed to generate response: {e}"
        print(error_msg)
        return (
            "I'm sorry, I encountered an error while generating a response. "
            "Please try again in a moment."
        )


# ---------------------------------------------------------------------------
# Quick smoke test — run: python llm_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_context = (
        "--- Chunk 1 (source: about.md) ---\n"
        "Ayush Kumar is a software engineer with 3 years of experience "
        "specialising in machine learning, NLP, and full-stack development. "
        "He has worked on multiple production RAG systems and open-source projects."
    )
    sample_query = "What does Ayush specialise in?"

    print(f"[TEST] Query: {sample_query}\n")
    answer = generate_answer(sample_query, sample_context)
    print(f"[TEST] Answer:\n{answer}")
