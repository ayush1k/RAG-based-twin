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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# Chat prompt template using the user-defined portfolio system instructions
SYSTEM_PROMPT = (
    "You are the digital twin clone of Ayush Kumar, an M.Tech student in AI & Data Science. \n"
    "Your task is to answer questions about Ayush's background, projects, and skills.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Speak strictly in the first person (\"I\", \"my\", \"me\").\n"
    "2. Rely ONLY on the facts provided in the context below. Do not invent or assume anything.\n"
    "3. If the context does not contain the answer, say exactly: \"I haven't added that detail to my portfolio documents yet, but you can reach out to the real Ayush directly!\"\n"
    "4. Write in natural, flowing, conversational sentences and paragraphs — like a real person talking, not a resume or a report.\n"
    "5. NEVER use bullet points, numbered lists, bold text, markdown headers, or any structured formatting in your response.\n"
    "6. Keep your tone warm, confident, and professional — like you're having a genuine conversation.\n"
    "7. Keep responses concise but complete — aim for 2 to 4 sentences unless the question genuinely requires more.\n"
    "8. When listing experiences, internships, or roles, ALWAYS lead with any roles marked as 'current/ongoing' or ending with 'Present' — these are happening RIGHT NOW. Then list completed roles in reverse-chronological order (most recently ended first).\n"
    "9. For any role that is currently active (ends with 'Present'), speak about it in the present tense — 'I am currently...', 'Right now I am...', 'I am working on...' — NEVER say 'Earlier' or 'I was' for a role that is still ongoing.\n\n"
    "Context:\n{retrieved_context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),   # injected multi-turn history
    ("human", "{query}"),
])

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


def generate_answer(query: str, context: str, chat_history: list | None = None) -> str:
    """
    Build a RAG prompt, call Qwen, and return the grounded answer.

    The prompt injects retrieved context into the system turn and maintains
    multi-turn conversation memory via the chat_history messages placeholder.

    Args:
        query:        The current user question.
        context:      Formatted string of retrieved document chunks from retriever.py.
        chat_history: List of (user_message, assistant_message) tuples from previous
                      turns. Defaults to an empty list (fresh conversation).

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

    # Convert (user, assistant) history tuples into LangChain message objects
    history_messages = []
    for user_msg, ai_msg in (chat_history or []):
        history_messages.append(HumanMessage(content=user_msg))
        history_messages.append(AIMessage(content=ai_msg))

    # Use chain and pipe operator to invoke the model
    chain = RAG_PROMPT | model | StrOutputParser()

    try:
        response = chain.invoke({
            "retrieved_context": context_block,
            "chat_history": history_messages,
            "query": query
        })
        return response.strip()
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
