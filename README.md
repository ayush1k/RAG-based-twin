# Portfolio chatbot — "talk to Ayush"

A RAG-powered chatbot that answers as you, built to drop into your portfolio. FastAPI backend (TF-IDF retrieval + Hugging Face Gemma 2 API by default, with Anthropic Claude and NVIDIA NIM support), React page for the frontend.

## 1. Fill in your knowledge base

Edit the markdown files in `data/` (`data/about.md`, `data/experience.md`, `data/faq.md`, `data/projects.md`) — replace the details with your real information. This content directly determines what the bot knows and how it sounds. Write the answers the way you'd actually say them out loud, not like resume bullets.

You can add more `.md` files to `data/` at any time — they are auto-compiled on startup into `data/knowledge_base.json`. Alternatively, you can edit or maintain `data/knowledge_base.json` directly.

## 2. Run the backend locally

Create a `.env` file in the root directory:
```bash
cp .env.example .env        # then edit .env and choose your LLM provider and paste your API key
```

By default, the backend is configured to use **Hugging Face Serverless Inference** with **`google/gemma-2-9b-it`**. Simply add your Hugging Face Token:
```bash
MODEL_PROVIDER=huggingface
MODEL_NAME=google/gemma-2-9b-it
HF_API_KEY=your_huggingface_token_here
```

Install requirements and run the FastAPI server:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## 3. Verify RAG retrieval locally

Run the test script to verify that queries search the database correctly:
```bash
python verify.py "What is your tech stack?"
```
This script will also trigger first-time compilation of `data/knowledge_base.json` from the markdown files if it hasn't been built yet.

Check the API health check:
```bash
curl http://localhost:8000/health
```

And test the `/chat` route:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are you working on right now?", "history": []}'
```

## 4. Swapping LLM Providers (Optional)

You can easily swap LLM backends in your `.env` without modifying code:

* **Anthropic (Claude):**
  ```bash
  MODEL_PROVIDER=anthropic
  MODEL_NAME=claude-haiku-4-5-20251001
  ANTHROPIC_API_KEY=your_anthropic_api_key
  ```
* **NVIDIA NIM:**
  ```bash
  MODEL_PROVIDER=nvidia
  MODEL_NAME=meta-llama/llama-3.1-8b-instruct
  NVIDIA_API_KEY=your_nvidia_nim_key
  ```

## 5. Deploy the backend

Any host that runs a Python web service works — Render and Railway both have simple free/cheap tiers for this kind of small API:

- Connect your repo, set the start command to:
  `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add the active API key (e.g., `HF_API_KEY`, `ANTHROPIC_API_KEY`, or `NVIDIA_API_KEY`) and `MODEL_PROVIDER`/`MODEL_NAME` as environment variables in the host's dashboard (never commit your real `.env`).
- Once deployed, update `ALLOWED_ORIGINS` in `main.py` to your real portfolio domain, and redeploy.

## 6. Add the frontend page

Copy `ChatbotPage.jsx` into your portfolio project and wire it up as a route (e.g. `/chat` or `/ask-me`). Update `BACKEND_URL` at the top of the file to your deployed backend's URL.

If your portfolio uses Next.js App Router, add `"use client"` as the very first line of the file.

The UI supports light and dark themes (toggled via the sun/moon button in the top right), includes a clear-chat history button, features interactive prompt chips, and contains an inline parser that formats links (`[label](url)` and raw URLs), bold text (`**`), and code blocks (`` ` ``) so recruiters can click directly on your socials and project URLs.

## How it works

- **Query Contextualization**: Each message is combined with your last few turns of chat history before retrieval, so follow-up questions ("what stack did you use there?") still pull the right context — see `main.py`'s `retrieval_query` line.
- **TF-IDF Retrieval**: Retrieval is TF-IDF based (`scikit-learn`), not a neural embedding model — deliberately lightweight so it deploys cleanly on a free tier with no GPU. It works well at this scale because queries usually share keywords with your source text. If you want richer semantic retrieval later, swap `KnowledgeBase.search()` in `rag.py` for an embedding-based approach — nothing else needs to change.
- **Strict Grounding**: The persona prompt in `main.py` explicitly tells the model to only use facts from the retrieved context and to say "not sure" rather than invent details — this is what keeps the bot from hallucinating a fake metric or project detail under questioning.
- **Conversation History Capping**: We limit the context history to the last `6` turns (i.e. 3 user prompts and 3 assistant replies). This prevents excessive token costs and keeps your API billing predictable.
