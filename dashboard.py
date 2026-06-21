import streamlit as st
import os
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Check for API key
hf_token = os.getenv("HF_ACCESS_TOKEN")

# Set page configuration with premium design aesthetics in mind
st.set_page_config(
    page_title="RAG based AI Twin",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (dark mode friendly, sleek fonts, gradients)
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    .context-card {
        background-color: #0F172A; /* Tailwind Slate 900 */
        color: #E2E8F0; /* Tailwind Slate 200 (Light grey) */
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
    .badge {
        background-color: #333;
        color: #FF4B4B;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise session state for conversation history
# messages: list of {"role": "user"|"assistant", "content": str, "context": str}
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# ---------------------------------------------------------------------------
# Sidebar configurations
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 RAG Engine Settings")

    if hf_token:
        st.success("HF Token Loaded successfully!")
    else:
        st.error("HF_ACCESS_TOKEN not found in environment/.env")

    st.markdown("---")

    # RAG Tuning Parameters
    top_k = st.slider("Top K Document Chunks", min_value=1, max_value=10, value=4, step=1,
                      help="Number of chunks to retrieve from the FAISS database.")

    show_raw_chunks = st.checkbox("Show Raw Context Chunks", value=False,
                                  help="Display the retrieved context blocks for the last question.")

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ System Specifications")
    st.markdown("- **Embedding Model (API):** `sentence-transformers/all-MiniLM-L6-v2`")
    st.markdown("- **Generative LLM (API):** `Qwen/Qwen2.5-7B-Instruct`")
    st.markdown("- **Vector Database:** `FAISS` (Local persistence)")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='main-title'>Ayush's Twin</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Streamlit dashboard to test and validate the RAG retrieval and generation pipeline.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main UI Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["💬 Chat with the Twin", "📚 Knowledge Source Documents"])

with tab1:
    # Hint row (only shown when conversation is empty)
    if not st.session_state.messages:
        st.info("💡 **Try asking:** *What projects has Ayush worked on?*, *What is Ayush's background?*, or *What technologies does he prefer?*")

    # -----------------------------------------------------------------------
    # Render full conversation history using st.chat_message
    # -----------------------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # -----------------------------------------------------------------------
    # Show raw context chunks for the LAST assistant reply (if toggled)
    # -----------------------------------------------------------------------
    if show_raw_chunks and st.session_state.last_context:
        with st.expander("🔍 Retrieved Context Chunks for last question", expanded=False):
            context = st.session_state.last_context
            if not context.strip():
                st.warning("No context chunks were retrieved from the FAISS index for that query.")
            else:
                chunks = context.split("--- Chunk ")
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    parts = chunk.split("\n", 1)
                    header = parts[0]
                    content = parts[1] if len(parts) > 1 else ""
                    source_info = header.split("source: ")[1].rstrip(")") if "source: " in header else "Source"
                    st.markdown(f"<span class='badge'>{source_info}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='context-card'>{content.strip()}</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Chat input — persists at the bottom of the page
    # -----------------------------------------------------------------------
    if prompt := st.chat_input("Ask anything about Ayush..."):
        # Import here to avoid circular issues on cold start
        from retriever import retrieve_context
        from llm_engine import generate_answer

        # Append user message immediately so it appears in the UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Build chat history from previous turns (exclude the just-added user msg)
        history = [
            (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
            for i in range(0, len(st.session_state.messages) - 2, 2)
            if st.session_state.messages[i]["role"] == "user"
            and st.session_state.messages[i + 1]["role"] == "assistant"
        ]

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching knowledge base and generating answer..."):
                # Step 1: Retrieve context for the current query
                context = retrieve_context(prompt, k=top_k)
                st.session_state.last_context = context

                # Step 2: Generate grounded answer with full history
                answer = generate_answer(prompt, context, chat_history=history)

            st.markdown(answer)

        # Persist the assistant reply to session history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Rerun to cleanly re-render the context expander at the correct position
        st.rerun()

with tab2:
    st.markdown("### 📖 Managed Knowledge Base")
    st.write("These are the Markdown files located inside the `/data` directory that build the FAISS index:")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(".md")]
        if files:
            for file in files:
                with st.expander(f"📄 {file}"):
                    with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                        st.markdown(f.read())
        else:
            st.warning("No Markdown files found in data/")
    else:
        st.error("Data directory not found.")
