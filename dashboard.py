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
    .chat-card {
        background-color: #1E293B; /* Tailwind Slate 800 */
        color: #F8FAFC; /* Tailwind Slate 50 (Crisp white) */
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
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

# Sidebar configurations
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
                      
    show_raw_chunks = st.checkbox("Show Raw Context Chunks", value=True,
                                  help="Display the exact text blocks retrieved from the Markdown files.")

    st.markdown("---")
    st.markdown("### ⚙️ System Specifications")
    st.markdown("- **Embedding Model (API):** `sentence-transformers/all-MiniLM-L6-v2`")
    st.markdown("- **Generative LLM (API):** `Qwen/Qwen2.5-7B-Instruct`")
    st.markdown("- **Vector Database:** `FAISS` (Local persistence)")

# Header section
st.markdown("<div class='main-title'>Ayush's Twin</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Streamlit dashboard to test and validate the RAG retrieval and generation pipeline.</div>", unsafe_allow_html=True)

# Main UI Tabs
tab1, tab2 = st.tabs(["💬 Ask the Twin", "📚 Knowledge Source Documents"])

with tab1:
    st.info("💡 **Try asking:** *What projects has Ayush worked on?*, *What is Ayush's background?*, or *What technologies does he prefer?*")
    
    query = st.text_input("Enter your query for the digital twin:", placeholder="e.g. What ML experience does Ayush have?", key="query_input")
    
    if st.button("Generate Answer", type="primary") and query:
        try:
            # Import retriever and llm_engine dynamically
            from retriever import retrieve_context
            from llm_engine import generate_answer
            
            with st.spinner("🔍 Querying FAISS and generating response from Hugging Face API..."):
                # Step 1: Retrieve context
                context = retrieve_context(query, k=top_k)
                
                # Step 2: Generate answer using the chain
                answer = generate_answer(query, context)
            
            # Display Answer
            st.markdown("### 🤖 Answer:")
            st.markdown(f"<div class='chat-card'>{answer}</div>", unsafe_allow_html=True)
            
            # Display Context Chunks if selected
            if show_raw_chunks:
                st.markdown("---")
                st.markdown("### 🔍 Retrieved Context Chunks:")
                if not context or context.strip() == "":
                    st.warning("No context chunks were retrieved from the FAISS index for this query.")
                else:
                    chunks = context.split("--- Chunk ")
                    for chunk in chunks:
                        if not chunk.strip():
                            continue
                        
                        # Parse source and content
                        parts = chunk.split("\n", 1)
                        header = parts[0]
                        content = parts[1] if len(parts) > 1 else ""
                        
                        source_info = "Source"
                        if "source: " in header:
                            source_info = header.split("source: ")[1].rstrip(")")
                        
                        st.markdown(f"<span class='badge'>{source_info}</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='context-card'>{content.strip()}</div>", unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Error executing RAG pipeline: {e}")
            st.exception(e)

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
