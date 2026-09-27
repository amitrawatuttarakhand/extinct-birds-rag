import os
import time
import streamlit as st
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Eco-Birds RAG | Sustainable AI Knowledge Base",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f4f9f5 0%, #ffffff 100%);
        font-family: 'Inter', sans-serif;
    }
    .eco-header {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.2);
    }
    .eco-header h1 { color: white !important; font-weight: 700; font-size: 2.2rem; margin: 0; }
    .eco-header p { color: #a7f3d0 !important; font-size: 1.05rem; margin-top: 6px; margin-bottom: 0; }
    .green-badge {
        background-color: #d1fae5; color: #065f46; padding: 4px 12px;
        border-radius: 20px; font-size: 0.85rem; font-weight: 600;
        display: inline-block; border: 1px solid #a7f3d0;
    }
    .cache-badge {
        background-color: #fef3c7; color: #92400e; padding: 4px 12px;
        border-radius: 20px; font-size: 0.85rem; font-weight: 600;
        display: inline-block; border: 1px solid #fde68a;
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 16px;
        border: 1px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Secrets & Model Setup
# ---------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ `GROQ_API_KEY` missing in Streamlit secrets!")
    st.stop()

@st.cache_resource(show_spinner="🌱 Loading embedding model...")
def load_embedder():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner="📚 Indexing extinct species database...")
def init_vector_store(_embeddings):
    if not os.path.exists("extinct_birds_data.txt"):
        st.error("File `extinct_birds_data.txt` missing! Please upload it to GitHub.")
        st.stop()

    loader = TextLoader("extinct_birds_data.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return Chroma.from_documents(docs, _embeddings)

@st.cache_resource
def init_green_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.2-1b-preview",
        temperature=0.1
    )

embeddings = load_embedder()
vectorstore = init_vector_store(embeddings)
llm = init_green_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Modern LCEL Prompt & Chain Definition
prompt_template = """Answer the question based ONLY on the following context. If you don't know, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL Chain (Replaces legacy RetrievalQA)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------
# Session State & Cache Logic
# ---------------------------------------------------------
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
if "fresh_queries" not in st.session_state:
    st.session_state.fresh_queries = 0

def get_cached_response(user_query: str, similarity_threshold: float = 0.88):
    if not st.session_state.semantic_cache:
        return None, None, 0.0

    query_vector = embeddings.embed_query(user_query)
    best_match = None
    highest_similarity = -1.0

    for item in st.session_state.semantic_cache:
        similarity = np.dot(query_vector, item["vector"]) / (
            np.linalg.norm(query_vector) * np.linalg.norm(item["vector"])
        )
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = item

    if highest_similarity >= similarity_threshold:
        return best_match["answer"], best_match.get("sources", []), highest_similarity
    
    return None, None, 0.0

def save_to_cache(user_query: str, answer: str, sources):
    query_vector = embeddings.embed_query(user_query)
    st.session_state.semantic_cache.append({
        "original_query": user_query,
        "vector": query_vector,
        "answer": answer,
        "sources": sources
    })

# ---------------------------------------------------------
# Sidebar Dashboard
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=64)
    st.title("🌱 Eco-Dashboard")
    st.caption("Real-time Sustainable Compute Metrics")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Fresh LLM Runs", value=st.session_state.fresh_queries)
    with col2:
        st.metric(label="Cache Hits ⚡", value=st.session_state.cache_hits)

    saved_tokens = st.session_state.cache_hits * 350
    st.markdown(f"""
    <div class="metric-card">
        <span style="color:#059669; font-weight:700;">⚡ Saved Tokens:</span> {saved_tokens:,}<br/>
        <span style="color:#059669; font-weight:700;">🌿 Saved FLOPs:</span> {saved_tokens * 2:.1e} FLOPs<br/>
        <span style="color:#059669; font-weight:700;">🔋 Energy Saved:</span> ~{(st.session_state.cache_hits * 0.002):.3f} Wh
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Cache & Chat", use_container_width=True):
        st.session_state.semantic_cache = []
        st.session_state.messages = []
        st.session_state.cache_hits = 0
        st.session_state.fresh_queries = 0
        st.rerun()

# ---------------------------------------------------------
# UI & Main Chat Loop
# ---------------------------------------------------------
st.markdown("""
<div class="eco-header">
    <h1>🦜 Extinct Bird Species Knowledge Base</h1>
    <p>A Green AI Retrieval-Augmented Generation (RAG) system built with ultra-low token compute.</p>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🦜" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])
        if "badge" in message:
            st.markdown(message["badge"], unsafe_allow_html=True)
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Context Chunks"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {idx+1}:** {doc.page_content}")

if prompt := st.chat_input("Ask about an extinct bird species (e.g., Dodo, Passenger Pigeon)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦜"):
        start_time = time.time()
        
        cached_answer, cached_sources, sim_score = get_cached_response(prompt)

        if cached_answer:
            st.session_state.cache_hits += 1
            latency = (time.time() - start_time) * 1000
            
            st.markdown(cached_answer)
            badge_html = f'<div class="cache-badge">⚡ Cached Hit ({sim_score:.1%} match) • {latency:.0f}ms • 0% Energy</div>'
            st.markdown(badge_html, unsafe_allow_html=True)

            if cached_sources:
                with st.expander("🔍 View Context Chunks"):
                    for idx, doc in enumerate(cached_sources):
                        st.markdown(f"**Chunk {idx+1}:** {doc.page_content}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": cached_answer,
                "badge": badge_html,
                "sources": cached_sources
            })

        else:
            st.session_state.fresh_queries += 1
            with st.spinner("Retrieving facts sustainably..."):
                try:
                    # Run LCEL chain
                    answer = rag_chain.invoke(prompt)
                    sources = retriever.invoke(prompt)
                    
                    latency = time.time() - start_time
                    save_to_cache(prompt, answer, sources)

                    st.markdown(answer)
                    badge_html = f'<div class="green-badge">🌱 Fresh Llama 3.2 1B • {latency:.2f}s</div>'
                    st.markdown(badge_html, unsafe_allow_html=True)

                    if sources:
                        with st.expander("🔍 View Context Chunks"):
                            for idx, doc in enumerate(sources):
                                st.markdown(f"**Chunk {idx+1}:** {doc.page_content}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "badge": badge_html,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error querying model: {e}")
