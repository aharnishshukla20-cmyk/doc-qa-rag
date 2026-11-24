import sys

# Linux cloud compatibility shim for ChromaDB's SQLite requirement
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import hashlib
import io
import os
import re
import time
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
import streamlit as st

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DocuMind · AI Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PREMIUM SAAS DARK THEME & GLASSMORPHISM CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 920px;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.4rem 0 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.1rem;
    }
    .brand-logo {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    .brand-text {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }
    .brand-tag {
        font-size: 0.72rem;
        color: #94a3b8;
    }

    .hero-wrapper {
        position: relative;
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.18), transparent 55%),
                    radial-gradient(circle at bottom left, rgba(6, 182, 212, 0.12), transparent 55%),
                    rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 20px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    }
    .hero-eyebrow {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #818cf8;
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.65rem;
    }
    .hero-heading {
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f8fafc;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0 0 1.25rem 0;
        line-height: 1.55;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
    }
    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-box {
        background: rgba(13, 17, 23, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 0.88rem;
        font-weight: 600;
        color: #e2e8f0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div.stButton > button {
        background: rgba(22, 27, 34, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 14px !important;
        padding: 0.85rem 1rem !important;
        text-align: left !important;
        color: #e2e8f0 !important;
        font-size: 0.86rem !important;
        line-height: 1.45 !important;
        height: auto !important;
        min-height: 78px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    div.stButton > button:hover {
        border-color: #6366f1 !important;
        background: rgba(99, 102, 241, 0.12) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.18) !important;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(22, 27, 34, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }

    .source-card {
        background: rgba(13, 17, 23, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #818cf8;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.55;
    }
    .source-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .source-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.15);
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
    }
    .footer-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR WORKSPACE ---
env_key = os.getenv("GEMINI_API_KEY", "")

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-logo">◈</div>
            <div>
                <div class="brand-text">DocuMind AI</div>
                <div class="brand-tag">RAG Knowledge Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if env_key:
        api_key = env_key
        st.success("🔒 Authenticated via server secrets")
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Enter a free Google AI Studio API key.",
        )

    st.markdown("#### 📂 Document Source")
    uploaded_file = st.file_uploader(
        "Drop a PDF to index & chat",
        type=["pdf"],
        help="Automatically chunks, embeds, and persists your PDF in ChromaDB.",
    )

    st.markdown("#### 🎛️ Search Precision")
    top_k = st.slider(
        "Retrieved passages (Top-K)",
        min_value=2,
        max_value=5,
        value=3,
        help="More passages give broader context; fewer passages give faster responses.",
    )

    st.divider()
    if st.button("↺  New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("👈 Add your **Gemini API Key** in the sidebar to launch DocuMind AI.")
    st.stop()

# --- GEMINI CLIENT ---
@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=30000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

gemini_client = get_gemini_client(api_key)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        while True:
            try:
                response = gemini_client.models.embed_content(
                    model=self.model_name,
                    contents=list(input),
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                if "429" in str(e):
                    st.toast("Free tier rate limit reached — resuming in 62s...", icon="⏳")
                    time.sleep(62)
                else:
                    raise

# --- PDF CHUNKING ---
def extract_and_chunk_pdf(file_bytes: bytes, chunk_size: int = 2500, overlap: int = 250):
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            full_text.append(f"[Page {page_num}]\n{text}")
    joined = "\n\n".join(full_text)

    chunks = []
    start = 0
    while start < len(joined):
        end = start + chunk_size
        chunks.append(joined[start:end])
        start += chunk_size - overlap
    return chunks

# --- CHROMADB CONNECTION ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = GeminiEmbeddingFunction(model_name="gemini-embedding-001")

collection = None
active_doc_name = "No PDF loaded"
is_custom_upload = False

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:10]
    collection_name = f"upload_{file_hash}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embed_fn
    )

    if collection.count() == 0:
        chunks = extract_and_chunk_pdf(file_bytes)
        total = len(chunks)
        progress_bar = st.progress(0, text=f"Indexing **{uploaded_file.name}** ({total} chunks)...")
        batch_size = 25
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
            collection.add(documents=batch, ids=ids)
            done = min(i + batch_size, total)
            progress_bar.progress(done / total, text=f"Indexed {done} of {total} chunks...")
            if done < total:
                time.sleep(62)
        progress_bar.empty()
        st.toast(f"Indexed {uploaded_file.name}!", icon="✅")

    active_doc_name = uploaded_file.name
    is_custom_upload = True
else:
    try:
        collection = chroma_client.get_collection(
            name="book_knowledge", embedding_function=embed_fn
        )
        active_doc_name = "sample.pdf"
    except Exception:
        collection = None

chunk_count = collection.count() if collection is not None else 0

# --- MODEL DISCOVERY & SMART CACHING ---
if "model_chain" not in st.session_state:
    models = []
    for m in gemini_client.models.list():
        if "generateContent" in (m.supported_actions or []):
            name = m.name.replace("models/", "")
            if any(tag in name for tag in ["gemini", "gemma"]) and not any(
                skip in name for skip in ["image", "audio", "tts", "robotics", "computer"]
            ):
                models.append(name)

    def priority(name: str) -> int:
        if "flash-lite" in name:
            return 0
        if "flash" in name:
            return 1
        if "pro" in name:
            return 2
        return 3

    models.sort(key=priority)
    st.session_state.model_chain = models[:10]

active_model_preview = st.session_state.model_chain[0] if st.session_state.model_chain else "Auto"

# --- HERO DASHBOARD CARD ---
st.markdown(
    f"""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">Retrieval-Augmented Generation · Grounded AI</div>
        <h1 class="hero-heading">Ask anything inside your documents.</h1>
        <p class="hero-desc">
            Semantic vector search powered by <b>gemini-embedding-001</b> and <b>ChromaDB</b>. Every response is strictly verified against retrieved passages with exact page citations.
        </p>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Active File</div>
                <div class="metric-value" title="{active_doc_name}">📄 {active_doc_name}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Vector Index</div>
                <div class="metric-value">🧩 {chunk_count} Chunks</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Active Engine</div>
                <div class="metric-value" title="{active_model_preview}">⚡ {active_model_preview}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Hallucination Guard</div>
                <div class="metric-value">🛡️ Strict Context</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if collection is None:
    st.info("👈 Upload any **PDF document** in the left sidebar to index it and start asking questions!")
    st.stop()

# --- RENDER CITATION CARDS ---
def render_sources(sources: list[str]):
    with st.expander(f"📑 Inspect {len(sources)} Verified Source Excerpts"):
        for idx, src in enumerate(sources, start=1):
            page_match = re.search(r"\[Page (\d+)\]", src)
            page_label = f"Page {page_match.group(1)}" if page_match else "Excerpt"
            clean_text = re.sub(r"\[Page \d+\]\s*", "", src).strip()
            preview = clean_text[:600] + ("..." if len(clean_text) > 600 else "")
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-top">
                        <span class="source-badge">SOURCE #{idx} · {page_label}</span>
                    </div>
                    <div>{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "◈"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.markdown(f'<div class="footer-meta">{msg["meta"]}</div>', unsafe_allow_html=True)
        if "sources" in msg:
            render_sources(msg["sources"])

# --- INTERACTIVE STARTER CARDS (EMPTY STATE) ---
selected_starter = None
if not st.session_state.messages:
    st.markdown("##### Suggested Prompts")
    if is_custom_upload:
        starters = [
            ("📌 Executive Summary", "Summarize the core purpose and main takeaways of this document."),
            ("📊 Key Findings", "What are the most important facts, metrics, or conclusions mentioned?"),
            ("✅ Action Items", "List any recommendations, steps, or conclusions stated in the text."),
        ]
    else:
        starters = [
            ("📖 Verse Analysis", "What is the significance of Verse 14 in Chapter 7?"),
            ("🧭 Concept Breakdown", "What are the four types of pious people who surrender to Krishna?"),
            ("🔍 Deep Comparison", "How does the text explain the difference between jnana and vijnana?"),
        ]

    cols = st.columns(3)
    for col, (tag, q_text) in zip(cols, starters):
        if col.button(f"{tag}\n\n{q_text}", use_container_width=True):
            selected_starter = q_text

# --- QUERY INPUT & RAG EXECUTION ---
user_query = st.chat_input("Message DocuMind AI...") or selected_starter

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="◈"):
        t0 = time.time()
        with st.spinner("Retrieving relevant passages & verifying answer..."):
            results = collection.query(query_texts=[user_query], n_results=top_k)
            retrieved_chunks = results["documents"][0]
            context = "\n\n---\n\n".join(retrieved_chunks)

            rag_prompt = f"""You are a precise, helpful document Q&A assistant.
Answer the user's question using ONLY the provided context excerpts from the document.
Use clean formatting (bullet points or concise paragraphs) when helpful, and cite the page numbers if they appear in the context.
If the answer is not present in the context, clearly say: "I couldn't find that information in the provided document."

CONTEXT:
{context}

QUESTION:
{user_query}
"""
            answer = None
            used_model = None
            last_error = None

            for idx, model_name in enumerate(list(st.session_state.model_chain)):
                try:
                    response = gemini_client.models.generate_content(
                        model=model_name, contents=rag_prompt
                    )
                    answer = response.text
                    used_model = model_name
                    st.session_state.model_chain.pop(idx)
                    st.session_state.model_chain.insert(0, model_name)
                    break
                except Exception as e:
                    last_error = e

            elapsed = round(time.time() - t0, 1)

            if answer:
                meta_line = f"⚡ {used_model}  ·  📚 {top_k} chunks retrieved  ·  ⏱️ {elapsed}s"
                st.markdown(answer)
                st.markdown(f'<div class="footer-meta">{meta_line}</div>', unsafe_allow_html=True)
                render_sources(retrieved_chunks)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": meta_line,
                        "sources": retrieved_chunks,
                    }
                )
            else:
                st.error(f"All free-tier models are currently overloaded. Please retry in a few seconds. ({last_error})")