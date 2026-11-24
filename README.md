# ◈ DocuMind AI — Grounded Document Q&A Assistant (RAG)

A full-stack **Retrieval-Augmented Generation (RAG)** application built with Python, Google Gemini (`google-genai`), ChromaDB, and Streamlit. DocuMind AI allows users to upload PDF documents, index them into persistent local vector collections, and ask natural-language questions backed by strict anti-hallucination guardrails and page-level source citations.

---

## ✨ Key Features

- **Multi-Document Persistent Vector Storage:** Uses **ChromaDB** (`PersistentClient`) with content-hash isolation (`MD5`) so multiple PDFs remain indexed on disk without re-embedding on subsequent sessions.
- **Page-Aware Chunking Pipeline:** Extracts PDF text via `pypdf` with embedded `[Page N]` metadata markers and overlapping character windows (`2,500` chars / `250` overlap) to preserve cross-paragraph context.
- **Resilient Multi-Model Failover & Caching:** Dynamically discovers available `generateContent` models, routes across Flash-Lite, Flash, Pro, and Gemma pools during Free Tier `503` high-demand spikes, and promotes healthy models in session state for low-latency follow-ups.
- **Rate-Limit-Safe Batch Indexing:** Automatically paces embedding requests (`gemini-embedding-001`) with adaptive backoff to stay within Free Tier RPM/TPM quotas on large documents.
- **Strict Grounding & Verifiable Citations:** Instructs the LLM to answer strictly from retrieved passages, refuse out-of-scope queries, and display expandable source excerpts with exact page numbers.
- **Dual Interfaces:** Includes both a modern **Streamlit Web UI (`app.py`)** with configurable `Top-K` retrieval precision and an interactive **CLI assistant (`step4_rag_qa.py`)**.

---

## 🏗️ System Architecture

```text
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐
│   PDF File   │────▶│  pypdf Text Extractor   │────▶│  Page-Tagged Chunks      │
└──────────────┘     │  (2500 chars / 250 ovlp)│     │  ([Page 1], [Page 2]...) │
                     └─────────────────────────┘     └────────────┬─────────────┘
                                                                  │
                                                gemini-embedding-001 (Batched)
                                                                  ▼
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐
│  User Query  │────▶│  ChromaDB Similarity    │◀───▶│  Persistent Vector Store │
└──────┬───────┘     │  Search (Top-K Chunks)  │     │  (./chroma_db)           │
       │             └────────────┬────────────┘     └──────────────────────────┘
       │                          │
       ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  Grounded Prompt + Multi-Model Failover (Gemini / Gemma) ──▶ Cited Response   │
└───────────────────────────────────────────────────────────────────────────────┘