# Architecture

## Overview

fifi.ai is a RAG (Retrieval-Augmented Generation) pipeline that turns a directory of Markdown blog posts into a grounded Q&A assistant. A user asks a question; the system retrieves the most relevant passages from the author's own content; those passages are injected into the LLM prompt as context. The LLM's answer is therefore grounded in the source material rather than in its general training.

```
blogs/ (Markdown)
     │
     ▼
 BlogLoader          — parses frontmatter, chunks text
     │
     ▼
 EmbeddingsManager   — calls OpenAI embeddings API, stores vectors
     │
     ▼
 VectorStore         — FAISS (local) or Pinecone (cloud)
     │
     ▼
 RAGEngine           — retrieves top-k chunks, builds prompt, calls GPT
     │
     ▼
 Streamlit UI / CLI  — streams response back to user
```

---

## Key Design Decisions

### Swappable vector stores (FAISS vs. Pinecone)

Both stores implement the same `VectorStore` abstract base. `EmbeddingsManager` talks only to that interface, so swapping backends requires only a config change (`VECTOR_STORE_TYPE=pinecone`), not code changes.

**Why FAISS as the default:** Zero infrastructure, no account needed, works offline. Suitable for a single author's blog corpus (tens of thousands of vectors). The index is an in-process file on disk.

**Why Pinecone as the alternative:** When the corpus grows or the service needs to run statelessly (e.g., on a serverless host that can't persist files between requests), Pinecone manages the index externally. The tradeoff is a network round-trip on every query and an external dependency.

**What I'd do at scale:** For a production multi-tenant system I'd use pgvector inside the existing Postgres instance rather than running a separate vector service. It keeps the stack simpler and lets you JOIN embeddings against metadata in a single query.

### Chunking strategy

Text is split by word count with overlap (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50` by default). Overlap prevents a sentence that straddles a chunk boundary from being lost entirely.

**The tradeoff I'd revisit:** Word-count chunking ignores document structure. A heading + its paragraph is more semantically coherent than an arbitrary 500-word window. For a future version I'd chunk by Markdown section (split on `##` headers) and fall back to word-count only when a section is too large.

### No reranker

Retrieved chunks are ranked by cosine similarity only. A cross-encoder reranker (e.g. Cohere Rerank or a local `cross-encoder/ms-marco-MiniLM`) would improve precision — especially for short queries where the top embedding hit is not always the most relevant passage. I left it out because it adds latency and an extra API dependency, and the corpus size doesn't justify it yet.

### Structured JSON logging

Every significant operation logs a JSON object with a `correlation_id` field. This means a single user query can be traced end-to-end across retrieval, embedding, and generation log lines by grepping for the same ID. In production this feeds directly into any log aggregation system (Datadog, CloudWatch, etc.) without a parsing step.

### Prompt customization via YAML

System prompts and fallback responses are loaded from `prompts/*.yaml` rather than hardcoded in Python. This lets the AI personality be updated without touching the code, and the YAML files can be swapped per deployment (a cooking blog and a security blog can share the same engine with different prompts).

---

## What I'd Change for Production

| Concern | Current | Production approach |
|---|---|---|
| Vector store | FAISS file on disk | pgvector in Postgres |
| Auth | None | API key or OAuth in front of Streamlit |
| Rate limiting | None | Token bucket at the request layer |
| Observability | JSON logs | Logs + span tracing (OpenTelemetry) |
| Eval | Keyword-match script | LLM-as-judge scoring on a fixed question set |
| Multi-tenancy | Single author | Namespace per tenant in the vector store |

---

## Repository Layout

```
src/
  blog_loader.py        — Markdown parsing and chunking
  embeddings_manager.py — embedding generation and vector store coordination
  rag_engine.py         — retrieval, prompt assembly, LLM call
  config.py             — pydantic-settings config (env vars + .env)
  logger.py             — structured JSON logging
  prompt_loader.py      — loads YAML prompt files
  vector_store/
    base.py             — VectorStore abstract interface
    faiss_store.py      — FAISS implementation
    pinecone_store.py   — Pinecone implementation
    factory.py          — creates the right store from config

examples/
  verify_setup.py       — checks env, deps, and API connectivity
  eval_rag_quality.py   — runs keyword-grounded eval against the live index
  generate_embeddings.py
  interactive_rag_demo.py

prompts/                — YAML system prompts (swap without code changes)
blogs/                  — Markdown source content
tests/                  — pytest suite, 127 tests, all mocked (no API calls)
```
