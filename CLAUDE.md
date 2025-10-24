# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**This is a TEMPLATE PROJECT** designed for developers to clone and customize as their own RAG chatbot.

This is a production-grade RAG (Retrieval-Augmented Generation) system that enables querying blog content through an AI-powered chatbot using OpenAI's API and FAISS vector database.

**Key Design Principles:**
- **Template-First**: Easy for others to clone and brand
- **5-Minute Setup**: Minimal configuration required
- **Production-Ready**: Real code, not demos
- **Well-Documented**: Comprehensive guides for users

**Current Status**: v1.0.0 Template Release - Ready for public use

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# Verify setup
python scripts/setup.py
```

### Running the Application

**Web Interface (Recommended)**
```bash
streamlit run streamlit_app.py
# Or use launcher script
./run_web.sh
```

**CLI Chatbot**
```bash
python chat.py
```

**Test Scripts**
```bash
# Test blog loading
python scripts/test_blog_loading.py

# Generate embeddings
python scripts/test_embeddings.py

# Test RAG pipeline
python scripts/test_rag.py
```

### Testing

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_blog_loader.py

# Run with verbose output
pytest -v

# Run only unit tests
pytest -m unit

# Run without coverage checking
pytest --no-cov

# Generate coverage report
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
pylint src/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## Architecture

### System Design

Fifi.ai follows a layered architecture:

1. **Data Layer** (`src/blog_loader.py`)
   - Loads markdown blog posts from `blogs/` directory
   - Parses YAML frontmatter for metadata
   - Validates blog structure

2. **Embeddings Layer** (`src/embeddings_manager.py`)
   - Chunks blog content into 500-token pieces
   - Generates embeddings using OpenAI's `text-embedding-3-small`
   - Manages FAISS vector index for similarity search
   - Handles save/load of index to disk

3. **RAG Engine** (`src/rag_engine.py`)
   - Core query processing pipeline
   - Retrieves top-5 relevant chunks via vector search
   - Constructs prompts with retrieved context
   - Calls OpenAI API (GPT-4o-mini by default)
   - Returns answers with source citations

4. **User Interfaces**
   - **Streamlit UI** (`streamlit_app.py`): Modern web interface with chat, sources, and statistics
   - **CLI** (`src/cli_chatbot.py`): Terminal-based chatbot with Rich formatting

5. **Infrastructure**
   - **Config** (`src/config.py`): Environment-specific configuration management
   - **Logger** (`src/logger.py`): Structured JSON logging with correlation IDs

### Data Flow

```
User Query → RAG Engine → Vector Search (FAISS) → Top-K Chunks
                              ↓
            Context Assembly → OpenAI API → Response with Sources
```

### Key Design Decisions

**Vector Database**: FAISS (local) for MVP, designed to migrate to Pinecone for scale
- Fast similarity search (<100ms p95)
- No external dependencies for development
- Save/load to disk for persistence

**LLM Model**: GPT-4o-mini by default
- Cost-effective: $0.00015/1K input tokens, $0.0006/1K output tokens
- Quality sufficient for most queries
- Can upgrade to GPT-4o for complex queries

**Chunking Strategy**: 500 tokens per chunk, 50 token overlap
- Balances context size vs retrieval precision
- Overlap ensures no information loss at boundaries

**Security-First**: Following `agent.md` standards
- All inputs validated
- No secrets in code (environment variables only)
- Structured logging without sensitive data
- Generic error messages to users, detailed logs server-side

## Important Implementation Details

### Testing Philosophy

- **Mock all external API calls** to avoid costs and ensure deterministic tests
- Target: >80% code coverage overall, >90% for critical paths (blog_loader, embeddings_manager, rag_engine)
- Current status: 127 tests total, 120 passing (94.5%)
- Use `pytest` fixtures extensively for setup/teardown

### Embeddings Management

The embeddings system is designed for incremental updates:
- `add_documents()`: Add new blogs to existing index
- `save()` / `load()`: Persist index to `data/faiss_index.faiss` and `data/faiss_metadata.pkl`
- Vector search uses cosine similarity
- Always retrieve top-5 chunks by default

### Configuration Pattern

Use `src/config.py` for all environment-specific settings:
- Development: Verbose logging, lower rate limits
- Production: JSON logging, strict validation
- Test: Mock API calls, fast timeouts

### Logging Standards

All logs must include:
- `correlation_id`: Track requests end-to-end
- `timestamp`: ISO 8601 format
- `level`: DEBUG/INFO/WARNING/ERROR/CRITICAL
- `metadata`: Context-specific information (tokens, latency, etc.)

Never log:
- API keys or secrets
- Full user queries (use hashed IDs in production)
- Sensitive personal information

### Cost Tracking

Every LLM call logs:
- Input tokens
- Output tokens
- Estimated cost (based on current OpenAI pricing)
- Model used

This enables cost monitoring and optimization.

## Common Tasks

### Adding New Blog Posts

1. Create markdown file in `blogs/` with YAML frontmatter:
```markdown
---
title: "Your Title"
date: 2025-10-24
author: "Your Name"
tags: [ai, rag]
---

Your content here...
```

2. Regenerate embeddings:
```bash
python scripts/test_embeddings.py
```

3. The index will auto-update on next query in the apps

### Modifying RAG Parameters

Edit constants in `src/rag_engine.py`:
- `DEFAULT_MODEL`: Change LLM model
- `DEFAULT_TEMPERATURE`: Adjust creativity (0.0-1.0)
- `DEFAULT_TOP_K`: Number of chunks to retrieve
- `DEFAULT_MAX_TOKENS`: Max response length

### Adding New Tests

Follow the pattern in `tests/`:
- Use `@pytest.fixture` for shared setup
- Mock OpenAI API calls with `pytest-mock`
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Add markers: `@pytest.mark.unit` or `@pytest.mark.integration`

## Development Standards

All code must follow `agent.md` guidelines:

1. **Security First**
   - Validate all inputs
   - Use environment variables for secrets
   - Generic error messages (log details server-side)

2. **Observability**
   - Structured logging for all operations
   - Track metrics: latency, tokens, costs
   - Include correlation IDs

3. **Code Quality**
   - Type hints throughout
   - >80% test coverage
   - Black formatting
   - Pylint compliance

4. **Production Ready**
   - Error handling with retries
   - Health check endpoints (when API is built)
   - Rate limiting
   - Graceful degradation

## Project Status & Roadmap

See `ROADMAP.md` for detailed timeline. Current phase:

✅ **Completed**:
- Phase 0: Foundation (config, logging)
- Phase 1: Blog loading & embeddings
- Phase 2: RAG engine
- Phase 3: CLI chatbot
- Phase 4: Testing (127 tests)
- Phase 6: Streamlit UI

⏸️ **Deferred**:
- Phase 5: FastAPI backend (will build if needed)

📅 **Next**:
- Create real blog content
- Deploy to Streamlit Cloud
- Phase 7+: Avatar, insights, advanced features

## Deployment

**Current**: Local development only

**Planned**:
- Streamlit Cloud (free tier) for web UI
- Vercel (free tier) for API (if needed)
- FAISS index backed by S3

## Troubleshooting

**"No module named 'src'"**
- Ensure virtual environment is activated
- Run from project root directory

**"OPENAI_API_KEY not set"**
- Copy `.env.example` to `.env`
- Add your OpenAI API key

**Tests failing with API errors**
- Ensure tests mock OpenAI API calls
- Check `tests/conftest.py` for fixtures

**Slow vector search**
- Check FAISS index size
- Consider migration to IVF index for >100K vectors

## Key Files Reference

- `streamlit_app.py`: Web UI entry point
- `chat.py`: CLI entry point
- `src/rag_engine.py`: Core RAG logic
- `src/embeddings_manager.py`: Vector database operations
- `src/blog_loader.py`: Blog ingestion
- `agent.md`: Development standards (security, monitoring, quality)
- `ROADMAP.md`: Detailed project plan and timeline
- `pytest.ini`: Test configuration
- `requirements.txt`: All dependencies

## Additional Resources

- Architecture details: `ROADMAP.md` sections on Architecture, Technology Stack
- Security guidelines: `agent.md`
- Test summary: `docs/TEST_SUMMARY.md`
- Deployment guide: `docs/DEPLOYMENT.md` (when created)
