# fifi.ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This started as a weekend idea: let anyone who writes turn their content into a grounded Q&A assistant backed by their own material. I made the first version public so others could use it as a starting point. It was built with Claude Code, with me making the architecture and design decisions throughout. Key choices were modular components, swappable vector stores (FAISS for local use, Pinecone for cloud), structured JSON logging, and a full test suite from the start.

---

## Features

- Drop `.md` files in `blogs/` and they become the knowledge base
- FAISS (local) or Pinecone (cloud) vector search with OpenAI embeddings
- Streamlit web UI with dark mode and streaming responses
- CLI chatbot for terminal use
- Customizable AI personality via YAML prompt files
- Structured JSON logging and query metrics
- 127 tests

---

## Documentation

- **[CUSTOMIZATION.md](docs/CUSTOMIZATION.md)** - Branding and customization
- **[BLOG_FORMAT.md](docs/BLOG_FORMAT.md)** - Blog writing guide
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deploy to production
- **[prompts/README.md](prompts/README.md)** - AI prompt customization
- **[FRONTEND.md](docs/FRONTEND.md)** - UI features
- **[ROADMAP.md](docs/ROADMAP.md)** - Project phases and timeline

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Blog Posts │────▶│   Embedding  │────▶│   Vector    │
│  (Markdown) │     │   Generator  │     │  Database   │
└─────────────┘     └──────────────┘     │(FAISS/Pine) │
                                          └──────┬──────┘
                                                 │
                    ┌────────────────────────────┘
                    │
                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    User     │────▶│  RAG Engine  │────▶│   OpenAI    │
│   Query     │     │              │     │   API       │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Response   │
                    │ with Sources │
                    └──────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/fehernandespaysan/fifi.ai.git
   cd fifi.ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

5. **Verify setup**
   ```bash
   python examples/verify_setup.py
   ```

### Usage

#### Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Includes dark mode, streaming responses, source citations, and a stats dashboard.

#### Command Line Interface

```bash
python chat.py

# Commands:
# /help     - Show help
# /stats    - View statistics
# /history  - Show conversation
# /clear    - Clear history
# /exit     - Exit
```

#### Example Scripts

```bash
python examples/verify_setup.py          # Verify your setup
python examples/load_and_inspect_blogs.py
python examples/generate_embeddings.py
python examples/interactive_rag_demo.py
```

---

## Technology Stack

- **Language:** Python 3.11+
- **LLM Provider:** OpenAI (GPT-4o-mini / GPT-4o)
- **Vector Database:** FAISS (local) or Pinecone (cloud)
- **Framework:** LangChain
- **UI:** Streamlit
- **Testing:** pytest, pytest-cov
- **Linting:** black, pylint, mypy, bandit

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | Complete | Project structure, configuration, logging |
| Phase 1 | Complete | Blog data handling and vector embeddings |
| Phase 2 | Complete | RAG query engine |
| Phase 3 | Complete | CLI chatbot |
| Phase 4 | Complete | Testing and quality assurance |
| Phase 5 | Deferred | FastAPI backend |
| Phase 6 | Complete | Streamlit UI |
| Phase 7+ | Planned | Advanced features |

**Test Results:** 111/127 tests passing

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss) and [Pinecone](https://www.pinecone.io/)
