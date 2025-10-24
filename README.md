# 🤖 RAG Chatbot Template

**Production-ready RAG chatbot you can customize in 5 minutes**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 💡 **This is a template!** Clone it, add your blogs, customize the branding, and deploy your own AI chatbot.

---

## ✨ Features

- 🚀 **5-Minute Setup** - Clone → Configure → Deploy
- 🎨 **Easy Branding** - Customize name, colors, messages via `.env`
- 📝 **Just Add Content** - Drop `.md` files in `blogs/` folder
- 🤖 **Production RAG** - FAISS vector search + OpenAI embeddings
- 💬 **Dual Interface** - Streamlit web UI + CLI chatbot
- 🔒 **Security Built-In** - API key sanitization, input validation
- 📊 **Observability** - Structured logging, metrics tracking
- ✅ **127 Tests Included** - Production-ready code quality
- 📱 **Deploy Anywhere** - Streamlit Cloud (free), Docker, Vercel

---

## 🚀 Quick Start

### 1. Clone & Install (2 min)

```bash
git clone https://github.com/yourusername/rag-chatbot-template
cd rag-chatbot-template
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (1 min)

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env - add your OpenAI API key
# Also customize: APP_NAME, WELCOME_MESSAGE, EXAMPLE_QUESTIONS
```

### 3. Add Your Content (1 min)

```bash
# Delete example blogs
rm blogs/example-*.md

# Add your own markdown files
cp your-blog.md blogs/
```

### 4. Run! (1 min)

```bash
# Web UI
streamlit run streamlit_app.py

# Or CLI
python chat.py
```

Visit `http://localhost:8501` - Your branded chatbot is live!

---

## 📚 What You Get

- **Production RAG System** - Not a toy demo, real production code
- **Full Test Suite** - 127 tests, 78% coverage
- **Example Blog Posts** - 3 different formats to guide you
- **Comprehensive Docs** - Setup, customization, deployment guides
- **Best Practices** - Security, logging, error handling built-in

---

## 📖 Documentation

- **[CUSTOMIZATION.md](CUSTOMIZATION.md)** - How to brand and customize
- **[BLOG_FORMAT.md](BLOG_FORMAT.md)** - How to write blog posts
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deploy to Streamlit Cloud (free!)
- **[ROADMAP.md](ROADMAP.md)** - Project roadmap and future plans
- **[CLAUDE.md](CLAUDE.md)** - Development guide for contributors

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Blog Posts │────▶│   Embedding  │────▶│   Vector    │
│  (Markdown) │     │   Generator  │     │  Database   │
└─────────────┘     └──────────────┘     │   (FAISS)   │
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

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fifi.ai.git
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
   python scripts/setup.py
   ```

### Using Fifi.ai

#### Option 1: Web Interface (Streamlit) ⭐ Recommended

The easiest way to use Fifi.ai is through the beautiful web interface:

```bash
# Run the Streamlit web app
streamlit run streamlit_app.py

# Or use the launcher script
./run_web.sh
```

Then open your browser to `http://localhost:8501`

**Features:**
- 💬 Beautiful chat interface
- 📊 Real-time statistics dashboard
- 📚 Source citations with expandable details
- 🎨 Markdown rendering
- 🔄 Conversation management

#### Option 2: Command Line Interface (CLI)

For terminal lovers, use the CLI chatbot:

```bash
# Start the CLI chatbot
python chat.py

# Available commands:
# /help     - Show help
# /stats    - View statistics
# /history  - Show conversation
# /clear    - Clear history
# /exit     - Exit chatbot
```

#### Option 3: Test Scripts

Run individual test scripts to explore functionality:

```bash
# Test blog loading
python scripts/test_blog_loading.py

# Generate embeddings from blog posts
python scripts/test_embeddings.py

# Test the complete RAG pipeline
python scripts/test_rag.py
```

---

## 📚 Documentation

- [ROADMAP.md](ROADMAP.md) - Detailed development roadmap and timeline
- [agent.md](agent.md) - Development standards and best practices
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute (coming soon)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture (coming soon)

---

## 🛠️ Technology Stack

### Core Technologies
- **Language:** Python 3.11+
- **LLM Provider:** OpenAI (GPT-4o-mini / GPT-4o)
- **Vector Database:** FAISS (local) → Pinecone (scale)
- **Framework:** LangChain
- **Web API:** FastAPI
- **UI:** Streamlit

### Testing & Quality
- **Testing:** pytest, pytest-cov
- **Linting:** black, pylint, mypy
- **Security:** bandit

---

## 📊 Project Status

**Current Phase:** ✅ Core Platform Complete - Content Creation & Deployment Prep

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Project structure, configuration, logging |
| **Phase 1** | ✅ Complete | Blog data handling and vector embeddings |
| **Phase 2** | ✅ Complete | RAG query engine |
| **Phase 3** | ✅ Complete | CLI chatbot |
| **Phase 4** | ✅ Complete | Testing & quality assurance (127 tests, 94.5% passing) |
| Phase 5 | ⏸️ Deferred | FastAPI backend (will build if needed) |
| **Phase 6** | ✅ Complete | Streamlit UI (production-ready) |
| Phase 7+ | 📅 Planned | Avatar, insights, and advanced features |

**Test Results:** 120/127 tests passing, ~75% code coverage
**Next Steps:** Creating real blog content, then deploying to Streamlit Cloud

See [ROADMAP.md](ROADMAP.md) for detailed timeline and [docs/TEST_SUMMARY.md](docs/TEST_SUMMARY.md) for test details.

---

## 💡 Use Cases

- **Personal Knowledge Base** - Query your blog posts and notes through an AI interface
- **Learning Platform** - Understand how production RAG systems work
- **Documentation Assistant** - Build a chatbot for your documentation
- **Content Discovery** - Help readers find relevant content from your blog
- **Portfolio Project** - Demonstrate AI engineering skills

---

## 🤝 Contributing

We welcome contributions! This project is designed for learning, so feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon).

---

## 📈 Roadmap Highlights

### Q4 2025
- ✅ Phase 0: Foundation & Setup
- 🎯 Phase 1-3: Core RAG functionality
- 🎯 Phase 4: Testing & QA
- 🎯 Phase 5-6: Web API & UI

### Q1 2026
- 🎯 Avatar implementation
- 🎯 Advanced features
- 🎯 Community growth

See [ROADMAP.md](ROADMAP.md) for complete details.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- Inspired by the open-source AI community

---

## 📞 Support & Questions

- 📧 **Email:** fernanda@example.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/fifi.ai/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/yourusername/fifi.ai/discussions)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Built with ❤️ for the AI Engineering community**
