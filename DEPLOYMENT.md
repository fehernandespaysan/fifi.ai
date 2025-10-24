# 🚀 Deployment Guide

## Option 1: Streamlit Cloud (Recommended - FREE!)

### Prerequisites
- GitHub account
- OpenAI API key

### Steps (5 minutes)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready to deploy"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"

3. **Configure**
   - Repository: Select your repo
   - Branch: `main`
   - Main file: `streamlit_app.py`

4. **Add Secrets**
   Click "Advanced settings" → "Secrets":
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   APP_NAME = "Your Bot Name"
   WELCOME_MESSAGE = "Your message..."
   ```

5. **Deploy!**
   - Click "Deploy"
   - Wait 2-3 minutes
   - Get your public URL: `https://yourapp.streamlit.app`

---

## Option 2: Docker (Self-Hosted)

```bash
# Create Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]
EOF

# Build and run
docker build -t rag-chatbot .
docker run -p 8501:8501 --env-file .env rag-chatbot
```

Visit `http://localhost:8501`

---

## Option 3: Vercel (API + UI)

If you build the FastAPI backend (Phase 5):

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

---

## Environment Variables

### Required
```bash
OPENAI_API_KEY=sk-...
```

### Recommended to Customize
```bash
APP_NAME="Your Bot"
WELCOME_MESSAGE="Your message"
EXAMPLE_QUESTION_1="Your question?"
# ... etc
```

### Optional
```bash
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
CHUNK_SIZE=500
```

---

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not set"
Add it to Streamlit secrets or `.env`

### "No blogs found"
Ensure `blogs/` folder has `.md` files

---

## Cost Estimates

**Streamlit Cloud:** FREE
**OpenAI API:** ~$0.0003 per query (GPT-4o-mini)

**Example monthly costs:**
- 100 users, 10 queries each = 1,000 queries = ~$0.30
- 1,000 users, 10 queries each = 10,000 queries = ~$3.00

Very affordable! 💰
