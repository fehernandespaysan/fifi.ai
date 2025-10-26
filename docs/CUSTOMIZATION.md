# 🎨 Customization Guide

Make this RAG chatbot your own in minutes!

## Quick Customization (2 minutes)

### 1. Edit `.env` File

```bash
# Copy the example
cp .env.example .env

# Edit these lines in .env:
APP_NAME="Your Bot Name"
APP_TAGLINE="Your Tagline Here"
WELCOME_TITLE="Welcome to YourBot"
WELCOME_MESSAGE="Your custom welcome message..."

# Customize example questions:
EXAMPLE_QUESTION_1="Your question 1?"
EXAMPLE_QUESTION_2="Your question 2?"
EXAMPLE_QUESTION_3="Your question 3?"
EXAMPLE_QUESTION_4="Your question 4?"
```

### 2. Add Your Content

```bash
# Delete example blogs
rm blogs/example-*.md

# Add your own .md files
cp your-blog-post.md blogs/
```

### 3. Run!

```bash
streamlit run streamlit_app.py
```

That's it! Your branded chatbot is ready.

---

## Advanced Customization

### Change Colors/Styling

Edit `streamlit_app.py`, find the `<style>` block (around line 44):

```python
st.markdown("""
<style>
    /* Change primary color */
    :root {
        --primary-color: #your-color-here;
    }

    /* Customize welcome screen */
    .welcome-title {
        color: #your-color;
    }
</style>
""", unsafe_allow_html=True)
```

### Adjust RAG Parameters

Edit `.env`:

```bash
# Retrieval settings
VECTOR_SEARCH_TOP_K=5        # Number of chunks to retrieve
CHUNK_SIZE=500               # Size of text chunks
CHUNK_OVERLAP=50             # Overlap between chunks

# LLM settings
OPENAI_MODEL=gpt-4o-mini     # or gpt-4o for better quality
OPENAI_TEMPERATURE=0.7       # 0=factual, 2=creative
```

### Change App Icon

Edit `streamlit_app.py`, line ~38:

```python
st.set_page_config(
    page_icon="🤖",  # Change this emoji or use a URL to an image
    ...
)
```

### Customize AI Prompts 🎯 NEW!

Edit how Fifi thinks, talks, and responds by modifying YAML files in `prompts/`:

```bash
# Edit Fifi's personality
nano prompts/system_prompt.yaml

# Edit how Fifi formats answers with context
nano prompts/user_template.yaml

# Edit how Fifi handles out-of-scope questions
nano prompts/fallback_prompt.yaml
```

**Quick examples:**

**Make Fifi more technical:**
```yaml
# prompts/system_prompt.yaml
prompt: |
  You are Fifi, an expert AI engineering assistant specialized in
  RAG systems and production ML deployments.

  Always provide technically accurate answers with code examples...
```

**Make Fifi more conversational:**
```yaml
# prompts/system_prompt.yaml
prompt: |
  You are Fifi, a friendly AI assistant who loves explaining AI
  concepts in simple, relatable terms.

  Use analogies and everyday examples...
```

**Change knowledge base topics:**
```yaml
# prompts/fallback_prompt.yaml
knowledge_base_topics:
  - "Machine Learning"
  - "Data Science"
  - "Python Programming"
```

**For detailed prompt customization guide**, see [`prompts/README.md`](../prompts/README.md)

---

## Tips

- **Keep it simple**: Only change branding variables, not code
- **Test locally**: Always test changes before deploying
- **Version control**: Commit after major customizations
- **Back up .env**: But never commit it to git!
- **Edit prompts safely**: Create custom variants in `prompts/custom/` for testing

---

**Need help?**
- Setup instructions: `README.md`
- Blog formatting: `BLOG_FORMAT.md`
- Prompt editing: `prompts/README.md`
