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

---

## Tips

- **Keep it simple**: Only change branding variables, not code
- **Test locally**: Always test changes before deploying
- **Version control**: Commit after major customizations
- **Back up .env**: But never commit it to git!

---

**Need help?** See `README.md` for setup instructions or `BLOG_FORMAT.md` for blog post formatting.
