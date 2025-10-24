# 📝 Blog Post Format Guide

## Quick Start

Create a `.md` file in the `blogs/` folder with this format:

```markdown
---
title: "Your Post Title"
date: 2025-10-24
author: "Your Name"
tags: [topic1, topic2, topic3]
---

# Your Post Title

Your content here...

## Section 1

Content...

## Section 2

More content...
```

## Required Fields

| Field | Format | Example |
|-------|--------|---------|
| `title` | String | "What is RAG?" |
| `date` | YYYY-MM-DD | 2025-10-24 |
| `author` | String | "Jane Doe" |
| `tags` | List | [ai, rag, tutorial] |

## Best Practices

### ✅ DO

- Use clear, descriptive titles
- Break content into sections with `##` headers
- Include code examples when relevant
- Aim for 800-2000 words per post
- Use bullet points and numbered lists
- Add links to external resources

### ❌ DON'T

- Don't skip the frontmatter (---  markers)
- Don't use dates in the future
- Don't create posts shorter than 300 words
- Don't forget to add tags

## Markdown Features

### Headers
```markdown
# H1 - Main Title
## H2 - Section
### H3 - Subsection
```

### Lists
```markdown
- Bullet point
- Another point

1. Numbered item
2. Another item
```

### Code Blocks
````markdown
```python
def hello():
    print("Hello, world!")
```
````

### Links & Images
```markdown
[Link text](https://example.com)
![Image alt text](https://example.com/image.png)
```

## Examples

See the `blogs/example-*.md` files for three different formats:

1. **example-tutorial-format.md** - Tutorial/how-to structure
2. **example-concept-explanation.md** - Technical concept explanation
3. **example-best-practices.md** - Tips/best practices format

## FAQs

**Q: How many blog posts should I start with?**
A: At least 3-5 for a good knowledge base.

**Q: Can I use HTML in markdown?**
A: Yes, but keep it simple. Stick to markdown when possible.

**Q: What happens when I add a new blog?**
A: The system will automatically index it on next query. Or run:
```bash
python scripts/test_embeddings.py
```

**Q: Can I edit existing blogs?**
A: Yes! Update the file, regenerate embeddings if needed.

---

**Ready to write?** Delete the example blogs and start adding your content!
