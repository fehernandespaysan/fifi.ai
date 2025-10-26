# Fifi.ai Prompts Configuration

This directory contains all prompt templates used by the Fifi.ai RAG chatbot. Prompts are stored as YAML files for easy editing and version control.

## 📁 Files

### Core Prompts
- **`system_prompt.yaml`** - Main personality and behavior of Fifi assistant
- **`user_template.yaml`** - Template for formatting RAG queries with context
- **`fallback_prompt.yaml`** - Response template when no relevant context is found

## 🎯 Editing Prompts

### System Prompt (`system_prompt.yaml`)
Defines Fifi's personality, tone, and overall behavior.

**What to customize:**
- Assistant name and role
- Tone and communication style
- Core instructions and guidelines
- Knowledge domain focus

**Example:**
```yaml
prompt: |
  You are Fifi, a helpful AI assistant that answers questions about AI engineering...
```

### User Template (`user_template.yaml`)
Template used when Fifi has retrieved relevant context from the knowledge base.

**Placeholders:**
- `{context}` - Retrieved text chunks from vector database
- `{question}` - User's original query

**What to customize:**
- Instructions for how to use context
- Citation requirements
- Response format preferences
- Tone for knowledge-based answers

**Example:**
```yaml
template: |
  Use the following context from blog posts to answer the user's question.

  Context:
  {context}

  Question: {question}
```

### Fallback Prompt (`fallback_prompt.yaml`)
Used when user queries don't match any content in the knowledge base.

**Placeholders:**
- `{query}` - User's original query

**What to customize:**
- Knowledge base topic list
- Greeting response style
- How to redirect users to covered topics
- Fallback system prompt

**Example:**
```yaml
template: |
  The user asked: "{query}"

  I don't have specific information about this in my knowledge base...
```

## 🔧 How Prompts are Loaded

1. **Automatic Loading** - The `PromptLoader` class automatically loads YAML files at startup
2. **Caching** - Prompts are cached in memory for performance
3. **Fallback** - If YAML loading fails, hardcoded fallback prompts are used
4. **Hot Reloading** - Call `loader.reload()` to refresh prompts without restart (dev mode)

## 🎨 Creating Custom Prompts

You can create custom prompt variations for testing or different use cases:

### Option 1: Custom Directory (Not committed to git)
```bash
mkdir prompts/custom
cp prompts/system_prompt.yaml prompts/custom/my_system_prompt.yaml
# Edit prompts/custom/my_system_prompt.yaml
```

### Option 2: Local Override Files (Not committed to git)
```bash
cp prompts/system_prompt.yaml prompts/system_prompt.local.yaml
# Edit prompts/system_prompt.local.yaml
```

Files matching these patterns are ignored by git:
- `prompts/custom/*.yaml`
- `prompts/*.local.yaml`
- `prompts/*_custom.yaml`

## 🧪 Testing Prompt Changes

### Method 1: Direct File Edit
1. Edit the YAML file (e.g., `system_prompt.yaml`)
2. Restart the application
3. Test your changes in the chatbot

### Method 2: Python API
```python
from src.prompt_loader import PromptLoader

loader = PromptLoader()
system_prompt = loader.get_system_prompt()
print(system_prompt)

# Reload after editing
loader.reload()
system_prompt = loader.get_system_prompt()  # Gets fresh version
```

### Method 3: Custom Loader in Code
```python
from src.rag_engine import RAGEngine
from src.prompt_loader import PromptLoader
from pathlib import Path

# Load from custom directory
custom_loader = PromptLoader(prompts_dir=Path("prompts/custom"))
engine = RAGEngine(prompt_loader=custom_loader)
```

## 📝 Best Practices

### DO ✅
- **Test changes thoroughly** before committing
- **Keep prompts focused** - one clear purpose per prompt
- **Use markdown formatting** in prompts for better readability
- **Document your changes** in git commits
- **Validate YAML syntax** before saving (use a YAML validator)
- **Keep placeholder names consistent** (`{context}`, `{question}`, etc.)

### DON'T ❌
- **Don't remove required placeholders** (`{context}`, `{question}`, `{query}`)
- **Don't commit sensitive information** in prompts
- **Don't make prompts too long** (affects token usage and latency)
- **Don't use special characters** that might break YAML parsing
- **Don't override core files** for personal testing (use custom/ instead)

## 🎓 Prompt Engineering Tips

### System Prompts
- Start with who the assistant is ("You are...")
- Define the knowledge source
- Set clear behavioral guidelines
- Specify response format preferences

### User Templates
- Provide clear context boundaries
- Give explicit instructions on how to use context
- Specify citation requirements
- Set tone and style expectations

### Fallback Prompts
- Be friendly and conversational
- Acknowledge the user's question
- Gently guide to available topics
- Invite engagement with covered material

## 🔍 Troubleshooting

**Prompts not loading?**
- Check YAML syntax (use https://yamlchecker.com)
- Verify file is in the correct directory
- Check file permissions
- Look at logs for specific error messages

**Changes not taking effect?**
- Restart the application completely
- Clear any application caches
- Verify you edited the correct file (not a backup or copy)

**Unexpected AI behavior?**
- Review prompt for clarity and specificity
- Check for conflicting instructions
- Test with simple queries first
- Validate placeholder formatting

## 📚 Additional Resources

- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [YAML Syntax Reference](https://yaml.org/spec/1.2.2/)
- [Fifi.ai Documentation](../docs/)

## 💡 Examples

### Example: Making Fifi More Technical
```yaml
# prompts/system_prompt.yaml
prompt: |
  You are Fifi, an expert AI engineering assistant specialized in RAG systems,
  vector databases, and production ML deployments.

  Your knowledge comes from curated technical blog posts and research papers. Always:
  - Provide technically accurate, detailed answers with code examples
  - Reference specific implementations and architectures
  - Use industry-standard terminology
  - Include performance considerations and best practices
  ...
```

### Example: Making Fifi More Conversational
```yaml
# prompts/system_prompt.yaml
prompt: |
  You are Fifi, a friendly AI assistant who loves explaining AI concepts in
  simple, relatable terms.

  You learn from blog posts and articles about AI. You always:
  - Use analogies and everyday examples
  - Break down complex topics into bite-sized pieces
  - Keep things conversational and approachable
  - Ask clarifying questions when needed
  ...
```

---

**Questions or suggestions?** Open an issue or submit a pull request!
