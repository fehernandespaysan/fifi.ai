# Fifi.ai Development Agent Guidelines

This document outlines the high standards we follow when building Fifi.ai. Claude Code should use these guidelines to ensure all code maintains production-quality standards for security, monitoring, and code quality.

---

## 🎯 Core Principles

- **Security First**: Never trust user input. Always validate and sanitize.
- **Observability**: Build monitoring and logging into every feature from day one.
- **Production Ready**: Code should be deployable and maintainable from the start.
- **Responsible AI**: Handle sensitive data carefully, especially API keys and user information.

---

## 🔐 Security Requirements

### 1. Input Validation & Sanitization

**Every external input must be validated:**

```python
# ✅ Good: Validate all inputs
def validate_user_message(message: str) -> str:
    if not isinstance(message, str):
        raise ValueError("Message must be a string")
    if len(message) > 5000:
        raise ValueError("Message exceeds maximum length")
    if len(message) < 1:
        raise ValueError("Message cannot be empty")
    return message.strip()
```

**Rules:**
- Validate data type, length, and format
- Use whitelisting (allow specific patterns) instead of blacklisting
- Never directly concatenate user input into queries or commands
- Sanitize HTML/JavaScript output to prevent XSS attacks

### 2. Environment Variables & Secrets

**Never hardcode secrets. Ever.**

```python
# ✅ Good: Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY not set in environment")
```

**Rules:**
- Store all API keys, tokens, and credentials in `.env` files (never commit to git)
- Use `.gitignore` to exclude `.env` files
- For production, use AWS Secrets Manager, Vault, or similar
- Rotate secrets regularly
- Never log API keys or sensitive tokens

### 3. Error Handling Without Information Leakage

**Never expose stack traces or sensitive details to users:**

```python
# ❌ Bad: Leaks sensitive info
try:
    result = rag_system.query(user_input)
except Exception as e:
    return {"error": str(e)}  # Shows SQL errors, stack traces!

# ✅ Good: Generic error message, log details server-side
import logging
logger = logging.getLogger(__name__)

try:
    result = rag_system.query(user_input)
except Exception as e:
    logger.error(f"RAG query failed: {e}", exc_info=True)
    return {"error": "Something went wrong. Please try again."}
```

### 4. Secure Logging

**Log for debugging, but never log sensitive data:**

```python
# ❌ Bad: Logs user credentials
logger.info(f"User login: {username} with password {password}")

# ✅ Good: Logs action without sensitive data
logger.info(f"User login attempt for {username}")
logger.debug(f"Login result: {'success' if is_valid else 'failed'}")
```

**Rules:**
- Never log passwords, API keys, or tokens
- Redact sensitive fields (emails, SSNs, credit cards)
- Use structured logging (JSON format) for easier monitoring
- Include correlation IDs to trace requests through the system

### 5. HTTP Headers & HTTPS

**Always use HTTPS. Set security headers:**

```python
# ✅ Good: Use Helmet-like headers (Python)
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response
```

---

## 📊 Monitoring & Observability

### 1. Structured Logging

**Every function should emit clear, contextual logs:**

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def query_rag_system(user_id: str, query: str):
    start_time = datetime.utcnow()
    correlation_id = request.headers.get('X-Correlation-ID', str(uuid4()))
    
    logger.info(json.dumps({
        "event": "rag_query_started",
        "user_id": user_id,
        "query_length": len(query),
        "correlation_id": correlation_id,
        "timestamp": start_time.isoformat()
    }))
    
    try:
        result = rag.query(query)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(json.dumps({
            "event": "rag_query_completed",
            "user_id": user_id,
            "status": "success",
            "duration_seconds": duration,
            "correlation_id": correlation_id
        }))
        return result
    
    except Exception as e:
        logger.error(json.dumps({
            "event": "rag_query_failed",
            "user_id": user_id,
            "error": type(e).__name__,
            "correlation_id": correlation_id
        }))
        raise
```

**Logging Best Practices:**
- Include correlation IDs to trace requests end-to-end
- Log with appropriate levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include timestamps, user context, and operation duration
- Use structured (JSON) logging for easier parsing
- Avoid logging user input directly in production

### 2. Key Metrics to Monitor

**Instrument these metrics in every feature:**

```python
# Response times
start = time.time()
result = process_request(data)
duration = time.time() - start
logger.info(f"Operation took {duration}s")

# Error rates
try:
    result = operation()
except Exception as e:
    error_count += 1
    log_error(e)

# Resource usage
import psutil
memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
logger.info(f"Memory usage: {memory_usage}MB")

# Claude API costs (important!)
tokens_used = response.usage.input_tokens + response.usage.output_tokens
cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015) / 1000
logger.info(f"API tokens: {tokens_used}, estimated cost: ${cost:.6f}")
```

### 3. Alerting Conditions

**Set up alerts for these critical conditions:**

- API error rate > 5% in 5 minutes
- Response time > 10 seconds (P95)
- Claude API rate limit approaching
- RAG system query failures
- Memory usage > 80%
- Dependency versions with known vulnerabilities

---

## ✅ Code Quality Standards

### 1. Type Hints (Python)

```python
from typing import Optional, List, Dict
from pydantic import BaseModel

class Message(BaseModel):
    user_id: str
    content: str
    timestamp: datetime

def process_message(message: Message) -> Dict[str, str]:
    """Process a user message and return response."""
    # Implementation
    pass
```

### 2. Testing & Validation

```python
# Unit test for input validation
def test_validate_user_message():
    # Valid message
    assert validate_user_message("Hello") == "Hello"
    
    # Invalid: too long
    with pytest.raises(ValueError):
        validate_user_message("x" * 5001)
    
    # Invalid: empty
    with pytest.raises(ValueError):
        validate_user_message("")
```

**Testing Requirements:**
- Write unit tests for all validation functions
- Test edge cases: empty input, max length, special characters
- Use `pytest` or similar for Python
- Aim for >80% code coverage for critical paths
- Use static analysis tools: `bandit` (security), `pylint`, `black` (formatting)

### 3. Dependency Management

```bash
# Always check for vulnerabilities
pip-audit
pip install pip-audit --upgrade

# Keep dependencies up to date (test after updates!)
pip list --outdated
pip install --upgrade package_name

# Lock versions in production
pip freeze > requirements.txt
```

**Rules:**
- Run security scans on every release
- Update dependencies regularly, but test thoroughly
- Keep `requirements.txt` with exact versions for reproducibility
- Remove unused dependencies
- Monitor for CVEs in your dependencies

---

## 🚀 Deployment & Production Readiness

### 1. Environment-Specific Config

```python
import os

class Config:
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # dev, staging, prod
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Secrets (should come from env)
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Thresholds
    MAX_MESSAGE_LENGTH = 5000
    MAX_RAG_RESULTS = 10
    REQUEST_TIMEOUT_SECONDS = 30
```

### 2. Health Checks

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """Verify system health."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/ready")
async def readiness_check():
    """Check if service is ready to receive requests."""
    try:
        # Check Claude API connectivity
        # Check database connectivity
        # Check RAG system
        return {"ready": True}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False, "error": str(e)}, 503
```

### 3. Graceful Error Recovery

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_claude_api(message: str) -> str:
    """Call Claude API with automatic retries."""
    try:
        response = await client.messages.create(...)
        return response.content[0].text
    except RateLimitError:
        logger.warning("Rate limited, retrying...")
        raise  # Let retry decorator handle it
    except APIError as e:
        logger.error(f"API error: {e}")
        raise
```

---

## 📋 Pre-Deployment Checklist

Before pushing to production:

- [ ] All inputs are validated and sanitized
- [ ] No secrets hardcoded (check with `git log --all -S "password\|api_key"`)
- [ ] Error messages don't leak sensitive info
- [ ] Logging is structured and doesn't log secrets
- [ ] All dependencies are up to date and vulnerability scanned
- [ ] Type hints are complete
- [ ] Unit tests pass with >80% coverage
- [ ] Security headers are set
- [ ] HTTPS is enforced
- [ ] Rate limiting is in place
- [ ] Health check endpoints work
- [ ] Monitoring/alerting is configured
- [ ] Correlation IDs are used throughout
- [ ] Documentation is up to date

---

## 🔍 Security Audit Checklist

When reviewing code, verify:

**Authentication & Authorization**
- [ ] User inputs are validated
- [ ] API keys are never logged
- [ ] Secrets use environment variables
- [ ] Rate limiting prevents abuse

**Data Protection**
- [ ] Sensitive data isn't in logs
- [ ] Error messages are generic
- [ ] HTTPS is enforced
- [ ] Database credentials are secured

**Code Quality**
- [ ] No hardcoded secrets
- [ ] Type hints present
- [ ] Error handling is proper
- [ ] Dependencies are up to date

---

## 📚 References

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Security Checklist: https://cheatsheetseries.owasp.org/
- Python Security: https://github.com/PyCQA/bandit

---

**Last Updated**: October 2025  
**Project**: Fifi.ai - Fun Interactive Forge for Insights