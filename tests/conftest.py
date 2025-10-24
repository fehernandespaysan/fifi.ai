"""
Pytest configuration and shared fixtures for Fifi.ai tests.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import reset_config
from src.logger import clear_correlation_id


@pytest.fixture(autouse=True)
def reset_environment():
    """
    Automatically reset configuration and correlation ID before each test.

    This ensures test isolation.
    """
    # Reset before test
    reset_config()
    clear_correlation_id()

    yield

    # Reset after test
    reset_config()
    clear_correlation_id()


@pytest.fixture
def mock_openai_api_key():
    """Provide a mock OpenAI API key for testing."""
    return "sk-test123456789012345678901234567890"


@pytest.fixture
def mock_env(mock_openai_api_key):
    """
    Provide a mock environment with required variables.

    Usage:
        def test_something(mock_env):
            # Environment is already set up
            config = Config()
    """
    env_vars = {
        "OPENAI_API_KEY": mock_openai_api_key,
        "ENVIRONMENT": "test",
        "TEST_MODE": "true",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Create a temporary directory structure for data files.

    Returns:
        Path: Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    blogs_dir = tmp_path / "blogs"
    blogs_dir.mkdir()

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    return tmp_path


@pytest.fixture
def sample_blog_content():
    """Provide sample blog content for testing."""
    return """---
title: Test Blog Post
date: 2025-10-23
tags: test, python, rag
---

# Introduction

This is a test blog post for Fifi.ai.

## Section 1

Some content about RAG systems and how they work.

## Section 2

More information about vector databases and embeddings.
"""


@pytest.fixture
def sample_env_file(tmp_path, mock_openai_api_key):
    """
    Create a sample .env file for testing.

    Returns:
        Path: Path to the .env file
    """
    env_file = tmp_path / ".env"
    env_content = f"""
OPENAI_API_KEY={mock_openai_api_key}
ENVIRONMENT=test
TEST_MODE=true
LOG_LEVEL=DEBUG
"""
    env_file.write_text(env_content)
    return env_file
