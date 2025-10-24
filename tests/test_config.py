"""
Unit tests for src/config.py

Tests configuration loading, validation, and security features.
"""

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Config, Environment, LogLevel, LogFormat, get_config, reset_config


class TestConfigModel:
    """Test the Config model and its validation."""

    def test_config_loads_with_valid_api_key(self):
        """Test that config loads successfully with valid API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config = Config()
            assert config.openai_api_key.startswith("sk-")
            assert config.app_name == "Fifi.ai"

    def test_config_fails_with_invalid_api_key(self):
        """Test that config validation fails with invalid API key."""
        # Use a key that's long enough (>20 chars) but doesn't start with sk-
        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-key-1234567890"}):
            reset_config()
            with pytest.raises(ValidationError) as exc_info:
                Config()
            assert "OpenAI API key must start with 'sk-'" in str(exc_info.value)

    def test_config_fails_with_placeholder_api_key(self):
        """Test that config validation fails with placeholder API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-your-api-key-here"}):
            reset_config()
            with pytest.raises(ValidationError) as exc_info:
                Config()
            assert "OPENAI_API_KEY not set" in str(exc_info.value)

    def test_environment_enum_values(self):
        """Test that environment enum has correct values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TEST.value == "test"

    def test_log_level_enum_values(self):
        """Test that log level enum has correct values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config = Config()

            assert config.app_name == "Fifi.ai"
            assert config.app_version == "0.1.0"
            assert config.environment == Environment.DEVELOPMENT
            assert config.openai_model == "gpt-4o-mini"
            assert config.openai_embedding_model == "text-embedding-3-small"
            assert config.log_level == LogLevel.INFO
            assert config.log_format == LogFormat.JSON
            assert config.vector_search_top_k == 5
            assert config.chunk_size == 500
            assert config.chunk_overlap == 50

    def test_custom_values_from_env(self):
        """Test that custom environment values override defaults."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "ENVIRONMENT": "production",
            "OPENAI_MODEL": "gpt-4o",
            "LOG_LEVEL": "DEBUG",
            "CHUNK_SIZE": "1000",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            reset_config()
            config = Config()

            assert config.environment == Environment.PRODUCTION
            assert config.openai_model == "gpt-4o"
            assert config.log_level == LogLevel.DEBUG
            assert config.chunk_size == 1000

    def test_vector_search_top_k_validation(self):
        """Test that vector_search_top_k is validated correctly."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "VECTOR_SEARCH_TOP_K": "0",  # Invalid: too low
        }
        with patch.dict(os.environ, env_vars):
            reset_config()
            with pytest.raises(ValidationError):
                Config()

        env_vars["VECTOR_SEARCH_TOP_K"] = "25"  # Invalid: too high
        with patch.dict(os.environ, env_vars):
            reset_config()
            with pytest.raises(ValidationError):
                Config()

    def test_chunk_size_validation(self):
        """Test that chunk_size is validated correctly."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "CHUNK_SIZE": "50",  # Invalid: too small
        }
        with patch.dict(os.environ, env_vars):
            reset_config()
            with pytest.raises(ValidationError):
                Config()

    def test_api_port_validation(self):
        """Test that API port is validated correctly."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "API_PORT": "100",  # Invalid: below 1024
        }
        with patch.dict(os.environ, env_vars):
            reset_config()
            with pytest.raises(ValidationError):
                Config()

    def test_secret_key_warning(self):
        """Test that using default secret key generates warning."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "SECRET_KEY": "change-this-to-a-random-secret-key-in-production"
        }):
            reset_config()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                Config()
                # Check if warning was issued
                assert len(w) > 0
                assert "SECRET_KEY" in str(w[0].message)


class TestConfigHelperMethods:
    """Test helper methods in the Config class."""

    def test_is_development(self):
        """Test is_development property."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "ENVIRONMENT": "development"
        }):
            reset_config()
            config = Config()
            assert config.is_development is True
            assert config.is_production is False
            assert config.is_test is False

    def test_is_production(self):
        """Test is_production property."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "ENVIRONMENT": "production"
        }):
            reset_config()
            config = Config()
            assert config.is_production is True
            assert config.is_development is False

    def test_is_test(self):
        """Test is_test property."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "ENVIRONMENT": "test"
        }):
            reset_config()
            config = Config()
            assert config.is_test is True

    def test_is_test_with_test_mode(self):
        """Test is_test property when test_mode is enabled."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "ENVIRONMENT": "development",
            "TEST_MODE": "true"
        }):
            reset_config()
            config = Config()
            assert config.is_test is True

    def test_mask_sensitive_value(self):
        """Test that sensitive values are masked correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config = Config()

            masked = config.mask_sensitive_value("sk-abc123xyz789")
            assert masked == "sk-***z789"
            assert "abc123" not in masked

    def test_mask_sensitive_value_short(self):
        """Test masking of short values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config = Config()

            masked = config.mask_sensitive_value("abc")
            assert masked == "***"

    def test_get_safe_config_dict(self):
        """Test that safe config dict masks sensitive values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config = Config()

            safe_dict = config.get_safe_config_dict()

            # Check that sensitive fields are masked
            assert "***" in safe_dict["openai_api_key"]
            assert "sk-test" not in safe_dict["openai_api_key"]
            assert "***" in safe_dict["secret_key"]

            # Check that non-sensitive fields are not masked
            assert safe_dict["app_name"] == "Fifi.ai"
            assert safe_dict["openai_model"] == "gpt-4o-mini"

    def test_create_data_directories(self, tmp_path):
        """Test that data directories are created."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "FAISS_INDEX_PATH": str(tmp_path / "data/faiss_index.faiss"),
            "BLOGS_DIRECTORY": str(tmp_path / "blogs/"),
        }):
            reset_config()
            config = Config()
            config.create_data_directories()

            assert config.faiss_index_path.parent.exists()
            assert config.blogs_directory.exists()


class TestConfigSingleton:
    """Test the singleton pattern for config."""

    def test_get_config_returns_same_instance(self):
        """Test that get_config returns the same instance."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config1 = get_config()
            config2 = get_config()
            assert config1 is config2

    def test_reset_config_clears_instance(self):
        """Test that reset_config clears the singleton instance."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            reset_config()
            config1 = get_config()
            reset_config()
            config2 = get_config()
            assert config1 is not config2


class TestConfigPathValidation:
    """Test path validation and conversion."""

    def test_path_strings_converted_to_paths(self):
        """Test that string paths are converted to Path objects."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "FAISS_INDEX_PATH": "data/test.faiss",
            "BLOGS_DIRECTORY": "blogs/",
        }):
            reset_config()
            config = Config()

            assert isinstance(config.faiss_index_path, Path)
            assert isinstance(config.blogs_directory, Path)


class TestConfigCORSParsing:
    """Test CORS origins parsing."""

    def test_cors_origins_from_string(self):
        """Test parsing CORS origins from comma-separated string."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "CORS_ORIGINS": "http://localhost:3000,http://localhost:8501,https://example.com",
        }):
            reset_config()
            config = Config()

            assert len(config.cors_origins) == 3
            assert "http://localhost:3000" in config.cors_origins
            assert "https://example.com" in config.cors_origins

    def test_cors_origins_strips_whitespace(self):
        """Test that CORS origins strips whitespace."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "CORS_ORIGINS": " http://localhost:3000 , http://localhost:8501 ",
        }):
            reset_config()
            config = Config()

            assert config.cors_origins[0] == "http://localhost:3000"
            assert " " not in config.cors_origins[0]


class TestConfigBlogExtensions:
    """Test blog extensions parsing."""

    def test_blog_extensions_from_string(self):
        """Test parsing blog extensions from comma-separated string."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "BLOG_EXTENSIONS": ".md,.markdown,.txt",
        }):
            reset_config()
            config = Config()

            assert len(config.blog_extensions) == 3
            assert ".md" in config.blog_extensions
            assert ".txt" in config.blog_extensions
