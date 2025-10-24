"""
Configuration management for Fifi.ai

This module provides centralized configuration management using Pydantic Settings.
It supports multiple environments (development, staging, production) and loads
configuration from environment variables with validation.

Security notes:
- Never log sensitive values (API keys, secrets)
- Always use environment variables for secrets
- Validate all configuration on startup
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    """Logging level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Logging format options."""

    JSON = "json"
    TEXT = "text"


class Config(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Usage:
        config = Config()
        api_key = config.openai_api_key
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars not defined in model
    )

    # =============================================================================
    # Environment Configuration
    # =============================================================================
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )

    # =============================================================================
    # Application Settings
    # =============================================================================
    app_name: str = Field(default="Fifi.ai", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=True, description="Enable debug mode")

    # =============================================================================
    # Branding & Customization
    # =============================================================================
    app_tagline: str = Field(
        default="Your AI Engineering Assistant",
        description="Application tagline/subtitle"
    )
    welcome_title: str = Field(
        default="Welcome to Fifi",
        description="Welcome screen title"
    )
    welcome_message: str = Field(
        default="Your AI assistant powered by RAG. Ask me anything about AI, machine learning, embeddings, or software engineering.",
        description="Welcome screen message"
    )
    example_question_1: str = Field(
        default="What is RAG and how does it work?",
        description="First example question"
    )
    example_question_2: str = Field(
        default="How do vector databases improve search?",
        description="Second example question"
    )
    example_question_3: str = Field(
        default="Explain embeddings in simple terms",
        description="Third example question"
    )
    example_question_4: str = Field(
        default="What are best practices for AI security?",
        description="Fourth example question"
    )

    # =============================================================================
    # OpenAI API Configuration
    # =============================================================================
    openai_api_key: str = Field(
        description="OpenAI API key (required)",
        min_length=20  # Basic validation
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for chat completions"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model to use for embeddings"
    )
    openai_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (0=deterministic, 2=creative)"
    )
    openai_max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4096,
        description="Maximum tokens in generated response"
    )

    # =============================================================================
    # Logging Configuration
    # =============================================================================
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    log_format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Logging format (json or text)"
    )

    # =============================================================================
    # Vector Database Configuration
    # =============================================================================
    faiss_index_path: Path = Field(
        default=Path("data/faiss_index.faiss"),
        description="Path to FAISS index file"
    )
    faiss_metadata_path: Path = Field(
        default=Path("data/faiss_metadata.pkl"),
        description="Path to FAISS metadata file"
    )
    vector_search_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve in vector search"
    )
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Size of text chunks in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens"
    )

    # =============================================================================
    # RAG Engine Configuration
    # =============================================================================
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve for RAG queries"
    )
    rag_min_relevance_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for retrieved chunks"
    )
    rag_max_history: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum number of conversation turns to keep in history"
    )

    # =============================================================================
    # Blog Content Configuration
    # =============================================================================
    blogs_directory: Path = Field(
        default=Path("blogs/"),
        description="Directory containing blog markdown files"
    )
    blog_extensions: str | List[str] = Field(
        default=[".md", ".markdown"],
        description="Supported blog file extensions"
    )

    # =============================================================================
    # API Configuration (FastAPI)
    # =============================================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    api_reload: bool = Field(
        default=True,
        description="Auto-reload on code changes (development only)"
    )

    # CORS Settings
    cors_origins: str | List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=10,
        ge=1,
        description="Maximum requests per minute per user"
    )
    rate_limit_requests_per_day: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per day per user"
    )

    # =============================================================================
    # Performance & Caching
    # =============================================================================
    cache_enabled: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="Cache TTL in seconds"
    )
    request_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )

    # =============================================================================
    # Monitoring & Observability
    # =============================================================================
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Metrics server port"
    )
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    sentry_environment: Optional[str] = Field(
        default=None,
        description="Sentry environment name"
    )

    # =============================================================================
    # Testing Configuration
    # =============================================================================
    test_mode: bool = Field(
        default=False,
        description="Enable test mode"
    )
    mock_openai_api: bool = Field(
        default=False,
        description="Mock OpenAI API calls (for testing)"
    )

    # =============================================================================
    # Security
    # =============================================================================
    secret_key: str = Field(
        default="change-this-to-a-random-secret-key-in-production",
        min_length=32,
        description="Secret key for session management"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authenticated endpoints"
    )

    # =============================================================================
    # Development Tools
    # =============================================================================
    watchfiles_force_polling: bool = Field(
        default=False,
        description="Force polling for file changes (Docker/WSL)"
    )
    json_pretty_print: bool = Field(
        default=True,
        description="Pretty print JSON responses"
    )

    # =============================================================================
    # Validators
    # =============================================================================

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v or v == "sk-your-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY not set. "
                "Get your API key from https://platform.openai.com/api-keys"
            )
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key is changed in production."""
        if v == "change-this-to-a-random-secret-key-in-production":
            import warnings
            warnings.warn(
                "Using default SECRET_KEY. Generate a secure key with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(32))\"",
                stacklevel=2
            )
        return v

    @field_validator("blogs_directory", "faiss_index_path", "faiss_metadata_path")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("blog_extensions", mode="before")
    @classmethod
    def parse_blog_extensions(cls, v: str | List[str]) -> List[str]:
        """Parse blog extensions from comma-separated string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    # =============================================================================
    # Helper Methods
    # =============================================================================

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self.environment == Environment.TEST or self.test_mode

    def create_data_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        directories = [
            self.faiss_index_path.parent,
            self.blogs_directory,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def mask_sensitive_value(self, value: str, show_chars: int = 4) -> str:
        """
        Mask sensitive values for logging.

        Args:
            value: The sensitive value to mask
            show_chars: Number of characters to show at the end

        Returns:
            Masked value like "sk-***xyz"
        """
        if not value or len(value) <= show_chars:
            return "***"
        return f"{value[:3]}***{value[-show_chars:]}"

    def get_safe_config_dict(self) -> dict:
        """
        Get configuration as dictionary with sensitive values masked.

        Safe for logging and debugging.
        """
        config_dict = self.model_dump()

        # Mask sensitive fields
        sensitive_fields = [
            "openai_api_key",
            "secret_key",
            "api_key",
            "sentry_dsn",
        ]

        for field in sensitive_fields:
            if field in config_dict and config_dict[field]:
                config_dict[field] = self.mask_sensitive_value(str(config_dict[field]))

        return config_dict


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Config: The application configuration

    Example:
        >>> from src.config import get_config
        >>> config = get_config()
        >>> print(config.openai_model)
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
