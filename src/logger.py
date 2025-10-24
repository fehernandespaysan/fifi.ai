"""
Structured logging setup for Fifi.ai

This module provides centralized logging with structured JSON output,
correlation IDs for request tracking, and security-first practices.

Security notes:
- Never log sensitive data (API keys, secrets, PII)
- Sanitize all user input before logging
- Use correlation IDs to track requests across services
- Log levels: DEBUG (dev only), INFO (normal), WARNING (unusual), ERROR (failures), CRITICAL (system issues)

Usage:
    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("User query processed", extra={"query_id": "123", "tokens": 500})
"""

import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_config

# Context variable for correlation ID (thread-safe)
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIDFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to all log records.

    Correlation IDs help track a single request across multiple services and operations.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the log record."""
        record.correlation_id = get_correlation_id()
        return True


class SecureFormatter(logging.Formatter):
    """
    Custom formatter that ensures sensitive data is never logged.

    Supports both JSON and text output formats.
    """

    # Fields that should never be logged (case-insensitive)
    SENSITIVE_FIELDS = {
        "password",
        "api_key",
        "apikey",
        "secret",
        "token",
        "authorization",
        "auth",
        "bearer",
        "cookie",
        "session",
        "csrf",
        "ssn",
        "credit_card",
        "cvv",
    }

    def __init__(self, format_type: str = "json") -> None:
        """
        Initialize the secure formatter.

        Args:
            format_type: Either 'json' or 'text'
        """
        super().__init__()
        self.format_type = format_type

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with security measures."""
        # Sanitize the record
        record = self._sanitize_record(record)

        if self.format_type == "json":
            return self._format_json(record)
        else:
            return self._format_text(record)

    def _sanitize_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """Remove or mask sensitive data from log record."""
        # Sanitize message
        if isinstance(record.msg, dict):
            record.msg = self._sanitize_dict(record.msg)
        elif isinstance(record.msg, str):
            record.msg = self._sanitize_string(record.msg)

        # Sanitize extra fields
        if hasattr(record, "extra"):
            record.extra = self._sanitize_dict(record.extra)

        # Sanitize args
        if record.args:
            record.args = tuple(
                self._sanitize_value(arg) for arg in record.args
            )

        return record

    def _sanitize_string(self, text: str) -> str:
        """Sanitize a string by redacting sensitive patterns."""
        import re

        # Pattern for API keys starting with sk-, pk-, etc. (at least 10 chars after prefix)
        text = re.sub(r'\bsk-[a-zA-Z0-9_-]{10,}', '***REDACTED***', text)
        text = re.sub(r'\bpk-[a-zA-Z0-9_-]{10,}', '***REDACTED***', text)

        # Pattern for Bearer tokens
        text = re.sub(r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*', '***REDACTED***', text)

        return text

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize a dictionary by masking sensitive fields."""
        sanitized = {}
        for key, value in data.items():
            # Check if value is a dict first - always recurse into nested dicts
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [self._sanitize_value(v) for v in value]
            # Only redact non-dict values with sensitive key names
            elif any(sensitive in key.lower() for sensitive in self.SENSITIVE_FIELDS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = self._sanitize_value(value)
        return sanitized

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        if isinstance(value, dict):
            return self._sanitize_dict(value)
        elif isinstance(value, str):
            # Check if it looks like an API key or token
            if value.startswith(("sk-", "Bearer ", "token ")):
                return "***REDACTED***"
        return value

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", None),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields from record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "correlation_id",
                ]:
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)

    def _format_text(self, record: logging.LogRecord) -> str:
        """Format log record as readable text."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        correlation_id = getattr(record, "correlation_id", "N/A")

        text = (
            f"[{timestamp}] "
            f"{record.levelname:8s} "
            f"[{correlation_id}] "
            f"{record.name}:{record.lineno} - "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)

        return text


def get_correlation_id() -> str:
    """
    Get or create a correlation ID for the current context.

    Returns:
        str: The correlation ID (UUID format)
    """
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set a correlation ID for the current context.

    Args:
        correlation_id: Optional correlation ID (generates new one if None)

    Returns:
        str: The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    correlation_id_var.set(None)


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional file path for logging to file
    """
    config = get_config()

    # Use config values if not provided
    if log_level is None:
        log_level = config.log_level.value
    if log_format is None:
        log_format = config.log_format.value

    # Create formatter
    formatter = SecureFormatter(format_type=log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIDFilter())
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIDFilter())
        root_logger.addHandler(file_handler)

    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing query", extra={"user_id": "123", "tokens": 500})
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    Context manager for setting correlation ID within a specific context.

    Usage:
        with LoggerContext(correlation_id="req_123"):
            logger.info("Processing request")  # Will include correlation_id
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize logger context.

        Args:
            correlation_id: Optional correlation ID (generates new one if None)
        """
        self.correlation_id = correlation_id
        self.previous_correlation_id = None

    def __enter__(self) -> str:
        """Enter the context and set correlation ID."""
        self.previous_correlation_id = correlation_id_var.get()
        self.correlation_id = set_correlation_id(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore previous correlation ID."""
        correlation_id_var.set(self.previous_correlation_id)


def log_function_call(logger: logging.Logger, log_level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments and return values.

    Security note: Only use on functions that don't handle sensitive data.

    Args:
        logger: Logger instance to use
        log_level: Logging level (default: DEBUG)

    Example:
        >>> logger = get_logger(__name__)
        >>> @log_function_call(logger)
        >>> def process_query(query: str) -> str:
        >>>     return f"Processed: {query}"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(
                log_level,
                f"Calling {func_name}",
                extra={
                    "function": func_name,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )
            try:
                result = func(*args, **kwargs)
                logger.log(
                    log_level,
                    f"Completed {func_name}",
                    extra={"function": func_name, "success": True},
                )
                return result
            except Exception as e:
                logger.error(
                    f"Error in {func_name}: {str(e)}",
                    extra={"function": func_name, "error_type": type(e).__name__},
                    exc_info=True,
                )
                raise

        return wrapper
    return decorator


# Initialize logging on module import
try:
    setup_logging()
except Exception:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
    logging.error("Failed to initialize structured logging, using basic config")
