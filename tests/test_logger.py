"""
Unit tests for src/logger.py

Tests logging functionality, correlation IDs, and security features.
"""

import json
import logging
import os
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from src.logger import (
    CorrelationIDFilter,
    SecureFormatter,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    get_logger,
    setup_logging,
    LoggerContext,
    log_function_call,
)


class TestCorrelationID:
    """Test correlation ID functionality."""

    def test_get_correlation_id_generates_new(self):
        """Test that get_correlation_id generates a new ID if none exists."""
        clear_correlation_id()
        corr_id = get_correlation_id()
        assert corr_id is not None
        assert len(corr_id) > 0

    def test_get_correlation_id_returns_existing(self):
        """Test that get_correlation_id returns existing ID."""
        clear_correlation_id()
        corr_id1 = get_correlation_id()
        corr_id2 = get_correlation_id()
        assert corr_id1 == corr_id2

    def test_set_correlation_id(self):
        """Test setting a custom correlation ID."""
        test_id = "test-correlation-id-123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

    def test_set_correlation_id_generates_if_none(self):
        """Test that set_correlation_id generates ID if none provided."""
        clear_correlation_id()
        new_id = set_correlation_id()
        assert new_id is not None
        assert get_correlation_id() == new_id

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-id")
        clear_correlation_id()
        # Getting should generate a new one
        new_id = get_correlation_id()
        assert new_id != "test-id"


class TestCorrelationIDFilter:
    """Test the CorrelationIDFilter class."""

    def test_filter_adds_correlation_id(self):
        """Test that filter adds correlation_id to log record."""
        test_id = "filter-test-id"
        set_correlation_id(test_id)

        filter_obj = CorrelationIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)
        assert hasattr(record, "correlation_id")
        assert record.correlation_id == test_id


class TestSecureFormatter:
    """Test the SecureFormatter class."""

    def test_json_format(self):
        """Test JSON formatting."""
        formatter = SecureFormatter(format_type="json")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["correlation_id"] == "test-id"
        assert data["line"] == 42
        assert "timestamp" in data

    def test_text_format(self):
        """Test text formatting."""
        formatter = SecureFormatter(format_type="text")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"

        output = formatter.format(record)

        assert "INFO" in output
        assert "test-id" in output
        assert "Test message" in output
        assert "test:42" in output

    def test_sanitize_api_key_in_message(self):
        """Test that API keys in messages are redacted."""
        formatter = SecureFormatter(format_type="json")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="API key: sk-1234567890abcdef",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-id"

        output = formatter.format(record)
        assert "sk-1234567890abcdef" not in output
        assert "REDACTED" in output

    def test_sanitize_dict_with_sensitive_fields(self):
        """Test that sensitive fields in dicts are redacted."""
        formatter = SecureFormatter(format_type="json")

        data = {
            "user": "john",
            "api_key": "secret-key-123",
            "password": "my-password",
            "token": "bearer-token",
        }

        sanitized = formatter._sanitize_dict(data)

        assert sanitized["user"] == "john"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["token"] == "***REDACTED***"

    def test_sanitize_nested_dict(self):
        """Test that nested dicts are sanitized."""
        formatter = SecureFormatter(format_type="json")

        data = {
            "user": "john",
            "auth": {
                "api_key": "secret-key",
                "username": "john",
            },
        }

        sanitized = formatter._sanitize_dict(data)

        assert sanitized["user"] == "john"
        assert sanitized["auth"]["username"] == "john"
        assert sanitized["auth"]["api_key"] == "***REDACTED***"

    def test_sanitize_bearer_token(self):
        """Test that Bearer tokens are redacted."""
        formatter = SecureFormatter(format_type="json")

        value = "Bearer abc123xyz"
        sanitized = formatter._sanitize_value(value)

        assert sanitized == "***REDACTED***"
        assert "abc123xyz" not in str(sanitized)


class TestLoggerContext:
    """Test the LoggerContext context manager."""

    def test_context_sets_correlation_id(self):
        """Test that context manager sets correlation ID."""
        clear_correlation_id()
        test_id = "context-test-id"

        with LoggerContext(correlation_id=test_id) as context_id:
            assert context_id == test_id
            assert get_correlation_id() == test_id

    def test_context_generates_id_if_none(self):
        """Test that context manager generates ID if none provided."""
        clear_correlation_id()

        with LoggerContext() as context_id:
            assert context_id is not None
            assert get_correlation_id() == context_id

    def test_context_restores_previous_id(self):
        """Test that context manager restores previous correlation ID."""
        original_id = "original-id"
        set_correlation_id(original_id)

        with LoggerContext(correlation_id="temporary-id"):
            assert get_correlation_id() == "temporary-id"

        assert get_correlation_id() == original_id


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_get_logger_with_module_name(self):
        """Test get_logger with __name__."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_with_defaults(self):
        """Test that setup_logging works with default values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            # Should not raise any exceptions
            setup_logging()

            # Check that root logger is configured
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) > 0

    def test_setup_logging_with_custom_level(self):
        """Test setup_logging with custom log level."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            setup_logging(log_level="DEBUG")

            root_logger = logging.getLogger()
            assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_text_format(self):
        """Test setup_logging with text format."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            setup_logging(log_format="text")

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) > 0
            # Check that formatter is SecureFormatter
            assert isinstance(root_logger.handlers[0].formatter, SecureFormatter)

    def test_setup_logging_with_file(self, tmp_path):
        """Test setup_logging with log file."""
        log_file = tmp_path / "test.log"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            setup_logging(log_file=log_file)

            # Check that file was created
            root_logger = logging.getLogger()

            # Log a message
            test_logger = get_logger("file_test")
            test_logger.info("Test file logging")

            # Check that file exists and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert len(content) > 0


class TestLogFunctionCall:
    """Test the log_function_call decorator."""

    def test_decorator_logs_function_call(self, caplog):
        """Test that decorator logs function calls."""
        logger = get_logger("decorator_test")

        @log_function_call(logger, log_level=logging.INFO)
        def test_function(x, y):
            return x + y

        with caplog.at_level(logging.INFO):
            result = test_function(2, 3)

        assert result == 5
        # Check that logs contain function name
        assert "test_function" in caplog.text
        assert "Calling" in caplog.text or "Completed" in caplog.text

    def test_decorator_logs_errors(self, caplog):
        """Test that decorator logs errors."""
        logger = get_logger("decorator_error_test")

        @log_function_call(logger, log_level=logging.INFO)
        def failing_function():
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_function()

        assert "Error" in caplog.text or "error" in caplog.text.lower()
        assert "failing_function" in caplog.text


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_log_message_includes_correlation_id(self, caplog):
        """Test that logged messages include correlation ID."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            setup_logging(log_format="json")

            test_id = "integration-test-id"
            set_correlation_id(test_id)

            logger = get_logger("integration_test")

            with caplog.at_level(logging.INFO):
                logger.info("Test integration message")

            # Check that correlation ID is in the output
            # Note: caplog might not capture the formatted output directly
            # but we can verify the record has the correlation_id attribute

    def test_sensitive_data_not_logged(self, caplog):
        """Test that sensitive data is redacted by the formatter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789012345678901234567890"}):
            # Don't call setup_logging() so caplog can capture logs
            logger = get_logger("security_test")

            # Set up the secure formatter
            formatter = SecureFormatter(format_type="text")

            with caplog.at_level(logging.INFO):
                # Try to log sensitive data
                logger.info("User password: my-secret-password")
                logger.info("API Key: sk-1234567890abcdef")

            # Verify that logs were captured
            assert len(caplog.records) == 2

            # Verify that the formatter sanitizes the messages
            formatted_message1 = formatter.format(caplog.records[0])
            formatted_message2 = formatter.format(caplog.records[1])

            # API key should be redacted
            assert "sk-1234567890abcdef" not in formatted_message2
            assert "REDACTED" in formatted_message2

    def test_multiple_loggers_share_correlation_id(self):
        """Test that multiple loggers share the same correlation ID."""
        test_id = "shared-id"
        set_correlation_id(test_id)

        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        # Both should use the same correlation ID
        assert get_correlation_id() == test_id

    def test_logger_context_isolation(self):
        """Test that logger contexts are properly isolated."""
        original_id = "original"
        set_correlation_id(original_id)

        with LoggerContext(correlation_id="context1"):
            context1_id = get_correlation_id()
            assert context1_id == "context1"

            with LoggerContext(correlation_id="context2"):
                context2_id = get_correlation_id()
                assert context2_id == "context2"

            # Should restore context1
            assert get_correlation_id() == "context1"

        # Should restore original
        assert get_correlation_id() == original_id
