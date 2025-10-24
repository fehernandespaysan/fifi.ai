#!/usr/bin/env python3
"""
Setup validation script for Fifi.ai

This script validates the environment setup, checks dependencies,
tests API connectivity, and ensures everything is ready for development.

Usage:
    python scripts/setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List, Tuple

from src.config import get_config
from src.logger import get_logger, setup_logging

# Setup logging for this script
setup_logging()
logger = get_logger(__name__)


class SetupValidator:
    """Validates the development environment setup."""

    def __init__(self):
        """Initialize the setup validator."""
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def run_all_checks(self) -> bool:
        """
        Run all validation checks.

        Returns:
            bool: True if all critical checks passed
        """
        print("\n" + "=" * 70)
        print("  Fifi.ai Setup Validation")
        print("=" * 70 + "\n")

        # Run all checks
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Configuration", self.check_configuration),
            ("Environment Variables", self.check_environment_variables),
            ("Directories", self.check_directories),
            ("OpenAI API", self.check_openai_api),
            ("Logging", self.check_logging),
        ]

        for check_name, check_func in checks:
            self._run_check(check_name, check_func)

        # Print summary
        self._print_summary()

        return self.checks_failed == 0

    def _run_check(self, name: str, check_func) -> None:
        """Run a single check and handle the result."""
        print(f"\n[{name}]")
        try:
            status, message = check_func()
            if status == "pass":
                print(f"  ✅ {message}")
                self.checks_passed += 1
            elif status == "fail":
                print(f"  ❌ {message}")
                self.checks_failed += 1
            elif status == "warning":
                print(f"  ⚠️  {message}")
                self.warnings += 1
        except Exception as e:
            print(f"  ❌ Error running check: {str(e)}")
            self.checks_failed += 1
            logger.error(f"Check '{name}' failed", exc_info=True)

    def check_python_version(self) -> Tuple[str, str]:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            return "pass", f"Python {version.major}.{version.minor}.{version.micro}"
        elif version.major == 3 and version.minor >= 9:
            return "warning", f"Python {version.major}.{version.minor}.{version.micro} (recommended: 3.11+)"
        else:
            return "fail", f"Python {version.major}.{version.minor}.{version.micro} (required: 3.11+)"

    def check_dependencies(self) -> Tuple[str, str]:
        """Check if required dependencies are installed."""
        required_packages = [
            "openai",
            "langchain",
            "langchain_openai",
            "faiss",
            "pydantic",
            "pydantic_settings",
            "fastapi",
            "uvicorn",
            "pytest",
        ]

        missing = []
        installed = []

        for package in required_packages:
            try:
                if package == "faiss":
                    __import__("faiss")
                elif package == "langchain_openai":
                    __import__("langchain_openai")
                elif package == "pydantic_settings":
                    __import__("pydantic_settings")
                else:
                    __import__(package)
                installed.append(package)
            except ImportError:
                missing.append(package)

        if missing:
            return "fail", f"Missing packages: {', '.join(missing)}\n     Run: pip install -r requirements.txt"
        return "pass", f"All required packages installed ({len(installed)}/{len(required_packages)})"

    def check_configuration(self) -> Tuple[str, str]:
        """Check if configuration can be loaded."""
        try:
            config = get_config()
            return "pass", f"Configuration loaded ({config.environment.value} environment)"
        except Exception as e:
            return "fail", f"Failed to load configuration: {str(e)}"

    def check_environment_variables(self) -> Tuple[str, str]:
        """Check if required environment variables are set."""
        try:
            config = get_config()

            issues = []

            # Check OpenAI API key
            if not config.openai_api_key or config.openai_api_key == "sk-your-api-key-here":
                issues.append("OPENAI_API_KEY not set")

            # Check secret key in production
            if config.is_production and config.secret_key == "change-this-to-a-random-secret-key-in-production":
                issues.append("SECRET_KEY should be changed in production")

            if issues:
                return "fail", "\n     ".join(["Issues found:"] + issues)

            return "pass", "All required environment variables set"
        except Exception as e:
            return "fail", f"Error checking environment variables: {str(e)}"

    def check_directories(self) -> Tuple[str, str]:
        """Check if required directories exist and create them if needed."""
        try:
            config = get_config()
            config.create_data_directories()

            directories = [
                config.blogs_directory,
                config.faiss_index_path.parent,
                Path("logs"),
            ]

            created = []
            existing = []

            for directory in directories:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    created.append(str(directory))
                else:
                    existing.append(str(directory))

            if created:
                return "pass", f"Created {len(created)} directories, {len(existing)} already exist"
            return "pass", f"All directories exist ({len(existing)} total)"
        except Exception as e:
            return "fail", f"Error with directories: {str(e)}"

    def check_openai_api(self) -> Tuple[str, str]:
        """Test OpenAI API connectivity."""
        try:
            from openai import OpenAI

            config = get_config()

            # Check if API key is set
            if not config.openai_api_key or config.openai_api_key == "sk-your-api-key-here":
                return "fail", "OpenAI API key not configured\n     Set OPENAI_API_KEY in .env file"

            # Initialize client
            client = OpenAI(api_key=config.openai_api_key)

            # Test API with a minimal request (costs ~$0.000015)
            try:
                response = client.chat.completions.create(
                    model=config.openai_model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )

                # Check if we got a response
                if response.choices and response.choices[0].message.content:
                    return "pass", f"OpenAI API connected (model: {config.openai_model})"
                else:
                    return "warning", "OpenAI API responded but no content received"

            except Exception as api_error:
                error_message = str(api_error)

                # Check for common errors
                if "api_key" in error_message.lower() or "authentication" in error_message.lower():
                    return "fail", f"Invalid API key\n     Error: {error_message}"
                elif "rate_limit" in error_message.lower():
                    return "warning", f"Rate limit reached\n     Error: {error_message}"
                elif "quota" in error_message.lower():
                    return "fail", f"Quota exceeded\n     Error: {error_message}"
                else:
                    return "fail", f"API request failed\n     Error: {error_message}"

        except ImportError:
            return "fail", "OpenAI package not installed\n     Run: pip install openai"
        except Exception as e:
            return "fail", f"Unexpected error: {str(e)}"

    def check_logging(self) -> Tuple[str, str]:
        """Test logging system."""
        try:
            test_logger = get_logger("test")
            test_logger.info("Setup validation test log")
            return "pass", "Logging system functional"
        except Exception as e:
            return "fail", f"Logging system error: {str(e)}"

    def _print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("  Validation Summary")
        print("=" * 70)
        print(f"\n  ✅ Passed:   {self.checks_passed}")
        print(f"  ❌ Failed:   {self.checks_failed}")
        print(f"  ⚠️  Warnings: {self.warnings}")

        if self.checks_failed == 0:
            print("\n  🎉 Setup validation successful! You're ready to start developing.")
            print("\n  Next steps:")
            print("    1. Add some blog posts to the blogs/ directory")
            print("    2. Run: python scripts/generate_embeddings.py (coming soon)")
            print("    3. Run: python main.py (coming soon)")
        else:
            print("\n  ⚠️  Setup validation failed. Please fix the issues above.")
            print("\n  Common fixes:")
            print("    - Copy .env.example to .env and add your OpenAI API key")
            print("    - Run: pip install -r requirements.txt")
            print("    - Get an API key from: https://platform.openai.com/api-keys")

        print("\n" + "=" * 70 + "\n")


def main() -> int:
    """
    Main entry point for setup validation.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        validator = SetupValidator()
        success = validator.run_all_checks()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error during validation: {str(e)}")
        logger.error("Fatal error during setup validation", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
