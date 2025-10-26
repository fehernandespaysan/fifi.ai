"""
Prompt Loader for Fifi.ai

Loads and manages prompts from YAML configuration files.
Supports prompt versioning, caching, and environment-specific overrides.

Usage:
    from src.prompt_loader import PromptLoader

    loader = PromptLoader()
    system_prompt = loader.get_system_prompt()
    user_template = loader.get_user_template()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.logger import get_logger

logger = get_logger(__name__)


class PromptLoaderError(Exception):
    """Exception raised for prompt loading errors."""
    pass


class PromptLoader:
    """
    Loads and manages prompts from YAML configuration files.

    Provides caching and validation for prompt templates.
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt YAML files (defaults to ./prompts)
        """
        if prompts_dir is None:
            # Default to prompts/ directory at project root
            project_root = Path(__file__).parent.parent
            prompts_dir = project_root / "prompts"

        self.prompts_dir = Path(prompts_dir)

        if not self.prompts_dir.exists():
            raise PromptLoaderError(
                f"Prompts directory not found: {self.prompts_dir}"
            )

        # Cache for loaded prompts
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"PromptLoader initialized",
            extra={"prompts_dir": str(self.prompts_dir)}
        )

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML file from the prompts directory.

        Args:
            filename: Name of the YAML file (e.g., "system_prompt.yaml")

        Returns:
            Dictionary with prompt configuration

        Raises:
            PromptLoaderError: If file doesn't exist or is invalid
        """
        # Check cache first
        if filename in self._cache:
            logger.debug(f"Using cached prompt: {filename}")
            return self._cache[filename]

        file_path = self.prompts_dir / filename

        if not file_path.exists():
            raise PromptLoaderError(
                f"Prompt file not found: {file_path}"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise PromptLoaderError(
                    f"Invalid YAML structure in {filename}: expected dict, got {type(data)}"
                )

            # Cache the loaded data
            self._cache[filename] = data

            logger.debug(
                f"Loaded prompt from {filename}",
                extra={
                    "name": data.get("name"),
                    "version": data.get("version")
                }
            )

            return data

        except yaml.YAMLError as e:
            raise PromptLoaderError(
                f"Failed to parse YAML file {filename}: {str(e)}"
            )
        except Exception as e:
            raise PromptLoaderError(
                f"Failed to load prompt file {filename}: {str(e)}"
            )

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the RAG assistant.

        Returns:
            System prompt string
        """
        data = self._load_yaml("system_prompt.yaml")
        prompt = data.get("prompt")

        if not prompt:
            raise PromptLoaderError(
                "system_prompt.yaml missing 'prompt' field"
            )

        # Strip leading/trailing whitespace but preserve internal formatting
        return prompt.strip()

    def get_user_template(self) -> str:
        """
        Get the user prompt template for RAG queries.

        Returns:
            Template string with {context} and {question} placeholders
        """
        data = self._load_yaml("user_template.yaml")
        template = data.get("template")

        if not template:
            raise PromptLoaderError(
                "user_template.yaml missing 'template' field"
            )

        # Validate required placeholders
        required_placeholders = ["{context}", "{question}"]
        for placeholder in required_placeholders:
            if placeholder not in template:
                logger.warning(
                    f"User template missing expected placeholder: {placeholder}"
                )

        return template.strip()

    def get_fallback_prompt(self) -> Dict[str, str]:
        """
        Get the fallback prompt for queries with no context.

        Returns:
            Dictionary with 'template' and 'system_prompt' keys
        """
        data = self._load_yaml("fallback_prompt.yaml")

        template = data.get("template")
        system_prompt = data.get("system_prompt")

        if not template:
            raise PromptLoaderError(
                "fallback_prompt.yaml missing 'template' field"
            )

        # Validate placeholder
        if "{query}" not in template:
            logger.warning(
                "Fallback template missing expected placeholder: {query}"
            )

        return {
            "template": template.strip(),
            "system_prompt": system_prompt or "You are a helpful AI assistant.",
            "topics": data.get("knowledge_base_topics", [])
        }

    def get_prompt_metadata(self, prompt_type: str) -> Dict[str, Any]:
        """
        Get metadata about a prompt.

        Args:
            prompt_type: Type of prompt ("system", "user", "fallback")

        Returns:
            Dictionary with prompt metadata
        """
        filename_map = {
            "system": "system_prompt.yaml",
            "user": "user_template.yaml",
            "fallback": "fallback_prompt.yaml"
        }

        filename = filename_map.get(prompt_type)
        if not filename:
            raise PromptLoaderError(
                f"Unknown prompt type: {prompt_type}"
            )

        data = self._load_yaml(filename)
        return data.get("metadata", {})

    def reload(self) -> None:
        """
        Clear the cache and force reload of all prompts.

        Useful for development when prompts are being edited.
        """
        self._cache.clear()
        logger.info("Prompt cache cleared - prompts will be reloaded")

    def list_prompts(self) -> list[str]:
        """
        List all available prompt files.

        Returns:
            List of YAML filenames in the prompts directory
        """
        if not self.prompts_dir.exists():
            return []

        return [
            f.name for f in self.prompts_dir.glob("*.yaml")
            if f.is_file()
        ]


# Singleton instance for easy access
_loader_instance: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """
    Get the global PromptLoader instance (singleton pattern).

    Returns:
        PromptLoader instance
    """
    global _loader_instance

    if _loader_instance is None:
        _loader_instance = PromptLoader()

    return _loader_instance
