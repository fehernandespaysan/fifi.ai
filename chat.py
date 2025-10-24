#!/usr/bin/env python3
"""
Fifi.ai CLI Chatbot Entry Point

Quick launcher for the interactive RAG chatbot.

Usage:
    python chat.py

    or

    python -m src.cli_chatbot
"""

import sys

from src.cli_chatbot import main

if __name__ == "__main__":
    sys.exit(main())
