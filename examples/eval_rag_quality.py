#!/usr/bin/env python3
"""
RAG quality evaluation script for fifi.ai

Loads blog content, generates embeddings, runs test queries, and scores
each answer against expected keywords. Use this to catch regressions in
retrieval quality or answer relevance before shipping changes.

Usage:
    python examples/eval_rag_quality.py

Requires OPENAI_API_KEY to be set. Makes real API calls.

Exit codes:
    0 — all cases passed
    1 — one or more cases failed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

from src.blog_loader import BlogLoader
from src.config import get_config
from src.embeddings_manager import EmbeddingsManager
from src.logger import get_logger, setup_logging
from src.rag_engine import RAGEngine

setup_logging()
logger = get_logger(__name__)


@dataclass
class EvalCase:
    """A single evaluation case."""
    query: str
    expected_keywords: List[str]          # answer must contain ALL of these
    forbidden_keywords: List[str] = field(default_factory=list)  # must contain NONE of these
    description: str = ""


@dataclass
class EvalResult:
    """Result of running one eval case."""
    case: EvalCase
    answer: str
    sources_found: int
    latency_ms: int
    passed: bool
    missing_keywords: List[str]
    forbidden_found: List[str]


def run_eval(engine: RAGEngine, cases: List[EvalCase]) -> List[EvalResult]:
    results = []
    for case in cases:
        start = time.time()
        response = engine.query(case.query)
        latency_ms = int((time.time() - start) * 1000)

        answer_lower = response.answer.lower()

        missing = [kw for kw in case.expected_keywords if kw.lower() not in answer_lower]
        forbidden_found = [kw for kw in case.forbidden_keywords if kw.lower() in answer_lower]
        passed = not missing and not forbidden_found

        results.append(EvalResult(
            case=case,
            answer=response.answer,
            sources_found=len(response.sources),
            latency_ms=latency_ms,
            passed=passed,
            missing_keywords=missing,
            forbidden_found=forbidden_found,
        ))
    return results


def print_results(results: List[EvalResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  RAG Quality Eval — {passed}/{total} passed")
    print(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        status = "PASS" if result.passed else "FAIL"
        label = f"[{status}]"
        desc = f" ({result.case.description})" if result.case.description else ""
        print(f"{label} {i}. {result.case.query[:70]}{desc}")

        if not result.passed:
            if result.missing_keywords:
                print(f"       Missing keywords: {result.missing_keywords}")
            if result.forbidden_found:
                print(f"       Forbidden found:  {result.forbidden_found}")
            # Show a short excerpt of the answer for diagnosis
            excerpt = result.answer[:200].replace("\n", " ")
            print(f"       Answer excerpt:   {excerpt}...")

        print(f"       Sources: {result.sources_found}  |  Latency: {result.latency_ms}ms\n")

    avg_latency = sum(r.latency_ms for r in results) // total if total else 0
    print(f"Average latency: {avg_latency}ms")
    print(f"{'='*60}\n")


def build_eval_cases(blogs_dir: Path) -> List[EvalCase]:
    """
    Build eval cases from the available blog content.

    If no blog files are found (e.g. only placeholder templates), the eval
    prints a warning and returns a minimal smoke-test set that checks the
    no-context fallback path instead.
    """
    # Check for real blog content (not the placeholder templates)
    md_files = list(blogs_dir.glob("*.md"))
    placeholder_markers = {"DELETE THIS FILE", "EXAMPLE BLOG POST"}

    real_blogs = []
    for f in md_files:
        text = f.read_text(errors="ignore")
        if not any(marker in text for marker in placeholder_markers):
            real_blogs.append(f)

    if not real_blogs:
        print(
            "\nNote: No user blog content found — running smoke tests only.\n"
            "Add .md files to blogs/ and re-run for content-grounded eval.\n"
        )
        # Smoke test: verify no-context fallback works correctly
        return [
            EvalCase(
                query="what does this knowledge base cover",
                expected_keywords=[],  # answer will vary; just check it doesn't crash
                description="no-context smoke test",
            ),
        ]

    # Build cases from detected topic keywords in real blog content
    cases: List[EvalCase] = []
    all_text = " ".join(f.read_text(errors="ignore").lower() for f in real_blogs)

    # Probe for common topics and add relevant cases
    if "vector" in all_text or "embedding" in all_text:
        cases.append(EvalCase(
            query="What are vector embeddings and how are they used?",
            expected_keywords=["vector", "embedding"],
            description="vector/embedding retrieval",
        ))

    if "rag" in all_text or "retrieval" in all_text:
        cases.append(EvalCase(
            query="How does retrieval-augmented generation work?",
            expected_keywords=["retrieval"],
            description="RAG concept retrieval",
        ))

    if "faiss" in all_text or "pinecone" in all_text:
        cases.append(EvalCase(
            query="What vector database options are available?",
            expected_keywords=["faiss"],
            description="vector DB retrieval",
        ))

    if "security" in all_text or "api key" in all_text:
        cases.append(EvalCase(
            query="What are security best practices?",
            expected_keywords=["security"],
            description="security retrieval",
        ))

    # Hallucination guard: completely out-of-scope question
    cases.append(EvalCase(
        query="What is the capital of France?",
        expected_keywords=[],
        forbidden_keywords=[],  # just verify it doesn't crash
        description="out-of-scope query (no hallucination check)",
    ))

    if not cases:
        # Fallback: at least one generic case
        cases.append(EvalCase(
            query="What topics does this knowledge base cover?",
            expected_keywords=[],
            description="generic coverage check",
        ))

    return cases


def main() -> int:
    config = get_config()

    if not config.openai_api_key or config.openai_api_key.startswith("sk-your"):
        print("Error: OPENAI_API_KEY not set. Set it in .env and retry.")
        return 1

    print("Loading blogs...")
    loader = BlogLoader()
    try:
        blogs = loader.load_all_blogs()
        print(f"  Loaded {len(blogs)} blog(s)")
    except Exception as e:
        print(f"Error loading blogs: {e}")
        return 1

    if not blogs:
        print("No blogs found. Add .md files to the blogs/ directory.")
        return 1

    print("Generating embeddings (this makes API calls)...")
    embeddings_manager = EmbeddingsManager()
    embeddings_manager.add_documents(blogs)
    stats = embeddings_manager.get_statistics()
    print(f"  Indexed {stats['total_vectors']} chunks from {len(blogs)} blog(s)")

    engine = RAGEngine(embeddings_manager=embeddings_manager)

    cases = build_eval_cases(config.blogs_directory)
    print(f"\nRunning {len(cases)} eval case(s)...\n")

    results = run_eval(engine, cases)
    print_results(results)

    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
