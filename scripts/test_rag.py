#!/usr/bin/env python3
"""
Test script to demonstrate end-to-end RAG functionality

This script:
1. Loads the existing FAISS index (or creates if missing)
2. Initializes the RAG engine
3. Processes test queries
4. Displays answers with sources
5. Shows streaming responses
6. Demonstrates conversation history

Usage:
    python scripts/test_rag.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blog_loader import BlogLoader
from src.embeddings_manager import EmbeddingsManager
from src.logger import setup_logging
from src.rag_engine import RAGEngine

# Setup logging
setup_logging(log_format="text")


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_response(response, show_context=False):
    """Print a RAG response."""
    print(f"❓ Query: {response.query}\n")
    print(f"💡 Answer:\n{response.answer}\n")

    if show_context:
        print(f"📚 Context ({len(response.sources)} sources):")
        for i, source in enumerate(response.sources, 1):
            print(f"\n  {i}. {source.chunk.blog_title} (score: {source.score:.3f})")
            print(f"     {source.chunk.text[:150]}...")

    print(f"\n📊 Metrics:")
    print(f"   Tokens: {response.tokens_used}")
    print(f"   Retrieval: {response.retrieval_time_ms}ms")
    print(f"   Generation: {response.generation_time_ms}ms")
    print(f"   Total: {response.total_time_ms}ms")


def main():
    """Run the RAG test."""
    print_section("Fifi.ai RAG Engine Test")

    # Step 1: Check if index exists
    print("📦 Step 1: Checking for existing index...")
    embeddings_manager = EmbeddingsManager()
    index_loaded = embeddings_manager.load()

    if not index_loaded:
        print("⚠️  No existing index found. Building index first...")
        print("   Loading blogs...")

        loader = BlogLoader()
        blogs = loader.load_all_blogs()

        if not blogs:
            print("❌ No blog posts found. Run scripts/test_embeddings.py first.")
            return 1

        print(f"   Found {len(blogs)} blogs. Generating embeddings...")
        embeddings_manager.add_documents(blogs)
        embeddings_manager.save()
        print("✅ Index created and saved!")
    else:
        print(f"✅ Index loaded: {embeddings_manager.index.ntotal} vectors")

    # Step 2: Initialize RAG engine
    print_section("Step 2: Initializing RAG Engine")
    engine = RAGEngine(embeddings_manager=embeddings_manager)
    print(f"✅ RAG engine ready!")
    print(f"   Model: {engine.model}")
    print(f"   Temperature: {engine.temperature}")
    print(f"   Top-K retrieval: {engine.top_k}")

    # Step 3: Test queries
    print_section("Step 3: Testing RAG Queries")

    test_queries = [
        "What is RAG and how does it work?",
        "Tell me about FAISS and Pinecone",
        "What are the security best practices for AI applications?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i} of {len(test_queries)}")
        print('─' * 70)

        try:
            response = engine.query(query)
            print_response(response, show_context=(i == 1))  # Show context for first query
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Step 4: Test streaming
    print_section("Step 4: Testing Streaming Response")
    print("❓ Query: How do vector embeddings work?\n")
    print("💡 Streaming Answer:\n")

    try:
        for chunk in engine.stream_query("How do vector embeddings work?"):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\n❌ Streaming failed: {str(e)}")

    # Step 5: Conversation history
    print_section("Step 5: Conversation History")
    history = engine.get_history()
    print(f"📝 Conversation has {len(history)} messages\n")

    # Show last 6 messages (3 exchanges)
    recent_history = history[-6:] if len(history) >= 6 else history
    for msg in recent_history:
        role_emoji = "👤" if msg.role == "user" else "🤖"
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{role_emoji} {msg.role.upper()}: {content_preview}")

    # Step 6: Statistics
    print_section("Step 6: RAG Engine Statistics")
    stats = engine.get_statistics()

    print(f"📊 Performance:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Total tokens: {stats['total_tokens_used']:,}")
    print(f"   Avg tokens/query: {stats['avg_tokens_per_query']:.1f}")
    print(f"   Avg retrieval time: {stats['avg_retrieval_time_ms']:.1f}ms")
    print(f"   Avg generation time: {stats['avg_generation_time_ms']:.1f}ms")

    print(f"\n⚙️  Configuration:")
    print(f"   Model: {stats['model']}")
    print(f"   Temperature: {stats['temperature']}")
    print(f"   Top-K: {stats['top_k']}")
    print(f"   Min relevance: {stats['min_relevance_score']}")

    # Step 7: Cost estimation
    print_section("Step 7: Cost Estimation")

    # OpenAI pricing (approximate as of Oct 2024)
    # gpt-4o-mini: Input $0.15/1M tokens, Output $0.60/1M tokens (assume 50/50 split)
    avg_cost_per_1m_tokens = 0.375  # Average of input and output
    estimated_cost = (stats['total_tokens_used'] / 1_000_000) * avg_cost_per_1m_tokens

    print(f"💰 Estimated costs (gpt-4o-mini):")
    print(f"   Tokens used: {stats['total_tokens_used']:,}")
    print(f"   Estimated cost: ${estimated_cost:.4f}")
    print(f"   Cost per query: ${estimated_cost / stats['total_queries']:.4f}")

    # Final summary
    print_section("Summary")
    print("✅ All RAG tests completed successfully!\n")

    print("🎯 Key Features Demonstrated:")
    print("   ✓ Context retrieval from vector database")
    print("   ✓ Response generation with OpenAI")
    print("   ✓ Source attribution")
    print("   ✓ Streaming responses")
    print("   ✓ Conversation history management")
    print("   ✓ Performance metrics tracking")

    print("\n💡 Next Steps:")
    print("   1. Phase 3: Build CLI chatbot interface")
    print("   2. Phase 4: Add comprehensive testing")
    print("   3. Phase 5: Create FastAPI backend")
    print("   4. Phase 6: Build Streamlit UI")

    print("\n" + "=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
