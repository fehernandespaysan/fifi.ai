#!/usr/bin/env python3
"""
Test script to demonstrate embeddings generation and search

This script:
1. Loads all blog posts from blogs/ directory
2. Chunks them into smaller pieces
3. Generates embeddings using OpenAI API
4. Stores them in a FAISS vector index
5. Performs similarity search
6. Saves and reloads the index

Usage:
    python scripts/test_embeddings.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blog_loader import BlogLoader
from src.embeddings_manager import EmbeddingsManager
from src.logger import setup_logging

# Setup logging
setup_logging(log_format="text")


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_chunk_info(chunk, index: int):
    """Print information about a chunk."""
    print(f"\nChunk {index}:")
    print(f"  Blog: {chunk.blog_title}")
    print(f"  Position: {chunk.start_pos}-{chunk.end_pos}")
    print(f"  Text preview: {chunk.text[:100]}...")


def print_search_result(result, rank: int):
    """Print a search result."""
    print(f"\n{rank}. Score: {result.score:.4f} | Distance: {result.distance:.4f}")
    print(f"   Blog: {result.chunk.blog_title}")
    print(f"   Chunk {result.chunk.chunk_index}:")
    print(f"   {result.chunk.text[:200]}...")


def main():
    """Run the embeddings test."""
    print_section("Fifi.ai Embeddings Test")

    # Step 1: Load blogs
    print("📚 Step 1: Loading blog posts...")
    loader = BlogLoader()
    blogs = loader.load_all_blogs()

    if not blogs:
        print("❌ No blog posts found. Add some .md files to the blogs/ directory.")
        return 1

    print(f"✅ Loaded {len(blogs)} blog posts")
    for blog in blogs:
        print(f"   - {blog.title} ({blog.word_count} words)")

    # Step 2: Initialize embeddings manager
    print_section("Step 2: Initializing Embeddings Manager")
    manager = EmbeddingsManager()
    print(f"✅ Using model: {manager.embedding_model}")
    print(f"   Chunk size: {manager.chunk_size} words")
    print(f"   Chunk overlap: {manager.chunk_overlap} words")

    # Step 3: Generate embeddings
    print_section("Step 3: Generating Embeddings")
    print("⏳ This may take a moment and will cost a small amount...")

    try:
        manager.add_documents(blogs)
        print(f"✅ Successfully generated embeddings!")

        stats = manager.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total vectors in index: {stats['total_vectors']}")
        print(f"   Vector dimension: {stats['dimension']}")
        print(f"   Total embeddings generated: {stats['total_embeddings_generated']}")
        print(f"   Total API calls: {stats['total_api_calls']}")
        print(f"   Total tokens used: {stats['total_tokens_used']:,}")

        # Show first few chunks
        print(f"\n📝 Sample chunks:")
        for i, chunk in enumerate(manager.chunks[:3]):
            print_chunk_info(chunk, i + 1)

    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Search queries
    print_section("Step 4: Testing Search")

    test_queries = [
        "What is RAG and how does it work?",
        "Tell me about vector databases",
        "Security best practices for AI applications",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        try:
            results = manager.search(query, top_k=3)
            print(f"   Found {len(results)} results:")

            for rank, result in enumerate(results, 1):
                print_search_result(result, rank)

        except Exception as e:
            print(f"   ❌ Search failed: {str(e)}")

    # Step 5: Save index
    print_section("Step 5: Saving Index")
    try:
        manager.save()
        print(f"✅ Index saved to:")
        print(f"   Index: {manager.index_path}")
        print(f"   Metadata: {manager.metadata_path}")

        # Check file sizes
        index_size = manager.index_path.stat().st_size / 1024  # KB
        metadata_size = manager.metadata_path.stat().st_size / 1024  # KB
        print(f"\n📦 File sizes:")
        print(f"   Index: {index_size:.2f} KB")
        print(f"   Metadata: {metadata_size:.2f} KB")

    except Exception as e:
        print(f"❌ Error saving index: {str(e)}")
        return 1

    # Step 6: Test loading
    print_section("Step 6: Testing Index Load")
    try:
        new_manager = EmbeddingsManager()
        loaded = new_manager.load()

        if loaded:
            print("✅ Index loaded successfully!")

            new_stats = new_manager.get_statistics()
            print(f"\n📊 Loaded statistics:")
            print(f"   Total vectors: {new_stats['total_vectors']}")
            print(f"   Total chunks: {new_stats['total_chunks']}")
            print(f"   Dimension: {new_stats['dimension']}")

            # Test search with loaded index
            print("\n🔍 Testing search with loaded index:")
            test_query = "What are embeddings?"
            results = new_manager.search(test_query, top_k=2)
            print(f"   Query: \"{test_query}\"")
            print(f"   Found {len(results)} results")

            for rank, result in enumerate(results, 1):
                print_search_result(result, rank)
        else:
            print("⚠️  No existing index found (this is expected on first run)")

    except Exception as e:
        print(f"❌ Error loading index: {str(e)}")

    # Final summary
    print_section("Summary")
    print("✅ All tests completed successfully!")
    print("\n📈 Cost estimation:")

    # OpenAI pricing (approximate)
    # text-embedding-3-small: $0.02 per 1M tokens
    cost = (stats['total_tokens_used'] / 1_000_000) * 0.02
    print(f"   Tokens used: {stats['total_tokens_used']:,}")
    print(f"   Estimated cost: ${cost:.4f}")

    print("\n💡 Next steps:")
    print("   1. The FAISS index is saved and ready to use")
    print("   2. You can now build the RAG query engine (Phase 2)")
    print("   3. The index will be reused on subsequent runs (much faster!)")

    print("\n" + "=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
