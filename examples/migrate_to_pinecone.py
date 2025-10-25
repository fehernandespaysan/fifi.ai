#!/usr/bin/env python3
"""
Migrate from FAISS to Pinecone Vector Store

This script helps you migrate your existing FAISS index to Pinecone cloud storage.

Usage:
    python examples/migrate_to_pinecone.py

Prerequisites:
    1. Set PINECONE_API_KEY in your .env file
    2. Have an existing FAISS index to migrate (or use --rebuild to create fresh)

Options:
    --rebuild: Rebuild index from blog posts instead of migrating existing FAISS index
    --dry-run: Show what would be migrated without actually doing it
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blog_loader import BlogLoader
from src.config import Config, VectorStoreType, get_config
from src.embeddings_manager import EmbeddingsManager
from src.logger import get_logger
from src.vector_store import FAISSVectorStore, PineconeVectorStore, create_vector_store

logger = get_logger(__name__)


def validate_pinecone_config(config: Config) -> bool:
    """Validate Pinecone configuration."""
    if not config.pinecone_api_key:
        print("❌ Error: PINECONE_API_KEY not set in .env file")
        print("   Get your API key from: https://app.pinecone.io/")
        return False

    if not config.pinecone_environment:
        print("❌ Error: PINECONE_ENVIRONMENT not set in .env file")
        return False

    if not config.pinecone_index_name:
        print("❌ Error: PINECONE_INDEX_NAME not set in .env file")
        return False

    return True


def migrate_from_faiss(dry_run: bool = False) -> bool:
    """
    Migrate existing FAISS index to Pinecone.

    Args:
        dry_run: If True, show what would be done without actually migrating

    Returns:
        True if migration successful, False otherwise
    """
    config = get_config()

    print("=" * 70)
    print("  FAISS → Pinecone Migration")
    print("=" * 70)
    print()

    # Validate Pinecone config
    if not validate_pinecone_config(config):
        return False

    # Check if FAISS index exists
    if not config.faiss_index_path.exists():
        print("❌ Error: FAISS index not found at", config.faiss_index_path)
        print("   Use --rebuild to create a fresh index from blog posts")
        return False

    print("📊 Migration Plan:")
    print(f"   Source: FAISS ({config.faiss_index_path})")
    print(f"   Target: Pinecone ({config.pinecone_index_name})")
    print(f"   Environment: {config.pinecone_environment}")
    print()

    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()

    # Load FAISS index
    print("Step 1: Loading FAISS index...")
    try:
        faiss_em = EmbeddingsManager(vector_store=create_vector_store(config, VectorStoreType.FAISS))
        loaded = faiss_em.load()

        if not loaded:
            print("❌ Failed to load FAISS index")
            return False

        faiss_stats = faiss_em.get_statistics()
        print(f"✅ Loaded FAISS index:")
        print(f"   Vectors: {faiss_stats['total_vectors']}")
        print(f"   Chunks: {faiss_stats['total_chunks']}")
        print()

    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return False

    if dry_run:
        print("✅ Dry run complete - migration plan validated")
        return True

    # Create Pinecone vector store
    print("Step 2: Connecting to Pinecone...")
    try:
        pinecone_store = create_vector_store(config, VectorStoreType.PINECONE)
        pinecone_stats = pinecone_store.get_stats()
        print(f"✅ Connected to Pinecone index: {config.pinecone_index_name}")
        print(f"   Current vectors: {pinecone_stats.get('total_vectors', 0)}")
        print()

    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return False

    # Migrate data
    print("Step 3: Migrating vectors to Pinecone...")
    print("   Note: This will re-index all blog posts")
    print()

    try:
        # Load blog posts
        blog_loader = BlogLoader()
        blogs = blog_loader.load_all_blogs()
        print(f"✅ Loaded {len(blogs)} blog posts")

        # Create new embeddings manager with Pinecone
        pinecone_em = EmbeddingsManager(vector_store=pinecone_store)

        # Add documents
        print("   Generating embeddings and uploading to Pinecone...")
        print("   (This may take a few moments...)")
        pinecone_em.add_documents(blogs)

        # Get final stats
        final_stats = pinecone_em.get_statistics()
        print(f"✅ Migration complete!")
        print(f"   Vectors: {final_stats['total_vectors']}")
        print(f"   Chunks: {final_stats['total_chunks']}")
        print(f"   Tokens used: {final_stats['total_tokens_used']:,}")
        print()

    except Exception as e:
        print(f"❌ Error during migration: {e}")
        logger.error("Migration failed", exc_info=True)
        return False

    # Test search
    print("Step 4: Testing Pinecone search...")
    try:
        results = pinecone_em.search("What is RAG?", top_k=3)
        print(f"✅ Search working! Found {len(results)} results")
        if results:
            print(f"   Top result: {results[0].chunk.blog_title} (score: {results[0].score:.3f})")
        print()

    except Exception as e:
        print(f"⚠️  Warning: Search test failed: {e}")

    print("=" * 70)
    print("🎉 Migration Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Update your .env file: VECTOR_STORE_TYPE=pinecone")
    print("  2. Test your application with Pinecone")
    print("  3. (Optional) Keep FAISS files as backup or delete them")
    print()

    return True


def rebuild_fresh(dry_run: bool = False) -> bool:
    """
    Build a fresh index in Pinecone from blog posts.

    Args:
        dry_run: If True, show what would be done without actually building

    Returns:
        True if successful, False otherwise
    """
    config = get_config()

    print("=" * 70)
    print("  Build Fresh Pinecone Index")
    print("=" * 70)
    print()

    # Validate Pinecone config
    if not validate_pinecone_config(config):
        return False

    print("📊 Build Plan:")
    print(f"   Target: Pinecone ({config.pinecone_index_name})")
    print(f"   Environment: {config.pinecone_environment}")
    print()

    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()

    # Load blog posts
    print("Step 1: Loading blog posts...")
    try:
        blog_loader = BlogLoader()
        blogs = blog_loader.load_all_blogs()
        print(f"✅ Loaded {len(blogs)} blog posts")
        for blog in blogs:
            print(f"   - {blog.title}")
        print()

    except Exception as e:
        print(f"❌ Error loading blogs: {e}")
        return False

    if dry_run:
        print("✅ Dry run complete - build plan validated")
        return True

    # Create Pinecone index
    print("Step 2: Creating/connecting to Pinecone index...")
    try:
        pinecone_store = create_vector_store(config, VectorStoreType.PINECONE)
        print(f"✅ Connected to Pinecone index: {config.pinecone_index_name}")
        print()

    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return False

    # Build index
    print("Step 3: Building index...")
    print("   (This may take a few moments...)")
    try:
        em = EmbeddingsManager(vector_store=pinecone_store)
        em.add_documents(blogs)

        stats = em.get_statistics()
        print(f"✅ Index built successfully!")
        print(f"   Vectors: {stats['total_vectors']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Tokens used: {stats['total_tokens_used']:,}")
        print()

    except Exception as e:
        print(f"❌ Error building index: {e}")
        logger.error("Build failed", exc_info=True)
        return False

    # Test search
    print("Step 4: Testing search...")
    try:
        results = em.search("What is RAG?", top_k=3)
        print(f"✅ Search working! Found {len(results)} results")
        if results:
            print(f"   Top result: {results[0].chunk.blog_title} (score: {results[0].score:.3f})")
        print()

    except Exception as e:
        print(f"⚠️  Warning: Search test failed: {e}")

    print("=" * 70)
    print("🎉 Build Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Update your .env file: VECTOR_STORE_TYPE=pinecone")
    print("  2. Test your application with Pinecone")
    print()

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate FAISS index to Pinecone or build fresh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from blog posts instead of migrating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    try:
        if args.rebuild:
            success = rebuild_fresh(dry_run=args.dry_run)
        else:
            success = migrate_from_faiss(dry_run=args.dry_run)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logger.error("Unexpected error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
