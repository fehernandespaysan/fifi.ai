#!/usr/bin/env python3
"""
Test script to load and display blog statistics

Usage:
    python scripts/test_blog_loading.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blog_loader import BlogLoader
from src.logger import setup_logging

# Setup logging
setup_logging(log_format="text")

def main():
    """Load blogs and display statistics."""
    print("\n" + "=" * 70)
    print("  Fifi.ai Blog Loader Test")
    print("=" * 70 + "\n")

    # Initialize loader
    loader = BlogLoader()
    print(f"📁 Loading blogs from: {loader.blogs_directory}\n")

    # Load all blogs
    try:
        blogs = loader.load_all_blogs()
        print(f"✅ Successfully loaded {len(blogs)} blog posts\n")

        if not blogs:
            print("⚠️  No blog posts found. Add some .md files to the blogs/ directory.")
            return

        # Display individual blogs
        print("📚 Blog Posts:")
        print("-" * 70)
        for i, blog in enumerate(blogs, 1):
            print(f"\n{i}. {blog.title}")
            print(f"   📅 Date: {blog.date.strftime('%Y-%m-%d')}")
            print(f"   👤 Author: {blog.author or 'N/A'}")
            print(f"   🏷️  Tags: {', '.join(blog.tags) if blog.tags else 'None'}")
            print(f"   📝 Words: {blog.word_count}")
            print(f"   📄 Characters: {blog.char_count}")

        # Display statistics
        print("\n" + "=" * 70)
        print("  Statistics")
        print("=" * 70)

        stats = loader.get_blog_statistics(blogs)
        print(f"\n📊 Total Blogs: {stats['total_blogs']}")
        print(f"📝 Total Words: {stats['total_words']:,}")
        print(f"📄 Total Characters: {stats['total_chars']:,}")
        print(f"📈 Average Words/Blog: {stats['avg_words_per_blog']:,}")

        if stats['unique_tags']:
            print(f"\n🏷️  Unique Tags ({len(stats['unique_tags'])}):")
            for tag in stats['unique_tags']:
                tag_blogs = loader.get_blogs_by_tag(blogs, tag)
                print(f"   - {tag}: {len(tag_blogs)} post(s)")

        if stats['date_range']:
            print(f"\n📅 Date Range:")
            print(f"   Earliest: {stats['date_range']['earliest']}")
            print(f"   Latest: {stats['date_range']['latest']}")

        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"❌ Error loading blogs: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
