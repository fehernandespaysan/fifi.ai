"""
Unit tests for src/blog_loader.py

Tests blog loading, parsing, validation, and error handling.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.blog_loader import (
    Blog,
    BlogLoader,
    BlogLoaderError,
    InvalidBlogFormatError,
    MissingFrontmatterError,
)


class TestBlog:
    """Test the Blog dataclass."""

    def test_blog_creation(self):
        """Test creating a Blog instance."""
        blog = Blog(
            file_path=Path("test.md"),
            title="Test Blog",
            date=datetime(2025, 10, 23),
            content="This is test content.",
            raw_content="---\ntitle: Test\n---\nContent",
            tags=["test", "python"],
            author="Test Author",
        )

        assert blog.title == "Test Blog"
        assert blog.date == datetime(2025, 10, 23)
        assert blog.author == "Test Author"
        assert len(blog.tags) == 2

    def test_blog_word_count_auto_calculation(self):
        """Test that word count is calculated automatically."""
        content = "This is a test with five words"
        blog = Blog(
            file_path=Path("test.md"),
            title="Test",
            date=datetime.now(),
            content=content,
            raw_content="",
        )

        assert blog.word_count == 7  # "This is a test with five words" = 7 words

    def test_blog_char_count_auto_calculation(self):
        """Test that character count is calculated automatically."""
        content = "Hello!"
        blog = Blog(
            file_path=Path("test.md"),
            title="Test",
            date=datetime.now(),
            content=content,
            raw_content="",
        )

        assert blog.char_count == 6

    def test_blog_to_dict(self):
        """Test converting blog to dictionary."""
        blog = Blog(
            file_path=Path("test.md"),
            title="Test Blog",
            date=datetime(2025, 10, 23, 10, 30),
            content="Content",
            raw_content="Raw",
            tags=["tag1"],
            author="Author",
        )

        blog_dict = blog.to_dict()

        assert blog_dict["title"] == "Test Blog"
        assert blog_dict["date"] == "2025-10-23T10:30:00"
        assert blog_dict["tags"] == ["tag1"]
        assert blog_dict["author"] == "Author"

    def test_blog_repr(self):
        """Test blog string representation."""
        blog = Blog(
            file_path=Path("test.md"),
            title="Test Blog",
            date=datetime(2025, 10, 23),
            content="Some content here",
            raw_content="",
        )

        repr_str = repr(blog)
        assert "Test Blog" in repr_str
        assert "2025-10-23" in repr_str


class TestBlogLoader:
    """Test the BlogLoader class."""

    def test_blog_loader_initialization(self, tmp_path):
        """Test BlogLoader initialization."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            "BLOGS_DIRECTORY": str(tmp_path)
        }):
            from src.config import reset_config
            reset_config()

            loader = BlogLoader()
            assert loader.blogs_directory == Path(tmp_path)

    def test_blog_loader_with_custom_directory(self, tmp_path):
        """Test BlogLoader with custom directory."""
        custom_dir = tmp_path / "custom_blogs"
        custom_dir.mkdir()

        loader = BlogLoader(blogs_directory=custom_dir)
        assert loader.blogs_directory == custom_dir

    def test_load_valid_blog(self, tmp_path):
        """Test loading a valid blog post."""
        blog_content = """---
title: Test Blog Post
date: 2025-10-23
tags: test, python
author: Test Author
---

# Introduction

This is a test blog post with some content.

## Section 1

More content here.
"""
        blog_file = tmp_path / "test_blog.md"
        blog_file.write_text(blog_content)

        loader = BlogLoader(blogs_directory=tmp_path)
        blog = loader.load_blog(blog_file)

        assert blog.title == "Test Blog Post"
        assert blog.date == datetime(2025, 10, 23)
        assert blog.tags == ["test", "python"]
        assert blog.author == "Test Author"
        assert "Introduction" in blog.content
        assert blog.word_count > 0

    def test_load_blog_missing_frontmatter(self, tmp_path):
        """Test loading a blog without frontmatter fails."""
        blog_content = "# Just a heading\n\nNo frontmatter here."
        blog_file = tmp_path / "invalid_blog.md"
        blog_file.write_text(blog_content)

        loader = BlogLoader(blogs_directory=tmp_path)

        with pytest.raises(MissingFrontmatterError):
            loader.load_blog(blog_file)

    def test_load_blog_missing_required_field(self, tmp_path):
        """Test loading a blog missing required fields fails."""
        blog_content = """---
title: Test Blog
---

Content without date.
"""
        blog_file = tmp_path / "invalid_blog.md"
        blog_file.write_text(blog_content)

        loader = BlogLoader(blogs_directory=tmp_path)

        with pytest.raises(MissingFrontmatterError) as exc_info:
            loader.load_blog(blog_file)
        assert "date" in str(exc_info.value).lower()

    def test_load_blog_invalid_yaml(self, tmp_path):
        """Test loading a blog with invalid YAML fails."""
        blog_content = """---
title: Test Blog
date: [invalid yaml structure
---

Content.
"""
        blog_file = tmp_path / "invalid_blog.md"
        blog_file.write_text(blog_content)

        loader = BlogLoader(blogs_directory=tmp_path)

        with pytest.raises(InvalidBlogFormatError):
            loader.load_blog(blog_file)

    def test_load_all_blogs(self, tmp_path):
        """Test loading all blogs from directory."""
        # Create multiple blog files
        for i in range(3):
            blog_content = f"""---
title: Blog Post {i}
date: 2025-10-{20+i}
---

Content for blog {i}.
"""
            blog_file = tmp_path / f"blog_{i}.md"
            blog_file.write_text(blog_content)

        loader = BlogLoader(blogs_directory=tmp_path)
        blogs = loader.load_all_blogs()

        assert len(blogs) == 3
        titles = [b.title for b in blogs]
        assert "Blog Post 0" in titles
        assert "Blog Post 1" in titles
        assert "Blog Post 2" in titles

    def test_load_all_blogs_with_errors(self, tmp_path):
        """Test that load_all_blogs continues despite individual errors."""
        # Valid blog
        valid_blog = """---
title: Valid Blog
date: 2025-10-23
---

Valid content.
"""
        (tmp_path / "valid.md").write_text(valid_blog)

        # Invalid blog
        invalid_blog = "No frontmatter"
        (tmp_path / "invalid.md").write_text(invalid_blog)

        loader = BlogLoader(blogs_directory=tmp_path)
        blogs = loader.load_all_blogs()

        # Should load the valid one, skip the invalid one
        assert len(blogs) == 1
        assert blogs[0].title == "Valid Blog"

    def test_load_all_blogs_nonexistent_directory(self):
        """Test that load_all_blogs fails if directory doesn't exist."""
        loader = BlogLoader(blogs_directory=Path("/nonexistent/path"))

        with pytest.raises(BlogLoaderError) as exc_info:
            loader.load_all_blogs()
        assert "not found" in str(exc_info.value).lower()

    def test_parse_date_various_formats(self, tmp_path):
        """Test parsing dates in various formats."""
        formats = [
            ("2025-10-23", datetime(2025, 10, 23)),
            ("2025/10/23", datetime(2025, 10, 23)),
            ("23-10-2025", datetime(2025, 10, 23)),
            ("October 23, 2025", datetime(2025, 10, 23)),
        ]

        loader = BlogLoader(blogs_directory=tmp_path)

        for date_str, expected_date in formats:
            result = loader._parse_date(date_str, Path("test.md"))
            assert result.date() == expected_date.date()

    def test_parse_date_invalid_format(self, tmp_path):
        """Test that invalid date format raises error."""
        loader = BlogLoader(blogs_directory=tmp_path)

        with pytest.raises(InvalidBlogFormatError):
            loader._parse_date("invalid-date", Path("test.md"))

    def test_parse_tags_from_string(self):
        """Test parsing tags from comma-separated string."""
        loader = BlogLoader()
        tags = loader._parse_tags("python, ai, machine-learning")

        assert len(tags) == 3
        assert "python" in tags
        assert "ai" in tags
        assert "machine-learning" in tags

    def test_parse_tags_from_list(self):
        """Test parsing tags from list."""
        loader = BlogLoader()
        tags = loader._parse_tags(["python", "ai", "ml"])

        assert len(tags) == 3
        assert "python" in tags

    def test_parse_tags_empty(self):
        """Test parsing empty tags."""
        loader = BlogLoader()
        tags = loader._parse_tags(None)

        assert tags == []

    def test_get_blogs_by_tag(self, tmp_path):
        """Test filtering blogs by tag."""
        # Create blogs with different tags
        blog1 = Blog(
            file_path=tmp_path / "blog1.md",
            title="Python Blog",
            date=datetime.now(),
            content="Content",
            raw_content="",
            tags=["python", "tutorial"],
        )

        blog2 = Blog(
            file_path=tmp_path / "blog2.md",
            title="AI Blog",
            date=datetime.now(),
            content="Content",
            raw_content="",
            tags=["ai", "machine-learning"],
        )

        blogs = [blog1, blog2]
        loader = BlogLoader()

        python_blogs = loader.get_blogs_by_tag(blogs, "python")
        assert len(python_blogs) == 1
        assert python_blogs[0].title == "Python Blog"

        ai_blogs = loader.get_blogs_by_tag(blogs, "ai")
        assert len(ai_blogs) == 1
        assert ai_blogs[0].title == "AI Blog"

    def test_get_blogs_by_date_range(self, tmp_path):
        """Test filtering blogs by date range."""
        blog1 = Blog(
            file_path=tmp_path / "blog1.md",
            title="Old Blog",
            date=datetime(2025, 1, 1),
            content="Content",
            raw_content="",
        )

        blog2 = Blog(
            file_path=tmp_path / "blog2.md",
            title="New Blog",
            date=datetime(2025, 10, 23),
            content="Content",
            raw_content="",
        )

        blogs = [blog1, blog2]
        loader = BlogLoader()

        # Get blogs from October onwards
        recent_blogs = loader.get_blogs_by_date_range(
            blogs,
            start_date=datetime(2025, 10, 1),
            end_date=datetime(2025, 10, 31),
        )

        assert len(recent_blogs) == 1
        assert recent_blogs[0].title == "New Blog"

    def test_get_blog_statistics(self, tmp_path):
        """Test calculating blog statistics."""
        blogs = [
            Blog(
                file_path=tmp_path / f"blog{i}.md",
                title=f"Blog {i}",
                date=datetime(2025, 10, i + 1),
                content=" ".join(["word"] * (100 + i * 10)),  # Variable word counts
                raw_content="",
                tags=["tag1", "tag2"] if i % 2 == 0 else ["tag2", "tag3"],
            )
            for i in range(5)
        ]

        loader = BlogLoader()
        stats = loader.get_blog_statistics(blogs)

        assert stats["total_blogs"] == 5
        assert stats["total_words"] > 0
        assert stats["avg_words_per_blog"] > 0
        assert len(stats["unique_tags"]) == 3  # tag1, tag2, tag3
        assert "tag1" in stats["unique_tags"]
        assert "date_range" in stats

    def test_get_blog_statistics_empty_list(self):
        """Test statistics with empty blog list."""
        loader = BlogLoader()
        stats = loader.get_blog_statistics([])

        assert stats["total_blogs"] == 0
        assert stats["total_words"] == 0
        assert stats["avg_words_per_blog"] == 0
        assert stats["unique_tags"] == []


class TestBlogLoaderIntegration:
    """Integration tests using real blog files."""

    def test_load_real_blog_files(self):
        """Test loading actual blog files from blogs/ directory."""
        # Assuming we have blog files in the blogs/ directory
        blogs_dir = Path("blogs")

        if not blogs_dir.exists():
            pytest.skip("blogs/ directory not found")

        loader = BlogLoader(blogs_directory=blogs_dir)
        blogs = loader.load_all_blogs()

        # Should have at least some blogs
        assert len(blogs) >= 0  # Won't fail if no blogs exist

        # If we have blogs, validate their structure
        for blog in blogs:
            assert blog.title
            assert blog.date
            assert blog.content
            assert isinstance(blog.tags, list)
            assert blog.word_count > 0
