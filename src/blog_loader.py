"""
Blog loader for Fifi.ai

Loads and parses markdown blog posts with YAML frontmatter.
Validates structure and handles errors gracefully.

Expected blog format:
    ---
    title: Blog Post Title
    date: 2025-10-23
    tags: tag1, tag2, tag3
    author: Author Name (optional)
    ---

    # Blog content in markdown

    Content goes here...

Usage:
    from src.blog_loader import BlogLoader, Blog

    loader = BlogLoader()
    blogs = loader.load_all_blogs()

    for blog in blogs:
        print(f"{blog.title}: {blog.word_count} words")
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.config import get_config
from src.logger import get_logger, LoggerContext

logger = get_logger(__name__)


@dataclass
class Blog:
    """
    Represents a blog post with metadata and content.

    Attributes:
        file_path: Path to the blog file
        title: Blog post title
        date: Publication date
        content: Markdown content (without frontmatter)
        raw_content: Full raw content including frontmatter
        tags: List of tags
        author: Author name (optional)
        metadata: Additional metadata from frontmatter
        word_count: Number of words in content
        char_count: Number of characters in content
    """

    file_path: Path
    title: str
    date: datetime
    content: str
    raw_content: str
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.char_count == 0:
            self.char_count = len(self.content)

    def to_dict(self) -> Dict:
        """
        Convert blog to dictionary for serialization.

        Returns:
            Dict with all blog fields
        """
        return {
            "file_path": str(self.file_path),
            "title": self.title,
            "date": self.date.isoformat(),
            "content": self.content,
            "tags": self.tags,
            "author": self.author,
            "metadata": self.metadata,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Blog(title='{self.title}', date={self.date.date()}, words={self.word_count})"


class BlogLoaderError(Exception):
    """Base exception for blog loader errors."""
    pass


class InvalidBlogFormatError(BlogLoaderError):
    """Raised when blog format is invalid."""
    pass


class MissingFrontmatterError(InvalidBlogFormatError):
    """Raised when blog is missing required frontmatter."""
    pass


class BlogLoader:
    """
    Loads and parses markdown blog posts.

    Handles:
    - YAML frontmatter parsing
    - Content validation
    - Error handling
    - Metrics logging
    """

    def __init__(self, blogs_directory: Optional[Path] = None):
        """
        Initialize the blog loader.

        Args:
            blogs_directory: Directory containing blog files (uses config if None)
        """
        config = get_config()
        self.blogs_directory = blogs_directory or config.blogs_directory
        self.supported_extensions = config.blog_extensions

        logger.info(
            "BlogLoader initialized",
            extra={
                "blogs_directory": str(self.blogs_directory),
                "supported_extensions": self.supported_extensions,
            },
        )

    def load_all_blogs(self) -> List[Blog]:
        """
        Load all blog posts from the blogs directory.

        Returns:
            List of Blog objects

        Raises:
            BlogLoaderError: If directory doesn't exist or can't be read
        """
        with LoggerContext() as correlation_id:
            logger.info(
                "Loading all blogs",
                extra={"directory": str(self.blogs_directory), "correlation_id": correlation_id},
            )

            if not self.blogs_directory.exists():
                error_msg = f"Blogs directory not found: {self.blogs_directory}"
                logger.error(error_msg)
                raise BlogLoaderError(error_msg)

            blogs = []
            errors = []

            # Find all markdown files
            for extension in self.supported_extensions:
                for file_path in self.blogs_directory.glob(f"**/*{extension}"):
                    try:
                        blog = self.load_blog(file_path)
                        blogs.append(blog)
                        logger.debug(
                            "Blog loaded successfully",
                            extra={
                                "file": file_path.name,
                                "title": blog.title,
                                "words": blog.word_count,
                            },
                        )
                    except Exception as e:
                        errors.append({"file": str(file_path), "error": str(e)})
                        logger.warning(
                            f"Failed to load blog: {file_path.name}",
                            extra={"file": str(file_path), "error": str(e)},
                        )

            # Log summary
            logger.info(
                "Finished loading blogs",
                extra={
                    "total_loaded": len(blogs),
                    "errors": len(errors),
                    "total_words": sum(b.word_count for b in blogs),
                },
            )

            return blogs

    def load_blog(self, file_path: Path) -> Blog:
        """
        Load a single blog post from a file.

        Args:
            file_path: Path to the blog file

        Returns:
            Blog object

        Raises:
            InvalidBlogFormatError: If blog format is invalid
            MissingFrontmatterError: If required frontmatter is missing
        """
        logger.debug(f"Loading blog from {file_path}")

        try:
            raw_content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise BlogLoaderError(f"Failed to read file {file_path}: {str(e)}")

        # Parse frontmatter and content
        frontmatter, content = self._parse_frontmatter(raw_content, file_path)

        # Validate required fields
        self._validate_frontmatter(frontmatter, file_path)

        # Parse date
        date = self._parse_date(frontmatter.get("date"), file_path)

        # Parse tags
        tags = self._parse_tags(frontmatter.get("tags", []))

        # Create Blog object
        blog = Blog(
            file_path=file_path,
            title=frontmatter["title"],
            date=date,
            content=content.strip(),
            raw_content=raw_content,
            tags=tags,
            author=frontmatter.get("author"),
            metadata=frontmatter,
        )

        return blog

    def _parse_frontmatter(self, raw_content: str, file_path: Path) -> tuple[Dict, str]:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            raw_content: Raw markdown content
            file_path: Path to file (for error messages)

        Returns:
            Tuple of (frontmatter_dict, content)

        Raises:
            MissingFrontmatterError: If frontmatter not found or invalid
        """
        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, raw_content, re.DOTALL)

        if not match:
            raise MissingFrontmatterError(
                f"No valid frontmatter found in {file_path.name}. "
                "Expected format:\n---\ntitle: Title\ndate: YYYY-MM-DD\n---"
            )

        frontmatter_str, content = match.groups()

        # Parse YAML
        try:
            frontmatter = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as e:
            raise InvalidBlogFormatError(
                f"Invalid YAML in frontmatter of {file_path.name}: {str(e)}"
            )

        return frontmatter, content

    def _validate_frontmatter(self, frontmatter: Dict, file_path: Path) -> None:
        """
        Validate required frontmatter fields.

        Args:
            frontmatter: Parsed frontmatter dictionary
            file_path: Path to file (for error messages)

        Raises:
            MissingFrontmatterError: If required fields are missing
        """
        required_fields = ["title", "date"]

        for field in required_fields:
            if field not in frontmatter or not frontmatter[field]:
                raise MissingFrontmatterError(
                    f"Missing required field '{field}' in {file_path.name}"
                )

    def _parse_date(self, date_value: any, file_path: Path) -> datetime:
        """
        Parse date from various formats.

        Args:
            date_value: Date value from frontmatter
            file_path: Path to file (for error messages)

        Returns:
            datetime object

        Raises:
            InvalidBlogFormatError: If date format is invalid
        """
        # YAML might parse dates automatically
        if isinstance(date_value, datetime):
            return date_value

        # Handle datetime.date objects from YAML
        import datetime as dt
        if isinstance(date_value, dt.date):
            return datetime.combine(date_value, datetime.min.time())

        if isinstance(date_value, str):
            # Try multiple date formats
            formats = [
                "%Y-%m-%d",           # 2025-10-23
                "%Y/%m/%d",           # 2025/10/23
                "%d-%m-%Y",           # 23-10-2025
                "%B %d, %Y",          # October 23, 2025
                "%Y-%m-%d %H:%M:%S",  # 2025-10-23 10:30:00
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue

            raise InvalidBlogFormatError(
                f"Invalid date format in {file_path.name}: '{date_value}'. "
                f"Expected format: YYYY-MM-DD"
            )

        raise InvalidBlogFormatError(
            f"Invalid date type in {file_path.name}: {type(date_value)}"
        )

    def _parse_tags(self, tags_value: any) -> List[str]:
        """
        Parse tags from various formats.

        Args:
            tags_value: Tags value from frontmatter

        Returns:
            List of tag strings
        """
        if not tags_value:
            return []

        if isinstance(tags_value, list):
            return [str(tag).strip() for tag in tags_value]

        if isinstance(tags_value, str):
            # Split by comma
            return [tag.strip() for tag in tags_value.split(",") if tag.strip()]

        return []

    def get_blogs_by_tag(self, blogs: List[Blog], tag: str) -> List[Blog]:
        """
        Filter blogs by tag.

        Args:
            blogs: List of Blog objects
            tag: Tag to filter by (case-insensitive)

        Returns:
            List of blogs with the specified tag
        """
        tag_lower = tag.lower()
        return [
            blog
            for blog in blogs
            if any(t.lower() == tag_lower for t in blog.tags)
        ]

    def get_blogs_by_date_range(
        self, blogs: List[Blog], start_date: datetime, end_date: datetime
    ) -> List[Blog]:
        """
        Filter blogs by date range.

        Args:
            blogs: List of Blog objects
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of blogs within the date range
        """
        return [
            blog
            for blog in blogs
            if start_date <= blog.date <= end_date
        ]

    def get_blog_statistics(self, blogs: List[Blog]) -> Dict:
        """
        Calculate statistics for a list of blogs.

        Args:
            blogs: List of Blog objects

        Returns:
            Dictionary with statistics
        """
        if not blogs:
            return {
                "total_blogs": 0,
                "total_words": 0,
                "total_chars": 0,
                "avg_words_per_blog": 0,
                "unique_tags": [],
                "date_range": None,
            }

        total_words = sum(b.word_count for b in blogs)
        total_chars = sum(b.char_count for b in blogs)
        unique_tags = sorted(set(tag for blog in blogs for tag in blog.tags))
        dates = [b.date for b in blogs]

        return {
            "total_blogs": len(blogs),
            "total_words": total_words,
            "total_chars": total_chars,
            "avg_words_per_blog": total_words // len(blogs),
            "unique_tags": unique_tags,
            "date_range": {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat(),
            },
        }
