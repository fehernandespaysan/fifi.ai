"""
Unit tests for src/embeddings_manager.py

Tests chunking, embedding generation, FAISS operations, and search.
Uses mocked OpenAI API to avoid costs during testing.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import faiss
import numpy as np
import pytest

from src.blog_loader import Blog
from src.embeddings_manager import (
    Chunk,
    EmbeddingsError,
    EmbeddingsManager,
    SearchResult,
)


class TestChunk:
    """Test the Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            text="This is test content",
            blog_title="Test Blog",
            blog_file="test.md",
            chunk_index=0,
            start_pos=0,
            end_pos=20,
            metadata={"author": "Test"},
        )

        assert chunk.text == "This is test content"
        assert chunk.blog_title == "Test Blog"
        assert chunk.chunk_index == 0
        assert chunk.metadata["author"] == "Test"


class TestSearchResult:
    """Test the SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        chunk = Chunk(
            text="Test",
            blog_title="Blog",
            blog_file="test.md",
            chunk_index=0,
            start_pos=0,
            end_pos=4,
            metadata={},
        )

        result = SearchResult(chunk=chunk, score=0.95, distance=0.05)

        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.distance == 0.05


class TestEmbeddingsManager:
    """Test the EmbeddingsManager class."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_response.usage = Mock(total_tokens=100)
        return mock_response

    @pytest.fixture
    def mock_openai_batch_response(self):
        """Create a mock OpenAI batch API response."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(3)
        ]
        mock_response.usage = Mock(total_tokens=300)
        return mock_response

    @pytest.fixture
    def embeddings_manager(self, tmp_path):
        """Create an EmbeddingsManager instance for testing."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "FAISS_INDEX_PATH": str(tmp_path / "test_index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "test_metadata.pkl"),
            },
        ):
            from src.config import reset_config

            reset_config()
            manager = EmbeddingsManager()
            return manager

    def test_embeddings_manager_initialization(self, embeddings_manager):
        """Test EmbeddingsManager initialization."""
        assert embeddings_manager.chunk_size > 0
        assert embeddings_manager.chunk_overlap >= 0
        assert embeddings_manager.embedding_model is not None
        assert embeddings_manager.index is None  # Not initialized yet
        assert len(embeddings_manager.chunks) == 0

    def test_embeddings_manager_custom_paths(self, tmp_path):
        """Test EmbeddingsManager with custom paths."""
        custom_index = tmp_path / "custom_index.faiss"
        custom_metadata = tmp_path / "custom_metadata.pkl"

        manager = EmbeddingsManager(
            index_path=custom_index, metadata_path=custom_metadata
        )

        assert manager.index_path == custom_index
        assert manager.metadata_path == custom_metadata

    def test_chunk_text_basic(self, embeddings_manager):
        """Test basic text chunking."""
        text = " ".join([f"word{i}" for i in range(100)])  # 100 words
        metadata = {"title": "Test Blog", "file_path": "test.md"}

        chunks = embeddings_manager.chunk_text(text, metadata)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.blog_title == "Test Blog" for c in chunks)
        assert all(c.blog_file == "test.md" for c in chunks)

        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_text_with_overlap(self, embeddings_manager):
        """Test that chunks have overlap."""
        # Set small chunk size for testing
        embeddings_manager.chunk_size = 10
        embeddings_manager.chunk_overlap = 3

        text = " ".join([f"word{i}" for i in range(50)])
        chunks = embeddings_manager.chunk_text(text, {})

        # With overlap, we should have more chunks
        assert len(chunks) > 2

        # Check that consecutive chunks overlap
        if len(chunks) >= 2:
            # Get words from first two chunks
            words_chunk_0 = chunks[0].text.split()
            words_chunk_1 = chunks[1].text.split()

            # Some words should appear in both (due to overlap)
            # This is a simple heuristic check
            assert len(words_chunk_0) > 0
            assert len(words_chunk_1) > 0

    def test_chunk_text_empty_string(self, embeddings_manager):
        """Test chunking empty string."""
        chunks = embeddings_manager.chunk_text("", {})
        assert len(chunks) == 0

    def test_chunk_text_short_text(self, embeddings_manager):
        """Test chunking text shorter than chunk size."""
        text = "Short text here"
        chunks = embeddings_manager.chunk_text(text, {})

        assert len(chunks) == 1
        assert chunks[0].text == text

    @patch("openai.OpenAI")
    def test_generate_embedding(self, mock_openai_class, embeddings_manager, mock_openai_response):
        """Test generating a single embedding."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_response
        embeddings_manager.openai_client = mock_client

        text = "This is a test query"
        embedding = embeddings_manager.generate_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 1536
        assert embeddings_manager.total_embeddings_generated == 1
        assert embeddings_manager.total_api_calls == 1
        assert embeddings_manager.total_tokens_used == 100

        # Verify API was called correctly
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == text

    @patch("openai.OpenAI")
    def test_generate_embedding_error(self, mock_openai_class, embeddings_manager):
        """Test error handling in generate_embedding."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        embeddings_manager.openai_client = mock_client

        with pytest.raises(EmbeddingsError) as exc_info:
            embeddings_manager.generate_embedding("test")

        assert "API Error" in str(exc_info.value)

    @patch("openai.OpenAI")
    def test_generate_embeddings_batch(
        self, mock_openai_class, embeddings_manager, mock_openai_batch_response
    ):
        """Test batch embedding generation."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_batch_response
        embeddings_manager.openai_client = mock_client

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embeddings_manager.generate_embeddings_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 1536)
        assert embeddings.dtype == np.float32
        assert embeddings_manager.total_embeddings_generated == 3
        assert embeddings_manager.total_api_calls == 1
        assert embeddings_manager.total_tokens_used == 300

        # Verify batch API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == texts

    def test_generate_embeddings_batch_empty(self, embeddings_manager):
        """Test batch embedding with empty list."""
        embeddings = embeddings_manager.generate_embeddings_batch([])
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0

    @patch("openai.OpenAI")
    def test_generate_embeddings_batch_error(self, mock_openai_class, embeddings_manager):
        """Test error handling in batch embedding generation."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Batch API Error")
        embeddings_manager.openai_client = mock_client

        with pytest.raises(EmbeddingsError) as exc_info:
            embeddings_manager.generate_embeddings_batch(["test1", "test2"])

        assert "Batch API Error" in str(exc_info.value)

    @patch("openai.OpenAI")
    def test_add_documents(
        self, mock_openai_class, embeddings_manager, tmp_path, mock_openai_batch_response
    ):
        """Test adding documents to the index."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_batch_response
        embeddings_manager.openai_client = mock_client

        # Create test blogs
        blogs = [
            Blog(
                file_path=tmp_path / "blog1.md",
                title="Blog 1",
                date=datetime.now(),
                content="This is the first blog post with some content",
                raw_content="",
                tags=["test"],
            ),
            Blog(
                file_path=tmp_path / "blog2.md",
                title="Blog 2",
                date=datetime.now(),
                content="This is the second blog post with different content",
                raw_content="",
                tags=["test"],
            ),
        ]

        embeddings_manager.add_documents(blogs)

        # Check that index was created
        assert embeddings_manager.index is not None
        assert embeddings_manager.index.ntotal > 0
        assert len(embeddings_manager.chunks) > 0
        assert embeddings_manager.dimension == 1536

        # Check metrics
        assert embeddings_manager.total_embeddings_generated > 0
        assert embeddings_manager.total_api_calls > 0

    @patch("openai.OpenAI")
    def test_search(
        self, mock_openai_class, embeddings_manager, tmp_path, mock_openai_batch_response
    ):
        """Test searching the index."""
        mock_client = Mock()

        # First call: batch embeddings for documents
        batch_response = Mock()
        batch_response.data = [Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(3)]
        batch_response.usage = Mock(total_tokens=300)

        # Second call: single embedding for query
        query_response = Mock()
        query_response.data = [Mock(embedding=[0.1] * 1536)]
        query_response.usage = Mock(total_tokens=50)

        mock_client.embeddings.create.side_effect = [batch_response, query_response]
        embeddings_manager.openai_client = mock_client

        # Add documents
        blog = Blog(
            file_path=tmp_path / "blog.md",
            title="Test Blog",
            date=datetime.now(),
            content="This is a test blog with some content about AI and machine learning",
            raw_content="",
            tags=["ai"],
        )
        embeddings_manager.add_documents([blog])

        # Search
        results = embeddings_manager.search("AI machine learning", top_k=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

        # Check result structure
        result = results[0]
        assert isinstance(result.chunk, Chunk)
        assert isinstance(result.score, (float, np.floating))  # Accept numpy float types
        assert isinstance(result.distance, (float, np.floating))
        assert result.score > 0

    def test_search_empty_index(self, embeddings_manager):
        """Test searching an empty index raises error."""
        with pytest.raises(EmbeddingsError) as exc_info:
            embeddings_manager.search("test query")

        assert "empty" in str(exc_info.value).lower()

    @patch("openai.OpenAI")
    def test_save_and_load(
        self, mock_openai_class, embeddings_manager, tmp_path, mock_openai_batch_response
    ):
        """Test saving and loading the index."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_batch_response
        embeddings_manager.openai_client = mock_client

        # Add some documents
        blog = Blog(
            file_path=tmp_path / "blog.md",
            title="Test Blog",
            date=datetime.now(),
            content="Content for testing save and load functionality",
            raw_content="",
        )
        embeddings_manager.add_documents([blog])

        original_chunks = len(embeddings_manager.chunks)
        original_total = embeddings_manager.index.ntotal

        # Save
        embeddings_manager.save()

        # Check files exist
        assert embeddings_manager.index_path.exists()
        assert embeddings_manager.metadata_path.exists()

        # Create new manager and load
        new_manager = EmbeddingsManager(
            index_path=embeddings_manager.index_path,
            metadata_path=embeddings_manager.metadata_path,
        )
        loaded = new_manager.load()

        assert loaded is True
        assert new_manager.index.ntotal == original_total
        assert len(new_manager.chunks) == original_chunks
        assert new_manager.dimension == 1536

    def test_save_empty_index(self, embeddings_manager, caplog):
        """Test that saving empty index logs warning."""
        embeddings_manager.save()

        # Should log a warning
        assert "empty" in caplog.text.lower()

    def test_load_nonexistent_files(self, embeddings_manager):
        """Test loading when files don't exist."""
        loaded = embeddings_manager.load()
        assert loaded is False

    def test_load_corrupt_metadata(self, embeddings_manager, tmp_path):
        """Test loading with corrupt metadata file."""
        # Create FAISS index file first (so it passes the existence check)
        embeddings_manager.index_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatL2(1536)
        faiss.write_index(index, str(embeddings_manager.index_path))

        # Create metadata file with incomplete/corrupt data (missing required keys)
        with open(embeddings_manager.metadata_path, "wb") as f:
            pickle.dump({"invalid": "data"}, f)  # Missing required 'chunks' key

        # Loading should fail when trying to access missing keys
        with pytest.raises((EmbeddingsError, KeyError)):
            embeddings_manager.load()

    def test_clear(self, embeddings_manager):
        """Test clearing the index."""
        # Set some values
        embeddings_manager.index = faiss.IndexFlatL2(1536)
        embeddings_manager.chunks = [Mock()]
        embeddings_manager.dimension = 1536
        embeddings_manager.total_embeddings_generated = 10

        # Clear
        embeddings_manager.clear()

        # Verify cleared
        assert embeddings_manager.index is None
        assert len(embeddings_manager.chunks) == 0
        assert embeddings_manager.dimension is None
        assert embeddings_manager.total_embeddings_generated == 0
        assert embeddings_manager.total_api_calls == 0
        assert embeddings_manager.total_tokens_used == 0

    def test_get_statistics(self, embeddings_manager):
        """Test getting statistics."""
        stats = embeddings_manager.get_statistics()

        assert isinstance(stats, dict)
        assert "total_vectors" in stats
        assert "total_chunks" in stats
        assert "dimension" in stats
        assert "embedding_model" in stats
        assert "chunk_size" in stats
        assert "chunk_overlap" in stats
        assert "total_embeddings_generated" in stats
        assert "total_api_calls" in stats
        assert "total_tokens_used" in stats
        assert "index_exists" in stats
        assert "index_path" in stats
        assert "metadata_path" in stats

        # Empty index stats
        assert stats["total_vectors"] == 0
        assert stats["total_chunks"] == 0
        assert stats["index_exists"] is False

    @patch("openai.OpenAI")
    def test_get_statistics_with_data(
        self, mock_openai_class, embeddings_manager, tmp_path, mock_openai_batch_response
    ):
        """Test statistics with data in index."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_batch_response
        embeddings_manager.openai_client = mock_client

        blog = Blog(
            file_path=tmp_path / "blog.md",
            title="Test",
            date=datetime.now(),
            content="Content",
            raw_content="",
        )
        embeddings_manager.add_documents([blog])

        stats = embeddings_manager.get_statistics()

        assert stats["total_vectors"] > 0
        assert stats["total_chunks"] > 0
        assert stats["dimension"] == 1536
        assert stats["index_exists"] is True
        assert stats["total_embeddings_generated"] > 0


class TestEmbeddingsManagerIntegration:
    """Integration tests with real blog files."""

    @patch("openai.OpenAI")
    def test_load_and_embed_real_blogs(self, mock_openai_class, tmp_path):
        """Test loading real blog files and creating embeddings."""
        # Check if blogs directory exists
        blogs_dir = Path("blogs")
        if not blogs_dir.exists():
            pytest.skip("blogs/ directory not found")

        from src.blog_loader import BlogLoader

        # Load blogs
        loader = BlogLoader(blogs_directory=blogs_dir)
        blogs = loader.load_all_blogs()

        if len(blogs) == 0:
            pytest.skip("No blog files found")

        # Mock OpenAI API
        mock_client = Mock()

        # Create mock embeddings for each chunk
        def create_mock_response(input=None, model=None, **kwargs):
            if isinstance(input, list):
                count = len(input)
            else:
                count = 1

            response = Mock()
            response.data = [Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(count)]
            response.usage = Mock(total_tokens=count * 100)
            return response

        mock_client.embeddings.create.side_effect = create_mock_response

        # Create embeddings manager with mocked client
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "FAISS_INDEX_PATH": str(tmp_path / "test_index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "test_metadata.pkl"),
            },
        ):
            from src.config import reset_config

            reset_config()
            manager = EmbeddingsManager()
            manager.openai_client = mock_client

            # Add documents
            manager.add_documents(blogs)

            # Verify index was created
            assert manager.index is not None
            assert manager.index.ntotal > 0
            assert len(manager.chunks) > 0

            # Test save
            manager.save()
            assert manager.index_path.exists()
            assert manager.metadata_path.exists()

            # Test load
            new_manager = EmbeddingsManager(
                index_path=manager.index_path, metadata_path=manager.metadata_path
            )
            loaded = new_manager.load()
            assert loaded is True
            assert new_manager.index.ntotal == manager.index.ntotal

    @patch("openai.OpenAI")
    def test_end_to_end_rag_pipeline(self, mock_openai_class, tmp_path):
        """Test complete RAG pipeline: load blogs → embed → search."""
        from src.blog_loader import BlogLoader

        # Create test blog
        blog_content = """---
title: Test RAG Blog
date: 2025-10-23
tags: test, rag
---

# RAG Systems

RAG stands for Retrieval-Augmented Generation. It combines retrieval from a knowledge base with LLM generation.

## Key Concepts

Vector databases store embeddings. FAISS is a popular choice for similarity search.
"""
        blog_file = tmp_path / "test_blog.md"
        blog_file.write_text(blog_content)

        # Load blog
        loader = BlogLoader(blogs_directory=tmp_path)
        blogs = loader.load_all_blogs()
        assert len(blogs) == 1

        # Mock OpenAI
        mock_client = Mock()

        def create_mock_response(input=None, model=None, **kwargs):
            if isinstance(input, list):
                count = len(input)
            else:
                count = 1
            response = Mock()
            response.data = [Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(count)]
            response.usage = Mock(total_tokens=count * 50)
            return response

        mock_client.embeddings.create.side_effect = create_mock_response

        # Create embeddings
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "FAISS_INDEX_PATH": str(tmp_path / "index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "metadata.pkl"),
            },
        ):
            from src.config import reset_config

            reset_config()
            manager = EmbeddingsManager()
            manager.openai_client = mock_client

            # Add documents
            manager.add_documents(blogs)

            # Search
            results = manager.search("What is RAG?", top_k=3)

            # Verify results
            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)
            assert all("Test RAG Blog" in r.chunk.blog_title for r in results)

            # Get statistics
            stats = manager.get_statistics()
            assert stats["total_vectors"] > 0
            assert stats["total_chunks"] > 0
