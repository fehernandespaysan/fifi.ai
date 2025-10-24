"""
Integration tests for Fifi.ai

Tests the complete RAG pipeline from blog loading to response generation.
These tests use mocked APIs to avoid costs but test real integration points.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.blog_loader import Blog, BlogLoader
from src.embeddings_manager import EmbeddingsManager
from src.rag_engine import RAGEngine


class TestBlogToEmbeddingsPipeline:
    """Test the pipeline from blogs to embeddings."""

    @patch("src.embeddings_manager.OpenAI")
    def test_load_blogs_and_create_embeddings(self, mock_openai, tmp_path):
        """Test loading blogs and generating embeddings."""
        # Create test blog file
        blog_content = """---
title: Integration Test Blog
date: 2025-10-24
tags: test, integration
author: Test Suite
---

# Integration Test

This is a test blog for integration testing.
It contains multiple sentences to test chunking.
The embeddings manager should process this correctly.
"""
        blog_file = tmp_path / "test_blog.md"
        blog_file.write_text(blog_content)

        # Mock OpenAI embeddings
        def create_embedding_response(input=None, model=None, **kwargs):
            if isinstance(input, list):
                count = len(input)
            else:
                count = 1
            response = Mock()
            response.data = [
                Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(count)
            ]
            response.usage = Mock(total_tokens=count * 50)
            return response

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = create_embedding_response
        mock_openai.return_value = mock_client

        # Test the pipeline
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "BLOGS_DIRECTORY": str(tmp_path),
                "FAISS_INDEX_PATH": str(tmp_path / "test_index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "test_metadata.pkl"),
            },
        ):
            from src.config import reset_config

            reset_config()

            # Load blogs
            loader = BlogLoader()
            blogs = loader.load_all_blogs()

            assert len(blogs) == 1
            assert blogs[0].title == "Integration Test Blog"

            # Create embeddings
            embeddings_manager = EmbeddingsManager()
            embeddings_manager.openai_client = mock_client
            embeddings_manager.add_documents(blogs)

            # Verify embeddings created
            assert embeddings_manager.index is not None
            assert embeddings_manager.index.ntotal > 0
            assert len(embeddings_manager.chunks) > 0

            # Verify save/load
            embeddings_manager.save()
            assert embeddings_manager.index_path.exists()
            assert embeddings_manager.metadata_path.exists()

            # Load in new manager
            new_manager = EmbeddingsManager()
            loaded = new_manager.load()

            assert loaded is True
            assert new_manager.index.ntotal == embeddings_manager.index.ntotal
            assert len(new_manager.chunks) == len(embeddings_manager.chunks)


class TestEmbeddingsToRAGPipeline:
    """Test the pipeline from embeddings to RAG responses."""

    @patch("src.rag_engine.OpenAI")
    @patch("src.embeddings_manager.OpenAI")
    def test_embeddings_to_rag_query(
        self, mock_embeddings_openai, mock_rag_openai, tmp_path
    ):
        """Test querying RAG engine with embeddings."""
        # Create test blog
        blog = Blog(
            file_path=tmp_path / "blog.md",
            title="RAG Test Blog",
            date=datetime.now(),
            content="RAG stands for Retrieval-Augmented Generation. It combines retrieval from a knowledge base with language model generation.",
            raw_content="",
            tags=["rag", "test"],
        )

        # Mock embeddings OpenAI
        def create_embedding_response(input=None, model=None, **kwargs):
            if isinstance(input, list):
                count = len(input)
            else:
                count = 1
            response = Mock()
            response.data = [
                Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(count)
            ]
            response.usage = Mock(total_tokens=count * 50)
            return response

        mock_embeddings_client = Mock()
        mock_embeddings_client.embeddings.create.side_effect = (
            create_embedding_response
        )
        mock_embeddings_openai.return_value = mock_embeddings_client

        # Mock RAG OpenAI
        mock_rag_response = Mock()
        mock_rag_response.choices = [Mock()]
        mock_rag_response.choices[0].message = Mock()
        mock_rag_response.choices[
            0
        ].message.content = (
            "RAG (Retrieval-Augmented Generation) is a technique that enhances language models by retrieving relevant information from a knowledge base before generating responses."
        )
        mock_rag_response.usage = Mock(total_tokens=150)

        mock_rag_client = Mock()
        mock_rag_client.chat.completions.create.return_value = mock_rag_response
        mock_rag_openai.return_value = mock_rag_client

        # Setup environment
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

            # Create embeddings
            embeddings_manager = EmbeddingsManager()
            embeddings_manager.openai_client = mock_embeddings_client
            embeddings_manager.add_documents([blog])

            # Create RAG engine
            rag_engine = RAGEngine(embeddings_manager=embeddings_manager)
            rag_engine.openai_client = mock_rag_client

            # Query
            response = rag_engine.query("What is RAG?")

            # Verify response
            assert response.query == "What is RAG?"
            assert len(response.answer) > 0
            assert "RAG" in response.answer or "Retrieval" in response.answer
            assert len(response.sources) > 0
            assert response.tokens_used == 150
            assert response.retrieval_time_ms >= 0
            assert response.generation_time_ms >= 0


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline."""

    @patch("src.rag_engine.OpenAI")
    @patch("src.embeddings_manager.OpenAI")
    def test_full_pipeline_blog_to_response(
        self, mock_embeddings_openai, mock_rag_openai, tmp_path
    ):
        """Test complete pipeline: blog files -> embeddings -> RAG -> response."""
        # Create multiple test blogs
        blogs_data = [
            {
                "filename": "rag_basics.md",
                "title": "RAG Basics",
                "content": "RAG (Retrieval-Augmented Generation) combines retrieval with generation. It searches a knowledge base and uses the results to generate better answers.",
            },
            {
                "filename": "vector_db.md",
                "title": "Vector Databases",
                "content": "Vector databases store embeddings and enable semantic search. FAISS is a popular open-source option.",
            },
            {
                "filename": "security.md",
                "title": "Security",
                "content": "Security best practices include: API key management, input validation, and rate limiting.",
            },
        ]

        blogs_dir = tmp_path / "blogs"
        blogs_dir.mkdir()

        for blog_data in blogs_data:
            content = f"""---
title: {blog_data['title']}
date: 2025-10-24
tags: test
---

{blog_data['content']}
"""
            (blogs_dir / blog_data["filename"]).write_text(content)

        # Mock embeddings
        def create_embedding_response(input=None, model=None, **kwargs):
            if isinstance(input, list):
                count = len(input)
            else:
                count = 1
            response = Mock()
            response.data = [
                Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(count)
            ]
            response.usage = Mock(total_tokens=count * 50)
            return response

        mock_embeddings_client = Mock()
        mock_embeddings_client.embeddings.create.side_effect = (
            create_embedding_response
        )
        mock_embeddings_openai.return_value = mock_embeddings_client

        # Mock RAG
        def create_rag_response(*args, **kwargs):
            # Extract query from messages
            messages = kwargs.get("messages", [])
            user_message = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )

            # Simple response based on query
            if "RAG" in user_message:
                answer = "RAG combines retrieval with generation to produce accurate answers."
            elif "vector" in user_message.lower():
                answer = "Vector databases store embeddings for semantic search."
            elif "security" in user_message.lower():
                answer = "Security requires API key management and input validation."
            else:
                answer = "I can help answer questions about RAG, vectors, and security."

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = answer
            response.usage = Mock(total_tokens=100)
            return response

        mock_rag_client = Mock()
        mock_rag_client.chat.completions.create.side_effect = create_rag_response
        mock_rag_openai.return_value = mock_rag_client

        # Setup environment
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "BLOGS_DIRECTORY": str(blogs_dir),
                "FAISS_INDEX_PATH": str(tmp_path / "index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "metadata.pkl"),
            },
        ):
            from src.config import reset_config

            reset_config()

            # Step 1: Load blogs
            loader = BlogLoader()
            blogs = loader.load_all_blogs()
            assert len(blogs) == 3

            # Step 2: Create embeddings
            embeddings_manager = EmbeddingsManager()
            embeddings_manager.openai_client = mock_embeddings_client
            embeddings_manager.add_documents(blogs)
            assert embeddings_manager.index.ntotal > 0

            # Step 3: Save and reload
            embeddings_manager.save()
            new_embeddings = EmbeddingsManager()
            loaded = new_embeddings.load()
            assert loaded is True

            # Step 4: Create RAG engine
            rag_engine = RAGEngine(embeddings_manager=new_embeddings)
            rag_engine.openai_client = mock_rag_client

            # Step 5: Test multiple queries
            queries = [
                "What is RAG?",
                "Tell me about vector databases",
                "What are security best practices?",
            ]

            for query in queries:
                response = rag_engine.query(query)

                # Verify response structure
                assert response.query == query
                assert len(response.answer) > 0
                assert len(response.sources) > 0
                assert response.tokens_used > 0

            # Step 6: Verify conversation history
            history = rag_engine.get_history()
            assert len(history) == 6  # 3 queries + 3 responses

            # Step 7: Verify statistics
            stats = rag_engine.get_statistics()
            assert stats["total_queries"] == 3
            assert stats["total_tokens_used"] == 300  # 100 * 3


class TestErrorHandlingAndRecovery:
    """Test error handling throughout the pipeline."""

    def test_missing_blogs_directory(self, tmp_path):
        """Test handling of missing blogs directory."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "BLOGS_DIRECTORY": str(tmp_path / "nonexistent"),
            },
        ):
            from src.blog_loader import BlogLoaderError
            from src.config import reset_config

            reset_config()

            loader = BlogLoader()

            with pytest.raises(BlogLoaderError):
                loader.load_all_blogs()

    @patch("src.embeddings_manager.OpenAI")
    def test_api_error_handling(self, mock_openai, tmp_path):
        """Test handling of API errors."""
        blog = Blog(
            file_path=tmp_path / "blog.md",
            title="Test",
            date=datetime.now(),
            content="Test content",
            raw_content="",
        )

        # Mock API error
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "FAISS_INDEX_PATH": str(tmp_path / "index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "metadata.pkl"),
            },
        ):
            from src.config import reset_config
            from src.embeddings_manager import EmbeddingsError

            reset_config()

            embeddings_manager = EmbeddingsManager()
            embeddings_manager.openai_client = mock_client

            with pytest.raises(EmbeddingsError):
                embeddings_manager.add_documents([blog])

    @patch("src.rag_engine.OpenAI")
    @patch("src.embeddings_manager.OpenAI")
    def test_empty_index_query(self, mock_embeddings_openai, mock_rag_openai, tmp_path):
        """Test querying with empty index."""
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

            embeddings_manager = EmbeddingsManager()
            rag_engine = RAGEngine(embeddings_manager=embeddings_manager)

            # Query with empty index
            response = rag_engine.query("Test query")

            # Should return no context response
            assert "couldn't find relevant information" in response.answer


class TestPerformanceAndScaling:
    """Test performance characteristics."""

    @patch("src.embeddings_manager.OpenAI")
    def test_large_document_processing(self, mock_openai, tmp_path):
        """Test processing large documents."""
        # Create a large blog (simulate)
        large_content = " ".join([f"sentence{i}" for i in range(2000)])  # 2000 words

        blog = Blog(
            file_path=tmp_path / "large.md",
            title="Large Blog",
            date=datetime.now(),
            content=large_content,
            raw_content="",
        )

        # Mock embeddings
        def create_embedding_response(input=None, model=None, **kwargs):
            if isinstance(input, list):
                count = len(input)
            else:
                count = 1
            response = Mock()
            response.data = [
                Mock(embedding=[0.1 + i * 0.01] * 1536) for i in range(count)
            ]
            response.usage = Mock(total_tokens=count * 50)
            return response

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = create_embedding_response
        mock_openai.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
                "FAISS_INDEX_PATH": str(tmp_path / "index.faiss"),
                "FAISS_METADATA_PATH": str(tmp_path / "metadata.pkl"),
                "CHUNK_SIZE": "500",
                "CHUNK_OVERLAP": "50",
            },
        ):
            from src.config import reset_config

            reset_config()

            embeddings_manager = EmbeddingsManager()
            embeddings_manager.openai_client = mock_client
            embeddings_manager.add_documents([blog])

            # Verify chunking
            assert len(embeddings_manager.chunks) > 1
            assert embeddings_manager.index.ntotal > 1

            # Verify statistics
            stats = embeddings_manager.get_statistics()
            assert stats["total_chunks"] > 1
            assert stats["total_embeddings_generated"] > 1
