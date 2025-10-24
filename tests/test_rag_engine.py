"""
Unit tests for src/rag_engine.py

Tests RAG query processing, context retrieval, response generation,
and conversation history management. Uses mocked APIs to avoid costs.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.embeddings_manager import Chunk, EmbeddingsManager, SearchResult
from src.rag_engine import (
    ConversationMessage,
    RAGEngine,
    RAGEngineError,
    RAGResponse,
)


class TestRAGResponse:
    """Test the RAGResponse dataclass."""

    def test_rag_response_creation(self):
        """Test creating a RAGResponse instance."""
        chunk = Chunk(
            text="Test content",
            blog_title="Test Blog",
            blog_file="test.md",
            chunk_index=0,
            start_pos=0,
            end_pos=12,
            metadata={},
        )
        source = SearchResult(chunk=chunk, score=0.9, distance=0.1)

        response = RAGResponse(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            sources=[source],
            context="Test context",
            model="gpt-4o-mini",
            tokens_used=100,
            retrieval_time_ms=50,
            generation_time_ms=200,
            total_time_ms=250,
        )

        assert response.query == "What is RAG?"
        assert "RAG stands for" in response.answer
        assert len(response.sources) == 1
        assert response.tokens_used == 100

    def test_rag_response_to_dict(self):
        """Test converting RAGResponse to dictionary."""
        chunk = Chunk(
            text="Content",
            blog_title="Blog",
            blog_file="blog.md",
            chunk_index=0,
            start_pos=0,
            end_pos=7,
            metadata={},
        )
        source = SearchResult(chunk=chunk, score=0.8, distance=0.2)

        response = RAGResponse(
            query="Test query",
            answer="Test answer",
            sources=[source],
            context="Context",
            model="gpt-4o-mini",
            tokens_used=50,
            retrieval_time_ms=30,
            generation_time_ms=100,
            total_time_ms=130,
        )

        response_dict = response.to_dict()

        assert response_dict["query"] == "Test query"
        assert response_dict["answer"] == "Test answer"
        assert len(response_dict["sources"]) == 1
        assert response_dict["sources"][0]["blog_title"] == "Blog"
        assert response_dict["tokens_used"] == 50


class TestConversationMessage:
    """Test the ConversationMessage dataclass."""

    def test_conversation_message_creation(self):
        """Test creating a ConversationMessage."""
        msg = ConversationMessage(
            role="user",
            content="Hello",
            metadata={"source": "cli"},
        )

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.metadata["source"] == "cli"
        assert msg.timestamp > 0


class TestRAGEngine:
    """Test the RAGEngine class."""

    @pytest.fixture
    def mock_embeddings_manager(self, tmp_path):
        """Create a mock EmbeddingsManager."""
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

            manager = Mock(spec=EmbeddingsManager)
            manager.index = Mock()
            manager.index.ntotal = 10
            return manager

    @pytest.fixture
    def mock_search_results(self, tmp_path):
        """Create mock search results."""
        chunks = [
            Chunk(
                text="RAG stands for Retrieval-Augmented Generation. It's a technique that combines retrieval with generation.",
                blog_title="What is RAG?",
                blog_file=str(tmp_path / "rag.md"),
                chunk_index=0,
                start_pos=0,
                end_pos=105,
                metadata={},
            ),
            Chunk(
                text="Vector databases store embeddings and enable semantic search.",
                blog_title="Vector Databases",
                blog_file=str(tmp_path / "vectors.md"),
                chunk_index=0,
                start_pos=0,
                end_pos=61,
                metadata={},
            ),
        ]

        return [
            SearchResult(chunk=chunks[0], score=0.9, distance=0.1),
            SearchResult(chunk=chunks[1], score=0.7, distance=0.3),
        ]

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI API response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = "RAG stands for Retrieval-Augmented Generation. It combines retrieval from a knowledge base with LLM generation to provide accurate, contextual answers."
        response.usage = Mock()
        response.usage.total_tokens = 150
        return response

    @patch("src.rag_engine.OpenAI")
    def test_rag_engine_initialization(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test RAGEngine initialization."""
        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        assert engine.embeddings_manager == mock_embeddings_manager
        assert engine.model is not None
        assert engine.temperature >= 0
        assert engine.top_k > 0
        assert engine.min_relevance_score >= 0
        assert len(engine.conversation_history) == 0

    @patch("src.rag_engine.OpenAI")
    def test_rag_engine_custom_system_prompt(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test RAGEngine with custom system prompt."""
        custom_prompt = "You are a specialized AI assistant."
        engine = RAGEngine(
            embeddings_manager=mock_embeddings_manager,
            system_prompt=custom_prompt,
        )

        assert engine.system_prompt == custom_prompt

    @patch("src.rag_engine.OpenAI")
    def test_query_success(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        mock_search_results,
        mock_openai_response,
    ):
        """Test successful query processing."""
        # Setup mocks
        mock_embeddings_manager.search.return_value = mock_search_results

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        # Create engine
        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.openai_client = mock_client

        # Execute query
        response = engine.query("What is RAG?")

        # Verify response
        assert isinstance(response, RAGResponse)
        assert response.query == "What is RAG?"
        assert len(response.answer) > 0
        assert len(response.sources) == 2
        assert response.tokens_used == 150
        assert response.retrieval_time_ms >= 0  # Can be 0 in tests
        assert response.generation_time_ms >= 0  # Can be 0 in tests

        # Verify mocks were called
        mock_embeddings_manager.search.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.rag_engine.OpenAI")
    def test_query_no_context_found(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test query when no relevant context is found."""
        # Setup mock to return empty results
        mock_embeddings_manager.search.return_value = []

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        # Execute query
        response = engine.query("Some obscure topic")

        # Verify response
        assert isinstance(response, RAGResponse)
        assert "couldn't find relevant information" in response.answer
        assert len(response.sources) == 0
        assert response.tokens_used == 0
        assert response.metadata.get("no_context") is True

    @patch("src.rag_engine.OpenAI")
    def test_query_filters_low_relevance(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        tmp_path,
        mock_openai_response,
    ):
        """Test that low relevance results are filtered out."""
        # Create results with low scores
        low_score_chunks = [
            Chunk(
                text="Irrelevant content",
                blog_title="Blog",
                blog_file=str(tmp_path / "blog.md"),
                chunk_index=0,
                start_pos=0,
                end_pos=18,
                metadata={},
            )
        ]
        low_score_results = [
            SearchResult(chunk=low_score_chunks[0], score=0.1, distance=0.9)
        ]

        mock_embeddings_manager.search.return_value = low_score_results

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.min_relevance_score = 0.3  # Set threshold

        response = engine.query("Test query")

        # Should get no context response because score too low
        assert "couldn't find relevant information" in response.answer

    @patch("src.rag_engine.OpenAI")
    def test_format_context(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        mock_search_results,
    ):
        """Test context formatting."""
        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        context = engine._format_context(mock_search_results)

        assert "[Source 1: What is RAG?]" in context
        assert "[Source 2: Vector Databases]" in context
        assert "RAG stands for" in context
        assert "Vector databases" in context

    @patch("src.rag_engine.OpenAI")
    def test_conversation_history(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        mock_search_results,
        mock_openai_response,
    ):
        """Test conversation history management."""
        mock_embeddings_manager.search.return_value = mock_search_results

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.openai_client = mock_client

        # First query
        response1 = engine.query("What is RAG?")

        # Check history after first query
        history = engine.get_history()
        assert len(history) == 2  # User message + assistant message
        assert history[0].role == "user"
        assert history[0].content == "What is RAG?"
        assert history[1].role == "assistant"

        # Second query
        response2 = engine.query("Tell me more")

        # Check history after second query
        history = engine.get_history()
        assert len(history) == 4  # 2 pairs of messages

    @patch("src.rag_engine.OpenAI")
    def test_clear_history(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test clearing conversation history."""
        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        # Add some messages
        engine._add_to_history("user", "Hello")
        engine._add_to_history("assistant", "Hi there!")

        assert len(engine.get_history()) == 2

        # Clear history
        engine.clear_history()

        assert len(engine.get_history()) == 0

    @patch("src.rag_engine.OpenAI")
    def test_history_trimming(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test that history is trimmed when it exceeds max length."""
        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.max_history_length = 2  # Keep only 2 turns (4 messages)

        # Add many messages
        for i in range(10):
            engine._add_to_history("user", f"Message {i}")
            engine._add_to_history("assistant", f"Response {i}")

        # Should only keep most recent messages
        history = engine.get_history()
        assert len(history) <= engine.max_history_length * 2

    @patch("src.rag_engine.OpenAI")
    def test_query_error_handling(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test error handling in query processing."""
        mock_embeddings_manager.search.side_effect = Exception("Search failed")

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        with pytest.raises(RAGEngineError) as exc_info:
            engine.query("Test query")

        assert "Query processing failed" in str(exc_info.value)

    @patch("src.rag_engine.OpenAI")
    def test_generate_response_error(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        mock_search_results,
    ):
        """Test error handling in response generation."""
        mock_embeddings_manager.search.return_value = mock_search_results

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.openai_client = mock_client

        with pytest.raises(RAGEngineError) as exc_info:
            engine.query("Test query")

        assert "Query processing failed" in str(exc_info.value)

    @patch("src.rag_engine.OpenAI")
    def test_statistics(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        mock_search_results,
        mock_openai_response,
    ):
        """Test statistics tracking."""
        mock_embeddings_manager.search.return_value = mock_search_results

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_openai_class.return_value = mock_client

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.openai_client = mock_client

        # Execute multiple queries
        for _ in range(3):
            engine.query("Test query")

        # Check statistics
        stats = engine.get_statistics()

        assert stats["total_queries"] == 3
        assert stats["total_tokens_used"] == 450  # 150 * 3
        assert stats["avg_tokens_per_query"] == 150
        assert stats["total_retrieval_time_ms"] >= 0  # Can be 0 in tests
        assert stats["total_generation_time_ms"] >= 0  # Can be 0 in tests
        assert stats["avg_retrieval_time_ms"] >= 0  # Can be 0 in tests
        assert stats["avg_generation_time_ms"] >= 0  # Can be 0 in tests

    @patch("src.rag_engine.OpenAI")
    def test_stream_query(
        self,
        mock_openai_class,
        mock_embeddings_manager,
        mock_search_results,
    ):
        """Test streaming query response."""
        mock_embeddings_manager.search.return_value = mock_search_results

        # Create mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="RAG "))]),
            Mock(choices=[Mock(delta=Mock(content="stands "))]),
            Mock(choices=[Mock(delta=Mock(content="for "))]),
            Mock(choices=[Mock(delta=Mock(content="Retrieval-Augmented "))]),
            Mock(choices=[Mock(delta=Mock(content="Generation."))]),
        ]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_chunks)
        mock_openai_class.return_value = mock_client

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)
        engine.openai_client = mock_client

        # Stream response
        chunks = list(engine.stream_query("What is RAG?"))

        # Verify chunks
        assert len(chunks) == 5
        assert "".join(chunks) == "RAG stands for Retrieval-Augmented Generation."

        # Verify history was updated
        history = engine.get_history()
        assert len(history) == 2
        assert history[1].content == "RAG stands for Retrieval-Augmented Generation."

    @patch("src.rag_engine.OpenAI")
    def test_stream_query_no_context(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test streaming query when no context is found."""
        mock_embeddings_manager.search.return_value = []

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        # Stream response
        chunks = list(engine.stream_query("Test query"))

        # Verify error message
        assert len(chunks) == 1
        assert "couldn't find relevant information" in chunks[0]

    @patch("src.rag_engine.OpenAI")
    def test_empty_index_handling(
        self, mock_openai_class, mock_embeddings_manager
    ):
        """Test handling of empty embeddings index."""
        mock_embeddings_manager.index = None

        engine = RAGEngine(embeddings_manager=mock_embeddings_manager)

        # Query should return no context response
        response = engine.query("Test query")

        assert "couldn't find relevant information" in response.answer


class TestRAGEngineIntegration:
    """Integration tests for RAG engine with real components."""

    @patch("src.rag_engine.OpenAI")
    @patch("src.embeddings_manager.OpenAI")
    def test_rag_with_real_embeddings_manager(
        self, mock_embeddings_openai, mock_rag_openai, tmp_path
    ):
        """Test RAG engine with real EmbeddingsManager (mocked API calls)."""
        from src.blog_loader import Blog

        # Mock OpenAI for embeddings
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
        mock_embeddings_client.embeddings.create.side_effect = create_embedding_response
        mock_embeddings_openai.return_value = mock_embeddings_client

        # Mock OpenAI for generation
        mock_generation_response = Mock()
        mock_generation_response.choices = [Mock()]
        mock_generation_response.choices[0].message = Mock()
        mock_generation_response.choices[
            0
        ].message.content = "RAG is a powerful technique for AI applications."
        mock_generation_response.usage = Mock()
        mock_generation_response.usage.total_tokens = 100

        mock_rag_client = Mock()
        mock_rag_client.chat.completions.create.return_value = mock_generation_response
        mock_rag_openai.return_value = mock_rag_client

        # Create blog and embeddings manager
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

            blog = Blog(
                file_path=tmp_path / "blog.md",
                title="Test RAG Blog",
                date=datetime.now(),
                content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation.",
                raw_content="",
                tags=["rag"],
            )

            embeddings_manager = EmbeddingsManager()
            embeddings_manager.openai_client = mock_embeddings_client
            embeddings_manager.add_documents([blog])

            # Create RAG engine
            engine = RAGEngine(embeddings_manager=embeddings_manager)
            engine.openai_client = mock_rag_client

            # Query
            response = engine.query("What is RAG?")

            # Verify response
            assert isinstance(response, RAGResponse)
            assert len(response.answer) > 0
            assert response.tokens_used == 100
            assert len(response.sources) > 0
