"""
RAG Query Engine for Fifi.ai

Implements Retrieval-Augmented Generation using:
- EmbeddingsManager for semantic search
- OpenAI for response generation
- LangChain for prompt management and chaining

This module is responsible for:
- Processing user queries
- Retrieving relevant context from vector database
- Generating responses using LLM
- Managing conversation history
- Streaming responses (optional)

Usage:
    from src.rag_engine import RAGEngine

    engine = RAGEngine()
    response = engine.query("What is RAG?")
    print(response.answer)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI

from src.config import get_config
from src.embeddings_manager import EmbeddingsManager, SearchResult
from src.logger import LoggerContext, get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """
    Represents a RAG query response.

    Attributes:
        query: Original user query
        answer: Generated answer
        sources: List of source chunks used
        context: Retrieved context text
        model: Model used for generation
        tokens_used: Total tokens consumed
        retrieval_time_ms: Time spent on retrieval
        generation_time_ms: Time spent on generation
        total_time_ms: Total query processing time
        metadata: Additional metadata
    """

    query: str
    answer: str
    sources: List[SearchResult]
    context: str
    model: str
    tokens_used: int
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {
                    "blog_title": s.chunk.blog_title,
                    "blog_file": s.chunk.blog_file,
                    "chunk_index": s.chunk.chunk_index,
                    "score": float(s.score),
                    "text": s.chunk.text,
                }
                for s in self.sources
            ],
            "context": self.context,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class ConversationMessage:
    """
    Represents a message in conversation history.

    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
        timestamp: Message timestamp
        metadata: Additional metadata
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGEngineError(Exception):
    """Base exception for RAG engine errors."""

    pass


class RAGEngine:
    """
    RAG Query Engine combining retrieval and generation.

    Uses EmbeddingsManager for retrieval and OpenAI for generation.
    Implements proper prompt engineering and context management.
    """

    def __init__(
        self,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the RAG engine.

        Args:
            embeddings_manager: EmbeddingsManager instance (creates new if None)
            system_prompt: Custom system prompt (uses default if None)
        """
        config = get_config()
        self.config = config

        # Initialize embeddings manager
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()

        # Load existing index if available
        if self.embeddings_manager.index is None:
            self.embeddings_manager.load()

        # Initialize OpenAI client for generation
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_model
        self.temperature = config.openai_temperature
        self.max_tokens = config.openai_max_tokens

        # Set system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Create prompt template
        self.user_prompt_template = self._get_user_prompt_template()

        # Conversation history
        self.conversation_history: List[ConversationMessage] = []
        self.max_history_length = config.rag_max_history

        # Retrieval settings
        self.top_k = config.rag_top_k
        self.min_relevance_score = config.rag_min_relevance_score

        # Metrics
        self.total_queries = 0
        self.total_tokens_used = 0
        self.total_retrieval_time_ms = 0
        self.total_generation_time_ms = 0

        logger.info(
            "RAGEngine initialized",
            extra={
                "model": self.model,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "min_relevance_score": self.min_relevance_score,
                "index_loaded": self.embeddings_manager.index is not None,
            },
        )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the RAG assistant."""
        return """You are Fifi, a helpful AI assistant that answers questions about AI engineering, RAG systems, and related topics.

Your knowledge comes from a curated collection of blog posts and articles. Always:
- Provide accurate, helpful answers based on the given context
- Cite your sources when referencing specific information
- Admit when you don't know something or when the context doesn't contain the answer
- Be concise but comprehensive
- Use examples and analogies when helpful
- Format your responses clearly with markdown

If the context doesn't contain enough information to answer the question, say so honestly and offer to help with related topics you do know about."""

    def _get_user_prompt_template(self) -> str:
        """Get the user prompt template for RAG queries."""
        template = """Use the following context from blog posts to answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't fully answer the question, say so
- Cite specific sources when referencing information
- Be clear, concise, and helpful
- Use markdown formatting for better readability

Answer:"""

        return template

    def query(self, query_text: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Process a query and generate a response.

        Args:
            query_text: User's question
            top_k: Number of chunks to retrieve (uses default if None)

        Returns:
            RAGResponse with answer and metadata

        Raises:
            RAGEngineError: If query processing fails
        """
        with LoggerContext() as correlation_id:
            start_time = time.time()

            logger.info(
                "Processing RAG query",
                extra={
                    "query_length": len(query_text),
                    "top_k": top_k or self.top_k,
                    "correlation_id": correlation_id,
                },
            )

            try:
                # Step 1: Retrieve relevant context
                retrieval_start = time.time()
                sources = self._retrieve_context(query_text, top_k)
                retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

                if not sources:
                    logger.warning(
                        "No relevant context found",
                        extra={"query": query_text[:100]},
                    )
                    return self._create_no_context_response(
                        query_text, retrieval_time_ms
                    )

                # Step 2: Format context
                context = self._format_context(sources)

                logger.debug(
                    "Context retrieved",
                    extra={
                        "num_sources": len(sources),
                        "context_length": len(context),
                        "top_score": sources[0].score if sources else None,
                    },
                )

                # Step 3: Generate response
                generation_start = time.time()
                answer, tokens_used = self._generate_response(query_text, context)
                generation_time_ms = int((time.time() - generation_start) * 1000)

                # Step 4: Update metrics
                total_time_ms = int((time.time() - start_time) * 1000)
                self.total_queries += 1
                self.total_tokens_used += tokens_used
                self.total_retrieval_time_ms += retrieval_time_ms
                self.total_generation_time_ms += generation_time_ms

                # Step 5: Update conversation history
                self._add_to_history("user", query_text)
                self._add_to_history("assistant", answer)

                # Create response
                response = RAGResponse(
                    query=query_text,
                    answer=answer,
                    sources=sources,
                    context=context,
                    model=self.model,
                    tokens_used=tokens_used,
                    retrieval_time_ms=retrieval_time_ms,
                    generation_time_ms=generation_time_ms,
                    total_time_ms=total_time_ms,
                    metadata={
                        "num_sources": len(sources),
                        "top_score": float(sources[0].score),
                        "correlation_id": correlation_id,
                    },
                )

                logger.info(
                    "RAG query completed",
                    extra={
                        "tokens_used": tokens_used,
                        "total_time_ms": total_time_ms,
                        "retrieval_time_ms": retrieval_time_ms,
                        "generation_time_ms": generation_time_ms,
                        "num_sources": len(sources),
                    },
                )

                return response

            except Exception as e:
                logger.error(f"RAG query failed: {str(e)}", exc_info=True)
                raise RAGEngineError(f"Query processing failed: {str(e)}")

    def _retrieve_context(
        self, query: str, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant context from embeddings manager.

        Args:
            query: User query
            top_k: Number of results to retrieve

        Returns:
            List of SearchResult objects
        """
        k = top_k or self.top_k

        if self.embeddings_manager.index is None or self.embeddings_manager.index.ntotal == 0:
            logger.warning("Embeddings index is empty")
            return []

        # Search for relevant chunks
        results = self.embeddings_manager.search(query, top_k=k)

        # Filter by minimum relevance score
        filtered_results = [
            r for r in results if r.score >= self.min_relevance_score
        ]

        logger.debug(
            f"Retrieved {len(filtered_results)} chunks (filtered from {len(results)})",
            extra={
                "query": query[:100],
                "top_k": k,
                "min_score": self.min_relevance_score,
            },
        )

        return filtered_results

    def _format_context(self, sources: List[SearchResult]) -> str:
        """
        Format retrieved sources into context string.

        Args:
            sources: List of SearchResult objects

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, source in enumerate(sources, 1):
            context_parts.append(f"[Source {i}: {source.chunk.blog_title}]")
            context_parts.append(source.chunk.text)
            context_parts.append("")  # Empty line between sources

        return "\n".join(context_parts)

    def _generate_response(self, query: str, context: str) -> tuple[str, int]:
        """
        Generate a response using OpenAI.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Tuple of (answer, tokens_used)
        """
        try:
            # Format user prompt with context and question
            user_prompt = self.user_prompt_template.format(
                context=context, question=query
            )

            # Create messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            return answer, tokens_used

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            raise RAGEngineError(f"Failed to generate response: {str(e)}")

    def _create_no_context_response(
        self, query: str, retrieval_time_ms: int
    ) -> RAGResponse:
        """Create a response when no relevant context is found."""
        answer = "I couldn't find relevant information in my knowledge base to answer your question. This might be because:\n\n1. The topic isn't covered in the blog posts I have access to\n2. Your question might need to be rephrased\n3. The question is too specific or too broad\n\nCould you try rephrasing your question or asking about a related topic?"

        return RAGResponse(
            query=query,
            answer=answer,
            sources=[],
            context="",
            model=self.model,
            tokens_used=0,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=0,
            total_time_ms=retrieval_time_ms,
            metadata={"no_context": True},
        )

    def _add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        message = ConversationMessage(role=role, content=content)
        self.conversation_history.append(message)

        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length * 2:
            # Keep system message + recent history
            self.conversation_history = self.conversation_history[
                -(self.max_history_length * 2) :
            ]

        logger.debug(
            f"Added {role} message to history",
            extra={"history_length": len(self.conversation_history)},
        )

    def query_with_history(self, query_text: str) -> RAGResponse:
        """
        Process a query with conversation history context.

        Args:
            query_text: User's question

        Returns:
            RAGResponse with answer and metadata
        """
        # For now, just call regular query
        # In future versions, we can use conversation history to enhance context
        return self.query(query_text)

    def stream_query(self, query_text: str, top_k: Optional[int] = None) -> Iterator[str]:
        """
        Process a query and stream the response.

        Args:
            query_text: User's question
            top_k: Number of chunks to retrieve

        Yields:
            Chunks of the generated response
        """
        with LoggerContext() as correlation_id:
            logger.info(
                "Processing streaming RAG query",
                extra={"query_length": len(query_text), "correlation_id": correlation_id},
            )

            try:
                # Retrieve context
                sources = self._retrieve_context(query_text, top_k)

                if not sources:
                    yield "I couldn't find relevant information to answer your question."
                    return

                context = self._format_context(sources)

                # Format user prompt
                user_prompt = self.user_prompt_template.format(
                    context=context, question=query_text
                )

                # Create messages
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Stream response
                stream = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )

                full_response = []
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response.append(content)
                        yield content

                # Add to history
                self._add_to_history("user", query_text)
                self._add_to_history("assistant", "".join(full_response))

            except Exception as e:
                logger.error(f"Streaming query failed: {str(e)}", exc_info=True)
                yield f"Error: {str(e)}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[ConversationMessage]:
        """Get conversation history."""
        return self.conversation_history

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG engine statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_queries": self.total_queries,
            "total_tokens_used": self.total_tokens_used,
            "total_retrieval_time_ms": self.total_retrieval_time_ms,
            "total_generation_time_ms": self.total_generation_time_ms,
            "avg_retrieval_time_ms": (
                self.total_retrieval_time_ms / self.total_queries
                if self.total_queries > 0
                else 0
            ),
            "avg_generation_time_ms": (
                self.total_generation_time_ms / self.total_queries
                if self.total_queries > 0
                else 0
            ),
            "avg_tokens_per_query": (
                self.total_tokens_used / self.total_queries
                if self.total_queries > 0
                else 0
            ),
            "conversation_history_length": len(self.conversation_history),
            "model": self.model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "min_relevance_score": self.min_relevance_score,
        }
