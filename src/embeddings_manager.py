"""
Embeddings Manager for Fifi.ai

Handles text chunking, embedding generation, and vector storage.

This module is responsible for:
- Splitting text into semantic chunks
- Generating embeddings via OpenAI API
- Storing vectors in vector database (FAISS or Pinecone)
- Similarity search
- Index persistence (save/load)

Usage:
    from src.embeddings_manager import EmbeddingsManager

    manager = EmbeddingsManager()
    manager.add_documents(blogs)
    results = manager.search("What is RAG?", top_k=5)
    manager.save()
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from src.blog_loader import Blog
from src.config import get_config
from src.logger import get_logger, LoggerContext
from src.vector_store import VectorStore, create_vector_store
from src.vector_store.base import SearchResult as VectorSearchResult

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    Represents a text chunk with metadata.

    Attributes:
        text: The chunk text content
        blog_title: Title of the source blog
        blog_file: Path to source blog file
        chunk_index: Index of this chunk in the blog
        start_pos: Starting position in original text
        end_pos: Ending position in original text
        metadata: Additional metadata
    """
    text: str
    blog_title: str
    blog_file: str
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """
    Represents a search result with score and metadata.

    Attributes:
        chunk: The matching text chunk
        score: Similarity score (lower is better for L2 distance)
        distance: Distance from query vector
    """
    chunk: Chunk
    score: float
    distance: float


class EmbeddingsError(Exception):
    """Base exception for embeddings-related errors."""
    pass


class EmbeddingsManager:
    """
    Manages embeddings generation and vector storage.

    Uses OpenAI for embeddings and supports multiple vector store backends
    (FAISS for local storage, Pinecone for cloud storage).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize the embeddings manager.

        Args:
            vector_store: Optional VectorStore instance (creates one from config if None)
        """
        config = get_config()
        self.config = config

        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.embedding_model = config.openai_embedding_model

        # Initialize vector store
        self.vector_store = vector_store or create_vector_store(config)

        # Chunk ID mapping (to retrieve chunks from vector store results)
        self.chunk_map: Dict[str, Chunk] = {}  # Maps chunk_id to Chunk object

        # Metrics
        self.total_embeddings_generated = 0
        self.total_api_calls = 0
        self.total_tokens_used = 0

        logger.info(
            "EmbeddingsManager initialized",
            extra={
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "vector_store": self.vector_store.__class__.__name__,
            },
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of Chunk objects
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        chunk_index = 0
        start_pos = 0

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            if not chunk_text.strip():
                continue

            end_pos = start_pos + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                blog_title=metadata.get("title", "Unknown"),
                blog_file=metadata.get("file_path", "Unknown"),
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=metadata,
            )
            chunks.append(chunk)
            chunk_index += 1
            start_pos = end_pos + 1

        logger.debug(
            f"Chunked text into {len(chunks)} chunks",
            extra={
                "total_words": len(words),
                "chunks_created": len(chunks),
                "blog_title": metadata.get("title"),
            },
        )

        return chunks

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingsError: If API call fails
        """
        try:
            start_time = time.time()

            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )

            embedding = response.data[0].embedding
            elapsed_time = time.time() - start_time

            # Update metrics
            self.total_embeddings_generated += 1
            self.total_api_calls += 1
            self.total_tokens_used += response.usage.total_tokens

            logger.debug(
                "Generated embedding",
                extra={
                    "text_length": len(text),
                    "tokens_used": response.usage.total_tokens,
                    "elapsed_ms": int(elapsed_time * 1000),
                    "dimension": len(embedding),
                },
            )

            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}", exc_info=True)
            raise EmbeddingsError(f"Embedding generation failed: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            Matrix of embeddings (one row per text)

        Raises:
            EmbeddingsError: If API call fails
        """
        if not texts:
            return np.array([])

        try:
            with LoggerContext() as correlation_id:
                start_time = time.time()

                logger.info(
                    f"Generating embeddings for {len(texts)} texts",
                    extra={"batch_size": len(texts), "correlation_id": correlation_id},
                )

                # OpenAI API supports batch embeddings
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                )

                embeddings = [data.embedding for data in response.data]
                embeddings_array = np.array(embeddings, dtype=np.float32)

                elapsed_time = time.time() - start_time

                # Update metrics
                self.total_embeddings_generated += len(texts)
                self.total_api_calls += 1
                self.total_tokens_used += response.usage.total_tokens

                logger.info(
                    "Batch embeddings generated",
                    extra={
                        "batch_size": len(texts),
                        "tokens_used": response.usage.total_tokens,
                        "elapsed_seconds": round(elapsed_time, 2),
                        "embeddings_per_second": round(len(texts) / elapsed_time, 2),
                    },
                )

                return embeddings_array

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}", exc_info=True)
            raise EmbeddingsError(f"Batch embedding generation failed: {str(e)}")

    def add_documents(self, blogs: List[Blog]) -> None:
        """
        Add blog documents to the vector store.

        Args:
            blogs: List of Blog objects to index
        """
        with LoggerContext() as correlation_id:
            logger.info(
                f"Adding {len(blogs)} documents to index",
                extra={"num_blogs": len(blogs), "correlation_id": correlation_id},
            )

            start_time = time.time()
            all_chunks = []

            # Chunk all documents
            for blog in blogs:
                metadata = {
                    "title": blog.title,
                    "file_path": str(blog.file_path),
                    "date": blog.date.isoformat(),
                    "author": blog.author,
                    "tags": blog.tags,
                }
                chunks = self.chunk_text(blog.content, metadata)
                all_chunks.extend(chunks)

            logger.info(
                f"Created {len(all_chunks)} chunks from {len(blogs)} blogs",
                extra={"total_chunks": len(all_chunks), "avg_chunks_per_blog": len(all_chunks) / len(blogs) if blogs else 0},
            )

            # Generate embeddings in batch
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings = self.generate_embeddings_batch(chunk_texts)

            # Prepare metadata for vector store
            chunk_ids = []
            vector_metadata = []

            for chunk in all_chunks:
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)

                # Store chunk in our map
                self.chunk_map[chunk_id] = chunk

                # Prepare metadata for vector store
                metadata_dict = {
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "blog_title": chunk.blog_title,
                    "blog_file": chunk.blog_file,
                    "chunk_index": chunk.chunk_index,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                }
                # Merge with original metadata
                metadata_dict.update(chunk.metadata)
                vector_metadata.append(metadata_dict)

            # Add to vector store
            self.vector_store.add_vectors(
                vectors=embeddings,
                metadata=vector_metadata,
                ids=chunk_ids,
            )

            elapsed_time = time.time() - start_time
            stats = self.vector_store.get_stats()

            logger.info(
                "Documents added to index",
                extra={
                    "total_chunks": len(self.chunk_map),
                    "index_size": stats.get("total_vectors", 0),
                    "elapsed_seconds": round(elapsed_time, 2),
                    "total_tokens": self.total_tokens_used,
                },
            )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar chunks to a query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by relevance

        Raises:
            EmbeddingsError: If search fails or index is empty
        """
        stats = self.vector_store.get_stats()
        if stats.get("total_vectors", 0) == 0:
            raise EmbeddingsError("Index is empty. Add documents first.")

        with LoggerContext() as correlation_id:
            start_time = time.time()

            logger.info(
                "Searching index",
                extra={
                    "query_length": len(query),
                    "top_k": top_k,
                    "correlation_id": correlation_id,
                },
            )

            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Search vector store
            vector_results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
            )

            # Convert vector store results to our SearchResult format
            results = []
            for vec_result in vector_results:
                chunk_id = vec_result.metadata.get("chunk_id")
                if not chunk_id:
                    logger.warning("Result missing chunk_id, skipping")
                    continue

                # Get chunk from our map
                chunk = self.chunk_map.get(chunk_id)
                if not chunk:
                    logger.warning(f"Chunk {chunk_id} not found in map, skipping")
                    continue

                result = SearchResult(
                    chunk=chunk,
                    score=vec_result.score,
                    distance=vec_result.distance,
                )
                results.append(result)

            elapsed_time = time.time() - start_time

            logger.info(
                "Search completed",
                extra={
                    "results_found": len(results),
                    "elapsed_ms": int(elapsed_time * 1000),
                    "top_score": results[0].score if results else None,
                },
            )

            return results

    def save(self) -> None:
        """
        Save vector store and metadata to disk.

        Raises:
            EmbeddingsError: If save fails
        """
        stats = self.vector_store.get_stats()
        if stats.get("total_vectors", 0) == 0:
            logger.warning("Cannot save empty index")
            return

        try:
            # Save vector store (delegates to backend)
            self.vector_store.save()

            # Save chunk map and metrics (for FAISS only - Pinecone stores metadata in cloud)
            if self.vector_store.get_stats().get("backend") == "faiss":
                metadata_path = self.config.faiss_metadata_path.parent / "chunks_metadata.pkl"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)

                import pickle
                metadata = {
                    "chunk_map": self.chunk_map,
                    "total_embeddings": self.total_embeddings_generated,
                    "total_tokens": self.total_tokens_used,
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }

                with open(metadata_path, "wb") as f:
                    pickle.dump(metadata, f)

                logger.info(
                    "Index and metadata saved",
                    extra={
                        "metadata_path": str(metadata_path),
                        "total_chunks": len(self.chunk_map),
                    },
                )
            else:
                logger.info("Vector store saved (cloud-managed)")

        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}", exc_info=True)
            raise EmbeddingsError(f"Save failed: {str(e)}")

    def load(self) -> bool:
        """
        Load vector store and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist

        Raises:
            EmbeddingsError: If load fails
        """
        try:
            # Load vector store (delegates to backend)
            self.vector_store.load()

            # Load chunk map and metrics (for FAISS only)
            if self.vector_store.get_stats().get("backend") == "faiss":
                metadata_path = self.config.faiss_metadata_path.parent / "chunks_metadata.pkl"

                if not metadata_path.exists():
                    logger.info("Chunk metadata not found, starting fresh")
                    return False

                import pickle
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

                self.chunk_map = metadata.get("chunk_map", {})
                self.total_embeddings_generated = metadata.get("total_embeddings", 0)
                self.total_tokens_used = metadata.get("total_tokens", 0)

                logger.info(
                    "Index and metadata loaded",
                    extra={
                        "total_chunks": len(self.chunk_map),
                        "total_vectors": self.vector_store.get_stats().get("total_vectors", 0),
                    },
                )
            else:
                # For Pinecone, we need to rebuild chunk_map from vector store metadata
                # This is a limitation - we'll log a warning
                logger.warning(
                    "Pinecone store loaded - chunk map may be incomplete. "
                    "Consider re-indexing documents if you experience issues."
                )

            return True

        except Exception as e:
            logger.info(f"Failed to load index (may not exist yet): {str(e)}")
            return False

    def clear(self) -> None:
        """Clear the index and metadata."""
        self.vector_store.clear()
        self.chunk_map.clear()
        self.total_embeddings_generated = 0
        self.total_api_calls = 0
        self.total_tokens_used = 0
        logger.info("Index cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with statistics
        """
        vector_stats = self.vector_store.get_stats()

        return {
            "total_vectors": vector_stats.get("total_vectors", 0),
            "total_chunks": len(self.chunk_map),
            "dimension": vector_stats.get("dimension"),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "vector_store_backend": vector_stats.get("backend"),
            "vector_store_stats": vector_stats,
        }
