"""
Embeddings Manager for Fifi.ai

Handles text chunking, embedding generation, and vector storage using FAISS.

This module is responsible for:
- Splitting text into semantic chunks
- Generating embeddings via OpenAI API
- Storing vectors in FAISS index
- Similarity search
- Index persistence (save/load)

Usage:
    from src.embeddings_manager import EmbeddingsManager

    manager = EmbeddingsManager()
    manager.add_documents(blogs)
    results = manager.search("What is RAG?", top_k=5)
    manager.save()
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from src.blog_loader import Blog
from src.config import get_config
from src.logger import get_logger, LoggerContext

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

    Uses OpenAI for embeddings and FAISS for vector storage.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        """
        Initialize the embeddings manager.

        Args:
            index_path: Path to FAISS index file (uses config if None)
            metadata_path: Path to metadata file (uses config if None)
        """
        config = get_config()
        self.config = config

        self.index_path = index_path or config.faiss_index_path
        self.metadata_path = metadata_path or config.faiss_metadata_path

        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.embedding_model = config.openai_embedding_model

        # FAISS index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.dimension: Optional[int] = None

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
                "index_path": str(self.index_path),
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
        Add blog documents to the index.

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
                extra={"total_chunks": len(all_chunks), "avg_chunks_per_blog": len(all_chunks) / len(blogs)},
            )

            # Generate embeddings in batch
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings = self.generate_embeddings_batch(chunk_texts)

            # Initialize or update FAISS index
            if self.index is None:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created new FAISS index with dimension {self.dimension}")

            # Add embeddings to index
            self.index.add(embeddings)
            self.chunks.extend(all_chunks)

            elapsed_time = time.time() - start_time

            logger.info(
                "Documents added to index",
                extra={
                    "total_chunks": len(self.chunks),
                    "index_size": self.index.ntotal,
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
        if self.index is None or self.index.ntotal == 0:
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
            query_vector = query_embedding.reshape(1, -1)

            # Search FAISS index
            distances, indices = self.index.search(query_vector, top_k)

            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(self.chunks):
                    logger.warning(f"Index {idx} out of range, skipping")
                    continue

                chunk = self.chunks[idx]
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity score

                result = SearchResult(
                    chunk=chunk,
                    score=score,
                    distance=float(distance),
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
        Save FAISS index and metadata to disk.

        Raises:
            EmbeddingsError: If save fails
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Cannot save empty index")
            return

        try:
            # Create directory if needed
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Save metadata
            metadata = {
                "chunks": self.chunks,
                "dimension": self.dimension,
                "total_embeddings": self.total_embeddings_generated,
                "total_tokens": self.total_tokens_used,
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(
                "Index saved",
                extra={
                    "index_path": str(self.index_path),
                    "metadata_path": str(self.metadata_path),
                    "total_vectors": self.index.ntotal,
                    "total_chunks": len(self.chunks),
                },
            )

        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}", exc_info=True)
            raise EmbeddingsError(f"Save failed: {str(e)}")

    def load(self) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist

        Raises:
            EmbeddingsError: If load fails
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info("Index files not found, starting fresh")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self.chunks = metadata["chunks"]
            self.dimension = metadata["dimension"]
            self.total_embeddings_generated = metadata.get("total_embeddings", 0)
            self.total_tokens_used = metadata.get("total_tokens", 0)

            logger.info(
                "Index loaded",
                extra={
                    "index_path": str(self.index_path),
                    "total_vectors": self.index.ntotal,
                    "total_chunks": len(self.chunks),
                    "dimension": self.dimension,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}", exc_info=True)
            raise EmbeddingsError(f"Load failed: {str(e)}")

    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = None
        self.chunks = []
        self.dimension = None
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
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "index_exists": self.index is not None,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }
