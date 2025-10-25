"""
Base Vector Store Interface for Fifi.ai

Defines the abstract interface that all vector store implementations must follow.
This enables swapping between different backends (FAISS, Pinecone, etc.) seamlessly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SearchResult:
    """
    Represents a single search result from vector store.

    Attributes:
        id: Unique identifier for the vector
        score: Similarity score (higher is better after normalization)
        distance: Raw distance metric from vector store
        metadata: Associated metadata dictionary
        text: The text chunk (if available in metadata)
    """

    id: str
    score: float
    distance: float
    metadata: Dict[str, Any]
    text: Optional[str] = None

    def __post_init__(self):
        """Extract text from metadata if available."""
        if self.text is None and "text" in self.metadata:
            self.text = self.metadata["text"]


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    All vector store backends (FAISS, Pinecone, etc.) must implement this interface.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the vector store.

        Args:
            **kwargs: Backend-specific configuration
        """
        pass

    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to the store with associated metadata.

        Args:
            vectors: NumPy array of shape (n_vectors, dimension)
            metadata: List of metadata dictionaries (one per vector)
            ids: Optional list of unique IDs for each vector

        Returns:
            List of IDs for the added vectors

        Raises:
            VectorStoreError: If operation fails
        """
        pass

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int = 5, filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector of shape (dimension,)
            top_k: Number of results to return
            filter: Optional metadata filter (backend-specific)

        Returns:
            List of SearchResult objects, sorted by relevance

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted

        Raises:
            VectorStoreError: If deletion fails
        """
        pass

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save the vector store to disk (for local stores like FAISS).

        Args:
            path: Optional path to save to

        Raises:
            VectorStoreError: If save fails
            NotImplementedError: If backend doesn't support local persistence
        """
        pass

    @abstractmethod
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load the vector store from disk (for local stores like FAISS).

        Args:
            path: Optional path to load from

        Raises:
            VectorStoreError: If load fails
            NotImplementedError: If backend doesn't support local persistence
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats like:
                - total_vectors: Number of vectors in store
                - dimension: Vector dimension
                - backend: Store backend type
                - Additional backend-specific stats
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all vectors from the store.

        Raises:
            VectorStoreError: If clear operation fails
        """
        pass

    def batch_add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        batch_size: int = 100,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors in batches for efficiency.

        Default implementation calls add_vectors in batches.
        Backends can override for optimized batch processing.

        Args:
            vectors: NumPy array of shape (n_vectors, dimension)
            metadata: List of metadata dictionaries
            batch_size: Number of vectors per batch
            ids: Optional list of unique IDs

        Returns:
            List of all added vector IDs
        """
        all_ids = []
        n_vectors = len(vectors)

        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch_vectors = vectors[i:end_idx]
            batch_metadata = metadata[i:end_idx]
            batch_ids = ids[i:end_idx] if ids else None

            batch_result_ids = self.add_vectors(batch_vectors, batch_metadata, batch_ids)
            all_ids.extend(batch_result_ids)

        return all_ids

    def __repr__(self) -> str:
        """String representation of the vector store."""
        stats = self.get_stats()
        return f"{self.__class__.__name__}(backend={stats.get('backend')}, vectors={stats.get('total_vectors', 0)})"
