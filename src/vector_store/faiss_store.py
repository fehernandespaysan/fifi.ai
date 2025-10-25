"""
FAISS Vector Store Implementation for Fifi.ai

Implements local vector storage using Facebook's FAISS library.
Provides fast similarity search with persistence to disk.
"""

import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.logger import get_logger
from src.vector_store.base import SearchResult, VectorStore, VectorStoreError

logger = get_logger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation.

    Stores vectors locally using FAISS index with metadata in pickle file.
    Suitable for small to medium datasets (up to ~1M vectors).
    """

    def __init__(
        self,
        dimension: int = 1536,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        metric: str = "l2",
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Vector dimension (e.g., 1536 for text-embedding-3-small)
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
            metric: Distance metric ('l2' or 'cosine')
        """
        self.dimension = dimension
        self.index_path = index_path or Path("data/faiss_index.faiss")
        self.metadata_path = metadata_path or Path("data/faiss_metadata.pkl")
        self.metric = metric

        # Initialize index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.id_to_index: Dict[str, int] = {}  # Map vector ID to FAISS index
        self.index_to_id: Dict[int, str] = {}  # Map FAISS index to vector ID
        self.metadata_store: Dict[str, Dict[str, Any]] = {}  # Store metadata by ID

        self._initialize_index()

        logger.debug(
            f"FAISSVectorStore initialized",
            extra={
                "dimension": dimension,
                "metric": metric,
                "index_path": str(self.index_path),
            },
        )

    def _initialize_index(self) -> None:
        """Initialize a new FAISS index."""
        if self.metric == "cosine":
            # For cosine similarity, we normalize vectors and use L2 distance
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        else:
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)

        logger.debug(f"Initialized new FAISS index with metric={self.metric}")

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to FAISS index.

        Args:
            vectors: NumPy array of shape (n_vectors, dimension)
            metadata: List of metadata dicts
            ids: Optional list of unique IDs

        Returns:
            List of vector IDs
        """
        if self.index is None:
            raise VectorStoreError("Index not initialized")

        n_vectors = len(vectors)
        if len(metadata) != n_vectors:
            raise VectorStoreError(
                f"Metadata count ({len(metadata)}) doesn't match vector count ({n_vectors})"
            )

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n_vectors)]
        elif len(ids) != n_vectors:
            raise VectorStoreError(
                f"ID count ({len(ids)}) doesn't match vector count ({n_vectors})"
            )

        # Validate vector dimensions
        if vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Vector dimension ({vectors.shape[1]}) doesn't match index dimension ({self.dimension})"
            )

        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        # Get starting index in FAISS
        start_idx = self.index.ntotal

        # Add vectors to FAISS index
        self.index.add(vectors.astype(np.float32))

        # Store ID mappings and metadata
        for i, vec_id in enumerate(ids):
            faiss_idx = start_idx + i
            self.id_to_index[vec_id] = faiss_idx
            self.index_to_id[faiss_idx] = vec_id
            self.metadata_store[vec_id] = metadata[i]

        logger.debug(
            f"Added {n_vectors} vectors to FAISS index",
            extra={"total_vectors": self.index.ntotal},
        )

        return ids

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors in FAISS index.

        Args:
            query_vector: Query vector of shape (dimension,)
            top_k: Number of results to return
            filter: Not implemented for FAISS (metadata filtering after search)

        Returns:
            List of SearchResult objects
        """
        if self.index is None or self.index.ntotal == 0:
            raise VectorStoreError("Index is empty")

        # Reshape query vector if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)

        # Perform search
        distances, indices = self.index.search(query_vector.astype(np.float32), top_k)

        # Convert to SearchResult objects
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled results
                continue

            vec_id = self.index_to_id.get(int(idx))
            if vec_id is None:
                logger.warning(f"No ID found for FAISS index {idx}")
                continue

            metadata = self.metadata_store.get(vec_id, {})

            # Apply metadata filter if provided
            if filter:
                if not self._matches_filter(metadata, filter):
                    continue

            # Convert distance to score (higher is better)
            if self.metric == "cosine":
                score = float(dist)  # Inner product is already similarity
            else:
                # L2 distance: convert to similarity (1 / (1 + distance))
                score = 1.0 / (1.0 + float(dist))

            results.append(
                SearchResult(
                    id=vec_id,
                    score=score,
                    distance=float(dist),
                    metadata=metadata,
                )
            )

        logger.debug(f"Search returned {len(results)} results")
        return results

    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs.

        Note: FAISS doesn't support deletion efficiently.
        This method marks vectors as deleted in metadata but doesn't remove from index.
        For true deletion, rebuild the index.
        """
        deleted_count = 0
        for vec_id in ids:
            if vec_id in self.id_to_index:
                idx = self.id_to_index[vec_id]
                del self.id_to_index[vec_id]
                del self.index_to_id[idx]
                del self.metadata_store[vec_id]
                deleted_count += 1

        logger.debug(f"Deleted {deleted_count} vectors (marked as deleted)")
        return deleted_count

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            path: Optional path to save index (uses self.index_path if None)
        """
        if self.index is None:
            raise VectorStoreError("No index to save")

        index_path = path or self.index_path
        metadata_path = self.metadata_path

        # Ensure parent directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_bundle = {
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "metadata_store": self.metadata_store,
            "dimension": self.dimension,
            "metric": self.metric,
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_bundle, f)

        logger.info(
            f"Saved FAISS index to {index_path}",
            extra={"vectors": self.index.ntotal, "metadata_path": str(metadata_path)},
        )

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            path: Optional path to load index from (uses self.index_path if None)
        """
        index_path = path or self.index_path
        metadata_path = self.metadata_path

        if not index_path.exists():
            raise VectorStoreError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise VectorStoreError(f"Metadata file not found: {metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata_bundle = pickle.load(f)

        self.id_to_index = metadata_bundle["id_to_index"]
        self.index_to_id = metadata_bundle["index_to_id"]
        self.metadata_store = metadata_bundle["metadata_store"]
        self.dimension = metadata_bundle["dimension"]
        self.metric = metadata_bundle["metric"]

        logger.info(
            f"Loaded FAISS index from {index_path}",
            extra={"vectors": self.index.ntotal, "dimension": self.dimension},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        return {
            "backend": "faiss",
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "metric": self.metric,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self._initialize_index()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.metadata_store.clear()
        logger.info("Cleared FAISS index")

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria.

        Simple implementation: checks if all filter key-value pairs exist in metadata.
        """
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True
