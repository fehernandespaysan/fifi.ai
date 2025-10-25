"""
Pinecone Vector Store Implementation for Fifi.ai

Implements cloud vector storage using Pinecone's managed service.
Provides scalable similarity search with automatic backups and high availability.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from src.logger import get_logger
from src.vector_store.base import SearchResult, VectorStore, VectorStoreError

logger = get_logger(__name__)


class PineconeVectorStore(VectorStore):
    """
    Pinecone-based vector store implementation.

    Stores vectors in Pinecone's cloud service with automatic scaling and management.
    Suitable for production use cases with large datasets.
    """

    def __init__(
        self,
        api_key: str,
        environment: str = "us-west1-gcp-free",
        index_name: str = "fifi-ai-blog-index",
        dimension: int = 1536,
        metric: str = "cosine",
        namespace: str = "default",
    ):
        """
        Initialize Pinecone vector store.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-west1-gcp-free')
            index_name: Name of the Pinecone index
            dimension: Vector dimension (e.g., 1536 for text-embedding-3-small)
            metric: Distance metric ('cosine', 'euclidean', or 'dotproduct')
            namespace: Namespace for vector isolation (useful for multi-tenancy)
        """
        if not api_key:
            raise VectorStoreError("Pinecone API key is required")

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.namespace = namespace

        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=api_key)
            logger.debug(f"Initialized Pinecone client for environment: {environment}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone client: {e}")

        # Initialize or connect to index
        self.index = None
        self._initialize_index()

        logger.info(
            f"PineconeVectorStore initialized",
            extra={
                "index_name": index_name,
                "environment": environment,
                "dimension": dimension,
                "metric": metric,
                "namespace": namespace,
            },
        )

    def _initialize_index(self) -> None:
        """Initialize or connect to Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_exists = any(idx["name"] == self.index_name for idx in existing_indexes)

            if not index_exists:
                # Create new index (Serverless for free tier)
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="gcp", region=self.environment.split("-")[0]),
                )

                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    logger.debug("Waiting for index to be ready...")
                    time.sleep(1)

                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.debug(f"Connecting to existing Pinecone index: {self.index_name}")

            # Connect to index
            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone index: {e}")

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to Pinecone index.

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

        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, vec_id in enumerate(ids):
            vectors_to_upsert.append(
                {
                    "id": vec_id,
                    "values": vectors[i].tolist(),
                    "metadata": metadata[i],
                }
            )

        # Upsert to Pinecone
        try:
            self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)
            logger.debug(
                f"Upserted {n_vectors} vectors to Pinecone",
                extra={"namespace": self.namespace},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vectors to Pinecone: {e}")

        return ids

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors in Pinecone index.

        Args:
            query_vector: Query vector of shape (dimension,)
            top_k: Number of results to return
            filter: Optional metadata filter (Pinecone query filter format)

        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise VectorStoreError("Index not initialized")

        # Ensure query vector is 1D
        if query_vector.ndim != 1:
            query_vector = query_vector.flatten()

        try:
            # Query Pinecone
            response = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
                filter=filter,
            )

            # Convert to SearchResult objects
            results = []
            for match in response.matches:
                results.append(
                    SearchResult(
                        id=match.id,
                        score=float(match.score),
                        distance=1.0 - float(match.score) if self.metric == "cosine" else float(match.score),
                        metadata=match.metadata or {},
                    )
                )

            logger.debug(
                f"Search returned {len(results)} results", extra={"top_k": top_k}
            )
            return results

        except Exception as e:
            raise VectorStoreError(f"Failed to search Pinecone index: {e}")

    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs from Pinecone.

        Args:
            ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        if self.index is None:
            raise VectorStoreError("Index not initialized")

        try:
            self.index.delete(ids=ids, namespace=self.namespace)
            logger.debug(f"Deleted {len(ids)} vectors from Pinecone")
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete vectors from Pinecone: {e}")

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save operation not needed for Pinecone (cloud-managed).

        Pinecone automatically persists all data in the cloud.
        This method is a no-op for compatibility with the interface.
        """
        logger.debug("Save operation not needed for Pinecone (cloud-managed)")

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load operation not needed for Pinecone (cloud-managed).

        Pinecone index is always available in the cloud.
        This method is a no-op for compatibility with the interface.
        """
        logger.debug("Load operation not needed for Pinecone (cloud-managed)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {
                "backend": "pinecone",
                "total_vectors": 0,
                "dimension": self.dimension,
                "metric": self.metric,
            }

        try:
            index_stats = self.index.describe_index_stats()
            total_vectors = index_stats.total_vector_count

            namespace_stats = index_stats.namespaces.get(self.namespace, {})
            namespace_count = namespace_stats.vector_count if namespace_stats else 0

            return {
                "backend": "pinecone",
                "total_vectors": total_vectors,
                "namespace_vectors": namespace_count,
                "dimension": self.dimension,
                "metric": self.metric,
                "index_name": self.index_name,
                "environment": self.environment,
                "namespace": self.namespace,
            }
        except Exception as e:
            logger.warning(f"Failed to get Pinecone stats: {e}")
            return {
                "backend": "pinecone",
                "total_vectors": 0,
                "dimension": self.dimension,
                "metric": self.metric,
                "error": str(e),
            }

    def clear(self) -> None:
        """
        Clear all vectors from the namespace.

        Note: This deletes all vectors in the current namespace.
        Use with caution in production!
        """
        if self.index is None:
            raise VectorStoreError("Index not initialized")

        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Cleared all vectors from namespace: {self.namespace}")
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Pinecone namespace: {e}")

    def batch_add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        batch_size: int = 100,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors in batches (optimized for Pinecone).

        Pinecone recommends batches of 100-200 vectors for optimal performance.

        Args:
            vectors: NumPy array of shape (n_vectors, dimension)
            metadata: List of metadata dictionaries
            batch_size: Number of vectors per batch (default: 100)
            ids: Optional list of unique IDs

        Returns:
            List of all added vector IDs
        """
        # Use parent implementation which calls add_vectors in batches
        # Pinecone's upsert handles batching efficiently
        return super().batch_add_vectors(vectors, metadata, batch_size, ids)
