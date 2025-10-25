"""
Vector Store Module for Fifi.ai

Provides abstraction layer for different vector database backends (FAISS, Pinecone).
"""

from src.vector_store.base import SearchResult, VectorStore, VectorStoreError
from src.vector_store.factory import create_vector_store
from src.vector_store.faiss_store import FAISSVectorStore
from src.vector_store.pinecone_store import PineconeVectorStore

__all__ = [
    "VectorStore",
    "VectorStoreError",
    "SearchResult",
    "create_vector_store",
    "FAISSVectorStore",
    "PineconeVectorStore",
]
