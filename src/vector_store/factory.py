"""
Vector Store Factory for Fifi.ai

Creates appropriate vector store instances based on configuration.
"""

from typing import Optional

from src.config import Config, VectorStoreType, get_config
from src.logger import get_logger
from src.vector_store.base import VectorStore, VectorStoreError
from src.vector_store.faiss_store import FAISSVectorStore
from src.vector_store.pinecone_store import PineconeVectorStore

logger = get_logger(__name__)


def create_vector_store(
    config: Optional[Config] = None,
    vector_store_type: Optional[VectorStoreType] = None,
) -> VectorStore:
    """
    Create a vector store instance based on configuration.

    Args:
        config: Configuration object (uses get_config() if None)
        vector_store_type: Override the vector store type from config

    Returns:
        VectorStore instance (FAISS or Pinecone)

    Raises:
        VectorStoreError: If configuration is invalid or store creation fails
    """
    if config is None:
        config = get_config()

    store_type = vector_store_type or config.vector_store_type

    logger.info(f"Creating vector store of type: {store_type}")

    if store_type == VectorStoreType.FAISS:
        return _create_faiss_store(config)
    elif store_type == VectorStoreType.PINECONE:
        return _create_pinecone_store(config)
    else:
        raise VectorStoreError(f"Unsupported vector store type: {store_type}")


def _create_faiss_store(config: Config) -> FAISSVectorStore:
    """Create and initialize FAISS vector store."""
    try:
        # Get embedding dimension from OpenAI model
        dimension = _get_embedding_dimension(config.openai_embedding_model)

        store = FAISSVectorStore(
            dimension=dimension,
            index_path=config.faiss_index_path,
            metadata_path=config.faiss_metadata_path,
            metric="cosine",  # Default to cosine similarity
        )

        logger.info(
            "Created FAISS vector store",
            extra={
                "dimension": dimension,
                "index_path": str(config.faiss_index_path),
            },
        )

        return store

    except Exception as e:
        raise VectorStoreError(f"Failed to create FAISS vector store: {e}")


def _create_pinecone_store(config: Config) -> PineconeVectorStore:
    """Create and initialize Pinecone vector store."""
    if not config.pinecone_api_key:
        raise VectorStoreError(
            "Pinecone API key not configured. Set PINECONE_API_KEY in .env file."
        )

    try:
        # Get embedding dimension from OpenAI model
        dimension = _get_embedding_dimension(config.openai_embedding_model)

        store = PineconeVectorStore(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
            index_name=config.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            namespace="default",
        )

        logger.info(
            "Created Pinecone vector store",
            extra={
                "dimension": dimension,
                "index_name": config.pinecone_index_name,
                "environment": config.pinecone_environment,
            },
        )

        return store

    except Exception as e:
        raise VectorStoreError(f"Failed to create Pinecone vector store: {e}")


def _get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a given OpenAI model.

    Args:
        model_name: OpenAI embedding model name

    Returns:
        Dimension of the embedding vectors
    """
    # OpenAI embedding model dimensions
    dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    dimension = dimensions.get(model_name)
    if dimension is None:
        logger.warning(
            f"Unknown embedding model: {model_name}, defaulting to dimension 1536"
        )
        dimension = 1536

    return dimension
