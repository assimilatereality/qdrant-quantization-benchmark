"""
Qdrant collection management operations.
"""

from typing import Optional
from qdrant_client import QdrantClient, models

from .config import CollectionConfig, EmbeddingConfig


class QdrantCollectionManager:
    """Manages Qdrant collection lifecycle operations."""
    
    def __init__(
        self, 
        client: QdrantClient,
        collection_config: Optional[CollectionConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize collection manager.
        
        Args:
            client: Qdrant client instance
            collection_config: Collection configuration
            embedding_config: Embedding configuration
        """
        self.client = client
        self.collection_config = collection_config or CollectionConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists, False otherwise
        """
        return self.client.collection_exists(collection_name)
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection if it exists.
        
        Args:
            collection_name: Name of the collection to delete
        """
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"✓ Deleted existing collection: {collection_name}")
    
    def create_hybrid_collection(self, collection_name: str) -> None:
        """
        Create a hybrid collection with both dense and sparse vectors.
        
        Args:
            collection_name: Name of the collection to create
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedding_config.vector_size,
                    distance=self.collection_config.distance
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
        print(f"✓ Created hybrid collection: {collection_name}")
    
    def create_standard_collection(self, collection_name: str) -> None:
        """
        Create a standard collection with dense vectors only.
        
        Args:
            collection_name: Name of the collection to create
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_config.vector_size,
                distance=self.collection_config.distance,
                on_disk=self.collection_config.on_disk
            )
        )
        print(f"✓ Created standard collection: {collection_name}")
    
    def create_quantized_collection(
        self, 
        collection_name: str,
        quantization_config: models.QuantizationConfig
    ) -> None:
        """
        Create a collection with quantization enabled.
        
        Args:
            collection_name: Name of the collection to create
            quantization_config: Quantization configuration
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_config.vector_size,
                distance=self.collection_config.distance,
                on_disk=True,  # Store originals on disk for quantization
            ),
            quantization_config=quantization_config
        )
        print(f"✓ Created quantized collection: {collection_name}")
    
    def recreate_collection(
        self, 
        collection_name: str,
        collection_type: str = "standard",
        quantization_config: Optional[models.QuantizationConfig] = None
    ) -> None:
        """
        Delete and recreate a collection.
        
        Args:
            collection_name: Name of the collection
            collection_type: Type of collection ("standard", "hybrid", "quantized")
            quantization_config: Quantization config (required for "quantized" type)
        """
        self.delete_collection(collection_name)
        
        if collection_type == "hybrid":
            self.create_hybrid_collection(collection_name)
        elif collection_type == "quantized":
            if quantization_config is None:
                raise ValueError("quantization_config required for quantized collection")
            self.create_quantized_collection(collection_name, quantization_config)
        else:
            self.create_standard_collection(collection_name)
    
    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information dictionary
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }