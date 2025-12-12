"""
Embedding generation and management.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig


class EmbeddingService:
    """Service for generating embeddings from text."""
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = SentenceTransformer(self.config.model_name)
        print(f"✓ Loaded embedding model: {self.config.model_name}")
    
    def encode_text(self, text: str) -> List[float]:
        """
        Encode a single text string to embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as list of floats
        """
        return self.model.encode(text).tolist()
    
    def encode_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Encode a batch of texts to embedding vectors.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 1000
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.model.encode(text).tolist() for text in batch]
            embeddings.extend(batch_embeddings)
            
            if show_progress and (i + batch_size) % 1000 == 0:
                print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} items...")
        
        if show_progress:
            print(f"✓ Encoded {len(texts)} items")
        
        return embeddings
    
    def encode_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        text_field: str = "description",
        title_field: str = "title",
        combine_fields: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Encode dataset items to embeddings.
        
        Args:
            dataset: List of dataset items (dicts)
            text_field: Field name for main text content
            title_field: Field name for title
            combine_fields: Whether to combine title and text
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        if combine_fields:
            texts = [
                f"{item.get(title_field, '')} {item.get(text_field, '')}"
                for item in dataset
            ]
        else:
            texts = [item.get(text_field, "") for item in dataset]
        
        if show_progress:
            print(f"Pre-computing embeddings for {len(texts)} items...")
        
        return self.encode_batch(texts, show_progress=show_progress)
    
    @property
    def vector_size(self) -> int:
        """Get the vector size of the embedding model."""
        return self.config.vector_size