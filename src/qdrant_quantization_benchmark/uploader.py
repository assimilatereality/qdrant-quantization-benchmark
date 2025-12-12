"""
Data upload operations for Qdrant with batch processing and retry logic.
"""

import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from .config import UploadConfig


class DataUploader:
    """Handles batch upload of data to Qdrant collections."""
    
    def __init__(self, client: QdrantClient, config: Optional[UploadConfig] = None):
        """
        Initialize data uploader.
        
        Args:
            client: Qdrant client instance
            config: Upload configuration
        """
        self.client = client
        self.config = config or UploadConfig()
    
    def upload_batch(
        self,
        collection_name: str,
        dataset: List[Dict[str, Any]],
        embeddings: List[List[float]],
        named_vector: bool = True,
        vector_name: str = "dense",
        show_progress: bool = True
    ) -> int:
        """
        Upload dataset with precomputed embeddings in batches.
        
        Args:
            collection_name: Name of the collection
            dataset: List of data items
            embeddings: Precomputed embeddings
            named_vector: Whether to use named vectors
            vector_name: Name of the vector field (if named_vector=True)
            show_progress: Whether to show progress
            
        Returns:
            Total number of points uploaded
        """
        if len(dataset) != len(embeddings):
            raise ValueError(
                f"Dataset size ({len(dataset)}) doesn't match embeddings size ({len(embeddings)})"
            )
        
        total_uploaded = 0
        batch_size = self.config.batch_size
        
        for i in range(0, len(dataset), batch_size):
            batch_dataset = dataset[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            points = self._prepare_points(
                batch_dataset, 
                batch_embeddings, 
                start_id=i,
                named_vector=named_vector,
                vector_name=vector_name
            )
            
            if self.config.enable_retry:
                uploaded = self._upload_with_retry(collection_name, points, batch_num=i // batch_size)
            else:
                self.client.upsert(collection_name=collection_name, points=points)
                uploaded = len(points)
            
            total_uploaded += uploaded
            
            if show_progress and total_uploaded % 1000 == 0:
                print(f"  Progress: {total_uploaded}/{len(dataset)} points...")
        
        if show_progress:
            print(f"✓ Uploaded {total_uploaded} points to {collection_name}")
        
        return total_uploaded
    
    def _prepare_points(
        self,
        batch_dataset: List[Dict[str, Any]],
        batch_embeddings: List[List[float]],
        start_id: int,
        named_vector: bool,
        vector_name: str
    ) -> List[models.PointStruct]:
        """
        Prepare PointStruct objects for upload.
        
        Args:
            batch_dataset: Batch of dataset items
            batch_embeddings: Batch of embeddings
            start_id: Starting ID for this batch
            named_vector: Whether to use named vectors
            vector_name: Name of the vector field
            
        Returns:
            List of PointStruct objects
        """
        points = []
        
        for idx, (item, embedding) in enumerate(zip(batch_dataset, batch_embeddings)):
            vector_data = {vector_name: embedding} if named_vector else embedding
            
            points.append(
                models.PointStruct(
                    id=start_id + idx,
                    vector=vector_data,
                    payload=item
                )
            )
        
        return points
    
    def _upload_with_retry(
        self, 
        collection_name: str, 
        points: List[models.PointStruct],
        batch_num: int
    ) -> int:
        """
        Upload points with retry logic and exponential backoff.
        
        Args:
            collection_name: Name of the collection
            points: Points to upload
            batch_num: Batch number (for logging)
            
        Returns:
            Number of points uploaded
            
        Raises:
            ResponseHandlingException: If all retries fail
        """
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                self.client.upsert(collection_name=collection_name, points=points)
                return len(points)
            except ResponseHandlingException as e:
                if attempt < max_retries - 1:
                    wait_time = self.config.initial_backoff * (attempt + 1)
                    print(
                        f"  ⚠ Timeout on batch {batch_num + 1}, "
                        f"retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"  ✗ Failed batch {batch_num + 1} "
                        f"after {max_retries} attempts"
                    )
                    raise
        
        return 0  # Should never reach here
    
    # PRESERVED COMMENTED RETRY CODE FOR REFERENCE
    """
    # Alternative retry implementation with more detailed error handling
    def _upload_with_advanced_retry(
        self,
        collection_name: str,
        points: List[models.PointStruct],
        batch_num: int
    ) -> int:
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                self.client.upsert(collection_name=collection_name, points=points)
                return len(points)
            except ResponseHandlingException as e:
                if attempt < max_retries - 1:
                    wait_time = self.config.initial_backoff * (attempt + 1)  # Linear backoff
                    # Exponential backoff alternative: wait_time = self.config.initial_backoff * (2 ** attempt)
                    print(
                        f"  ⚠ Timeout on batch {batch_num + 1}, "
                        f"retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"  ✗ Failed batch starting at index {batch_num * self.config.batch_size} "
                        f"after {max_retries} attempts"
                    )
                    raise
            except Exception as e:
                print(f"  ✗ Unexpected error on batch {batch_num + 1}: {e}")
                raise
        
        return 0
    """