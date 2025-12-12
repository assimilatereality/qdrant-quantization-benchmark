"""
Tests for Qdrant collection management operations.
"""

import pytest
from unittest.mock import Mock, call
from qdrant_client.models import Distance, VectorParams

from qdrant_quantization_benchmark.qdrant_manager import QdrantCollectionManager
from qdrant_quantization_benchmark.config import CollectionConfig, EmbeddingConfig


class TestQdrantCollectionManager:
    """Tests for QdrantCollectionManager class."""
    
    def test_initialization(self, mock_qdrant_client, collection_config, embedding_config):
        """Test manager initialization."""
        manager = QdrantCollectionManager(
            mock_qdrant_client,
            collection_config,
            embedding_config
        )
        
        assert manager.client == mock_qdrant_client
        assert manager.collection_config == collection_config
        assert manager.embedding_config == embedding_config
    
    def test_initialization_with_defaults(self, mock_qdrant_client):
        """Test initialization with default configs."""
        manager = QdrantCollectionManager(mock_qdrant_client)
        
        assert isinstance(manager.collection_config, CollectionConfig)
        assert isinstance(manager.embedding_config, EmbeddingConfig)
    
    def test_collection_exists_true(self, mock_qdrant_client):
        """Test collection_exists when collection exists."""
        mock_qdrant_client.collection_exists.return_value = True
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        result = manager.collection_exists("test_collection")
        
        assert result is True
        mock_qdrant_client.collection_exists.assert_called_once_with("test_collection")
    
    def test_collection_exists_false(self, mock_qdrant_client):
        """Test collection_exists when collection doesn't exist."""
        mock_qdrant_client.collection_exists.return_value = False
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        result = manager.collection_exists("nonexistent")
        
        assert result is False
    
    def test_delete_collection_when_exists(self, mock_qdrant_client, capsys):
        """Test deleting an existing collection."""
        mock_qdrant_client.collection_exists.return_value = True
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        manager.delete_collection("test_collection")
        
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")
        captured = capsys.readouterr()
        assert "Deleted existing collection" in captured.out
    
    def test_delete_collection_when_not_exists(self, mock_qdrant_client):
        """Test deleting a non-existent collection."""
        mock_qdrant_client.collection_exists.return_value = False
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        manager.delete_collection("nonexistent")
        
        # Should not call delete
        mock_qdrant_client.delete_collection.assert_not_called()
    
    def test_create_hybrid_collection(self, mock_qdrant_client, capsys):
        """Test creating a hybrid collection."""
        manager = QdrantCollectionManager(mock_qdrant_client)
        manager.create_hybrid_collection("test_hybrid")
        
        # Verify create_collection was called with correct parameters
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        
        assert call_args[1]['collection_name'] == "test_hybrid"
        assert 'dense' in call_args[1]['vectors_config']
        assert 'sparse' in call_args[1]['sparse_vectors_config']
        
        captured = capsys.readouterr()
        assert "Created hybrid collection" in captured.out
    
    def test_create_standard_collection(self, mock_qdrant_client, capsys):
        """Test creating a standard collection."""
        manager = QdrantCollectionManager(mock_qdrant_client)
        manager.create_standard_collection("test_standard")
        
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        
        assert call_args[1]['collection_name'] == "test_standard"
        assert isinstance(call_args[1]['vectors_config'], VectorParams)
        
        captured = capsys.readouterr()
        assert "Created standard collection" in captured.out
    
    def test_create_quantized_collection(self, mock_qdrant_client, quantization_config, capsys):
        """Test creating a quantized collection."""
        manager = QdrantCollectionManager(mock_qdrant_client)
        
        quant_config = quantization_config.get_all_configs()['scalar']['config']
        manager.create_quantized_collection("test_quantized", quant_config)
        
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        
        assert call_args[1]['collection_name'] == "test_quantized"
        assert call_args[1]['quantization_config'] == quant_config
        
        captured = capsys.readouterr()
        assert "Created quantized collection" in captured.out
    
    def test_recreate_collection_standard(self, mock_qdrant_client):
        """Test recreating a standard collection."""
        mock_qdrant_client.collection_exists.return_value = True
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        manager.recreate_collection("test_collection", collection_type="standard")
        
        # Should delete then create
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")
        mock_qdrant_client.create_collection.assert_called_once()
    
    def test_recreate_collection_hybrid(self, mock_qdrant_client):
        """Test recreating a hybrid collection."""
        mock_qdrant_client.collection_exists.return_value = True
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        manager.recreate_collection("test_collection", collection_type="hybrid")
        
        mock_qdrant_client.delete_collection.assert_called_once()
        
        # Verify hybrid config was used
        call_args = mock_qdrant_client.create_collection.call_args
        assert 'sparse_vectors_config' in call_args[1]
    
    def test_recreate_collection_quantized(self, mock_qdrant_client, quantization_config):
        """Test recreating a quantized collection."""
        mock_qdrant_client.collection_exists.return_value = True
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        quant_config = quantization_config.get_all_configs()['binary']['config']
        
        manager.recreate_collection(
            "test_collection",
            collection_type="quantized",
            quantization_config=quant_config
        )
        
        mock_qdrant_client.delete_collection.assert_called_once()
        
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args[1]['quantization_config'] == quant_config
    
    def test_recreate_quantized_without_config_raises_error(self, mock_qdrant_client):
        """Test that recreating quantized collection without config raises error."""
        manager = QdrantCollectionManager(mock_qdrant_client)
        
        with pytest.raises(ValueError, match="quantization_config required"):
            manager.recreate_collection("test", collection_type="quantized")
    
    def test_get_collection_info(self, mock_qdrant_client):
        """Test retrieving collection information."""
        mock_qdrant_client.collection_exists.return_value = True
        
        # Setup mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_info
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        info = manager.get_collection_info("test_collection")
        
        assert info['name'] == "test_collection"
        assert info['vectors_count'] == 1000
        assert info['points_count'] == 1000
        assert info['status'] == "green"
        
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")
    
    def test_get_collection_info_nonexistent_raises_error(self, mock_qdrant_client):
        """Test that getting info for nonexistent collection raises error."""
        mock_qdrant_client.collection_exists.return_value = False
        
        manager = QdrantCollectionManager(mock_qdrant_client)
        
        with pytest.raises(ValueError, match="does not exist"):
            manager.get_collection_info("nonexistent")
    
    def test_vector_size_from_embedding_config(self, mock_qdrant_client):
        """Test that vector size comes from embedding config."""
        embedding_config = EmbeddingConfig(vector_size=512)
        
        manager = QdrantCollectionManager(
            mock_qdrant_client,
            embedding_config=embedding_config
        )
        manager.create_standard_collection("test")
        
        call_args = mock_qdrant_client.create_collection.call_args
        vectors_config = call_args[1]['vectors_config']
        assert vectors_config.size == 512
    
    def test_distance_metric_from_collection_config(self, mock_qdrant_client):
        """Test that distance metric comes from collection config."""
        collection_config = CollectionConfig(distance=Distance.EUCLID)
        
        manager = QdrantCollectionManager(
            mock_qdrant_client,
            collection_config=collection_config
        )
        manager.create_standard_collection("test")
        
        call_args = mock_qdrant_client.create_collection.call_args
        vectors_config = call_args[1]['vectors_config']
        assert vectors_config.distance == Distance.EUCLID