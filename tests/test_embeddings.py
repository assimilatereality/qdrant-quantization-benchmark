"""
Tests for embedding generation service.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from qdrant_quantization_benchmark.embeddings import EmbeddingService
from qdrant_quantization_benchmark.config import EmbeddingConfig


@pytest.fixture
def mock_embedding_service(mocker):
    """Create EmbeddingService with mocked SentenceTransformer."""
    # Mock the SentenceTransformer class
    mock_model = mocker.Mock()
    mock_model.encode.return_value = np.array([0.1] * 384)
    
    mocker.patch(
        'qdrant_quantization_benchmark.embeddings.SentenceTransformer',
        return_value=mock_model
    )
    
    config = EmbeddingConfig(model_name="test-model")
    service = EmbeddingService(config)
    
    return service


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a test document",
        "Another test document",
        "Yet another document"
    ]


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {'title': 'Title 1', 'description': 'Description 1'},
        {'title': 'Title 2', 'description': 'Description 2'},
        {'title': 'Title 3', 'description': 'Description 3'}
    ]


class TestEmbeddingService:
    """Tests for EmbeddingService class."""
    
    def test_initialization(self):
        """Test service initialization."""
        config = EmbeddingConfig()
        service = EmbeddingService(config)
        assert service.config == config
    
    def test_encode_text(self, mock_embedding_service):
        """Test single text encoding."""
        embedding = mock_embedding_service.encode_text("test text")
        
        assert isinstance(embedding, np.ndarray) or isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_encode_batch(self, mock_embedding_service, sample_texts):
        """Test batch encoding."""
        embeddings = mock_embedding_service.encode_batch(sample_texts)
        
        assert len(embeddings) == len(sample_texts)
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_encode_batch_without_progress(self, mock_embedding_service, sample_texts):
        """Test batch encoding without progress display."""
        embeddings = mock_embedding_service.encode_batch(
            sample_texts, 
            show_progress=False
        )
        
        assert len(embeddings) == len(sample_texts)
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_encode_dataset(self, mock_embedding_service, sample_dataset):
        """Test dataset encoding (combines title + description)."""
        embeddings = mock_embedding_service.encode_dataset(sample_dataset)
        
        assert len(embeddings) == len(sample_dataset)
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_encode_empty_batch(self, mock_embedding_service):
        """Test encoding empty batch."""
        embeddings = mock_embedding_service.encode_batch([])
        
        assert len(embeddings) == 0