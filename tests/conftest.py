"""
Shared pytest fixtures for qdrant-quantization-benchmark tests.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo, Distance, VectorParams

from qdrant_quantization_benchmark.config import (
    BenchmarkSuiteConfig,
    EmbeddingConfig,
    CollectionConfig,
    UploadConfig,
    BenchmarkConfig,
    QuantizationConfig,
    QdrantConnectionConfig,
    LoggingConfig,
)
from qdrant_quantization_benchmark.data_generator import DatasetGenerator
from qdrant_quantization_benchmark.query_generator import QueryGenerator


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for testing without actual Qdrant connection."""
    client = Mock(spec=QdrantClient)
    
    # Mock collection_exists
    client.collection_exists.return_value = False
    
    # Mock get_collection
    mock_collection_info = Mock(spec=CollectionInfo)
    mock_collection_info.vectors_count = 1000
    mock_collection_info.points_count = 1000
    mock_collection_info.status = "green"
    client.get_collection.return_value = mock_collection_info
    
    # Mock upsert (returns None on success)
    client.upsert.return_value = None
    
    # Mock query_points
    mock_response = Mock()
    mock_response.points = []
    client.query_points.return_value = mock_response
    
    return client


@pytest.fixture
def mock_sentence_transformer(mocker):
    """Mock SentenceTransformer to avoid downloading models."""
    mock_model = Mock()
    
    # encode() returns a numpy array of shape (384,)
    mock_model.encode.return_value = np.random.rand(384)
    
    mocker.patch(
        'qdrant_quantization_benchmark.embeddings.SentenceTransformer',
        return_value=mock_model
    )
    
    return mock_model


@pytest.fixture
def embedding_config():
    """Standard embedding configuration."""
    return EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        vector_size=384
    )


@pytest.fixture
def collection_config():
    """Standard collection configuration."""
    return CollectionConfig(
        distance=Distance.COSINE,
        on_disk=True,
        timeout=60
    )


@pytest.fixture
def upload_config():
    """Standard upload configuration."""
    return UploadConfig(
        batch_size=50,
        enable_retry=False,
        max_retries=3,
        initial_backoff=2.0
    )


@pytest.fixture
def benchmark_config():
    """Standard benchmark configuration."""
    return BenchmarkConfig(
        warmup_enabled=True,
        limit=10,
        oversampling_factors=[2.0, 3.0, 5.0],
        test_queries=[
            "python machine learning tutorial",
            "javascript web development",
            "rust security programming"
        ]
    )


@pytest.fixture
def quantization_config():
    """Standard quantization configuration."""
    return QuantizationConfig()


@pytest.fixture
def logging_config():
    """Standard logging configuration."""
    return LoggingConfig(
        level="INFO",
        json_output=False,
        verbose=False,
        quiet=False
    )


@pytest.fixture
def qdrant_connection_config(monkeypatch):
    """Mock Qdrant connection configuration with environment variables."""
    monkeypatch.setenv("QDRANT_URL", "http://test-qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key-12345")
    return QdrantConnectionConfig()


@pytest.fixture
def full_config(monkeypatch, logging_config):
    """Complete benchmark suite configuration with mocked environment."""
    monkeypatch.setenv("QDRANT_URL", "http://test-qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key-12345")
    return BenchmarkSuiteConfig.from_env(logging_config)


@pytest.fixture
def sample_dataset():
    """Generate a small test dataset."""
    generator = DatasetGenerator(seed=42)
    return generator.generate(
        n=20,
        domain_mix={
            'tech': 0.5,
            'medical': 0.5
        }
    )


@pytest.fixture
def sample_queries():
    """Generate sample test queries."""
    generator = QueryGenerator(seed=42)
    return generator.generate_auto_queries(
        n=5,
        domain_mix={
            'tech': 0.6,
            'medical': 0.4
        }
    )


@pytest.fixture
def temp_dataset_file(tmp_path, sample_dataset):
    """Create a temporary dataset file."""
    filepath = tmp_path / "test_dataset.json"
    generator = DatasetGenerator()
    generator.save_dataset(sample_dataset, str(filepath))
    return filepath


@pytest.fixture
def temp_queries_file(tmp_path, sample_queries):
    """Create a temporary queries file."""
    filepath = tmp_path / "test_queries.json"
    generator = QueryGenerator()
    generator.add_manual_queries(sample_queries)
    generator.save_queries(str(filepath))
    return filepath


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings matching sample_dataset."""
    # 20 embeddings of size 384
    return [np.random.rand(384).tolist() for _ in range(20)]


@pytest.fixture
def mock_benchmark_results():
    """Mock benchmark results for testing visualization."""
    baseline = {
        "avg": 45.23,
        "p50": 42.15,
        "p90": 52.18,
        "p95": 55.67,
        "p99": 58.92,
        "p99.5": 59.45,
        "p99.9": 59.87
    }
    
    quantization = {
        "scalar": {
            "no_rescoring": {
                "avg": 22.15,
                "p50": 20.10,
                "p90": 25.20,
                "p95": 27.50,
                "p99": 29.80,
                "p99.5": 30.20,
                "p99.9": 30.50
            },
            "with_rescoring": {
                "avg": 28.45,
                "p50": 26.30,
                "p90": 32.40,
                "p95": 34.70,
                "p99": 37.00,
                "p99.5": 37.50,
                "p99.9": 37.80
            }
        },
        "binary": {
            "no_rescoring": {
                "avg": 15.67,
                "p50": 14.20,
                "p90": 18.30,
                "p95": 19.80,
                "p99": 21.50,
                "p99.5": 22.00,
                "p99.9": 22.30
            },
            "with_rescoring": {
                "avg": 19.23,
                "p50": 17.80,
                "p90": 22.10,
                "p95": 23.90,
                "p99": 25.70,
                "p99.5": 26.20,
                "p99.9": 26.50
            }
        }
    }
    
    return baseline, quantization