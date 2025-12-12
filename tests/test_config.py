"""
Tests for configuration dataclasses and validation.
"""

import pytest
import os
from qdrant_client.models import Distance

from qdrant_quantization_benchmark.config import (
    LoggingConfig,
    EmbeddingConfig,
    CollectionConfig,
    UploadConfig,
    BenchmarkConfig,
    QuantizationConfig,
    QdrantConnectionConfig,
    BenchmarkSuiteConfig,
)


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""
    
    def test_default_values(self):
        """Test default logging configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.json_output is False
        assert config.verbose is False
        assert config.quiet is False
    
    def test_custom_values(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            level="DEBUG",
            json_output=True,
            verbose=True,
            quiet=False
        )
        assert config.level == "DEBUG"
        assert config.json_output is True
        assert config.verbose is True
        assert config.quiet is False


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""
    
    def test_default_values(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.vector_size == 384
    
    def test_custom_model(self):
        """Test custom model configuration."""
        config = EmbeddingConfig(
            model_name="custom-model",
            vector_size=768
        )
        assert config.model_name == "custom-model"
        assert config.vector_size == 768


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""
    
    def test_default_values(self):
        """Test default collection configuration."""
        config = CollectionConfig()
        assert config.distance == Distance.COSINE
        assert config.on_disk is True
        assert config.timeout == 60
    
    def test_custom_configuration(self):
        """Test custom collection configuration."""
        config = CollectionConfig(
            distance=Distance.EUCLID,
            on_disk=False,
            timeout=120
        )
        assert config.distance == Distance.EUCLID
        assert config.on_disk is False
        assert config.timeout == 120


class TestUploadConfig:
    """Tests for UploadConfig dataclass."""
    
    def test_default_values(self):
        """Test default upload configuration."""
        config = UploadConfig()
        assert config.batch_size == 50
        assert config.enable_retry is False  # Default is False in actual config
        assert config.max_retries == 3
        assert config.initial_backoff == 2.0
    
    def test_custom_retry_settings(self):
        """Test custom retry configuration."""
        config = UploadConfig(
            batch_size=100,
            enable_retry=True,
            max_retries=5,
            initial_backoff=5.0
        )
        assert config.batch_size == 100
        assert config.enable_retry is True
        assert config.max_retries == 5
        assert config.initial_backoff == 5.0


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""
    
    def test_default_values(self):
        """Test default benchmark configuration."""
        config = BenchmarkConfig()
        assert config.warmup_enabled is True
        assert config.limit == 10
        assert config.oversampling_factors == [2.0, 3.0, 5.0, 8.0, 10.0]
        assert len(config.test_queries) == 5
    
    def test_custom_configuration(self):
        """Test custom benchmark configuration."""
        config = BenchmarkConfig(
            warmup_enabled=False,
            limit=20,
            oversampling_factors=[3.0, 5.0],
            test_queries=["query1", "query2"]
        )
        assert config.warmup_enabled is False
        assert config.limit == 20
        assert config.oversampling_factors == [3.0, 5.0]
        assert config.test_queries == ["query1", "query2"]


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""
    
    def test_default_values(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()
        
        # Check that all three configs exist
        assert "config" in config.scalar
        assert "expected_speedup" in config.scalar
        assert "expected_compression" in config.scalar
        
        assert "config" in config.binary
        assert "expected_speedup" in config.binary
        assert "expected_compression" in config.binary
        
        assert "config" in config.binary_2bit
        assert "expected_speedup" in config.binary_2bit
        assert "expected_compression" in config.binary_2bit
    
    def test_get_all_configs(self):
        """Test retrieving all quantization configs."""
        config = QuantizationConfig()
        all_configs = config.get_all_configs()
        
        assert "scalar" in all_configs
        assert "binary" in all_configs
        assert "binary_2bit" in all_configs
        
        # Verify structure
        for method in ["scalar", "binary", "binary_2bit"]:
            assert "config" in all_configs[method]
            assert "expected_speedup" in all_configs[method]
            assert "expected_compression" in all_configs[method]


class TestQdrantConnectionConfig:
    """Tests for QdrantConnectionConfig dataclass."""
    
    def test_initialization_with_env_vars(self, monkeypatch):
        """Test initialization with environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        config = QdrantConnectionConfig()
        assert config.url == "http://localhost:6333"
        assert config.api_key == "test-key"
        assert config.timeout == 60
    
    def test_missing_url_raises_error(self, monkeypatch):
        """Test that missing URL raises ValueError."""
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        with pytest.raises(ValueError, match="QDRANT_URL must be set"):
            QdrantConnectionConfig()
    
    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="QDRANT_API_KEY must be set"):
            QdrantConnectionConfig()
    
    def test_custom_timeout(self, monkeypatch):
        """Test custom timeout configuration."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        config = QdrantConnectionConfig(timeout=120)
        assert config.timeout == 120


class TestBenchmarkSuiteConfig:
    """Tests for BenchmarkSuiteConfig dataclass."""
    
    def test_default_initialization(self, monkeypatch):
        """Test that BenchmarkSuiteConfig can be initialized with defaults."""
        # Set required environment variables for connection config
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        config = BenchmarkSuiteConfig()
        
        assert config.logging.level == "INFO"
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        assert config.embedding.vector_size == 384
        assert config.collection.distance == Distance.COSINE
        assert config.upload.batch_size == 50
        assert config.benchmark.warmup_enabled is True
    
    def test_from_env_basic(self, monkeypatch):
        """Test creating config from environment."""
        monkeypatch.setenv("QDRANT_URL", "http://test:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "secret-key")
        
        config = BenchmarkSuiteConfig.from_env()
        
        assert config.connection.url == "http://test:6333"
        assert config.connection.api_key == "secret-key"
        assert config.logging.level == "INFO"
    
    def test_from_env_with_custom_logging(self, monkeypatch):
        """Test from_env with custom logging config."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        custom_logging = LoggingConfig(level="DEBUG", verbose=True)
        config = BenchmarkSuiteConfig.from_env(logging_config=custom_logging)
        
        assert config.logging.level == "DEBUG"
        assert config.logging.verbose is True
    
    def test_from_env_requires_url(self, monkeypatch):
        """Test that from_env requires QDRANT_URL."""
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="QDRANT_URL must be set"):
            BenchmarkSuiteConfig.from_env()
    
    def test_nested_configs_initialized(self, monkeypatch):
        """Test that all nested configs are properly initialized."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        config = BenchmarkSuiteConfig()
        
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.connection, QdrantConnectionConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.collection, CollectionConfig)
        assert isinstance(config.upload, UploadConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert isinstance(config.quantization, QuantizationConfig)
    
    def test_quantization_configs_accessible(self, monkeypatch):
        """Test accessing quantization configs through suite config."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        config = BenchmarkSuiteConfig()
        quant_configs = config.quantization.get_all_configs()
        
        assert "scalar" in quant_configs
        assert "binary" in quant_configs
        assert "binary_2bit" in quant_configs