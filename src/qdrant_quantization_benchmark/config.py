"""
Configuration management for Qdrant benchmark suite.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from qdrant_client import models


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    json_output: bool = False
    verbose: bool = False
    quiet: bool = False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "all-MiniLM-L6-v2"
    vector_size: int = 384
    

@dataclass
class CollectionConfig:
    """Configuration for Qdrant collection."""
    distance: models.Distance = models.Distance.COSINE
    on_disk: bool = True
    timeout: int = 60


@dataclass
class UploadConfig:
    """Configuration for data upload."""
    batch_size: int = 50
    enable_retry: bool = False
    max_retries: int = 3
    initial_backoff: float = 2.0
    

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    warmup_enabled: bool = True
    limit: int = 10
    oversampling_factors: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0, 8.0, 10.0])
    test_queries: List[str] = field(default_factory=lambda: [
        "python machine learning tutorial",
        "javascript web development",
        "learn rust programming security",
        "intermediate algorithms and data structures",
        "practical examples best practices"
    ])


@dataclass
class QuantizationConfig:
    """Configuration for quantization methods."""
    scalar: Dict = field(default_factory=lambda: {
        "config": models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
        "expected_speedup": "2x",
        "expected_compression": "4x"
    })
    binary: Dict = field(default_factory=lambda: {
        "config": models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True,
            )
        ),
        "expected_speedup": "40x",
        "expected_compression": "32x"
    })
    binary_2bit: Dict = field(default_factory=lambda: {
        "config": models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                encoding=models.BinaryQuantizationEncoding.TWO_BITS,
                always_ram=True,
            )
        ),
        "expected_speedup": "20x",
        "expected_compression": "16x"
    })
    
    def get_all_configs(self) -> Dict[str, Dict]:
        """Return all quantization configurations."""
        return {
            "scalar": self.scalar,
            "binary": self.binary,
            "binary_2bit": self.binary_2bit
        }


@dataclass
class QdrantConnectionConfig:
    """Configuration for Qdrant connection."""
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    timeout: int = 60
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.url:
            raise ValueError("QDRANT_URL must be set in environment or .env file")
        if not self.api_key:
            raise ValueError("QDRANT_API_KEY must be set in environment or .env file")


@dataclass
class BenchmarkSuiteConfig:
    """Master configuration for the entire benchmark suite."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    connection: QdrantConnectionConfig = field(default_factory=QdrantConnectionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    @classmethod
    def from_env(cls, logging_config: LoggingConfig = None) -> "BenchmarkSuiteConfig":
        """Create configuration from environment variables."""
        return cls(
            logging=logging_config or LoggingConfig(),
            connection=QdrantConnectionConfig()
        )