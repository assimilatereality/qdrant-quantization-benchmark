"""
Tests for performance benchmarking operations.
"""

import pytest
import time
from unittest.mock import Mock
import numpy as np

from qdrant_quantization_benchmark.benchmarking import PerformanceBenchmark
from qdrant_quantization_benchmark.config import BenchmarkConfig


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark class."""
    
    def test_initialization(self, mock_qdrant_client, mock_sentence_transformer, benchmark_config):
        """Test benchmark initialization."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(
            mock_qdrant_client,
            embedding_service,
            benchmark_config
        )
        
        assert benchmark.client == mock_qdrant_client
        assert benchmark.embedding_service == embedding_service
        assert benchmark.config == benchmark_config
    
    def test_warmup_enabled(self, mock_qdrant_client, mock_sentence_transformer):
        """Test warmup when enabled."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        config = BenchmarkConfig(warmup_enabled=True)
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        benchmark.warmup("test_collection", using="dense")
        
        # Should call query_points once for warmup
        mock_qdrant_client.query_points.assert_called_once()
    
    def test_warmup_disabled(self, mock_qdrant_client, mock_sentence_transformer):
        """Test warmup when disabled."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        config = BenchmarkConfig(warmup_enabled=False)
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        benchmark.warmup("test_collection", using="dense")
        
        # Should not call query_points
        mock_qdrant_client.query_points.assert_not_called()
    
    def test_measure_search_latency(self, mock_qdrant_client, mock_sentence_transformer):
        """Test measuring search latency."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        # Setup mock response
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        
        config = BenchmarkConfig(test_queries=["query1", "query2", "query3"])
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        metrics = benchmark.measure_search_latency(
            collection_name="test",
            test_queries=config.test_queries,
            using="dense",
            label="Test"
        )
        
        # Check all expected metrics are present
        assert "avg" in metrics
        assert "p50" in metrics
        assert "p90" in metrics
        assert "p95" in metrics
        assert "p99" in metrics
        assert "p99.5" in metrics
        assert "p99.9" in metrics
        
        # Should be called for warmup + 3 queries
        assert mock_qdrant_client.query_points.call_count == 4
    
    def test_calculate_metrics(self, mock_qdrant_client, mock_sentence_transformer):
        """Test metrics calculation."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service)
        
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        metrics = benchmark._calculate_metrics(latencies)
        
        assert metrics["avg"] == 55.0
        assert metrics["p50"] == 55.0  # Median of 10 values
        assert metrics["p90"] == 91.0
        assert metrics["p99"] == 99.1
    
    def test_benchmark_quantization(self, mock_qdrant_client, mock_sentence_transformer):
        """Test benchmarking quantized collections."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        
        config = BenchmarkConfig(test_queries=["query1", "query2"])
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        results = benchmark.benchmark_quantization(
            collection_name="quantized_test",
            test_queries=config.test_queries,
            method_name="scalar"
        )
        
        # Check structure
        assert "no_rescoring" in results
        assert "with_rescoring" in results
        
        # Each should have metrics
        assert "avg" in results["no_rescoring"]
        assert "avg" in results["with_rescoring"]
    
    def test_tune_oversampling(self, mock_qdrant_client, mock_sentence_transformer):
        """Test oversampling factor tuning."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        
        config = BenchmarkConfig(test_queries=["query1"])
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        factors = [2.0, 3.0, 5.0]
        results = benchmark.tune_oversampling(
            collection_name="test",
            test_queries=config.test_queries,
            factors=factors
        )
        
        # Check all factors tested
        assert 2.0 in results
        assert 3.0 in results
        assert 5.0 in results
        
        # Each should have latency metrics
        for factor, metrics in results.items():
            assert "avg_latency" in metrics
            assert "p95_latency" in metrics
    
    def test_measure_accuracy_retention(self, mock_qdrant_client, mock_sentence_transformer):
        """Test accuracy retention measurement."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        # Setup mock responses
        baseline_response = Mock()
        baseline_response.points = [Mock(id=i) for i in range(10)]
        
        quantized_response = Mock()
        # Simulate 80% overlap
        quantized_response.points = [Mock(id=i) for i in range(8)] + [Mock(id=20), Mock(id=21)]
        
        mock_qdrant_client.query_points.side_effect = [baseline_response, quantized_response] * 3
        
        config = BenchmarkConfig(test_queries=["query1"])
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        factors = [2.0, 3.0]
        results = benchmark.measure_accuracy_retention(
            original_collection="original",
            quantized_collection="quantized",
            test_queries=config.test_queries,
            factors=factors
        )
        
        # Check structure
        assert 2.0 in results
        assert 3.0 in results
        
        # Each should have accuracy
        for factor, metrics in results.items():
            assert "avg_accuracy" in metrics
            # Accuracy should be around 0.8 (80% overlap)
            assert 0 <= metrics["avg_accuracy"] <= 1.0
    
    def test_latency_values_reasonable(self, mock_qdrant_client, mock_sentence_transformer):
        """Test that measured latencies are reasonable."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        
        config = BenchmarkConfig(test_queries=["query1", "query2"])
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        metrics = benchmark.measure_search_latency(
            collection_name="test",
            test_queries=config.test_queries,
            label="Test"
        )
        
        # Latencies should be positive and in milliseconds (reasonable range)
        assert metrics["avg"] > 0
        assert metrics["avg"] < 10000  # Less than 10 seconds
        assert metrics["p50"] > 0
        assert metrics["p99"] >= metrics["p50"]  # P99 should be >= P50
    
    def test_custom_test_queries(self, mock_qdrant_client, mock_sentence_transformer):
        """Test using custom test queries."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service)
        
        custom_queries = ["custom1", "custom2", "custom3", "custom4"]
        metrics = benchmark.measure_search_latency(
            collection_name="test",
            test_queries=custom_queries,
            label="Custom"
        )
        
        # Should query 4 times (plus warmup)
        assert mock_qdrant_client.query_points.call_count == 5
    
    def test_search_params_passed_correctly(self, mock_qdrant_client, mock_sentence_transformer):
        """Test that search params are passed to query."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        from qdrant_client.models import SearchParams, QuantizationSearchParams
        
        mock_response = Mock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        
        config = BenchmarkConfig(test_queries=["query1"])
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service, config)
        
        search_params = SearchParams(
            quantization=QuantizationSearchParams(rescore=True, oversampling=3.0)
        )
        
        benchmark.measure_search_latency(
            collection_name="test",
            test_queries=config.test_queries,
            search_params=search_params
        )
        
        # Check search_params was passed
        call_args = mock_qdrant_client.query_points.call_args_list[-1]  # Last call
        assert 'search_params' in call_args[1]
    
    def test_print_metrics(self, mock_qdrant_client, mock_sentence_transformer, capsys):
        """Test printing metrics."""
        from qdrant_quantization_benchmark.embeddings import EmbeddingService
        
        embedding_service = EmbeddingService()
        benchmark = PerformanceBenchmark(mock_qdrant_client, embedding_service)
        
        metrics = {
            "avg": 45.23,
            "p50": 42.15,
            "p90": 52.18,
            "p95": 55.67,
            "p99": 58.92,
            "p99.5": 59.45,
            "p99.9": 59.87
        }
        
        benchmark._print_metrics("Test Label", metrics)
        
        captured = capsys.readouterr()
        assert "Test Label:" in captured.out
        assert "P50:" in captured.out
        assert "42.15ms" in captured.out
        assert "P99.9:" in captured.out