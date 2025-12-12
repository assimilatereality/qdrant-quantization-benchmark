"""
Performance benchmarking operations for Qdrant collections.
"""

import time
from typing import List, Dict, Any, Optional, Set
import numpy as np
from qdrant_client import QdrantClient, models

from .config import BenchmarkConfig
from .embeddings import EmbeddingService


class PerformanceBenchmark:
    """Benchmark performance of Qdrant collections."""
    
    def __init__(
        self, 
        client: QdrantClient,
        embedding_service: EmbeddingService,
        config: Optional[BenchmarkConfig] = None
    ):
        """
        Initialize performance benchmark.
        
        Args:
            client: Qdrant client instance
            embedding_service: Embedding service for encoding queries
            config: Benchmark configuration
        """
        self.client = client
        self.embedding_service = embedding_service
        self.config = config or BenchmarkConfig()
    
    def warmup(self, collection_name: str, using: Optional[str] = None) -> None:
        """
        Warm up caches with a throwaway query.
        
        Args:
            collection_name: Name of the collection
            using: Vector name (for named vectors)
        """
        if not self.config.warmup_enabled:
            return
        
        warmup_vector = self.embedding_service.encode_text("warmup query")
        
        query_params = {
            "collection_name": collection_name,
            "query": warmup_vector,
            "limit": self.config.limit
        }
        
        if using:
            query_params["using"] = using
        
        self.client.query_points(**query_params)
    
    def measure_search_latency(
        self,
        collection_name: str,
        test_queries: Optional[List[str]] = None,
        using: Optional[str] = None,
        label: str = "Benchmark",
        search_params: Optional[models.SearchParams] = None
    ) -> Dict[str, float]:
        """
        Measure search performance across multiple queries.
        
        Args:
            collection_name: Name of the collection
            test_queries: List of test query strings
            using: Vector name (for named vectors)
            label: Label for this benchmark run
            search_params: Optional search parameters (for quantization)
            
        Returns:
            Dictionary with latency percentiles
        """
        if test_queries is None:
            test_queries = self.config.test_queries
        
        # Warmup
        self.warmup(collection_name, using)
        
        latencies = []
        
        for query in test_queries:
            query_vector = self.embedding_service.encode_text(query)
            
            start_time = time.time()
            
            query_params = {
                "collection_name": collection_name,
                "query": query_vector,
                "limit": self.config.limit
            }
            
            if using:
                query_params["using"] = using
            
            if search_params:
                query_params["search_params"] = search_params
            
            self.client.query_points(**query_params)
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        metrics = self._calculate_metrics(latencies)
        self._print_metrics(label, metrics)
        
        return metrics
    
    def benchmark_quantization(
        self,
        collection_name: str,
        test_queries: Optional[List[str]] = None,
        method_name: str = "quantized"
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark quantized collection with and without rescoring.
        
        Args:
            collection_name: Name of the quantized collection
            test_queries: List of test query strings
            method_name: Name of quantization method (for labeling)
            
        Returns:
            Dictionary with metrics for both no_rescoring and with_rescoring
        """
        if test_queries is None:
            test_queries = self.config.test_queries
        
        # Test without rescoring
        no_rescoring_metrics = self.measure_search_latency(
            collection_name=collection_name,
            test_queries=test_queries,
            label=f"{method_name} (No Rescoring)"
        )
        
        # Test with rescoring
        rescoring_latencies = []
        
        for query in test_queries:
            query_vector = self.embedding_service.encode_text(query)
            
            start_time = time.time()
            
            self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=self.config.limit,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        rescore=True,
                        oversampling=3.0,
                    )
                ),
            )
            
            latency = (time.time() - start_time) * 1000
            rescoring_latencies.append(latency)
        
        with_rescoring_metrics = self._calculate_metrics(rescoring_latencies)
        self._print_metrics(f"{method_name} (With Rescoring)", with_rescoring_metrics)
        
        return {
            "no_rescoring": no_rescoring_metrics,
            "with_rescoring": with_rescoring_metrics
        }
    
    def tune_oversampling(
        self,
        collection_name: str,
        test_queries: Optional[List[str]] = None,
        factors: Optional[List[float]] = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Test different oversampling factors to find optimal performance.
        
        Args:
            collection_name: Name of the quantized collection
            test_queries: List of test query strings
            factors: List of oversampling factors to test
            
        Returns:
            Dictionary mapping factor to metrics
        """
        if test_queries is None:
            test_queries = self.config.test_queries
        
        if factors is None:
            factors = self.config.oversampling_factors
        
        results = {}
        
        for factor in factors:
            latencies = []
            
            for query in test_queries:
                query_vector = self.embedding_service.encode_text(query)
                
                start_time = time.time()
                
                self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=self.config.limit,
                    search_params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            rescore=True,
                            oversampling=factor,
                        )
                    ),
                )
                
                latencies.append((time.time() - start_time) * 1000)
            
            results[factor] = {
                "avg_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95)
            }
        
        return results
    
    def measure_accuracy_retention(
        self,
        original_collection: str,
        quantized_collection: str,
        test_queries: Optional[List[str]] = None,
        factors: Optional[List[float]] = None,
        using_original: str = "dense"
    ) -> Dict[float, Dict[str, float]]:
        """
        Compare search results between original and quantized collections.
        
        Args:
            original_collection: Name of original collection
            quantized_collection: Name of quantized collection
            test_queries: List of test query strings
            factors: List of oversampling factors to test
            using_original: Vector name for original collection
            
        Returns:
            Dictionary mapping factor to accuracy metrics
        """
        if test_queries is None:
            test_queries = self.config.test_queries
        
        if factors is None:
            factors = self.config.oversampling_factors
        
        results = {}
        
        for factor in factors:
            accuracy_scores = []
            
            for query in test_queries:
                query_vector = self.embedding_service.encode_text(query)
                
                # Get baseline results
                baseline_results = self.client.query_points(
                    collection_name=original_collection,
                    query=query_vector,
                    using=using_original,
                    limit=self.config.limit
                )
                baseline_ids = {point.id for point in baseline_results.points}
                
                # Get quantized results
                quantized_results = self.client.query_points(
                    collection_name=quantized_collection,
                    query=query_vector,
                    limit=self.config.limit,
                    search_params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            rescore=True,
                            oversampling=factor,
                        )
                    ),
                )
                quantized_ids = {point.id for point in quantized_results.points}
                
                # Calculate overlap
                overlap = len(baseline_ids & quantized_ids)
                accuracy = overlap / len(baseline_ids) if baseline_ids else 0
                accuracy_scores.append(accuracy)
            
            results[factor] = {
                "avg_accuracy": np.mean(accuracy_scores)
            }
        
        return results
    
    def _calculate_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """
        Calculate statistical metrics from latency measurements.
        
        Args:
            latencies: List of latency measurements in milliseconds
            
        Returns:
            Dictionary with calculated metrics
        """
        return {
            "avg": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p90": np.percentile(latencies, 90),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "p99.5": np.percentile(latencies, 99.5),
            "p99.9": np.percentile(latencies, 99.9)
        }
    
    def _print_metrics(self, label: str, metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted way.
        
        Args:
            label: Label for this metric set
            metrics: Dictionary of metrics
        """
        print(f"{label}:")
        print(f"  P50:     {metrics['p50']:.2f}ms")
        print(f"  P90:     {metrics['p90']:.2f}ms")
        print(f"  P95:     {metrics['p95']:.2f}ms")
        print(f"  P99:     {metrics['p99']:.2f}ms")
        print(f"  P99.5:   {metrics['p99.5']:.2f}ms")
        print(f"  P99.9:   {metrics['p99.9']:.2f}ms")