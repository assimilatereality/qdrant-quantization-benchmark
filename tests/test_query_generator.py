"""
Tests for query generation functionality.
"""

import pytest
import json

from qdrant_quantization_benchmark.query_generator import QueryGenerator


class TestQueryGenerator:
    """Tests for QueryGenerator class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        generator = QueryGenerator(seed=42)
        assert generator is not None
        assert generator.queries == []
    
    def test_initialization_with_seed(self):
        """Test that same seed produces reproducible queries."""
        # Generate twice with same seed to verify reproducibility
        gen1 = QueryGenerator(seed=42)
        queries1 = gen1.generate_auto_queries(n=5, domain_mix={'tech': 1.0})
    
        gen2 = QueryGenerator(seed=42)
        queries2 = gen2.generate_auto_queries(n=5, domain_mix={'tech': 1.0})
    
        # Same seed should produce identical queries
        assert queries1 == queries2
    
    def test_different_seeds_produce_different_queries(self):
        """Test that different seeds produce different queries."""
        gen1 = QueryGenerator(seed=42)
        gen2 = QueryGenerator(seed=99)
        
        queries1 = gen1.generate_auto_queries(n=10, domain_mix={'tech': 1.0})
        queries2 = gen2.generate_auto_queries(n=10, domain_mix={'tech': 1.0})
        
        # Should have at least SOME different queries
        different_count = sum(1 for q1, q2 in zip(queries1, queries2) if q1 != q2)
        assert different_count > 0
    
    def test_generate_auto_queries(self):
        """Test auto-generation of queries."""
        generator = QueryGenerator(seed=42)
        queries = generator.generate_auto_queries(n=20)
        
        assert len(queries) == 20
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q) > 0 for q in queries)
    
    def test_domain_mix_distribution(self):
        """Test that domain mix is respected."""
        generator = QueryGenerator(seed=42)
        
        # Generate with specific domain mix
        queries = generator.generate_auto_queries(
            n=20,
            domain_mix={'tech': 0.5, 'medical': 0.5}
        )
        
        assert len(queries) == 20
        
        # Count queries by keywords (approximate)
        tech_keywords = ['python', 'javascript', 'programming', 'code']
        medical_keywords = ['cardiology', 'surgery', 'clinical', 'patient']
        
        tech_count = sum(1 for q in queries 
                        if any(kw in q.lower() for kw in tech_keywords))
        medical_count = sum(1 for q in queries 
                           if any(kw in q.lower() for kw in medical_keywords))
        
        # Should have both types present
        assert tech_count > 0
        assert medical_count > 0
    
    def test_add_manual_queries(self):
        """Test adding manual queries."""
        generator = QueryGenerator()
        manual_queries = ["query 1", "query 2", "query 3"]
        
        generator.add_manual_queries(manual_queries)
        
        assert len(generator.queries) == 3
        assert generator.queries == manual_queries
    
    def test_add_manual_query(self):
        """Test adding single manual query."""
        generator = QueryGenerator()
        generator.add_manual_query("test query")
        
        assert len(generator.queries) == 1
        assert generator.queries[0] == "test query"
    
    def test_remove_query(self):
        """Test removing a query."""
        generator = QueryGenerator()
        generator.add_manual_queries(["query 1", "query 2", "query 3"])
        
        generator.remove_query("query 2")
        
        assert len(generator.queries) == 2
        assert "query 2" not in generator.queries
    
    def test_remove_nonexistent_query(self):
        """Test removing query that doesn't exist."""
        generator = QueryGenerator()
        generator.add_manual_query("query 1")
        
        # Should not raise error
        generator.remove_query("nonexistent")
        
        assert len(generator.queries) == 1
    
    def test_get_queries(self):
        """Test retrieving all queries."""
        generator = QueryGenerator()
        queries = ["q1", "q2", "q3"]
        generator.add_manual_queries(queries)
        
        retrieved = generator.get_queries()
        assert retrieved == queries
    
    def test_clear_queries(self):
        """Test clearing all queries."""
        generator = QueryGenerator()
        generator.add_manual_queries(["q1", "q2", "q3"])
        
        generator.clear_queries()
        
        assert len(generator.queries) == 0
    
    def test_save_queries(self, tmp_path):
        """Test saving queries to file."""
        generator = QueryGenerator()
        queries = ["query 1", "query 2", "query 3"]
        generator.add_manual_queries(queries)
        
        filepath = tmp_path / "test_queries.json"
        generator.save_queries(str(filepath))
        
        assert filepath.exists()
        
        # Verify content
        with open(filepath) as f:
            data = json.load(f)
        
        assert data['queries'] == queries
        assert data['count'] == 3
    
    def test_load_queries(self, tmp_path):
        """Test loading queries from file."""
        # Create test file
        generator = QueryGenerator()
        original_queries = ["q1", "q2", "q3"]
        generator.add_manual_queries(original_queries)
        
        filepath = tmp_path / "test_queries.json"
        generator.save_queries(str(filepath))
        
        # Load with new generator
        new_generator = QueryGenerator()
        loaded_queries = new_generator.load_queries(str(filepath))
        
        assert loaded_queries == original_queries
        assert new_generator.queries == original_queries
    
    def test_get_domain_distribution(self):
        """Test domain distribution analysis."""
        generator = QueryGenerator()
        generator.add_manual_queries([
            "python machine learning",
            "cardiology treatment",
            "antibiotic prescription",
            "health insurance plan"
        ])
        
        distribution = generator.get_domain_distribution()
        
        assert 'tech' in distribution
        assert 'medical' in distribution
        assert 'pharmaceutical' in distribution
        assert 'health_insurance' in distribution