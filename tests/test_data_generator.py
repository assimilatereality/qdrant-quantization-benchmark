"""
Tests for dataset generation functionality.
"""

import pytest
import json
from pathlib import Path

from qdrant_quantization_benchmark.data_generator import DatasetGenerator


class TestDatasetGenerator:
    """Tests for DatasetGenerator class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        generator = DatasetGenerator(seed=42)
        assert generator is not None
    
    def test_initialization_with_seed(self):
        """Test that same seed produces reproducible core structure."""
        gen1 = DatasetGenerator(seed=42)
        gen2 = DatasetGenerator(seed=42)
        
        dataset1 = gen1.generate(n=10, domain_mix={'tech': 1.0})
        dataset2 = gen2.generate(n=10, domain_mix={'tech': 1.0})

        # With same seed, deterministic parts should match
        for item1, item2 in zip(dataset1, dataset2):
            # Domain should be same
            assert item1['domain'] == item2['domain']
        
            # Metadata language and topic should be deterministic (based on index)
            assert item1['metadata']['language'] == item2['metadata']['language']
            assert item1['metadata']['topic'] == item2['metadata']['topic']
        
            # Title is built from language, topic, edition - should be same
            assert item1['title'] == item2['title']
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different datasets."""
        gen1 = DatasetGenerator(seed=42)
        gen2 = DatasetGenerator(seed=99)
        
        dataset1 = gen1.generate(n=10, domain_mix={'tech': 1.0})
        dataset2 = gen2.generate(n=10, domain_mix={'tech': 1.0})
        
        # At least some descriptions should be different due to random.choice()
        descriptions1 = [item['description'] for item in dataset1]
        descriptions2 = [item['description'] for item in dataset2]
        
        # Count how many are different
        different_count = sum(1 for d1, d2 in zip(descriptions1, descriptions2) if d1 != d2)
        
        # Should have at least SOME differences (not all the same)
        assert different_count > 0, f"Expected some differences, but all {len(dataset1)} items were identical"
    
    def test_generate_default_size(self):
        """Test generating with default parameters."""
        generator = DatasetGenerator()
        dataset = generator.generate(n=100)
        assert len(dataset) == 100
    
    def test_domain_distribution(self):
        """Test that domain mix is respected."""
        generator = DatasetGenerator(seed=42)
        
        # Generate with 50% tech, 50% medical
        dataset = generator.generate(
            n=100,
            domain_mix={'tech': 0.5, 'medical': 0.5}
        )
        
        tech_count = sum(1 for item in dataset if item['domain'] == 'tech')
        medical_count = sum(1 for item in dataset if item['domain'] == 'medical')
        
        # Should be roughly 50/50 (allow ±2 for rounding)
        assert 48 <= tech_count <= 52
        assert 48 <= medical_count <= 52
    
    def test_all_domains_generated(self):
        """Test that all domains can be generated."""
        generator = DatasetGenerator(seed=42)
        
        domains = ['tech', 'medical', 'pharmaceutical', 'health_insurance']
        
        for domain in domains:
            dataset = generator.generate(n=5, domain_mix={domain: 1.0})
            assert all(item['domain'] == domain for item in dataset)
    
    def test_generated_items_have_required_fields(self):
        """Test that generated items have all required fields."""
        generator = DatasetGenerator()
        dataset = generator.generate(n=10)
        
        required_fields = ['id', 'domain', 'title', 'description', 'metadata']
        
        for item in dataset:
            for field in required_fields:
                assert field in item
    
    def test_unique_ids(self):
        """Test that all items have unique IDs."""
        generator = DatasetGenerator()
        dataset = generator.generate(n=100)
        
        ids = [item['id'] for item in dataset]
        assert len(ids) == len(set(ids)), "IDs should be unique"
    
    def test_save_dataset(self, tmp_path):
        """Test saving dataset to file."""
        generator = DatasetGenerator()
        dataset = generator.generate(n=10)
        
        filepath = tmp_path / "test_dataset.json"
        generator.save_dataset(dataset, str(filepath))
        
        assert filepath.exists()
        
        # Verify content
        with open(filepath) as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 10
        assert loaded_data == dataset
    
    def test_load_dataset(self, tmp_path):
        """Test loading dataset from file."""
        # Create test dataset
        generator = DatasetGenerator()
        original_dataset = generator.generate(n=10)
        
        # Save it
        filepath = tmp_path / "test_dataset.json"
        generator.save_dataset(original_dataset, str(filepath))
        
        # Load it back
        loaded_dataset = DatasetGenerator.load_dataset(str(filepath))
        
        assert loaded_dataset == original_dataset
    
    def test_even_distribution_with_none(self):
        """Test even distribution when domain_mix is None."""
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate(n=100, domain_mix=None)
        
        # Count domains
        domain_counts = {}
        for item in dataset:
            domain = item['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Should have 4 domains with roughly equal distribution
        assert len(domain_counts) == 4
        # Each should be around 25 (±3 for rounding)
        for count in domain_counts.values():
            assert 22 <= count <= 28