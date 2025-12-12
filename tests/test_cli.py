"""
Tests for CLI functionality.
"""

import pytest
from argparse import Namespace
from unittest.mock import Mock, patch, MagicMock
import json

from qdrant_quantization_benchmark.cli import (
    cmd_generate_data,
    cmd_generate_queries,
    cmd_upload,
    cmd_benchmark,
    cmd_create_quantized,
    cmd_visualize,
    add_logging_arguments,
)


class TestCmdGenerateData:
    """Tests for cmd_generate_data command."""
    
    def test_generate_data_basic(self, tmp_path):
        """Test basic dataset generation command."""
        args = Namespace(
            size=10,
            output=str(tmp_path / "test.json"),
            tech=1.0,
            medical=0.0,
            pharma=0.0,
            insurance=0.0,
            seed=42,
            log_level="INFO",
            json_logs=False,
            verbose=False,
            quiet=True  # Quiet to reduce output noise
        )
        
        # Should not raise error
        cmd_generate_data(args)
        
        # Check file was created
        assert (tmp_path / "test.json").exists()
        
        # Verify content
        with open(tmp_path / "test.json", 'r') as f:
            data = json.load(f)
        
        assert len(data) == 10
    
    def test_generate_data_domain_mix(self, tmp_path):
        """Test dataset generation with domain mix."""
        args = Namespace(
            size=20,
            output=str(tmp_path / "mixed.json"),
            tech=0.5,
            medical=0.5,
            pharma=0.0,
            insurance=0.0,
            seed=42,
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        cmd_generate_data(args)
        
        with open(tmp_path / "mixed.json", 'r') as f:
            data = json.load(f)
        
        # Check domains
        domains = [item['domain'] for item in data]
        assert 'tech' in domains
        assert 'medical' in domains


class TestCmdGenerateQueries:
    """Tests for cmd_generate_queries command."""
    
    def test_generate_queries_basic(self, tmp_path):
        """Test basic query generation command."""
        args = Namespace(
            num_queries=5,
            output=str(tmp_path / "queries.json"),
            tech=1.0,
            medical=0.0,
            pharma=0.0,
            insurance=0.0,
            seed=42,
            display=False,
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        cmd_generate_queries(args)
        
        assert (tmp_path / "queries.json").exists()
        
        with open(tmp_path / "queries.json", 'r') as f:
            data = json.load(f)
        
        assert 'queries' in data
        assert len(data['queries']) == 5


class TestCmdUpload:
    """Tests for cmd_upload command."""
    
    @patch('qdrant_quantization_benchmark.cli.QdrantClient')
    def test_upload_command(self, mock_client_class, temp_dataset_file, monkeypatch, mock_sentence_transformer):
        """Test upload command."""
        # Setup environment
        monkeypatch.setenv("QDRANT_URL", "http://test:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        # Mock client
        mock_client = Mock()
        mock_client.collection_exists.return_value = False
        mock_client_class.return_value = mock_client
        
        args = Namespace(
            collection="test_collection",
            dataset=str(temp_dataset_file),
            batch_size=10,
            enable_retry=False,
            recreate=True,
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        # Should not raise error
        cmd_upload(args)
        
        # Verify client was called
        mock_client.create_collection.assert_called()
        mock_client.upsert.assert_called()


class TestCmdCreateQuantized:
    """Tests for cmd_create_quantized command."""
    
    @patch('qdrant_quantization_benchmark.cli.QdrantClient')
    def test_create_quantized_command(self, mock_client_class, temp_dataset_file, monkeypatch, mock_sentence_transformer):
        """Test creating quantized collections."""
        monkeypatch.setenv("QDRANT_URL", "http://test:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        args = Namespace(
            dataset=str(temp_dataset_file),
            methods=['scalar'],
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        cmd_create_quantized(args)
        
        # Should create collection
        mock_client.create_collection.assert_called()


class TestCmdBenchmark:
    """Tests for cmd_benchmark command."""
    
    @patch('qdrant_quantization_benchmark.cli.QdrantClient')
    def test_benchmark_command(self, mock_client_class, temp_queries_file, tmp_path, monkeypatch, mock_sentence_transformer):
        """Test benchmark command."""
        monkeypatch.setenv("QDRANT_URL", "http://test:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        output_file = tmp_path / "results.json"
        
        args = Namespace(
            collection="test_collection",
            queries=str(temp_queries_file),
            quantization=None,
            output=str(output_file),
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        cmd_benchmark(args)
        
        # Check results file was created
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        assert 'baseline' in results


class TestCmdVisualize:
    """Tests for cmd_visualize command."""
    
    def test_visualize_command(self, tmp_path, mock_benchmark_results, mocker):
        """Test visualize command."""
        baseline, quantization = mock_benchmark_results
        
        # Create results file
        results_file = tmp_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "baseline": baseline,
                "quantization": quantization
            }, f)
        
        output_file = tmp_path / "plot.png"
        
        # Mock savefig
        mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
        
        args = Namespace(
            results=str(results_file),
            output=str(output_file),
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        cmd_visualize(args)
        
        # Should call savefig
        mock_savefig.assert_called_once()
    
    def test_visualize_missing_data(self, tmp_path, capsys):
        """Test visualize with incomplete results."""
        # Create results file with missing data
        results_file = tmp_path / "incomplete.json"
        with open(results_file, 'w') as f:
            json.dump({"baseline": {}}, f)  # Missing quantization
        
        args = Namespace(
            results=str(results_file),
            output=str(tmp_path / "plot.png"),
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        cmd_visualize(args)
        
        captured = capsys.readouterr()
        # Should report error
        assert "invalid_results_file" in captured.out or "error" in captured.out.lower()


class TestAddLoggingArguments:
    """Tests for add_logging_arguments function."""
    
    def test_adds_logging_arguments(self):
        """Test that logging arguments are added to parser."""
        import argparse
        
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        
        # Parse with logging flags
        args = parser.parse_args(['--log-level', 'DEBUG', '--verbose', '--json-logs'])
        
        assert args.log_level == 'DEBUG'
        assert args.verbose is True
        assert args.json_logs is True
    
    def test_default_logging_arguments(self):
        """Test default values for logging arguments."""
        import argparse
        
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        
        args = parser.parse_args([])
        
        assert args.log_level == 'INFO'
        assert args.verbose is False
        assert args.quiet is False
        assert args.json_logs is False


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_full_workflow(self, tmp_path, monkeypatch, mock_sentence_transformer):
        """Test complete workflow from generation to visualization."""
        monkeypatch.setenv("QDRANT_URL", "http://test:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        # 1. Generate dataset
        dataset_file = tmp_path / "dataset.json"
        data_args = Namespace(
            size=10, output=str(dataset_file),
            tech=1.0, medical=0.0, pharma=0.0, insurance=0.0,
            seed=42, log_level="ERROR", json_logs=False, verbose=False, quiet=True
        )
        cmd_generate_data(data_args)
        assert dataset_file.exists()
        
        # 2. Generate queries
        queries_file = tmp_path / "queries.json"
        queries_args = Namespace(
            num_queries=5, output=str(queries_file),
            tech=1.0, medical=0.0, pharma=0.0, insurance=0.0,
            seed=42, display=False, log_level="ERROR", json_logs=False, verbose=False, quiet=True
        )
        cmd_generate_queries(queries_args)
        assert queries_file.exists()
        
        # Rest would require mocking Qdrant client, which we did in individual tests


class TestCLIErrorHandling:
    """Tests for CLI error handling."""
    
    def test_upload_missing_env_vars(self, temp_dataset_file, monkeypatch):
        """Test upload fails gracefully without environment variables."""
        # Remove environment variables
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        
        args = Namespace(
            collection="test",
            dataset=str(temp_dataset_file),
            batch_size=50,
            enable_retry=False,
            recreate=False,
            log_level="ERROR",
            json_logs=False,
            verbose=False,
            quiet=True
        )
        
        # Should raise ValueError about missing config
        with pytest.raises(ValueError):
            cmd_upload(args)