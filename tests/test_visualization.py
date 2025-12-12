"""
Tests for visualization and analysis functionality.
"""

import pytest
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from qdrant_quantization_benchmark.visualization import BenchmarkVisualizer


class TestBenchmarkVisualizer:
    """Tests for BenchmarkVisualizer class."""
    
    def test_plot_quantization_results(self, mock_benchmark_results, tmp_path, mocker):
        """Test generating quantization results plot."""
        baseline, quantization = mock_benchmark_results
        
        # Mock plt.savefig to avoid actually creating file
        mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
        mock_tight_layout = mocker.patch('matplotlib.pyplot.tight_layout')
        
        output_path = str(tmp_path / "test_plot.png")
        
        # Should not raise error
        BenchmarkVisualizer.plot_quantization_results(
            baseline_metrics=baseline,
            quantization_results=quantization,
            output_path=output_path
        )
        
        # Verify savefig was called
        mock_savefig.assert_called_once()
        mock_tight_layout.assert_called_once()
    
    def test_print_analysis_summary(self, mock_benchmark_results, capsys):
        """Test printing analysis summary."""
        baseline, quantization = mock_benchmark_results
        
        BenchmarkVisualizer.print_analysis_summary(baseline, quantization)
        
        captured = capsys.readouterr()
        
        # Check output contains expected sections
        assert "QUANTIZATION PERFORMANCE ANALYSIS" in captured.out
        assert "Baseline Performance:" in captured.out
        assert "Quantization Results:" in captured.out
        assert "SCALAR" in captured.out
        assert "BINARY" in captured.out
        
        # Check metrics are displayed
        assert "P50:" in captured.out
        assert "P95:" in captured.out
        assert "ms" in captured.out
    
    def test_print_analysis_shows_speedup(self, mock_benchmark_results, capsys):
        """Test that analysis summary shows speedup calculations."""
        baseline, quantization = mock_benchmark_results
        
        BenchmarkVisualizer.print_analysis_summary(baseline, quantization)
        
        captured = capsys.readouterr()
        
        # Should show speedup factors
        assert "x)" in captured.out  # Format like "2.5x)"
    
    def test_print_oversampling_analysis(self, capsys):
        """Test printing oversampling factor analysis."""
        latency_results = {
            2.0: {"avg_latency": 28.45, "p95_latency": 34.70},
            3.0: {"avg_latency": 30.12, "p95_latency": 36.50},
            5.0: {"avg_latency": 32.89, "p95_latency": 38.20},
        }
        
        accuracy_results = {
            2.0: {"avg_accuracy": 0.85},
            3.0: {"avg_accuracy": 0.92},
            5.0: {"avg_accuracy": 0.97},
        }
        
        BenchmarkVisualizer.print_oversampling_analysis(
            latency_results,
            accuracy_results
        )
        
        captured = capsys.readouterr()
        
        assert "OVERSAMPLING FACTOR OPTIMIZATION" in captured.out
        assert "2.0x:" in captured.out
        assert "3.0x:" in captured.out
        assert "5.0x:" in captured.out
        assert "avg latency" in captured.out
        assert "avg accuracy retention" in captured.out
    
    def test_plot_with_empty_quantization_results(self, tmp_path, mocker):
        """Test plotting with minimal quantization results."""
        baseline = {
            "avg": 45.23,
            "p50": 42.15,
            "p90": 52.18,
            "p95": 55.67,
            "p99": 58.92,
            "p99.5": 59.45,
            "p99.9": 59.87
        }
        
        # Minimal quantization result
        quantization = {
            "scalar": {
                "no_rescoring": baseline.copy(),
                "with_rescoring": baseline.copy()
            }
        }
        
        mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
        
        output_path = str(tmp_path / "minimal_plot.png")
        
        # Should handle minimal data
        BenchmarkVisualizer.plot_quantization_results(
            baseline_metrics=baseline,
            quantization_results=quantization,
            output_path=output_path
        )
        
        mock_savefig.assert_called_once()
    
    def test_print_analysis_with_multiple_methods(self, capsys):
        """Test analysis with multiple quantization methods."""
        baseline = {"avg": 50.0, "p50": 48.0, "p90": 55.0, "p95": 58.0, "p99": 62.0, "p99.5": 63.0}
        
        quantization = {
            "scalar": {
                "no_rescoring": {"avg": 25.0, "p50": 24.0, "p90": 28.0, "p95": 29.0, "p99": 31.0, "p99.5": 31.5},
                "with_rescoring": {"avg": 30.0, "p50": 29.0, "p90": 33.0, "p95": 34.0, "p99": 36.0, "p99.5": 36.5}
            },
            "binary": {
                "no_rescoring": {"avg": 15.0, "p50": 14.0, "p90": 18.0, "p95": 19.0, "p99": 21.0, "p99.5": 21.5},
                "with_rescoring": {"avg": 19.0, "p50": 18.0, "p90": 22.0, "p95": 23.0, "p99": 25.0, "p99.5": 25.5}
            },
            "binary_2bit": {
                "no_rescoring": {"avg": 18.0, "p50": 17.0, "p90": 21.0, "p95": 22.0, "p99": 24.0, "p99.5": 24.5},
                "with_rescoring": {"avg": 22.0, "p50": 21.0, "p90": 25.0, "p95": 26.0, "p99": 28.0, "p99.5": 28.5}
            }
        }
        
        BenchmarkVisualizer.print_analysis_summary(baseline, quantization)
        
        captured = capsys.readouterr()
        
        assert "SCALAR" in captured.out
        assert "BINARY" in captured.out
        assert "BINARY_2BIT" in captured.out
    
    def test_speedup_calculations_correct(self, capsys):
        """Test that speedup calculations are correct."""
        baseline = {"avg": 100.0, "p50": 100.0, "p90": 100.0, "p95": 100.0, "p99": 100.0, "p99.5": 100.0}
        
        quantization = {
            "scalar": {
                "no_rescoring": {"avg": 50.0, "p50": 50.0, "p90": 50.0, "p95": 50.0, "p99": 50.0, "p99.5": 50.0},
                "with_rescoring": {"avg": 50.0, "p50": 50.0, "p90": 50.0, "p95": 50.0, "p99": 50.0, "p99.5": 50.0}
            }
        }
        
        BenchmarkVisualizer.print_analysis_summary(baseline, quantization)
        
        captured = capsys.readouterr()
        
        # Should show 2.0x speedup (100/50 = 2.0)
        assert "2.0x" in captured.out


class TestVisualizationHelpers:
    """Tests for visualization helper methods."""
    
    def test_plot_percentile_comparison_callable(self, mock_benchmark_results, mocker):
        """Test that _plot_percentile_comparison doesn't raise errors."""
        baseline, quantization = mock_benchmark_results
        
        # Create a mock axes object
        fig, ax = plt.subplots()
        
        # Should not raise error
        BenchmarkVisualizer._plot_percentile_comparison(ax, baseline, quantization)
        
        plt.close(fig)
    
    def test_plot_speedup_comparison_callable(self, mock_benchmark_results):
        """Test that _plot_speedup_comparison doesn't raise errors."""
        baseline, quantization = mock_benchmark_results
        
        fig, ax = plt.subplots()
        
        # Should not raise error
        BenchmarkVisualizer._plot_speedup_comparison(ax, baseline, quantization)
        
        plt.close(fig)
    
    def test_plot_rescoring_impact_callable(self, mock_benchmark_results):
        """Test that _plot_rescoring_impact doesn't raise errors."""
        baseline, quantization = mock_benchmark_results
        
        fig, ax = plt.subplots()
        
        # Should not raise error
        BenchmarkVisualizer._plot_rescoring_impact(ax, baseline, quantization)
        
        plt.close(fig)
    
    def test_plot_p95_table_callable(self, mock_benchmark_results):
        """Test that _plot_p95_table doesn't raise errors."""
        baseline, quantization = mock_benchmark_results
        
        fig, ax = plt.subplots()
        
        # Should not raise error
        BenchmarkVisualizer._plot_p95_table(ax, baseline, quantization)
        
        plt.close(fig)