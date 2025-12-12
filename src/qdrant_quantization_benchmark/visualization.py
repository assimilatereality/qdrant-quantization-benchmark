"""
Visualization and analysis of benchmark results.
"""

from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkVisualizer:
    """Create visualizations from benchmark results."""
    
    @staticmethod
    def plot_quantization_results(
        baseline_metrics: Dict[str, float],
        quantization_results: Dict[str, Dict[str, Dict[str, float]]],
        output_path: str = "quantization_analysis.png"
    ) -> None:
        """
        Create comprehensive visualization of quantization performance.
        
        Args:
            baseline_metrics: Baseline performance metrics
            quantization_results: Results from quantization benchmarks
            output_path: Path to save the output figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quantization Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. PERCENTILE COMPARISON (Top Left)
        BenchmarkVisualizer._plot_percentile_comparison(
            axes[0, 0], baseline_metrics, quantization_results
        )
        
        # 2. SPEEDUP COMPARISON (Top Right)
        BenchmarkVisualizer._plot_speedup_comparison(
            axes[0, 1], baseline_metrics, quantization_results
        )
        
        # 3. WITH vs WITHOUT RESCORING (Bottom Left)
        BenchmarkVisualizer._plot_rescoring_impact(
            axes[1, 0], baseline_metrics, quantization_results
        )
        
        # 4. P95 COMPARISON TABLE (Bottom Right)
        BenchmarkVisualizer._plot_p95_table(
            axes[1, 1], baseline_metrics, quantization_results
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    
    @staticmethod
    def _plot_percentile_comparison(
        ax: plt.Axes,
        baseline_metrics: Dict[str, float],
        quantization_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """Plot percentile comparison across methods."""
        percentiles = ['P50', 'P90', 'P95', 'P99', 'P99.5']
        x = np.arange(len(percentiles))
        width = 0.15
        
        # Baseline
        baseline_values = [
            baseline_metrics['p50'], baseline_metrics['p90'],
            baseline_metrics['p95'], baseline_metrics['p99'],
            baseline_metrics['p99.5']
        ]
        ax.bar(x - 2*width, baseline_values, width, label='Baseline', color='#2E86AB')
        
        # Quantized methods
        colors = {'scalar': '#A23B72', 'binary': '#F18F01', 'binary_2bit': '#C73E1D'}
        for i, (method, results) in enumerate(quantization_results.items()):
            no_rescore = results['no_rescoring']
            values = [
                no_rescore['p50'], no_rescore['p90'], no_rescore['p95'],
                no_rescore['p99'], no_rescore['p99.5']
            ]
            ax.bar(x + (i-1)*width, values, width,
                  label=f'{method.upper()} (No Rescore)',
                  color=colors.get(method, '#999999'), alpha=0.7)
        
        ax.set_xlabel('Percentile', fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title('Latency Distribution Across Percentiles')
        ax.set_xticks(x)
        ax.set_xticklabels(percentiles)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    @staticmethod
    def _plot_speedup_comparison(
        ax: plt.Axes,
        baseline_metrics: Dict[str, float],
        quantization_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """Plot speedup comparison."""
        methods = list(quantization_results.keys())
        x_pos = np.arange(len(methods))
        
        speedup_no_rescore = []
        speedup_with_rescore = []
        
        for method in methods:
            results = quantization_results[method]
            speedup_no = baseline_metrics['avg'] / results['no_rescoring']['avg']
            speedup_with = baseline_metrics['avg'] / results['with_rescoring']['avg']
            speedup_no_rescore.append(speedup_no)
            speedup_with_rescore.append(speedup_with)
        
        bar_width = 0.35
        ax.barh(x_pos - bar_width/2, speedup_no_rescore, bar_width,
               label='Without Rescoring', color='#E63946')
        ax.barh(x_pos + bar_width/2, speedup_with_rescore, bar_width,
               label='With Rescoring', color='#06A77D')
        
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax.set_yticks(x_pos)
        ax.set_yticklabels([m.upper() for m in methods])
        ax.set_xlabel('Speedup Factor (higher is better)', fontweight='bold')
        ax.set_title('Average Latency Speedup vs Baseline')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (no, yes) in enumerate(zip(speedup_no_rescore, speedup_with_rescore)):
            ax.text(no + 0.05, i - bar_width/2, f'{no:.2f}x',
                   va='center', fontsize=9, fontweight='bold')
            ax.text(yes + 0.05, i + bar_width/2, f'{yes:.2f}x',
                   va='center', fontsize=9, fontweight='bold')
    
    @staticmethod
    def _plot_rescoring_impact(
        ax: plt.Axes,
        baseline_metrics: Dict[str, float],
        quantization_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """Plot impact of rescoring on latency."""
        methods = list(quantization_results.keys())
        x_pos = np.arange(len(methods))
        
        no_rescore_avg = [
            quantization_results[m]['no_rescoring']['avg'] for m in methods
        ]
        with_rescore_avg = [
            quantization_results[m]['with_rescoring']['avg'] for m in methods
        ]
        
        bar_width = 0.35
        ax.bar(x_pos - bar_width/2, no_rescore_avg, bar_width,
              label='Without Rescoring', color='#E63946', alpha=0.8)
        ax.bar(x_pos + bar_width/2, with_rescore_avg, bar_width,
              label='With Rescoring', color='#06A77D', alpha=0.8)
        ax.axhline(y=baseline_metrics['avg'], color='#2E86AB', linestyle='--',
                  linewidth=2, label=f'Baseline ({baseline_metrics["avg"]:.1f}ms)')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylabel('Average Latency (ms)', fontweight='bold')
        ax.set_title('Impact of Rescoring on Latency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    @staticmethod
    def _plot_p95_table(
        ax: plt.Axes,
        baseline_metrics: Dict[str, float],
        quantization_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """Create P95 comparison table."""
        ax.axis('off')
        
        # Create table data
        table_data = [['Method', 'Baseline\nP95 (ms)', 'Quantized\nP95 (ms)', 'Speedup']]
        table_data.append(['Baseline', f"{baseline_metrics['p95']:.1f}", '-', '1.0x'])
        
        methods = list(quantization_results.keys())
        for method in methods:
            results = quantization_results[method]
            quant_p95 = results['with_rescoring']['p95']
            speedup = baseline_metrics['p95'] / quant_p95
            table_data.append([
                method.upper(),
                f"{baseline_metrics['p95']:.1f}",
                f"{quant_p95:.1f}",
                f"{speedup:.2f}x"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8E8E8')
        
        ax.set_title('P95 Latency Summary', fontweight='bold', pad=20)
    
    @staticmethod
    def print_analysis_summary(
        baseline_metrics: Dict[str, float],
        quantization_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """
        Print detailed analysis summary.
        
        Args:
            baseline_metrics: Baseline performance metrics
            quantization_results: Results from quantization benchmarks
        """
        print("=" * 60)
        print("QUANTIZATION PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        print(f"\nBaseline Performance:")
        print(f"  Average: {baseline_metrics['avg']:.2f}ms")
        print(f"  P50:     {baseline_metrics['p50']:.2f}ms")
        print(f"  P90:     {baseline_metrics['p90']:.2f}ms")
        print(f"  P95:     {baseline_metrics['p95']:.2f}ms")
        print(f"  P99:     {baseline_metrics['p99']:.2f}ms")
        print(f"  P99.5:   {baseline_metrics['p99.5']:.2f}ms")
        
        print(f"\nQuantization Results:")
        for method, results in quantization_results.items():
            no_rescoring = results['no_rescoring']
            with_rescoring = results['with_rescoring']
            
            speedup_avg_no = baseline_metrics['avg'] / no_rescoring['avg']
            speedup_avg_with = baseline_metrics['avg'] / with_rescoring['avg']
            speedup_p95_no = baseline_metrics['p95'] / no_rescoring['p95']
            speedup_p95_with = baseline_metrics['p95'] / with_rescoring['p95']
            
            print(f"\n{method.upper()}:")
            print(f"  Without rescoring:")
            print(f"    Average: {no_rescoring['avg']:.2f}ms ({speedup_avg_no:.1f}x)")
            print(f"    P95:     {no_rescoring['p95']:.2f}ms ({speedup_p95_no:.1f}x)")
            print(f"  With rescoring:")
            print(f"    Average: {with_rescoring['avg']:.2f}ms ({speedup_avg_with:.1f}x)")
            print(f"    P95:     {with_rescoring['p95']:.2f}ms ({speedup_p95_with:.1f}x)")
    
    @staticmethod
    def print_oversampling_analysis(
        oversampling_results_latency: Dict[float, Dict[str, float]],
        oversampling_results_accuracy: Dict[float, Dict[str, float]]
    ) -> None:
        """
        Print oversampling factor analysis.
        
        Args:
            oversampling_results_latency: Latency results per factor
            oversampling_results_accuracy: Accuracy results per factor
        """
        print("\n" + "=" * 60)
        print("OVERSAMPLING FACTOR OPTIMIZATION")
        print("=" * 60)
        
        for factor in sorted(oversampling_results_latency.keys()):
            latency = oversampling_results_latency[factor]
            accuracy = oversampling_results_accuracy[factor]
            
            print(f"\n  {factor}x:")
            print(f"    {latency['avg_latency']:.2f}ms avg latency, "
                  f"{latency['p95_latency']:.2f}ms P95 latency")
            print(f"    {accuracy['avg_accuracy']:.2f} avg accuracy retention")