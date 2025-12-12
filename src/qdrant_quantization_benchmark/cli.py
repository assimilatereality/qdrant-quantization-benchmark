"""
Command-line interface for Qdrant quantization benchmark suite.
"""

import argparse
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from .config import BenchmarkSuiteConfig, LoggingConfig
from .logging import setup_logging, Timer
from .qdrant_manager import QdrantCollectionManager
from .embeddings import EmbeddingService
from .uploader import DataUploader
from .benchmarking import PerformanceBenchmark
from .visualization import BenchmarkVisualizer
from .data_generator import DatasetGenerator
from .query_generator import QueryGenerator


def cmd_generate_data(args: argparse.Namespace) -> None:
    """Generate test dataset."""
    log = setup_logging(
        level=args.log_level,
        json_output=args.json_logs,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    with Timer(log, "generate_dataset", size=args.size):
        log.info("generating_dataset", size=args.size, output=args.output)
        
        domain_mix = {
            'tech': args.tech,
            'medical': args.medical,
            'pharmaceutical': args.pharma,
            'health_insurance': args.insurance
        }
        
        generator = DatasetGenerator(seed=args.seed)
        dataset = generator.generate(n=args.size, domain_mix=domain_mix)
        generator.save_dataset(dataset, args.output)
        
        log.info("dataset_generated", 
                items=len(dataset),
                output=args.output,
                domain_mix=domain_mix)


def cmd_generate_queries(args: argparse.Namespace) -> None:
    """Generate test queries."""
    log = setup_logging(
        level=args.log_level,
        json_output=args.json_logs,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    with Timer(log, "generate_queries", num_queries=args.num_queries):
        log.info("generating_queries", num_queries=args.num_queries)
        
        domain_mix = {
            'tech': args.tech,
            'medical': args.medical,
            'pharmaceutical': args.pharma,
            'health_insurance': args.insurance
        }
        
        generator = QueryGenerator(seed=args.seed)
        auto_queries = generator.generate_auto_queries(n=args.num_queries, domain_mix=domain_mix)
        generator.add_manual_queries(auto_queries)
        
        metadata = {
            'auto_generated': args.num_queries,
            'domain_mix': domain_mix,
            'seed': args.seed
        }
        
        generator.save_queries(args.output, metadata=metadata)
        
        if args.display:
            generator.display_queries()
        
        log.info("queries_generated",
                count=len(generator.queries),
                output=args.output)


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload dataset to Qdrant collection."""
    log = setup_logging(
        level=args.log_level,
        json_output=args.json_logs,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    with Timer(log, "upload_dataset", collection=args.collection):
        # Load configuration
        logging_config = LoggingConfig(
            level=args.log_level,
            json_output=args.json_logs,
            verbose=args.verbose,
            quiet=args.quiet
        )
        config = BenchmarkSuiteConfig.from_env(logging_config)
        config.upload.batch_size = args.batch_size
        config.upload.enable_retry = args.enable_retry
        
        log.info("upload_started",
                collection=args.collection,
                dataset=args.dataset,
                batch_size=args.batch_size,
                retry_enabled=args.enable_retry)
        
        # Initialize clients
        client = QdrantClient(
            url=config.connection.url,
            api_key=config.connection.api_key,
            timeout=config.connection.timeout
        )
        
        # Load dataset
        log.info("loading_dataset", path=args.dataset)
        dataset = DatasetGenerator.load_dataset(args.dataset)
        log.info("dataset_loaded", items=len(dataset))
        
        # Initialize services
        embedding_service = EmbeddingService(config.embedding)
        collection_manager = QdrantCollectionManager(client, config.collection, config.embedding)
        uploader = DataUploader(client, config.upload)
        
        # Create collection
        if args.recreate or not collection_manager.collection_exists(args.collection):
            log.info("creating_collection", collection=args.collection)
            collection_manager.recreate_collection(args.collection, collection_type="hybrid")
        
        # Generate embeddings
        log.info("generating_embeddings", count=len(dataset))
        embeddings = embedding_service.encode_dataset(dataset)
        
        # Upload
        log.info("uploading_data", collection=args.collection, points=len(dataset))
        uploader.upload_batch(
            collection_name=args.collection,
            dataset=dataset,
            embeddings=embeddings,
            named_vector=True,
            vector_name="dense"
        )
        
        log.info("upload_completed",
                collection=args.collection,
                points_uploaded=len(dataset))


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run performance benchmarks."""
    log = setup_logging(
        level=args.log_level,
        json_output=args.json_logs,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    with Timer(log, "benchmark_suite", collection=args.collection):
        # Load configuration
        logging_config = LoggingConfig(
            level=args.log_level,
            json_output=args.json_logs,
            verbose=args.verbose,
            quiet=args.quiet
        )
        config = BenchmarkSuiteConfig.from_env(logging_config)
        
        log.info("benchmark_started", collection=args.collection)
        
        # Initialize clients
        client = QdrantClient(
            url=config.connection.url,
            api_key=config.connection.api_key,
            timeout=config.connection.timeout
        )
        
        # Load queries
        if args.queries:
            log.info("loading_queries", path=args.queries)
            query_gen = QueryGenerator()
            test_queries = query_gen.load_queries(args.queries)
            log.info("queries_loaded", count=len(test_queries))
        else:
            test_queries = config.benchmark.test_queries
            log.info("using_default_queries", count=len(test_queries))
        
        # Initialize services
        embedding_service = EmbeddingService(config.embedding)
        benchmark = PerformanceBenchmark(client, embedding_service, config.benchmark)
        
        # Run baseline benchmark
        log.info("benchmarking_baseline", collection=args.collection)
        baseline_metrics = benchmark.measure_search_latency(
            collection_name=args.collection,
            test_queries=test_queries,
            using="dense",
            label="Baseline (No Quantization)"
        )
        log.info("baseline_completed", metrics=baseline_metrics)
        
        # Run quantization benchmarks if requested
        quantization_results = {}
        if args.quantization:
            for method in args.quantization:
                quantized_collection = f"quantized_{method}"
                log.info("benchmarking_quantization", method=method, collection=quantized_collection)
                
                quantization_results[method] = benchmark.benchmark_quantization(
                    collection_name=quantized_collection,
                    test_queries=test_queries,
                    method_name=method
                )
                log.info("quantization_completed", method=method)
        
        # Save results
        if args.output:
            results = {
                "baseline": baseline_metrics,
                "quantization": quantization_results
            }
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            log.info("results_saved", path=args.output)
        
        # Print analysis
        if quantization_results:
            BenchmarkVisualizer.print_analysis_summary(baseline_metrics, quantization_results)


def cmd_create_quantized(args: argparse.Namespace) -> None:
    """Create quantized collections from existing collection."""
    log = setup_logging(
        level=args.log_level,
        json_output=args.json_logs,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    with Timer(log, "create_quantized_collections", methods=args.methods):
        # Load configuration
        logging_config = LoggingConfig(
            level=args.log_level,
            json_output=args.json_logs,
            verbose=args.verbose,
            quiet=args.quiet
        )
        config = BenchmarkSuiteConfig.from_env(logging_config)
        
        log.info("creating_quantized_collections", methods=args.methods)
        
        # Initialize clients
        client = QdrantClient(
            url=config.connection.url,
            api_key=config.connection.api_key,
            timeout=config.connection.timeout
        )
        
        # Load dataset
        log.info("loading_dataset", path=args.dataset)
        dataset = DatasetGenerator.load_dataset(args.dataset)
        log.info("dataset_loaded", items=len(dataset))
        
        # Initialize services
        embedding_service = EmbeddingService(config.embedding)
        collection_manager = QdrantCollectionManager(client, config.collection, config.embedding)
        uploader = DataUploader(client, config.upload)
        
        # Generate embeddings once
        log.info("generating_embeddings", count=len(dataset))
        embeddings = embedding_service.encode_dataset(dataset)
        
        # Get quantization configurations
        quant_configs = config.quantization.get_all_configs()
        
        # Create each quantized collection
        for method_name in args.methods:
            if method_name not in quant_configs:
                log.error("unknown_quantization_method", method=method_name)
                continue
            
            collection_name = f"quantized_{method_name}"
            log.info("creating_quantized_collection", method=method_name, collection=collection_name)
            
            # Create collection
            collection_manager.recreate_collection(
                collection_name=collection_name,
                collection_type="quantized",
                quantization_config=quant_configs[method_name]["config"]
            )
            
            # Upload data (unnamed vectors for quantized collections)
            uploader.upload_batch(
                collection_name=collection_name,
                dataset=dataset,
                embeddings=embeddings,
                named_vector=False
            )
            
            log.info("quantized_collection_created", method=method_name, collection=collection_name)


def cmd_visualize(args: argparse.Namespace) -> None:
    """Generate visualization from benchmark results."""
    log = setup_logging(
        level=args.log_level,
        json_output=args.json_logs,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    with Timer(log, "generate_visualization", results=args.results):
        # Load results
        log.info("loading_results", path=args.results)
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        baseline_metrics = results.get("baseline", {})
        quantization_results = results.get("quantization", {})
        
        if not baseline_metrics or not quantization_results:
            log.error("invalid_results_file", 
                     message="Results file must contain both 'baseline' and 'quantization' data")
            return
        
        log.info("generating_visualization", output=args.output)
        
        # Generate visualization
        BenchmarkVisualizer.plot_quantization_results(
            baseline_metrics=baseline_metrics,
            quantization_results=quantization_results,
            output_path=args.output
        )
        
        log.info("visualization_saved", path=args.output)


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add logging-related arguments to parser."""
    log_group = parser.add_argument_group('logging options')
    log_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    log_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (sets log level to DEBUG)'
    )
    log_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress most output (sets log level to ERROR)'
    )
    log_group.add_argument(
        '--json-logs',
        action='store_true',
        help='Output logs in JSON format (for CloudWatch/monitoring)'
    )


def main() -> None:
    """Main CLI entry point."""
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description='Qdrant Quantization Benchmark Suite - Vector Database Performance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset
  qdrant-qbench generate-data --size 10000 --output data/dataset.json
  
  # Generate queries
  qdrant-qbench generate-queries --num-queries 20 --output data/queries.json
  
  # Upload to Qdrant
  qdrant-qbench upload --collection test --dataset data/dataset.json
  
  # Create quantized collections
  qdrant-qbench create-quantized --dataset data/dataset.json --methods scalar binary
  
  # Run benchmarks
  qdrant-qbench benchmark --collection test --queries data/queries.json --quantization scalar binary
  
  # Generate visualization
  qdrant-qbench visualize --results results.json --output report.png

Logging Options:
  -v, --verbose          Enable verbose output (DEBUG level)
  -q, --quiet           Suppress most output (ERROR level only)
  --log-level LEVEL     Set specific log level
  --json-logs           Output logs as JSON for monitoring systems

Note: You can use either 'qdrant-qbench' (short) or 'qdrant-quantization-benchmark' (full).
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate data command
    gen_data = subparsers.add_parser('generate-data', help='Generate test dataset')
    gen_data.add_argument('-n','--size', type=int, default=10000, help='Dataset size')
    gen_data.add_argument('-o','--output', default='data/dataset.json', help='Output file path')
    gen_data.add_argument('--tech', type=float, default=0.25, help='Tech domain proportion')
    gen_data.add_argument('--medical', type=float, default=0.25, help='Medical domain proportion')
    gen_data.add_argument('--pharma', type=float, default=0.25, help='Pharmaceutical proportion')
    gen_data.add_argument('--insurance', type=float, default=0.25, help='Health insurance proportion')
    gen_data.add_argument('--seed', type=int, default=42, help='Random seed')
    add_logging_arguments(gen_data)
    
    # Generate queries command
    gen_queries = subparsers.add_parser('generate-queries', help='Generate test queries')
    gen_queries.add_argument('-n','--num-queries', type=int, default=20, help='Number of queries')
    gen_queries.add_argument('-o', '--output', default='data/queries.json', help='Output file path')
    gen_queries.add_argument('--tech', type=float, default=0.25, help='Tech query proportion')
    gen_queries.add_argument('--medical', type=float, default=0.25, help='Medical query proportion')
    gen_queries.add_argument('--pharma', type=float, default=0.25, help='Pharmaceutical proportion')
    gen_queries.add_argument('--insurance', type=float, default=0.25, help='Insurance proportion')
    gen_queries.add_argument('--seed', type=int, default=42, help='Random seed')
    gen_queries.add_argument('--display', action='store_true', help='Display generated queries')
    add_logging_arguments(gen_queries)
    
    # Upload command
    upload = subparsers.add_parser('upload', help='Upload dataset to Qdrant')
    upload.add_argument('-c', '--collection', required=True, help='Collection name')
    upload.add_argument('-d', '--dataset', required=True, help='Dataset file path')
    upload.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size for upload')
    upload.add_argument('--enable-retry', action='store_true', help='Enable retry on timeout')
    upload.add_argument('--recreate', action='store_true', help='Recreate collection if exists')
    add_logging_arguments(upload)
    
    # Create quantized collections command
    create_quant = subparsers.add_parser('create-quantized', help='Create quantized collections')
    create_quant.add_argument('-d', '--dataset', required=True, help='Dataset file path')
    create_quant.add_argument('-m', '--methods', nargs='+', 
                             choices=['scalar', 'binary', 'binary_2bit'],
                             default=['scalar', 'binary', 'binary_2bit'],
                             help='Quantization methods to use')
    add_logging_arguments(create_quant)
    
    # Benchmark command
    bench = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench.add_argument('-c', '--collection', required=True, help='Collection name')
    bench.add_argument('-q', '--queries', help='Queries file path (optional)')
    bench.add_argument('--quantization', nargs='+',
                      choices=['scalar', 'binary', 'binary_2bit'],
                      help='Quantization methods to benchmark')
    bench.add_argument('-o', '--output', help='Output file for results (JSON)')
    add_logging_arguments(bench)
    
    # Visualize command
    viz = subparsers.add_parser('visualize', help='Generate visualization')
    viz.add_argument('-r', '--results', required=True, help='Results file (JSON)')
    viz.add_argument('-o', '--output', default='analysis.png', help='Output image path')
    add_logging_arguments(viz)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'generate-data':
            cmd_generate_data(args)
        elif args.command == 'generate-queries':
            cmd_generate_queries(args)
        elif args.command == 'upload':
            cmd_upload(args)
        elif args.command == 'create-quantized':
            cmd_create_quantized(args)
        elif args.command == 'benchmark':
            cmd_benchmark(args)
        elif args.command == 'visualize':
            cmd_visualize(args)
    except Exception as e:
        # Use basic logging if structlog not yet initialized
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()