# Qdrant Benchmark Suite

Performance benchmarking suite for Qdrant vector database with comprehensive quantization analysis.

## Features

- **Multi-domain dataset generation**: Tech, medical, pharmaceutical, and health insurance domains
- **Flexible query generation**: Auto-generated and manual query support
- **Batch upload with retry logic**: Robust data upload with configurable retry mechanisms
- **Comprehensive benchmarking**: Measure P50, P90, P95, P99, P99.5, P99.9 latencies
- **Quantization comparison**: Test scalar, binary, and 2-bit binary quantization
- **Performance visualization**: Generate detailed performance analysis charts
- **SRE-focused design**: Built with operational simplicity and monitoring in mind

## Project Structure

```
qdrant-benchmark/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── TESTING_INSTRUCTIONS.md     # Detailed testing guide
├── .env                        # Environment variables (create this)
├── src/
│   └── qdrant_benchmark/
│       ├── __init__.py         # Package initialization
│       ├── config.py           # Configuration management
│       ├── qdrant_manager.py   # Collection operations
│       ├── embeddings.py       # Embedding service
│       ├── uploader.py         # Data upload with retry
│       ├── benchmarking.py     # Performance measurement
│       ├── visualization.py    # Plotting and analysis
│       ├── cli.py              # CLI interface
│       ├── data_generator.py   # Dataset generation
│       └── query_generator.py  # Query generation
├── tests/                      # Tests (to be implemented)
├── data/                       # Generated datasets
└── results/                    # Benchmark results and plots
```

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Qdrant instance (cloud or local)
- `uv` package manager (recommended)

### 2. Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <your-repo-url>
cd qdrant-benchmark

# Create .env file
cat > .env << EOF
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
EOF

# Install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. Basic Workflow

```bash
# Generate dataset
qdrant-benchmark generate-data --size 10000 --output data/dataset.json

# Generate queries
qdrant-benchmark generate-queries --num-queries 20 --output data/queries.json

# Upload to Qdrant
qdrant-benchmark upload --collection my_benchmark --dataset data/dataset.json --recreate

# Create quantized collections
qdrant-benchmark create-quantized --dataset data/dataset.json --methods scalar binary

# Run benchmarks
qdrant-benchmark benchmark \
  --collection my_benchmark \
  --queries data/queries.json \
  --quantization scalar binary \
  --output results/results.json

# Generate visualization
qdrant-benchmark visualize --results results/results.json --output results/analysis.png
```

## CLI Commands

### generate-data

Generate synthetic test dataset across multiple domains.

```bash
qdrant-benchmark generate-data \
  --size 10000 \
  --output data/dataset.json \
  --tech 0.25 \
  --medical 0.25 \
  --pharma 0.25 \
  --insurance 0.25 \
  --seed 42
```

### generate-queries

Generate test queries for benchmarking.

```bash
qdrant-benchmark generate-queries \
  --num-queries 20 \
  --output data/queries.json \
  --tech 0.25 \
  --medical 0.25 \
  --display
```

### upload

Upload dataset to Qdrant collection.

```bash
qdrant-benchmark upload \
  --collection my_collection \
  --dataset data/dataset.json \
  --batch-size 50 \
  --enable-retry \
  --recreate
```

Options:
- `--enable-retry`: Enable retry logic for unstable connections
- `--recreate`: Delete and recreate collection if exists
- `--batch-size`: Number of points per batch (default: 50)

### create-quantized

Create quantized collections from existing dataset.

```bash
qdrant-benchmark create-quantized \
  --dataset data/dataset.json \
  --methods scalar binary binary_2bit
```

### benchmark

Run performance benchmarks.

```bash
qdrant-benchmark benchmark \
  --collection my_collection \
  --queries data/queries.json \
  --quantization scalar binary binary_2bit \
  --output results/results.json
```

### visualize

Generate performance visualization.

```bash
qdrant-benchmark visualize \
  --results results/results.json \
  --output results/analysis.png
```

## Configuration

All configuration is managed through:
1. **Environment variables** (`.env` file): Qdrant connection credentials
2. **CLI arguments**: Runtime parameters
3. **Config classes** (`config.py`): Default values and structures

### Key Configuration Options

- **Embedding model**: `all-MiniLM-L6-v2` (384-dimensional vectors)
- **Batch size**: 50 points per upload batch
- **Retry logic**: Optional, with exponential backoff
- **Benchmark queries**: Default set of 5 queries, customizable
- **Oversampling factors**: [2.0, 3.0, 5.0, 8.0, 10.0] for quantization tuning

## Performance Expectations

With a 10,000 item dataset on Qdrant Cloud:

| Method | Expected Speedup | Memory Compression |
|--------|-----------------|-------------------|
| Scalar | 2x | 4x |
| Binary | 40x | 32x |
| Binary 2-bit | 20x | 16x |

*Note: Actual results vary based on instance specifications and network conditions.*

## Development

### Code Style

- **Black**: Code formatting
- **Ruff**: Linting
- **mypy**: Type checking

Run checks:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Testing

Comprehensive testing instructions available in `TESTING_INSTRUCTIONS.md`.

Run tests (after pytest implementation):
```bash
pytest tests/ --cov=src/qdrant_benchmark --cov-report=html
```

Target: 80% code coverage

## Architecture Principles

Following SRE best practices:

1. **Start Simple**: Core functionality first, add complexity as needed
2. **Operational Excellence**: Designed for monitoring and observability
3. **Fail Gracefully**: Robust error handling and retry logic
4. **Modular Design**: Each component has a single responsibility
5. **Configuration Management**: All parameters externalized and documented

## Monitoring Integration

Designed for integration with:
- **CloudWatch**: Metrics and logs
- **X-Ray**: Distributed tracing
- **Prometheus**: Time-series metrics
- **Dynatrace**: Full-stack observability

(Logging and monitoring to be implemented in future steps)

## Contributing

1. Follow existing code structure and style
2. Add type hints to all functions
3. Write docstrings for all public methods
4. Update tests to maintain 80% coverage
5. Run linters before committing

## Roadmap

- [x] Core benchmarking functionality
- [x] Multi-domain dataset generation
- [x] Quantization comparison
- [x] Performance visualization
- [ ] Structured logging (Step 3)
- [ ] Enhanced CLI with logging (Step 4)
- [ ] Pytest implementation with 80% coverage (Step 5)
- [ ] CloudWatch integration
- [ ] X-Ray tracing
- [ ] Prometheus metrics export

## License

MIT License - See LICENSE file for details

## Author

Rodney D. Tigges - Senior Site Reliability Engineer

## Acknowledgments

- Built with Qdrant vector database
- Uses Sentence Transformers for embeddings
- Inspired by SRE principles and operational excellence