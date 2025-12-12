# Qdrant Quantization Benchmark

A comprehensive benchmarking suite for evaluating and optimizing vector quantization methods in Qdrant vector databases. This tool helps you make data-driven decisions about quantization strategies by measuring performance, accuracy retention, and resource usage across different quantization techniques.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Purpose & Benefits

### Why Use This Tool?

**Vector quantization** can dramatically reduce memory usage and improve query speeds in vector databases, but choosing the right quantization method requires careful benchmarking. This tool provides:

- **Performance Analysis**: Measure actual latency improvements (P50, P95, P99) across quantization methods
- **Accuracy Testing**: Validate that quantization doesn't degrade search quality below acceptable thresholds
- **Resource Optimization**: Compare memory usage and compute requirements
- **Production Insights**: Test with realistic multi-domain datasets (tech, medical, pharmaceutical, insurance)
- **Visual Reports**: Generate publication-ready performance visualizations

### Quantization Methods Tested

1. **Scalar Quantization** - Convert float32 to int8 (4x compression, ~2x speedup)
2. **Binary Quantization** - Convert to binary vectors (32x compression, ~40x speedup)
3. **2-Bit Binary Quantization** - Balanced approach (16x compression, ~20x speedup)
4. **Product Quantization** - Advanced compression with configurable parameters

### Real-World Use Cases

- **AI/ML Engineers**: Optimize RAG (Retrieval-Augmented Generation) pipelines
- **SRE/DevOps**: Right-size infrastructure for vector search workloads
- **Data Scientists**: Validate search quality after quantization
- **Cost Optimization**: Reduce cloud costs by 4-32x through intelligent compression

## 🚀 Quick Start
```bash
# Install
pip install -e .

# Generate test data
qdrant-benchmark generate-data -n 10000 --output data/dataset.json

# Upload to Qdrant
qdrant-benchmark upload data/dataset.json --collection benchmark_test

# Run benchmarks
qdrant-benchmark benchmark benchmark_test --output results/

# Visualize results
qdrant-benchmark visualize results/ --output benchmark_report.png
```

## 📋 Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Performance Results](#performance-results)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Qdrant Cloud account or self-hosted Qdrant instance
- 4GB+ RAM recommended for testing

### Standard Installation
```bash
# Clone the repository
git clone https://github.com/assimilatereality/qdrant-quantization-benchmark.git
cd qdrant-quantization-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Development Installation
```bash
# Install with development dependencies (testing, linting)
pip install -e ".[dev]"
```

### Verify Installation
```bash
# Check installation
qdrant-benchmark --version

# Run health check
qdrant-benchmark --help
```

## 📦 Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qdrant-client` | >=1.7.0 | Qdrant vector database client |
| `sentence-transformers` | >=2.2.0 | Text embedding generation |
| `numpy` | >=1.24.0 | Numerical operations |
| `matplotlib` | >=3.7.0 | Performance visualization |
| `structlog` | >=23.1.0 | Structured logging |
| `python-dotenv` | >=1.0.0 | Environment configuration |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=7.4.0 | Testing framework |
| `pytest-cov` | >=4.1.0 | Coverage reporting |
| `pytest-mock` | >=3.11.0 | Mocking utilities |
| `black` | >=23.7.0 | Code formatting |
| `ruff` | >=0.0.282 | Linting |
| `mypy` | >=1.4.0 | Type checking |

### Installing Dependencies Manually
```bash
# Core dependencies only
pip install qdrant-client sentence-transformers numpy matplotlib structlog python-dotenv

# Development dependencies
pip install pytest pytest-cov pytest-mock black ruff mypy
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:
```bash
# Required
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key-here

# Optional
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Default embedding model
VECTOR_SIZE=384                    # Default vector dimension
BATCH_SIZE=50                      # Upload batch size
LOG_LEVEL=INFO                     # Logging level
```

### Configuration File

Alternatively, use a Python configuration:
```python
from qdrant_quantization_benchmark.config import BenchmarkSuiteConfig, LoggingConfig

# Custom configuration
config = BenchmarkSuiteConfig.from_env(
    logging_config=LoggingConfig(level="DEBUG", verbose=True)
)
```

### Qdrant Setup

**Option 1: Qdrant Cloud** (Recommended for testing)
```bash
# Sign up at https://cloud.qdrant.io
# Create a cluster and copy credentials to .env
```

**Option 2: Docker (Local testing)**
```bash
docker run -p 6333:6333 qdrant/qdrant
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=your-local-key
```

**Option 3: Self-hosted**
```bash
# Follow: https://qdrant.tech/documentation/guides/installation/
```

## 📖 Usage

### 1. Generate Test Data

Create synthetic datasets across multiple domains:
```bash
# Generate 10,000 balanced items
qdrant-benchmark generate-data -n 10000 --output data/dataset.json

# Tech-focused dataset (70% tech, 30% other)
qdrant-benchmark generate-data -n 10000 \
  --tech 0.7 --medical 0.1 --pharma 0.1 --insurance 0.1 \
  --output data/tech_dataset.json

# Large dataset for production testing
qdrant-benchmark generate-data -n 100000 --output data/large_dataset.json
```

**Available Domains:**
- `tech` - Programming books, development topics
- `medical` - Clinical guides, medical textbooks
- `pharmaceutical` - Medications, drug information
- `health_insurance` - Insurance plans, coverage options

### 2. Generate Test Queries

Create queries for benchmarking:
```bash
# Auto-generate 20 queries
qdrant-benchmark generate-queries -n 20 --output data/queries.json

# Domain-specific queries
qdrant-benchmark generate-queries -n 50 \
  --tech 0.6 --medical 0.4 \
  --output data/medical_queries.json

# Add custom manual queries
qdrant-benchmark generate-queries \
  --manual "python machine learning best practices" \
  --manual "cardiology treatment protocols" \
  --output data/custom_queries.json
```

### 3. Upload Data to Qdrant
```bash
# Upload with default settings
qdrant-benchmark upload data/dataset.json --collection my_collection

# Custom batch size and retry settings
qdrant-benchmark upload data/dataset.json \
  --collection my_collection \
  --batch-size 100 \
  --enable-retry \
  --max-retries 5

# Upload to multiple collections
qdrant-benchmark upload data/dataset.json --collection baseline
qdrant-benchmark upload data/dataset.json --collection test_v2
```

### 4. Run Benchmarks
```bash
# Basic benchmark
qdrant-benchmark benchmark my_collection --output results/

# Test specific quantization methods
qdrant-benchmark benchmark my_collection \
  --methods scalar,binary \
  --output results/quick_test/

# Comprehensive benchmark with custom queries
qdrant-benchmark benchmark my_collection \
  --queries data/custom_queries.json \
  --methods scalar,binary,binary_2bit,product \
  --warmup \
  --limit 20 \
  --output results/full_benchmark/

# Test oversampling factors
qdrant-benchmark benchmark my_collection \
  --tune-oversampling 2,3,5,8,10 \
  --output results/oversampling_test/
```

**Benchmark Options:**
- `--methods`: Quantization methods to test (scalar, binary, binary_2bit, product)
- `--queries`: Custom query file (default: auto-generated)
- `--warmup`: Enable cache warmup before testing
- `--limit`: Number of results per query (default: 10)
- `--tune-oversampling`: Test different oversampling factors

### 5. Visualize Results
```bash
# Generate performance report
qdrant-benchmark visualize results/ --output report.png

# Multiple comparison formats
qdrant-benchmark visualize results/ \
  --output report.png \
  --format png \
  --dpi 300

# Generate summary table
qdrant-benchmark visualize results/ --summary-only
```

### 6. Cleanup
```bash
# Delete test collections
qdrant-benchmark delete-collection my_collection

# Delete multiple collections
qdrant-benchmark delete-collection baseline test_v2 experimental
```

## 🧪 Testing

This project includes a comprehensive test suite with **90% code coverage**.

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/qdrant_quantization_benchmark --cov-report=html

# Run specific test file
pytest tests/test_benchmarking.py -v

# Run tests matching pattern
pytest -k "test_quantization" -v

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Structure
```
tests/
├── conftest.py                 # Shared fixtures
├── test_config.py             # Configuration tests
├── test_data_generator.py     # Dataset generation tests
├── test_query_generator.py    # Query generation tests
├── test_embeddings.py         # Embedding service tests
├── test_logging.py            # Logging utilities tests
├── test_qdrant_manager.py     # Collection management tests
├── test_uploader.py           # Upload operations tests
├── test_benchmarking.py       # Performance measurement tests
├── test_visualization.py      # Visualization tests
└── test_cli.py                # CLI integration tests
```

### Test Coverage by Module

| Module | Coverage | Notes |
|--------|----------|-------|
| config.py | 100% | Configuration validation |
| data_generator.py | 100% | Dataset generation |
| query_generator.py | 99% | Query generation |
| embeddings.py | 100% | Embedding service |
| qdrant_manager.py | 100% | Collection operations |
| uploader.py | 98% | Upload logic |
| visualization.py | 100% | Visualization |
| benchmarking.py | 94% | Performance testing |
| logging.py | 81% | Logging utilities |
| cli.py | 64% | CLI interface |

See [README_TESTING.md](README_TESTING.md) for detailed testing documentation.

## 📁 Project Structure
```
qdrant-quantization-benchmark/
├── src/
│   └── qdrant_quantization_benchmark/
│       ├── __init__.py           # Package initialization
│       ├── config.py             # Configuration dataclasses
│       ├── data_generator.py    # Multi-domain dataset generation
│       ├── query_generator.py   # Test query generation
│       ├── embeddings.py        # Embedding service wrapper
│       ├── qdrant_manager.py    # Qdrant collection management
│       ├── uploader.py          # Batch upload with retry logic
│       ├── benchmarking.py      # Performance measurement
│       ├── visualization.py     # Results visualization
│       ├── logging.py           # Structured logging utilities
│       └── cli.py               # Command-line interface
├── tests/                       # Test suite (90% coverage)
│   ├── conftest.py             # Shared test fixtures
│   ├── test_*.py               # Module-specific tests
│   └── README_TESTING.md       # Testing documentation
├── examples/                    # Usage examples
│   ├── generate_data_and_queries.py
│   └── end_to_end_benchmark.py
├── data/                        # Generated datasets (gitignored)
├── results/                     # Benchmark results (gitignored)
├── htmlcov/                     # Coverage reports (gitignored)
├── pyproject.toml              # Package configuration
├── README.md                    # This file
├── README_TESTING.md           # Testing guide
├── STEP_5_SUMMARY.md           # Development summary
└── .env                         # Environment config (gitignored)
```

## 🏗️ Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface (cli.py)                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Data         │    │ Benchmarking     │    │ Qdrant      │
│ Generation   │    │ Engine           │    │ Manager     │
├──────────────┤    ├──────────────────┤    ├─────────────┤
│ • Datasets   │    │ • Performance    │    │ • Collections│
│ • Queries    │───▶│ • Quantization   │───▶│ • Upload    │
│ • Embeddings │    │ • Visualization  │    │ • Query     │
└──────────────┘    └──────────────────┘    └─────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Structured       │
                    │ Logging          │
                    │ (structlog)      │
                    └──────────────────┘
```

### Key Design Patterns

1. **Dataclass Configuration**: Type-safe, hierarchical configuration management
2. **Dependency Injection**: Mocked external dependencies for testing
3. **Structured Logging**: Consistent, queryable logs with context
4. **Batch Processing**: Efficient data upload with retry logic
5. **Fixture-Based Testing**: Reusable test components

## 📊 Performance Results

### Example Benchmark Results

Based on 10,000 item dataset with all-MiniLM-L6-v2 embeddings:

| Method | P95 Latency | Speedup | Compression | Accuracy Retention |
|--------|-------------|---------|-------------|-------------------|
| **Baseline** | 45ms | 1.0x | 1.0x | 100% |
| **Scalar** | 23ms | 2.0x | 4.0x | 99.5% |
| **Binary** | 1.2ms | 37.5x | 32.0x | 97.8% |
| **Binary 2-bit** | 2.3ms | 19.6x | 16.0x | 98.9% |

*Results vary based on dataset size, vector dimensions, and hardware.*

### Visualization Output

The tool generates comprehensive performance visualizations:

- **Percentile Comparison**: P50, P90, P95, P99 latencies across methods
- **Speedup Analysis**: Relative performance improvements
- **Rescoring Impact**: With vs without rescoring comparison
- **Summary Tables**: Quick reference performance data

## 🤝 Contributing

Contributions are welcome! This project follows standard open-source practices.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/assimilatereality/qdrant-quantization-benchmark.git
cd qdrant-quantization-benchmark

# Install dev dependencies
pip install -e ".[dev]"

# Run tests before committing
pytest --cov

# Format code
black src/ tests/

# Run linter
ruff check src/ tests/

# Type checking
mypy src/
```

### Code Standards

- **Testing**: Maintain 80%+ coverage for new code
- **Formatting**: Use Black with default settings
- **Linting**: Pass Ruff checks
- **Type Hints**: Add type annotations for public APIs
- **Documentation**: Update README for new features

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ValueError: QDRANT_URL must be set`
```bash
# Solution: Create .env file with credentials
echo "QDRANT_URL=https://your-cluster.qdrant.io" > .env
echo "QDRANT_API_KEY=your-key" >> .env
```

**Issue**: Import errors after installation
```bash
# Solution: Reinstall in editable mode
pip install -e .
```

**Issue**: Out of memory during embedding generation
```bash
# Solution: Reduce batch size or dataset size
qdrant-benchmark generate-data -n 1000  # Smaller dataset
# Or adjust batch size in config
```

**Issue**: Tests failing with connection errors
```bash
# Solution: Tests use mocked clients, ensure pytest-mock is installed
pip install pytest-mock
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qdrant** - High-performance vector database
- **Sentence Transformers** - State-of-the-art text embeddings
- **Structlog** - Structured logging for Python

## 📮 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/assimilatereality/qdrant-quantization-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/assimilatereality/qdrant-quantization-benchmark/discussions)
- **Email**: AssimilateReality@gmail.com

## 🗺️ Roadmap

### Planned Features

- [ ] PostgreSQL with pgvector support
- [ ] Multi-cloud deployment (AWS, GCP, Azure)
- [ ] Real-time streaming benchmarks
- [ ] Cost optimization calculator
- [ ] CI/CD pipeline templates
- [ ] Grafana dashboard templates
- [ ] Benchmark result database
- [ ] API for programmatic access

### Version History

- **v0.2.0** (Current) - Complete test suite with 90% coverage
- **v0.1.0** - Initial release with core benchmarking features

---

**Built with ❤️ for the vector database community**

*Star ⭐ this repo if you find it useful!*