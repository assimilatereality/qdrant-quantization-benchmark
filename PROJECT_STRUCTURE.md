# Project Structure Reference

## Complete Directory Layout

```
qdrant-quantization-benchmark/
│
├── pyproject.toml                 # Package configuration, dependencies, build settings
├── README.md                      # Project overview and quick start guide
├── TESTING_INSTRUCTIONS.md        # Detailed step-by-step testing procedures
├── PROJECT_STRUCTURE.md           # This file - code organization reference
├── .env                           # Environment variables (QDRANT_URL, QDRANT_API_KEY)
├── .gitignore                     # Git ignore patterns
│
├── src/
│   └── qdrant_quantization_benchmark/  # Main package directory
│       ├── __init__.py            # Package exports and version
│       ├── config.py              # Configuration dataclasses
│       ├── qdrant_manager.py      # Collection lifecycle management
│       ├── embeddings.py          # Embedding generation service
│       ├── uploader.py            # Batch upload with retry logic
│       ├── benchmarking.py        # Performance measurement
│       ├── visualization.py       # Chart generation
│       ├── cli.py                 # Argparse CLI interface
│       ├── data_generator.py      # Multi-domain dataset generation
│       └── query_generator.py     # Test query generation
│
├── tests/                         # Test suite (Step 5 - to be implemented)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_qdrant_manager.py
│   ├── test_embeddings.py
│   ├── test_uploader.py
│   ├── test_benchmarking.py
│   ├── test_visualization.py
│   ├── test_data_generator.py
│   └── test_query_generator.py
│
├── examples/                      # Example scripts and notebooks
│   └── generate_data_and_queries.py
│
├── data/                          # Generated datasets (gitignored)
│   ├── datasets/
│   └── queries/
│
└── results/                       # Benchmark results and visualizations (gitignored)
    ├── *.json
    └── *.png
```

## Module Responsibilities

### config.py - Configuration Management

**Purpose**: Centralize all configuration using dataclasses

**Key Classes**:
- `EmbeddingConfig`: Model name, vector size
- `CollectionConfig`: Distance metric, storage options
- `UploadConfig`: Batch size, retry settings
- `BenchmarkConfig`: Test queries, oversampling factors
- `QuantizationConfig`: Scalar, binary, 2-bit binary configs
- `QdrantConnectionConfig`: URL, API key, timeout
- `BenchmarkSuiteConfig`: Master config combining all above

**Usage**:
```python
config = BenchmarkSuiteConfig.from_env()
print(config.embedding.model_name)  # "all-MiniLM-L6-v2"
```

---

### qdrant_manager.py - Collection Operations

**Purpose**: Manage Qdrant collection lifecycle

**Key Class**: `QdrantCollectionManager`

**Main Methods**:
- `collection_exists()`: Check if collection exists
- `delete_collection()`: Delete collection if exists
- `create_hybrid_collection()`: Create dense + sparse vectors
- `create_standard_collection()`: Create dense vectors only
- `create_quantized_collection()`: Create with quantization
- `recreate_collection()`: Delete and recreate
- `get_collection_info()`: Get collection metadata

**Usage**:
```python
manager = QdrantCollectionManager(client)
manager.recreate_collection("my_collection", collection_type="standard")
```

---

### embeddings.py - Embedding Service

**Purpose**: Generate embeddings from text using SentenceTransformers

**Key Class**: `EmbeddingService`

**Main Methods**:
- `encode_text()`: Single text → vector
- `encode_batch()`: Multiple texts → vectors (with progress)
- `encode_dataset()`: Dataset items → vectors (combines title + description)

**Usage**:
```python
service = EmbeddingService()
vector = service.encode_text("machine learning tutorial")
embeddings = service.encode_dataset(dataset, show_progress=True)
```

---

### uploader.py - Data Upload

**Purpose**: Batch upload with optional retry logic

**Key Class**: `DataUploader`

**Main Methods**:
- `upload_batch()`: Upload dataset with precomputed embeddings
- `_prepare_points()`: Create PointStruct objects
- `_upload_with_retry()`: Retry with exponential backoff

**Preserved Retry Code**: Commented alternative implementation at bottom

**Usage**:
```python
uploader = DataUploader(client, config)
uploader.upload_batch(
    collection_name="my_collection",
    dataset=dataset,
    embeddings=embeddings,
    named_vector=True
)
```

---

### benchmarking.py - Performance Measurement

**Purpose**: Measure search latency and accuracy

**Key Class**: `PerformanceBenchmark`

**Main Methods**:
- `warmup()`: Warm up caches with throwaway query
- `measure_search_latency()`: Measure P50-P99.9 latencies
- `benchmark_quantization()`: Test with/without rescoring
- `tune_oversampling()`: Find optimal oversampling factor
- `measure_accuracy_retention()`: Compare quantized vs original

**Metrics Tracked**: avg, p50, p90, p95, p99, p99.5, p99.9

**Usage**:
```python
benchmark = PerformanceBenchmark(client, embedding_service)
metrics = benchmark.measure_search_latency(
    collection_name="my_collection",
    test_queries=queries,
    label="Baseline"
)
```

---

### visualization.py - Chart Generation

**Purpose**: Create matplotlib visualizations from results

**Key Class**: `BenchmarkVisualizer` (all static methods)

**Main Methods**:
- `plot_quantization_results()`: Generate 4-panel analysis chart
- `print_analysis_summary()`: Console output of results
- `print_oversampling_analysis()`: Oversampling factor comparison

**Charts Generated**:
1. Percentile comparison across methods
2. Speedup comparison (horizontal bar chart)
3. Rescoring impact (grouped bar chart)
4. P95 summary table

**Usage**:
```python
BenchmarkVisualizer.plot_quantization_results(
    baseline_metrics=baseline,
    quantization_results=results,
    output_path="analysis.png"
)
```

---

### cli.py - Command Line Interface

**Purpose**: Argparse-based CLI tying all modules together

**Commands**:
- `generate-data`: Create synthetic dataset
- `generate-queries`: Create test queries
- `upload`: Upload dataset to Qdrant
- `create-quantized`: Create quantized collections
- `benchmark`: Run performance tests
- `visualize`: Generate charts

**Each command**:
- Parses arguments
- Initializes required services
- Executes workflow
- Handles errors gracefully

**Entry Point**: `main()` function

---

### data_generator.py - Dataset Generation

**Purpose**: Generate synthetic multi-domain datasets

**Key Class**: `DatasetGenerator`

**Domains Supported**:
- Tech (programming books)
- Medical (textbooks/guides)
- Pharmaceutical (medications)
- Health Insurance (plans)

**Main Methods**:
- `generate()`: Create n items with domain mix
- `save_dataset()`: Save to JSON
- `load_dataset()`: Load from JSON (static)

**Item Structure**:
```python
{
    "id": 0,
    "domain": "tech",
    "title": "Python for Machine Learning - Edition 1",
    "description": "Comprehensive guide to...",
    "metadata": {...}
}
```

---

### query_generator.py - Query Generation

**Purpose**: Generate test queries across domains

**Key Class**: `QueryGenerator`

**Main Methods**:
- `generate_auto_queries()`: Auto-generate based on templates
- `add_manual_queries()`: Add user-specified queries
- `save_queries()`: Save to JSON with metadata
- `load_queries()`: Load from JSON
- `display_queries()`: Pretty print
- `get_domain_distribution()`: Analyze query distribution

**Query Templates**: Per-domain templates for realistic queries

---

## Data Flow

### 1. Dataset Generation Flow
```
CLI (generate-data)
  → DatasetGenerator.generate()
  → Save to JSON file
```

### 2. Upload Flow
```
CLI (upload)
  → Load dataset from JSON
  → EmbeddingService.encode_dataset()
  → QdrantCollectionManager.recreate_collection()
  → DataUploader.upload_batch()
    → Batch processing
    → Optional retry logic
    → Upload to Qdrant
```

### 3. Benchmark Flow
```
CLI (benchmark)
  → Load queries
  → PerformanceBenchmark.measure_search_latency() [baseline]
  → For each quantization method:
    → PerformanceBenchmark.benchmark_quantization()
  → Save results to JSON
  → BenchmarkVisualizer.print_analysis_summary()
```

### 4. Visualization Flow
```
CLI (visualize)
  → Load results from JSON
  → BenchmarkVisualizer.plot_quantization_results()
  → Save PNG chart
```

## Configuration Flow

```
.env file
  ↓
os.getenv()
  ↓
QdrantConnectionConfig (validates)
  ↓
BenchmarkSuiteConfig (aggregates all configs)
  ↓
Individual service initialization
```

## Error Handling Strategy

Each module handles errors at appropriate level:

- **Config**: Validates at initialization, raises ValueError
- **Qdrant Manager**: Checks existence before operations
- **Uploader**: Optional retry with backoff
- **Benchmarking**: Catches query failures, logs, continues
- **CLI**: Try-catch at command level, user-friendly messages

## Testing Strategy (Step 5)

Target 80% coverage:

1. **Unit tests**: Each module independently
2. **Integration tests**: Module interactions
3. **Mocking**: Qdrant client for offline tests
4. **Fixtures**: Reusable test data
5. **Parametrized tests**: Multiple scenarios

## Future Enhancements

### Step 3: Structured Logging
- Add `structlog` integration
- JSON log output for CloudWatch
- Contextual logging (operation, duration, collection)

### Step 4: Enhanced CLI
- Add `--verbose` and `--quiet` flags
- Integrate logging throughout
- Progress bars for long operations

### Step 5: Pytest Suite
- Comprehensive test coverage (80%+)
- Mock Qdrant client
- Test fixtures for datasets
- CI/CD integration

## Quick Navigation

**Need to**:
- **Change vector size?** → `config.py` → `EmbeddingConfig`
- **Add new domain?** → `data_generator.py` → Add to `_generate_*_item()`
- **Modify retry logic?** → `uploader.py` → `UploadConfig` and `_upload_with_retry()`
- **Add new metric?** → `benchmarking.py` → `_calculate_metrics()`
- **Customize charts?** → `visualization.py` → `_plot_*()` methods
- **Add CLI command?** → `cli.py` → Add subparser and `cmd_*()` function

## Dependencies

**Core**:
- `qdrant-client`: Vector database client
- `sentence-transformers`: Embedding generation
- `numpy`: Numerical operations
- `matplotlib`: Visualization

**Dev** (optional):
- `pytest`: Testing framework
- `black`: Code formatting
- `ruff`: Linting
- `mypy`: Type checking

All managed via `pyproject.toml` and installed via `uv`.