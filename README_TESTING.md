# Testing Guide for Qdrant Quantization Benchmark

## Overview

This project uses **pytest** for comprehensive testing with **80%+ code coverage**. All tests are designed to run quickly (< 30 seconds) without requiring external dependencies like actual Qdrant instances or embedding model downloads.

## Quick Start

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage report
pytest --cov

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v

# Run and generate HTML coverage report
pytest --cov=src/qdrant_quantization_benchmark --cov-report=html
open htmlcov/index.html
```

## Test Structure

```
tests/
├── conftest.py                  # Shared fixtures and test utilities
├── test_config.py               # Configuration dataclasses (95%+ coverage)
├── test_data_generator.py       # Dataset generation (85%+ coverage)
├── test_query_generator.py      # Query generation (85%+ coverage)
├── test_qdrant_manager.py       # Collection management (90%+ coverage)
├── test_embeddings.py           # Embedding service (90%+ coverage)
├── test_uploader.py             # Upload & retry logic (90%+ coverage)
├── test_benchmarking.py         # Performance measurement (85%+ coverage)
├── test_logging.py              # Logging utilities (80%+ coverage)
├── test_visualization.py        # Visualization (70%+ coverage)
└── test_cli.py                  # CLI integration (70%+ coverage)
```

## Test Coverage Targets

| Module | Target | Achieved |
|--------|--------|----------|
| config.py | 95%+ | ✅ |
| data_generator.py | 85%+ | ✅ |
| query_generator.py | 85%+ | ✅ |
| qdrant_manager.py | 90%+ | ✅ |
| embeddings.py | 90%+ | ✅ |
| uploader.py | 90%+ | ✅ |
| benchmarking.py | 85%+ | ✅ |
| logging.py | 80%+ | ✅ |
| visualization.py | 70%+ | ✅ |
| cli.py | 70%+ | ✅ |
| **Overall** | **80%+** | **✅** |

## Key Testing Patterns

### 1. Mocking External Dependencies

We mock external services to keep tests fast and reliable:

```python
# Qdrant client is mocked
@pytest.fixture
def mock_qdrant_client():
    client = Mock(spec=QdrantClient)
    client.collection_exists.return_value = False
    return client

# SentenceTransformer is mocked
@pytest.fixture
def mock_sentence_transformer(mocker):
    mock_model = Mock()
    mock_model.encode.return_value = np.random.rand(384)
    mocker.patch('qdrant_quantization_benchmark.embeddings.SentenceTransformer',
                 return_value=mock_model)
    return mock_model
```

### 2. Using Fixtures for Test Data

Shared test data is defined in `conftest.py`:

```python
@pytest.fixture
def sample_dataset():
    """Generate a small test dataset."""
    generator = DatasetGenerator(seed=42)
    return generator.generate(n=20, domain_mix={'tech': 0.5, 'medical': 0.5})

@pytest.fixture
def temp_dataset_file(tmp_path, sample_dataset):
    """Create a temporary dataset file."""
    filepath = tmp_path / "test_dataset.json"
    generator = DatasetGenerator()
    generator.save_dataset(sample_dataset, str(filepath))
    return filepath
```

### 3. Parametrized Tests

Testing multiple scenarios efficiently:

```python
@pytest.mark.parametrize("domain_mix,expected_counts", [
    ({'tech': 1.0}, {'tech': 100}),
    ({'tech': 0.5, 'medical': 0.5}, {'tech': 50, 'medical': 50}),
])
def test_domain_mix_distribution(domain_mix, expected_counts):
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate(n=100, domain_mix=domain_mix)
    # assertions...
```

### 4. Testing Retry Logic

```python
def test_retry_succeeds_after_one_failure(mock_qdrant_client):
    config = UploadConfig(enable_retry=True, max_retries=3)
    uploader = DataUploader(mock_qdrant_client, config)
    
    # First call fails, second succeeds
    mock_qdrant_client.upsert.side_effect = [
        ResponseHandlingException("Timeout"),
        None  # Success
    ]
    
    result = uploader._upload_with_retry("test", [], 0)
    assert mock_qdrant_client.upsert.call_count == 2
```

## Running Specific Test Categories

### Fast Tests Only

```bash
# Run only fast unit tests (skip slow integration tests if marked)
pytest -m "not slow"
```

### By Module

```bash
# Test configuration
pytest tests/test_config.py -v

# Test data generation
pytest tests/test_data_generator.py tests/test_query_generator.py

# Test core logic
pytest tests/test_qdrant_manager.py tests/test_embeddings.py tests/test_uploader.py

# Test benchmarking
pytest tests/test_benchmarking.py

# Test CLI
pytest tests/test_cli.py
```

### With Coverage

```bash
# HTML report (opens in browser)
pytest --cov --cov-report=html && open htmlcov/index.html

# Terminal report with missing lines
pytest --cov --cov-report=term-missing

# Fail if coverage is below 80%
pytest --cov --cov-fail-under=80
```

## Test Fixtures Reference

### Configuration Fixtures

- `embedding_config` - Standard embedding configuration
- `collection_config` - Standard collection configuration
- `upload_config` - Standard upload configuration
- `benchmark_config` - Standard benchmark configuration
- `full_config` - Complete benchmark suite configuration

### Mock Fixtures

- `mock_qdrant_client` - Mocked Qdrant client
- `mock_sentence_transformer` - Mocked embedding model

### Data Fixtures

- `sample_dataset` - 20-item test dataset
- `sample_queries` - 5 test queries
- `sample_embeddings` - 20 embeddings (384-dim)
- `temp_dataset_file` - Temporary dataset JSON file
- `temp_queries_file` - Temporary queries JSON file
- `mock_benchmark_results` - Fake benchmark results for visualization

## Common Test Commands

```bash
# Run all tests
pytest

# Run with output from print statements
pytest -s

# Run specific test
pytest tests/test_config.py::TestLoggingConfig::test_default_values

# Run tests matching pattern
pytest -k "test_upload"

# Stop on first failure
pytest -x

# Show test execution times
pytest --durations=10

# Run in parallel (if pytest-xdist installed)
pytest -n auto
```

## Debugging Failed Tests

### 1. Verbose Output

```bash
pytest -vv tests/test_config.py
```

### 2. Show Print Statements

```bash
pytest -s tests/test_embeddings.py
```

### 3. Drop into Debugger

```bash
pytest --pdb tests/test_uploader.py
```

### 4. Last Failed

```bash
# Re-run only tests that failed last time
pytest --lf
```

## Coverage Analysis

### Check Uncovered Lines

```bash
pytest --cov --cov-report=term-missing
```

Output shows which lines aren't covered:
```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src/qdrant_benchmark/config.py           45      2    96%   123-124
src/qdrant_benchmark/embeddings.py       67      5    93%   45, 89-92
```

### Generate HTML Report

```bash
pytest --cov --cov-report=html
```

Then open `htmlcov/index.html` to see:
- Line-by-line coverage
- Branch coverage
- Which tests cover which lines

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov --cov-fail-under=80
```

## What's NOT Tested

To maintain fast, reliable tests, we explicitly don't test:

1. **Actual Qdrant connections** - Would be slow and require infrastructure
2. **Actual embedding model downloads** - Would download large models
3. **Actual image rendering** - We test data processing, not matplotlib output
4. **Real file I/O** - We use pytest's `tmp_path` fixture
5. **Network calls** - Everything external is mocked

## Writing New Tests

### Template for New Test Module

```python
"""
Tests for [module_name] functionality.
"""

import pytest
from qdrant_quantization_benchmark.[module] import [Class]


class Test[ClassName]:
    """Tests for [ClassName] class."""
    
    def test_basic_functionality(self):
        """Test basic operation."""
        obj = Class()
        result = obj.method()
        assert result == expected
    
    def test_error_handling(self):
        """Test error cases."""
        obj = Class()
        with pytest.raises(ValueError):
            obj.method_that_fails()
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
    ])
    def test_multiple_cases(self, input, expected):
        """Test multiple input scenarios."""
        assert function(input) == expected
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Descriptive test names** - `test_upload_fails_with_invalid_data` not `test_error`
3. **Use fixtures** - Don't repeat setup code
4. **Mock external dependencies** - Keep tests fast
5. **Test edge cases** - Empty lists, None values, boundary conditions
6. **Test error paths** - Not just happy paths

## Troubleshooting

### Tests Pass Locally But Fail in CI

- Check Python version matches
- Verify all dependencies are installed
- Check for environment-specific code

### Flaky Tests

- Look for time-dependent code
- Check for race conditions
- Ensure proper mocking

### Coverage Not Increasing

- Check for defensive code that's hard to trigger
- Look for error handling paths
- Consider if 100% coverage is necessary

### Slow Tests

- Profile with `pytest --durations=10`
- Check for unmocked external calls
- Consider splitting into unit/integration tests

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)

## Success Criteria

Tests are successful when:

✅ All tests pass  
✅ Coverage ≥ 80%  
✅ Tests run in < 30 seconds  
✅ No external dependencies required  
✅ Tests are deterministic (no flakiness)  

Current status: **All criteria met! ✅**