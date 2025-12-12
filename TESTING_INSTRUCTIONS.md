# Testing Instructions for Qdrant Benchmark Suite

## Prerequisites

1. **Qdrant Instance**: You need access to a Qdrant instance (cloud or local)
2. **Environment Variables**: Set up `.env` file with credentials
3. **Python 3.9+**: Ensure you have Python 3.9 or higher

## Installation Steps

### 1. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
```

### 2. Install with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. Verify Installation

```bash
# Check that the CLI is available
qdrant-benchmark --help
```

You should see the help message with all available commands.

## Basic Functionality Tests

### Test 1: Generate Dataset

```bash
# Generate a small test dataset (1000 items for quick testing)
qdrant-benchmark generate-data \
  --size 1000 \
  --output data/test_dataset.json \
  --tech 0.5 \
  --medical 0.5

# Expected output:
# ✓ Saved 1000 items to data/test_dataset.json
```

**Verify**: Check that `data/test_dataset.json` exists and contains 1000 items:
```bash
python -c "import json; data = json.load(open('data/test_dataset.json')); print(f'Items: {len(data)}')"
```

### Test 2: Generate Queries

```bash
# Generate test queries
qdrant-benchmark generate-queries \
  --num-queries 10 \
  --output data/test_queries.json \
  --display

# Expected output:
# Test Queries (10/10 shown)
# 1. python machine learning tutorial
# 2. javascript web development
# ...
# ✓ Saved 10 queries to data/test_queries.json
```

**Verify**: Check that queries file exists:
```bash
cat data/test_queries.json
```

### Test 3: Upload to Qdrant

```bash
# Upload the test dataset to a collection
qdrant-benchmark upload \
  --collection test_benchmark \
  --dataset data/test_dataset.json \
  --batch-size 50 \
  --recreate

# Expected output:
# ✓ Loaded embedding model: all-MiniLM-L6-v2
# Loading dataset from data/test_dataset.json...
# ✓ Loaded 1000 items from data/test_dataset.json
# ✓ Deleted existing collection: test_benchmark
# ✓ Created hybrid collection: test_benchmark
# Pre-computing embeddings for 1000 items...
# ✓ Encoded 1000 items
# ✓ Uploaded 1000 points to test_benchmark
```

**Verify**: Check collection in Qdrant dashboard or via Python:
```python
from qdrant_client import QdrantClient
import os

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
info = client.get_collection("test_benchmark")
print(f"Points: {info.points_count}")  # Should be 1000
```

### Test 4: Run Baseline Benchmark

```bash
# Run benchmark on the base collection
qdrant-benchmark benchmark \
  --collection test_benchmark \
  --queries data/test_queries.json \
  --output results/baseline_results.json

# Expected output:
# Loading queries from data/test_queries.json...
# ✓ Loaded 10 queries from data/test_queries.json
# 
# Benchmarking baseline collection: test_benchmark
# Baseline (No Quantization):
#   P50:     45.23ms
#   P90:     52.18ms
#   P95:     55.67ms
#   P99:     58.92ms
#   P99.5:   59.45ms
#   P99.9:   59.87ms
# 
# ✓ Saved results to results/baseline_results.json
```

**Verify**: Check results file:
```bash
cat results/baseline_results.json | python -m json.tool
```

### Test 5: Create Quantized Collections

```bash
# Create quantized collections (using smaller dataset for faster testing)
qdrant-benchmark create-quantized \
  --dataset data/test_dataset.json \
  --methods scalar binary

# Expected output:
# Loading dataset from data/test_dataset.json...
# ✓ Loaded 1000 items from data/test_dataset.json
# ✓ Loaded embedding model: all-MiniLM-L6-v2
# ✓ Encoded 1000 items
# 
# Creating scalar quantized collection...
# ✓ Deleted existing collection: quantized_scalar
# ✓ Created quantized collection: quantized_scalar
# ✓ Uploaded 1000 points to quantized_scalar
# 
# Creating binary quantized collection...
# ✓ Deleted existing collection: quantized_binary
# ✓ Created quantized collection: quantized_binary
# ✓ Uploaded 1000 points to quantized_binary
```

### Test 6: Benchmark Quantized Collections

```bash
# Benchmark quantized collections
qdrant-benchmark benchmark \
  --collection test_benchmark \
  --queries data/test_queries.json \
  --quantization scalar binary \
  --output results/quantized_results.json

# Expected output:
# Baseline (No Quantization):
#   P50: 45.23ms...
#   ...
# 
# scalar (No Rescoring):
#   P50: 22.15ms...
#   ...
# 
# scalar (With Rescoring):
#   P50: 28.45ms...
#   ...
# 
# binary (No Rescoring):
#   P50: 15.67ms...
#   ...
# 
# binary (With Rescoring):
#   P50: 19.23ms...
#   ...
```

### Test 7: Generate Visualization

```bash
# Generate performance visualization
qdrant-benchmark visualize \
  --results results/quantized_results.json \
  --output results/analysis.png

# Expected output:
# Loading results from results/quantized_results.json...
# ✓ Saved visualization to results/analysis.png
```

**Verify**: Open `results/analysis.png` - should show 4 charts:
1. Percentile comparison
2. Speedup comparison
3. Rescoring impact
4. P95 summary table

## Test with Retry Logic

```bash
# Test upload with retry enabled (useful for unstable connections)
qdrant-benchmark upload \
  --collection test_retry \
  --dataset data/test_dataset.json \
  --batch-size 50 \
  --enable-retry \
  --recreate

# Should complete successfully, potentially with retry messages if timeouts occur
```

## Full Integration Test (Larger Dataset)

For a more comprehensive test:

```bash
# 1. Generate larger dataset (10k items)
qdrant-benchmark generate-data --size 10000 --output data/full_dataset.json

# 2. Generate more queries
qdrant-benchmark generate-queries --num-queries 20 --output data/full_queries.json

# 3. Upload to Qdrant
qdrant-benchmark upload \
  --collection full_benchmark \
  --dataset data/full_dataset.json \
  --recreate

# 4. Create all quantized collections
qdrant-benchmark create-quantized \
  --dataset data/full_dataset.json \
  --methods scalar binary binary_2bit

# 5. Run full benchmark
qdrant-benchmark benchmark \
  --collection full_benchmark \
  --queries data/full_queries.json \
  --quantization scalar binary binary_2bit \
  --output results/full_results.json

# 6. Generate visualization
qdrant-benchmark visualize \
  --results results/full_results.json \
  --output results/full_analysis.png
```

## Troubleshooting

### Issue: "QDRANT_URL must be set"
- **Solution**: Ensure `.env` file exists with correct credentials
- Verify: `cat .env`

### Issue: "Collection does not exist"
- **Solution**: Run the upload command first with `--recreate` flag
- Or check collection name spelling

### Issue: Slow embedding generation
- **Expected**: First run downloads the sentence-transformers model (~80MB)
- Subsequent runs use cached model

### Issue: Import errors
- **Solution**: Ensure package is installed: `uv pip install -e .`
- Verify: `python -c "import qdrant_benchmark; print(qdrant_benchmark.__version__)"`

### Issue: Permission denied when creating data directory
- **Solution**: Create directory manually: `mkdir -p data results`

## Expected Performance Characteristics

With a 10,000 item dataset on Qdrant Cloud (free tier):
- **Baseline P95**: 40-80ms
- **Scalar quantization speedup**: ~1.5-2x
- **Binary quantization speedup**: ~2-4x
- **Binary 2-bit speedup**: ~2-3x

Note: Performance varies based on:
- Qdrant instance specs
- Network latency
- Query complexity
- Dataset characteristics

## Cleanup

To remove test data:
```bash
# Remove local data
rm -rf data/ results/

# Delete test collections from Qdrant (via Python)
python -c "
from qdrant_client import QdrantClient
import os

client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
for name in ['test_benchmark', 'test_retry', 'full_benchmark', 'quantized_scalar', 'quantized_binary', 'quantized_binary_2bit']:
    try:
        client.delete_collection(name)
        print(f'Deleted: {name}')
    except:
        pass
"
```

## Next Steps

Once all tests pass successfully:
1. Confirm authorization to proceed with Step 3 (Structured Logging)
2. Confirm authorization to proceed with Step 4 (Enhanced CLI with logging)
3. Plan Step 5 (pytest implementation for 80% coverage)

## Test Checklist

- [ ] Installation via uv completed
- [ ] Dataset generation works
- [ ] Query generation works
- [ ] Upload to Qdrant successful
- [ ] Baseline benchmark runs
- [ ] Quantized collections created
- [ ] Quantized benchmarks run
- [ ] Visualization generates correctly
- [ ] Retry logic tested (if needed)
- [ ] Full integration test passes