# Logging Features Testing Guide

## What's New in Steps 3 & 4

### Step 3: Structured Logging
- ✅ Added `structlog` for structured, parseable logs
- ✅ JSON output mode for CloudWatch/monitoring systems
- ✅ Colored console output for development
- ✅ Context-aware logging with automatic metadata
- ✅ Progress tracking and timing utilities

### Step 4: Enhanced CLI
- ✅ `--verbose` / `-v` flag for debug output
- ✅ `--quiet` / `-q` flag for minimal output
- ✅ `--log-level` for precise control
- ✅ `--json-logs` for structured JSON output
- ✅ All commands now support logging options

## Installation

### 1. Update Dependencies

```bash
# Reinstall with new dependencies
uv pip install -e .

# Verify new packages are installed
uv pip list | grep -E "structlog|colorama"
```

**Expected output:**
```
colorama               0.4.6
structlog              23.1.0 (or higher)
```

### 2. Verify Installation

```bash
# Test that logging module imports correctly
python -c "from qdrant_quantization_benchmark.logging import setup_logging, get_logger; print('✓ Logging module loaded')"
```

## Testing Logging Features

### Test 1: Default Logging (INFO level)

```bash
qdrant-qbench generate-data \
  --size 100 \
  --output data/test_logging.json
```

**Expected output:**
- Colorized console output
- Operation timing information
- Clear status messages

**Example:**
```
[info     ] generating_dataset         size=100 output=data/test_logging.json
[info     ] dataset_generated         items=100 output=data/test_logging.json
```

### Test 2: Verbose Mode (DEBUG level)

```bash
qdrant-qbench generate-data \
  --size 100 \
  --output data/test_verbose.json \
  --verbose
```

**Expected output:**
- All INFO messages PLUS
- DEBUG messages showing internal operations
- Detailed configuration information

**Example:**
```
[debug    ] logging_configured         level=DEBUG json_output=False verbose=True
[info     ] generate_dataset_started   operation=generate_dataset size=100
[info     ] generating_dataset        size=100 output=data/test_verbose.json
[info     ] dataset_generated         items=100
[info     ] generate_dataset_completed operation=generate_dataset duration_ms=150.23
```

### Test 3: Quiet Mode (ERROR level only)

```bash
qdrant-qbench generate-data \
  --size 100 \
  --output data/test_quiet.json \
  --quiet
```

**Expected output:**
- Minimal output (only print statements and errors)
- No INFO or DEBUG messages
- Only see the final "✓ Saved" messages from the generator

### Test 4: JSON Logs (for monitoring)

```bash
qdrant-qbench generate-data \
  --size 100 \
  --output data/test_json.json \
  --json-logs
```

**Expected output:**
- One JSON object per line
- Structured, parseable format
- Ready for CloudWatch/ELK/Splunk

**Example:**
```json
{"app": "qdrant-quantization-benchmark", "version": "0.1.0", "level": "info", "timestamp": "2025-12-12T15:30:45.123456Z", "event": "generating_dataset", "size": 100, "output": "data/test_json.json"}
{"app": "qdrant-quantization-benchmark", "version": "0.1.0", "level": "info", "timestamp": "2025-12-12T15:30:45.456789Z", "event": "dataset_generated", "items": 100, "output": "data/test_json.json"}
```

### Test 5: Specific Log Level

```bash
qdrant-qbench generate-data \
  --size 100 \
  --output data/test_warning.json \
  --log-level WARNING
```

**Expected output:**
- Only WARNING, ERROR, and CRITICAL messages
- Suppresses INFO and DEBUG

### Test 6: Upload with Logging

```bash
qdrant-qbench upload \
  --collection test_logging \
  --dataset data/test_logging.json \
  --recreate \
  --verbose
```

**Expected output:**
- Detailed progress through upload stages:
  - Collection creation
  - Dataset loading
  - Embedding generation
  - Batch uploads with progress

**Example:**
```
[info     ] upload_started            collection=test_logging batch_size=50 retry_enabled=False
[info     ] loading_dataset          path=data/test_logging.json
[info     ] dataset_loaded           items=100
[info     ] creating_collection       collection=test_logging
[info     ] generating_embeddings    count=100
[info     ] uploading_data           collection=test_logging points=100
[info     ] upload_completed         collection=test_logging points_uploaded=100
[info     ] upload_dataset_completed operation=upload_dataset duration_ms=5432.18
```

### Test 7: Benchmark with JSON Logs

```bash
# First create a small dataset
qdrant-qbench generate-data --size 100 --output data/bench_test.json --quiet
qdrant-qbench generate-queries --num-queries 5 --output data/bench_queries.json --quiet
qdrant-qbench upload --collection bench_test --dataset data/bench_test.json --recreate --quiet

# Run benchmark with JSON logging
qdrant-qbench benchmark \
  --collection bench_test \
  --queries data/bench_queries.json \
  --output results/bench_logs.json \
  --json-logs
```

**Expected output:**
- JSON-formatted logs suitable for log aggregation
- Each operation logged with structured context
- Easy to parse and analyze

### Test 8: Multiple Verbosity Flags (Error Handling)

```bash
# Test that quiet overrides verbose
qdrant-qbench generate-data \
  --size 100 \
  --output data/test_conflict.json \
  --verbose \
  --quiet
```

**Expected behavior:**
- `--quiet` should take precedence
- Only ERROR level messages shown

### Test 9: Full Workflow with Verbose Logging

```bash
# Complete workflow with detailed logging
qdrant-qbench generate-data --size 500 --output data/full_test.json -v
qdrant-qbench generate-queries --num-queries 10 --output data/full_queries.json -v
qdrant-qbench upload --collection full_test --dataset data/full_test.json --recreate -v
qdrant-qbench create-quantized --dataset data/full_test.json --methods scalar -v
qdrant-qbench benchmark --collection full_test --queries data/full_queries.json --quantization scalar --output results/full_test.json -v
qdrant-qbench visualize --results results/full_test.json --output results/full_analysis.png -v
```

**Expected output:**
- Detailed DEBUG logs for every operation
- Timing information for each stage
- Progress updates throughout

## Log Output Comparison

### Console Mode (Default)
```
[info     ] upload_started            collection=test batch_size=50
[info     ] loading_dataset           path=data/test.json
✓ Loaded 1000 items from data/test.json
[info     ] dataset_loaded            items=1000
```

### JSON Mode (--json-logs)
```json
{"app": "qdrant-quantization-benchmark", "version": "0.1.0", "level": "info", "event": "upload_started", "collection": "test", "batch_size": 50}
{"app": "qdrant-quantization-benchmark", "version": "0.1.0", "level": "info", "event": "loading_dataset", "path": "data/test.json"}
{"app": "qdrant-quantization-benchmark", "version": "0.1.0", "level": "info", "event": "dataset_loaded", "items": 1000}
```

## Integration with Monitoring Systems

### CloudWatch Logs

```bash
# Pipe JSON logs to CloudWatch
qdrant-qbench benchmark \
  --collection prod \
  --queries data/queries.json \
  --json-logs 2>&1 | \
  aws logs put-log-events \
    --log-group-name /qdrant/benchmark \
    --log-stream-name $(date +%Y%m%d)
```

### Save Logs to File

```bash
# Capture logs for later analysis
qdrant-qbench benchmark \
  --collection test \
  --queries data/queries.json \
  --verbose \
  --json-logs > logs/benchmark-$(date +%Y%m%d-%H%M%S).jsonl
```

## Troubleshooting

### Issue: No color in logs
**Solution:** Some terminals don't support color. Use `--json-logs` for structured output instead.

### Issue: Too much output with verbose
**Solution:** Use `--log-level INFO` instead of `--verbose` for balanced output.

### Issue: Want logs in file AND console
**Solution:** Use `tee` to duplicate output:
```bash
qdrant-qbench benchmark --collection test --verbose 2>&1 | tee logs/output.log
```

### Issue: Structlog import error
**Solution:** Reinstall dependencies:
```bash
uv pip install -e .
```

## Log Fields Reference

All logs include these fields:

### Automatic Fields
- `app`: Always "qdrant-quantization-benchmark"
- `version`: Package version
- `level`: Log level (debug, info, warning, error, critical)
- `timestamp`: ISO 8601 timestamp (UTC) in JSON mode
- `event`: Operation or event name

### Context Fields (varies by operation)
- `operation`: Name of long-running operation
- `duration_ms`: Operation duration in milliseconds
- `collection`: Qdrant collection name
- `size`, `items`, `count`: Data quantities
- `path`, `output`: File paths
- `batch_size`: Upload batch size
- `processed`, `total`: Progress tracking

## Performance Impact

Logging overhead is minimal:
- **Console mode**: < 1ms per log message
- **JSON mode**: < 0.5ms per log message
- **Quiet mode**: Near-zero overhead

For production workloads, use:
- `--log-level INFO` or `--log-level WARNING`
- `--json-logs` for structured output
- Avoid `--verbose` unless debugging

## Next Steps

After verifying logging works:
1. Confirm all tests pass
2. Try different log levels in your workflows
3. Test JSON output with your monitoring system
4. Ready to proceed to **Step 5: Pytest Testing**

## Success Criteria

All tests pass if you see:

✅ Colored console output in default mode  
✅ Debug messages with `--verbose`  
✅ Minimal output with `--quiet`  
✅ Valid JSON with `--json-logs`  
✅ Timing information for operations  
✅ Progress tracking during long operations  
✅ Error messages are clear and actionable