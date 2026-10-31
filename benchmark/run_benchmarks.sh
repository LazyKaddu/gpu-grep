#!/bin/bash
set -e

# Navigate to the project root
cd "$(dirname "$0")/.."

DATASET="benchmark/massive_test.log"
PATTERN="AashishNegi"

echo "building gpu-grep release binary..."
cargo build --release

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found. Generating 10GB log file (this will take a few minutes)..."
    python3 benchmark/generate_logs.py
else
    echo "Dataset found: $DATASET"
fi

echo "Starting Hyperfine Showdown..."
# Use the compiled binary directly from the target folder
GPU_GREP="./target/release/gpu-grep"

hyperfine --warmup 2 \
  --export-markdown benchmark/results/benchmark_report.md \
  "grep -a '$PATTERN' $DATASET" \
  "rg '$PATTERN' $DATASET" \
  "$GPU_GREP '$PATTERN' $DATASET"

echo "📊 Benchmark complete! Results saved to benchmarks/results/benchmark_report.md"