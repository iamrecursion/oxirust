#!/bin/bash
# Comprehensive benchmark runner for scirs2-special
#
# This script runs both SciPy and Rust benchmarks and generates comparison reports.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== SciRS2 Special Functions Benchmark Suite ==="
echo "Project directory: $PROJECT_DIR"
echo

# Check if Python and required packages are available
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Warning: python3 not found. Skipping SciPy benchmarks."
    SKIP_SCIPY=1
else
    if ! python3 -c "import numpy, scipy.special" 2>/dev/null; then
        echo "Warning: numpy/scipy not available. Skipping SciPy benchmarks."
        SKIP_SCIPY=1
    else
        echo "✓ Python environment ready"
        SKIP_SCIPY=0
    fi
fi

echo

# Run SciPy benchmarks if available
if [ "$SKIP_SCIPY" -eq 0 ]; then
    echo "Running SciPy benchmarks..."
    cd "$PROJECT_DIR"
    python3 benches/scipy_comparison.py
    echo "✓ SciPy benchmarks completed"
    echo
fi

# Run Rust benchmarks
echo "Running Rust benchmarks..."
cd "$PROJECT_DIR"

echo "Building benchmark binaries..."
cargo build --release --benches

echo
echo "Running original Bessel benchmarks..."
cargo bench --bench bessel_bench

echo
echo "Running comprehensive benchmarks..."
cargo bench --bench comprehensive_bench

echo
echo "=== Benchmark Summary ==="

# Generate summary report
if [ "$SKIP_SCIPY" -eq 0 ] && [ -f "benches/scipy_benchmark_results.json" ]; then
    echo "SciPy vs Rust comparison results are included in the benchmark output above."
    echo "Detailed SciPy results saved in: benches/scipy_benchmark_results.json"
else
    echo "SciPy comparison not available (missing Python/NumPy/SciPy)."
fi

echo "Rust benchmark results saved in: target/criterion/"
echo
echo "To view detailed HTML reports, open: target/criterion/report/index.html"
echo "To install Python dependencies: pip install numpy scipy"
echo
echo "Benchmark suite completed successfully!"