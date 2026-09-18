#!/bin/bash
#
# SciRS2 SIMD Ultra-Optimization vs NumPy Benchmark
#
# This script runs comprehensive benchmarks comparing:
# - SciRS2 Ultra-Optimized SIMD functions
# - SciRS2 Standard SIMD functions
# - NumPy (backed by BLAS/Accelerate)
#
# Results are saved to $TMPDIR (or /tmp if not set)
#
# Usage:
#   ./benches/run_simd_comparison.sh
#   ./benches/run_simd_comparison.sh --numpy-only
#   ./benches/run_simd_comparison.sh --rust-only
#   ./benches/run_simd_comparison.sh --compare-only
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Use TMPDIR or fallback to /tmp
TEMP_DIR="${TMPDIR:-/tmp}"

echo "================================================================================"
echo "SciRS2 SIMD Ultra-Optimization vs NumPy Benchmark"
echo "================================================================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Temp directory:    $TEMP_DIR"
echo ""

# Parse arguments
NUMPY_ONLY=false
RUST_ONLY=false
COMPARE_ONLY=false

for arg in "$@"; do
    case $arg in
        --numpy-only)
            NUMPY_ONLY=true
            ;;
        --rust-only)
            RUST_ONLY=true
            ;;
        --compare-only)
            COMPARE_ONLY=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --numpy-only    Run only NumPy benchmark"
            echo "  --rust-only     Run only Rust/SciRS2 benchmark"
            echo "  --compare-only  Run only comparison (requires existing CSVs)"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
    esac
done

# Check dependencies
check_dependencies() {
    echo "Checking dependencies..."

    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 not found"
        exit 1
    fi

    if ! python3 -c "import numpy" 2>/dev/null; then
        echo "Error: NumPy not found. Install with: pip3 install numpy"
        exit 1
    fi

    if ! command -v cargo &> /dev/null; then
        echo "Error: cargo not found"
        exit 1
    fi

    echo "  python3: OK"
    echo "  numpy:   OK"
    echo "  cargo:   OK"
    echo ""
}

# Run NumPy benchmark
run_numpy_benchmark() {
    echo "--------------------------------------------------------------------------------"
    echo "Step 1: Running NumPy benchmark..."
    echo "--------------------------------------------------------------------------------"
    python3 "$SCRIPT_DIR/simd_ultra_comparison.py" --numpy-only
    echo ""
}

# Run Rust benchmark
run_rust_benchmark() {
    echo "--------------------------------------------------------------------------------"
    echo "Step 2: Running SciRS2 SIMD benchmark..."
    echo "--------------------------------------------------------------------------------"
    cd "$PROJECT_DIR"
    cargo run --release --features simd --example simd_ultra_benchmark_csv
    echo ""
}

# Run comparison
run_comparison() {
    echo "--------------------------------------------------------------------------------"
    echo "Step 3: Generating comparison report..."
    echo "--------------------------------------------------------------------------------"
    python3 "$SCRIPT_DIR/simd_ultra_comparison.py" --compare-only
    echo ""
}

# Main execution
check_dependencies

if [ "$COMPARE_ONLY" = true ]; then
    run_comparison
elif [ "$NUMPY_ONLY" = true ]; then
    run_numpy_benchmark
elif [ "$RUST_ONLY" = true ]; then
    run_rust_benchmark
else
    run_numpy_benchmark
    run_rust_benchmark
    run_comparison
fi

echo "================================================================================"
echo "Benchmark Complete!"
echo "================================================================================"
echo ""
echo "Output files (in $TEMP_DIR):"
echo "  - NumPy results:      scirs2_numpy_benchmark.csv"
echo "  - SciRS2 results:     scirs2_ultra_benchmark.csv"
echo "  - Comparison report:  scirs2_simd_comparison_report.txt"
echo ""
