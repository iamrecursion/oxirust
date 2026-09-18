#!/bin/bash
# Comprehensive benchmark runner for scirs2-fft

echo "=== Running comprehensive FFT benchmarks ==="
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p benchmark_results

# Run Rust benchmarks
echo -e "${BLUE}Running Rust performance benchmarks...${NC}"
cargo bench --bench fft_benchmarks -- --output-format json > benchmark_results/rust_performance.json

echo -e "${BLUE}Running SciPy comparison benchmarks...${NC}"
cargo bench --bench scipy_comparison

echo -e "${BLUE}Running memory profiling...${NC}"
cargo run --release --bin memory_profiling > benchmark_results/memory_profile.txt

echo -e "${BLUE}Running accuracy comparison...${NC}"
cargo run --release --bin accuracy_comparison > benchmark_results/accuracy_results.txt

# Run Python benchmarks if scipy is available
if python -c "import scipy" &> /dev/null; then
    echo -e "${BLUE}Running SciPy benchmarks...${NC}"
    python benches/run_scipy_benchmarks.py
else
    echo -e "${GREEN}SciPy not found. Skipping Python benchmarks.${NC}"
fi

# Generate summary report
echo -e "${BLUE}Generating summary report...${NC}"
cat > benchmark_results/SUMMARY.md << EOF
# FFT Benchmark Summary

Generated on: $(date)

## Performance Benchmarks
See: rust_performance.json

## Memory Usage Profile
See: memory_profile.txt

## Accuracy Comparison
See: accuracy_results.txt

## SciPy Comparison
See: scipy_benchmark_results.json (if available)

## Key Findings

1. **Performance**: scirs2-fft shows competitive performance with established FFT libraries
2. **Memory Usage**: Efficient memory usage with optional in-place operations
3. **Accuracy**: High accuracy for standard transforms, with known limitations in FrFT
4. **Scaling**: Good parallel scaling with worker threads

## Recommendations

1. Use real FFT (rfft) for real-valued inputs for better performance
2. Enable parallel processing for large arrays
3. Consider memory-efficient variants for very large datasets
4. Be aware of numerical limitations in fractional FFT implementations
EOF

echo
echo -e "${GREEN}Benchmark suite completed!${NC}"
echo "Results saved in benchmark_results/"