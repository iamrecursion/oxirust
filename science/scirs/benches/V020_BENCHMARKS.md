# SciRS2 v0.2.0 Performance Benchmark Suite

Comprehensive performance validation and comparison suite for SciRS2 v0.2.0 release.

## Overview

This benchmark suite provides production-ready performance validation across all major SciRS2 modules with the following focus areas:

1. **Comprehensive Coverage**: All modules and operations
2. **SIMD Performance**: Quantify vectorization benefits
3. **GPU Acceleration**: Measure GPU vs CPU speedups
4. **Memory Efficiency**: Profile allocations and bandwidth
5. **Scalability**: Thread and data size scaling
6. **Python Comparison**: Competitive analysis vs SciPy/NumPy/PyTorch

## Quick Start

### Run All Benchmarks

```bash
cd benches
./v020_run_all_benchmarks.sh
```

This will run all benchmarks and generate comprehensive reports (~1-2 hours).

### Run Individual Benchmark Suites

```bash
# Comprehensive suite (all modules)
cargo bench --bench v020_comprehensive_suite

# SIMD vs scalar comparison
cargo bench --bench v020_simd_comparison

# GPU vs CPU comparison (requires GPU backend)
cargo bench --bench v020_gpu_comparison --features cuda

# Memory profiling
cargo bench --bench v020_memory_profiling

# Scalability analysis
cargo bench --bench v020_scalability

# Python comparison
cargo bench --bench v020_python_comparison
python3 benches/v020_python_comparison.py
```

### Quick Benchmarks (Reduced Measurement Time)

```bash
./v020_run_all_benchmarks.sh --quick
```

### Skip Python Comparison

```bash
./v020_run_all_benchmarks.sh --skip-python
```

### Skip GPU Benchmarks

```bash
./v020_run_all_benchmarks.sh --skip-gpu
```

## Benchmark Suites

### 1. Comprehensive Suite (`v020_comprehensive_suite.rs`)

**Purpose**: Validate performance across all SciRS2 modules

**Categories**:
- Core array operations (creation, elementwise ops)
- Linear algebra (decompositions, solvers, matmul)
- FFT operations (1D, 2D, batched)
- Statistical operations (mean, std, quantiles)
- Integration (quad, trapz, simps)
- Special functions (Bessel, Gamma, Erf)
- Clustering (KMeans, hierarchical)
- Signal processing (convolution, correlation)

**Output**: `/tmp/scirs2_v020_benchmark_results.json`

**View Results**:
```bash
cat /tmp/scirs2_v020_benchmark_results.json | jq .
```

### 2. SIMD Comparison (`v020_simd_comparison.rs`)

**Purpose**: Quantify SIMD vectorization benefits

**Comparisons**:
- Naive scalar implementation
- Unrolled scalar implementation
- SIMD optimized implementation

**Operations**:
- Dot product (f32, f64) - various sizes
- Matrix multiplication (square, rectangular)
- Elementwise operations (add, multiply)
- Cache-level performance analysis

**Performance Targets**:
- Dot product (f32): 8-12x speedup (AVX2), 12-18x (AVX-512)
- Dot product (f64): 4-6x speedup (AVX2), 6-10x (AVX-512)
- Matrix multiply: 10-30x speedup for large matrices

**View Results**:
```bash
open target/criterion/simd_vs_scalar/report/index.html
```

### 3. GPU Comparison (`v020_gpu_comparison.rs`)

**Purpose**: Measure GPU acceleration benefits

**Backends Supported**:
- CUDA (NVIDIA)
- Metal (Apple)
- WGPU (cross-platform)

**Operations**:
- Matrix multiplication (various sizes)
- FFT operations
- Batch operations
- Element-wise operations
- Data transfer overhead measurement

**Performance Targets**:
- Matrix multiply: 5-50x speedup for large matrices
- FFT: 3-20x speedup for large FFTs
- Neural ops: 10-100x speedup for batch operations

**Enable GPU Benchmarks**:
```bash
# CUDA
cargo bench --bench v020_gpu_comparison --features cuda

# Metal
cargo bench --bench v020_gpu_comparison --features metal-backend

# WGPU
cargo bench --bench v020_gpu_comparison --features wgpu-backend
```

### 4. Memory Profiling (`v020_memory_profiling.rs`)

**Purpose**: Profile memory usage and efficiency

**Measurements**:
- Allocation overhead (Vec, preallocated, uninitialized)
- Memory reuse patterns
- Memory bandwidth (sequential, strided, random access)
- Cache efficiency (L1, L2, L3, DRAM)
- Chunked processing overhead
- Out-of-core performance
- Memory pool effectiveness
- Zero-copy operations

**Performance Targets**:
- Allocation overhead: <5% vs preallocated
- Cache efficiency: >80% L1 hit rate for small matrices
- Memory bandwidth: >50% of theoretical peak
- Out-of-core: <20% performance degradation

**View Results**:
```bash
open target/criterion/memory/report/index.html
```

### 5. Scalability Analysis (`v020_scalability.rs`)

**Purpose**: Measure scaling characteristics

**Dimensions**:
- **Thread Scaling**: 1, 2, 4, 8, 16 threads
- **Data Size Scaling**: 1K to 1M+ elements
- **Complexity Validation**: O(n), O(n log n), O(n²), O(n³)

**Operations**:
- Parallel reduction (sum, mean)
- Parallel map (elementwise operations)
- Parallel matrix operations
- Batch processing
- Dimension scaling (1D, 2D)

**Performance Targets**:
- 2 threads: >80% parallel efficiency
- 4 threads: >60% parallel efficiency
- 8 threads: >40% parallel efficiency

**View Results**:
```bash
open target/criterion/scalability/report/index.html
```

### 6. Python Comparison (`v020_python_comparison.rs` + `.py`)

**Purpose**: Compare against Python scientific computing stack

**Python Libraries**:
- NumPy: Array operations
- SciPy: Scientific functions, linear algebra
- (Future: PyTorch for neural operations)

**Methodology**:
1. Rust benchmarks save results to JSON
2. Python script loads JSON and runs equivalent operations
3. Generate comparison report with speedup ratios
4. Create visualization plots

**Operations Compared**:
- Array operations (zeros, arange, add, multiply, sum, mean)
- Linear algebra (det, inv, lu, qr, solve, matmul)
- FFT (forward, inverse)
- Statistics (mean, std, var, median)
- Special functions (Bessel, Gamma, Erf)
- Integration (trapz, simps, quad)

**Performance Targets**:
- Array operations: 2-10x faster than NumPy
- Linear algebra: Competitive with SciPy (using OxiBLAS)
- FFT: Competitive with SciPy/NumPy
- Special functions: Competitive or faster than SciPy

**Run Comparison**:
```bash
# 1. Run Rust benchmarks
cargo bench --bench v020_python_comparison

# 2. Run Python benchmarks and generate comparison
python3 benches/v020_python_comparison.py

# 3. View results
cat /tmp/scirs2_v020_comparison_results.json | jq .summary
open /tmp/scirs2_v020_speedup_plot.png
```

## Understanding Results

### Criterion HTML Reports

After running benchmarks, detailed HTML reports are generated in `target/criterion/`:

```bash
open target/criterion/report/index.html
```

The reports provide:
- Interactive performance graphs
- Statistical analysis (mean, median, std dev)
- Regression detection
- Comparison between implementations
- Throughput measurements

### JSON Result Files

Programmatic access to results:

```bash
# Comprehensive results
cat /tmp/scirs2_v020_benchmark_results.json | jq .

# Python comparison summary
cat /tmp/scirs2_v020_comparison_results.json | jq .summary

# Rust results for Python comparison
cat /tmp/scirs2_v020_python_comparison_rust.json | jq .

# Python benchmark results
cat /tmp/scirs2_v020_python_comparison_python.json | jq .
```

### Performance Metrics

**Throughput**: Operations or elements processed per second
- Higher is better
- Useful for comparing different implementations

**Latency**: Time per operation in nanoseconds
- Lower is better
- Includes mean, median, standard deviation

**Speedup**: Ratio of baseline time to optimized time
- `speedup = baseline_time / optimized_time`
- `speedup > 1.0`: Optimized is faster
- `speedup < 1.0`: Baseline is faster

**Parallel Efficiency**: `efficiency = speedup / num_threads * 100%`
- 100%: Perfect linear scaling
- >80%: Excellent scaling
- 50-80%: Good scaling
- <50%: Poor scaling, bottlenecks present

## Dependencies

### Rust Dependencies

All required dependencies are in `benches/Cargo.toml`:

- `criterion`: Benchmarking framework
- `ndarray`: N-dimensional arrays
- `rayon`: Parallel processing
- `sysinfo`: Memory monitoring
- All `scirs2-*` modules

### Python Dependencies (Optional)

For Python comparison benchmarks:

```bash
pip install numpy scipy pandas matplotlib
```

### GPU Dependencies (Optional)

For GPU benchmarks:

```bash
# CUDA
# Install CUDA toolkit and drivers

# Metal
# Built-in on macOS

# WGPU
# Cross-platform, no additional setup needed
```

## Performance Targets Summary

### v0.2.0 Release Goals

| Category | Target | Status |
|----------|--------|--------|
| SIMD dot product (f32) | 8-12x speedup | ✓ |
| SIMD dot product (f64) | 4-6x speedup | ✓ |
| Matrix multiply | Competitive with BLAS | ✓ |
| 2-thread efficiency | >80% | ✓ |
| 4-thread efficiency | >60% | ✓ |
| Memory overhead | <10% | ✓ |
| vs NumPy (arrays) | 2-10x faster | ✓ |
| vs SciPy (linalg) | Competitive | ✓ |
| GPU matmul (large) | 5-50x speedup | ✓ |

## Continuous Integration

The benchmark suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions
- name: Run Performance Benchmarks
  run: |
    cd benches
    ./v020_run_all_benchmarks.sh --skip-python --skip-gpu

- name: Upload Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: /tmp/scirs2_v020_*
```

## Troubleshooting

### Out of Memory

If benchmarks exhaust memory:

```bash
# Run individual suites instead of full suite
cargo bench --bench v020_comprehensive_suite
# Then run others individually
```

### Python Dependencies Missing

```bash
pip install -r requirements.txt
# Or skip Python benchmarks:
./v020_run_all_benchmarks.sh --skip-python
```

### GPU Benchmarks Failing

```bash
# Skip GPU benchmarks if no GPU available:
./v020_run_all_benchmarks.sh --skip-gpu
```

### Slow Benchmark Execution

```bash
# Use quick mode (reduced measurement time):
./v020_run_all_benchmarks.sh --quick
```

## Contributing

### Adding New Benchmarks

1. Create benchmark file in `benches/`
2. Add entry to `benches/Cargo.toml`:
   ```toml
   [[bench]]
   name = "my_benchmark"
   path = "my_benchmark.rs"
   harness = false
   ```
3. Follow existing structure and naming conventions
4. Update this documentation

### Benchmark Design Guidelines

- Use consistent random seeds for reproducibility
- Include both performance and accuracy verification
- Provide meaningful progress output
- Save results in JSON format for analysis
- Include error handling for edge cases
- Use representative data sizes
- Warm up before measurement
- Measure multiple iterations

## License

This benchmarking suite is part of SciRS2 and is licensed under Apache-2.0.

## Support

For issues or questions:
- GitHub Issues: https://github.com/cool-japan/scirs
- Documentation: https://docs.rs/scirs2

## Version History

### v0.2.0 (Current)
- Initial comprehensive benchmark suite
- SIMD vs scalar comparison
- GPU vs CPU comparison
- Memory profiling
- Scalability analysis
- Python comparison (NumPy, SciPy)
- Automated report generation

### Future Enhancements
- PyTorch comparison for neural operations
- Distributed computing benchmarks
- Energy efficiency measurements
- Hardware-specific optimizations
- Automated regression testing
- Performance dashboard
