# scirs2-stats Benchmarks

This directory contains comprehensive benchmarks for the scirs2-stats module, including comparisons with SciPy.

## Running Benchmarks

### Rust Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark suite:
```bash
cargo bench --bench distributions
cargo bench --bench statistical_tests
cargo bench --bench scipy_comparison
```

### Python/SciPy Benchmarks

Run the SciPy comparison benchmarks:
```bash
python3 benches/scipy_benchmark.py
```

### Comparing Results

To compare Rust and Python performance:
```bash
# First run the Rust benchmarks and save output
cargo bench > rust_benchmarks.txt

# Run Python benchmarks
python3 benches/scipy_benchmark.py

# Compare results
python3 benches/compare_benchmarks.py rust_benchmarks.txt scipy_benchmark_results.json
```

## Benchmark Suites

### 1. Distributions (`distributions.rs`)
- PDF/PMF calculations
- CDF calculations
- Random number generation
- Statistical moments (mean, variance)
- Tests various distribution types and sample sizes

### 2. Statistical Tests (`statistical_tests.rs`)
- T-tests (one-sample, independent, paired)
- Non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Correlation functions (Pearson, Spearman, Kendall)
- ANOVA tests

### 3. SciPy Comparison (`scipy_comparison.rs`)
- Descriptive statistics on large datasets
- Quantile operations
- Correlation matrix calculations
- Memory-intensive operations
- Parallel operation opportunities

## Performance Optimization Tips

When running benchmarks for accurate comparison:

1. **Disable CPU frequency scaling**:
   ```bash
   sudo cpupower frequency-set --governor performance
   ```

2. **Run with release optimizations**:
   ```bash
   cargo bench --release
   ```

3. **Minimize system load**:
   - Close unnecessary applications
   - Disable background services
   - Run benchmarks multiple times and average results

4. **Use consistent hardware**:
   - Run both Rust and Python benchmarks on the same machine
   - Ensure thermal throttling is not affecting results

## Interpreting Results

### Criterion Output
- **time**: Average execution time per iteration
- **thrpt**: Throughput (operations per second)
- **R²**: Coefficient of determination (higher is better, indicates consistent measurements)

### Performance Metrics
- **PDF/CDF Performance**: Critical for Monte Carlo simulations
- **RNG Performance**: Important for sampling-based methods
- **Statistical Test Performance**: Affects interactive data analysis
- **Memory Bandwidth**: Important for large dataset operations

## Expected Performance Characteristics

Based on the implementation strategy:

1. **Distribution Operations**: Should be competitive with SciPy due to:
   - Efficient Rust implementations
   - SIMD optimizations from scirs2-core
   - Cache-friendly data structures

2. **Statistical Tests**: May show significant speedup due to:
   - Zero-copy operations
   - Optimized sorting algorithms
   - Parallel implementations where applicable

3. **Large Dataset Operations**: Should excel due to:
   - Memory-efficient algorithms
   - SIMD vectorization
   - Parallel processing capabilities

## Adding New Benchmarks

To add a new benchmark:

1. Create a new function in the appropriate benchmark file
2. Use `criterion::BenchmarkGroup` for related benchmarks
3. Test various input sizes to understand scaling behavior
4. Add corresponding Python benchmark for comparison
5. Update this README with the new benchmark description

## Continuous Benchmarking

For tracking performance over time:

1. Save benchmark results:
   ```bash
   cargo bench -- --save-baseline <name>
   ```

2. Compare against baseline:
   ```bash
   cargo bench -- --baseline <name>
   ```

3. Use CI/CD to detect performance regressions