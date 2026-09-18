# SciRS2-Optimize Benchmarking Suite

This directory contains comprehensive benchmarks comparing SciRS2-Optimize with SciPy's optimize module.

## Overview

The benchmarking suite tests various optimization algorithms on standard test problems to compare:
- Execution speed
- Iteration counts
- Convergence reliability
- Accuracy of solutions
- Scaling with problem size

## Test Problems

The suite includes the following standard optimization test functions:

### Unconstrained Optimization
- **Rosenbrock**: Classic banana-shaped valley (2D)
- **Sphere**: Simple quadratic function (N-D)
- **Rastrigin**: Highly multimodal function (N-D)
- **Ackley**: Multimodal with a global funnel (N-D)
- **Beale**: Multiple local minima (2D)
- **Himmelblau**: Four equal local minima (2D)
- **Levy**: Multimodal with regular structure (N-D)

### Methods Tested
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- L-BFGS (Limited-memory BFGS)
- CG (Conjugate Gradient)
- Nelder-Mead (Simplex method)
- Powell (Direction set method)
- Differential Evolution (Global optimization)
- Basin-hopping (Global optimization)

## Running the Benchmarks

### Full Comparison Suite

Run the complete benchmark comparison:

```bash
./run_full_comparison.sh
```

This will:
1. Run Rust benchmarks using Criterion
2. Run Python/SciPy benchmarks
3. Generate comparison plots and reports

### Individual Components

#### Rust Benchmarks Only
```bash
cargo bench --bench scipy_comparison_bench
```

#### Python/SciPy Benchmarks Only
```bash
python3 scipy_benchmarks.py
```

#### Analysis Only (requires existing results)
```bash
python3 analyze_comparison.py
```

## Output Files

- `scipy_benchmark_results.json`: SciPy benchmark results
- `scirs_benchmark_results.json`: SciRS2 benchmark results  
- `performance_comparison.png`: Visual performance comparison
- `comparison_report.md`: Detailed comparison report

## Dependencies

### Rust
- criterion (for benchmarking)
- scirs2-optimize and all its features

### Python
```bash
pip install numpy scipy pandas matplotlib seaborn
```

## Interpreting Results

### Performance Metrics
- **Speedup**: Ratio of SciPy time to SciRS2 time
  - \> 1.0: SciRS2 is faster
  - < 1.0: SciPy is faster
  - ≈ 1.0: Similar performance

### Accuracy Metrics
- **Absolute Difference**: |f_scipy - f_scirs|
- **Relative Difference**: |f_scipy - f_scirs| / |f_scipy|
- **Acceptable**: Absolute difference < 1e-6

### Success Rates
- Percentage of runs that converged successfully
- Should be similar between implementations

## Customizing Benchmarks

### Adding New Test Functions

1. Add to `test_functions` module in both Rust and Python
2. Update `get_benchmark_problems()` in both files
3. Ensure consistent initial points and bounds

### Adding New Methods

1. Add method to benchmark loops
2. Ensure method exists in both implementations
3. Map method names between Rust and Python if different

### Adjusting Parameters

Edit the following in the benchmark files:
- `sample_size`: Number of benchmark runs (default: 50)
- `measurement_time`: Time per benchmark (default: 10s)
- Problem dimensions in `dimensions` array
- Population sizes for global methods

## Performance Tips

1. **Warm-up**: First run may be slower due to compilation
2. **Background Processes**: Close other applications for consistent results
3. **Power Settings**: Use performance power profile
4. **Multiple Runs**: Results are averaged over multiple runs

## Known Limitations

1. Some SciPy methods may not have exact SciRS2 equivalents
2. Global optimization methods use randomization (results may vary)
3. Criterion benchmarks measure wall-clock time (includes overhead)
4. Python benchmarks include interpreter overhead

## Contributing

When adding new benchmarks:
1. Ensure fair comparison (same initial points, tolerances, etc.)
2. Document any implementation differences
3. Test on multiple platforms if possible
4. Update this README with new additions