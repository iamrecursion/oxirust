"""
Statistics Performance Benchmarks

Compares scirs2 vs SciPy/NumPy for statistical operations.
"""

import numpy as np
import scipy.stats
import scirs2
import time
import pandas as pd
from typing import List, Dict


def benchmark_stat_operation(
    name: str,
    scipy_func,
    scirs2_func,
    sizes: List[int],
    n_runs: int = 50
) -> List[Dict]:
    """Benchmark a statistical operation"""
    results = []

    for size in sizes:
        print(f"  Testing {name} with size {size}...", end="", flush=True)

        # Generate test data
        np.random.seed(42)
        data = np.ascontiguousarray(np.random.randn(size))

        # Warmup
        try:
            _ = scipy_func(data)
            _ = scirs2_func(data)
        except Exception as e:
            print(f" SKIP ({e})")
            continue

        # Benchmark SciPy
        scipy_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = scipy_func(data)
            scipy_times.append(time.perf_counter() - start)
        scipy_time = np.median(scipy_times)

        # Benchmark SciRS2
        scirs2_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = scirs2_func(data)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = scipy_time / scirs2_time
        print(f" ✓ (speedup: {speedup:.2f}x)")

        results.append({
            'operation': name,
            'size': size,
            'scipy_us': scipy_time * 1_000_000,  # microseconds
            'scirs2_us': scirs2_time * 1_000_000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'scipy'
        })

    return results


def main():
    """Run all statistics benchmarks"""
    print("="*80)
    print("STATISTICS BENCHMARKS")
    print("="*80)
    print()

    all_results = []
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    # Basic descriptive statistics
    print("Benchmarking: Mean")
    all_results.extend(benchmark_stat_operation(
        "mean",
        lambda x: np.mean(x),
        lambda x: scirs2.mean_py(x),
        sizes
    ))

    print("Benchmarking: Std")
    all_results.extend(benchmark_stat_operation(
        "std",
        lambda x: np.std(x, ddof=1),
        lambda x: scirs2.std_py(x, ddof=1),
        sizes
    ))

    print("Benchmarking: Var")
    all_results.extend(benchmark_stat_operation(
        "var",
        lambda x: np.var(x, ddof=1),
        lambda x: scirs2.var_py(x, ddof=1),
        sizes
    ))

    print("Benchmarking: Median")
    all_results.extend(benchmark_stat_operation(
        "median",
        lambda x: np.median(x),
        lambda x: scirs2.median_py(x),
        sizes
    ))

    # SIMD-optimized operations
    print("Benchmarking: Mean SIMD")
    all_results.extend(benchmark_stat_operation(
        "mean_simd",
        lambda x: np.mean(x),
        lambda x: scirs2.mean_simd_py(x),
        sizes
    ))

    print("Benchmarking: Std SIMD")
    all_results.extend(benchmark_stat_operation(
        "std_simd",
        lambda x: np.std(x, ddof=1),
        lambda x: scirs2.std_simd_py(x, ddof=1),
        sizes
    ))

    print("Benchmarking: Var SIMD")
    all_results.extend(benchmark_stat_operation(
        "var_simd",
        lambda x: np.var(x, ddof=1),
        lambda x: scirs2.variance_simd_py(x, ddof=1),
        sizes
    ))

    # Higher moments
    print("Benchmarking: Skewness")
    all_results.extend(benchmark_stat_operation(
        "skewness",
        lambda x: scipy.stats.skew(x),
        lambda x: scirs2.skew_py(x),
        sizes
    ))

    print("Benchmarking: Kurtosis")
    all_results.extend(benchmark_stat_operation(
        "kurtosis",
        lambda x: scipy.stats.kurtosis(x),
        lambda x: scirs2.kurtosis_py(x),
        sizes
    ))

    print("Benchmarking: Skewness SIMD")
    all_results.extend(benchmark_stat_operation(
        "skewness_simd",
        lambda x: scipy.stats.skew(x),
        lambda x: scirs2.skewness_simd_py(x),
        sizes
    ))

    print("Benchmarking: Kurtosis SIMD")
    all_results.extend(benchmark_stat_operation(
        "kurtosis_simd",
        lambda x: scipy.stats.kurtosis(x),
        lambda x: scirs2.kurtosis_simd_py(x),
        sizes
    ))

    # Percentiles
    print("Benchmarking: Percentile (50th)")
    all_results.extend(benchmark_stat_operation(
        "percentile_50",
        lambda x: np.percentile(x, 50),
        lambda x: scirs2.percentile_py(x, 50),
        sizes
    ))

    print("Benchmarking: IQR")
    all_results.extend(benchmark_stat_operation(
        "iqr",
        lambda x: scipy.stats.iqr(x),
        lambda x: scirs2.iqr_py(x),
        sizes
    ))

    # Correlation (2D operations)
    print("\nBenchmarking 2D Operations:")
    correlation_results = []

    for size in [100, 1_000, 10_000]:
        print(f"  Testing correlation with size {size}...", end="", flush=True)

        np.random.seed(42)
        x = np.ascontiguousarray(np.random.randn(size))
        y = np.ascontiguousarray(2 * x + np.random.randn(size) * 0.5)

        # Benchmark SciPy
        scipy_times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = scipy.stats.pearsonr(x, y)
            scipy_times.append(time.perf_counter() - start)
        scipy_time = np.median(scipy_times)

        # Benchmark SciRS2
        scirs2_times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = scirs2.pearsonr_py(x, y)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = scipy_time / scirs2_time
        print(f" ✓ (speedup: {speedup:.2f}x)")

        correlation_results.append({
            'operation': 'pearsonr',
            'size': size,
            'scipy_us': scipy_time * 1_000_000,
            'scirs2_us': scirs2_time * 1_000_000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'scipy'
        })

    all_results.extend(correlation_results)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    summary = df.groupby('operation').agg({
        'speedup': ['mean', 'min', 'max'],
        'winner': lambda x: (x == 'scirs2').sum() / len(x) * 100
    }).round(2)
    summary.columns = ['avg_speedup', 'min_speedup', 'max_speedup', 'scirs2_win_rate_%']
    print(summary.to_string())

    overall_speedup = df['speedup'].mean()
    scirs2_wins = (df['winner'] == 'scirs2').sum()
    total_tests = len(df)

    print(f"\n**Overall Average Speedup**: {overall_speedup:.2f}x")
    print(f"**SciRS2 Wins**: {scirs2_wins}/{total_tests} ({scirs2_wins/total_tests*100:.1f}%)")

    # SIMD comparison
    simd_ops = df[df['operation'].str.contains('simd')]
    non_simd = df[~df['operation'].str.contains('simd') & df['operation'].isin(['mean', 'std', 'var', 'skewness', 'kurtosis'])]

    if not simd_ops.empty and not non_simd.empty:
        print("\n" + "="*80)
        print("SIMD ACCELERATION")
        print("="*80)
        simd_speedup = simd_ops['speedup'].mean()
        non_simd_speedup = non_simd['speedup'].mean()
        print(f"SIMD Operations Average Speedup: {simd_speedup:.2f}x")
        print(f"Non-SIMD Operations Average Speedup: {non_simd_speedup:.2f}x")
        print(f"SIMD Advantage: {simd_speedup / non_simd_speedup:.2f}x faster")

    # Save results
    output_file = '/tmp/bench_stats_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create markdown report
    md_file = '/tmp/bench_stats_report.md'
    with open(md_file, 'w') as f:
        f.write("# Statistics Performance Benchmarks\n\n")
        f.write("## Test Configuration\n\n")
        f.write(f"- **Test Sizes**: {sizes}\n")
        f.write(f"- **Runs per Test**: 50 (median time)\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")
        f.write(summary.to_markdown())
        f.write(f"\n\n**Overall Average Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"**SciRS2 Win Rate**: {scirs2_wins/total_tests*100:.1f}%\n")

        if not simd_ops.empty:
            f.write("\n## SIMD Acceleration\n\n")
            f.write(f"SIMD Operations Average Speedup: {simd_speedup:.2f}x\n")
            f.write(f"Non-SIMD Operations Average Speedup: {non_simd_speedup:.2f}x\n")
            f.write(f"SIMD Advantage: {simd_speedup / non_simd_speedup:.2f}x faster\n")

    print(f"Markdown report saved to: {md_file}")

    return df


if __name__ == '__main__':
    main()
