"""
Quick Benchmark Suite - Fast Performance Overview

Runs representative benchmarks across all modules without exhaustive testing.
Designed to complete in under 5 minutes.
"""

import numpy as np
import scipy.linalg
import scipy.stats
from sklearn.cluster import KMeans as SKLearnKMeans
import scirs2
import time
import pandas as pd


def quick_benchmark(name, scipy_func, scirs2_func, data, n_runs=10):
    """Quick benchmark of a single operation"""
    # Warmup
    try:
        _ = scipy_func(data)
        _ = scirs2_func(data)
    except Exception as e:
        return None

    # Benchmark
    scipy_times = [time.perf_counter(), scipy_func(data), time.perf_counter()]
    scipy_time = scipy_times[2] - scipy_times[0]

    scirs2_times = [time.perf_counter(), scirs2_func(data), time.perf_counter()]
    scirs2_time = scirs2_times[2] - scirs2_times[0]

    # Run multiple times for better estimate
    scipy_total = 0
    scirs2_total = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = scipy_func(data)
        scipy_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = scirs2_func(data)
        scirs2_total += time.perf_counter() - t0

    scipy_time = scipy_total / n_runs
    scirs2_time = scirs2_total / n_runs

    return {
        'operation': name,
        'scipy_ms': scipy_time * 1000,
        'scirs2_ms': scirs2_time * 1000,
        'speedup': scipy_time / scirs2_time,
        'winner': 'scirs2' if scipy_time > scirs2_time else 'scipy'
    }


def main():
    print("="*80)
    print("QUICK BENCHMARK SUITE")
    print("="*80)
    print("\nRunning representative benchmarks from each module...")
    print("This should complete in under 5 minutes.\n")

    results = []
    np.random.seed(42)

    # === LINEAR ALGEBRA ===
    print("Linear Algebra:")

    A = np.ascontiguousarray(np.random.randn(100, 100))
    res = quick_benchmark("det_100x100", lambda x: scipy.linalg.det(x),
                          lambda x: scirs2.det_py(x), A)
    if res:
        results.append(res)
        print(f"  det (100x100): {res['speedup']:.2f}x")

    res = quick_benchmark("inv_100x100", lambda x: scipy.linalg.inv(x),
                          lambda x: scirs2.inv_py(x), A)
    if res:
        results.append(res)
        print(f"  inv (100x100): {res['speedup']:.2f}x")

    b = np.ascontiguousarray(np.random.randn(100))
    res = quick_benchmark("solve_100", lambda _: scipy.linalg.solve(A, b),
                          lambda _: scirs2.solve_py(A, b), None)
    if res:
        results.append(res)
        print(f"  solve (100): {res['speedup']:.2f}x")

    res = quick_benchmark("qr_100x100", lambda x: scipy.linalg.qr(x),
                          lambda x: scirs2.qr_py(x), A)
    if res:
        results.append(res)
        print(f"  qr (100x100): {res['speedup']:.2f}x")

    res = quick_benchmark("svd_100x100", lambda x: scipy.linalg.svd(x),
                          lambda x: scirs2.svd_py(x), A)
    if res:
        results.append(res)
        print(f"  svd (100x100): {res['speedup']:.2f}x")

    # === STATISTICS ===
    print("\nStatistics:")

    data = np.ascontiguousarray(np.random.randn(10000))

    res = quick_benchmark("mean_10k", lambda x: np.mean(x),
                          lambda x: scirs2.mean_py(x), data)
    if res:
        results.append(res)
        print(f"  mean (10k): {res['speedup']:.2f}x")

    res = quick_benchmark("std_10k", lambda x: np.std(x, ddof=1),
                          lambda x: scirs2.std_py(x, ddof=1), data)
    if res:
        results.append(res)
        print(f"  std (10k): {res['speedup']:.2f}x")

    res = quick_benchmark("median_10k", lambda x: np.median(x),
                          lambda x: scirs2.median_py(x), data)
    if res:
        results.append(res)
        print(f"  median (10k): {res['speedup']:.2f}x")

    res = quick_benchmark("skewness_10k", lambda x: scipy.stats.skew(x),
                          lambda x: scirs2.skew_py(x), data)
    if res:
        results.append(res)
        print(f"  skewness (10k): {res['speedup']:.2f}x")

    res = quick_benchmark("kurtosis_10k", lambda x: scipy.stats.kurtosis(x),
                          lambda x: scirs2.kurtosis_py(x), data)
    if res:
        results.append(res)
        print(f"  kurtosis (10k): {res['speedup']:.2f}x")

    x = np.ascontiguousarray(np.random.randn(1000))
    y = np.ascontiguousarray(2 * x + np.random.randn(1000) * 0.5)
    res = quick_benchmark("pearsonr_1k", lambda _: scipy.stats.pearsonr(x, y),
                          lambda _: scirs2.pearsonr_py(x, y), None)
    if res:
        results.append(res)
        print(f"  pearsonr (1k): {res['speedup']:.2f}x")

    # === FFT ===
    print("\nFFT:")

    signal_real = np.ascontiguousarray(np.random.randn(8192))
    signal_complex = np.ascontiguousarray(np.random.randn(8192) + 1j * np.random.randn(8192))

    res = quick_benchmark("fft_8k", lambda x: np.fft.fft(x),
                          lambda x: scirs2.fft_py(x), signal_complex, n_runs=20)
    if res:
        results.append(res)
        print(f"  fft (8k): {res['speedup']:.2f}x")

    res = quick_benchmark("rfft_8k", lambda x: np.fft.rfft(x),
                          lambda x: scirs2.rfft_py(x), signal_real, n_runs=20)
    if res:
        results.append(res)
        print(f"  rfft (8k): {res['speedup']:.2f}x")

    # === CLUSTERING ===
    print("\nClustering:")

    X = np.ascontiguousarray(np.random.randn(1000, 10))
    res = quick_benchmark("kmeans_1kx10",
                          lambda x: SKLearnKMeans(n_clusters=5, random_state=42, n_init=10).fit(x),
                          lambda x: scirs2.kmeans_py(x, n_clusters=5, random_state=42),
                          X, n_runs=3)
    if res:
        results.append(res)
        print(f"  kmeans (1k×10, k=5): {res['speedup']:.2f}x")

    # Create quick results report
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    overall_speedup = df['speedup'].mean()
    scirs2_wins = (df['winner'] == 'scirs2').sum()
    total = len(df)

    print(f"\n**Overall Average Speedup**: {overall_speedup:.2f}x")
    print(f"**SciRS2 Wins**: {scirs2_wins}/{total} ({scirs2_wins/total*100:.1f}%)")

    # Category breakdown
    df['category'] = df['operation'].str.extract(r'^([a-z_]+)')[0]
    category_summary = df.groupby('category').agg({
        'speedup': ['mean', 'count']
    }).round(2)
    print(f"\n{'='*80}")
    print("BY CATEGORY")
    print(f"{'='*80}\n")
    print(category_summary.to_string())

    # Save results
    output_csv = '/tmp/bench_quick_suite_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # Create markdown report
    md_file = '/tmp/bench_quick_suite_report.md'
    with open(md_file, 'w') as f:
        f.write("# SciRS2-Python Quick Benchmark Report\n\n")
        f.write("## Overview\n\n")
        f.write("Representative performance benchmarks across all major modules.\n\n")
        f.write(f"- **Total Tests**: {total}\n")
        f.write(f"- **Overall Average Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"- **SciRS2 Win Rate**: {scirs2_wins/total*100:.1f}%\n\n")
        f.write("## Results\n\n")
        f.write(df[['operation', 'scipy_ms', 'scirs2_ms', 'speedup', 'winner']].to_markdown(index=False))
        f.write("\n\n## Category Performance\n\n")
        f.write(category_summary.to_markdown())

    print(f"Markdown report saved to: {md_file}")

    return df


if __name__ == '__main__':
    main()
