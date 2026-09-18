"""
Minimal Benchmark Suite - Ultra-Fast Overview

Tests only the smallest sizes to complete quickly.
"""

import numpy as np
import scipy.linalg
import scipy.stats
from sklearn.cluster import KMeans as SKLearnKMeans
import scirs2
import time
import pandas as pd


def minimal_bench(name, scipy_func, scirs2_func, data, n_runs=5):
    """Minimal benchmark"""
    try:
        # Single warmup
        _ = scipy_func(data)
        _ = scirs2_func(data)

        # Quick timing
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = scipy_func(data)
        scipy_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = scirs2_func(data)
        scirs2_time = (time.perf_counter() - t0) / n_runs

        speedup = scipy_time / scirs2_time

        return {
            'category': name.split('_')[0],
            'operation': name,
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'scipy'
        }
    except Exception as e:
        print(f"  SKIP {name}: {str(e)[:50]}")
        return None


def main():
    print("="*80)
    print("MINIMAL BENCHMARK SUITE")
    print("="*80)
    print("\nTesting small data sizes for quick overview...\n")

    results = []
    np.random.seed(42)

    # === LINEAR ALGEBRA (size 50) ===
    print("Linear Algebra (50x50):")
    A = np.ascontiguousarray(np.random.randn(50, 50) + np.eye(50) * 5)
    b = np.ascontiguousarray(np.random.randn(50))

    for op_name, scipy_f, scirs2_f, data in [
        ('linalg_det', lambda x: scipy.linalg.det(x), lambda x: scirs2.det_py(x), A),
        ('linalg_inv', lambda x: scipy.linalg.inv(x), lambda x: scirs2.inv_py(x), A),
        ('linalg_solve', lambda _: scipy.linalg.solve(A, b), lambda _: scirs2.solve_py(A, b), None),
        ('linalg_qr', lambda x: scipy.linalg.qr(x), lambda x: scirs2.qr_py(x), A),
    ]:
        res = minimal_bench(op_name, scipy_f, scirs2_f, data)
        if res:
            results.append(res)
            print(f"  {op_name.split('_')[1]}: {res['speedup']:.2f}x")

    # === STATISTICS (size 1000) ===
    print("\nStatistics (1000 elements):")
    data = np.ascontiguousarray(np.random.randn(1000))
    x = np.ascontiguousarray(np.random.randn(1000))
    y = np.ascontiguousarray(2 * x + np.random.randn(1000) * 0.5)

    for op_name, scipy_f, scirs2_f, d in [
        ('stats_mean', lambda x: np.mean(x), lambda x: scirs2.mean_py(x), data),
        ('stats_std', lambda x: np.std(x, ddof=1), lambda x: scirs2.std_py(x, ddof=1), data),
        ('stats_median', lambda x: np.median(x), lambda x: scirs2.median_py(x), data),
        ('stats_skewness', lambda x: scipy.stats.skew(x), lambda x: scirs2.skew_py(x), data),
        ('stats_kurtosis', lambda x: scipy.stats.kurtosis(x), lambda x: scirs2.kurtosis_py(x), data),
        ('stats_pearsonr', lambda _: scipy.stats.pearsonr(x, y), lambda _: scirs2.pearsonr_py(x, y), None),
    ]:
        res = minimal_bench(op_name, scipy_f, scirs2_f, d)
        if res:
            results.append(res)
            print(f"  {op_name.split('_')[1]}: {res['speedup']:.2f}x")

    # === FFT (size 1024) ===
    print("\nFFT (1024 elements):")
    signal_real = np.ascontiguousarray(np.random.randn(1024))
    signal_complex = np.ascontiguousarray(np.random.randn(1024) + 1j * np.random.randn(1024))

    for op_name, scipy_f, scirs2_f, d in [
        ('fft_fft', lambda x: np.fft.fft(x), lambda x: scirs2.fft_py(x), signal_complex),
        ('fft_rfft', lambda x: np.fft.rfft(x), lambda x: scirs2.rfft_py(x), signal_real),
    ]:
        res = minimal_bench(op_name, scipy_f, scirs2_f, d, n_runs=10)
        if res:
            results.append(res)
            print(f"  {op_name.split('_')[1]}: {res['speedup']:.2f}x")

    # === CLUSTERING (500x5, k=3) ===
    print("\nClustering (500x5, k=3):")
    X = np.ascontiguousarray(np.random.randn(500, 5))

    res = minimal_bench('cluster_kmeans',
                       lambda x: SKLearnKMeans(n_clusters=3, random_state=42, n_init=10).fit(x),
                       lambda x: scirs2.kmeans_py(x, n_clusters=3, random_state=42),
                       X, n_runs=3)
    if res:
        results.append(res)
        print(f"  kmeans: {res['speedup']:.2f}x")

    # Create summary
    if not results:
        print("\nNo results collected!")
        return

    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")

    total = len(df)
    overall_speedup = df['speedup'].mean()
    scirs2_wins = (df['winner'] == 'scirs2').sum()

    print(f"Total Tests: {total}")
    print(f"Overall Average Speedup: {overall_speedup:.2f}x")
    print(f"SciRS2 Wins: {scirs2_wins}/{total} ({scirs2_wins/total*100:.1f}%)\n")

    # Category breakdown
    category_summary = df.groupby('category').agg({
        'speedup': ['mean', 'count']
    }).round(2)
    category_summary.columns = ['avg_speedup', 'tests']

    print("By Category:")
    print(category_summary.to_string())

    # Save results
    output_file = '/tmp/bench_minimal_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}\n")
    print(df[['operation', 'scipy_ms', 'scirs2_ms', 'speedup', 'winner']].to_string(index=False))

    # Markdown report
    md_file = '/tmp/bench_minimal_report.md'
    with open(md_file, 'w') as f:
        f.write("# Minimal Performance Benchmark - Quick Overview\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Tests**: {total}\n")
        f.write(f"- **Overall Average Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"- **SciRS2 Win Rate**: {scirs2_wins/total*100:.1f}%\n\n")
        f.write("## By Category\n\n")
        f.write(category_summary.to_markdown())
        f.write("\n\n## Detailed Results\n\n")
        f.write(df[['operation', 'scipy_ms', 'scirs2_ms', 'speedup', 'winner']].to_markdown(index=False))

    print(f"\nMarkdown report saved to: {md_file}")

    return df


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f} seconds")
