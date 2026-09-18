#!/usr/bin/env python3
"""
Quick Performance Benchmark

Fast benchmark to get initial performance comparison.
Tests representative operations at moderate sizes.
"""

import numpy as np
import scipy.linalg
import scipy.stats
import scirs2
import time


def time_operation(func, *args, iterations=5):
    """Time an operation with multiple iterations"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000  # Return median time in ms


def main():
    print("=" * 80)
    print("Quick Performance Benchmark: scirs2 vs SciPy")
    print("=" * 80)
    print()

    results = []

    # Linear Algebra Benchmarks
    print("Linear Algebra (100x100 matrices)")
    print("-" * 80)

    A = np.ascontiguousarray(np.random.randn(100, 100))
    b = np.ascontiguousarray(np.random.randn(100))

    # Determinant
    scipy_time = time_operation(scipy.linalg.det, A)
    scirs2_time = time_operation(scirs2.det_py, A)
    ratio = scirs2_time / scipy_time
    results.append(("det", scipy_time, scirs2_time, ratio))
    print(f"  Determinant:    SciPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Matrix Inverse
    scipy_time = time_operation(scipy.linalg.inv, A)
    scirs2_time = time_operation(scirs2.inv_py, A)
    ratio = scirs2_time / scipy_time
    results.append(("inv", scipy_time, scirs2_time, ratio))
    print(f"  Inverse:        SciPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Linear Solve
    scipy_time = time_operation(lambda: scipy.linalg.solve(A, b))
    scirs2_time = time_operation(lambda: scirs2.solve_py(A, b))
    ratio = scirs2_time / scipy_time
    results.append(("solve", scipy_time, scirs2_time, ratio))
    print(f"  Solve:          SciPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # SVD
    scipy_time = time_operation(scipy.linalg.svd, A, iterations=3)
    scirs2_time = time_operation(scirs2.svd_py, A, iterations=3)
    ratio = scirs2_time / scipy_time
    results.append(("svd", scipy_time, scirs2_time, ratio))
    print(f"  SVD:            SciPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Statistics Benchmarks
    print("\nStatistics (100,000 elements)")
    print("-" * 80)

    data = np.ascontiguousarray(np.random.randn(100000))

    # Mean
    scipy_time = time_operation(np.mean, data, iterations=10)
    scirs2_time = time_operation(scirs2.mean_py, data, iterations=10)
    ratio = scirs2_time / scipy_time
    results.append(("mean", scipy_time, scirs2_time, ratio))
    print(f"  Mean:           NumPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Std
    scipy_time = time_operation(lambda: np.std(data, ddof=1), iterations=10)
    scirs2_time = time_operation(lambda: scirs2.std_py(data, ddof=1), iterations=10)
    ratio = scirs2_time / scipy_time
    results.append(("std", scipy_time, scirs2_time, ratio))
    print(f"  Std Dev:        NumPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Median
    scipy_time = time_operation(np.median, data, iterations=5)
    scirs2_time = time_operation(scirs2.median_py, data, iterations=5)
    ratio = scirs2_time / scipy_time
    results.append(("median", scipy_time, scirs2_time, ratio))
    print(f"  Median:         NumPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # FFT Benchmarks
    print("\nFFT (8192 points)")
    print("-" * 80)

    signal = np.ascontiguousarray(np.random.randn(8192))

    # FFT
    scipy_time = time_operation(np.fft.fft, signal, iterations=10)
    scirs2_time = time_operation(scirs2.fft_py, signal, iterations=10)
    ratio = scirs2_time / scipy_time
    results.append(("fft", scipy_time, scirs2_time, ratio))
    print(f"  FFT:            NumPy {scipy_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Clustering Benchmarks
    print("\nClustering (1000 samples, 10 features)")
    print("-" * 80)

    from sklearn.cluster import KMeans as SKLearnKMeans
    X = np.ascontiguousarray(np.random.randn(1000, 10))

    # KMeans
    def sklearn_kmeans():
        km = SKLearnKMeans(n_clusters=3, random_state=42, n_init=1, max_iter=10)
        km.fit(X)

    def scirs2_kmeans():
        km = scirs2.KMeans(n_clusters=3, random_state=42, max_iter=10)
        km.fit(X)

    sklearn_time = time_operation(sklearn_kmeans, iterations=5)
    scirs2_time = time_operation(scirs2_kmeans, iterations=5)
    ratio = scirs2_time / sklearn_time
    results.append(("kmeans", sklearn_time, scirs2_time, ratio))
    print(f"  KMeans:         SKLearn {sklearn_time:6.2f}ms | SciRS2 {scirs2_time:6.2f}ms | Ratio: {ratio:.2f}x")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    avg_ratio = np.mean([r[3] for r in results])
    print(f"\nAverage Performance: {avg_ratio:.2f}x slower than baseline")

    # Categorize
    good = sum(1 for r in results if r[3] < 2.0)
    acceptable = sum(1 for r in results if 2.0 <= r[3] < 5.0)
    slow = sum(1 for r in results if r[3] >= 5.0)

    print(f"  ✅ Good (<2x):       {good}/{len(results)}")
    print(f"  🟡 Acceptable (2-5x): {acceptable}/{len(results)}")
    print(f"  ❌ Slow (>5x):       {slow}/{len(results)}")

    if avg_ratio < 2.0:
        print(f"\n🎉 Overall Status: EXCELLENT - Within 2x target!")
    elif avg_ratio < 3.0:
        print(f"\n✅ Overall Status: GOOD - Within 3x")
    elif avg_ratio < 5.0:
        print(f"\n🟡 Overall Status: ACCEPTABLE - Within 5x")
    else:
        print(f"\n❌ Overall Status: NEEDS OPTIMIZATION")

    return results


if __name__ == '__main__':
    results = main()
