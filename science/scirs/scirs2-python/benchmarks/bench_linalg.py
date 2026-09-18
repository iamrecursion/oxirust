#!/usr/bin/env python3
"""
Performance Benchmarks: Linear Algebra Module

Compares scirs2 linear algebra performance against SciPy/NumPy.
Target: Performance within 2x of SciPy for most operations.
"""

import numpy as np
import scipy.linalg
import scirs2
import time
import pandas as pd


def benchmark_determinant():
    """Benchmark determinant calculation"""
    sizes = [10, 50, 100, 200, 500]
    results = []

    for n in sizes:
        A = np.ascontiguousarray(np.random.randn(n, n))

        # SciPy baseline
        times_scipy = []
        for _ in range(10):
            start = time.perf_counter()
            scipy.linalg.det(A)
            times_scipy.append(time.perf_counter() - start)
        scipy_time = np.median(times_scipy)

        # SciRS2
        times_scirs2 = []
        for _ in range(10):
            start = time.perf_counter()
            scirs2.det_py(A)
            times_scirs2.append(time.perf_counter() - start)
        scirs2_time = np.median(times_scirs2)

        results.append({
            'operation': 'determinant',
            'size': f'{n}x{n}',
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': scipy_time / scirs2_time,
            'ratio': scirs2_time / scipy_time
        })

    return pd.DataFrame(results)


def benchmark_matrix_inverse():
    """Benchmark matrix inversion"""
    sizes = [10, 50, 100, 200]
    results = []

    for n in sizes:
        A = np.ascontiguousarray(np.random.randn(n, n) + np.eye(n) * 5)

        # SciPy baseline
        times_scipy = []
        for _ in range(10):
            start = time.perf_counter()
            scipy.linalg.inv(A)
            times_scipy.append(time.perf_counter() - start)
        scipy_time = np.median(times_scipy)

        # SciRS2
        times_scirs2 = []
        for _ in range(10):
            start = time.perf_counter()
            scirs2.inv_py(A)
            times_scirs2.append(time.perf_counter() - start)
        scirs2_time = np.median(times_scirs2)

        results.append({
            'operation': 'inverse',
            'size': f'{n}x{n}',
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': scipy_time / scirs2_time,
            'ratio': scirs2_time / scipy_time
        })

    return pd.DataFrame(results)


def benchmark_linear_solve():
    """Benchmark linear system solving"""
    sizes = [10, 50, 100, 200, 500]
    results = []

    for n in sizes:
        A = np.ascontiguousarray(np.random.randn(n, n) + np.eye(n) * 5)
        b = np.ascontiguousarray(np.random.randn(n))

        # SciPy baseline
        times_scipy = []
        for _ in range(10):
            start = time.perf_counter()
            scipy.linalg.solve(A, b)
            times_scipy.append(time.perf_counter() - start)
        scipy_time = np.median(times_scipy)

        # SciRS2
        times_scirs2 = []
        for _ in range(10):
            start = time.perf_counter()
            scirs2.solve_py(A, b)
            times_scirs2.append(time.perf_counter() - start)
        scirs2_time = np.median(times_scirs2)

        results.append({
            'operation': 'solve',
            'size': f'{n}x{n}',
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': scipy_time / scirs2_time,
            'ratio': scirs2_time / scipy_time
        })

    return pd.DataFrame(results)


def benchmark_svd():
    """Benchmark SVD decomposition"""
    sizes = [10, 50, 100, 200]
    results = []

    for n in sizes:
        A = np.ascontiguousarray(np.random.randn(n, n))

        # SciPy baseline
        times_scipy = []
        for _ in range(5):
            start = time.perf_counter()
            scipy.linalg.svd(A)
            times_scipy.append(time.perf_counter() - start)
        scipy_time = np.median(times_scipy)

        # SciRS2
        times_scirs2 = []
        for _ in range(5):
            start = time.perf_counter()
            scirs2.svd_py(A)
            times_scirs2.append(time.perf_counter() - start)
        scirs2_time = np.median(times_scirs2)

        results.append({
            'operation': 'SVD',
            'size': f'{n}x{n}',
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': scipy_time / scirs2_time,
            'ratio': scirs2_time / scipy_time
        })

    return pd.DataFrame(results)


def benchmark_qr():
    """Benchmark QR decomposition"""
    sizes = [10, 50, 100, 200]
    results = []

    for n in sizes:
        A = np.ascontiguousarray(np.random.randn(n, n))

        # SciPy baseline
        times_scipy = []
        for _ in range(10):
            start = time.perf_counter()
            scipy.linalg.qr(A)
            times_scipy.append(time.perf_counter() - start)
        scipy_time = np.median(times_scipy)

        # SciRS2
        times_scirs2 = []
        for _ in range(10):
            start = time.perf_counter()
            scirs2.qr_py(A)
            times_scirs2.append(time.perf_counter() - start)
        scirs2_time = np.median(times_scirs2)

        results.append({
            'operation': 'QR',
            'size': f'{n}x{n}',
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': scipy_time / scirs2_time,
            'ratio': scirs2_time / scipy_time
        })

    return pd.DataFrame(results)


def main():
    """Run all linear algebra benchmarks"""
    print("=" * 80)
    print("Linear Algebra Performance Benchmarks")
    print("Comparing scirs2 vs SciPy")
    print("=" * 80)
    print()

    # Run benchmarks
    benchmarks = [
        ("Determinant", benchmark_determinant),
        ("Matrix Inverse", benchmark_matrix_inverse),
        ("Linear Solve", benchmark_linear_solve),
        ("SVD", benchmark_svd),
        ("QR", benchmark_qr),
    ]

    all_results = []

    for name, bench_func in benchmarks:
        print(f"\n{name}")
        print("-" * 80)
        df = bench_func()
        print(df.to_string(index=False))
        print()

        # Calculate summary stats
        avg_ratio = df['ratio'].mean()
        status = "✅ GOOD" if avg_ratio < 2.0 else "🟡 ACCEPTABLE" if avg_ratio < 5.0 else "❌ SLOW"
        print(f"Average Performance: {avg_ratio:.2f}x slower than SciPy - {status}")

        all_results.append(df)

    # Combined summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nOverall average performance: {combined['ratio'].mean():.2f}x slower than SciPy")
    print(f"Best speedup: {combined['speedup'].max():.2f}x faster")
    print(f"Worst slowdown: {combined['ratio'].max():.2f}x slower")

    # Save results
    combined.to_csv('benchmarks/linalg_results.csv', index=False)
    print(f"\nResults saved to benchmarks/linalg_results.csv")

    return combined


if __name__ == '__main__':
    results = main()
