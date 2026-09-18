"""
Quick Linear Algebra Benchmarks

Focused benchmark set for linear algebra operations with reasonable timeouts.
Avoids large matrices that may hang.
"""

import numpy as np
import scipy.linalg
import scirs2
import time
import pandas as pd


def quick_bench(name, scipy_func, scirs2_func, data, n_runs=10):
    """Quick benchmark with timeout protection"""
    try:
        # Warmup
        _ = scipy_func(data)
        _ = scirs2_func(data)

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

        return {
            'operation': name,
            'scipy_ms': scipy_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'scipy'
        }
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        return None


def main():
    print("="*80)
    print("LINEAR ALGEBRA QUICK BENCHMARKS")
    print("="*80)
    print()

    results = []
    np.random.seed(42)

    # Test sizes: small to medium (avoid hanging)
    sizes = [10, 50, 100, 200]

    for size in sizes:
        print(f"Testing size {size}x{size}:")

        A = np.ascontiguousarray(np.random.randn(size, size))
        b = np.ascontiguousarray(np.random.randn(size))

        # Determinant
        res = quick_bench(f"det_{size}",
                         lambda x: scipy.linalg.det(x),
                         lambda x: scirs2.det_py(x), A)
        if res:
            results.append(res)
            print(f"  det: {res['speedup']:.2f}x")

        # Matrix inverse (make well-conditioned)
        A_inv = A + np.eye(size) * 10
        res = quick_bench(f"inv_{size}",
                         lambda x: scipy.linalg.inv(x),
                         lambda x: scirs2.inv_py(x), A_inv)
        if res:
            results.append(res)
            print(f"  inv: {res['speedup']:.2f}x")

        # Linear solve
        res = quick_bench(f"solve_{size}",
                         lambda _: scipy.linalg.solve(A_inv, b),
                         lambda _: scirs2.solve_py(A_inv, b), None)
        if res:
            results.append(res)
            print(f"  solve: {res['speedup']:.2f}x")

        # QR decomposition
        res = quick_bench(f"qr_{size}",
                         lambda x: scipy.linalg.qr(x),
                         lambda x: scirs2.qr_py(x), A)
        if res:
            results.append(res)
            print(f"  qr: {res['speedup']:.2f}x")

        # LU decomposition
        res = quick_bench(f"lu_{size}",
                         lambda x: scipy.linalg.lu(x),
                         lambda x: scirs2.lu_py(x), A)
        if res:
            results.append(res)
            print(f"  lu: {res['speedup']:.2f}x")

        # Only do SVD and eigenvalues for small sizes (slow operations)
        if size <= 100:
            res = quick_bench(f"svd_{size}",
                             lambda x: scipy.linalg.svd(x),
                             lambda x: scirs2.svd_py(x), A, n_runs=5)
            if res:
                results.append(res)
                print(f"  svd: {res['speedup']:.2f}x")

            res = quick_bench(f"eigvals_{size}",
                             lambda x: scipy.linalg.eigvals(x),
                             lambda x: scirs2.eigvals_py(x), A, n_runs=5)
            if res:
                results.append(res)
                print(f"  eigvals: {res['speedup']:.2f}x")

        # Cholesky (positive definite)
        A_pd = A @ A.T + np.eye(size)
        A_pd = np.ascontiguousarray(A_pd)
        res = quick_bench(f"cholesky_{size}",
                         lambda x: scipy.linalg.cholesky(x),
                         lambda x: scirs2.cholesky_py(x), A_pd)
        if res:
            results.append(res)
            print(f"  cholesky: {res['speedup']:.2f}x")

    # Create results
    if not results:
        print("No results collected!")
        return

    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))

    overall_speedup = df['speedup'].mean()
    scirs2_wins = (df['winner'] == 'scirs2').sum()
    total = len(df)

    print(f"\n**Overall Average Speedup**: {overall_speedup:.2f}x")
    print(f"**SciRS2 Wins**: {scirs2_wins}/{total} ({scirs2_wins/total*100:.1f}%)")

    # Category summary
    df['op_type'] = df['operation'].str.extract(r'^([a-z]+)_')[0]
    summary = df.groupby('op_type').agg({
        'speedup': ['mean', 'count']
    }).round(2)
    summary.columns = ['avg_speedup', 'tests']

    print(f"\n{'='*80}")
    print("BY OPERATION")
    print(f"{'='*80}\n")
    print(summary.to_string())

    # Save results
    output_file = '/tmp/bench_linalg_quick_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Markdown report
    md_file = '/tmp/bench_linalg_quick_report.md'
    with open(md_file, 'w') as f:
        f.write("# Linear Algebra Quick Benchmarks\n\n")
        f.write(f"- **Total Tests**: {total}\n")
        f.write(f"- **Overall Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"- **SciRS2 Win Rate**: {scirs2_wins/total*100:.1f}%\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## By Operation\n\n")
        f.write(summary.to_markdown())

    print(f"Markdown report saved to: {md_file}")

    return df


if __name__ == '__main__':
    main()
