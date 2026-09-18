"""
FFT Performance Benchmarks

Compares scirs2 vs NumPy/SciPy for FFT operations.
"""

import numpy as np
import scipy.fft
import scirs2
import time
import pandas as pd
from typing import List, Dict


def benchmark_fft_operation(
    name: str,
    numpy_func,
    scirs2_func,
    sizes: List[int],
    n_runs: int = 30
) -> List[Dict]:
    """Benchmark an FFT operation"""
    results = []

    for size in sizes:
        print(f"  Testing {name} with size {size}...", end="", flush=True)

        # Generate test data
        np.random.seed(42)
        if 'rfft' in name or 'dct' in name:
            data = np.ascontiguousarray(np.random.randn(size))
        else:
            data = np.ascontiguousarray(np.random.randn(size) + 1j * np.random.randn(size))

        # Warmup
        try:
            _ = numpy_func(data)
            _ = scirs2_func(data)
        except Exception as e:
            print(f" SKIP ({e})")
            continue

        # Benchmark NumPy/SciPy
        numpy_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = numpy_func(data)
            numpy_times.append(time.perf_counter() - start)
        numpy_time = np.median(numpy_times)

        # Benchmark SciRS2
        scirs2_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = scirs2_func(data)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = numpy_time / scirs2_time
        print(f" ✓ (speedup: {speedup:.2f}x)")

        results.append({
            'operation': name,
            'size': size,
            'numpy_us': numpy_time * 1_000_000,
            'scirs2_us': scirs2_time * 1_000_000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'numpy'
        })

    return results


def main():
    """Run all FFT benchmarks"""
    print("="*80)
    print("FFT BENCHMARKS")
    print("="*80)
    print()

    all_results = []
    # Use power-of-2 sizes for optimal FFT performance
    sizes = [128, 512, 2048, 8192, 32768, 131072]

    # FFT
    print("Benchmarking: FFT (Complex)")
    all_results.extend(benchmark_fft_operation(
        "fft",
        lambda x: np.fft.fft(x),
        lambda x: scirs2.fft_py(x),
        sizes
    ))

    # IFFT
    print("Benchmarking: IFFT (Complex)")
    all_results.extend(benchmark_fft_operation(
        "ifft",
        lambda x: np.fft.ifft(x),
        lambda x: scirs2.ifft_py(x),
        sizes
    ))

    # RFFT (Real FFT)
    print("Benchmarking: RFFT (Real)")
    def rfft_setup_and_run(name, numpy_func, scirs2_func):
        results = []
        for size in sizes:
            print(f"  Testing {name} with size {size}...", end="", flush=True)

            np.random.seed(42)
            data = np.ascontiguousarray(np.random.randn(size))

            try:
                _ = numpy_func(data)
                _ = scirs2_func(data)
            except Exception as e:
                print(f" SKIP ({e})")
                continue

            # Benchmark NumPy
            numpy_times = []
            for _ in range(30):
                start = time.perf_counter()
                _ = numpy_func(data)
                numpy_times.append(time.perf_counter() - start)
            numpy_time = np.median(numpy_times)

            # Benchmark SciRS2
            scirs2_times = []
            for _ in range(30):
                start = time.perf_counter()
                _ = scirs2_func(data)
                scirs2_times.append(time.perf_counter() - start)
            scirs2_time = np.median(scirs2_times)

            speedup = numpy_time / scirs2_time
            print(f" ✓ (speedup: {speedup:.2f}x)")

            results.append({
                'operation': name,
                'size': size,
                'numpy_us': numpy_time * 1_000_000,
                'scirs2_us': scirs2_time * 1_000_000,
                'speedup': speedup,
                'winner': 'scirs2' if speedup > 1.0 else 'numpy'
            })
        return results

    all_results.extend(rfft_setup_and_run(
        "rfft",
        lambda x: np.fft.rfft(x),
        lambda x: scirs2.rfft_py(x)
    ))

    # IRFFT
    print("Benchmarking: IRFFT (Inverse Real)")
    # For IRFFT, we need complex input from RFFT
    irfft_results = []
    for size in sizes:
        print(f"  Testing irfft with size {size}...", end="", flush=True)

        np.random.seed(42)
        real_data = np.random.randn(size)
        complex_data = np.ascontiguousarray(np.fft.rfft(real_data))

        try:
            _ = np.fft.irfft(complex_data)
            _ = scirs2.irfft_py(complex_data)
        except Exception as e:
            print(f" SKIP ({e})")
            continue

        # Benchmark NumPy
        numpy_times = []
        for _ in range(30):
            start = time.perf_counter()
            _ = np.fft.irfft(complex_data, n=size)
            numpy_times.append(time.perf_counter() - start)
        numpy_time = np.median(numpy_times)

        # Benchmark SciRS2
        scirs2_times = []
        for _ in range(30):
            start = time.perf_counter()
            _ = scirs2.irfft_py(complex_data, n=size)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = numpy_time / scirs2_time
        print(f" ✓ (speedup: {speedup:.2f}x)")

        irfft_results.append({
            'operation': 'irfft',
            'size': size,
            'numpy_us': numpy_time * 1_000_000,
            'scirs2_us': scirs2_time * 1_000_000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'numpy'
        })

    all_results.extend(irfft_results)

    # DCT (Discrete Cosine Transform)
    print("Benchmarking: DCT")
    all_results.extend(rfft_setup_and_run(
        "dct",
        lambda x: scipy.fft.dct(x),
        lambda x: scirs2.dct_py(x)
    ))

    # FFT Shift
    print("Benchmarking: FFT Shift")
    all_results.extend(rfft_setup_and_run(
        "fftshift",
        lambda x: np.fft.fftshift(x),
        lambda x: scirs2.fftshift_py(x)
    ))

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

    # Save results
    output_file = '/tmp/bench_fft_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create markdown report
    md_file = '/tmp/bench_fft_report.md'
    with open(md_file, 'w') as f:
        f.write("# FFT Performance Benchmarks\n\n")
        f.write("## Test Configuration\n\n")
        f.write(f"- **Test Sizes**: {sizes} (power-of-2 for optimal FFT)\n")
        f.write(f"- **Runs per Test**: 30 (median time)\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")
        f.write(summary.to_markdown())
        f.write(f"\n\n**Overall Average Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"**SciRS2 Win Rate**: {scirs2_wins/total_tests*100:.1f}%\n")

    print(f"Markdown report saved to: {md_file}")

    return df


if __name__ == '__main__':
    main()
