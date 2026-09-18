#!/usr/bin/env python3
"""
SciRS2 v0.2.0 Python Comparison Benchmarks

This script runs equivalent Python benchmarks for comparison with SciRS2.
It reads the Rust benchmark results and runs matching Python implementations
using NumPy, SciPy, and PyTorch.

Requirements:
    pip install numpy scipy matplotlib pandas tabulate
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import linalg, fft, special, integrate
import pandas as pd

class PythonBenchmarkSuite:
    """Run Python equivalent benchmarks"""

    def __init__(self):
        self.results = []

    def benchmark(self, func, *args, n_runs=10, warmup=2):
        """Benchmark a function with warmup and multiple runs"""
        # Warmup
        for _ in range(warmup):
            func(*args)

        # Measure
        times = []
        for _ in range(n_runs):
            start = time.perf_counter_ns()
            func(*args)
            end = time.perf_counter_ns()
            times.append(end - start)

        return {
            'mean_time_ns': np.mean(times),
            'median_time_ns': np.median(times),
            'std_dev_ns': np.std(times),
            'min_time_ns': np.min(times),
            'max_time_ns': np.max(times),
        }

    def bench_array_operations(self):
        """Benchmark NumPy array operations"""
        print("  Running array operations benchmarks...")
        sizes = [1000, 10_000, 100_000, 1_000_000]

        for size in sizes:
            # Zeros
            result = self.benchmark(np.zeros, size, dtype=np.float64)
            self.results.append({
                'category': 'array_ops',
                'operation': 'zeros',
                'size': size,
                **result
            })

            # Arange
            result = self.benchmark(np.arange, size, dtype=np.float64)
            self.results.append({
                'category': 'array_ops',
                'operation': 'arange',
                'size': size,
                **result
            })

            # Elementwise operations
            a = np.arange(size, dtype=np.float64)
            b = np.arange(1, size + 1, dtype=np.float64)

            result = self.benchmark(np.add, a, b)
            self.results.append({
                'category': 'array_ops',
                'operation': 'add',
                'size': size,
                **result
            })

            result = self.benchmark(np.multiply, a, b)
            self.results.append({
                'category': 'array_ops',
                'operation': 'multiply',
                'size': size,
                **result
            })

            result = self.benchmark(np.sum, a)
            self.results.append({
                'category': 'array_ops',
                'operation': 'sum',
                'size': size,
                **result
            })

            result = self.benchmark(np.mean, a)
            self.results.append({
                'category': 'array_ops',
                'operation': 'mean',
                'size': size,
                **result
            })

    def bench_linalg_operations(self):
        """Benchmark SciPy linear algebra operations"""
        print("  Running linear algebra benchmarks...")
        sizes = [50, 100, 200, 500]

        for size in sizes:
            rng = np.random.default_rng(42)
            matrix = rng.uniform(-1.0, 1.0, (size, size))
            vector = rng.uniform(-1.0, 1.0, size)

            # Determinant
            result = self.benchmark(np.linalg.det, matrix)
            self.results.append({
                'category': 'linalg',
                'operation': 'det',
                'size': size,
                **result
            })

            # Inverse
            if size <= 200:
                result = self.benchmark(np.linalg.inv, matrix)
                self.results.append({
                    'category': 'linalg',
                    'operation': 'inv',
                    'size': size,
                    **result
                })

            # LU decomposition
            result = self.benchmark(linalg.lu, matrix)
            self.results.append({
                'category': 'linalg',
                'operation': 'lu',
                'size': size,
                **result
            })

            # QR decomposition
            result = self.benchmark(np.linalg.qr, matrix)
            self.results.append({
                'category': 'linalg',
                'operation': 'qr',
                'size': size,
                **result
            })

            # Linear solve
            result = self.benchmark(np.linalg.solve, matrix, vector)
            self.results.append({
                'category': 'linalg',
                'operation': 'solve',
                'size': size,
                **result
            })

            # Matrix multiplication
            b_matrix = rng.uniform(-1.0, 1.0, (size, size))
            result = self.benchmark(np.dot, matrix, b_matrix)
            self.results.append({
                'category': 'linalg',
                'operation': 'matmul',
                'size': size,
                **result
            })

    def bench_fft_operations(self):
        """Benchmark FFT operations"""
        print("  Running FFT benchmarks...")
        sizes = [128, 512, 2048, 8192, 32768]

        for size in sizes:
            rng = np.random.default_rng(42)
            signal = rng.uniform(-1.0, 1.0, size)

            # Forward FFT
            result = self.benchmark(fft.fft, signal)
            self.results.append({
                'category': 'fft',
                'operation': 'fft',
                'size': size,
                **result
            })

            # Inverse FFT
            freq = fft.fft(signal)
            result = self.benchmark(fft.ifft, freq)
            self.results.append({
                'category': 'fft',
                'operation': 'ifft',
                'size': size,
                **result
            })

    def bench_stats_operations(self):
        """Benchmark statistical operations"""
        print("  Running statistical benchmarks...")
        sizes = [10_000, 100_000, 1_000_000]

        for size in sizes:
            rng = np.random.default_rng(42)
            data = rng.uniform(-1.0, 1.0, size)

            # Mean
            result = self.benchmark(np.mean, data)
            self.results.append({
                'category': 'stats',
                'operation': 'mean',
                'size': size,
                **result
            })

            # Standard deviation
            result = self.benchmark(np.std, data)
            self.results.append({
                'category': 'stats',
                'operation': 'std',
                'size': size,
                **result
            })

            # Variance
            result = self.benchmark(np.var, data)
            self.results.append({
                'category': 'stats',
                'operation': 'var',
                'size': size,
                **result
            })

            # Median
            if size <= 100_000:
                result = self.benchmark(np.median, data)
                self.results.append({
                    'category': 'stats',
                    'operation': 'median',
                    'size': size,
                    **result
                })

    def bench_special_functions(self):
        """Benchmark special mathematical functions"""
        print("  Running special functions benchmarks...")
        n_points = 10_000
        x_values = np.arange(n_points) * 0.01

        # Bessel J0
        result = self.benchmark(lambda x: special.j0(x).sum(), x_values)
        self.results.append({
            'category': 'special',
            'operation': 'bessel_j0',
            'size': n_points,
            **result
        })

        # Bessel J1
        result = self.benchmark(lambda x: special.j1(x).sum(), x_values)
        self.results.append({
            'category': 'special',
            'operation': 'bessel_j1',
            'size': n_points,
            **result
        })

        # Gamma function
        result = self.benchmark(lambda x: special.gamma(x + 1.0).sum(), x_values)
        self.results.append({
            'category': 'special',
            'operation': 'gamma',
            'size': n_points,
            **result
        })

        # Error function
        result = self.benchmark(lambda x: special.erf(x).sum(), x_values)
        self.results.append({
            'category': 'special',
            'operation': 'erf',
            'size': n_points,
            **result
        })

        # Complementary error function
        result = self.benchmark(lambda x: special.erfc(x).sum(), x_values)
        self.results.append({
            'category': 'special',
            'operation': 'erfc',
            'size': n_points,
            **result
        })

    def bench_integration_operations(self):
        """Benchmark numerical integration"""
        print("  Running integration benchmarks...")

        def f(x):
            return np.sin(x)

        # Trapezoid rule
        sample_counts = [100, 1000, 10_000]
        for n in sample_counts:
            x = np.linspace(0, np.pi, n)
            y = f(x)

            result = self.benchmark(integrate.trapezoid, y, x)
            self.results.append({
                'category': 'integration',
                'operation': 'trapz',
                'size': n,
                **result
            })

            result = self.benchmark(integrate.simpson, y, x=x)
            self.results.append({
                'category': 'integration',
                'operation': 'simps',
                'size': n,
                **result
            })

        # Adaptive quadrature
        result = self.benchmark(lambda: integrate.quad(np.sin, 0, np.pi))
        self.results.append({
            'category': 'integration',
            'operation': 'quad',
            'size': 0,  # Adaptive
            **result
        })

    def run_all(self):
        """Run all benchmarks"""
        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║       Running Python Comparison Benchmarks                ║")
        print("╚════════════════════════════════════════════════════════════╝\n")

        self.bench_array_operations()
        self.bench_linalg_operations()
        self.bench_fft_operations()
        self.bench_stats_operations()
        self.bench_special_functions()
        self.bench_integration_operations()

        return self.results


def load_rust_results(filename):
    """Load Rust benchmark results from JSON"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data.get('results', [])


def compare_results(rust_results, python_results):
    """Compare Rust and Python results and generate report"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║          Generating Comparison Report                     ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Create comparison DataFrame
    comparisons = []

    for rust in rust_results:
        # Find matching Python result
        python = next(
            (p for p in python_results
             if p['category'] == rust['category']
             and p['operation'] == rust['operation']
             and p['size'] == rust['size']),
            None
        )

        if python:
            rust_time = rust['mean_time_ns']
            python_time = python['mean_time_ns']
            speedup = python_time / rust_time

            comparisons.append({
                'Category': rust['category'],
                'Operation': rust['operation'],
                'Size': rust['size'],
                'Rust (ms)': rust_time / 1_000_000,
                'Python (ms)': python_time / 1_000_000,
                'Speedup': speedup,
                'Winner': 'Rust' if speedup > 1.0 else 'Python',
            })

    df = pd.DataFrame(comparisons)

    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===\n")
    print(f"Total comparisons: {len(comparisons)}")
    print(f"Rust faster: {sum(1 for c in comparisons if c['Speedup'] > 1.0)}")
    print(f"Python faster: {sum(1 for c in comparisons if c['Speedup'] <= 1.0)}")
    print(f"\nAverage speedup: {df['Speedup'].mean():.2f}x")
    print(f"Median speedup: {df['Speedup'].median():.2f}x")
    print(f"Max speedup: {df['Speedup'].max():.2f}x")
    print(f"Min speedup: {df['Speedup'].min():.2f}x")

    # Print detailed table by category
    print("\n=== DETAILED RESULTS BY CATEGORY ===\n")
    for category in df['Category'].unique():
        cat_df = df[df['Category'] == category]
        print(f"\n{category.upper()}:")
        print(cat_df[['Operation', 'Size', 'Rust (ms)', 'Python (ms)', 'Speedup']].to_string(index=False))

    # Save to JSON
    output_file = '/tmp/scirs2_v020_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_comparisons': len(comparisons),
                'rust_faster': sum(1 for c in comparisons if c['Speedup'] > 1.0),
                'python_faster': sum(1 for c in comparisons if c['Speedup'] <= 1.0),
                'average_speedup': float(df['Speedup'].mean()),
                'median_speedup': float(df['Speedup'].median()),
                'max_speedup': float(df['Speedup'].max()),
                'min_speedup': float(df['Speedup'].min()),
            },
            'comparisons': comparisons,
        }, f, indent=2)

    print(f"\n✓ Saved comparison results to {output_file}")

    return df


def generate_plots(df):
    """Generate comparison plots"""
    try:
        import matplotlib.pyplot as plt

        # Speedup by category
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = df['Category'].unique()
        for category in categories:
            cat_df = df[df['Category'] == category]
            ax.scatter(cat_df['Size'], cat_df['Speedup'], label=category, alpha=0.6, s=100)

        ax.axhline(y=1.0, color='r', linestyle='--', label='Equal performance')
        ax.set_xlabel('Size')
        ax.set_ylabel('Speedup (Python time / Rust time)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('SciRS2 vs Python Performance Comparison (v0.2.0)')

        output_file = '/tmp/scirs2_v020_speedup_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved speedup plot to {output_file}")

    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")


def main():
    """Main entry point"""
    rust_file = '/tmp/scirs2_v020_python_comparison_rust.json'
    python_file = '/tmp/scirs2_v020_python_comparison_python.json'

    # Check if Rust results exist
    if not Path(rust_file).exists():
        print(f"Error: Rust benchmark results not found at {rust_file}")
        print("Please run: cargo bench --bench v020_python_comparison")
        sys.exit(1)

    # Run Python benchmarks
    suite = PythonBenchmarkSuite()
    python_results = suite.run_all()

    # Save Python results
    with open(python_file, 'w') as f:
        json.dump({
            'timestamp': pd.Timestamp.now().isoformat(),
            'platform': sys.platform,
            'results': python_results,
        }, f, indent=2)

    print(f"\n✓ Saved {len(python_results)} Python benchmark results to {python_file}")

    # Load Rust results
    rust_results = load_rust_results(rust_file)
    print(f"✓ Loaded {len(rust_results)} Rust benchmark results from {rust_file}")

    # Compare results
    df = compare_results(rust_results, python_results)

    # Generate plots
    generate_plots(df)

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║          Comparison Complete!                              ║")
    print("╚════════════════════════════════════════════════════════════╝\n")


if __name__ == '__main__':
    main()
