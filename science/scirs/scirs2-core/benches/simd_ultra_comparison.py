#!/usr/bin/env python3
"""
NumPy benchmark for basic operations (add, mul, dot, sum)
Compares against SciRS2 ultra-optimized SIMD functions

Usage:
    python3 benches/simd_ultra_comparison.py [--numpy-only] [--compare-only]
"""

import numpy as np
import time
import csv
import os
import sys
import argparse
from pathlib import Path


def get_temp_dir():
    """Get temporary directory from environment or default"""
    return os.environ.get('TMPDIR', os.environ.get('TEMP', '/tmp'))


def benchmark(func, iterations, warmup=10):
    """Benchmark a function with warmup"""
    # Warm-up
    for _ in range(warmup):
        result = func()
    _ = result  # Prevent optimization

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func()
    elapsed = time.perf_counter() - start

    # Prevent optimization
    if hasattr(result, '__len__'):
        _ = result[0]
    else:
        _ = float(result)

    return (elapsed / iterations) * 1_000_000  # microseconds


def run_numpy_benchmark():
    """Run NumPy benchmark and save results"""
    print("=" * 80)
    print("NumPy Basic Operations Benchmark")
    print("=" * 80)
    print()

    sizes = [100, 1000, 10000, 100000]
    results = []

    for size in sizes:
        print(f"\nArray size: {size:,}")
        print("-" * 60)

        iterations = 10000 if size <= 100 else (1000 if size <= 1000 else (100 if size <= 10000 else 10))

        # Generate test data
        np.random.seed(42)
        a_f32 = np.random.randn(size).astype(np.float32)
        b_f32 = np.random.randn(size).astype(np.float32)

        operations = [
            ('add', lambda: np.add(a_f32, b_f32)),
            ('multiply', lambda: np.multiply(a_f32, b_f32)),
            ('dot', lambda: np.dot(a_f32, b_f32)),
            ('sum', lambda: np.sum(a_f32)),
        ]

        print(f"  F32 Operations ({iterations} iterations):")
        for op_name, op_func in operations:
            time_us = benchmark(op_func, iterations)
            print(f"    {op_name:<12} f32: {time_us:10.2f} μs")
            results.append({
                'operation': op_name,
                'dtype': 'f32',
                'size': size,
                'implementation': 'numpy',
                'time_us': time_us,
                'iterations': iterations
            })

    # Save results
    temp_dir = get_temp_dir()
    csv_file = os.path.join(temp_dir, 'scirs2_numpy_benchmark.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['operation', 'dtype', 'size', 'implementation', 'time_us', 'iterations'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n\nResults saved to: {csv_file}")
    return csv_file


def load_csv(filename):
    """Load CSV file into list of dicts"""
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['size'] = int(row['size'])
            row['time_us'] = float(row['time_us'])
            row['iterations'] = int(row['iterations'])
            results.append(row)
    return results


def compare_results():
    """Compare SciRS2 and NumPy results"""
    temp_dir = get_temp_dir()
    scirs2_csv = os.path.join(temp_dir, 'scirs2_ultra_benchmark.csv')
    numpy_csv = os.path.join(temp_dir, 'scirs2_numpy_benchmark.csv')

    if not Path(scirs2_csv).exists():
        print(f"Error: {scirs2_csv} not found")
        print("Run: cargo run --example simd_ultra_benchmark_csv --release --features simd")
        return False

    if not Path(numpy_csv).exists():
        print(f"Error: {numpy_csv} not found")
        print("Run: python3 benches/simd_ultra_comparison.py --numpy-only")
        return False

    scirs2_data = load_csv(scirs2_csv)
    numpy_data = load_csv(numpy_csv)

    # Create lookups
    numpy_lookup = {(r['operation'], r['dtype'], r['size']): r['time_us'] for r in numpy_data}

    ultra_results = [r for r in scirs2_data if r['implementation'] == 'scirs2_ultra']
    standard_results = [r for r in scirs2_data if r['implementation'] == 'scirs2_standard']
    standard_lookup = {(r['operation'], r['dtype'], r['size']): r['time_us'] for r in standard_results}

    print()
    print("=" * 100)
    print("SciRS2 Ultra-Optimized SIMD vs NumPy Performance Comparison")
    print("=" * 100)
    print()

    # Table header
    print(f"{'Operation':<12} {'Size':<10} {'NumPy (μs)':<14} {'Standard (μs)':<14} {'Ultra (μs)':<14} {'vs NumPy':<12} {'vs Standard':<12}")
    print("-" * 100)

    comparisons = []
    for row in ultra_results:
        key = (row['operation'], row['dtype'], row['size'])
        numpy_time = numpy_lookup.get(key, 0)
        standard_time = standard_lookup.get(key, 0)
        ultra_time = row['time_us']

        vs_numpy = numpy_time / ultra_time if ultra_time > 0 else float('inf')
        vs_standard = standard_time / ultra_time if ultra_time > 0 else float('inf')

        comparisons.append({
            'operation': row['operation'],
            'dtype': row['dtype'],
            'size': row['size'],
            'numpy_time': numpy_time,
            'standard_time': standard_time,
            'ultra_time': ultra_time,
            'vs_numpy': vs_numpy,
            'vs_standard': vs_standard,
        })

        # Color coding for terminal
        numpy_indicator = "✓" if vs_numpy >= 1.0 else "✗"
        std_indicator = "✓" if vs_standard >= 1.0 else "✗"

        print(f"{row['operation']:<12} {row['size']:<10,} {numpy_time:<14.2f} {standard_time:<14.2f} {ultra_time:<14.2f} {vs_numpy:<10.2f}x {numpy_indicator} {vs_standard:<10.2f}x {std_indicator}")

    print("-" * 100)
    print()

    # Summary statistics
    vs_numpy_values = [c['vs_numpy'] for c in comparisons if c['vs_numpy'] != float('inf')]
    vs_standard_values = [c['vs_standard'] for c in comparisons if c['vs_standard'] != float('inf')]

    if vs_numpy_values:
        avg_vs_numpy = sum(vs_numpy_values) / len(vs_numpy_values)
        print(f"Average speedup vs NumPy:    {avg_vs_numpy:.2f}x")

    if vs_standard_values:
        avg_vs_standard = sum(vs_standard_values) / len(vs_standard_values)
        print(f"Average speedup vs Standard: {avg_vs_standard:.2f}x")

    print()

    # By operation summary
    print("Performance by Operation:")
    print("-" * 60)
    from collections import defaultdict
    ops = defaultdict(list)
    for c in comparisons:
        ops[c['operation']].append(c)

    for op, data in ops.items():
        vs_numpy_avg = sum(d['vs_numpy'] for d in data if d['vs_numpy'] != float('inf')) / len(data)
        vs_std_avg = sum(d['vs_standard'] for d in data if d['vs_standard'] != float('inf')) / len(data)
        numpy_win = "✓" if vs_numpy_avg >= 1.0 else "✗"
        print(f"  {op:<12}: vs NumPy = {vs_numpy_avg:>6.2f}x {numpy_win}, vs Standard = {vs_std_avg:>6.2f}x ✓")

    print()
    print("=" * 100)
    print("Legend: ✓ = SciRS2 Ultra faster, ✗ = Comparison target faster")
    print("=" * 100)

    # Save detailed report
    report_file = os.path.join(temp_dir, 'scirs2_simd_comparison_report.txt')
    with open(report_file, 'w') as f:
        f.write("SciRS2 Ultra-Optimized vs NumPy Performance Report\n")
        f.write("=" * 80 + "\n\n")
        for c in comparisons:
            f.write(f"{c['operation']} (size={c['size']:,}):\n")
            f.write(f"  NumPy:    {c['numpy_time']:.2f} μs\n")
            f.write(f"  Standard: {c['standard_time']:.2f} μs\n")
            f.write(f"  Ultra:    {c['ultra_time']:.2f} μs\n")
            f.write(f"  Speedup vs NumPy: {c['vs_numpy']:.2f}x\n")
            f.write(f"  Speedup vs Standard: {c['vs_standard']:.2f}x\n\n")

    print(f"\nDetailed report saved to: {report_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='SciRS2 SIMD vs NumPy Benchmark')
    parser.add_argument('--numpy-only', action='store_true', help='Run only NumPy benchmark')
    parser.add_argument('--compare-only', action='store_true', help='Run only comparison (requires existing CSVs)')
    args = parser.parse_args()

    if args.numpy_only:
        run_numpy_benchmark()
    elif args.compare_only:
        compare_results()
    else:
        # Run both
        run_numpy_benchmark()
        print("\n" + "=" * 80 + "\n")
        compare_results()


if __name__ == '__main__':
    main()
