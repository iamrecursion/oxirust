#!/usr/bin/env python3
"""
Run both Rust and Python benchmarks and generate comparison report

This script:
1. Runs the Rust benchmarks using cargo bench
2. Runs the Python SciPy benchmarks
3. Parses results from both
4. Generates a comparison report with speedup factors
"""

import subprocess
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, Any

def run_rust_benchmarks() -> Dict[str, Any]:
    """Run Rust benchmarks and parse results"""
    print("Running Rust benchmarks...")
    
    # Run cargo bench for comprehensive comparison
    cmd = ["cargo", "bench", "--bench", "comprehensive_scipy_comparison"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running Rust benchmarks: {result.stderr}")
        return {}
    
    # Parse Criterion output
    # Criterion saves results in target/criterion/
    criterion_dir = Path("../target/criterion")
    rust_results = {}
    
    if criterion_dir.exists():
        for bench_dir in criterion_dir.iterdir():
            if bench_dir.is_dir() and not bench_dir.name.startswith('.'):
                # Parse benchmark group results
                group_name = bench_dir.name
                rust_results[group_name] = {}
                
                for size_dir in bench_dir.iterdir():
                    if size_dir.is_dir() and size_dir.name.isdigit():
                        size = int(size_dir.name)
                        # Read the benchmark.json file
                        json_file = size_dir / "base" / "benchmark.json"
                        if json_file.exists():
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                # Convert nanoseconds to microseconds
                                mean_ns = data.get('mean', {}).get('point_estimate', 0)
                                rust_results[group_name][size] = mean_ns / 1000.0
    
    return rust_results

def run_scipy_benchmarks() -> Dict[str, Any]:
    """Run SciPy benchmarks"""
    print("\nRunning SciPy benchmarks...")
    
    # Run the Python benchmark script
    cmd = [sys.executable, "scipy_comprehensive_benchmark.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running SciPy benchmarks: {result.stderr}")
        return {}
    
    # Load results from JSON file
    results_file = "scipy_benchmark_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return {}

def calculate_speedup(rust_results: Dict[str, Any], scipy_results: Dict[str, Any]) -> pd.DataFrame:
    """Calculate speedup factors (SciPy time / Rust time)"""
    speedup_data = []
    
    for func_name in scipy_results:
        if func_name in rust_results:
            for size in scipy_results[func_name]:
                if size in rust_results[func_name]:
                    scipy_time = scipy_results[func_name][size]
                    rust_time = rust_results[func_name][size]
                    speedup = scipy_time / rust_time if rust_time > 0 else float('inf')
                    
                    speedup_data.append({
                        'function': func_name,
                        'size': int(size),
                        'scipy_time_us': scipy_time,
                        'rust_time_us': rust_time,
                        'speedup': speedup
                    })
    
    return pd.DataFrame(speedup_data)

def generate_report(df: pd.DataFrame) -> None:
    """Generate comparison report with visualizations"""
    print("\n=== Performance Comparison Report ===")
    
    # Summary statistics
    print("\nOverall Performance Summary:")
    print(f"Average speedup: {df['speedup'].mean():.2f}x")
    print(f"Median speedup: {df['speedup'].median():.2f}x")
    print(f"Min speedup: {df['speedup'].min():.2f}x")
    print(f"Max speedup: {df['speedup'].max():.2f}x")
    
    # Function-wise summary
    print("\nFunction-wise Average Speedup:")
    func_speedup = df.groupby('function')['speedup'].mean().sort_values(ascending=False)
    for func, speedup in func_speedup.items():
        print(f"  {func}: {speedup:.2f}x")
    
    # Create visualizations
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall speedup distribution
    ax = axes[0, 0]
    df['speedup'].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_xlabel('Speedup Factor')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Speedup Factors')
    ax.axvline(df['speedup'].mean(), color='red', linestyle='--', label=f'Mean: {df["speedup"].mean():.2f}x')
    ax.legend()
    
    # 2. Speedup by function
    ax = axes[0, 1]
    func_speedup.plot(kind='barh', ax=ax)
    ax.set_xlabel('Average Speedup Factor')
    ax.set_title('Average Speedup by Function')
    ax.grid(True, alpha=0.3)
    
    # 3. Speedup vs sample size
    ax = axes[1, 0]
    for func in df['function'].unique()[:5]:  # Top 5 functions
        func_df = df[df['function'] == func]
        ax.plot(func_df['size'], func_df['speedup'], marker='o', label=func)
    ax.set_xscale('log')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Speedup vs Sample Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. Performance comparison scatter plot
    ax = axes[1, 1]
    ax.scatter(df['scipy_time_us'], df['rust_time_us'], alpha=0.6)
    
    # Add diagonal line (equal performance)
    max_time = max(df['scipy_time_us'].max(), df['rust_time_us'].max())
    ax.plot([0, max_time], [0, max_time], 'r--', label='Equal performance')
    
    ax.set_xlabel('SciPy Time (µs)')
    ax.set_ylabel('Rust Time (µs)')
    ax.set_title('Rust vs SciPy Execution Time')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'benchmark_comparison.png'")
    
    # Save detailed results
    df.to_csv('benchmark_comparison_detailed.csv', index=False)
    print("Detailed results saved as 'benchmark_comparison_detailed.csv'")
    
    # Generate markdown report
    with open('benchmark_report.md', 'w') as f:
        f.write("# scirs2-stats vs SciPy Performance Comparison\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Average speedup**: {df['speedup'].mean():.2f}x\n")
        f.write(f"- **Median speedup**: {df['speedup'].median():.2f}x\n")
        f.write(f"- **Range**: {df['speedup'].min():.2f}x - {df['speedup'].max():.2f}x\n\n")
        
        f.write("## Function Performance\n\n")
        f.write("| Function | Average Speedup |\n")
        f.write("|----------|----------------|\n")
        for func, speedup in func_speedup.items():
            f.write(f"| {func} | {speedup:.2f}x |\n")
        
        f.write("\n## Methodology\n\n")
        f.write("- Warmup time: 1 second\n")
        f.write("- Measurement time: 3 seconds\n")
        f.write("- Sample sizes: 10, 50, 100, 500, 1000, 5000, 10000\n")
        f.write("- All measurements in microseconds (µs)\n")
    
    print("Markdown report saved as 'benchmark_report.md'")

def main():
    """Main entry point"""
    # Change to benches directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run benchmarks
    rust_results = run_rust_benchmarks()
    scipy_results = run_scipy_benchmarks()
    
    if not rust_results or not scipy_results:
        print("Failed to collect benchmark results")
        return
    
    # Calculate speedup and generate report
    df = calculate_speedup(rust_results, scipy_results)
    
    if df.empty:
        print("No matching benchmarks found between Rust and SciPy")
        return
    
    generate_report(df)

if __name__ == "__main__":
    main()