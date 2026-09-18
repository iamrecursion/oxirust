#!/usr/bin/env python3
"""
Compare benchmark results between scirs2-stats and SciPy.

This script parses the output from both Rust criterion benchmarks and
Python SciPy benchmarks to generate a comparison report.
"""

import json
import sys
from pathlib import Path
import re

def parse_criterion_output(output_file):
    """Parse criterion benchmark output."""
    results = {}
    
    # Criterion output format varies, this is a simplified parser
    # In practice, you would use criterion's JSON output format
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    current_group = None
    for line in lines:
        # Parse group headers
        if line.startswith('Benchmarking'):
            match = re.search(r'Benchmarking (\w+)', line)
            if match:
                current_group = match.group(1)
        
        # Parse timing results
        if 'time:' in line and current_group:
            match = re.search(r'time:\s+\[([\d.]+)\s+(\w+)', line)
            if match:
                time = float(match.group(1))
                unit = match.group(2)
                
                # Convert to milliseconds
                if unit == 'ns':
                    time = time / 1_000_000
                elif unit == 'us':
                    time = time / 1_000
                elif unit == 's':
                    time = time * 1_000
                
                results[current_group] = time
    
    return results

def load_scipy_results(json_file):
    """Load SciPy benchmark results from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)

def compare_results(rust_results, scipy_results):
    """Compare Rust and SciPy benchmark results."""
    comparison = {}
    
    # Map test names between Rust and Python
    test_mapping = {
        'normal_pdf': 'normal_pdf',
        't_pdf': 't_pdf',
        'normal_cdf': 'normal_cdf',
        'ttest_1samp': 'ttest_1samp',
        'ttest_ind': 'ttest_ind',
        'pearson': 'pearson',
        'spearman': 'spearman',
        'mean': 'mean',
        'variance': 'variance',
    }
    
    for rust_name, scipy_name in test_mapping.items():
        rust_times = [v for k, v in rust_results.items() if rust_name in k]
        scipy_times = []
        
        for category in scipy_results.values():
            scipy_times.extend([v['mean'] * 1000 for k, v in category.items() if scipy_name in k])
        
        if rust_times and scipy_times:
            comparison[rust_name] = {
                'rust_avg': sum(rust_times) / len(rust_times),
                'scipy_avg': sum(scipy_times) / len(scipy_times),
                'speedup': sum(scipy_times) / sum(rust_times)
            }
    
    return comparison

def generate_report(comparison):
    """Generate a comparison report."""
    print("=== scirs2-stats vs SciPy Performance Comparison ===\n")
    print(f"{'Test':<20} {'Rust (ms)':<12} {'SciPy (ms)':<12} {'Speedup':<10}")
    print("-" * 54)
    
    total_rust = 0
    total_scipy = 0
    
    for test, data in sorted(comparison.items()):
        rust_time = data['rust_avg']
        scipy_time = data['scipy_avg']
        speedup = data['speedup']
        
        total_rust += rust_time
        total_scipy += scipy_time
        
        print(f"{test:<20} {rust_time:<12.3f} {scipy_time:<12.3f} {speedup:<10.2f}x")
    
    print("-" * 54)
    overall_speedup = total_scipy / total_rust if total_rust > 0 else 0
    print(f"{'Overall':<20} {total_rust:<12.3f} {total_scipy:<12.3f} {overall_speedup:<10.2f}x")
    
    print("\nNote: These are approximate comparisons. Actual performance may vary based on:")
    print("- Hardware configuration")
    print("- Compiler optimizations")
    print("- Data characteristics")
    print("- Operating system and background processes")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compare_benchmarks.py <criterion_output.txt> [scipy_results.json]")
        sys.exit(1)
    
    criterion_file = sys.argv[1]
    scipy_file = sys.argv[2] if len(sys.argv) > 2 else "scipy_benchmark_results.json"
    
    # Parse results
    rust_results = parse_criterion_output(criterion_file)
    scipy_results = load_scipy_results(scipy_file)
    
    # Compare and report
    comparison = compare_results(rust_results, scipy_results)
    generate_report(comparison)

if __name__ == "__main__":
    main()