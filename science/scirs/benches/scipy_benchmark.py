#!/usr/bin/env python3
"""
Benchmark script for comparing SciPy performance with scirs2-stats.

This script runs equivalent benchmarks to those in the Rust implementation,
allowing for direct performance comparison.
"""

import numpy as np
import scipy.stats as stats
import time
from typing import List, Tuple
import json

def time_function(func, *args, iterations=100):
    """Time a function execution over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)

def benchmark_distributions():
    """Benchmark distribution operations."""
    results = {}
    
    # Test different sample sizes
    sample_sizes = [10, 100, 1000, 10000]
    
    for n in sample_sizes:
        x = np.linspace(-3, 3, n)
        
        # Normal distribution PDF
        norm_dist = stats.norm(0, 1)
        mean_time, std_time = time_function(lambda: [norm_dist.pdf(xi) for xi in x])
        results[f'normal_pdf_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Student's t distribution PDF
        t_dist = stats.t(5, 0, 1)
        mean_time, std_time = time_function(lambda: [t_dist.pdf(xi) for xi in x])
        results[f't_pdf_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Normal distribution CDF
        mean_time, std_time = time_function(lambda: [norm_dist.cdf(xi) for xi in x])
        results[f'normal_cdf_{n}'] = {'mean': mean_time, 'std': std_time}
    
    # Random number generation
    for n in [100, 1000, 10000, 100000]:
        mean_time, std_time = time_function(lambda: stats.norm.rvs(0, 1, size=n))
        results[f'normal_rvs_{n}'] = {'mean': mean_time, 'std': std_time}
        
        mean_time, std_time = time_function(lambda: stats.uniform.rvs(0, 1, size=n))
        results[f'uniform_rvs_{n}'] = {'mean': mean_time, 'std': std_time}
    
    return results

def benchmark_statistical_tests():
    """Benchmark statistical tests."""
    results = {}
    
    sample_sizes = [10, 50, 100, 500, 1000]
    
    for n in sample_sizes:
        # Generate test data
        data1 = np.random.normal(5.0, 1.0, n)
        data2 = np.random.normal(5.5, 1.0, n)
        
        # One-sample t-test
        mean_time, std_time = time_function(lambda: stats.ttest_1samp(data1, 5.0))
        results[f'ttest_1samp_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Independent samples t-test
        mean_time, std_time = time_function(lambda: stats.ttest_ind(data1, data2))
        results[f'ttest_ind_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Paired t-test
        mean_time, std_time = time_function(lambda: stats.ttest_rel(data1, data2))
        results[f'ttest_rel_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Mann-Whitney U test
        mean_time, std_time = time_function(lambda: stats.mannwhitneyu(data1, data2))
        results[f'mann_whitney_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Wilcoxon signed-rank test
        mean_time, std_time = time_function(lambda: stats.wilcoxon(data1 - 5.0))
        results[f'wilcoxon_{n}'] = {'mean': mean_time, 'std': std_time}
    
    # Normality tests
    for n in [20, 50, 100]:
        data = np.random.normal(0, 1, n)
        
        if n <= 50:  # Shapiro-Wilk has size limitations
            mean_time, std_time = time_function(lambda: stats.shapiro(data))
            results[f'shapiro_{n}'] = {'mean': mean_time, 'std': std_time}
        
        mean_time, std_time = time_function(lambda: stats.anderson(data))
        results[f'anderson_{n}'] = {'mean': mean_time, 'std': std_time}
    
    return results

def benchmark_correlations():
    """Benchmark correlation calculations."""
    results = {}
    
    sample_sizes = [10, 50, 100, 500, 1000]
    
    for n in sample_sizes:
        # Generate correlated data
        x = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 1, n)
        y = 0.8 * x + 0.2 * noise
        
        # Pearson correlation
        mean_time, std_time = time_function(lambda: stats.pearsonr(x, y))
        results[f'pearson_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Spearman correlation
        mean_time, std_time = time_function(lambda: stats.spearmanr(x, y))
        results[f'spearman_{n}'] = {'mean': mean_time, 'std': std_time}
        
        # Kendall tau (limit to smaller sizes due to O(n²) complexity)
        if n <= 100:
            mean_time, std_time = time_function(lambda: stats.kendalltau(x, y))
            results[f'kendall_{n}'] = {'mean': mean_time, 'std': std_time}
    
    return results

def benchmark_descriptive_stats():
    """Benchmark basic descriptive statistics."""
    results = {}
    
    sizes = [100, 1000, 10000, 100000, 1000000]
    
    for size in sizes:
        data = np.random.normal(0, 1, size)
        
        # Mean
        mean_time, std_time = time_function(lambda: np.mean(data))
        results[f'mean_{size}'] = {'mean': mean_time, 'std': std_time}
        
        # Variance
        mean_time, std_time = time_function(lambda: np.var(data, ddof=1))
        results[f'variance_{size}'] = {'mean': mean_time, 'std': std_time}
        
        # Standard deviation
        mean_time, std_time = time_function(lambda: np.std(data, ddof=1))
        results[f'std_{size}'] = {'mean': mean_time, 'std': std_time}
        
        # Skewness (for smaller sizes)
        if size <= 100000:
            mean_time, std_time = time_function(lambda: stats.skew(data))
            results[f'skewness_{size}'] = {'mean': mean_time, 'std': std_time}
        
        # Kurtosis (for smaller sizes)
        if size <= 100000:
            mean_time, std_time = time_function(lambda: stats.kurtosis(data, fisher=False))
            results[f'kurtosis_{size}'] = {'mean': mean_time, 'std': std_time}
    
    return results

def main():
    """Run all benchmarks and save results."""
    print("Running SciPy benchmarks...")
    
    all_results = {}
    
    print("Benchmarking distributions...")
    all_results['distributions'] = benchmark_distributions()
    
    print("Benchmarking statistical tests...")
    all_results['statistical_tests'] = benchmark_statistical_tests()
    
    print("Benchmarking correlations...")
    all_results['correlations'] = benchmark_correlations()
    
    print("Benchmarking descriptive statistics...")
    all_results['descriptive_stats'] = benchmark_descriptive_stats()
    
    # Save results to JSON
    with open('scipy_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Results saved to scipy_benchmark_results.json")
    
    # Print summary
    print("\n=== Summary ===")
    for category, results in all_results.items():
        print(f"\n{category}:")
        for test_name, times in results.items():
            print(f"  {test_name}: {times['mean']*1000:.3f} ± {times['std']*1000:.3f} ms")

if __name__ == "__main__":
    main()