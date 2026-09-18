#!/usr/bin/env python3
"""
Comprehensive benchmark suite for SciPy to compare with scirs2-stats

This script runs the same benchmarks as comprehensive_scipy_comparison.rs
using SciPy functions to enable direct performance comparison.
"""

import numpy as np
import scipy.stats as stats
from scipy import special
import scipy.linalg
import time
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd

class BenchmarkRunner:
    """Manages benchmark execution and timing"""
    
    def __init__(self, warmup_time: float = 1.0, measurement_time: float = 3.0):
        self.warmup_time = warmup_time
        self.measurement_time = measurement_time
        self.results: Dict[str, Dict[int, float]] = {}
    
    def run_benchmark(self, name: str, func, data_sizes: List[int], 
                     data_generator, *args, **kwargs) -> None:
        """Run a benchmark across different data sizes"""
        if name not in self.results:
            self.results[name] = {}
        
        for size in data_sizes:
            data = data_generator(size, *args, **kwargs)
            
            # Warmup
            start_warmup = time.time()
            while time.time() - start_warmup < self.warmup_time:
                if isinstance(data, tuple):
                    func(*data)
                else:
                    func(data)
            
            # Measurement
            start_measure = time.time()
            iterations = 0
            while time.time() - start_measure < self.measurement_time:
                if isinstance(data, tuple):
                    func(*data)
                else:
                    func(data)
                iterations += 1
            
            elapsed = time.time() - start_measure
            avg_time = elapsed / iterations
            self.results[name][size] = avg_time * 1e6  # Convert to microseconds
            
            print(f"{name} (n={size}): {avg_time*1e6:.2f} µs")

# Data generators
def generate_normal_data(n: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """Generate normal distributed data"""
    return np.random.normal(mean, std, n)

def generate_uniform_data(n: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Generate uniform distributed data"""
    return np.random.uniform(low, high, n)

def generate_exponential_data(n: int, scale: float = 1.0) -> np.ndarray:
    """Generate exponential distributed data"""
    return np.random.exponential(scale, n)

def generate_multivariate_normal(n: int, dim: int) -> np.ndarray:
    """Generate multivariate normal data"""
    return np.random.randn(n, dim)

def generate_correlated_data(n: int, correlation: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated data"""
    x = np.random.randn(n)
    noise = np.random.randn(n)
    y = correlation * x + np.sqrt(1 - correlation**2) * noise
    return x, y

def benchmark_descriptive_stats(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark descriptive statistics functions"""
    print("\n=== Descriptive Statistics ===")
    
    # Mean
    runner.run_benchmark("mean", np.mean, sample_sizes, generate_normal_data)
    
    # Variance
    runner.run_benchmark("variance", lambda x: np.var(x, ddof=1), 
                        sample_sizes, generate_normal_data)
    
    # Standard deviation
    runner.run_benchmark("std", lambda x: np.std(x, ddof=1), 
                        sample_sizes, generate_normal_data)
    
    # Skewness
    runner.run_benchmark("skewness", stats.skew, sample_sizes, generate_normal_data)
    
    # Kurtosis
    runner.run_benchmark("kurtosis", stats.kurtosis, sample_sizes, generate_normal_data)
    
    # Median
    runner.run_benchmark("median", np.median, sample_sizes, generate_normal_data)

def benchmark_correlation(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark correlation functions"""
    print("\n=== Correlation Functions ===")
    
    # Pearson correlation
    runner.run_benchmark("pearson", lambda data: stats.pearsonr(data[0], data[1]), 
                        sample_sizes, generate_correlated_data)
    
    # Spearman correlation
    runner.run_benchmark("spearman", lambda data: stats.spearmanr(data[0], data[1]), 
                        sample_sizes, generate_correlated_data)
    
    # Kendall tau (only for smaller samples due to O(n²) complexity)
    kendall_sizes = [s for s in sample_sizes if s <= 1000]
    runner.run_benchmark("kendalltau", lambda data: stats.kendalltau(data[0], data[1]), 
                        kendall_sizes, generate_correlated_data)

def benchmark_statistical_tests(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark statistical tests"""
    print("\n=== Statistical Tests ===")
    
    # One-sample t-test
    runner.run_benchmark("ttest_1samp", lambda x: stats.ttest_1samp(x, 0.0), 
                        sample_sizes, generate_normal_data)
    
    # Independent t-test
    def ttest_ind_bench(n):
        x1 = generate_normal_data(n, 0.0, 1.0)
        x2 = generate_normal_data(n, 0.5, 1.0)
        return (x1, x2)
    
    runner.run_benchmark("ttest_ind", lambda data: stats.ttest_ind(data[0], data[1]), 
                        sample_sizes, ttest_ind_bench)
    
    # Mann-Whitney U test
    runner.run_benchmark("mann_whitney", lambda data: stats.mannwhitneyu(data[0], data[1]), 
                        sample_sizes, ttest_ind_bench)
    
    # Shapiro-Wilk test (expensive for large samples)
    shapiro_sizes = [s for s in sample_sizes if s <= 5000]
    runner.run_benchmark("shapiro_wilk", stats.shapiro, 
                        shapiro_sizes, generate_normal_data)
    
    # Anderson-Darling test
    runner.run_benchmark("anderson_darling", lambda x: stats.anderson(x, 'norm'), 
                        sample_sizes, generate_normal_data)

def benchmark_distributions(runner: BenchmarkRunner, eval_points: List[int]) -> None:
    """Benchmark distribution functions"""
    print("\n=== Distribution Functions ===")
    
    # Normal distribution PDF
    def normal_pdf_bench(n):
        x = generate_uniform_data(n, -3.0, 3.0)
        return x
    
    runner.run_benchmark("normal_pdf", lambda x: stats.norm.pdf(x), 
                        eval_points, normal_pdf_bench)
    
    # Normal distribution CDF
    runner.run_benchmark("normal_cdf", lambda x: stats.norm.cdf(x), 
                        eval_points, normal_pdf_bench)
    
    # Student's t distribution PDF
    runner.run_benchmark("t_pdf", lambda x: stats.t.pdf(x, df=10), 
                        eval_points, normal_pdf_bench)
    
    # Chi-square distribution PDF
    def chi2_pdf_bench(n):
        x = generate_uniform_data(n, 0.1, 10.0)
        return x
    
    runner.run_benchmark("chi2_pdf", lambda x: stats.chi2.pdf(x, df=5), 
                        eval_points, chi2_pdf_bench)

def benchmark_regression(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark regression functions"""
    print("\n=== Regression ===")
    
    # Simple linear regression
    def linear_regression_data(n):
        x = generate_uniform_data(n, 0.0, 10.0)
        y = 2.0 * x + generate_normal_data(n, 0.0, 0.1)
        return (x, y)
    
    runner.run_benchmark("linear_regression", 
                        lambda data: stats.linregress(data[0], data[1]), 
                        sample_sizes, linear_regression_data)
    
    # Polynomial regression (using numpy polyfit)
    runner.run_benchmark("polynomial_regression_deg3", 
                        lambda data: np.polyfit(data[0], data[1], 3), 
                        sample_sizes, linear_regression_data)

def benchmark_quantiles(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark quantile functions"""
    print("\n=== Quantiles ===")
    
    # Single quantile (median)
    runner.run_benchmark("quantile_50", lambda x: np.quantile(x, 0.5), 
                        sample_sizes, generate_normal_data)
    
    # Multiple quantiles
    runner.run_benchmark("quantiles_multiple", 
                        lambda x: np.quantile(x, [0.25, 0.5, 0.75]), 
                        sample_sizes, generate_normal_data)

def benchmark_dispersion(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark dispersion measures"""
    print("\n=== Dispersion Measures ===")
    
    # MAD (Median Absolute Deviation)
    runner.run_benchmark("mad", lambda x: stats.median_abs_deviation(x), 
                        sample_sizes, generate_normal_data)
    
    # IQR (Interquartile Range)
    runner.run_benchmark("iqr", lambda x: stats.iqr(x), 
                        sample_sizes, generate_normal_data)
    
    # Coefficient of Variation
    runner.run_benchmark("cv", lambda x: stats.variation(x), 
                        sample_sizes, generate_normal_data)
    
    # Gini coefficient
    def gini_coefficient(x):
        """Calculate Gini coefficient"""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(1, n+1) * sorted_x))) / (n * cumsum[-1]) - (n + 1) / n
    
    runner.run_benchmark("gini", gini_coefficient, 
                        sample_sizes, generate_normal_data)

def benchmark_random_sampling(runner: BenchmarkRunner, sample_sizes: List[int]) -> None:
    """Benchmark random sampling functions"""
    print("\n=== Random Sampling ===")
    
    # Choice with replacement
    def choice_with_replacement(data):
        n = len(data)
        return np.random.choice(data, size=min(100, n), replace=True)
    
    runner.run_benchmark("choice_with_replacement", choice_with_replacement, 
                        sample_sizes, generate_normal_data)
    
    # Choice without replacement
    def choice_without_replacement(data):
        n = len(data)
        if n >= 100:
            return np.random.choice(data, size=min(100, n), replace=False)
        return data
    
    runner.run_benchmark("choice_without_replacement", choice_without_replacement, 
                        [s for s in sample_sizes if s >= 100], generate_normal_data)
    
    # Permutation
    runner.run_benchmark("permutation", np.random.permutation, 
                        sample_sizes, lambda n: np.arange(n))

def save_results(results: Dict[str, Dict[int, float]], filename: str = "scipy_benchmark_results.json"):
    """Save benchmark results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def plot_comparison(scipy_results: Dict[str, Dict[int, float]], 
                   rust_results: Dict[str, Dict[int, float]] = None):
    """Plot comparison between SciPy and Rust implementations"""
    # This function can be extended to load and compare with Rust results
    pass

def main():
    """Run all benchmarks"""
    # Configuration
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    eval_points = [10, 100, 1000, 10000]
    
    # Create benchmark runner
    runner = BenchmarkRunner(warmup_time=1.0, measurement_time=3.0)
    
    # Run all benchmarks
    benchmark_descriptive_stats(runner, sample_sizes)
    benchmark_correlation(runner, sample_sizes)
    benchmark_statistical_tests(runner, sample_sizes)
    benchmark_distributions(runner, eval_points)
    benchmark_regression(runner, sample_sizes)
    benchmark_quantiles(runner, sample_sizes)
    benchmark_dispersion(runner, sample_sizes)
    benchmark_random_sampling(runner, sample_sizes)
    
    # Save results
    save_results(runner.results)
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    for func_name, sizes in runner.results.items():
        print(f"\n{func_name}:")
        for size, time_us in sizes.items():
            print(f"  n={size}: {time_us:.2f} µs")

if __name__ == "__main__":
    main()