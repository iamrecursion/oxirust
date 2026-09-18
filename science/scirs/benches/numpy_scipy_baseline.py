#!/usr/bin/env python3
"""
NumPy/SciPy Baseline Performance Benchmarks

This script provides baseline performance measurements for NumPy/SciPy operations
that correspond to the Rust benchmarks in numpy_scipy_comparison_bench.rs
"""

import numpy as np
import scipy.stats
import time
import json
from typing import Dict, List, Tuple
import platform
import psutil

# Test sizes matching the Rust benchmarks
SIZES = [100, 1000, 10000, 100000]
MATRIX_SIZES = [10, 50, 100, 500, 1000]
ITERATIONS = 100

def timeit(func, iterations: int = ITERATIONS) -> float:
    """Time a function execution over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times)

def benchmark_array_creation() -> Dict[str, Dict[int, float]]:
    """Benchmark array creation and initialization operations."""
    results = {
        "zeros": {},
        "ones": {},
        "random": {},
        "linspace": {}
    }
    
    for size in SIZES:
        # Zeros initialization
        results["zeros"][size] = timeit(lambda: np.zeros(size))
        
        # Ones initialization
        results["ones"][size] = timeit(lambda: np.ones(size))
        
        # Random initialization
        results["random"][size] = timeit(lambda: np.random.random(size))
        
        # Linspace
        results["linspace"][size] = timeit(lambda: np.linspace(0, 100, size))
    
    return results

def benchmark_element_wise_ops() -> Dict[str, Dict[int, float]]:
    """Benchmark element-wise operations (ufuncs)."""
    results = {
        "add": {},
        "multiply": {},
        "sqrt": {},
        "exp": {},
        "sin": {}
    }
    
    for size in SIZES:
        arr1 = np.random.random(size)
        arr2 = np.random.random(size)
        
        # Addition
        results["add"][size] = timeit(lambda: arr1 + arr2)
        
        # Multiplication
        results["multiply"][size] = timeit(lambda: arr1 * arr2)
        
        # Square root
        results["sqrt"][size] = timeit(lambda: np.sqrt(arr1))
        
        # Exponential
        results["exp"][size] = timeit(lambda: np.exp(arr1))
        
        # Trigonometric
        results["sin"][size] = timeit(lambda: np.sin(arr1))
    
    return results

def benchmark_reduction_ops() -> Dict[str, Dict[int, float]]:
    """Benchmark reduction operations."""
    results = {
        "sum": {},
        "mean": {},
        "std": {},
        "min_max": {}
    }
    
    for size in SIZES:
        arr = np.random.random(size)
        
        # Sum
        results["sum"][size] = timeit(lambda: np.sum(arr))
        
        # Mean
        results["mean"][size] = timeit(lambda: np.mean(arr))
        
        # Standard deviation
        results["std"][size] = timeit(lambda: np.std(arr))
        
        # Min/Max
        results["min_max"][size] = timeit(lambda: (np.min(arr), np.max(arr)))
    
    return results

def benchmark_matrix_ops() -> Dict[str, Dict[int, float]]:
    """Benchmark matrix operations."""
    results = {
        "matmul": {},
        "matvec": {},
        "transpose": {},
        "diagonal": {}
    }
    
    for size in MATRIX_SIZES:
        mat_a = np.random.random((size, size))
        mat_b = np.random.random((size, size))
        vec = np.random.random(size)
        
        # Matrix multiplication
        results["matmul"][size] = timeit(lambda: np.dot(mat_a, mat_b), iterations=50)
        
        # Matrix-vector multiplication
        results["matvec"][size] = timeit(lambda: np.dot(mat_a, vec))
        
        # Transpose
        results["transpose"][size] = timeit(lambda: mat_a.T.copy())
        
        # Diagonal
        results["diagonal"][size] = timeit(lambda: np.diag(mat_a).copy())
    
    return results

def benchmark_array_manipulation() -> Dict[str, Dict[int, float]]:
    """Benchmark array manipulation operations."""
    results = {
        "reshape": {},
        "concatenate": {},
        "slice": {},
        "sort": {}
    }
    
    for size in SIZES:
        arr = np.random.random(size)
        
        # Reshape
        if size >= 100:
            rows = 10
            cols = size // rows
            if rows * cols == size:
                results["reshape"][size] = timeit(lambda: arr.reshape(rows, cols))
        
        # Concatenation
        arr2 = np.random.random(size)
        results["concatenate"][size] = timeit(lambda: np.concatenate([arr, arr2]))
        
        # Slicing
        mid = size // 2
        results["slice"][size] = timeit(lambda: arr[:mid].copy())
        
        # Sorting
        results["sort"][size] = timeit(lambda: np.sort(arr.copy()))
    
    return results

def benchmark_statistical_ops() -> Dict[str, Dict[int, float]]:
    """Benchmark statistical operations."""
    results = {
        "variance": {},
        "covariance": {},
        "median": {}
    }
    
    for size in SIZES:
        arr1 = np.random.random(size)
        arr2 = np.random.random(size)
        
        # Variance
        results["variance"][size] = timeit(lambda: np.var(arr1))
        
        # Covariance (simplified - single value)
        results["covariance"][size] = timeit(lambda: np.cov(arr1, arr2)[0, 1])
        
        # Median
        results["median"][size] = timeit(lambda: np.median(arr1))
    
    return results

def benchmark_memory_ops() -> Dict[str, Dict[int, float]]:
    """Benchmark memory-intensive operations."""
    results = {
        "copy": {},
        "view": {},
        "alloc_pattern": {}
    }
    
    for size in [1000, 10000, 50000]:
        arr = np.random.random(size)
        
        # Copy
        results["copy"][size] = timeit(lambda: arr.copy())
        
        # View
        results["view"][size] = timeit(lambda: arr.view())
        
        # Allocation pattern
        def alloc_pattern():
            arrays = []
            for _ in range(10):
                arrays.append(np.zeros(size // 10))
            return arrays
        
        results["alloc_pattern"][size] = timeit(alloc_pattern)
    
    return results

def get_system_info() -> Dict[str, str]:
    """Get system information for the benchmark report."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }

def format_results(results: Dict[str, Dict[str, Dict[int, float]]]) -> str:
    """Format benchmark results as a markdown table."""
    output = []
    output.append("# NumPy/SciPy Baseline Performance Results\n")
    
    # System info
    sys_info = get_system_info()
    output.append("## System Information")
    for key, value in sys_info.items():
        output.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    output.append("")
    
    # Results by category
    for category, ops in results.items():
        output.append(f"\n## {category.replace('_', ' ').title()}")
        output.append("\n| Operation | Size | Time (μs) | Throughput (ops/sec) |")
        output.append("|-----------|------|-----------|---------------------|")
        
        for op_name, sizes in ops.items():
            for size, time_sec in sizes.items():
                time_us = time_sec * 1_000_000
                throughput = 1 / time_sec if time_sec > 0 else 0
                output.append(f"| {op_name} | {size:,} | {time_us:.2f} | {throughput:,.0f} |")
    
    return "\n".join(output)

def main():
    """Run all benchmarks and save results."""
    print("Running NumPy/SciPy baseline benchmarks...")
    
    results = {
        "array_creation": benchmark_array_creation(),
        "element_wise_ops": benchmark_element_wise_ops(),
        "reduction_ops": benchmark_reduction_ops(),
        "matrix_ops": benchmark_matrix_ops(),
        "array_manipulation": benchmark_array_manipulation(),
        "statistical_ops": benchmark_statistical_ops(),
        "memory_ops": benchmark_memory_ops()
    }
    
    # Save raw results as JSON
    with open("numpy_scipy_baseline_results.json", "w") as f:
        json.dump({
            "system_info": get_system_info(),
            "results": results
        }, f, indent=2)
    
    # Save formatted results as markdown
    formatted = format_results(results)
    with open("numpy_scipy_baseline_results.md", "w") as f:
        f.write(formatted)
    
    print("Benchmarks complete. Results saved to:")
    print("  - numpy_scipy_baseline_results.json")
    print("  - numpy_scipy_baseline_results.md")

if __name__ == "__main__":
    main()