#!/usr/bin/env python3
"""
SciPy benchmark suite for comparison with SciRS2-Optimize

This script runs the same optimization problems using SciPy's optimize module
to generate comparison data.
"""

import numpy as np
import time
import json
from scipy import optimize
from typing import Dict, List, Tuple, Callable
import pandas as pd


class TestFunctions:
    """Standard test functions for optimization benchmarking"""
    
    @staticmethod
    def rosenbrock(x):
        """Rosenbrock function (2D)"""
        a = 1.0
        b = 100.0
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    @staticmethod
    def rastrigin(x):
        """Rastrigin function (N-dimensional)"""
        a = 10.0
        n = len(x)
        return a * n + sum(xi**2 - a * np.cos(2 * np.pi * xi) for xi in x)
    
    @staticmethod
    def ackley(x):
        """Ackley function (N-dimensional)"""
        a = 20.0
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        sum1 = sum(xi**2 for xi in x) / n
        sum2 = sum(np.cos(c * xi) for xi in x) / n
        return -a * np.exp(-b * np.sqrt(sum1)) - np.exp(sum2) + a + np.e
    
    @staticmethod
    def sphere(x):
        """Sphere function (N-dimensional)"""
        return sum(xi**2 for xi in x)
    
    @staticmethod
    def beale(x):
        """Beale function (2D)"""
        x1, x2 = x[0], x[1]
        return ((1.5 - x1 + x1*x2)**2 + 
                (2.25 - x1 + x1*x2**2)**2 + 
                (2.625 - x1 + x1*x2**3)**2)
    
    @staticmethod
    def goldstein_price(x):
        """Goldstein-Price function (2D)"""
        x1, x2 = x[0], x[1]
        a = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        b = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        return a * b
    
    @staticmethod
    def himmelblau(x):
        """Himmelblau function (2D)"""
        x1, x2 = x[0], x[1]
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    
    @staticmethod
    def booth(x):
        """Booth function (2D)"""
        x1, x2 = x[0], x[1]
        return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
    
    @staticmethod
    def matyas(x):
        """Matyas function (2D)"""
        x1, x2 = x[0], x[1]
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
    
    @staticmethod
    def levy(x):
        """Levy function (N-dimensional)"""
        n = len(x)
        w = [1 + (xi - 1) / 4 for xi in x]
        term1 = np.sin(np.pi * w[0])**2
        term2 = sum((w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2) 
                   for i in range(n-1))
        term3 = (w[n-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[n-1])**2)
        return term1 + term2 + term3


def get_benchmark_problems():
    """Get standard benchmark problems"""
    return [
        {
            'name': 'Rosenbrock',
            'function': TestFunctions.rosenbrock,
            'initial_points': [
                np.array([0.0, 0.0]),
                np.array([-1.0, 1.0]),
                np.array([2.0, 2.0]),
            ],
            'optimal_value': 0.0,
            'dimensions': [2],
        },
        {
            'name': 'Sphere',
            'function': TestFunctions.sphere,
            'initial_points': [
                np.array([1.0, 1.0]),
                np.array([5.0, 5.0]),
            ],
            'optimal_value': 0.0,
            'dimensions': [2, 10, 50],
        },
        {
            'name': 'Rastrigin',
            'function': TestFunctions.rastrigin,
            'initial_points': [
                np.array([1.0, 1.0]),
                np.array([4.0, 4.0]),
            ],
            'optimal_value': 0.0,
            'dimensions': [2, 10],
        },
        {
            'name': 'Ackley',
            'function': TestFunctions.ackley,
            'initial_points': [
                np.array([1.0, 1.0]),
                np.array([2.5, 2.5]),
            ],
            'optimal_value': 0.0,
            'dimensions': [2, 10],
        },
        {
            'name': 'Beale',
            'function': TestFunctions.beale,
            'initial_points': [
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
            ],
            'optimal_value': 0.0,
            'dimensions': [2],
        },
        {
            'name': 'Himmelblau',
            'function': TestFunctions.himmelblau,
            'initial_points': [
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0]),
            ],
            'optimal_value': 0.0,
            'dimensions': [2],
        },
    ]


def benchmark_scipy_method(func: Callable, x0: np.ndarray, method: str, 
                         options: Dict = None) -> Dict:
    """Benchmark a single SciPy optimization method"""
    
    # Set default options
    if options is None:
        options = {'maxiter': 1000}
    
    # Time the optimization
    start_time = time.time()
    result = optimize.minimize(func, x0, method=method, options=options)
    end_time = time.time()
    
    return {
        'time': end_time - start_time,
        'iterations': result.nit if hasattr(result, 'nit') else -1,
        'function_evals': result.nfev if hasattr(result, 'nfev') else -1,
        'gradient_evals': result.njev if hasattr(result, 'njev') else -1,
        'success': result.success,
        'final_value': result.fun,
        'final_x': result.x.tolist(),
        'message': result.message if hasattr(result, 'message') else str(result),
    }


def run_unconstrained_benchmarks():
    """Run benchmarks for unconstrained optimization methods"""
    
    problems = get_benchmark_problems()
    methods = ['BFGS', 'L-BFGS-B', 'CG', 'Nelder-Mead', 'Powell']
    
    results = []
    
    for problem in problems[:3]:  # Use first 3 problems
        print(f"\nBenchmarking {problem['name']}...")
        
        for method in methods:
            print(f"  Method: {method}")
            
            for i, x0 in enumerate(problem['initial_points']):
                # Run multiple trials
                times = []
                for trial in range(10):
                    result = benchmark_scipy_method(
                        problem['function'], x0, method
                    )
                    times.append(result['time'])
                
                # Get final result for other metrics
                final_result = benchmark_scipy_method(
                    problem['function'], x0, method
                )
                
                results.append({
                    'problem': problem['name'],
                    'method': method,
                    'initial_point': i,
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    **final_result
                })
    
    return results


def run_dimension_benchmarks():
    """Run benchmarks for different problem dimensions"""
    
    sphere = TestFunctions.sphere
    dimensions = [2, 5, 10, 20, 50]
    methods = ['BFGS', 'L-BFGS-B']
    
    results = []
    
    print("\nBenchmarking different dimensions...")
    
    for dim in dimensions:
        print(f"  Dimension: {dim}")
        x0 = np.ones(dim)
        
        for method in methods:
            times = []
            for trial in range(10):
                result = benchmark_scipy_method(sphere, x0, method)
                times.append(result['time'])
            
            final_result = benchmark_scipy_method(sphere, x0, method)
            
            results.append({
                'dimension': dim,
                'method': method,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                **final_result
            })
    
    return results


def run_global_benchmarks():
    """Run benchmarks for global optimization methods"""
    
    problems = [
        ('Rastrigin', TestFunctions.rastrigin),
        ('Ackley', TestFunctions.ackley),
    ]
    
    results = []
    
    print("\nBenchmarking global optimization...")
    
    for name, func in problems:
        print(f"  Problem: {name}")
        bounds = [(-5.0, 5.0)] * 5
        
        # Differential Evolution
        times = []
        for trial in range(5):
            start_time = time.time()
            result = optimize.differential_evolution(
                func, bounds, maxiter=100, popsize=50
            )
            end_time = time.time()
            times.append(end_time - start_time)
        
        results.append({
            'problem': name,
            'method': 'differential_evolution',
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'success': result.success,
            'final_value': result.fun,
            'iterations': result.nit,
            'function_evals': result.nfev,
        })
        
        # Basin-hopping
        x0 = np.random.uniform(-5, 5, 5)
        times = []
        for trial in range(5):
            start_time = time.time()
            result = optimize.basinhopping(func, x0, niter=100)
            end_time = time.time()
            times.append(end_time - start_time)
        
        results.append({
            'problem': name,
            'method': 'basinhopping',
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'final_value': result.fun,
            'iterations': result.nit,
            'function_evals': result.nfev,
        })
    
    return results


def run_least_squares_benchmarks():
    """Run benchmarks for least squares problems"""
    
    def residual(params, x, y):
        """Linear model residual"""
        return y - (params[0] + params[1] * x)
    
    n_points_list = [10, 50, 100]
    results = []
    
    print("\nBenchmarking least squares...")
    
    for n in n_points_list:
        print(f"  N points: {n}")
        
        # Generate test data
        x_data = np.linspace(0, 1, n)
        y_data = 2.0 + 3.0 * x_data + 0.1 * np.random.randn(n)
        
        x0 = np.array([0.0, 0.0])
        
        # Levenberg-Marquardt
        times = []
        for trial in range(10):
            start_time = time.time()
            result = optimize.least_squares(
                residual, x0, args=(x_data, y_data), method='lm'
            )
            end_time = time.time()
            times.append(end_time - start_time)
        
        results.append({
            'n_points': n,
            'method': 'levenberg-marquardt',
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'success': result.success,
            'final_cost': result.cost,
            'iterations': result.nfev,
        })
        
        # Trust Region Reflective
        times = []
        for trial in range(10):
            start_time = time.time()
            result = optimize.least_squares(
                residual, x0, args=(x_data, y_data), method='trf'
            )
            end_time = time.time()
            times.append(end_time - start_time)
        
        results.append({
            'n_points': n,
            'method': 'trust-region-reflective',
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'success': result.success,
            'final_cost': result.cost,
            'iterations': result.nfev,
        })
    
    return results


def save_results(results: Dict[str, List[Dict]], filename: str = 'scipy_benchmark_results.json'):
    """Save benchmark results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


def generate_comparison_table(scipy_results: Dict, scirs_results: Dict = None):
    """Generate a comparison table between SciPy and SciRS2 results"""
    
    print("\n" + "="*80)
    print("SCIPY BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Unconstrained optimization results
    print("\nUnconstrained Optimization:")
    print("-"*60)
    df = pd.DataFrame(scipy_results['unconstrained'])
    summary = df.groupby(['problem', 'method'])['avg_time'].agg(['mean', 'std'])
    print(summary)
    
    # Dimension scaling results
    print("\nDimension Scaling:")
    print("-"*60)
    df = pd.DataFrame(scipy_results['dimensions'])
    pivot = df.pivot_table(values='avg_time', index='dimension', columns='method')
    print(pivot)
    
    # Global optimization results
    print("\nGlobal Optimization:")
    print("-"*60)
    df = pd.DataFrame(scipy_results['global'])
    summary = df[['problem', 'method', 'avg_time', 'final_value', 'iterations']]
    print(summary)
    
    # Least squares results
    print("\nLeast Squares:")
    print("-"*60)
    df = pd.DataFrame(scipy_results['least_squares'])
    pivot = df.pivot_table(values='avg_time', index='n_points', columns='method')
    print(pivot)


def main():
    """Run all benchmarks and save results"""
    
    print("Running SciPy optimization benchmarks...")
    print("This may take several minutes...")
    
    all_results = {
        'unconstrained': run_unconstrained_benchmarks(),
        'dimensions': run_dimension_benchmarks(),
        'global': run_global_benchmarks(),
        'least_squares': run_least_squares_benchmarks(),
    }
    
    # Save results
    save_results(all_results)
    
    # Generate summary table
    generate_comparison_table(all_results)
    
    print("\nBenchmarking complete!")


if __name__ == '__main__':
    main()