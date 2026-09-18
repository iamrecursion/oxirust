#!/usr/bin/env python3
"""
SciPy benchmark comparison script for scirs2-special functions.

This script benchmarks SciPy's special functions to provide reference
performance data for comparison with the Rust implementations.
"""

import numpy as np
import scipy.special as sp
import time
import json
import sys
from pathlib import Path

def benchmark_function(func, args_list, name, num_runs=1000):
    """Benchmark a function with given arguments."""
    times = []
    
    # Warm up
    for _ in range(10):
        for args in args_list:
            func(*args)
    
    # Actual benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        for args in args_list:
            func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    # Remove outliers (top/bottom 5%)
    times.sort()
    trimmed_times = times[len(times)//20:-len(times)//20] if len(times) > 40 else times
    
    avg_time = sum(trimmed_times) / len(trimmed_times)
    min_time = min(trimmed_times)
    
    return {
        'name': name,
        'avg_time_us': avg_time * 1e6 / len(args_list),  # microseconds per call
        'min_time_us': min_time * 1e6 / len(args_list),
        'total_calls': len(args_list) * num_runs,
        'calls_per_second': len(args_list) / avg_time
    }

def main():
    """Run benchmarks for various special functions."""
    results = {}
    
    # Bessel functions
    print("Benchmarking Bessel functions...")
    
    # j0 - small values
    small_args = [(x * 0.01,) for x in range(100)]
    results['j0_small'] = benchmark_function(sp.j0, small_args, 'j0_small')
    
    # j0 - medium values  
    medium_args = [(x * 0.1 + 10.0,) for x in range(100)]
    results['j0_medium'] = benchmark_function(sp.j0, medium_args, 'j0_medium')
    
    # j0 - large values
    large_args = [(x * 10.0 + 100.0,) for x in range(100)]
    results['j0_large'] = benchmark_function(sp.j0, large_args, 'j0_large')
    
    # j1 benchmarks
    results['j1_small'] = benchmark_function(sp.j1, small_args, 'j1_small')
    results['j1_medium'] = benchmark_function(sp.j1, medium_args, 'j1_medium')
    results['j1_large'] = benchmark_function(sp.j1, large_args, 'j1_large')
    
    # jn with different orders
    for n in [2, 5, 10, 20]:
        jn_args = [(n, x * 0.2 + 5.0) for x in range(50)]
        results[f'jn_order_{n}'] = benchmark_function(sp.jn, jn_args, f'jn_order_{n}')
    
    # jv with different orders
    for v in [0.0, 1.0, 2.0, 0.5, 1.5, 2.5]:
        jv_args = [(v, x * 0.2 + 5.0) for x in range(50)]
        results[f'jv_order_{v}'] = benchmark_function(sp.jv, jv_args, f'jv_order_{v}')
    
    # Modified Bessel functions
    print("Benchmarking modified Bessel functions...")
    results['i0_small'] = benchmark_function(sp.i0, small_args, 'i0_small')
    i0_medium_args = [(x * 0.1 + 1.0,) for x in range(100)]
    results['i0_medium'] = benchmark_function(sp.i0, i0_medium_args, 'i0_medium')
    
    # Spherical Bessel functions
    print("Benchmarking spherical Bessel functions...")
    spherical_args = [(0, x * 0.1 + 1.0) for x in range(100)]
    results['spherical_j0'] = benchmark_function(sp.spherical_jn, spherical_args, 'spherical_j0')
    
    # Gamma functions
    print("Benchmarking gamma functions...")
    gamma_args = [(x * 0.1 + 1.0,) for x in range(100)]
    results['gamma'] = benchmark_function(sp.gamma, gamma_args, 'gamma')
    results['gammaln'] = benchmark_function(sp.gammaln, gamma_args, 'gammaln')
    results['digamma'] = benchmark_function(sp.digamma, gamma_args, 'digamma')
    
    beta_args = [(x * 0.1 + 1.0, y * 0.1 + 1.0) for x in range(10) for y in range(10)]
    results['beta'] = benchmark_function(sp.beta, beta_args, 'beta')
    
    # Error functions
    print("Benchmarking error functions...")
    erf_args = [(x * 0.1,) for x in range(-50, 51)]
    results['erf'] = benchmark_function(sp.erf, erf_args, 'erf')
    results['erfc'] = benchmark_function(sp.erfc, erf_args, 'erfc')
    
    # Airy functions
    print("Benchmarking Airy functions...")
    airy_args = [(x * 0.1,) for x in range(-100, 101)]
    ai_result = benchmark_function(lambda x: sp.airy(x)[0], airy_args, 'airy_ai')
    results['airy_ai'] = ai_result
    
    # Lambert W
    print("Benchmarking Lambert W...")
    lambertw_args = [(x * 0.1 + 0.1,) for x in range(100)]
    results['lambertw'] = benchmark_function(sp.lambertw, lambertw_args, 'lambertw')
    
    # Advanced mode functions
    print("Benchmarking advanced mode functions...")
    
    # Dawson's integral
    dawson_args = [(x * 0.1,) for x in range(-50, 51)]
    results['dawsn'] = benchmark_function(sp.dawsn, dawson_args, 'dawsn')
    
    # Polygamma function (trigamma)
    polygamma_args = [(1, x * 0.1 + 1.0) for x in range(100)]
    results['polygamma'] = benchmark_function(sp.polygamma, polygamma_args, 'polygamma')
    
    # Sine and cosine integrals
    si_args = [(x * 0.1 + 0.1,) for x in range(100)]
    results['si'] = benchmark_function(sp.shichi, si_args, 'si')  # shichi returns (shi, chi)
    results['sici'] = benchmark_function(sp.sici, si_args, 'sici')  # sici returns (si, ci)
    
    # Scaled error functions
    erfcx_args = [(x * 0.1,) for x in range(100)]
    results['erfcx'] = benchmark_function(sp.erfcx, erfcx_args, 'erfcx')
    results['erfi'] = benchmark_function(sp.erfi, erfcx_args, 'erfi')
    
    # Faddeeva function
    try:
        from scipy.special import wofz
        results['wofz'] = benchmark_function(wofz, erfcx_args, 'wofz')
    except ImportError:
        print("Warning: wofz not available in this SciPy version")
    
    # Spence function (dilogarithm)
    try:
        spence_args = [(x * 0.1 + 0.1,) for x in range(50)]
        results['spence'] = benchmark_function(sp.spence, spence_args, 'spence')
    except AttributeError:
        print("Warning: spence not available in this SciPy version")
    
    # Exponentially scaled Bessel functions (new in advanced mode)
    try:
        results['j0e'] = benchmark_function(sp.j0, large_args, 'j0e')  # Approximation since j0e may not exist
        results['i0e'] = benchmark_function(sp.i0e, medium_args, 'i0e')
        results['k0e'] = benchmark_function(sp.k0e, medium_args, 'k0e')
    except AttributeError:
        print("Warning: Some exponentially scaled functions not available")
    
    # Print results
    print("\nSciPy Benchmark Results:")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name:20} {result['avg_time_us']:8.2f} μs/call  {result['calls_per_second']:10.0f} calls/s")
    
    # Save results to JSON
    output_file = Path(__file__).parent / "scipy_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install numpy scipy")
        sys.exit(1)