"""
SciRS2 Async Operations Examples

Demonstrates async API for non-blocking scientific computing operations.
Useful for web applications, dashboards, and interactive notebooks.
"""

import asyncio
import time
import numpy as np
import scirs2


async def async_fft_example():
    """Async FFT for large signals"""
    print("\n=== Async FFT Example ===")

    # Generate large signal
    signal = np.random.randn(1_000_000)

    print(f"Processing FFT of {len(signal):,} samples asynchronously...")
    start = time.time()

    # Async FFT - doesn't block event loop
    result = await scirs2.fft_async(signal)

    elapsed = time.time() - start
    print(f"Async FFT completed in {elapsed:.3f}s")
    print(f"Result shape: {result['real'].shape}")
    print(f"DC component: {result['real'][0]:.2f}")


async def async_svd_example():
    """Async SVD for large matrices"""
    print("\n=== Async SVD Example ===")

    # Generate large matrix
    matrix = np.random.randn(1000, 1000)

    print(f"Computing SVD of {matrix.shape} matrix asynchronously...")
    start = time.time()

    # Async SVD
    result = await scirs2.svd_async(matrix, full_matrices=True)

    elapsed = time.time() - start
    print(f"Async SVD completed in {elapsed:.3f}s")
    print(f"U shape: {result['U'].shape}")
    print(f"S shape: {result['S'].shape}")
    print(f"Vt shape: {result['Vt'].shape}")
    print(f"Largest singular value: {result['S'][0]:.2f}")


async def async_qr_example():
    """Async QR decomposition"""
    print("\n=== Async QR Example ===")

    matrix = np.random.randn(500, 500)

    print(f"Computing QR decomposition asynchronously...")
    start = time.time()

    result = await scirs2.qr_async(matrix)

    elapsed = time.time() - start
    print(f"Async QR completed in {elapsed:.3f}s")
    print(f"Q shape: {result['Q'].shape}")
    print(f"R shape: {result['R'].shape}")


async def async_integration_example():
    """Async numerical integration"""
    print("\n=== Async Integration Example ===")

    # Define expensive integrand (Python function)
    def expensive_integrand(x):
        # Simulate expensive computation
        result = np.sin(x) * np.exp(-x**2)
        return result

    print("Computing expensive integral asynchronously...")
    start = time.time()

    # Async integration
    result = await scirs2.quad_async(
        expensive_integrand,
        a=0.0,
        b=10.0,
        epsabs=1e-8,
        epsrel=1e-8
    )

    elapsed = time.time() - start
    print(f"Async integration completed in {elapsed:.3f}s")
    print(f"Integral value: {result['value']:.6f}")
    print(f"Error estimate: {result['error']:.2e}")


async def async_optimization_example():
    """Async optimization"""
    print("\n=== Async Optimization Example ===")

    # Define objective function
    def rosenbrock(x):
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
        x = np.array(x)
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    # Initial guess
    x0 = np.array([-1.0, 1.0])

    print("Optimizing Rosenbrock function asynchronously...")
    start = time.time()

    # Async optimization
    result = await scirs2.minimize_async(
        rosenbrock,
        x0,
        method="BFGS",
        maxiter=1000
    )

    elapsed = time.time() - start
    print(f"Async optimization completed in {elapsed:.3f}s")
    print(f"Optimal point: x = {result['x']}")
    print(f"Optimal value: f(x) = {result['fun']:.6f}")
    print(f"Iterations: {result['nit']}")


async def parallel_async_example():
    """Run multiple async operations in parallel"""
    print("\n=== Parallel Async Operations Example ===")

    # Create multiple tasks
    signal1 = np.random.randn(500_000)
    signal2 = np.random.randn(500_000)
    matrix1 = np.random.randn(500, 500)
    matrix2 = np.random.randn(500, 500)

    print("Running 4 operations in parallel...")
    start = time.time()

    # Run all tasks concurrently
    results = await asyncio.gather(
        scirs2.fft_async(signal1),
        scirs2.fft_async(signal2),
        scirs2.qr_async(matrix1),
        scirs2.qr_async(matrix2),
    )

    elapsed = time.time() - start
    print(f"All 4 operations completed in {elapsed:.3f}s")
    print(f"(Would take ~{elapsed * 4:.1f}s if run sequentially)")


async def main():
    """Run all examples"""
    print("SciRS2 Async Operations Examples")
    print("=" * 50)

    await async_fft_example()
    await async_svd_example()
    await async_qr_example()
    await async_integration_example()
    await async_optimization_example()
    await parallel_async_example()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    # Run async examples
    asyncio.run(main())
