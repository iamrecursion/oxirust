#!/usr/bin/env python3
"""
PyTorch optimizer benchmark template for cross-framework comparison with SciRS2.
This script benchmarks various PyTorch optimizers on standard optimization problems.
"""

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import psutil
import tracemalloc


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_test_function(name: str, dim: int):
    """Get test function implementation."""
    if name.lower() == "quadratic":
        def quadratic_fn(x):
            return torch.sum(x * x)
        return quadratic_fn
    
    elif name.lower() == "rosenbrock":
        def rosenbrock_fn(x):
            if len(x) != 2:
                raise ValueError("Rosenbrock function requires 2D input")
            a, b = 1.0, 100.0
            return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        return rosenbrock_fn
    
    elif name.lower() == "beale":
        def beale_fn(x):
            if len(x) != 2:
                raise ValueError("Beale function requires 2D input")
            x1, x2 = x[0], x[1]
            term1 = (1.5 - x1 + x1 * x2)**2
            term2 = (2.25 - x1 + x1 * x2**2)**2
            term3 = (2.625 - x1 + x1 * x2**3)**2
            return term1 + term2 + term3
        return beale_fn
    
    elif name.lower() == "sphere":
        def sphere_fn(x):
            return torch.sum(x * x)
        return sphere_fn
    
    else:
        raise ValueError(f"Unknown test function: {name}")


class OptimizationProblem:
    """Wrapper for optimization problems compatible with PyTorch optimizers."""
    
    def __init__(self, function_name: str, dim: int):
        self.function_name = function_name
        self.dim = dim
        self.test_fn = get_test_function(function_name, dim)
        
        # Create parameter tensor
        self.params = nn.Parameter(torch.randn(dim, dtype=torch.float64) * 0.5)
        
    def forward(self):
        """Compute function value."""
        return self.test_fn(self.params)
    
    def get_gradient_norm(self):
        """Get current gradient norm."""
        if self.params.grad is not None:
            return torch.norm(self.params.grad).item()
        return float('inf')


def benchmark_optimizer(optimizer_class, optimizer_kwargs: Dict, 
                       problem: OptimizationProblem, max_iterations: int, 
                       tolerance: float) -> Dict[str, Any]:
    """Benchmark a single optimizer on a problem."""
    
    # Reset parameters
    with torch.no_grad():
        problem.params.data = torch.randn(problem.dim, dtype=torch.float64) * 0.5
        if problem.params.grad is not None:
            problem.params.grad.zero_()
    
    # Create optimizer
    optimizer = optimizer_class([problem.params], **optimizer_kwargs)
    
    # Tracking variables
    start_time = time.time()
    convergence_curve = []
    converged = False
    final_iteration = max_iterations
    
    # Start memory tracking
    tracemalloc.start()
    initial_memory = psutil.Process().memory_info().rss
    peak_memory = initial_memory
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        loss = problem.forward()
        convergence_curve.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Check convergence
        grad_norm = problem.get_gradient_norm()
        if grad_norm < tolerance:
            converged = True
            final_iteration = iteration
            break
        
        # Optimizer step
        optimizer.step()
        
        # Track memory usage
        current_memory = psutil.Process().memory_info().rss
        peak_memory = max(peak_memory, current_memory)
    
    end_time = time.time()
    
    # Stop memory tracking
    current_memory, peak_traced_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "converged": converged,
        "convergence_time_ms": (end_time - start_time) * 1000,
        "final_function_value": problem.forward().item(),
        "final_gradient_norm": problem.get_gradient_norm(),
        "iterations": final_iteration + 1,
        "convergence_curve": convergence_curve,
        "peak_memory_bytes": peak_memory,
        "avg_memory_bytes": (initial_memory + peak_memory) // 2,
        "allocation_count": 0,  # PyTorch doesn't easily provide this
        "fragmentation_ratio": 0.0,
        "gpu_utilization": None,  # Would need GPU monitoring
    }


def run_benchmark_suite(function_name: str, problem_dim: int, batch_size: int,
                       max_iterations: int, tolerance: float, num_runs: int,
                       random_seed: int) -> Dict[str, Any]:
    """Run complete benchmark suite."""
    
    # Setup
    setup_seed(random_seed)
    
    # Define optimizers to test
    optimizers_config = {
        "SGD": {"lr": 0.01},
        "SGD_momentum": {"lr": 0.01, "momentum": 0.9},
        "Adam": {"lr": 0.001},
        "AdamW": {"lr": 0.001, "weight_decay": 0.01},
        "RMSprop": {"lr": 0.01},
        "Adagrad": {"lr": 0.01},
        "Adadelta": {"lr": 1.0},
        "LBFGS": {"lr": 1.0, "max_iter": 20},
    }
    
    # Map optimizer names to classes
    optimizer_classes = {
        "SGD": optim.SGD,
        "SGD_momentum": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        "Adadelta": optim.Adadelta,
        "LBFGS": optim.LBFGS,
    }
    
    results = {}
    
    for optimizer_name, config in optimizers_config.items():
        print(f"Benchmarking {optimizer_name}...")
        
        # Run multiple times for statistical significance
        run_results = []
        successful_runs = 0
        
        for run in range(num_runs):
            setup_seed(random_seed + run)
            problem = OptimizationProblem(function_name, problem_dim)
            
            try:
                if optimizer_name == "LBFGS":
                    # LBFGS requires special handling
                    result = benchmark_lbfgs(problem, config, max_iterations, tolerance)
                else:
                    result = benchmark_optimizer(
                        optimizer_classes[optimizer_name],
                        config,
                        problem,
                        max_iterations,
                        tolerance
                    )
                
                run_results.append(result)
                if result["converged"]:
                    successful_runs += 1
                    
            except Exception as e:
                print(f"Error in {optimizer_name} run {run}: {e}")
                # Add failed run
                run_results.append({
                    "converged": False,
                    "convergence_time_ms": max_iterations * 100,  # Penalty time
                    "final_function_value": 1e6,
                    "final_gradient_norm": 1e6,
                    "iterations": max_iterations,
                    "convergence_curve": [1e6] * max_iterations,
                    "peak_memory_bytes": 0,
                    "avg_memory_bytes": 0,
                    "allocation_count": 0,
                    "fragmentation_ratio": 0.0,
                    "gpu_utilization": None,
                })
        
        # Aggregate results
        if run_results:
            results[optimizer_name] = aggregate_results(run_results, successful_runs, num_runs)
        
    return results


def benchmark_lbfgs(problem: OptimizationProblem, config: Dict, 
                   max_iterations: int, tolerance: float) -> Dict[str, Any]:
    """Special handling for L-BFGS optimizer."""
    
    # Reset parameters
    with torch.no_grad():
        problem.params.data = torch.randn(problem.dim, dtype=torch.float64) * 0.5
        if problem.params.grad is not None:
            problem.params.grad.zero_()
    
    optimizer = optim.LBFGS([problem.params], **config)
    
    start_time = time.time()
    convergence_curve = []
    converged = False
    iteration_count = 0
    
    initial_memory = psutil.Process().memory_info().rss
    peak_memory = initial_memory
    
    def closure():
        nonlocal iteration_count, converged, peak_memory
        
        optimizer.zero_grad()
        loss = problem.forward()
        convergence_curve.append(loss.item())
        loss.backward()
        
        # Check convergence
        grad_norm = problem.get_gradient_norm()
        if grad_norm < tolerance:
            converged = True
        
        # Track memory
        current_memory = psutil.Process().memory_info().rss
        peak_memory = max(peak_memory, current_memory)
        
        iteration_count += 1
        return loss
    
    # Run L-BFGS
    for _ in range(max_iterations // config.get("max_iter", 20)):
        if converged:
            break
        optimizer.step(closure)
    
    end_time = time.time()
    
    return {
        "converged": converged,
        "convergence_time_ms": (end_time - start_time) * 1000,
        "final_function_value": problem.forward().item(),
        "final_gradient_norm": problem.get_gradient_norm(),
        "iterations": iteration_count,
        "convergence_curve": convergence_curve,
        "peak_memory_bytes": peak_memory,
        "avg_memory_bytes": (initial_memory + peak_memory) // 2,
        "allocation_count": 0,
        "fragmentation_ratio": 0.0,
        "gpu_utilization": None,
    }


def aggregate_results(run_results: List[Dict], successful_runs: int, total_runs: int) -> Dict[str, Any]:
    """Aggregate results from multiple runs."""
    
    if not run_results:
        return {}
    
    # Extract metrics
    convergence_times = [r["convergence_time_ms"] for r in run_results]
    final_values = [r["final_function_value"] for r in run_results]
    iterations = [r["iterations"] for r in run_results]
    gradient_norms = [r["final_gradient_norm"] for r in run_results]
    convergence_curves = [r["convergence_curve"] for r in run_results]
    
    # Calculate statistics
    def safe_mean(values):
        return sum(values) / len(values) if values else 0.0
    
    def safe_std(values, mean_val):
        if len(values) <= 1:
            return 0.0
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    mean_time = safe_mean(convergence_times)
    mean_value = safe_mean(final_values)
    mean_iterations = safe_mean(iterations)
    mean_grad_norm = safe_mean(gradient_norms)
    
    return {
        "successful_runs": successful_runs,
        "total_runs": total_runs,
        "success_rate": successful_runs / total_runs,
        "mean_convergence_time_ms": mean_time,
        "std_convergence_time_ms": safe_std(convergence_times, mean_time),
        "mean_final_value": mean_value,
        "std_final_value": safe_std(final_values, mean_value),
        "mean_iterations": mean_iterations,
        "std_iterations": safe_std(iterations, mean_iterations),
        "mean_gradient_norm": mean_grad_norm,
        "std_gradient_norm": safe_std(gradient_norms, mean_grad_norm),
        "convergence_curves": convergence_curves,
        "peak_memory_bytes": max(r["peak_memory_bytes"] for r in run_results),
        "avg_memory_bytes": safe_mean([r["avg_memory_bytes"] for r in run_results]),
        "allocation_count": 0,
        "fragmentation_ratio": 0.0,
        "gpu_utilization": None,
    }


if __name__ == "__main__":
    # Configuration from template substitution
    function_name = "{{FUNCTION_NAME}}"
    problem_dim = {{PROBLEM_DIM}}
    batch_size = {{BATCH_SIZE}}
    max_iterations = {{MAX_ITERATIONS}}
    tolerance = {{TOLERANCE}}
    num_runs = {{NUM_RUNS}}
    random_seed = {{RANDOM_SEED}}
    
    try:
        results = run_benchmark_suite(
            function_name, problem_dim, batch_size,
            max_iterations, tolerance, num_runs, random_seed
        )
        
        # Output results as JSON
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        # Error handling
        error_result = {
            "error": str(e),
            "traceback": tracemalloc.format_exc() if 'tracemalloc' in globals() else str(e)
        }
        print(json.dumps(error_result))