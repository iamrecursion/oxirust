"""
Demonstration of New Statistical Distributions in scirs2-python

This script demonstrates the 6 newly added statistical distributions:
1. Lognormal
2. Weibull
3. Laplace
4. Logistic
5. Pareto
6. Geometric

Each distribution showcases:
- Creation with custom parameters
- PDF/PMF evaluation
- CDF evaluation
- PPF (inverse CDF) evaluation
- Random variate generation
- Basic statistical properties
"""

import numpy as np
import scirs2

print("=" * 80)
print("New Statistical Distributions in scirs2-python")
print("=" * 80)

# 1. LOGNORMAL DISTRIBUTION
print("\n" + "=" * 80)
print("1. LOGNORMAL DISTRIBUTION")
print("=" * 80)
print("\nThe lognormal distribution describes a variable whose logarithm is normally")
print("distributed. Commonly used in finance, reliability analysis, and natural phenomena.")

# Create lognormal distribution (mu=0, sigma=1)
dist_lognorm = scirs2.lognorm(mu=0.0, sigma=1.0)
print(f"\nDistribution: Lognormal(mu=0, sigma=1)")

# PDF evaluation
x_vals = [0.5, 1.0, 2.0, 3.0]
print(f"\nPDF values:")
for x in x_vals:
    print(f"  pdf({x:.1f}) = {dist_lognorm.pdf(x):.6f}")

# CDF evaluation
print(f"\nCDF values:")
for x in x_vals:
    print(f"  cdf({x:.1f}) = {dist_lognorm.cdf(x):.6f}")

# PPF evaluation
q_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
print(f"\nPPF (quantiles):")
for q in q_vals:
    print(f"  ppf({q:.2f}) = {dist_lognorm.ppf(q):.6f}")

# Random samples
samples_lognorm = dist_lognorm.rvs(1000)
print(f"\nRandom samples (n=1000):")
print(f"  Mean: {np.mean(samples_lognorm):.4f} (theoretical: {np.exp(0.5):.4f})")
print(f"  Std:  {np.std(samples_lognorm):.4f}")
print(f"  Min:  {np.min(samples_lognorm):.4f}")
print(f"  Max:  {np.max(samples_lognorm):.4f}")

# 2. WEIBULL DISTRIBUTION
print("\n" + "=" * 80)
print("2. WEIBULL DISTRIBUTION")
print("=" * 80)
print("\nThe Weibull distribution is widely used in reliability engineering,")
print("survival analysis, and modeling failure times.")

# Create Weibull distribution (shape=2.0, scale=1.0)
dist_weibull = scirs2.weibull_min(shape=2.0, scale=1.0)
print(f"\nDistribution: Weibull(shape=2, scale=1)")

# PDF evaluation
x_vals = [0.5, 1.0, 1.5, 2.0]
print(f"\nPDF values:")
for x in x_vals:
    print(f"  pdf({x:.1f}) = {dist_weibull.pdf(x):.6f}")

# CDF evaluation
print(f"\nCDF values:")
for x in x_vals:
    print(f"  cdf({x:.1f}) = {dist_weibull.cdf(x):.6f}")

# PPF evaluation
print(f"\nPPF (quantiles):")
for q in q_vals:
    print(f"  ppf({q:.2f}) = {dist_weibull.ppf(q):.6f}")

# Random samples
samples_weibull = dist_weibull.rvs(1000)
print(f"\nRandom samples (n=1000):")
print(f"  Mean: {np.mean(samples_weibull):.4f}")
print(f"  Std:  {np.std(samples_weibull):.4f}")
print(f"  Min:  {np.min(samples_weibull):.4f}")
print(f"  Max:  {np.max(samples_weibull):.4f}")

# 3. LAPLACE DISTRIBUTION
print("\n" + "=" * 80)
print("3. LAPLACE DISTRIBUTION")
print("=" * 80)
print("\nThe Laplace distribution (double exponential) has heavier tails than normal.")
print("Used in robust statistics, signal processing, and Bayesian inference.")

# Create Laplace distribution (loc=0, scale=1)
dist_laplace = scirs2.laplace(loc=0.0, scale=1.0)
print(f"\nDistribution: Laplace(loc=0, scale=1)")

# PDF evaluation
x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
print(f"\nPDF values:")
for x in x_vals:
    print(f"  pdf({x:+.1f}) = {dist_laplace.pdf(x):.6f}")

# CDF evaluation
print(f"\nCDF values:")
for x in x_vals:
    print(f"  cdf({x:+.1f}) = {dist_laplace.cdf(x):.6f}")

# PPF evaluation
print(f"\nPPF (quantiles):")
for q in q_vals:
    print(f"  ppf({q:.2f}) = {dist_laplace.ppf(q):+.6f}")

# Random samples
samples_laplace = dist_laplace.rvs(1000)
print(f"\nRandom samples (n=1000):")
print(f"  Mean: {np.mean(samples_laplace):+.4f}")
print(f"  Std:  {np.std(samples_laplace):.4f} (theoretical: {np.sqrt(2):.4f})")
print(f"  Min:  {np.min(samples_laplace):+.4f}")
print(f"  Max:  {np.max(samples_laplace):+.4f}")

# 4. LOGISTIC DISTRIBUTION
print("\n" + "=" * 80)
print("4. LOGISTIC DISTRIBUTION")
print("=" * 80)
print("\nThe logistic distribution is used in logistic regression, neural networks,")
print("and modeling growth processes.")

# Create Logistic distribution (loc=0, scale=1)
dist_logistic = scirs2.logistic(loc=0.0, scale=1.0)
print(f"\nDistribution: Logistic(loc=0, scale=1)")

# PDF evaluation
x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
print(f"\nPDF values:")
for x in x_vals:
    print(f"  pdf({x:+.1f}) = {dist_logistic.pdf(x):.6f}")

# CDF evaluation (logistic CDF is the sigmoid function)
print(f"\nCDF values (sigmoid function):")
for x in x_vals:
    print(f"  cdf({x:+.1f}) = {dist_logistic.cdf(x):.6f}")

# PPF evaluation
print(f"\nPPF (quantiles):")
for q in q_vals:
    print(f"  ppf({q:.2f}) = {dist_logistic.ppf(q):+.6f}")

# Random samples
samples_logistic = dist_logistic.rvs(1000)
print(f"\nRandom samples (n=1000):")
print(f"  Mean: {np.mean(samples_logistic):+.4f}")
print(f"  Std:  {np.std(samples_logistic):.4f} (theoretical: {np.pi/np.sqrt(3):.4f})")
print(f"  Min:  {np.min(samples_logistic):+.4f}")
print(f"  Max:  {np.max(samples_logistic):+.4f}")

# 5. PARETO DISTRIBUTION
print("\n" + "=" * 80)
print("5. PARETO DISTRIBUTION")
print("=" * 80)
print("\nThe Pareto distribution models the '80-20 rule' and power-law phenomena.")
print("Used in economics, physics, and network analysis.")

# Create Pareto distribution (shape=3.0, scale=1.0)
dist_pareto = scirs2.pareto(shape=3.0, scale=1.0)
print(f"\nDistribution: Pareto(shape=3, scale=1)")

# PDF evaluation
x_vals = [1.0, 1.5, 2.0, 3.0, 5.0]
print(f"\nPDF values (x > scale):")
for x in x_vals:
    print(f"  pdf({x:.1f}) = {dist_pareto.pdf(x):.6f}")

# CDF evaluation
print(f"\nCDF values:")
for x in x_vals:
    print(f"  cdf({x:.1f}) = {dist_pareto.cdf(x):.6f}")

# PPF evaluation
print(f"\nPPF (quantiles):")
for q in q_vals:
    print(f"  ppf({q:.2f}) = {dist_pareto.ppf(q):.6f}")

# Random samples
samples_pareto = dist_pareto.rvs(1000)
print(f"\nRandom samples (n=1000):")
print(f"  Mean: {np.mean(samples_pareto):.4f} (theoretical: 1.5 for shape=3)")
print(f"  Std:  {np.std(samples_pareto):.4f}")
print(f"  Min:  {np.min(samples_pareto):.4f} (should be >= scale=1.0)")
print(f"  Max:  {np.max(samples_pareto):.4f}")

# 6. GEOMETRIC DISTRIBUTION
print("\n" + "=" * 80)
print("6. GEOMETRIC DISTRIBUTION (Discrete)")
print("=" * 80)
print("\nThe geometric distribution models the number of failures before first success.")
print("Used in reliability analysis, queuing theory, and probability problems.")

# Create Geometric distribution (p=0.3)
dist_geom = scirs2.geom(p=0.3)
print(f"\nDistribution: Geometric(p=0.3)")

# PMF evaluation
k_vals = [0, 1, 2, 3, 5, 10]
print(f"\nPMF values:")
for k in k_vals:
    print(f"  pmf({k:2d}) = {dist_geom.pmf(float(k)):.6f}")

# CDF evaluation
print(f"\nCDF values:")
for k in k_vals:
    print(f"  cdf({k:2d}) = {dist_geom.cdf(float(k)):.6f}")

# PPF evaluation
print(f"\nPPF (quantiles):")
for q in q_vals:
    print(f"  ppf({q:.2f}) = {dist_geom.ppf(q):.0f}")

# Random samples
samples_geom = dist_geom.rvs(1000)
print(f"\nRandom samples (n=1000):")
theoretical_mean = (1.0 - 0.3) / 0.3
print(f"  Mean: {np.mean(samples_geom):.4f} (theoretical: {theoretical_mean:.4f})")
print(f"  Std:  {np.std(samples_geom):.4f}")
print(f"  Min:  {int(np.min(samples_geom))}")
print(f"  Max:  {int(np.max(samples_geom))}")

# COMPARISON: CDF-PPF Inverse Relationship
print("\n" + "=" * 80)
print("VERIFICATION: CDF-PPF Inverse Relationship")
print("=" * 80)
print("\nVerifying that PPF(CDF(x)) ≈ x for all distributions:\n")

distributions = [
    ("Lognormal", dist_lognorm, 1.5),
    ("Weibull", dist_weibull, 1.0),
    ("Laplace", dist_laplace, 0.5),
    ("Logistic", dist_logistic, 0.5),
    ("Pareto", dist_pareto, 2.0),
    ("Geometric", dist_geom, 2.0),
]

for name, dist, test_x in distributions:
    cdf_val = dist.cdf(test_x)
    ppf_val = dist.ppf(cdf_val)
    error = abs(ppf_val - test_x)
    status = "✓" if error < 0.01 else "✗"
    print(f"{status} {name:12s}: x={test_x:.2f}, CDF(x)={cdf_val:.6f}, PPF(CDF(x))={ppf_val:.6f}, error={error:.6f}")

# SUMMARY
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nSuccessfully demonstrated 6 new statistical distributions:")
print("  ✓ Lognormal   - Continuous, x > 0, multiplicative processes")
print("  ✓ Weibull     - Continuous, x ≥ 0, reliability engineering")
print("  ✓ Laplace     - Continuous, x ∈ ℝ, robust statistics")
print("  ✓ Logistic    - Continuous, x ∈ ℝ, sigmoid/neural networks")
print("  ✓ Pareto      - Continuous, x ≥ scale, power laws")
print("  ✓ Geometric   - Discrete, k ∈ {0,1,2,...}, waiting times")
print("\nAll distributions provide:")
print("  • PDF/PMF - Probability density/mass function")
print("  • CDF     - Cumulative distribution function")
print("  • PPF     - Percent point function (inverse CDF)")
print("  • RVS     - Random variate sampling")
print("\nTotal distributions in scirs2-python: 17")
print("=" * 80)
