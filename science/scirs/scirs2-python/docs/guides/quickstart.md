# SciRS2-Python Quick Start Guide

**Version**: 0.2.0
**Time to Complete**: 10 minutes

---

## Installation

### From PyPI (Recommended)

```bash
pip install scirs2
```

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/cool-japan/scirs.git
cd scirs/scirs2-python

# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
maturin develop --release

# Install test dependencies
pip install pytest numpy scipy scikit-learn
```

### Verify Installation

```python
import scirs2
print(f"SciRS2 version: {scirs2.__version__}")
```

Expected output: `SciRS2 version: 0.2.0`

---

## Your First 5 Minutes with SciRS2

### Example 1: Complex Statistical Analysis (SciRS2's Strength!)

```python
import numpy as np
import scirs2

# Generate sample data
np.random.seed(42)
data = np.random.randn(1000)

# ✅ Use scirs2 for complex statistics (FAST!)
skewness = scirs2.skew_py(data)
kurtosis = scirs2.kurtosis_py(data)

print(f"Skewness: {skewness:.4f}")  # 6x faster than SciPy!
print(f"Kurtosis: {kurtosis:.4f}")  # 5x faster than SciPy!

# Get comprehensive statistics
stats = scirs2.describe_py(data)
print(f"Mean: {stats['mean']:.4f}")
print(f"Std: {stats['std']:.4f}")
print(f"Min: {stats['min']:.4f}")
print(f"Max: {stats['max']:.4f}")
```

### Example 2: Recommended Hybrid Approach

```python
import numpy as np
import scipy.stats
import scirs2

# Generate data
data = np.random.randn(500)

# ✅ Use NumPy for basic statistics (FAST)
mean = np.mean(data)
std = np.std(data)
median = np.median(data)

# ✅ Use scirs2 for complex statistics (FASTER!)
skewness = scirs2.skew_py(data)
kurtosis = scirs2.kurtosis_py(data)
iqr = scirs2.iqr_py(data)

print("Basic Statistics (NumPy):")
print(f"  Mean: {mean:.4f}")
print(f"  Std: {std:.4f}")
print(f"  Median: {median:.4f}")

print("\nDistribution Shape (scirs2):")
print(f"  Skewness: {skewness:.4f}")
print(f"  Kurtosis: {kurtosis:.4f}")
print(f"  IQR: {iqr:.4f}")
```

### Example 3: Distribution Analysis

```python
import numpy as np
import scirs2

# Generate different distributions
normal_data = np.random.randn(1000)
skewed_data = np.random.gamma(2, 2, 1000)
uniform_data = np.random.uniform(-3, 3, 1000)

def analyze_distribution(data, name):
    """Analyze distribution shape using scirs2"""
    skew = scirs2.skew_py(data)
    kurt = scirs2.kurtosis_py(data)

    print(f"\n{name}:")
    print(f"  Skewness: {skew:7.4f}")
    print(f"  Kurtosis: {kurt:7.4f}")

    # Interpret
    if abs(skew) < 0.5:
        print(f"  → Approximately symmetric")
    elif skew > 0:
        print(f"  → Right-skewed (positive skew)")
    else:
        print(f"  → Left-skewed (negative skew)")

analyze_distribution(normal_data, "Normal Distribution")
analyze_distribution(skewed_data, "Gamma Distribution")
analyze_distribution(uniform_data, "Uniform Distribution")
```

---

## Common Use Cases

### Use Case 1: Exploratory Data Analysis

```python
import numpy as np
import scirs2

# Load your data
data = np.random.randn(500)  # Replace with your data

# Quick statistical summary
stats = scirs2.describe_py(data)

print("=== Data Summary ===")
for key, value in stats.items():
    print(f"{key:12s}: {value:10.4f}")

# Distribution analysis
skew = scirs2.skew_py(data)
kurt = scirs2.kurtosis_py(data)

print("\n=== Distribution Shape ===")
print(f"Skewness: {skew:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# Assess normality
if abs(skew) < 0.5 and abs(kurt) < 1.0:
    print("✅ Data appears approximately normal")
else:
    print("⚠️  Data deviates from normal distribution")
```

### Use Case 2: Correlation Analysis

```python
import numpy as np
import scirs2

# Generate correlated data
np.random.seed(42)
x = np.random.randn(500)
y = 2 * x + np.random.randn(500) * 0.5  # Strong positive correlation

# Pearson correlation (fast on small data!)
corr = scirs2.pearsonr_py(x, y)
print(f"Pearson correlation: {corr:.4f}")

# Spearman correlation
spearman = scirs2.spearmanr_py(x, y)
print(f"Spearman correlation: {spearman:.4f}")

# Covariance
cov = scirs2.covariance_py(x, y, ddof=1)
print(f"Covariance: {cov:.4f}")
```

### Use Case 3: Time Series Analysis

```python
import numpy as np
import scirs2

# Create time series
np.random.seed(42)
data = np.cumsum(np.random.randn(100))  # Random walk
ts = scirs2.PyTimeSeries(data, None)

# Descriptive statistics
stats = ts.describe()
print("Time Series Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.4f}")

# Differencing
diff1 = scirs2.apply_differencing(ts, 1)
print(f"\nFirst difference computed: {len(diff1.data)} points")

# ARIMA modeling
arima = scirs2.PyARIMA(1, 1, 0)  # AR(1), I(1), MA(0)
arima.fit(ts)

# Forecast
forecast = arima.forecast(10)
print(f"\n10-step forecast: {forecast[:5]}...")

# Model summary
print("\nModel Summary:")
print(arima.summary())
```

---

## What NOT to Do ❌

### Don't Use scirs2 for Linear Algebra

```python
import numpy as np
import scipy.linalg
import scirs2

matrix = np.random.randn(100, 100)

# ❌ BAD - 377x slower!
det_slow = scirs2.det_py(matrix)

# ✅ GOOD - Use SciPy
det_fast = scipy.linalg.det(matrix)

# ❌ BAD - 714x slower!
inv_slow = scirs2.inv_py(matrix)

# ✅ GOOD - Use SciPy
inv_fast = scipy.linalg.inv(matrix)
```

### Don't Use scirs2 for FFT

```python
import numpy as np
import scirs2

signal = np.random.randn(1024)

# ❌ BAD - 62x slower!
spectrum_slow = scirs2.rfft_py(signal)

# ✅ GOOD - Use NumPy
spectrum_fast = np.fft.rfft(signal)
```

### Don't Use scirs2 for Basic Stats on Large Data

```python
import numpy as np
import scirs2

large_data = np.random.randn(100000)

# ❌ BAD - 50x slower!
mean_slow = scirs2.mean_py(large_data)

# ✅ GOOD - Use NumPy
mean_fast = np.mean(large_data)
```

---

## Performance Tips

### 1. Use Contiguous Arrays

```python
import numpy as np
import scirs2

# Create non-contiguous array
data = np.random.randn(1000, 1000)
data_slice = data[::2, ::2]  # Non-contiguous

# ❌ Slower
result_slow = scirs2.skew_py(data_slice.flatten())

# ✅ Faster - make contiguous first
data_contiguous = np.ascontiguousarray(data_slice.flatten())
result_fast = scirs2.skew_py(data_contiguous)
```

### 2. Choose the Right Tool for Your Data Size

```python
import numpy as np
import scipy.stats
import scirs2

# Small data (<1,000) - Use scirs2
small_data = np.random.randn(500)
skew_small = scirs2.skew_py(small_data)  # ✅ 6x faster

# Large data (>10,000) - Use NumPy/SciPy
large_data = np.random.randn(50000)
skew_large = scipy.stats.skew(large_data)  # ✅ Faster than scirs2
```

### 3. Profile Your Specific Use Case

```python
import time
import numpy as np
import scipy.stats
import scirs2

data = np.random.randn(1000)

# Benchmark SciPy
start = time.perf_counter()
for _ in range(100):
    _ = scipy.stats.skew(data)
scipy_time = time.perf_counter() - start

# Benchmark scirs2
start = time.perf_counter()
for _ in range(100):
    _ = scirs2.skew_py(data)
scirs2_time = time.perf_counter() - start

speedup = scipy_time / scirs2_time
print(f"Speedup: {speedup:.2f}x")
```

---

## Next Steps

### 1. Read the Performance Guide

**Essential**: [Performance Guide](../performance.md)

Learn exactly when to use scirs2 vs NumPy/SciPy with detailed benchmarks and decision matrices.

### 2. Explore Module Documentation

- [Linear Algebra API](../api/linalg.md) (⚠️ with performance warnings)
- [Statistics API](../api/stats.md) (✅ recommended)
- [FFT API](../api/fft.md) (⚠️ with performance warnings)
- [Time Series API](../api/series.md)
- [Clustering API](../api/cluster.md)

### 3. Check Out Examples

See [examples/](../../examples/) directory for 20+ working examples:
- Distribution analysis
- Statistical testing
- Time series forecasting
- Correlation analysis
- And more!

### 4. Migration from SciPy

If you're coming from SciPy, read: [Migration Guide](./migration.md)

### 5. Troubleshooting

Having issues? Check: [Troubleshooting Guide](./troubleshooting.md)

---

## Getting Help

- **Documentation**: https://docs.rs/scirs2-python
- **GitHub Issues**: https://github.com/cool-japan/scirs/issues
- **Performance Questions**: See [Performance Guide](../performance.md)

---

## Summary

**Use scirs2 for**:
- ✅ Complex statistics (skewness, kurtosis) on small-medium data
- ✅ Distribution analysis
- ✅ Type-safe Rust integration

**Use NumPy/SciPy for**:
- ❌ Linear algebra (all operations)
- ❌ FFT (all operations)
- ❌ Basic statistics on large data
- ❌ Performance-critical code paths

**Best Approach**: Hybrid - use both libraries together for optimal performance!

---

**Happy Computing!** 🦀📊

For questions or issues, visit: https://github.com/cool-japan/scirs
