# SciRS2-Python API Documentation

**Version**: 0.2.0
**Last Updated**: 2025-12-29

---

## Overview

This directory contains detailed API documentation for all scirs2-python modules.

**Note**: Full API reference with type signatures available in `scirs2.pyi` type stub file.

---

## Module Index

### Statistics (`scirs2.stats`)

✅ **Performance Strength**: Complex statistics are **4-23x faster** than SciPy on small-medium datasets (<10,000 elements).

**Key Functions**:
- `skew_py(data)` - Skewness (Fisher's definition) - **6-23x faster!**
- `kurtosis_py(data)` - Kurtosis (Fisher's definition) - **5-24x faster!**
- `describe_py(data)` - Comprehensive statistical summary
- `mean_py(data)` - Arithmetic mean
- `std_py(data, ddof=0)` - Standard deviation
- `var_py(data, ddof=0)` - Variance
- `median_py(data)` - Median value
- `percentile_py(data, q)` - Percentile calculation
- `iqr_py(data)` - Interquartile range - **1.7x faster**
- `pearsonr_py(x, y)` - Pearson correlation - **3x faster on small data**
- `spearmanr_py(x, y)` - Spearman rank correlation
- `covariance_py(x, y, ddof=1)` - Covariance

**Performance Notes**:
- ✅ **Best for**: Complex statistics (skew, kurt) on datasets <10K
- ⚠️ **Avoid**: Basic stats (mean, std) on large data (>10K) - use NumPy instead

**Example**:
```python
import numpy as np
import scirs2

data = np.random.randn(500)

# ✅ Excellent performance
skew = scirs2.skew_py(data)  # 6x faster!
kurt = scirs2.kurtosis_py(data)  # 5x faster!

# Get all statistics at once
stats = scirs2.describe_py(data)
# Returns: {'mean', 'std', 'var', 'min', 'max', 'skewness', 'kurtosis', ...}
```

---

### Linear Algebra (`scirs2.linalg`)

⚠️ **Performance Warning**: Linear algebra operations are **200-700x slower** than SciPy. **Use NumPy/SciPy for ALL linear algebra operations.**

**Available Functions** (not recommended for production):
- `det_py(A)` - Determinant - **377x slower**
- `inv_py(A)` - Matrix inverse - **714x slower**
- `solve_py(A, b)` - Solve linear system - **207x slower**
- `qr_py(A)` - QR decomposition - **430x slower**
- `lu_py(A)` - LU decomposition
- `svd_py(A)` - Singular value decomposition
- `eig_py(A)` - Eigenvalue decomposition
- `cholesky_py(A)` - Cholesky decomposition
- `matrix_norm_py(A, ord)` - Matrix norms
- `vector_norm_py(v, ord)` - Vector norms

**Recommendation**: **DO NOT USE** - Use `scipy.linalg` or `numpy.linalg` instead.

**Example (AVOID)**:
```python
# ❌ BAD - Extremely slow!
import scirs2
det = scirs2.det_py(matrix)  # 377x slower

# ✅ GOOD - Use SciPy
import scipy.linalg
det = scipy.linalg.det(matrix)
```

---

### FFT (`scirs2.fft`)

⚠️ **Performance Warning**: FFT operations are **62x slower** than NumPy. **Use NumPy/SciPy for ALL FFT operations.**

**Available Functions** (not recommended for production):
- `fft_py(data)` - Fast Fourier Transform
- `ifft_py(real, imag)` - Inverse FFT
- `rfft_py(data)` - Real FFT - **62x slower**
- `irfft_py(real, imag, n)` - Inverse real FFT
- `dct_py(data, type)` - Discrete Cosine Transform
- `idct_py(data, type)` - Inverse DCT
- `fftfreq_py(n, d)` - FFT frequencies
- `rfftfreq_py(n, d)` - Real FFT frequencies
- `fftshift_py(data)` - Shift zero-frequency component
- `next_fast_len_py(n, real)` - Next fast FFT length

**Recommendation**: **DO NOT USE** - Use `numpy.fft` instead.

**Example (AVOID)**:
```python
# ❌ BAD - Extremely slow!
import scirs2
spectrum = scirs2.rfft_py(signal)  # 62x slower

# ✅ GOOD - Use NumPy
import numpy as np
spectrum = np.fft.rfft(signal)
```

---

### Clustering (`scirs2.cluster`)

**Classes**:
- `KMeans` - K-Means clustering algorithm

**Functions**:
- `silhouette_score_py(X, labels)` - Silhouette coefficient
- `davies_bouldin_score_py(X, labels)` - Davies-Bouldin index
- `calinski_harabasz_score_py(X, labels)` - Calinski-Harabasz index
- `standardize_py(X, with_std)` - Standardize features
- `normalize_py(X, norm)` - Normalize samples

**Example**:
```python
import numpy as np
import scirs2

# Generate sample data
X = np.random.randn(100, 2)

# K-Means clustering
kmeans = scirs2.KMeans(n_clusters=3)
kmeans.fit(X)

labels = kmeans.labels
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

# Evaluate
silhouette = scirs2.silhouette_score_py(X, labels)
print(f"Silhouette score: {silhouette:.4f}")
```

---

### Time Series (`scirs2.series`)

**Classes**:
- `PyTimeSeries` - Time series container
- `PyARIMA` - ARIMA model for forecasting

**Functions**:
- `apply_differencing(ts, lag)` - Apply differencing
- `apply_seasonal_differencing(ts, period)` - Seasonal differencing
- `boxcox_transform(ts, lambda)` - Box-Cox transformation
- `boxcox_inverse(data, lambda)` - Inverse Box-Cox
- `adf_test(ts, max_lags)` - Augmented Dickey-Fuller test
- `stl_decomposition(ts, period)` - STL decomposition

**Example**:
```python
import numpy as np
import scirs2

# Create time series
data = np.cumsum(np.random.randn(100))
ts = scirs2.PyTimeSeries(data, None)

# ARIMA modeling
arima = scirs2.PyARIMA(1, 1, 0)  # AR(1), I(1), MA(0)
arima.fit(ts)

# Forecast
forecast = arima.forecast(10)
print(f"10-step forecast: {forecast}")

# Model summary
print(arima.summary())
```

---

## Quick Reference by Use Case

### Use Case: Distribution Analysis

**Best Module**: `scirs2.stats` ✅

```python
import scirs2
import numpy as np

data = np.random.randn(500)

# Comprehensive analysis
stats = scirs2.describe_py(data)

# Distribution shape
skew = scirs2.skew_py(data)  # 6x faster!
kurt = scirs2.kurtosis_py(data)  # 5x faster!
```

**Performance**: **Excellent** (4-23x faster than SciPy)

---

### Use Case: Linear System Solving

**Best Module**: **NumPy/SciPy** (NOT scirs2) ❌

```python
import scipy.linalg

# ✅ GOOD - Use SciPy
x = scipy.linalg.solve(A, b)

# ❌ BAD - Don't use scirs2
# x = scirs2.solve_py(A, b)  # 207x slower!
```

---

### Use Case: Signal Processing

**Best Module**: **NumPy** (NOT scirs2) ❌

```python
import numpy as np

# ✅ GOOD - Use NumPy
spectrum = np.fft.rfft(signal)

# ❌ BAD - Don't use scirs2
# spectrum = scirs2.rfft_py(signal)  # 62x slower!
```

---

### Use Case: Time Series Forecasting

**Best Module**: `scirs2.series` ✅

```python
import scirs2

ts = scirs2.PyTimeSeries(data, None)
arima = scirs2.PyARIMA(1, 1, 0)
arima.fit(ts)
forecast = arima.forecast(steps)
```

---

### Use Case: Clustering

**Best Module**: `scirs2.cluster` (compare with scikit-learn)

```python
import scirs2

kmeans = scirs2.KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels
```

---

## Type Hints

All functions have type stubs in `scirs2.pyi`. Your IDE will provide:
- Autocompletion
- Type checking
- Function signatures
- Documentation

**Example IDE support**:
```python
import scirs2

# IDE shows:
# skew_py(data: np.ndarray) -> float
result = scirs2.skew_py(data)  # ← Type hint appears
```

---

## Performance Comparison Table

| Operation | scirs2 | NumPy/SciPy | Recommendation |
|-----------|--------|-------------|----------------|
| **Skewness** (100-1K) | ✅ | 6-23x slower | **Use scirs2** |
| **Kurtosis** (100-1K) | ✅ | 5-24x slower | **Use scirs2** |
| **Mean/Std/Var** | ❌ | 10-50x faster | **Use NumPy** |
| **Linear algebra** | ❌ | 200-700x faster | **Use SciPy** |
| **FFT** | ❌ | 62x faster | **Use NumPy** |
| **Correlation** (<1K) | ✅ | 3x slower | **Use scirs2** |
| **IQR** (<1K) | ✅ | 1.7x slower | **Use scirs2** |

---

## Best Practices

### DO ✅

1. **Use scirs2 for complex statistics on small data**:
   ```python
   skew = scirs2.skew_py(data)  # <10K elements
   ```

2. **Use contiguous arrays**:
   ```python
   data = np.ascontiguousarray(data)
   result = scirs2.skew_py(data)
   ```

3. **Profile your specific use case**:
   ```python
   import time
   # Benchmark both scirs2 and SciPy for your data
   ```

### DON'T ❌

1. **Don't use scirs2 for linear algebra**:
   ```python
   # ❌ 377x slower!
   det = scirs2.det_py(matrix)
   ```

2. **Don't use scirs2 for FFT**:
   ```python
   # ❌ 62x slower!
   spectrum = scirs2.rfft_py(signal)
   ```

3. **Don't use scirs2 for basic stats on large data**:
   ```python
   # ❌ 50x slower on >10K elements!
   mean = scirs2.mean_py(large_data)
   ```

---

## Additional Resources

- **Performance Guide**: [../performance.md](../performance.md)
- **Quick Start**: [../guides/quickstart.md](../guides/quickstart.md)
- **Migration Guide**: [../guides/migration.md](../guides/migration.md)
- **Troubleshooting**: [../guides/troubleshooting.md](../guides/troubleshooting.md)
- **Type Stubs**: `scirs2.pyi` in package root

---

## Contributing

To add or improve API documentation:
1. Check function signatures in `scirs2.pyi`
2. Test performance vs SciPy/NumPy
3. Add examples and performance notes
4. Submit PR to https://github.com/cool-japan/scirs

---

**Last Updated**: 2025-12-29
**Version**: 0.2.0
