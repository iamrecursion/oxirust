# Migrating from SciPy to SciRS2

**Target Audience**: Python users familiar with NumPy/SciPy
**Migration Strategy**: Hybrid approach (not full replacement)
**Version**: SciRS2 0.2.0

---

## Important: Not a Drop-In Replacement

**Critical Understanding**: SciRS2 is **NOT** a general-purpose SciPy replacement. It's a **specialized tool** for:
- Complex statistical analysis on small-medium datasets
- Type-safe Rust integration
- Higher-order statistical moments

**Recommended Strategy**: Use scirs2 **alongside** NumPy/SciPy in a hybrid approach, not as a complete replacement.

---

## Quick Reference: SciPy → SciRS2

### ✅ Recommended Migrations (Performance Gains)

#### Complex Statistics (4-23x faster on small data)

| SciPy | SciRS2 | Speedup | Notes |
|-------|--------|---------|-------|
| `scipy.stats.skew(data)` | `scirs2.skew_py(data)` | **6-23x** | Best on <1K elements |
| `scipy.stats.kurtosis(data)` | `scirs2.kurtosis_py(data)` | **5-24x** | Best on <1K elements |
| `scipy.stats.iqr(data)` | `scirs2.iqr_py(data)` | **1.7x** | Small datasets |
| `scipy.stats.pearsonr(x, y)` | `scirs2.pearsonr_py(x, y)` | **3x** | Small datasets |

#### Descriptive Statistics

| SciPy | SciRS2 | Notes |
|-------|--------|-------|
| `scipy.stats.describe(data)` | `scirs2.describe_py(data)` | Returns dict, not tuple |

### ⚠️ NOT Recommended (Use SciPy Instead)

#### Linear Algebra (200-700x slower!)

| NumPy/SciPy | scirs2 (AVOID!) | Slowdown |
|-------------|-----------------|----------|
| `np.linalg.det(A)` | ❌ `scirs2.det_py(A)` | **377x slower** |
| `np.linalg.inv(A)` | ❌ `scirs2.inv_py(A)` | **714x slower** |
| `np.linalg.solve(A, b)` | ❌ `scirs2.solve_py(A, b)` | **207x slower** |
| `scipy.linalg.qr(A)` | ❌ `scirs2.qr_py(A)` | **430x slower** |

**Action**: **NEVER migrate linear algebra code to scirs2**

#### FFT (62x slower!)

| NumPy | scirs2 (AVOID!) | Slowdown |
|-------|-----------------|----------|
| `np.fft.rfft(signal)` | ❌ `scirs2.rfft_py(signal)` | **62x slower** |
| `np.fft.fft(signal)` | ❌ `scirs2.fft_py(signal)` | **62x slower** |

**Action**: **NEVER migrate FFT code to scirs2**

#### Basic Statistics on Large Data (10-50x slower)

| NumPy | scirs2 (AVOID for >10K!) | Slowdown |
|-------|--------------------------|----------|
| `np.mean(data)` | ❌ `scirs2.mean_py(data)` | **50x slower** |
| `np.std(data)` | ❌ `scirs2.std_py(data)` | **13x slower** |
| `np.var(data)` | ❌ `scirs2.var_py(data)` | **12x slower** |
| `np.median(data)` | ❌ `scirs2.median_py(data)` | **14x slower** |

---

## Migration Examples

### Example 1: Distribution Analysis (RECOMMENDED ✅)

**Before (Pure SciPy)**:
```python
import numpy as np
from scipy import stats

data = np.random.randn(500)

# All using SciPy
mean = np.mean(data)
std = np.std(data)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

print(f"Mean: {mean:.4f}, Std: {std:.4f}")
print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
```

**After (Hybrid Approach - Optimal ✅)**:
```python
import numpy as np
import scirs2

data = np.random.randn(500)

# ✅ Use NumPy for basics (FAST)
mean = np.mean(data)
std = np.std(data)

# ✅ Use scirs2 for complex stats (FASTER!)
skewness = scirs2.skew_py(data)  # 6x faster!
kurtosis = scirs2.kurtosis_py(data)  # 5x faster!

print(f"Mean: {mean:.4f}, Std: {std:.4f}")
print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
```

**Performance Improvement**: Complex statistics 6x faster, same results!

---

### Example 2: Statistical Summary (RECOMMENDED ✅)

**Before (SciPy)**:
```python
from scipy import stats
import numpy as np

data = np.random.randn(1000)

# SciPy's describe returns namedtuple
result = stats.describe(data)
print(f"Mean: {result.mean}")
print(f"Variance: {result.variance}")
print(f"Skewness: {result.skewness}")
print(f"Kurtosis: {result.kurtosis}")
```

**After (scirs2)**:
```python
import scirs2
import numpy as np

data = np.random.randn(1000)

# scirs2's describe returns dict
result = scirs2.describe_py(data)
print(f"Mean: {result['mean']}")
print(f"Variance: {result['var']}")
print(f"Skewness: {result['skewness']}")  # Faster!
print(f"Kurtosis: {result['kurtosis']}")  # Faster!
```

**API Difference**: `result.attribute` → `result['key']` (dict instead of namedtuple)

---

### Example 3: Correlation Analysis (RECOMMENDED for small data ✅)

**Before (SciPy)**:
```python
from scipy import stats
import numpy as np

x = np.random.randn(500)
y = 2 * x + np.random.randn(500) * 0.5

# Returns (correlation, p-value)
corr, pval = stats.pearsonr(x, y)
print(f"Correlation: {corr:.4f}, p-value: {pval:.4e}")
```

**After (scirs2 for small data)**:
```python
import scirs2
import numpy as np

x = np.random.randn(500)
y = 2 * x + np.random.randn(500) * 0.5

# Returns correlation only (3x faster on small data!)
corr = scirs2.pearsonr_py(x, y)
print(f"Correlation: {corr:.4f}")

# Note: p-value not currently available in scirs2
# Use SciPy if you need p-values
```

**API Difference**: No p-value returned (use SciPy if needed)

---

### Example 4: Linear Algebra (NOT RECOMMENDED ❌)

**Before (NumPy/SciPy)**:
```python
import numpy as np
from scipy import linalg

A = np.random.randn(100, 100)
b = np.random.randn(100)

det = linalg.det(A)
inv = linalg.inv(A)
x = linalg.solve(A, b)
```

**DO NOT Migrate to scirs2** ❌:
```python
import scirs2
import numpy as np

A = np.random.randn(100, 100)
b = np.random.randn(100)

# ❌ 377x slower!
det = scirs2.det_py(A)

# ❌ 714x slower!
inv = scirs2.inv_py(A)

# ❌ 207x slower!
x = scirs2.solve_py(A, b)
```

**Recommendation**: **Keep using NumPy/SciPy for ALL linear algebra**

---

### Example 5: FFT (NOT RECOMMENDED ❌)

**Before (NumPy)**:
```python
import numpy as np

signal = np.random.randn(1024)
spectrum = np.fft.rfft(signal)
```

**DO NOT Migrate to scirs2** ❌:
```python
import scirs2

signal = np.random.randn(1024)
spectrum = scirs2.rfft_py(signal)  # ❌ 62x slower!
```

**Recommendation**: **Keep using NumPy for ALL FFT operations**

---

## API Differences

### Return Types

#### SciPy `describe` (namedtuple) vs scirs2 (dict)

**SciPy**:
```python
from scipy import stats
result = stats.describe(data)
# Access: result.mean, result.variance, result.skewness
```

**scirs2**:
```python
import scirs2
result = scirs2.describe_py(data)
# Access: result['mean'], result['var'], result['skewness']
```

#### Pearson Correlation

**SciPy** (returns correlation + p-value):
```python
from scipy import stats
corr, pval = stats.pearsonr(x, y)
```

**scirs2** (returns correlation only):
```python
import scirs2
corr = scirs2.pearsonr_py(x, y)
# No p-value - use SciPy if needed
```

### Function Naming

| Concept | SciPy | scirs2 |
|---------|-------|--------|
| Skewness | `scipy.stats.skew()` | `scirs2.skew_py()` |
| Kurtosis | `scipy.stats.kurtosis()` | `scirs2.kurtosis_py()` |
| Percentile | `np.percentile()` | `scirs2.percentile_py()` |
| IQR | `scipy.stats.iqr()` | `scirs2.iqr_py()` |

**Note**: All scirs2 functions have `_py` suffix for Python bindings

### Parameter Differences

#### Standard Deviation (ddof parameter)

**NumPy** (default ddof=0):
```python
import numpy as np
std = np.std(data)  # Population std (ddof=0)
std = np.std(data, ddof=1)  # Sample std
```

**scirs2** (same):
```python
import scirs2
std = scirs2.std_py(data)  # Population std (ddof=0)
std = scirs2.std_py(data, ddof=1)  # Sample std
```

---

## Migration Decision Tree

```
START: Do I need to migrate this SciPy code?
│
├─> Is it LINEAR ALGEBRA?
│   └─> ❌ NO - Keep using SciPy (200-700x faster)
│
├─> Is it FFT?
│   └─> ❌ NO - Keep using NumPy (62x faster)
│
├─> Is it BASIC STATS (mean/std/var) on LARGE data (>10K)?
│   └─> ❌ NO - Keep using NumPy (10-50x faster)
│
├─> Is it COMPLEX STATS (skew/kurt) on SMALL data (<10K)?
│   └─> ✅ YES - Migrate to scirs2 (4-23x faster!)
│
├─> Is it CORRELATION on SMALL data (<1K)?
│   └─> ✅ YES - Consider scirs2 (3x faster)
│       └─> But keep SciPy if you need p-values
│
└─> Otherwise
    └─> Profile both, choose based on your specific case
```

---

## Gradual Migration Strategy

### Phase 1: Identify Opportunities (Week 1)

1. **Audit your codebase**:
   ```bash
   # Find all scipy.stats usage
   grep -r "scipy.stats" your_project/
   ```

2. **Identify hot paths**:
   - Profile your code
   - Find frequently called statistical functions
   - Note data sizes

3. **Categorize operations**:
   - ✅ Complex stats on small data → Migrate to scirs2
   - ❌ Linear algebra → Keep SciPy
   - ❌ FFT → Keep NumPy
   - ⚠️ Others → Benchmark first

### Phase 2: Pilot Migration (Week 2)

1. **Start small**:
   ```python
   # Before
   from scipy import stats
   skew = stats.skew(data)

   # After
   import scirs2
   skew = scirs2.skew_py(data)
   ```

2. **Verify results**:
   ```python
   import numpy as np
   from scipy import stats
   import scirs2

   data = np.random.randn(500)

   scipy_skew = stats.skew(data)
   scirs2_skew = scirs2.skew_py(data)

   assert np.allclose(scipy_skew, scirs2_skew, rtol=1e-10)
   ```

3. **Benchmark**:
   ```python
   import time

   # Benchmark SciPy
   start = time.perf_counter()
   for _ in range(100):
       stats.skew(data)
   scipy_time = time.perf_counter() - start

   # Benchmark scirs2
   start = time.perf_counter()
   for _ in range(100):
       scirs2.skew_py(data)
   scirs2_time = time.perf_counter() - start

   print(f"Speedup: {scipy_time / scirs2_time:.2f}x")
   ```

### Phase 3: Expand (Week 3-4)

1. **Migrate related operations**:
   ```python
   # Migrate entire distribution analysis
   import numpy as np
   import scirs2

   # All complex stats together
   skew = scirs2.skew_py(data)
   kurt = scirs2.kurtosis_py(data)
   iqr = scirs2.iqr_py(data)
   ```

2. **Update tests**:
   ```python
   def test_distribution_analysis():
       data = np.random.randn(500)

       # Test scirs2 matches SciPy
       scipy_skew = stats.skew(data)
       scirs2_skew = scirs2.skew_py(data)

       assert np.allclose(scipy_skew, scirs2_skew, rtol=1e-10)
   ```

3. **Document changes**:
   ```python
   def analyze_distribution(data):
       """
       Analyze distribution shape.

       Uses scirs2 for complex statistics (6-23x faster than SciPy
       on small-medium datasets).

       Args:
           data: numpy array of data points

       Returns:
           dict with skewness and kurtosis
       """
       return {
           'skewness': scirs2.skew_py(data),
           'kurtosis': scirs2.kurtosis_py(data)
       }
   ```

### Phase 4: Optimize (Ongoing)

1. **Profile hybrid code**:
   - Measure end-to-end performance
   - Identify remaining bottlenecks
   - Optimize based on actual usage

2. **Fine-tune data size thresholds**:
   ```python
   def compute_skewness(data):
       """Choose implementation based on data size"""
       if len(data) < 10000:
           return scirs2.skew_py(data)  # Faster for small data
       else:
           return stats.skew(data)  # Faster for large data
   ```

---

## Common Migration Pitfalls

### Pitfall 1: Migrating Everything

**Wrong** ❌:
```python
# Replacing ALL SciPy with scirs2
import scirs2 as scipy  # DON'T DO THIS!
```

**Right** ✅:
```python
# Use both libraries together
import scipy.stats
import scipy.linalg
import scirs2

# Use each for what it's best at
```

### Pitfall 2: Ignoring Data Size

**Wrong** ❌:
```python
# Using scirs2 for large data
large_data = np.random.randn(100000)
mean = scirs2.mean_py(large_data)  # 50x slower!
```

**Right** ✅:
```python
# Check data size first
if len(data) < 10000:
    result = scirs2.skew_py(data)  # Fast
else:
    result = stats.skew(data)  # Faster for large data
```

### Pitfall 3: Not Profiling

**Wrong** ❌:
```python
# Assuming all scirs2 operations are faster
result = scirs2.some_function_py(data)
```

**Right** ✅:
```python
# Always profile your specific use case
import time

# Test both implementations
scipy_time = time_function(lambda: stats.func(data))
scirs2_time = time_function(lambda: scirs2.func_py(data))

if scirs2_time < scipy_time:
    use_scirs2 = True
```

---

## Testing Your Migration

### Unit Tests

```python
import numpy as np
from scipy import stats
import scirs2
import pytest

def test_skewness_matches_scipy():
    """scirs2 skewness should match SciPy within tolerance"""
    np.random.seed(42)
    data = np.random.randn(500)

    scipy_result = stats.skew(data)
    scirs2_result = scirs2.skew_py(data)

    assert np.allclose(scipy_result, scirs2_result, rtol=1e-10)

def test_kurtosis_matches_scipy():
    """scirs2 kurtosis should match SciPy within tolerance"""
    np.random.seed(42)
    data = np.random.randn(500)

    scipy_result = stats.kurtosis(data)
    scirs2_result = scirs2.kurtosis_py(data)

    assert np.allclose(scipy_result, scirs2_result, rtol=1e-10)
```

### Performance Tests

```python
def test_skewness_performance():
    """scirs2 skewness should be faster on small data"""
    import time

    data = np.random.randn(500)

    # Benchmark SciPy
    start = time.perf_counter()
    for _ in range(100):
        stats.skew(data)
    scipy_time = time.perf_counter() - start

    # Benchmark scirs2
    start = time.perf_counter()
    for _ in range(100):
        scirs2.skew_py(data)
    scirs2_time = time.perf_counter() - start

    speedup = scipy_time / scirs2_time
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"
```

---

## Rollback Plan

If migration causes issues:

1. **Conditional import**:
   ```python
   try:
       import scirs2
       USE_SCIRS2 = True
   except ImportError:
       import scipy.stats
       USE_SCIRS2 = False

   def compute_skewness(data):
       if USE_SCIRS2:
           return scirs2.skew_py(data)
       else:
           return scipy.stats.skew(data)
   ```

2. **Feature flag**:
   ```python
   # config.py
   ENABLE_SCIRS2 = os.environ.get('USE_SCIRS2', 'false').lower() == 'true'

   # your_code.py
   if ENABLE_SCIRS2:
       skew = scirs2.skew_py(data)
   else:
       skew = stats.skew(data)
   ```

3. **Gradual rollout**:
   ```python
   import random

   def compute_skewness(data):
       # Use scirs2 for 10% of requests
       if random.random() < 0.1:
           return scirs2.skew_py(data)
       else:
           return stats.skew(data)
   ```

---

## Success Metrics

Track these metrics during migration:

1. **Performance**:
   - [ ] Complex stats 4-23x faster
   - [ ] Overall statistical analysis >2x faster
   - [ ] No degradation in linear algebra or FFT

2. **Correctness**:
   - [ ] All unit tests pass
   - [ ] Results match SciPy within tolerance (rtol=1e-10)
   - [ ] No numerical instabilities

3. **Maintainability**:
   - [ ] Code is well-documented
   - [ ] Clear comments on why scirs2 is used
   - [ ] Rollback plan in place

---

## Summary

### DO Migrate ✅
- Complex statistics (skewness, kurtosis) on small-medium data
- Distribution analysis
- Code where type safety is valuable

### DON'T Migrate ❌
- Linear algebra (200-700x slower)
- FFT operations (62x slower)
- Basic statistics on large data (10-50x slower)

### Hybrid Approach (Recommended) 🎯
```python
import numpy as np
import scipy.stats
import scipy.linalg
import scirs2

# Use NumPy for basics
mean = np.mean(data)

# Use scirs2 for complex stats
skew = scirs2.skew_py(data)

# Use SciPy for linalg
det = scipy.linalg.det(matrix)
```

---

**Need Help?**
- [Performance Guide](../performance.md)
- [Troubleshooting](./troubleshooting.md)
- [GitHub Issues](https://github.com/cool-japan/scirs/issues)

**Happy Migrating!** 🦀📊
