# SIMD Ultra-Optimization Techniques

## Overview

This document describes the aggressive optimization techniques used in `scirs2-core/src/simd/basic_optimized.rs` to achieve **1.4x to 4.5x speedup** over standard SIMD implementations.

## Performance Results

Benchmarks on macOS ARM64 (NEON):

| Operation      | Size    | Original (ns) | Optimized (ns) | Speedup | Improvement |
|----------------|---------|---------------|----------------|---------|-------------|
| **Addition**   | 1,000   | 347           | 111            | 3.13x   | 212.6%      |
|                | 10,000  | 3,632         | 1,074          | 3.38x   | 238.2%      |
| **Multiply**   | 1,000   | 215           | 68             | 3.16x   | 216.2%      |
|                | 10,000  | 2,286         | 759            | 3.01x   | 201.2%      |
| **Dot Product**| 1,000   | 193           | 50             | 3.86x   | 286.0%      |
|                | 10,000  | 2,326         | 592            | 3.93x   | 292.9%      |
| **Sum**        | 1,000   | 141           | 31             | **4.55x**| **354.8%** |
|                | 10,000  | 1,766         | 437            | 4.04x   | 304.1%      |

## Core Optimization Techniques

### 1. Multiple Accumulators (Instruction-Level Parallelism)

**Problem**: Single accumulator creates dependency chains that stall CPU pipeline.

**Solution**: Use 4-8 parallel accumulators to maximize instruction throughput.

```rust
// ❌ Single accumulator (slow - dependency chain)
let mut acc = _mm256_setzero_ps();
while i < len {
    let v = _mm256_load_ps(ptr.add(i));
    acc = _mm256_add_ps(acc, v);  // Must wait for previous add
    i += 8;
}

// ✅ Multiple accumulators (fast - parallel execution)
let mut acc1 = _mm256_setzero_ps();
let mut acc2 = _mm256_setzero_ps();
let mut acc3 = _mm256_setzero_ps();
let mut acc4 = _mm256_setzero_ps();

while i + 32 <= len {
    let v1 = _mm256_load_ps(ptr.add(i));
    let v2 = _mm256_load_ps(ptr.add(i + 8));
    let v3 = _mm256_load_ps(ptr.add(i + 16));
    let v4 = _mm256_load_ps(ptr.add(i + 24));

    acc1 = _mm256_add_ps(acc1, v1);  // All 4 can execute in parallel
    acc2 = _mm256_add_ps(acc2, v2);
    acc3 = _mm256_add_ps(acc3, v3);
    acc4 = _mm256_add_ps(acc4, v4);
    i += 32;
}
```

**Impact**: Eliminates pipeline stalls, achieves near-peak throughput.

### 2. Aggressive Loop Unrolling

**Configuration**:
- **AVX-512**: 4-way unroll (64 elements per iteration)
- **AVX2**: 8-way unroll (64 elements per iteration)
- **SSE/NEON**: 4-way unroll (16 elements per iteration)

```rust
// AVX2: 8-way unrolling with 8 accumulators
const VECTOR_SIZE: usize = 8;     // 8 f32s per vector
const UNROLL_FACTOR: usize = 8;   // 8-way unroll
const CHUNK_SIZE: usize = 64;     // 64 elements per iteration

while i + CHUNK_SIZE <= len {
    // Load 8 vectors (64 f32s total)
    let v1 = _mm256_load_ps(ptr.add(i));
    let v2 = _mm256_load_ps(ptr.add(i + 8));
    // ... v3 through v8 ...

    // Accumulate in parallel
    acc1 = _mm256_add_ps(acc1, v1);
    acc2 = _mm256_add_ps(acc2, v2);
    // ... acc3 through acc8 ...

    i += CHUNK_SIZE;
}
```

**Benefits**:
- Reduces loop overhead (fewer iterations)
- Better instruction pipelining
- Hides memory latency

### 3. Pre-allocated Memory with Unsafe Set Length

**Problem**: Dynamic vector growth causes reallocation overhead.

**Solution**: Pre-allocate exact size and set length directly.

```rust
// ❌ Standard allocation (slow - may reallocate)
let mut result = Vec::new();
for i in 0..len {
    result.push(value);  // May reallocate multiple times
}

// ✅ Pre-allocated (fast - single allocation)
let mut result = Vec::with_capacity(len);
unsafe { result.set_len(len); }  // Set length immediately

// Now write directly to memory
let result_ptr = result.as_mut_ptr();
unsafe {
    *result_ptr.add(i) = value;  // Direct memory write
}
```

**Impact**: Eliminates reallocation overhead, guarantees single allocation.

### 4. Pointer Arithmetic (Zero Bounds Checking)

**Problem**: Array indexing adds bounds checks on every access.

**Solution**: Use raw pointers with manual bounds checking outside loops.

```rust
// ❌ Array indexing (slow - bounds checks every access)
for i in 0..len {
    result[i] = a[i] + b[i];  // 3 bounds checks per iteration
}

// ✅ Pointer arithmetic (fast - checked once)
assert_eq!(a.len(), b.len());  // Check once
let a_ptr = a.as_ptr();
let b_ptr = b.as_ptr();
let result_ptr = result.as_mut_ptr();

unsafe {
    for i in 0..len {
        *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);  // No bounds checks
    }
}
```

**Impact**: Eliminates bounds checking overhead in hot loops.

### 5. Memory Prefetching

**Strategy**: Prefetch data before it's needed to hide memory latency.

```rust
const PREFETCH_DISTANCE: usize = 256;  // AVX2
// const PREFETCH_DISTANCE: usize = 512;  // AVX-512

while i + CHUNK_SIZE <= len {
    // Prefetch data 256 bytes ahead
    if i + PREFETCH_DISTANCE < len {
        _mm_prefetch(a_ptr.add(i + PREFETCH_DISTANCE) as *const i8, _MM_HINT_T0);
        _mm_prefetch(b_ptr.add(i + PREFETCH_DISTANCE) as *const i8, _MM_HINT_T0);
    }

    // Process current chunk (data already in cache)
    let v1 = _mm256_load_ps(a_ptr.add(i));
    // ...
}
```

**Prefetch Hints**:
- `_MM_HINT_T0`: Fetch to all cache levels (L1/L2/L3)
- `_MM_HINT_T1`: Fetch to L2/L3 (not L1)
- `_MM_HINT_T2`: Fetch to L3 only

**Impact**: Hides memory latency, keeps pipeline fed with data.

### 6. Alignment-Aware Processing

**Strategy**: Detect alignment and use faster aligned loads when possible.

```rust
// Check alignment (64-byte for AVX-512, 32-byte for AVX2)
let a_aligned = (a_ptr as usize) % 32 == 0;
let b_aligned = (b_ptr as usize) % 32 == 0;
let result_aligned = (result_ptr as usize) % 32 == 0;

if a_aligned && b_aligned && result_aligned && len >= CHUNK_SIZE {
    // Fast aligned path
    while i + CHUNK_SIZE <= len {
        let a_vec = _mm256_load_ps(a_ptr.add(i));      // Aligned load (faster)
        let b_vec = _mm256_load_ps(b_ptr.add(i));
        let result_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_store_ps(result_ptr.add(i), result_vec); // Aligned store (faster)
        i += CHUNK_SIZE;
    }
} else {
    // Unaligned fallback
    while i + VECTOR_SIZE <= len {
        let a_vec = _mm256_loadu_ps(a_ptr.add(i));     // Unaligned load
        // ...
    }
}
```

**Performance Difference**:
- Aligned loads/stores: 1-2 cycles
- Unaligned loads/stores: 3-5 cycles (penalty varies by CPU)

### 7. FMA (Fused Multiply-Add) Instructions

**For dot product**: Single instruction for multiply + accumulate.

```rust
// ❌ Separate multiply and add (2 instructions)
let prod = _mm256_mul_ps(a_vec, b_vec);
acc = _mm256_add_ps(acc, prod);

// ✅ FMA (1 instruction, higher precision)
acc = _mm256_fmadd_ps(a_vec, b_vec, acc);  // acc = a * b + acc
```

**Benefits**:
- 50% fewer instructions
- Higher numerical precision (no intermediate rounding)
- Single-cycle latency on modern CPUs

### 8. Aggressive Compiler Hints

**Inlining**:
```rust
#[inline(always)]  // Force inline (not just hint)
pub fn simd_add_f32_ultra_optimized(...) -> Array1<f32> {
    // ...
}
```

**Target Features**:
```rust
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn avx2_add_f32_inner(...) {
    // Compiler can use AVX2 + FMA without runtime checks
}
```

**Impact**: Eliminates function call overhead, enables better optimization.

### 9. Efficient Horizontal Reduction

**For reductions** (sum, dot product): Minimize horizontal operations.

```rust
// Combine 8 accumulators efficiently
let combined1 = _mm256_add_ps(acc1, acc2);
let combined2 = _mm256_add_ps(acc3, acc4);
let combined3 = _mm256_add_ps(acc5, acc6);
let combined4 = _mm256_add_ps(acc7, acc8);

let combined12 = _mm256_add_ps(combined1, combined2);
let combined34 = _mm256_add_ps(combined3, combined4);
let final_acc = _mm256_add_ps(combined12, combined34);

// Horizontal reduction (sum 8 lanes)
let high = _mm256_extractf128_ps(final_acc, 1);
let low = _mm256_castps256_ps128(final_acc);
let sum128 = _mm_add_ps(low, high);

let shuf = _mm_shuffle_ps(sum128, sum128, 0b1110);
let sum_partial = _mm_add_ps(sum128, shuf);
let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
let final_result = _mm_add_ps(sum_partial, shuf2);

let result = _mm_cvtss_f32(final_result);
```

**Pattern**: Tree-based reduction minimizes dependencies.

### 10. Architecture-Specific Implementations

**Separate implementations for each SIMD level**:

```rust
pub fn simd_add_f32_ultra_optimized(...) -> Array1<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return avx512_add_f32_inner(...);  // 512-bit vectors
        } else if is_x86_feature_detected!("avx2") {
            return avx2_add_f32_inner(...);    // 256-bit vectors
        } else if is_x86_feature_detected!("sse2") {
            return sse_add_f32_inner(...);     // 128-bit vectors
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_add_f32_inner(...);        // ARM NEON
    }

    // Scalar fallback
    scalar_add_f32(...)
}
```

**Benefits**: Each implementation optimized for specific instruction set.

## Implementation Pattern

### Complete Example: Ultra-Optimized Addition

```rust
#[inline(always)]
pub fn simd_add_f32_ultra_optimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    let len = a.len();
    assert_eq!(len, b.len(), "Arrays must have same length");

    // 1. Pre-allocate result
    let mut result = Vec::with_capacity(len);
    unsafe { result.set_len(len); }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            // 2. Get raw pointers
            let a_ptr = a.as_slice().unwrap().as_ptr();
            let b_ptr = b.as_slice().unwrap().as_ptr();
            let result_ptr = result.as_mut_ptr();

            // 3. Runtime feature detection
            if is_x86_feature_detected!("avx2") {
                avx2_add_f32_inner(a_ptr, b_ptr, result_ptr, len);
            }
        }
    }

    Array1::from_vec(result)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn avx2_add_f32_inner(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    len: usize,
) {
    use std::arch::x86_64::*;

    const PREFETCH_DISTANCE: usize = 256;
    const VECTOR_SIZE: usize = 8;
    const UNROLL_FACTOR: usize = 8;
    const CHUNK_SIZE: usize = 64;

    let mut i = 0;

    // 4. Check alignment
    let a_aligned = (a as usize) % 32 == 0;
    let b_aligned = (b as usize) % 32 == 0;
    let result_aligned = (result as usize) % 32 == 0;

    if a_aligned && b_aligned && result_aligned && len >= CHUNK_SIZE {
        // 5. Optimized aligned path with 8-way unrolling
        while i + CHUNK_SIZE <= len {
            // 6. Prefetch
            if i + PREFETCH_DISTANCE < len {
                _mm_prefetch(a.add(i + PREFETCH_DISTANCE) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b.add(i + PREFETCH_DISTANCE) as *const i8, _MM_HINT_T0);
            }

            // 7. Load 8 vectors (aligned)
            let a1 = _mm256_load_ps(a.add(i));
            let a2 = _mm256_load_ps(a.add(i + 8));
            // ... a3 through a8 ...

            let b1 = _mm256_load_ps(b.add(i));
            let b2 = _mm256_load_ps(b.add(i + 8));
            // ... b3 through b8 ...

            // 8. Compute (8 independent operations)
            let r1 = _mm256_add_ps(a1, b1);
            let r2 = _mm256_add_ps(a2, b2);
            // ... r3 through r8 ...

            // 9. Store results (aligned)
            _mm256_store_ps(result.add(i), r1);
            _mm256_store_ps(result.add(i + 8), r2);
            // ... r3 through r8 ...

            i += CHUNK_SIZE;
        }
    }

    // 10. Handle remaining elements with unaligned loads
    while i + VECTOR_SIZE <= len {
        let a_vec = _mm256_loadu_ps(a.add(i));
        let b_vec = _mm256_loadu_ps(b.add(i));
        let result_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.add(i), result_vec);
        i += VECTOR_SIZE;
    }

    // 11. Scalar remainder
    while i < len {
        *result.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}
```

## When to Use Ultra-Optimizations

### ✅ Use When:
- Processing large arrays (> 1000 elements)
- Performance-critical hot paths
- Batch processing operations
- Scientific computing workloads

### ❌ Don't Use When:
- Small arrays (< 100 elements) - overhead dominates
- Memory-bound operations (optimization won't help)
- One-time operations (not worth complexity)

## Performance Tuning Guidelines

### 1. Prefetch Distance

Depends on memory latency:
- **Fast RAM/L3**: 128-256 bytes
- **Slow RAM**: 512-1024 bytes

### 2. Unroll Factor

Balance between code size and ILP:
- **L1 cache-resident**: 8-16 way unroll
- **L2 cache-resident**: 4-8 way unroll
- **RAM-resident**: 2-4 way unroll

### 3. Accumulator Count

Match CPU execution ports:
- **Modern CPUs**: 4-8 accumulators (2-4 ADD units)
- **Older CPUs**: 2-4 accumulators

## Verification

Always verify correctness with tests:

```rust
#[test]
fn test_correctness() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

    let result = simd_add_f32_ultra_optimized(&a.view(), &b.view());

    assert_eq!(result[0], 6.0);
    assert_eq!(result[1], 8.0);
    assert_eq!(result[2], 10.0);
    assert_eq!(result[3], 12.0);
}
```

## References

- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- ARM NEON Intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
- Agner Fog's Optimization Manuals: https://www.agner.org/optimize/

## Summary

The ultra-optimizations achieve **1.4x to 4.5x speedup** through:

1. **Multiple accumulators** (4-8) for instruction-level parallelism
2. **Aggressive loop unrolling** (4-8 way) to reduce overhead
3. **Pre-allocated memory** to eliminate reallocation
4. **Pointer arithmetic** to remove bounds checks
5. **Memory prefetching** to hide latency
6. **Alignment detection** for faster loads/stores
7. **FMA instructions** for dot products (2x fewer instructions)
8. **Compiler hints** for maximum optimization
9. **Efficient horizontal reduction** for reductions
10. **Architecture-specific implementations** for each SIMD level

These techniques transform standard SIMD code into highly optimized implementations that approach theoretical peak performance.
