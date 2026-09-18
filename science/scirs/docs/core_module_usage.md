# SciRS2 Core Module Usage Guidelines

## Introduction

This document provides guidelines for effectively using the `scirs2-core` module throughout the SciRS2 project. The Core Module Usage Policy aims to:

1. **Standardize Code Patterns**: Use consistent approaches across the codebase
2. **Reduce Duplication**: Centralize common functionality to avoid redundant implementations
3. **Improve Maintainability**: Make code more maintainable by leveraging the core module
4. **Simplify API**: Provide a clean, well-documented interface for common operations

## Core Module Features

The `scirs2-core` module provides several key features that should be used consistently across the project:

### Validation Utilities (`validation` feature)

The validation module provides functions for validating various types of inputs:

```rust
use scirs2_core::validation::{
    check_probability,
    check_probabilities,
    check_probabilities_sum_to_one,
    check_positive,
    check_non_negative,
    check_in_bounds,
    check_finite,
    check_array_finite,
    check_same_shape,
    check_shape,
    check_square,
    check_1d,
    check_2d,
};
```

**When to use**: Always prefer these validation functions over custom validation code. They provide comprehensive error messages and proper error propagation.

**Example usage**:
```rust
use scirs2_core::validation::check_probability;
use scirs2_stats::error::StatsError;

pub fn new(p: f64) -> StatsResult<Self> {
    // Validate probability is between 0 and 1
    check_probability(p, "Success probability").map_err(StatsError::from)?;
    
    // ...rest of implementation
}
```

### SIMD Acceleration (`simd` feature)

The SIMD module provides vectorized operations for improved performance:

```rust
use scirs2_core::simd::{
    simd_add,
    simd_subtract,
    simd_multiply,
    simd_divide,
    simd_min,
    simd_max,
};
```

**When to use**: Use SIMD operations for performance-critical array operations, especially in inner loops that process large amounts of data.

**Example usage**:
```rust
use scirs2_core::simd::simd_add;

// Instead of:
for i in 0..a.len() {
    c[i] = a[i] + b[i];
}

// Use:
simd_add(&a, &b, &mut c);
```

### Parallel Processing (`parallel` feature)

The parallel module provides utilities for parallel processing using Rayon:

```rust
use scirs2_core::parallel::{
    parallel_map,
    parallel_reduce,
    parallel_for_each,
};
```

**When to use**: Use parallel processing for CPU-bound operations that can be parallelized, such as applying operations to large arrays or independent computations.

**Example usage**:
```rust
use scirs2_core::parallel::parallel_map;

// Instead of:
let results: Vec<f64> = input.iter().map(|x| expensive_function(*x)).collect();

// Use:
let results: Vec<f64> = parallel_map(&input, |x| expensive_function(*x));
```

### Caching and Memoization (`cache` feature)

The cache module provides utilities for caching computation results:

```rust
use scirs2_core::cache::{
    CacheBuilder,
    TTLSizedCache,
};
```

**When to use**: Use caching for expensive computations that are likely to be repeated, such as loading datasets or computing complex functions with the same inputs.

**Example usage**:
```rust
use scirs2_core::cache::CacheBuilder;
use std::cell::RefCell;

struct Calculator {
    // Cache expensive computation results
    computation_cache: RefCell<TTLSizedCache<Input, Output>>,
}

impl Calculator {
    pub fn new() -> Self {
        let computation_cache = RefCell::new(
            CacheBuilder::new()
                .with_size(100)
                .with_ttl(3600) // 1 hour TTL
                .build_sized_cache()
        );
        
        Self { computation_cache }
    }
    
    pub fn compute(&self, input: Input) -> Output {
        // Check cache first
        if let Some(result) = self.computation_cache.borrow().get(&input) {
            return result.clone();
        }
        
        // Perform expensive computation
        let result = expensive_computation(&input);
        
        // Cache the result
        self.computation_cache.borrow_mut().insert(input, result.clone());
        
        result
    }
}
```

### Numeric Utilities

The numeric module provides utilities for working with numeric types:

```rust
use scirs2_core::numeric::{
    ScalarType,
    RealNum,
    format_scientific,
};
```

**When to use**: Use these utilities when working with generic numeric types or when formatting numeric output.

### Utility Functions

The utils module provides various utility functions:

```rust
use scirs2_core::utils::{
    // Array comparison
    is_close,
    points_equal,
    arrays_equal,
    
    // Array generation and manipulation
    linspace,
    logspace,
    arange,
    fill_diagonal,
    pad_array,
    get_window,
    
    // Element-wise operations
    maximum,
    minimum,
    
    // Vector operations
    normalize,
    
    // Numerical calculus
    differentiate,
    integrate,
    
    // General utilities
    prod,
    all,
    any,
};
```

**When to use**: Always prefer these utility functions over custom implementations to ensure consistency and avoid duplication.

**Example usage**:
```rust
use scirs2_core::utils::{normalize, pad_array, differentiate};

// Normalize a vector to unit energy
let signal = vec![1.0, 2.0, 3.0, 4.0];
let normalized = normalize(&signal, "energy").unwrap();

// Pad an array
let arr = array![1.0, 2.0, 3.0];
let padded = pad_array(&arr, &[(1, 2)], "constant", Some(0.0)).unwrap();

// Differentiate a function
let f = |x| x * x;
let derivative = differentiate(3.0, 0.001, |x| Ok(f(x))).unwrap();
```

## Feature Flags

The core module uses feature flags to enable optional functionality:

```toml
[dependencies]
scirs2-core = { workspace = true, features = ["simd", "parallel", "cache", "validation"] }
```

Each module should enable only the features it requires:

- Enable `simd` for modules performing numerical computations
- Enable `parallel` for modules with parallelizable operations
- Enable `cache` for modules that need caching/memoization
- Enable `validation` for modules that need input validation

## Error Handling

All modules should properly handle and propagate errors from the core module:

```rust
pub enum ModuleError {
    // Other module-specific errors...
    
    // Core error propagation
    #[error("{0}")]
    CoreError(#[from] scirs2_core::error::CoreError),
}
```

## Best Practices

1. **Check Core First**: Before implementing new functionality, check if `scirs2-core` already provides it.

2. **Feature Consistency**: Ensure feature flags are used consistently across modules.

3. **Error Conversion**: Implement proper error conversion between module-specific errors and core errors.

4. **Documentation**: Document the use of core module features in your code.

5. **Centralization Requests**: If you find yourself implementing functionality that could benefit other modules, consider contributing it to `scirs2-core`.

## Contribution Process for Core Module

When you identify functionality that should be moved to the core module:

1. **Evaluate Usage**: Ensure the functionality is general enough to benefit multiple modules.

2. **Standardize Interface**: Design a clean, well-documented interface that will work across modules.

3. **Implement Tests**: Provide comprehensive tests for the functionality.

4. **Update Documentation**: Update this guide if necessary to reflect the new functionality.

5. **Migrate Existing Code**: Help migrate existing code to use the new core functionality.

## Examples of Proper Core Module Usage

### Example 1: Validation in Distribution Implementation

```rust
// Good example - using core validation
use scirs2_core::validation::check_probability;

impl<F: Float + std::fmt::Display> Bernoulli<F> {
    pub fn new(p: F) -> StatsResult<Self> {
        // Use core validation function
        check_probability(p, "Success probability").map_err(StatsError::from)?;
        
        Ok(Self { p })
    }
}

// Bad example - custom validation
impl<F: Float> Bernoulli<F> {
    pub fn new(p: F) -> StatsResult<Self> {
        // Duplicated validation logic
        if p < F::zero() || p > F::one() {
            return Err(StatsError::ValueError(
                format!("Success probability must be between 0 and 1, got {}", p)
            ));
        }
        
        Ok(Self { p })
    }
}
```

### Example 2: Using SIMD in Numerical Computations

```rust
// Good example - using core SIMD
use scirs2_core::simd::simd_add;

pub fn add_arrays(a: &[f64], b: &[f64], c: &mut [f64]) {
    simd_add(a, b, c);
}

// Bad example - manual loop
pub fn add_arrays(a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}
```

### Example 3: Caching Expensive Computations

```rust
// Good example - using core cache
use scirs2_core::cache::CacheBuilder;
use std::cell::RefCell;

pub struct DataLoader {
    cache: RefCell<TTLSizedCache<String, Vec<f64>>>,
}

impl DataLoader {
    pub fn new() -> Self {
        let cache = RefCell::new(
            CacheBuilder::new()
                .with_size(100)
                .with_ttl(3600)
                .build_sized_cache()
        );
        
        Self { cache }
    }
}

// Bad example - custom cache implementation
pub struct DataLoader {
    cache: HashMap<String, (Vec<f64>, Instant)>,
    max_size: usize,
}

impl DataLoader {
    // Custom cache implementation with expiration logic
    // ...
}
```

## Conclusion

Consistent use of the core module features will lead to a more cohesive, maintainable codebase. By leveraging the functionality provided by `scirs2-core`, we can reduce duplication, standardize code patterns, and improve the overall quality of the SciRS2 project.