# SciRS2 API Standardization Guide

## Overview

This document defines the API standards for all SciRS2 modules to ensure consistency and ease of use across the ecosystem.

## 1. Error Handling

### Standard Pattern
- Each module MUST define its own error type named `{Module}Error`
- Each module MUST provide a type alias `{Module}Result<T> = Result<T, {Module}Error>`
- Error types MUST implement `std::error::Error`, `Display`, and `Debug` traits
- Errors MUST be convertible from `CoreError` using the `From` trait

### Example
```rust
// In scirs2-stats/src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum StatsError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Core error: {0}")]
    Core(#[from] CoreError),
    
    // ... other variants
}

pub type StatsResult<T> = Result<T, StatsError>;
```

## 2. Function Naming Conventions

### General Rules
- Use snake_case for all function names
- Follow SciPy naming conventions where applicable for API familiarity
- Use full words rather than abbreviations for clarity, unless the abbreviation is well-established (e.g., `fft`, `svd`)

### Standard Names
- Matrix operations: `determinant`, `inverse`, `transpose` (not `det`, `inv`, `trans`)
- Statistical functions: Use SciPy names: `mean`, `std`, `var` (established abbreviations)
- Transformations: `forward`, `inverse` (not `fwd`, `inv`)

## 3. Parameter Naming Conventions

### Standard Parameter Names by Domain

#### Linear Algebra Operations
- `a` - Primary matrix/array input
- `b` - Secondary matrix/array input (for binary operations)
- `alpha`, `beta` - Scalar multipliers
- `trans_a`, `trans_b` - Transpose flags

#### Statistical Operations
- `data` or `x` - Input data array
- `axis` - Axis along which to compute
- `ddof` - Delta degrees of freedom
- `weights` - Optional weights array

#### Signal/Image Processing
- `input` - Input array/signal/image
- `output` - Output array (for in-place operations)
- `filter` or `kernel` - Filter/kernel array
- `mode` - Boundary handling mode
- `cval` - Constant value for padding

#### Optimization
- `x0` - Initial guess/starting point
- `f` or `func` - Objective function
- `grad` - Gradient function
- `bounds` - Parameter bounds
- `tol` - Tolerance
- `max_iter` - Maximum iterations

### Optional Parameters
- Use `Option<T>` for truly optional parameters
- Provide sensible defaults where possible
- Consider builder patterns for functions with many optional parameters

## 4. Return Type Conventions

### Standard Patterns
- Always return `Result<T, ModuleError>` for fallible operations
- Return owned types for computed results
- Use tuples for multiple return values: `Result<(Array2<f64>, Array1<f64>)>`
- Avoid returning references unless lifetime management is trivial

### Examples
```rust
// Good
pub fn solve(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> LinalgResult<Array1<f64>>

// Avoid
pub fn solve<'a>(a: &'a ArrayView2<f64>, b: &ArrayView1<f64>) -> LinalgResult<&'a Array1<f64>>
```

## 5. Documentation Standards

### Required Sections (in order)
1. Brief one-line description
2. Detailed description (if needed)
3. Arguments
4. Returns
5. Errors
6. Examples
7. Notes (optional)
8. References (optional)

### Template
```rust
/// Computes the mean of array elements.
///
/// This function calculates the arithmetic mean along the specified axis.
/// NaN values are propagated.
///
/// # Arguments
///
/// * `data` - Input array
/// * `axis` - Axis along which to compute the mean
/// * `keepdims` - Whether to preserve dimensions
///
/// # Returns
///
/// Array containing the mean values
///
/// # Errors
///
/// Returns `StatsError::EmptyInput` if the input array is empty
///
/// # Examples
///
/// ```
/// use scirs2_stats::descriptive::mean;
/// use ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0];
/// let result = mean(&data.view(), None, false)?;
/// assert_eq!(result, 2.5);
/// # Ok::<(), scirs2_stats::StatsError>(())
/// ```
///
/// # Notes
///
/// This implementation follows NumPy's behavior for NaN handling
```

## 6. Generic Type Parameters

### Standard Names
- `T` or `A` - Array element type
- `D` - Dimension type
- `S` - Storage type
- `F` - Floating point type (when constraining to floats)
- `N` - Numeric type (when constraining to numbers)

### Trait Bounds
- Place simple bounds inline: `fn foo<T: Clone>`
- Use where clauses for complex bounds
- Order bounds consistently: lifetime, type, trait bounds

## 7. Module Organization

### Standard Structure
```
src/
├── lib.rs          # Module root with re-exports
├── error.rs        # Error types
├── types.rs        # Common type definitions
├── traits.rs       # Module-specific traits
├── utils.rs        # Internal utilities
└── feature/        # Feature-specific submodules
    ├── mod.rs      # Feature module root
    └── impl.rs     # Implementation
```

### Re-exports
- Re-export commonly used types at module root
- Keep internal implementation details private
- Use `pub use` for clean API surface

## 8. Testing Standards

### Test Organization
- Unit tests in same file as implementation
- Integration tests in `tests/` directory
- Doc tests for all public APIs
- Property-based tests where applicable

### Test Naming
- `test_function_name_condition_expected`
- `test_edge_case_description`
- Use descriptive names that explain what is being tested

## 9. Validation Functions

### Standard Signatures
```rust
// In scirs2-core/src/validation.rs
pub fn check_finite<T: Float>(value: T, name: &str) -> CoreResult<()>
pub fn check_positive<T: PartialOrd + Zero>(value: T, name: &str) -> CoreResult<()>
pub fn check_shape<D: Dimension>(actual: &D, expected: &D) -> CoreResult<()>
```

### Usage Pattern
```rust
use scirs2_core::validation::{check_finite, check_positive};

pub fn sqrt(x: f64) -> StatsResult<f64> {
    check_positive(x, "x")?;
    check_finite(x, "x")?;
    Ok(x.sqrt())
}
```

## 10. Performance Considerations

### Guidelines
- Prefer zero-copy operations where possible
- Use `ArrayView` for read-only access
- Implement both single-threaded and parallel versions for expensive operations
- Use feature flags for optional optimizations
- Document computational complexity in function docs

### Parallel Operations
```rust
pub fn operation(data: &ArrayView2<f64>) -> Result<Array2<f64>>
pub fn operation_parallel(data: &ArrayView2<f64>, n_jobs: Option<usize>) -> Result<Array2<f64>>
```

## Implementation Checklist

When implementing new functionality:

- [ ] Follow naming conventions
- [ ] Use standard error types
- [ ] Document all public APIs
- [ ] Include examples in documentation
- [ ] Add comprehensive tests
- [ ] Run clippy and fix all warnings
- [ ] Benchmark against reference implementation
- [ ] Update module documentation