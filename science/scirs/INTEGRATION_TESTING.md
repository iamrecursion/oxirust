# SciRS2 v0.2.0 Integration Testing Strategy

This document describes the comprehensive integration testing strategy for SciRS2 v0.2.0, focusing on cross-crate workflows, API compatibility, and performance validation.

## Objectives

1. **Validate Cross-Module Workflows**: Ensure that different SciRS2 modules work correctly together in real-world scenarios
2. **Test Data Flow**: Verify that data can flow seamlessly between modules without unnecessary copying
3. **API Compatibility**: Ensure that module APIs are compatible and work together as expected
4. **Performance Validation**: Measure and validate end-to-end performance of integrated workflows
5. **Memory Efficiency**: Verify that cross-module operations don't cause memory issues

## Test Organization

Integration tests are organized in the `scirs2-integration-tests` package:

```
scirs2-integration-tests/
├── Cargo.toml                  # Test package configuration
├── README.md                   # Package documentation
├── src/lib.rs                  # Library placeholder
└── tests/
    ├── README.md               # Detailed test documentation
    ├── integration.rs          # Main test entry point
    └── integration/
        ├── mod.rs              # Module declarations
        ├── common/             # Common utilities
        ├── fixtures/           # Test data
        ├── neural_optimize.rs  # Neural + Optimize tests
        ├── fft_signal.rs       # FFT + Signal tests
        ├── sparse_linalg.rs    # Sparse + Linalg tests
        ├── ndimage_vision.rs   # NDImage + Vision tests
        ├── stats_datasets.rs   # Stats + Datasets tests
        └── performance.rs      # Performance tests
```

## Test Categories

### 1. Neural Network + Optimization (neural_optimize.rs)

**Workflow**: ML training pipelines

**Tests**:
- SGD/Adam optimizer integration with neural networks
- Gradient computation and flow verification
- Hyperparameter optimization workflows
- Learning rate scheduling
- Early stopping integration
- Batch processing efficiency
- Zero-copy tensor passing
- Memory-efficient training loops

**Property Tests**:
- Loss should decrease with training
- Gradient descent convergence properties
- Optimizer state consistency

**Performance Targets**:
- Training 1000 samples with 100 features: < 10 seconds

### 2. FFT + Signal Processing (fft_signal.rs)

**Workflow**: Spectral analysis pipelines

**Tests**:
- FFT-based filtering
- Spectral analysis (PSD, spectrograms)
- Window function integration
- Convolution via FFT
- Filter design and application
- Hilbert transform
- Real-valued FFT (RFFT)
- 2D FFT for images

**Property Tests**:
- Parseval's theorem (energy conservation)
- FFT linearity
- FFT/IFFT roundtrip accuracy

**Performance Targets**:
- 65536-point FFT pipeline: < 100 ms

### 3. Sparse + Linear Algebra (sparse_linalg.rs)

**Workflow**: Sparse linear algebra operations

**Tests**:
- Sparse-dense matrix operations
- Sparse linear system solving
- Eigenvalue computation
- Matrix factorizations (LU, Cholesky, QR)
- Sparse format conversions
- Iterative solvers with preconditioning
- Sparse SVD
- Graph operations on sparse matrices

**Property Tests**:
- Sparse-dense consistency
- Matrix symmetry preservation
- Solver accuracy verification

**Performance Targets**:
- Sparse operations should be faster than dense for low-density matrices

### 4. Image Processing + Vision (ndimage_vision.rs)

**Workflow**: Image processing and computer vision pipelines

**Tests**:
- Image filtering workflows
- Edge detection (Sobel, Canny)
- Morphological operations
- Image segmentation
- Feature detection and matching
- Image pyramids
- Image registration
- Optical flow computation

**Property Tests**:
- Filter commutativity
- Scale covariance of features
- Morphological duality

**Performance Targets**:
- 2048x2048 image processing: < 500 ms

### 5. Statistics + Datasets (stats_datasets.rs)

**Workflow**: Statistical data analysis

**Tests**:
- Statistical analysis on loaded datasets
- Data normalization and standardization
- Correlation analysis
- Hypothesis testing
- Distribution fitting
- Cross-validation integration
- Outlier detection
- PCA and dimensionality reduction
- Time series analysis

**Property Tests**:
- Mean invariance under centering
- Correlation bounds
- Variance positivity
- Standardization unit variance

**Performance Targets**:
- Statistical analysis on 10,000 samples: efficient memory usage

### 6. Performance Integration (performance.rs)

**Focus**: End-to-end performance validation

**Tests**:
- Neural training pipeline performance
- FFT signal processing performance
- Sparse linear algebra performance
- Image processing pipeline performance
- Statistical analysis performance
- Memory efficiency validation
- Zero-copy transfer verification
- Parallel processing efficiency
- GPU/CPU handoff testing (when available)
- Cache efficiency
- Performance scaling analysis

## Test Infrastructure

### Common Utilities (common/test_utils.rs)

Helper functions for all tests:
- `create_test_array_1d()`: Deterministic 1D array generation
- `create_test_array_2d()`: Deterministic 2D array generation
- `measure_time()`: Execution time measurement
- `arrays_approx_equal()`: Approximate array comparison
- `assert_memory_efficient()`: Memory usage validation
- Property test generators (dimensions, tolerances, etc.)

### Test Fixtures (fixtures/mod.rs)

Standard test datasets:
- XOR dataset (classification)
- Linear regression data
- Sinusoidal signals (FFT/signal tests)
- Sparse matrices (configurable density)
- Test images (gradients, patterns)
- Normal distribution samples

## Testing Policies

### No-Unwrap Policy
All tests follow the no-unwrap policy:
```rust
// ❌ Bad
let result = operation().unwrap();

// ✅ Good
let result = operation().expect("Operation should succeed");

// ✅ Better
let result = operation()?;
```

### Property-Based Testing
Use `proptest` for mathematical properties:
```rust
proptest! {
    #[test]
    fn prop_fft_parseval_theorem(signal_len in 64usize..256) {
        // Test energy conservation in FFT
    }
}
```

### Memory Efficiency Validation
Use helper for memory checks:
```rust
assert_memory_efficient(
    || perform_operation(),
    max_mb: 500.0,
    description: "Operation description",
)?;
```

### Performance Targets
Include performance assertions:
```rust
assert!(duration.as_secs() < 10,
        "Operation too slow: {:.3}s", duration.as_secs_f64());
```

## Running Integration Tests

### Basic Usage
```bash
# All integration tests
cargo test --package scirs2-integration-tests

# Specific category
cargo test --package scirs2-integration-tests --test integration neural_optimize

# With verbose output
cargo test --package scirs2-integration-tests -- --nocapture

# Performance benchmarks (ignored by default)
cargo test --package scirs2-integration-tests --ignored
```

### Feature Flags
```bash
# With GPU/CUDA support
cargo test --package scirs2-integration-tests --features cuda

# With SIMD optimizations
cargo test --package scirs2-integration-tests --features simd
```

### Continuous Integration
Integration tests should be run:
- On every pull request
- Before releases
- Nightly for performance regression detection

## Implementation Status

### ✅ Complete
- Test infrastructure and utilities
- Test fixtures and data generators
- Test organization and structure
- Property-based testing framework
- Performance measurement utilities
- Comprehensive test documentation

### 🚧 In Progress
Many test implementations contain `TODO` markers where actual test logic
should be added once module APIs stabilize. The patterns and structure are
ready for implementation.

### 📋 TODO
- Fill in neural + optimize test implementations
- Complete FFT + signal test implementations
- Implement advanced sparse matrix tests
- Add GPU-specific integration tests
- Implement distributed processing tests

## Integration Patterns

### Pattern 1: Data Pipeline
```rust
// Load data -> Preprocess -> Train -> Evaluate
let (features, labels) = load_dataset()?;
let normalized = normalize(&features)?;
let model = train_model(&normalized, &labels)?;
let metrics = evaluate(&model, &test_data)?;
```

### Pattern 2: Signal Processing
```rust
// Load signal -> Window -> FFT -> Filter -> IFFT
let signal = load_signal()?;
let windowed = apply_window(&signal)?;
let spectrum = fft(&windowed)?;
let filtered = apply_filter(&spectrum)?;
let result = ifft(&filtered)?;
```

### Pattern 3: Image Processing
```rust
// Load image -> Filter -> Detect edges -> Extract features
let image = load_image()?;
let smoothed = gaussian_filter(&image)?;
let edges = canny_detector(&smoothed)?;
let features = detect_corners(&edges)?;
```

## Best Practices

### 1. Use Fixtures
Prefer standard fixtures over creating custom data:
```rust
let (x, y) = TestDatasets::xor_dataset();
```

### 2. Measure Everything Important
```rust
let (result, perf) = measure_time("Operation", || {
    perform_operation()
})?;
println!("Completed in {:.3} ms", perf.duration_ms);
```

### 3. Test Error Paths
Don't just test the happy path:
```rust
// Test invalid inputs
let result = operation_with_invalid_input();
assert!(result.is_err());
```

### 4. Document Test Intent
```rust
/// Test that neural network can be trained with SGD optimizer
/// Verifies gradient flow and weight updates work correctly
#[test]
fn test_neural_with_sgd() {
    // ...
}
```

## Troubleshooting

### Test Compilation Issues
If tests fail to compile, check:
1. Module APIs may have changed
2. Feature flags may need updating
3. Dependencies may need version updates

### Test Failures
If tests fail at runtime:
1. Check if module implementations are complete
2. Verify test assumptions are still valid
3. Check for numerical precision issues
4. Review error messages for API changes

### Performance Issues
If performance tests fail:
1. Run in release mode: `cargo test --release`
2. Check system load during testing
3. Verify data sizes match expectations
4. Profile to identify bottlenecks

## Contributing

When adding new integration tests:
1. Choose appropriate test file for modules involved
2. Follow existing patterns and conventions
3. Add both positive and negative test cases
4. Include property-based tests where applicable
5. Document test objectives and expectations
6. Add performance targets for critical operations
7. Update this document with new test categories

## References

- [Rust Testing Best Practices](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [proptest Documentation](https://docs.rs/proptest/)
- [Integration Testing in Cargo](https://doc.rust-lang.org/cargo/guide/tests.html)

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)
