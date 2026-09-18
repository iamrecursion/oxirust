# SciRS2 v0.2.0 Cross-Crate Integration Tests

This directory contains comprehensive integration tests that verify cross-module workflows, data flow between modules, and API compatibility across the SciRS2 ecosystem.

## Test Structure

```
tests/
├── integration.rs              # Main test entry point
├── integration/
│   ├── mod.rs                  # Module declarations
│   ├── common/                 # Common test utilities
│   │   ├── mod.rs
│   │   └── test_utils.rs      # Helper functions, property test generators
│   ├── fixtures/               # Test data and fixtures
│   │   └── mod.rs             # Standard test datasets
│   ├── neural_optimize.rs     # Neural + Optimize integration tests
│   ├── fft_signal.rs          # FFT + Signal integration tests
│   ├── sparse_linalg.rs       # Sparse + Linalg integration tests
│   ├── ndimage_vision.rs      # NDImage + Vision integration tests
│   ├── stats_datasets.rs      # Stats + Datasets integration tests
│   └── performance.rs         # Performance and memory tests
└── README.md                   # This file
```

## Test Categories

### 1. Neural + Optimize Integration (`neural_optimize.rs`)
Tests ML training pipelines combining neural networks with optimization algorithms:
- SGD and Adam optimizer integration
- Gradient flow verification
- Hyperparameter optimization
- Learning rate scheduling
- Early stopping integration
- Batch processing
- Zero-copy tensor passing
- Property-based tests for convergence

### 2. FFT + Signal Integration (`fft_signal.rs`)
Tests spectral analysis pipelines:
- FFT-based filtering workflows
- Spectral analysis (PSD, spectrogram)
- Window functions integration
- Convolution via FFT
- Filter design and application
- Hilbert transform
- Real-valued FFT (RFFT)
- 2D FFT for image processing
- Property-based tests (Parseval's theorem, linearity, roundtrip)

### 3. Sparse + Linalg Integration (`sparse_linalg.rs`)
Tests sparse linear algebra operations:
- Sparse-dense matrix multiplication
- Sparse linear system solving
- Eigenvalue computation
- Matrix factorizations (LU, Cholesky)
- Sparse format conversions (CSR, CSC, COO)
- Iterative solvers with preconditioning
- Sparse QR and SVD
- Graph operations on sparse matrices
- Property-based tests for consistency

### 4. NDImage + Vision Integration (`ndimage_vision.rs`)
Tests image processing pipelines:
- Image filtering workflows
- Edge detection (Sobel, Canny)
- Morphological operations
- Image segmentation
- Feature detection and description
- Image pyramids
- Image registration
- Object detection pipelines
- Optical flow
- Property-based tests for filter properties

### 5. Stats + Datasets Integration (`stats_datasets.rs`)
Tests statistical data analysis workflows:
- Statistical analysis on loaded datasets
- Data normalization and standardization
- Correlation analysis
- Hypothesis testing
- Distribution fitting
- Cross-validation with statistical validation
- Outlier detection
- PCA integration
- Time series analysis
- Property-based tests for statistical invariants

### 6. Performance Tests (`performance.rs`)
End-to-end performance and memory efficiency tests:
- Neural training pipeline performance
- FFT signal processing performance
- Sparse linear algebra performance
- Image processing performance
- Statistical analysis performance
- Memory efficiency validation
- Zero-copy transfer verification
- Parallel processing efficiency
- GPU/CPU handoff testing (when available)
- Cache efficiency
- Performance scaling analysis

## Running Tests

### Run all integration tests
```bash
cargo test --workspace --test integration
```

### Run specific test category
```bash
# Neural + Optimize tests
cargo test --test integration neural_optimize

# FFT + Signal tests
cargo test --test integration fft_signal

# Performance tests
cargo test --test integration performance
```

### Run property-based tests
```bash
cargo test --test integration prop_
```

### Run performance benchmarks (ignored by default)
```bash
cargo test --test integration --ignored
```

### Run with verbose output
```bash
cargo test --test integration -- --nocapture
```

## Test Policies

### No-Unwrap Policy
All tests follow the no-unwrap policy:
- Use `expect()` with descriptive messages
- Use proper `Result` types
- Handle errors gracefully in tests

### Property-Based Testing
Tests use `proptest` for property-based testing where applicable:
- Verify mathematical properties (commutativity, associativity, etc.)
- Test invariants across module boundaries
- Validate edge cases automatically

### Memory Efficiency
Memory efficiency is validated using:
- `assert_memory_efficient()` helper
- Zero-copy transfer verification
- Memory usage profiling

### Performance Targets
Performance tests include targets to catch regressions:
- Neural training: < 10s for 1000 samples
- FFT pipeline: < 100ms for 65536 samples
- Image processing: < 500ms for 2048x2048 images

## Test Data

### Fixtures (`fixtures/mod.rs`)
Standard test datasets:
- XOR dataset (4 samples, 2 features)
- Linear regression dataset (configurable size)
- Sinusoidal signals (for FFT/signal tests)
- Sparse matrices (with configurable density)
- Test images (gradients, patterns)
- Normal distribution samples

### Test Utilities (`common/test_utils.rs`)
Helper functions:
- `create_test_array_1d()` - Create deterministic 1D arrays
- `create_test_array_2d()` - Create deterministic 2D arrays
- `measure_time()` - Measure execution time
- `arrays_approx_equal()` - Compare arrays with tolerance
- `create_synthetic_classification_data()` - Generate labeled datasets
- `assert_memory_efficient()` - Validate memory usage
- Property test generators (dimensions, tolerances, etc.)

## Current Status

### Implemented
- ✅ Test infrastructure and utilities
- ✅ Test fixtures and data generators
- ✅ Neural + Optimize test structure (TODO: API-dependent implementations)
- ✅ FFT + Signal test structure
- ✅ Sparse + Linalg test structure
- ✅ NDImage + Vision test structure
- ✅ Stats + Datasets test structure
- ✅ Performance test structure
- ✅ Property-based test framework

### TODO (Awaiting API Stability)
Many test implementations are marked with `TODO` comments. These need to be
filled in once the module APIs stabilize. The test structure and patterns
are in place, ready for implementation.

Specific areas:
- Actual neural network training loops (depends on scirs2-neural API)
- Optimizer integration (depends on scirs2-optimize API)
- Some advanced signal processing features
- Some computer vision algorithms
- Statistical test implementations

## Integration Patterns

### Data Flow Testing
Tests verify that data can flow seamlessly between modules:
1. Load data from `scirs2-datasets`
2. Preprocess with `scirs2-stats`
3. Train model with `scirs2-neural` and `scirs2-optimize`
4. Evaluate performance

### Zero-Copy Verification
Tests check that data is not unnecessarily copied:
- Pointer comparison
- Memory usage monitoring
- View/reference usage validation

### Error Propagation
Tests verify error handling across boundaries:
- Error types are compatible
- Error messages are informative
- Errors propagate correctly

### Type Compatibility
Tests verify type compatibility:
- Arrays can be passed between modules
- Numeric types are consistent
- No precision loss in conversions

## Contributing

When adding new integration tests:
1. Follow the no-unwrap policy
2. Use property-based testing where applicable
3. Add performance targets for critical paths
4. Document test objectives and expected behavior
5. Use fixtures from `TestDatasets` when possible
6. Add memory efficiency checks for large operations

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)
