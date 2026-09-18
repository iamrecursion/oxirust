# SciRS2-Core API Reference

## Table of Contents

1. [Core Modules](#core-modules)
2. [Array Operations](#array-operations)
3. [GPU Computing](#gpu-computing)
4. [Memory Management](#memory-management)
5. [Parallel Operations](#parallel-operations)
6. [Mathematical Constants](#mathematical-constants)
7. [Error Handling](#error-handling)
8. [Validation](#validation)
9. [Testing Infrastructure](#testing-infrastructure)
10. [Performance Optimization](#performance-optimization)

## Core Modules

### `scirs2_core::array_protocol`

The array protocol module provides a unified interface for working with different array types and backends.

#### Core Types

##### `ArrayProtocol`
```rust
pub trait ArrayProtocol<T> {
    type Output;
    fn shape(&self) -> &[usize];
    fn ndim(&self) -> usize;
    fn len(&self) -> usize;
}
```

**Purpose**: Universal trait for array-like objects providing shape information and basic properties.

**Parameters**:
- `T`: Element type of the array

**Methods**:
- `shape()`: Returns the dimensions of the array
- `ndim()`: Returns the number of dimensions
- `len()`: Returns the total number of elements

**Example**:
```rust
use scirs2_core::array_protocol::ArrayProtocol;
use ndarray::Array2;

let matrix = Array2::<f64>::zeros((100, 50));
assert_eq!(matrix.shape(), &[100, 50]);
assert_eq!(matrix.ndim(), 2);
assert_eq!(matrix.len(), 5000);
```

### `scirs2_core::error`

Comprehensive error handling system with location tracking and error chaining.

#### Core Types

##### `CoreError`
```rust
pub enum CoreError {
    ComputationError(ErrorContext),
    DomainError(ErrorContext),
    DimensionError(ErrorContext),
    ValueError(ErrorContext),
    // ... and many more
}
```

**Purpose**: Comprehensive error enumeration covering all scientific computing error scenarios.

**Variants**:
- `ComputationError`: Generic computation failures
- `DomainError`: Input outside valid mathematical domain
- `DimensionError`: Array dimension mismatches
- `ValueError`: Invalid parameter values
- `MemoryError`: Memory allocation failures
- `GpuError`: GPU computation errors

##### `ErrorContext`
```rust
pub struct ErrorContext {
    pub message: String,
    pub location: Option<ErrorLocation>,
    pub cause: Option<Box<CoreError>>,
}
```

**Purpose**: Rich error context with location information and error chaining.

**Fields**:
- `message`: Human-readable error description
- `location`: File, line, and function where error occurred
- `cause`: Optional underlying error cause

**Example**:
```rust
use scirs2_core::error::{CoreError, ErrorContext, ErrorLocation};

let error = CoreError::ValueError(
    ErrorContext::new("Invalid matrix dimensions")
        .with_location(ErrorLocation::new(file!(), line!()))
);
```

#### Convenience Macros

##### `error_context!`
```rust
error_context!("Error message")
error_context!("Error message", "function_name")
```

**Purpose**: Create error contexts with automatic location tracking.

##### `domainerror!`, `valueerror!`, `computationerror!`
```rust
domainerror!("Value outside valid domain")
valueerror!("Invalid parameter value")
computationerror!("Computation failed")
```

**Purpose**: Quickly create specific error types with location information.

## Array Operations

### `scirs2_core::simd_ops`

High-performance SIMD operations for scientific computing.

#### Core Traits

##### `SimdUnifiedOps`
```rust
pub trait SimdUnifiedOps<T> {
    fn simd_add(a: &ArrayView1<T>, b: &ArrayView1<T>) -> Array1<T>;
    fn simd_mul(a: &ArrayView1<T>, b: &ArrayView1<T>) -> Array1<T>;
    fn simd_dot(a: &ArrayView1<T>, b: &ArrayView1<T>) -> T;
    // ... more operations
}
```

**Purpose**: Unified interface for SIMD-accelerated array operations.

**Methods**:
- `simd_add()`: Element-wise addition with SIMD acceleration
- `simd_mul()`: Element-wise multiplication with SIMD acceleration
- `simd_dot()`: Dot product computation
- `simd_sum()`: Sum all elements
- `simd_norm()`: L2 norm calculation

**Performance**: Up to 14x faster than scalar operations on compatible hardware.

**Example**:
```rust
use scirs2_core::simd_ops::SimdUnifiedOps;
use ndarray::Array1;

let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

let result = f64::simd_add(&a.view(), &b.view());
// Result: [6.0, 8.0, 10.0, 12.0]
```

#### Ultra-Optimized Functions

##### `simd_mul_f32_hyperoptimized`
```rust
pub fn simd_mul_f32_hyperoptimized(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>
) -> Array1<f32>
```

**Purpose**: Adaptive SIMD multiplication with automatic strategy selection based on data size.

**Strategies**:
- Small arrays (<256): Lightweight SIMD
- Medium arrays (256-4K): Pipelined operations
- Large arrays (4K-64K): Cache-line optimization
- Very large arrays (64K-512K): Branch-free processing
- Huge arrays (>512K): TLB-optimized access

**Performance**: Up to 14.17x faster than scalar baseline.

### `scirs2_core::parallel_ops`

Advanced parallel processing operations with intelligent work distribution.

#### Core Functions

##### `parallel_map`
```rust
pub fn parallel_map<T, U, F>(items: &[T], op: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync,
```

**Purpose**: Parallel map operation with automatic chunking.

**Parameters**:
- `items`: Input slice to process
- `op`: Function to apply to each element

**Returns**: Vector of transformed results

##### `parallel_reduce`
```rust
pub fn parallel_reduce<T, F>(items: &[T], identity: T, op: F) -> T
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Sync,
```

**Purpose**: Parallel reduction with tree-based aggregation.

**Example**:
```rust
use scirs2_core::parallel_ops::parallel_reduce;

let data = vec![1, 2, 3, 4, 5];
let sum = parallel_reduce(&data, 0, |a, b| a + b);
assert_eq!(sum, 15);
```

##### `parallel_scan`
```rust
pub fn parallel_scan<T, F>(items: &[T], init: T, op: F) -> Vec<T>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> T + Sync,
```

**Purpose**: Parallel prefix scan (cumulative operation).

##### `parallel_matrix_rows`
```rust
pub fn parallel_matrix_rows<T, U, F>(matrix: &[&[T]], op: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&[T]) -> U + Sync,
```

**Purpose**: Process matrix rows in parallel with load balancing.

## GPU Computing

### `scirs2_core::gpu`

Multi-backend GPU acceleration support for scientific computing.

#### Backend Support

- **CUDA**: NVIDIA GPU acceleration
- **ROCm**: AMD GPU acceleration
- **Metal**: Apple Silicon and macOS GPU acceleration
- **WGPU**: Cross-platform WebGPU support
- **OpenCL**: Universal GPU compute support

#### Core Types

##### `GpuKernel`
```rust
pub trait GpuKernel {
    fn name(&self) -> &str;
    fn launch(&self, context: &GpuContext, args: &[GpuBuffer]) -> Result<(), GpuError>;
    fn required_features(&self) -> &[GpuFeature];
}
```

**Purpose**: Abstract interface for GPU compute kernels.

##### `GpuContext`
```rust
pub struct GpuContext {
    pub backend: GpuBackend,
    pub device_id: usize,
    pub memory_pool: GpuMemoryPool,
}
```

**Purpose**: GPU execution context with device management and memory pooling.

#### Kernel Categories

##### BLAS Kernels
- `GemvKernel`: General matrix-vector multiplication
- `GemmKernel`: General matrix-matrix multiplication
- `DotKernel`: Vector dot product
- `AxpyKernel`: Vector scaling and addition

##### Elementwise Kernels
- `ElementwiseAddKernel`: Element-wise addition
- `ElementwiseMulKernel`: Element-wise multiplication
- `ElementwiseSinKernel`: Element-wise sine function
- `ElementwiseExpKernel`: Element-wise exponential function

##### ML Kernels
- `ActivationKernel`: Neural network activation functions
- `ConvolutionKernel`: Convolution operations
- `BatchNormKernel`: Batch normalization

**Example**:
```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, ElementwiseAddKernel};

let context = GpuContext::new(GpuBackend::CUDA, 0)?;
let kernel = ElementwiseAddKernel::new();

let result = kernel.launch(&context, &[buffer_a, buffer_b, buffer_result])?;
```

## Memory Management

### `scirs2_core::memory`

Advanced memory management with multiple allocation strategies and performance optimization.

#### Core Types

##### `AdvancedBufferPool<T>`
```rust
pub struct AdvancedBufferPool<T: Clone + Default> {
    // Internal fields
}
```

**Purpose**: High-performance memory pool with size classes and strategy selection.

**Allocation Strategies**:
- `Pool`: Size-class based pooling
- `Arena`: Batch allocation with reset capability
- `NumaAware`: NUMA topology optimization
- `CacheAligned`: Cache-line aligned allocation
- `HugePage`: Large page allocation for big datasets

**Methods**:
```rust
impl<T: Clone + Default> AdvancedBufferPool<T> {
    pub fn new() -> Self;
    pub fn with_config(config: MemoryConfig) -> Self;
    pub fn acquire_vec_advanced(&mut self, capacity: usize) -> Vec<T>;
    pub fn release_vec_advanced(&mut self, vec: Vec<T>);
    pub fn memory_pressure(&self) -> MemoryPressure;
    pub fn get_statistics(&self) -> PoolStatistics;
    pub fn memory_report(&self) -> MemoryReport;
}
```

##### `MemoryConfig`
```rust
pub struct MemoryConfig {
    pub strategy: AllocationStrategy,
    pub access_pattern: AccessPattern,
    pub enable_prefetch: bool,
    pub alignment: usize,
    pub numa_aware: bool,
    pub max_memory: Option<usize>,
}
```

**Purpose**: Configuration for memory allocation behavior.

**Access Patterns**:
- `Sequential`: Linear memory access
- `Random`: Random access patterns
- `Temporal`: Time-based locality
- `Spatial`: Space-based locality
- `Streaming`: One-pass streaming access

##### `SmartAllocator`
```rust
pub struct SmartAllocator {
    // Internal adaptive logic
}
```

**Purpose**: Adaptive allocator that learns from usage patterns and optimizes strategy selection.

**Features**:
- Pattern recognition and adaptation
- Performance metric tracking
- Automatic strategy optimization
- Usage history analysis

**Example**:
```rust
use scirs2_core::memory::{AdvancedBufferPool, MemoryConfig, AllocationStrategy};

let config = MemoryConfig {
    strategy: AllocationStrategy::CacheAligned,
    alignment: 64,
    numa_aware: true,
    ..Default::default()
};

let mut pool = AdvancedBufferPool::<f64>::with_config(config);
let buffer = pool.acquire_vec_advanced(1000);

// Use buffer for computation
pool.release_vec_advanced(buffer);

let report = pool.memory_report();
println!("Pool efficiency: {:.2}%", report.pool_efficiency * 100.0);
```

#### Convenience Functions

```rust
pub fn create_optimized_pool<T: Clone + Default + 'static>() -> AdvancedBufferPool<T>;
pub fn create_scientific_pool<T: Clone + Default + 'static>() -> AdvancedBufferPool<T>;
pub fn create_large_data_pool<T: Clone + Default + 'static>() -> AdvancedBufferPool<T>;
```

## Mathematical Constants

### `scirs2_core::constants`

Comprehensive collection of mathematical, physical, and scientific constants.

#### Mathematical Constants

##### Basic Constants
```rust
pub mod math {
    pub const PI: f64 = std::f64::consts::PI;
    pub const E: f64 = std::f64::consts::E;
    pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
    pub const EULER: f64 = 0.577_215_664_901_532_9;
}
```

##### Special Mathematical Constants
```rust
pub const CATALAN: f64 = 0.915_965_594_177_219;         // Catalan's constant
pub const APERY: f64 = 1.202_056_903_159_594;           // Apéry's constant ζ(3)
pub const FEIGENBAUM_DELTA: f64 = 4.669_201_609_102_990; // Feigenbaum constant δ
pub const PLASTIC: f64 = 1.324_717_957_244_746;         // Plastic number
```

#### Physical Constants

##### Fundamental Constants
```rust
pub mod physical {
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;      // m/s
    pub const PLANCK: f64 = 6.626_070_15e-34;           // J·s
    pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19; // C
    pub const BOLTZMANN: f64 = 1.380_649e-23;           // J/K
}
```

##### Astrophysical Constants
```rust
pub const SOLAR_MASS: f64 = 1.988_47e30;               // kg
pub const EARTH_MASS: f64 = 5.972_168e24;              // kg
pub const HUBBLE_CONSTANT: f64 = 70.0;                 // km/s/Mpc
```

#### Numerical Analysis Constants

```rust
pub mod numerical {
    pub const MACHINE_EPSILON_F64: f64 = f64::EPSILON;
    pub const DEFAULT_TOLERANCE: f64 = 1e-12;
    pub const STRICT_TOLERANCE: f64 = 1e-15;
    pub const DEFAULT_MAX_ITERATIONS: usize = 1000;
}
```

#### Complex Number Constants

```rust
pub mod complex {
    pub const I: Complex64 = Complex64::new(0.0, 1.0);
    pub const E_TO_I_PI: Complex64 = NEG_ONE;           // Euler's identity
    pub const SQRT_I: Complex64 = Complex64::new(0.707_106_781_186_547, 0.707_106_781_186_547);
}
```

#### Usage Examples

```rust
use scirs2_core::constants::{math, physical, numerical};

// Mathematical computations
let circle_area = math::PI * radius.powi(2);
let exponential_growth = math::E.powf(growth_rate * time);

// Physical calculations
let energy = physical::PLANCK * frequency;
let thermal_energy = physical::BOLTZMANN * temperature;

// Numerical algorithms
let tolerance = numerical::DEFAULT_TOLERANCE;
let max_iterations = numerical::DEFAULT_MAX_ITERATIONS;
```

## Performance Optimization

### `scirs2_core::chunking`

Advanced chunking strategies for optimal parallel performance across different workload types.

#### Core Types

##### `ChunkConfig`
```rust
pub struct ChunkConfig {
    pub strategy: ChunkStrategy,
    pub memory_pattern: MemoryPattern,
    pub compute_intensity: ComputeIntensity,
    pub cache_awareness: CacheAwareness,
    pub numa_strategy: NumaStrategy,
}
```

**Purpose**: Comprehensive configuration for chunking behavior optimization.

##### `ChunkStrategy`
```rust
pub enum ChunkStrategy {
    Fixed(usize),
    Adaptive,
    CacheOptimized,
    MemoryOptimized,
    WorkStealingBalanced,
    NumaAware,
    LinearAlgebra,        // Optimized for matrix operations
    SparseMatrix,         // Optimized for sparse data
    SignalProcessing,     // Optimized for FFT-friendly sizes
    ImageProcessing,      // Optimized for 2D block processing
    MonteCarlo,           // Optimized for independent sampling
    GpuAware,            // Optimized for hybrid CPU/GPU workloads
}
```

#### Specialized Configurations

##### Scientific Computing Configurations
```rust
impl ChunkConfig {
    pub fn linear_algebra() -> Self;      // Matrix operations
    pub fn sparse_matrix() -> Self;       // Sparse data structures
    pub fn signal_processing() -> Self;   // FFT and signal analysis
    pub fn image_processing() -> Self;    // 2D image operations
    pub fn monte_carlo() -> Self;         // Random sampling
    pub fn iterative_solver() -> Self;   // Convergence algorithms
    pub fn gpu_hybrid() -> Self;          // CPU/GPU workloads
}
```

##### Usage Examples
```rust
use scirs2_core::chunking::{ChunkConfig, ChunkingUtils};

// Matrix multiplication chunking
let config = ChunkConfig::linear_algebra();
let optimal_size = ChunkingUtils::optimal_chunk_size(matrix_size, &config);

// Monte Carlo simulation chunking
let config = ChunkConfig::monte_carlo().with_monitoring();
let results = ChunkingUtils::chunked_map(&samples, &config, |sample| {
    monte_carlo_step(sample)
});

// GPU-aware chunking
let config = ChunkConfig::gpu_hybrid();
let chunks = ChunkingUtils::optimal_chunk_size(data_size, &config);
```

#### Performance Monitoring

##### `ChunkPerformanceMonitor`
```rust
pub struct ChunkPerformanceMonitor {
    // Adaptive performance tracking
}

impl ChunkPerformanceMonitor {
    pub fn record_measurement(&mut self, measurement: ChunkMeasurement);
    pub fn get_optimal_size(&self, operation_type: &str, data_size: usize) -> Option<usize>;
    pub fn get_statistics(&self) -> ChunkStatistics;
}
```

**Purpose**: Dynamic performance monitoring with automatic optimization discovery.

### Matrix-Specific Utilities

##### `MatrixChunking`
```rust
impl MatrixChunking {
    pub fn matrix_multiply_chunks(rows_a: usize, cols_a: usize, cols_b: usize)
        -> (usize, usize, usize);

    pub fn array_2d_chunks(rows: usize, cols: usize, thread_count: usize)
        -> (usize, usize);

    pub fn array_3d_chunks(depth: usize, rows: usize, cols: usize, thread_count: usize)
        -> (usize, usize, usize);
}
```

**Purpose**: Cache-oblivious chunking algorithms for multi-dimensional arrays.

## Testing Infrastructure

### `scirs2_core::testing`

Comprehensive testing framework designed for scientific computing applications.

#### Numerical Assertions

##### `NumericAssertion` Trait
```rust
pub trait NumericAssertion<T> {
    fn assert_approx_eq(&self, other: &T, tolerance: f64);
    fn assert_relative_eq(&self, other: &T, relative_tolerance: f64);
    fn assert_finite(&self);
    fn assert_in_range(&self, min: T, max: T);
}
```

**Purpose**: Robust numerical testing with configurable tolerances for floating-point comparisons.

**Implementations**:
- `f32`, `f64`: Scalar floating-point numbers
- `Array1<f64>`, `Array2<f64>`, etc.: Multi-dimensional arrays
- Complex numbers and custom numeric types

**Example**:
```rust
use scirs2_core::testing::NumericAssertion;

let result = 0.1 + 0.2;
result.assert_approx_eq(&0.3, 1e-10);

let matrix_a = compute_result();
let matrix_b = expected_result();
matrix_a.assert_relative_eq(&matrix_b, 1e-12);
```

#### Test Data Generation

##### `TestDataGenerator`
```rust
pub struct TestDataGenerator {
    // Reproducible random generation
}

impl TestDataGenerator {
    pub fn new() -> Self;
    pub fn with_seed(seed: u64) -> Self;

    // Basic random data
    pub fn random_float(&mut self, min: f64, max: f64) -> f64;
    pub fn random_array1(&mut self, size: usize, min: f64, max: f64) -> Array1<f64>;
    pub fn random_matrix(&mut self, shape: (usize, usize), min: f64, max: f64) -> Array2<f64>;

    // Specialized mathematical objects
    pub fn positive_definite_matrix(&mut self, size: usize) -> Array2<f64>;
    pub fn symmetric_matrix(&mut self, size: usize, min: f64, max: f64) -> Array2<f64>;
    pub fn orthogonal_matrix(&mut self, size: usize) -> Array2<f64>;
    pub fn sparse_matrix(&mut self, shape: (usize, usize), sparsity: f64, min: f64, max: f64) -> Array2<f64>;

    // Statistical distributions
    pub fn normal_distribution(&mut self, size: usize, mean: f64, std_dev: f64) -> Array1<f64>;
    pub fn time_series(&mut self, length: usize, trend: f64, noise_level: f64) -> Array1<f64>;
}
```

**Purpose**: Generate reproducible test data for scientific computing scenarios.

#### Property-Based Testing

##### `PropertyTest`
```rust
pub struct PropertyTest {
    // Configuration and execution
}

impl PropertyTest {
    pub fn new() -> Self;
    pub fn with_iterations(self, iterations: usize) -> Self;
    pub fn with_shrinking(self, enabled: bool) -> Self;

    pub fn test_property<F>(&self, name: &str, property: F)
    where F: FnMut(&mut TestDataGenerator);
}
```

**Purpose**: Test mathematical properties with automatic test case generation and shrinking.

**Example**:
```rust
use scirs2_core::testing::PropertyTest;

PropertyTest::new()
    .with_iterations(1000)
    .test_property("matrix_multiplication_associative", |gen| {
        let a = gen.random_matrix((10, 10), -1.0, 1.0);
        let b = gen.random_matrix((10, 10), -1.0, 1.0);
        let c = gen.random_matrix((10, 10), -1.0, 1.0);

        let result1 = (a.dot(&b)).dot(&c);
        let result2 = a.dot(&(b.dot(&c)));

        result1.assert_approx_eq(&result2, 1e-10);
    });
```

#### Performance Benchmarking

##### `BenchmarkSuite`
```rust
pub struct BenchmarkSuite {
    // Benchmark management and execution
}

impl BenchmarkSuite {
    pub fn new() -> Self;
    pub fn with_regression_threshold(self, threshold: f64) -> Self;

    pub fn add_benchmark<F>(&mut self, name: &str, benchmark: F)
    where F: Fn() -> Duration + Send + Sync + 'static;

    pub fn set_baseline(&mut self, name: &str, baseline: Duration);
    pub fn run_all(&self) -> BenchmarkResults;
}
```

**Purpose**: Performance testing with regression detection and automatic warmup.

**Example**:
```rust
use scirs2_core::testing::BenchmarkSuite;
use std::time::Instant;

let mut suite = BenchmarkSuite::new()
    .with_regression_threshold(1.1); // 10% slowdown threshold

suite.add_benchmark("matrix_multiply", || {
    let start = Instant::now();
    let result = matrix_a.dot(&matrix_b);
    start.elapsed()
});

let results = suite.run_all();
results.print_results();
```

#### Test Organization

##### `TestRunner`
```rust
pub struct TestRunner {
    // Test execution management
}

impl TestRunner {
    pub fn new() -> Self;
    pub fn with_parallel_execution(self) -> Self;

    pub fn add_test<F>(&mut self, name: &str, test: F)
    where F: Fn() + Send + Sync + 'static;

    pub fn add_fixture(&mut self, name: &str, fixture: TestFixture);
    pub fn run_all(&self) -> TestResults;
}
```

**Purpose**: Organized test execution with fixtures, parallel execution, and comprehensive reporting.

## Validation

### `scirs2_core::validation`

Advanced validation framework with production-ready features and cross-platform support.

#### Production Validation

##### `ProductionValidator`
```rust
pub struct ProductionValidator {
    // Enterprise-grade validation
}

impl ProductionValidator {
    pub fn validate_with_context<T>(&self, value: &T, context: &ValidationContext) -> ValidationResult;
    pub fn batch_validate<T>(&self, items: &[T]) -> Vec<ValidationResult>;
    pub fn validate_schema<T>(&self, data: &T, schema: &ValidationSchema) -> ValidationResult;
}
```

##### `ValidationContext`
```rust
pub struct ValidationContext {
    pub operation_name: String,
    pub severity_level: ValidationSeverity,
    pub environment: ValidationEnvironment,
    pub timeout: Option<Duration>,
}
```

#### Cross-Platform Validation

##### `CrossPlatformValidator`
```rust
pub struct CrossPlatformValidator {
    // Platform-aware validation
}

impl CrossPlatformValidator {
    pub fn validate_file_path(&mut self, path: &str) -> ValidationResult;
    pub fn validate_numeric_cross_platform<T>(&mut self, value: T, fieldname: &str) -> ValidationResult;
    pub fn validate_simd_operation(&mut self, operation: &str, data_size: usize, vector_size: usize) -> ValidationResult;
    pub fn validate_memory_allocation(&mut self, size: usize, purpose: &str) -> ValidationResult;
}
```

**Features**:
- Windows path validation (reserved names, invalid characters)
- Unix path validation (system directories, length limits)
- WASM sandbox validation
- NUMA topology awareness
- SIMD capability checking

#### Data Validation

##### `DataValidator`
```rust
pub struct DataValidator {
    // Schema-based data validation
}

impl DataValidator {
    pub fn validate_with_schema<T>(&mut self, data: &T, schema_name: &str) -> ValidationResult;
    pub fn validate_array<S, D>(&mut self, array: &ArrayBase<S, D>, constraints: &ArrayConstraints) -> ValidationResult;
    pub fn register_schema(&mut self, schema: DataSchema);
}
```

##### `DataSchema`
```rust
pub struct DataSchema {
    pub name: String,
    pub data_type: DataType,
    pub constraints: Vec<Constraint>,
    pub required: bool,
}
```

#### Constraint Types

```rust
pub enum Constraint {
    // Range constraints
    MinValue(f64),
    MaxValue(f64),
    Range(f64, f64),

    // Array constraints
    UniqueElements,
    Sorted,
    Normalized,

    // Scientific constraints
    Symmetric,
    PositiveDefinite,
    Orthogonal,

    // Custom constraints
    Custom(String, Box<dyn CustomConstraint>),
}
```

**Example**:
```rust
use scirs2_core::validation::{DataValidator, SchemaBuilder, DataType, Constraint};

let schema = SchemaBuilder::new("probability")
    .with_type(DataType::Float64)
    .with_constraint(Constraint::Range(0.0, 1.0))
    .with_constraint(Constraint::Finite)
    .build()?;

let mut validator = DataValidator::new();
validator.register_schema(schema);

let probability = 0.75;
let result = validator.validate_with_schema(&probability, "probability");
assert!(result.is_valid);
```

## Integration Examples

### Complete Workflow Example

```rust
use scirs2_core::{
    simd_ops::SimdUnifiedOps,
    parallel_ops::parallel_map,
    memory::{create_scientific_pool, MemoryConfig},
    chunking::{ChunkConfig, ChunkingUtils},
    testing::{NumericAssertion, TestDataGenerator, PropertyTest},
    validation::{ProductionValidator, ValidationContext},
    constants::math,
};
use ndarray::{Array1, Array2};

fn scientific_computation_pipeline() -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    // 1. Memory Management Setup
    let mut memory_pool = create_scientific_pool::<f64>();

    // 2. Generate test data
    let mut generator = TestDataGenerator::with_seed(42);
    let matrix = generator.random_matrix((1000, 1000), -1.0, 1.0);

    // 3. Validation
    let validator = ProductionValidator::new();
    let context = ValidationContext::production();
    validator.validate_with_context(&matrix, &context)?;

    // 4. Optimized chunking for computation
    let chunk_config = ChunkConfig::linear_algebra();
    let optimal_chunk = ChunkingUtils::optimal_chunk_size(matrix.len(), &chunk_config);

    // 5. SIMD-accelerated computation
    let rows: Vec<Array1<f64>> = (0..matrix.nrows())
        .map(|i| matrix.row(i).to_owned())
        .collect();

    let results = parallel_map(&rows, |row| {
        // Apply SIMD-accelerated operations
        let scaled = f64::simd_mul(&row.view(), &Array1::from_elem(row.len(), math::PI).view());
        let normalized = {
            let norm = f64::simd_norm(&scaled.view());
            if norm > 0.0 {
                f64::simd_mul(&scaled.view(), &Array1::from_elem(scaled.len(), 1.0 / norm).view())
            } else {
                scaled
            }
        };
        normalized
    });

    // 6. Reconstruct result matrix
    let mut result_matrix = Array2::zeros((matrix.nrows(), matrix.ncols()));
    for (i, row) in results.into_iter().enumerate() {
        result_matrix.row_mut(i).assign(&row);
    }

    // 7. Validation of results
    result_matrix.assert_finite();

    // 8. Property-based testing
    PropertyTest::new()
        .with_iterations(100)
        .test_property("normalized_rows_have_unit_norm", |gen| {
            for i in 0..std::cmp::min(10, result_matrix.nrows()) {
                let row = result_matrix.row(i);
                let norm = f64::simd_norm(&row);
                norm.assert_approx_eq(&1.0, 1e-10);
            }
        });

    Ok(result_matrix)
}
```

This comprehensive API reference covers all major components of the SciRS2-Core library with detailed documentation, examples, and usage patterns for scientific computing applications.