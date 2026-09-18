# SciRS2 Datasets Tutorials

Welcome to the comprehensive tutorial collection for SciRS2 datasets! These tutorials provide in-depth coverage of all aspects of the datasets module, from basic usage to advanced performance optimization.

## Tutorial Overview

### 🚀 [Getting Started](01_getting_started.md)
**Perfect for beginners**
- Introduction to SciRS2 datasets
- Loading toy datasets (Iris, Boston, Digits, etc.)
- Basic dataset operations and exploration
- Understanding the Dataset structure
- Error handling and common patterns

**Topics covered:**
- Dataset loading fundamentals
- Exploring dataset properties
- Basic data access patterns
- Error handling strategies

### 🔬 [Data Generation](02_data_generation.md)
**Create synthetic datasets for testing and development**
- Classification datasets (linear and non-linear)
- Regression datasets with configurable noise
- Clustering data (blobs, hierarchical patterns)
- Non-linear patterns (spirals, moons, swiss roll)
- Time series generation with trends and seasonality
- Data corruption utilities (missing values, outliers)

**Topics covered:**
- Synthetic data generation
- Configurable dataset parameters
- Non-linear pattern creation
- Time series synthesis
- Data corruption simulation

### ✂️ [Cross-Validation](03_cross_validation.md)
**Master model evaluation and validation strategies**
- K-Fold and Stratified K-Fold cross-validation
- Time series cross-validation (respecting temporal order)
- Group-based cross-validation
- Leave-One-Out cross-validation
- Nested cross-validation for hyperparameter tuning
- Custom validation strategies

**Topics covered:**
- Standard cross-validation techniques
- Time-aware validation for sequential data
- Advanced validation strategies
- Performance evaluation best practices

### 🛠️ [Dataset Utilities](04_dataset_utilities.md)
**Comprehensive data preprocessing and manipulation**
- Data scaling and normalization (Standard, Min-Max, Robust)
- Missing value detection and imputation
- Feature engineering (polynomial features, binning)
- Sampling strategies (random, stratified, bootstrap, SMOTE)
- Data balancing for imbalanced datasets
- Statistical analysis and outlier detection

**Topics covered:**
- Data preprocessing pipelines
- Feature engineering techniques
- Sampling and balancing strategies
- Data quality assessment

### 📁 [Custom Datasets](05_custom_datasets.md)
**Load and work with your own data**
- CSV, JSON, ARFF, and LIBSVM format support
- Custom format implementation
- Database integration (SQL, NoSQL)
- Streaming for large datasets
- Data validation and quality checks
- Performance optimization for custom loaders

**Topics covered:**
- Multi-format data loading
- Custom loader implementation
- Database integration
- Large dataset handling
- Data validation strategies

### ⚡ [Performance Optimization](06_performance.md)
**Achieve maximum performance with large datasets**
- Memory management and optimization
- Parallel processing and multi-threading
- SIMD operations for numerical computations
- Intelligent caching strategies
- GPU acceleration (CUDA support)
- Streaming for memory-efficient processing
- Performance profiling and monitoring

**Topics covered:**
- Memory optimization techniques
- Parallel and GPU acceleration
- Advanced caching strategies
- Performance monitoring and profiling

## Quick Start Guide

### 1. Basic Dataset Loading
```rust
use scirs2_datasets::{load_iris, load_boston};

// Load classic datasets
let iris = load_iris()?;
let boston = load_boston()?;

println!("Iris: {} samples, {} features", iris.n_samples(), iris.n_features());
println!("Boston: {} samples, {} features", boston.n_samples(), boston.n_features());
```

### 2. Generate Synthetic Data
```rust
use scirs2_datasets::{make_classification, make_regression};

// Generate classification dataset
let classification = make_classification(1000, 20, 5, 2, 15, Some(42))?;

// Generate regression dataset  
let regression = make_regression(500, 10, 5, 0.1, Some(42))?;
```

### 3. Cross-Validation
```rust
use scirs2_datasets::{load_wine, stratified_k_fold_split};

let wine = load_wine()?;
if let Some(target) = &wine.target {
    let folds = stratified_k_fold_split(target, 5, true, Some(42))?;
    println!("Created {} folds for cross-validation", folds.len());
}
```

### 4. Data Preprocessing
```rust
use scirs2_datasets::{load_digits, utils::{StandardScaler, train_test_split}};

let digits = load_digits()?;

// Split and scale data
let (train, test) = train_test_split(&digits, 0.2, Some(42))?;

let mut scaler = StandardScaler::new();
let mut train_data = train.data.clone();
scaler.fit(&train_data)?;
scaler.transform(&mut train_data)?;
```

## Learning Path Recommendations

### For Machine Learning Beginners
1. Start with **Getting Started** to understand the basics
2. Learn **Data Generation** to create test datasets
3. Master **Cross-Validation** for proper model evaluation
4. Explore **Dataset Utilities** for data preprocessing

### For Data Scientists
1. Begin with **Getting Started** for API familiarization
2. Study **Cross-Validation** for advanced validation strategies
3. Dive into **Dataset Utilities** for comprehensive preprocessing
4. Use **Custom Datasets** to integrate your own data
5. Apply **Performance Optimization** for large-scale workflows

### For ML Engineers and Researchers
1. Review **Getting Started** for quick API reference
2. Focus on **Performance Optimization** for production systems
3. Study **Custom Datasets** for flexible data integration
4. Use **Data Generation** for testing and benchmarking
5. Master **Cross-Validation** for rigorous evaluation

## Advanced Topics

### Production Deployment
- Memory-efficient streaming for large datasets
- GPU acceleration for massive computational workloads
- Intelligent caching for improved performance
- Parallel processing for multi-core systems

### Research and Development
- Custom data generator implementation
- Advanced cross-validation strategies
- Statistical analysis and data quality assessment
- Performance profiling and optimization

### Integration Patterns
- Database connectivity (SQL and NoSQL)
- Custom file format support
- Streaming data processing
- Multi-format data loading pipelines

## Code Examples Repository

Each tutorial includes comprehensive, runnable examples. You can find additional examples in:

- `examples/` directory - Complete example programs
- `src/` code documentation - Inline examples in documentation
- Tutorial code blocks - All code is tested and verified

## Performance Benchmarks

The tutorials include performance benchmarks comparing SciRS2 with:
- scikit-learn (Python)
- Native Rust implementations
- GPU-accelerated versions
- Memory-optimized versions

Run benchmarks with:
```bash
cargo run --example scikit_learn_benchmark --release
```

## Contributing to Tutorials

We welcome contributions to improve these tutorials:

1. **Error corrections** - Fix any inaccuracies or typos
2. **Additional examples** - Provide more use cases
3. **Performance tips** - Share optimization techniques
4. **Best practices** - Document effective patterns

## Getting Help

- **Documentation**: Comprehensive API documentation
- **Examples**: Practical examples for common use cases
- **Community**: GitHub Discussions for questions and sharing
- **Issues**: GitHub Issues for bug reports and feature requests

## What's Next?

After completing these tutorials, you'll be equipped to:

- Load and manipulate datasets efficiently
- Generate synthetic data for testing and development
- Implement robust cross-validation strategies
- Optimize performance for large-scale data processing
- Integrate custom data sources and formats
- Build production-ready data pipelines

The SciRS2 datasets module provides a solid foundation for all your scientific computing and machine learning data needs!

---

*These tutorials are part of the SciRS2 scientific computing ecosystem. For more information about other SciRS2 modules, visit the main documentation.*