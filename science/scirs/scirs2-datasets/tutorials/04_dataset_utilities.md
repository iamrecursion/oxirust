# Dataset Utilities Tutorial

This tutorial covers the comprehensive dataset manipulation and utility functions provided by SciRS2 datasets for preprocessing, transformation, and analysis.

## Overview

SciRS2 provides extensive utilities for:

- **Data Preprocessing**: Scaling, normalization, missing value handling
- **Feature Engineering**: Polynomial features, binning, encoding
- **Sampling Strategies**: Random, stratified, bootstrap, SMOTE
- **Data Balancing**: Over/under-sampling for imbalanced datasets
- **Statistical Analysis**: Descriptive statistics, correlations
- **Data Quality**: Outlier detection, duplicate handling

## Data Scaling and Normalization

### Standard Scaling (Z-score normalization)

```rust
use scirs2_datasets::{load_boston, utils::StandardScaler};

let boston = load_boston()?;
let mut data = boston.data.clone();

// Create and fit scaler
let mut scaler = StandardScaler::new();
scaler.fit(&data)?;

// Transform the data
scaler.transform(&mut data)?;

println!("Data standardized:");
println!("  Mean ≈ 0, Std ≈ 1 for each feature");

// Check first feature statistics
let first_feature = data.column(0);
let mean = first_feature.mean().unwrap();
let std = first_feature.std(0.0);
println!("  Feature 0: mean={:.3}, std={:.3}", mean, std);
```

### Min-Max Scaling

```rust
use scirs2_datasets::{load_iris, utils::MinMaxScaler};

let iris = load_iris()?;
let mut data = iris.data.clone();

// Scale to [0, 1] range
let mut scaler = MinMaxScaler::new(0.0, 1.0);
scaler.fit(&data)?;
scaler.transform(&mut data)?;

println!("Data scaled to [0, 1]:");
for i in 0..data.ncols() {
    let col = data.column(i);
    let min_val = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!("  Feature {}: [{:.3}, {:.3}]", i, min_val, max_val);
}
```

### Robust Scaling

```rust
use scirs2_datasets::{make_regression, utils::RobustScaler, utils::add_outliers};

// Create dataset with outliers
let mut dataset = make_regression(200, 5, 4, 0.1, Some(42))?;
add_outliers(&mut dataset.data, 0.1, 5.0, Some(42))?;

let mut data = dataset.data.clone();

// Robust scaling uses median and IQR instead of mean and std
let mut scaler = RobustScaler::new();
scaler.fit(&data)?;
scaler.transform(&mut data)?;

println!("Data robustly scaled (resistant to outliers)");
```

### Unit Vector Scaling

```rust
use scirs2_datasets::{load_digits, utils::normalize_l2};

let digits = load_digits()?;
let mut data = digits.data.clone();

// Normalize each sample to unit length (L2 norm = 1)
normalize_l2(&mut data)?;

println!("Each sample normalized to unit vector");

// Verify normalization
for i in 0..std::cmp::min(5, data.nrows()) {
    let row = data.row(i);
    let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
    println!("  Sample {}: L2 norm = {:.3}", i, norm);
}
```

## Missing Value Handling

### Detecting Missing Values

```rust
use scirs2_datasets::{make_classification, utils::add_missing_values, utils::detect_missing};

// Create dataset and add missing values
let mut dataset = make_classification(100, 10, 3, 2, 8, Some(42))?;
add_missing_values(&mut dataset.data, 0.15, Some(42))?; // 15% missing

// Detect missing values
let missing_info = detect_missing(&dataset.data)?;

println!("Missing value analysis:");
println!("  Total missing: {}", missing_info.total_missing);
println!("  Missing percentage: {:.1}%", missing_info.missing_percentage);
println!("  Rows with missing: {}", missing_info.rows_with_missing);
println!("  Cols with missing: {}", missing_info.cols_with_missing);
```

### Simple Imputation

```rust
use scirs2_datasets::{make_regression, utils::{add_missing_values, SimpleImputer, ImputationStrategy}};

let mut dataset = make_regression(150, 8, 6, 0.1, Some(42))?;
add_missing_values(&mut dataset.data, 0.1, Some(42))?;

let mut data = dataset.data.clone();

// Impute with mean
let mut imputer = SimpleImputer::new(ImputationStrategy::Mean);
imputer.fit(&data)?;
imputer.transform(&mut data)?;

println!("Missing values imputed with mean");
```

### Advanced Imputation

```rust
use scirs2_datasets::{load_boston, utils::{add_missing_values, KNNImputer}};

let mut dataset = load_boston()?;
add_missing_values(&mut dataset.data, 0.05, Some(42))?;

let mut data = dataset.data.clone();

// K-Nearest Neighbors imputation
let mut imputer = KNNImputer::new(5); // k=5 neighbors
imputer.fit(&data)?;
imputer.transform(&mut data)?;

println!("Missing values imputed using KNN (k=5)");
```

## Feature Engineering

### Polynomial Features

```rust
use scirs2_datasets::{make_regression, utils::PolynomialFeatures};

let dataset = make_regression(100, 3, 3, 0.1, Some(42))?;
let data = dataset.data.clone();

// Generate polynomial features up to degree 2
let mut poly = PolynomialFeatures::new(2, true); // degree=2, include_bias=true
let poly_data = poly.fit_transform(&data)?;

println!("Polynomial feature expansion:");
println!("  Original features: {}", data.ncols());
println!("  Polynomial features: {}", poly_data.ncols());
println!("  Feature names: {:?}", poly.get_feature_names(&["x0", "x1", "x2"]));
```

### Feature Binning

```rust
use scirs2_datasets::{load_boston, utils::{KBinsDiscretizer, BinningStrategy}};

let boston = load_boston()?;
let data = &boston.data;

// Discretize continuous features into bins
let mut discretizer = KBinsDiscretizer::new(
    5,                           // n_bins
    BinningStrategy::Quantile,   // binning strategy
    true                         // encode as onehot
);

let binned_data = discretizer.fit_transform(&data.column(0).to_owned().insert_axis(ndarray::Axis(1)))?;

println!("Feature binning:");
println!("  Original feature: continuous");
println!("  Binned features: {} bins", binned_data.ncols());
```

### Feature Selection

```rust
use scirs2_datasets::{make_classification, utils::{SelectKBest, mutual_info_score}};

let dataset = make_classification(200, 20, 3, 2, 10, Some(42))?;

if let Some(target) = &dataset.target {
    // Select top k features based on mutual information
    let mut selector = SelectKBest::new(10, mutual_info_score); // Select top 10 features
    let selected_data = selector.fit_transform(&dataset.data, target)?;
    
    println!("Feature selection:");
    println!("  Original features: {}", dataset.data.ncols());
    println!("  Selected features: {}", selected_data.ncols());
    println!("  Selected indices: {:?}", selector.get_selected_indices());
}
```

## Sampling Strategies

### Random Sampling

```rust
use scirs2_datasets::{load_digits, utils::random_sample};

let digits = load_digits()?;

// Random sample of 100 examples
let sample_indices = random_sample(digits.n_samples(), 100, Some(42))?;
let sampled_data = digits.data.select(ndarray::Axis(0), &sample_indices);

println!("Random sampling:");
println!("  Original: {} samples", digits.n_samples());
println!("  Sampled: {} samples", sampled_data.nrows());
```

### Stratified Sampling

```rust
use scirs2_datasets::{load_wine, utils::stratified_sample};

let wine = load_wine()?;

if let Some(target) = &wine.target {
    // Stratified sample maintaining class proportions
    let sample_indices = stratified_sample(target, 50, Some(42))?; // 50 total samples
    
    println!("Stratified sampling:");
    println!("  Original: {} samples", wine.n_samples());
    println!("  Sampled: {} samples", sample_indices.len());
    
    // Check class distribution preservation
    let mut original_dist = std::collections::HashMap::new();
    let mut sampled_dist = std::collections::HashMap::new();
    
    for &class in target.iter() {
        *original_dist.entry(class as i32).or_insert(0) += 1;
    }
    
    for &idx in &sample_indices {
        let class = target[idx] as i32;
        *sampled_dist.entry(class).or_insert(0) += 1;
    }
    
    println!("  Original distribution: {:?}", original_dist);
    println!("  Sampled distribution: {:?}", sampled_dist);
}
```

### Bootstrap Sampling

```rust
use scirs2_datasets::{load_iris, utils::bootstrap_sample};

let iris = load_iris()?;

// Bootstrap sample (sampling with replacement)
let bootstrap_indices = bootstrap_sample(iris.n_samples(), Some(42))?;

println!("Bootstrap sampling:");
println!("  Original: {} samples", iris.n_samples());
println!("  Bootstrap: {} samples", bootstrap_indices.len());

// Count unique samples in bootstrap
let unique_samples: std::collections::HashSet<_> = bootstrap_indices.iter().collect();
println!("  Unique samples in bootstrap: {}", unique_samples.len());
```

## Data Balancing

### Random Over/Under Sampling

```rust
use scirs2_datasets::{generators::ClassificationConfig, utils::{RandomOverSampler, RandomUnderSampler}};

// Create imbalanced dataset
let config = ClassificationConfig {
    n_samples: 1000,
    n_features: 10,
    n_classes: 3,
    weights: Some(vec![0.7, 0.2, 0.1]), // Imbalanced
    random_state: Some(42),
    ..Default::default()
};

let dataset = config.generate()?;

if let Some(target) = &dataset.target {
    // Check original distribution
    let mut original_dist = std::collections::HashMap::new();
    for &class in target.iter() {
        *original_dist.entry(class as i32).or_insert(0) += 1;
    }
    println!("Original distribution: {:?}", original_dist);
    
    // Random over-sampling
    let mut over_sampler = RandomOverSampler::new(Some(42));
    let (balanced_data, balanced_target) = over_sampler.fit_transform(&dataset.data, target)?;
    
    let mut balanced_dist = std::collections::HashMap::new();
    for &class in balanced_target.iter() {
        *balanced_dist.entry(class as i32).or_insert(0) += 1;
    }
    println!("After over-sampling: {:?}", balanced_dist);
}
```

### SMOTE (Synthetic Minority Oversampling)

```rust
use scirs2_datasets::{make_classification, utils::SMOTE};

let dataset = make_classification(500, 15, 3, 2, 10, Some(42))?;

if let Some(target) = &dataset.target {
    // SMOTE generates synthetic examples for minority classes
    let mut smote = SMOTE::new(
        5,     // k_neighbors
        0.5,   // sampling_strategy (ratio of minority to majority)
        Some(42)
    );
    
    let (synthetic_data, synthetic_target) = smote.fit_transform(&dataset.data, target)?;
    
    println!("SMOTE balancing:");
    println!("  Original: {} samples", dataset.n_samples());
    println!("  After SMOTE: {} samples", synthetic_data.nrows());
}
```

## Statistical Analysis

### Descriptive Statistics

```rust
use scirs2_datasets::{load_boston, utils::describe_dataset};

let boston = load_boston()?;

// Comprehensive statistical description
let stats = describe_dataset(&boston)?;

println!("Dataset Statistics:");
println!("  Samples: {}", stats.n_samples);
println!("  Features: {}", stats.n_features);
println!("  Missing values: {}", stats.missing_values);

for (i, feature_stats) in stats.feature_stats.iter().enumerate() {
    println!("  Feature {}: mean={:.2}, std={:.2}, min={:.2}, max={:.2}", 
             i, feature_stats.mean, feature_stats.std, 
             feature_stats.min, feature_stats.max);
}
```

### Correlation Analysis

```rust
use scirs2_datasets::{load_wine, utils::correlation_matrix};

let wine = load_wine()?;

// Calculate feature correlation matrix
let corr_matrix = correlation_matrix(&wine.data)?;

println!("Correlation analysis:");
println!("  Matrix shape: {:?}", corr_matrix.shape());

// Find highly correlated features (|correlation| > 0.8)
let mut high_corr = Vec::new();
for i in 0..corr_matrix.nrows() {
    for j in (i+1)..corr_matrix.ncols() {
        let corr = corr_matrix[[i, j]].abs();
        if corr > 0.8 {
            high_corr.push((i, j, corr));
        }
    }
}

println!("  Highly correlated pairs (|r| > 0.8):");
for (i, j, corr) in high_corr {
    println!("    Features {} and {}: r = {:.3}", i, j, corr);
}
```

### Outlier Detection

```rust
use scirs2_datasets::{load_boston, utils::{detect_outliers_iqr, detect_outliers_zscore}};

let boston = load_boston()?;

// IQR method (outliers beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR)
let iqr_outliers = detect_outliers_iqr(&boston.data, 1.5)?;

// Z-score method (outliers with |z-score| > threshold)
let zscore_outliers = detect_outliers_zscore(&boston.data, 3.0)?;

println!("Outlier detection:");
println!("  IQR method: {} outlier samples", iqr_outliers.len());
println!("  Z-score method: {} outlier samples", zscore_outliers.len());

// Intersection of both methods
let consensus_outliers: Vec<_> = iqr_outliers.iter()
    .filter(|&&idx| zscore_outliers.contains(&idx))
    .collect();
println!("  Consensus outliers: {}", consensus_outliers.len());
```

## Data Quality Assessment

### Duplicate Detection

```rust
use scirs2_datasets::{make_classification, utils::find_duplicates};

let mut dataset = make_classification(200, 10, 3, 2, 8, Some(42))?;

// Artificially add some duplicates
let duplicate_row = dataset.data.row(0).to_owned();
dataset.data.push_row(duplicate_row.view())?;

// Find duplicate rows
let duplicates = find_duplicates(&dataset.data, 1e-10)?; // tolerance for floating point comparison

println!("Duplicate detection:");
println!("  Found {} duplicate pairs", duplicates.len());

for (idx1, idx2) in duplicates {
    println!("    Rows {} and {} are duplicates", idx1, idx2);
}
```

### Data Validation

```rust
use scirs2_datasets::{load_digits, utils::{validate_dataset, ValidationReport}};

let digits = load_digits()?;

// Comprehensive data validation
let report = validate_dataset(&digits)?;

println!("Data Validation Report:");
println!("  Valid dataset: {}", report.is_valid);
println!("  Warnings: {}", report.warnings.len());
println!("  Errors: {}", report.errors.len());

for warning in &report.warnings {
    println!("  ⚠️  {}", warning);
}

for error in &report.errors {
    println!("  ❌ {}", error);
}
```

## Advanced Dataset Operations

### Dataset Concatenation

```rust
use scirs2_datasets::{make_classification, utils::concatenate_datasets};

let dataset1 = make_classification(100, 10, 3, 2, 8, Some(42))?;
let dataset2 = make_classification(150, 10, 3, 2, 8, Some(43))?;

let combined = concatenate_datasets(&[dataset1, dataset2])?;

println!("Dataset concatenation:");
println!("  Combined: {} samples", combined.n_samples());
```

### Dataset Filtering

```rust
use scirs2_datasets::{load_iris, utils::filter_dataset};

let iris = load_iris()?;

// Filter samples based on a condition (e.g., first feature > 5.0)
let condition = |row: ndarray::ArrayView1<f64>| row[0] > 5.0;
let filtered = filter_dataset(&iris, condition)?;

println!("Dataset filtering:");
println!("  Original: {} samples", iris.n_samples());
println!("  Filtered: {} samples", filtered.n_samples());
```

### Memory-Efficient Operations

```rust
use scirs2_datasets::{load_digits, utils::chunked_operation};

let digits = load_digits()?;

// Process data in chunks to save memory
let chunk_size = 100;
let mut chunk_means = Vec::new();

chunked_operation(&digits.data, chunk_size, |chunk| {
    let mean = chunk.mean_axis(ndarray::Axis(0)).unwrap();
    chunk_means.push(mean);
    Ok(())
})?;

println!("Chunked processing:");
println!("  Processed {} chunks of size {}", chunk_means.len(), chunk_size);
```

This tutorial covered the extensive dataset utility functions available in SciRS2. These tools provide comprehensive support for data preprocessing, feature engineering, quality assessment, and advanced manipulations needed in real-world machine learning workflows.