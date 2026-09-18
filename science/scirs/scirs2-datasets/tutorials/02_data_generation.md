# Data Generation Tutorial

This tutorial covers the comprehensive data generation capabilities of SciRS2 datasets, allowing you to create synthetic datasets for various machine learning tasks.

## Overview

SciRS2 provides powerful data generators for:

- **Classification**: Linear and non-linear classification problems
- **Regression**: Single and multi-output regression with noise
- **Clustering**: Blob-like and hierarchical clustering datasets
- **Non-linear patterns**: Spirals, moons, circles, swiss roll
- **Time series**: Synthetic time series with trends and seasonality
- **Corrupted data**: Datasets with missing values and outliers

## Classification Data Generation

### Basic Classification Dataset

```rust
use scirs2_datasets::make_classification;

// Generate a basic classification dataset
let dataset = make_classification(
    1000,      // n_samples: number of samples
    20,        // n_features: total number of features
    5,         // n_classes: number of target classes
    3,         // n_clusters_per_class: clusters per class
    10,        // n_informative: number of informative features
    Some(42),  // random_state: for reproducibility
)?;

println!("Classification dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());
println!("  Classes: 5");
```

### Advanced Classification with Custom Parameters

```rust
use scirs2_datasets::generators::ClassificationConfig;

// Advanced configuration
let config = ClassificationConfig {
    n_samples: 2000,
    n_features: 50,
    n_classes: 4,
    n_clusters_per_class: 2,
    n_informative: 20,
    n_redundant: 5,
    n_repeated: 3,
    class_sep: 1.5,        // Class separation (higher = easier)
    flip_y: 0.01,          // Label noise (1% of labels flipped)
    weights: Some(vec![0.4, 0.3, 0.2, 0.1]), // Class imbalance
    random_state: Some(42),
};

let dataset = config.generate()?;

// Check class distribution
if let Some(target) = &dataset.target {
    let mut class_counts = std::collections::HashMap::new();
    for &class in target.iter() {
        *class_counts.entry(class as i32).or_insert(0) += 1;
    }
    println!("Class distribution: {:?}", class_counts);
}
```

## Regression Data Generation

### Single-Output Regression

```rust
use scirs2_datasets::make_regression;

let dataset = make_regression(
    500,       // n_samples
    10,        // n_features
    5,         // n_informative: number of features that affect target
    0.1,       // noise: standard deviation of gaussian noise
    Some(42),  // random_state
)?;

println!("Regression dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());

// Target statistics
if let Some(target) = &dataset.target {
    let mean = target.mean().unwrap();
    let std = target.std(0.0);
    println!("  Target mean: {:.3}, std: {:.3}", mean, std);
}
```

### Multi-Output Regression

```rust
use scirs2_datasets::generators::RegressionConfig;

let config = RegressionConfig {
    n_samples: 1000,
    n_features: 15,
    n_targets: 3,          // Multiple output targets
    n_informative: 10,
    noise: 0.05,
    bias: 100.0,           // Bias term added to targets
    tail_strength: 0.5,    // Tail strength for heavy-tailed noise
    random_state: Some(42),
};

let dataset = config.generate()?;
println!("Multi-output regression: {} targets", 3);
```

## Clustering Data Generation

### Blob Clusters

```rust
use scirs2_datasets::make_blobs;

// Generate blob-like clusters
let dataset = make_blobs(
    800,       // n_samples
    2,         // n_features (2D for visualization)
    4,         // n_centers: number of clusters
    1.0,       // cluster_std: standard deviation of clusters
    Some(42),  // random_state
)?;

println!("Blob clusters:");
println!("  Samples: {}", dataset.n_samples());
println!("  Clusters: 4");
```

### Custom Cluster Centers

```rust
use scirs2_datasets::generators::BlobConfig;
use ndarray::Array2;

// Define custom cluster centers
let centers = Array2::from_shape_vec(
    (3, 2),
    vec![0.0, 0.0,    // Center 1
         5.0, 5.0,    // Center 2
         0.0, 5.0],   // Center 3
)?;

let config = BlobConfig {
    n_samples: 600,
    centers: Some(centers),
    cluster_std: vec![0.5, 1.0, 1.5], // Different std for each cluster
    random_state: Some(42),
};

let dataset = config.generate()?;
```

## Non-Linear Pattern Generation

### Two Moons

```rust
use scirs2_datasets::make_moons;

// Generate two interleaving half circles
let dataset = make_moons(
    400,       // n_samples
    0.1,       // noise level
    Some(42),  // random_state
)?;

println!("Two moons pattern: {} samples", dataset.n_samples());
```

### Concentric Circles

```rust
use scirs2_datasets::make_circles;

let dataset = make_circles(
    500,       // n_samples
    0.05,      // noise level
    0.6,       // factor: scale factor between inner and outer circle
    Some(42),
)?;

println!("Concentric circles: {} samples", dataset.n_samples());
```

### Spiral Patterns

```rust
use scirs2_datasets::make_spirals;

let dataset = make_spirals(
    600,       // n_samples
    2,         // n_spirals: number of spiral arms
    0.1,       // noise level
    Some(42),
)?;

println!("Spiral pattern: {} samples", dataset.n_samples());
```

### Swiss Roll (3D Manifold)

```rust
use scirs2_datasets::make_swiss_roll;

let dataset = make_swiss_roll(
    1000,      // n_samples
    0.1,       // noise level
    Some(42),
)?;

println!("Swiss roll: {} samples, {} features", 
         dataset.n_samples(), dataset.n_features());
assert_eq!(dataset.n_features(), 3); // 3D embedding
```

## Time Series Generation

### Basic Time Series

```rust
use scirs2_datasets::make_time_series;

let dataset = make_time_series(
    100,       // n_timesteps
    3,         // n_features/variables
    true,      // with_trend: include linear trend
    true,      // with_seasonality: include seasonal component
    0.1,       // noise_level
    Some(42),
)?;

println!("Time series:");
println!("  Timesteps: {}", dataset.n_samples());
println!("  Variables: {}", dataset.n_features());
```

### Advanced Time Series with Custom Patterns

```rust
use scirs2_datasets::generators::{TimeSeriesConfig, SeasonalPattern, TrendType};

let config = TimeSeriesConfig {
    n_timesteps: 365,  // One year of daily data
    n_features: 2,
    trend_type: TrendType::Polynomial(2), // Quadratic trend
    seasonal_patterns: vec![
        SeasonalPattern {
            period: 7,     // Weekly seasonality
            amplitude: 2.0,
        },
        SeasonalPattern {
            period: 30,    // Monthly seasonality  
            amplitude: 1.0,
        },
    ],
    noise_type: scirs2_datasets::generators::NoiseType::ARMA(1, 1),
    noise_level: 0.2,
    random_state: Some(42),
};

let dataset = config.generate()?;
```

## Data Corruption and Noise

### Adding Missing Values

```rust
use scirs2_datasets::{make_classification, utils::add_missing_values};

let mut dataset = make_classification(500, 10, 3, 2, 8, Some(42))?;

// Add 10% missing values randomly
add_missing_values(&mut dataset.data, 0.1, Some(42))?;

println!("Added missing values to dataset");
```

### Adding Outliers

```rust
use scirs2_datasets::{make_regression, utils::add_outliers};

let mut dataset = make_regression(300, 5, 4, 0.05, Some(42))?;

// Add 5% outliers
add_outliers(&mut dataset.data, 0.05, 3.0, Some(42))?; // 3.0 = outlier strength

println!("Added outliers to dataset");
```

## Combining Datasets

### Concatenating Datasets

```rust
use scirs2_datasets::{make_classification, utils::concatenate_datasets};

let dataset1 = make_classification(200, 10, 2, 1, 8, Some(42))?;
let dataset2 = make_classification(300, 10, 2, 1, 8, Some(43))?;

let combined = concatenate_datasets(&[dataset1, dataset2])?;
println!("Combined dataset: {} samples", combined.n_samples()); // 500 samples
```

### Feature Augmentation

```rust
use scirs2_datasets::{make_classification, utils::add_polynomial_features};

let mut dataset = make_classification(100, 3, 2, 1, 3, Some(42))?;

// Add polynomial features (degree 2)
add_polynomial_features(&mut dataset.data, 2)?;

println!("Augmented features: {}", dataset.n_features());
```

## Best Practices for Data Generation

### Reproducible Experiments

```rust
// Always use random_state for reproducible results
let seed = 42;
let dataset1 = make_classification(100, 10, 2, 1, 8, Some(seed))?;
let dataset2 = make_classification(100, 10, 2, 1, 8, Some(seed))?;

// These datasets will be identical
assert_eq!(dataset1.data, dataset2.data);
```

### Realistic Data Characteristics

```rust
use scirs2_datasets::generators::ClassificationConfig;

// Create realistic, challenging classification data
let config = ClassificationConfig {
    n_samples: 1000,
    n_features: 50,
    n_classes: 5,
    n_informative: 20,     // Not all features are informative
    n_redundant: 10,       // Some features are linear combinations
    n_repeated: 5,         // Some features are duplicated
    class_sep: 0.8,        // Moderate class separation (not too easy)
    flip_y: 0.05,          // 5% label noise (realistic)
    weights: Some(vec![0.4, 0.25, 0.2, 0.1, 0.05]), // Imbalanced classes
    random_state: Some(42),
};

let realistic_dataset = config.generate()?;
```

### Memory-Efficient Generation

```rust
// For large datasets, generate in chunks
fn generate_large_dataset(total_samples: usize, batch_size: usize) -> Result<Dataset, Box<dyn std::error::Error>> {
    let mut batches = Vec::new();
    
    for i in (0..total_samples).step_by(batch_size) {
        let current_batch_size = std::cmp::min(batch_size, total_samples - i);
        let batch = make_classification(
            current_batch_size, 10, 3, 2, 8, 
            Some(42 + i as u64)  // Different seed for each batch
        )?;
        batches.push(batch);
    }
    
    concatenate_datasets(&batches)
}

let large_dataset = generate_large_dataset(100_000, 1000)?;
```

## Performance Considerations

### Benchmarking Data Generation

```rust
use std::time::Instant;

let start = Instant::now();
let dataset = make_classification(10_000, 100, 5, 2, 50, Some(42))?;
let duration = start.elapsed();

println!("Generated {} samples in {:.2}ms", 
         dataset.n_samples(), duration.as_millis());
println!("Throughput: {:.1} samples/s", 
         dataset.n_samples() as f64 / duration.as_secs_f64());
```

### Parallel Generation

```rust
use rayon::prelude::*;

// Generate multiple datasets in parallel
let seeds: Vec<u64> = (0..10).collect();
let datasets: Vec<_> = seeds.par_iter()
    .map(|&seed| make_classification(1000, 20, 3, 2, 15, Some(seed)))
    .collect::<Result<Vec<_>, _>>()?;

println!("Generated {} datasets in parallel", datasets.len());
```

This tutorial covered the comprehensive data generation capabilities of SciRS2. These tools enable you to create diverse, realistic synthetic datasets for algorithm development, testing, and benchmarking in machine learning applications.