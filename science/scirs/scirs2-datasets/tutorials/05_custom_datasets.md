# Custom Datasets Tutorial

This tutorial covers how to load, create, and work with custom datasets in SciRS2, including various file formats and data sources.

## Overview

SciRS2 supports loading custom datasets from:

- **CSV files**: Most common format with flexible parsing
- **JSON files**: Structured data with metadata
- **ARFF files**: Weka's Attribute-Relation File Format
- **LIBSVM files**: Sparse format for SVMs
- **NumPy arrays**: Direct integration with Python ecosystem
- **Custom formats**: Extensible format support
- **Streaming data**: Large datasets that don't fit in memory

## Loading CSV Files

### Basic CSV Loading

```rust
use scirs2_datasets::{load_csv, CsvConfig};
use std::path::Path;

// Basic CSV loading with default configuration
let config = CsvConfig::default();
let dataset = load_csv("path/to/your/data.csv", config)?;

println!("Loaded CSV dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());
```

### Advanced CSV Configuration

```rust
use scirs2_datasets::{loaders::{CsvConfig, CsvLoadingMode}, load_csv};

// Comprehensive CSV configuration
let config = CsvConfig {
    has_header: true,
    delimiter: ',',
    quote_char: '"',
    escape_char: Some('\\'),
    target_column: Some("target".to_string()),      // Column name for target
    feature_columns: Some(vec![                     // Specific columns to load
        "feature1".to_string(),
        "feature2".to_string(),
        "feature3".to_string(),
    ]),
    skip_rows: 0,
    max_rows: None,
    missing_values: vec!["NA".to_string(), "NULL".to_string(), "".to_string()],
    loading_mode: CsvLoadingMode::InMemory,
    data_types: None, // Auto-detect data types
    date_format: None,
    encoding: "utf-8".to_string(),
};

let dataset = load_csv("data/custom_data.csv", config)?;

println!("Advanced CSV loading:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());
if let Some(feature_names) = dataset.feature_names() {
    println!("  Feature names: {:?}", feature_names);
}
```

### CSV with Custom Preprocessing

```rust
use scirs2_datasets::{load_csv_with_preprocessing, CsvConfig, PreprocessingConfig};

let csv_config = CsvConfig::default().with_header(true);
let preprocessing_config = PreprocessingConfig {
    standardize: true,
    handle_missing: true,
    remove_outliers: true,
    outlier_threshold: 3.0,
    encoding_strategy: EncodingStrategy::OneHot,
};

let dataset = load_csv_with_preprocessing(
    "data/raw_data.csv", 
    csv_config, 
    preprocessing_config
)?;

println!("CSV with preprocessing applied");
```

## Loading JSON Files

### Structured JSON Data

```rust
use scirs2_datasets::{load_json, JsonConfig};

// JSON configuration
let config = JsonConfig {
    data_path: "data".to_string(),           // Path to data array in JSON
    target_path: Some("target".to_string()), // Path to target array
    feature_names_path: Some("feature_names".to_string()),
    metadata_path: Some("metadata".to_string()),
};

let dataset = load_json("data/dataset.json", config)?;

println!("Loaded JSON dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());
```

### JSON with Nested Structure

```rust
use scirs2_datasets::{load_json_nested, JsonConfig};

// Example JSON structure:
// {
//   "experiment": {
//     "data": [[1, 2, 3], [4, 5, 6]],
//     "labels": [0, 1],
//     "metadata": {"description": "Test dataset"}
//   }
// }

let config = JsonConfig {
    data_path: "experiment.data".to_string(),
    target_path: Some("experiment.labels".to_string()),
    metadata_path: Some("experiment.metadata".to_string()),
    ..Default::default()
};

let dataset = load_json_nested("data/nested.json", config)?;
```

## Loading ARFF Files

### Basic ARFF Loading

```rust
use scirs2_datasets::{load_arff, ArffConfig};

// ARFF files include metadata about attributes
let config = ArffConfig::default();
let dataset = load_arff("data/dataset.arff", config)?;

println!("Loaded ARFF dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());

// ARFF files provide rich metadata
if let Some(feature_names) = dataset.feature_names() {
    println!("  Feature names: {:?}", feature_names);
}
if let Some(target_names) = dataset.target_names() {
    println!("  Target names: {:?}", target_names);
}
```

### ARFF with Custom Attribute Handling

```rust
use scirs2_datasets::{load_arff, ArffConfig, AttributeType};

let config = ArffConfig {
    ignore_attributes: vec!["id".to_string(), "timestamp".to_string()],
    categorical_encoding: CategoricalEncoding::LabelEncoding,
    handle_missing: true,
    attribute_types: Some(vec![
        ("age".to_string(), AttributeType::Numeric),
        ("gender".to_string(), AttributeType::Categorical),
        ("income".to_string(), AttributeType::Numeric),
    ]),
};

let dataset = load_arff("data/complex.arff", config)?;
```

## Loading LIBSVM Files

### Sparse Data Format

```rust
use scirs2_datasets::{load_libsvm, LibsvmConfig};

// LIBSVM format: label feature_id:value feature_id:value ...
let config = LibsvmConfig {
    n_features: None,        // Auto-detect or specify
    zero_based: false,       // Feature indices start from 1
    multilabel: false,       // Single-label classification
    max_features: None,      // Limit number of features
};

let dataset = load_libsvm("data/sparse_data.libsvm", config)?;

println!("Loaded LIBSVM dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());
```

## Creating Custom Datasets

### From Raw Arrays

```rust
use scirs2_datasets::{Dataset, DatasetMetadata};
use ndarray::{Array2, Array1};

// Create dataset from raw data
let data = Array2::from_shape_vec(
    (100, 5),
    (0..500).map(|x| x as f64).collect()
)?;

let target = Array1::from_vec(
    (0..100).map(|x| (x % 3) as f64).collect()
);

let metadata = DatasetMetadata {
    name: "Custom Dataset".to_string(),
    description: "A custom dataset created from raw arrays".to_string(),
    feature_names: Some(vec![
        "feature_1".to_string(),
        "feature_2".to_string(),
        "feature_3".to_string(),
        "feature_4".to_string(),
        "feature_5".to_string(),
    ]),
    target_names: Some(vec!["class_0".to_string(), "class_1".to_string(), "class_2".to_string()]),
    ..Default::default()
};

let dataset = Dataset {
    data,
    target: Some(target),
    metadata,
};

println!("Created custom dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Features: {}", dataset.n_features());
```

### From Function Generator

```rust
use scirs2_datasets::{Dataset, DatasetMetadata};
use ndarray::{Array2, Array1};

// Create dataset using a mathematical function
fn create_function_dataset(n_samples: usize) -> Result<Dataset, Box<dyn std::error::Error>> {
    let mut data = Array2::zeros((n_samples, 2));
    let mut target = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        let x = (i as f64 / n_samples as f64) * 4.0 * std::f64::consts::PI;
        let y = x.sin() + 0.1 * (i as f64).cos();
        
        data[[i, 0]] = x;
        data[[i, 1]] = y;
        target[i] = if y > 0.0 { 1.0 } else { 0.0 };
    }
    
    let metadata = DatasetMetadata {
        name: "Sine Wave Dataset".to_string(),
        description: "Dataset generated from sine wave function".to_string(),
        feature_names: Some(vec!["x".to_string(), "sin(x)".to_string()]),
        target_names: Some(vec!["negative".to_string(), "positive".to_string()]),
        ..Default::default()
    };
    
    Ok(Dataset {
        data,
        target: Some(target),
        metadata,
    })
}

let dataset = create_function_dataset(200)?;
```

## Working with Large Datasets

### Streaming CSV Loading

```rust
use scirs2_datasets::{load_csv_streaming, CsvConfig, StreamingConfig};

let csv_config = CsvConfig::default().with_header(true);
let streaming_config = StreamingConfig {
    chunk_size: 1000,           // Process 1000 rows at a time
    parallel: true,             // Use parallel processing
    memory_limit: Some(1024),   // Memory limit in MB
    cache_chunks: false,        // Don't cache chunks in memory
};

// Load large CSV file in streaming mode
let dataset_stream = load_csv_streaming(
    "data/large_dataset.csv", 
    csv_config, 
    streaming_config
)?;

// Process chunks one at a time
for (chunk_idx, chunk) in dataset_stream.enumerate() {
    let chunk = chunk?;
    println!("Processing chunk {}: {} samples", chunk_idx, chunk.n_samples());
    
    // Process chunk data here
    // Your analysis/training code
}
```

### Memory-Mapped Files

```rust
use scirs2_datasets::{load_mmap, MmapConfig};

// Memory-map large files without loading into RAM
let config = MmapConfig {
    data_type: DataType::F64,
    shape: (1_000_000, 100),    // 1M samples, 100 features
    header_size: 0,             // No header
    endianness: Endianness::Little,
};

let dataset = load_mmap("data/huge_dataset.bin", config)?;

println!("Memory-mapped dataset:");
println!("  Samples: {}", dataset.n_samples());
println!("  Memory usage: minimal (data stays on disk)");
```

## Database Integration

### SQL Database Loading

```rust
use scirs2_datasets::{load_from_sql, SqlConfig};

let config = SqlConfig {
    connection_string: "sqlite:///data/database.db".to_string(),
    query: "SELECT feature1, feature2, feature3, target FROM training_data".to_string(),
    target_column: Some("target".to_string()),
    batch_size: 1000,
};

let dataset = load_from_sql(config)?;

println!("Loaded from SQL database:");
println!("  Samples: {}", dataset.n_samples());
```

### NoSQL Database Loading

```rust
use scirs2_datasets::{load_from_mongodb, MongoConfig};

let config = MongoConfig {
    connection_string: "mongodb://localhost:27017".to_string(),
    database: "ml_data".to_string(),
    collection: "training_samples".to_string(),
    query: Some(r#"{"status": "active"}"#.to_string()),
    projection: Some(r#"{"features": 1, "label": 1, "_id": 0}"#.to_string()),
};

let dataset = load_from_mongodb(config)?;
```

## Custom Format Support

### Implementing Custom Loader

```rust
use scirs2_datasets::{Dataset, DatasetMetadata, error::Result};
use std::path::Path;

// Custom loader for your specific format
pub fn load_custom_format<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)?;
    
    // Parse your custom format
    let lines: Vec<&str> = content.lines().collect();
    let n_samples = lines.len();
    let n_features = lines[0].split(',').count() - 1; // Assuming last column is target
    
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);
    
    for (i, line) in lines.iter().enumerate() {
        let values: Vec<f64> = line.split(',')
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        
        for j in 0..n_features {
            data[[i, j]] = values[j];
        }
        target[i] = values[n_features];
    }
    
    let metadata = DatasetMetadata {
        name: path.file_stem().unwrap().to_string_lossy().to_string(),
        description: "Custom format dataset".to_string(),
        source: Some(path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    Ok(Dataset {
        data,
        target: Some(target),
        metadata,
    })
}

// Usage
let dataset = load_custom_format("data/my_format.dat")?;
```

## Data Validation and Quality Checks

### Automatic Validation

```rust
use scirs2_datasets::{load_csv, CsvConfig, utils::validate_dataset};

let config = CsvConfig::default().with_header(true);
let dataset = load_csv("data/questionable_data.csv", config)?;

// Comprehensive validation
let validation_report = validate_dataset(&dataset)?;

println!("Data validation results:");
println!("  Valid: {}", validation_report.is_valid);
println!("  Warnings: {}", validation_report.warnings.len());
println!("  Errors: {}", validation_report.errors.len());

for warning in &validation_report.warnings {
    println!("  ⚠️  {}", warning);
}

for error in &validation_report.errors {
    println!("  ❌ {}", error);
}
```

### Custom Validation Rules

```rust
use scirs2_datasets::{Dataset, ValidationRule, ValidationResult};

// Define custom validation rules
struct CustomValidator;

impl ValidationRule for CustomValidator {
    fn validate(&self, dataset: &Dataset) -> ValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // Custom rule: Check if all features are non-negative
        if dataset.data.iter().any(|&x| x < 0.0) {
            warnings.push("Dataset contains negative values".to_string());
        }
        
        // Custom rule: Check target distribution
        if let Some(target) = &dataset.target {
            let mut class_counts = std::collections::HashMap::new();
            for &class in target.iter() {
                *class_counts.entry(class as i32).or_insert(0) += 1;
            }
            
            let min_count = class_counts.values().min().unwrap_or(&0);
            let max_count = class_counts.values().max().unwrap_or(&0);
            
            if max_count > min_count * 10 {
                warnings.push("Highly imbalanced classes detected".to_string());
            }
        }
        
        ValidationResult {
            is_valid: errors.is_empty(),
            warnings,
            errors,
        }
    }
}

// Apply custom validation
let validator = CustomValidator;
let result = validator.validate(&dataset);
```

## Performance Optimization

### Parallel Loading

```rust
use scirs2_datasets::{load_csv_parallel, CsvConfig, StreamingConfig};
use rayon::prelude::*;

// Load multiple files in parallel
let files = vec![
    "data/batch1.csv",
    "data/batch2.csv", 
    "data/batch3.csv",
];

let datasets: Result<Vec<_>, _> = files.par_iter()
    .map(|&file| {
        let config = CsvConfig::default().with_header(true);
        let streaming_config = StreamingConfig::default().with_parallel(true);
        load_csv_parallel(file, config, streaming_config)
    })
    .collect();

let datasets = datasets?;
println!("Loaded {} datasets in parallel", datasets.len());
```

### Caching Strategy

```rust
use scirs2_datasets::{load_csv_cached, CsvConfig, CacheConfig};

let csv_config = CsvConfig::default().with_header(true);
let cache_config = CacheConfig {
    enable_cache: true,
    cache_dir: "cache/datasets".to_string(),
    ttl_seconds: Some(3600), // Cache for 1 hour
    compression: true,
    checksum_validation: true,
};

// First load: reads from file and caches
let dataset1 = load_csv_cached("data/large_file.csv", csv_config.clone(), cache_config.clone())?;

// Second load: reads from cache (much faster)
let dataset2 = load_csv_cached("data/large_file.csv", csv_config, cache_config)?;

println!("Cached loading improves performance significantly");
```

This tutorial covered comprehensive custom dataset loading and creation capabilities in SciRS2. These tools enable you to work with diverse data sources and formats, making it easy to integrate your own datasets into the SciRS2 ecosystem.