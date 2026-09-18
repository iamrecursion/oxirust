//! Real-world datasets demonstration
//!
//! This example demonstrates how to load and work with real-world datasets
//! from various domains including finance, healthcare, and machine learning research.
//!
//! Usage:
//!   cargo run --example real_world_datasets --release

use scirs2_core::ndarray::compat::ArrayStatCompat;
use scirs2_datasets::{
    list_real_world_datasets, load_adult, load_california_housing, load_heart_disease,
    load_red_wine_quality, load_titanic,
    utils::{k_fold_split, train_test_split},
    BenchmarkRunner, MLPipeline, RealWorldConfig,
};
use statrs::statistics::Statistics;
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 Real-World Datasets Demonstration");
    println!("====================================\n");

    // List all available real-world datasets
    demonstrate_dataset_catalog();

    // Load and explore different types of datasets
    demonstrate_classification_datasets()?;
    demonstrate_regression_datasets()?;
    demonstrate_healthcare_datasets()?;

    // Advanced dataset operations
    demonstrate_advanced_operations()?;

    // Performance comparison
    demonstrate_performance_comparison()?;

    println!("\n🎉 Real-world datasets demonstration completed!");
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_dataset_catalog() {
    println!("📋 AVAILABLE REAL-WORLD DATASETS");
    println!("{}", "-".repeat(40));

    let datasets = list_real_world_datasets();

    // Group datasets by domain
    let mut classification = Vec::new();
    let mut regression = Vec::new();
    let mut time_series = Vec::new();
    let mut healthcare = Vec::new();
    let mut financial = Vec::new();

    for dataset in &datasets {
        match dataset.as_str() {
            "adult" | "bank_marketing" | "credit_approval" | "german_credit" | "mushroom"
            | "spam" | "titanic" => classification.push(dataset),
            "auto_mpg" | "california_housing" | "concrete_strength" | "energy_efficiency"
            | "red_wine_quality" | "white_wine_quality" => regression.push(dataset),
            "air_passengers" | "bitcoin_prices" | "electricity_load" | "stock_prices" => {
                time_series.push(dataset)
            }
            "diabetes_readmission" | "heart_disease" => healthcare.push(dataset),
            "credit_card_fraud" | "loan_default" => financial.push(dataset),
            _ => {}
        }
    }

    println!("Classification Datasets ({}):", classification.len());
    for dataset in classification {
        println!("  • {dataset}");
    }

    println!("\nRegression Datasets ({}):", regression.len());
    for dataset in regression {
        println!("  • {dataset}");
    }

    println!("\nTime Series Datasets ({}):", time_series.len());
    for dataset in time_series {
        println!("  • {dataset}");
    }

    println!("\nHealthcare Datasets ({}):", healthcare.len());
    for dataset in healthcare {
        println!("  • {dataset}");
    }

    println!("\nFinancial Datasets ({}):", financial.len());
    for dataset in financial {
        println!("  • {dataset}");
    }

    println!(
        "\nTotal: {} real-world datasets available\n",
        datasets.len()
    );
}

#[allow(dead_code)]
fn demonstrate_classification_datasets() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 CLASSIFICATION DATASETS");
    println!("{}", "-".repeat(40));

    // Titanic dataset
    println!("Loading Titanic dataset...");
    let titanic = load_titanic()?;

    println!("Titanic Dataset:");
    println!(
        "  Description: {}",
        titanic
            .metadata
            .get("description")
            .unwrap_or(&"Unknown".to_string())
    );
    println!("  Samples: {}", titanic.n_samples());
    println!("  Features: {}", titanic.n_features());

    if let Some(featurenames) = titanic.featurenames() {
        println!("  Features: {featurenames:?}");
    }

    if let Some(targetnames) = titanic.targetnames() {
        println!("  Classes: {targetnames:?}");
    }

    // Analyze class distribution
    if let Some(target) = &titanic.target {
        let mut class_counts = HashMap::new();
        for &class in target.iter() {
            *class_counts.entry(class as i32).or_insert(0) += 1;
        }
        println!("  Class distribution: {class_counts:?}");

        // Calculate survival rate
        let survived = class_counts.get(&1).unwrap_or(&0);
        let total = titanic.n_samples();
        println!(
            "  Survival rate: {:.1}%",
            (*survived as f64 / total as f64) * 100.0
        );
    }

    // Demonstrate train/test split
    let (train, test) = train_test_split(&titanic, 0.2, Some(42))?;
    println!(
        "  Train/test split: {} train, {} test",
        train.n_samples(),
        test.n_samples()
    );

    // Adult (Census Income) dataset
    println!("\nLoading Adult (Census Income) dataset...");
    match load_adult() {
        Ok(adult) => {
            println!("Adult Dataset:");
            println!(
                "  Description: {}",
                adult
                    .metadata
                    .get("description")
                    .unwrap_or(&"Unknown".to_string())
            );
            println!("  Samples: {}", adult.n_samples());
            println!("  Features: {}", adult.n_features());
            println!("  Task: Predict income >$50K based on census data");
        }
        Err(e) => {
            println!("  Note: Adult dataset requires download: {e}");
            println!("  This is expected for the demonstration");
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_regression_datasets() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 REGRESSION DATASETS");
    println!("{}", "-".repeat(40));

    // California Housing dataset
    println!("Loading California Housing dataset...");
    let housing = load_california_housing()?;

    println!("California Housing Dataset:");
    println!(
        "  Description: {}",
        housing
            .metadata
            .get("description")
            .unwrap_or(&"Unknown".to_string())
    );
    println!("  Samples: {}", housing.n_samples());
    println!("  Features: {}", housing.n_features());

    if let Some(featurenames) = housing.featurenames() {
        println!("  Features: {featurenames:?}");
    }

    // Analyze target distribution
    if let Some(target) = &housing.target {
        let mean = target.mean_or(0.0);
        let std = target.std(0.0);
        let min = target.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = target.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!("  Target (house value) statistics:");
        println!("    Mean: {mean:.2} (hundreds of thousands)");
        println!("    Std:  {std:.2}");
        println!("    Range: [{min:.2}, {max:.2}]");
    }

    // Red Wine Quality dataset
    println!("\nLoading Red Wine Quality dataset...");
    let wine = load_red_wine_quality()?;

    println!("Red Wine Quality Dataset:");
    println!(
        "  Description: {}",
        wine.metadata
            .get("description")
            .unwrap_or(&"Unknown".to_string())
    );
    println!("  Samples: {}", wine.n_samples());
    println!("  Features: {}", wine.n_features());

    if let Some(target) = &wine.target {
        let mean_quality = target.mean_or(0.0);
        println!("  Average wine quality: {mean_quality:.1}/10");

        // Quality distribution
        let mut quality_counts = HashMap::new();
        for &quality in target.iter() {
            let q = quality.round() as i32;
            *quality_counts.entry(q).or_insert(0) += 1;
        }
        println!("  Quality distribution: {quality_counts:?}");
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_healthcare_datasets() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏥 HEALTHCARE DATASETS");
    println!("{}", "-".repeat(40));

    // Heart Disease dataset
    println!("Loading Heart Disease dataset...");
    let heart = load_heart_disease()?;

    println!("Heart Disease Dataset:");
    println!(
        "  Description: {}",
        heart
            .metadata
            .get("description")
            .unwrap_or(&"Unknown".to_string())
    );
    println!("  Samples: {}", heart.n_samples());
    println!("  Features: {}", heart.n_features());

    if let Some(featurenames) = heart.featurenames() {
        println!("  Clinical features: {:?}", &featurenames[..5]); // Show first 5
        println!("  ... and {} more features", featurenames.len() - 5);
    }

    // Analyze risk factors
    if let Some(target) = &heart.target {
        let mut disease_counts = HashMap::new();
        for &disease in target.iter() {
            *disease_counts.entry(disease as i32).or_insert(0) += 1;
        }

        let with_disease = disease_counts.get(&1).unwrap_or(&0);
        let total = heart.n_samples();
        println!(
            "  Disease prevalence: {:.1}% ({}/{})",
            (*with_disease as f64 / total as f64) * 100.0,
            with_disease,
            total
        );
    }

    // Demonstrate feature analysis
    println!("  Sample clinical parameter ranges:");
    let age_col = heart.data.column(0);
    let age_mean = age_col.mean();
    let age_std = age_col.std(0.0);
    println!("    Age: {age_mean:.1} ± {age_std:.1} years");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 ADVANCED DATASET OPERATIONS");
    println!("{}", "-".repeat(40));

    let housing = load_california_housing()?;

    // Data preprocessing pipeline
    println!("Preprocessing pipeline for California Housing:");

    // 1. Train/test split
    let (mut train, test) = train_test_split(&housing, 0.2, Some(42))?;
    println!(
        "  1. Split: {} train, {} test",
        train.n_samples(),
        test.n_samples()
    );

    // 2. Feature scaling
    let mut pipeline = MLPipeline::default();
    train = pipeline.prepare_dataset(&train)?;
    println!("  2. Standardized features");

    // 3. Cross-validation setup
    let cv_folds = k_fold_split(train.n_samples(), 5, true, Some(42))?;
    println!("  3. Created {} CV folds", cv_folds.len());

    // Feature correlation analysis (simplified)
    println!("  4. Feature analysis:");
    println!("     • {} numerical features", train.n_features());
    println!("     • Ready for machine learning models");

    // Custom dataset configuration
    println!("\nCustom dataset loading configuration:");
    let config = RealWorldConfig {
        use_cache: true,
        download_if_missing: false, // Don't download in demo
        return_preprocessed: true,
        subset: Some("small".to_string()),
        random_state: Some(42),
        ..Default::default()
    };

    println!("  • Caching: {}", config.use_cache);
    println!("  • Download missing: {}", config.download_if_missing);
    println!("  • Preprocessed: {}", config.return_preprocessed);
    println!("  • Subset: {:?}", config.subset);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ PERFORMANCE COMPARISON");
    println!("{}", "-".repeat(40));

    let runner = BenchmarkRunner::new().with_iterations(3).with_warmup(1);

    // Benchmark real-world dataset loading
    println!("Benchmarking real-world dataset operations...");

    // Titanic loading benchmark
    let titanic_params = HashMap::from([("dataset".to_string(), "titanic".to_string())]);
    let titanic_result =
        runner.run_benchmark("load_titanic", titanic_params, || match load_titanic() {
            Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
            Err(e) => Err(format!("Failed to load Titanic: {e}")),
        });

    // California Housing loading benchmark
    let housing_params = HashMap::from([("dataset".to_string(), "california_housing".to_string())]);
    let housing_result = runner.run_benchmark("load_california_housing", housing_params, || {
        match load_california_housing() {
            Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
            Err(e) => Err(format!("Failed to load California Housing: {e}")),
        }
    });

    // Heart Disease loading benchmark
    let heart_params = HashMap::from([("dataset".to_string(), "heart_disease".to_string())]);
    let heart_result =
        runner.run_benchmark(
            "load_heart_disease",
            heart_params,
            || match load_heart_disease() {
                Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                Err(e) => Err(format!("Failed to load Heart Disease: {e}")),
            },
        );

    // Display results
    println!("\nReal-world dataset loading performance:");

    let results = vec![
        ("Titanic", &titanic_result),
        ("California Housing", &housing_result),
        ("Heart Disease", &heart_result),
    ];

    for (name, result) in results {
        if result.success {
            println!(
                "  {}: {} ({} samples, {} features, {:.1} samples/s)",
                name,
                result.formatted_duration(),
                result.samples,
                result.features,
                result.throughput
            );
        } else {
            println!(
                "  {}: Failed - {}",
                name,
                result
                    .error
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            );
        }
    }

    // Memory usage estimation
    let total_samples = titanic_result.samples + housing_result.samples + heart_result.samples;
    let total_features = titanic_result.features + housing_result.features + heart_result.features;
    let estimated_memory_mb = (total_samples * total_features * 8) as f64 / (1024.0 * 1024.0);

    println!("\nMemory usage estimate:");
    println!("  Total samples: {total_samples}");
    println!("  Total features: {total_features}");
    println!("  Estimated memory: {estimated_memory_mb:.1} MB");

    // Performance recommendations
    println!("\nPerformance recommendations:");
    if estimated_memory_mb > 100.0 {
        println!("  • Consider using streaming for large datasets");
        println!("  • Enable caching for frequently accessed datasets");
    }
    println!("  • Use train/test splitting to reduce memory usage");
    println!("  • Apply feature selection to reduce dimensionality");

    println!();
    Ok(())
}

/// Helper function to format large numbers
#[allow(dead_code)]
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Demonstrate dataset information display
#[allow(dead_code)]
fn show_dataset_info(name: &str, dataset: &scirs2_datasets::utils::Dataset) {
    println!("{name}:");
    println!("  Samples: {}", format_number(dataset.n_samples()));
    println!("  Features: {}", dataset.n_features());
    println!(
        "  Task: {}",
        dataset
            .metadata
            .get("task_type")
            .unwrap_or(&"Unknown".to_string())
    );

    if let Some(source) = dataset.metadata.get("source") {
        println!("  Source: {source}");
    }

    if dataset.target.is_some() {
        println!("  Supervised: Yes");
    } else {
        println!("  Supervised: No");
    }
}
