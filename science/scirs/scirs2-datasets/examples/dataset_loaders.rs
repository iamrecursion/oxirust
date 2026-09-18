use scirs2_datasets::loaders;
use scirs2_datasets::utils::{train_test_split, Dataset};
use std::env;
use std::path::Path;

#[allow(dead_code)]
fn main() {
    // Check if a CSV file is provided as a command-line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path_to_csv_file>", args[0]);
        println!("Example: {} examples/sampledata.csv", args[0]);
        return;
    }

    let filepath = &args[1];

    // Verify the file exists
    if !Path::new(filepath).exists() {
        println!("Error: File '{filepath}' does not exist");
        return;
    }

    // Load CSV file
    println!("Loading CSV file: {filepath}");
    let csv_config = loaders::CsvConfig {
        has_header: true,
        target_column: None,
        ..Default::default()
    };
    match loaders::load_csv(filepath, csv_config) {
        Ok(dataset) => {
            print_dataset_info(&dataset, "Loaded CSV");

            // Split the dataset for demonstration
            println!("\nDemonstrating train-test split...");
            match train_test_split(&dataset, 0.2, Some(42)) {
                Ok((train, test)) => {
                    println!("Training set: {} samples", train.n_samples());
                    println!("Test set: {} samples", test.n_samples());

                    // Save as JSON for demonstration
                    let jsonpath = format!("{filepath}.json");
                    println!("\nSaving training dataset to JSON: {jsonpath}");
                    if let Err(e) = loaders::save_json(&train, &jsonpath) {
                        println!("Error saving JSON: {e}");
                    } else {
                        println!("Successfully saved JSON file");

                        // Load back the JSON file
                        println!("\nLoading back from JSON file...");
                        match loaders::load_json(&jsonpath) {
                            Ok(loaded) => {
                                print_dataset_info(&loaded, "Loaded JSON");
                            }
                            Err(e) => println!("Error loading JSON: {e}"),
                        }
                    }
                }
                Err(e) => println!("Error splitting dataset: {e}"),
            }
        }
        Err(e) => println!("Error loading CSV: {e}"),
    }
}

#[allow(dead_code)]
fn print_dataset_info(dataset: &Dataset, name: &str) {
    println!("=== {name} Dataset ===");
    println!("Number of samples: {}", dataset.n_samples());
    println!("Number of features: {}", dataset.n_features());

    if let Some(featurenames) = &dataset.featurenames {
        println!(
            "Features: {:?}",
            &featurenames[0..std::cmp::min(5, featurenames.len())]
        );
        if featurenames.len() > 5 {
            println!("... and {} more", featurenames.len() - 5);
        }
    }

    if let Some(target) = &dataset.target {
        println!("Target shape: {}", target.len());

        if let Some(targetnames) = &dataset.targetnames {
            println!("Target classes: {targetnames:?}");
        }
    }

    for (key, value) in &dataset.metadata {
        println!("Metadata - {key}: {value}");
    }
}
