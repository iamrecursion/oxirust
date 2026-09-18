use scirs2_datasets::loaders::load_csv_legacy;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a CSV file with headers and target column
    let dataset = load_csv_legacy(
        "scirs2-datasets/data/example.csv",
        true,    // has header
        Some(3), // target column index (0-based)
    )?;

    println!("CSV dataset loaded successfully:");
    println!("  Samples: {}", dataset.n_samples());
    println!("  Features: {}", dataset.n_features());
    println!("  Feature names: {:?}", dataset.featurenames);

    // Access data and target
    println!("\nFirst 3 samples:");
    for i in 0..3 {
        let features = dataset.data.row(i);
        let target = dataset.target.as_ref().map(|t| t[i]);
        println!("  Sample {i}: Features = {features:?}, Target = {target:?}");
    }

    Ok(())
}
