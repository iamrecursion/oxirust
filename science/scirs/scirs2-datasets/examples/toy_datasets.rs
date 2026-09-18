use scirs2_datasets::{load_boston, load_iris};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iris = load_iris()?;
    println!("Iris dataset loaded:");
    println!("  Samples: {}", iris.n_samples());
    println!("  Features: {}", iris.n_features());
    println!(
        "  Target classes: {}",
        iris.targetnames.as_ref().map_or(0, |v| v.len())
    );

    let boston = load_boston()?;
    println!("\nBoston Housing dataset loaded:");
    println!("  Samples: {}", boston.n_samples());
    println!("  Features: {}", boston.n_features());

    Ok(())
}
