//! Inspect the built-in `oximedia::ml::ModelZoo`.
//!
//! The model zoo is a lightweight, in-memory catalogue of models known to
//! `oximedia-ml` pipelines. It does **not** download weights; users bring
//! their own `.onnx` file. This example:
//!
//!   1. Constructs the default zoo via [`ModelZoo::with_defaults`].
//!   2. Prints every registered [`ModelEntry`].
//!   3. Registers a custom entry (via [`ModelZoo::register`]) and re-lists.
//!   4. Demonstrates the `.get()` lookup path.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oximedia --example ml_model_zoo --features ml
//! ```

use oximedia::prelude::*;

fn format_opt_tuple(value: Option<(u32, u32)>) -> String {
    match value {
        Some((w, h)) => format!("{w}x{h}"),
        None => "—".to_string(),
    }
}

fn format_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "—".to_string(), |v| v.to_string())
}

fn print_zoo(title: &str, zoo: &ModelZoo) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
    if zoo.is_empty() {
        println!("Zoo is empty; register entries via ModelZoo::register(ModelEntry {{ .. }}).");
        return;
    }
    println!(
        "  {:<28} {:<18} {:<10} {:<10} Notes",
        "ID", "Task", "Input", "Classes"
    );
    println!("  {}", "-".repeat(96));
    // Collect + sort for deterministic output across runs.
    let mut sorted: Vec<&ModelEntry> = zoo.entries().collect();
    sorted.sort_by_key(|e| e.id);
    for entry in sorted {
        println!(
            "  {:<28} {:<18} {:<10} {:<10} {}",
            entry.id,
            format!("{:?}", entry.task),
            format_opt_tuple(entry.input_size),
            format_opt_usize(entry.num_classes),
            entry.notes,
        );
    }
    println!("\n  {} entries total.", zoo.len());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia — Sovereign ML Model Zoo");
    println!("=================================");

    // Step 1: inspect the default registry.
    let mut zoo = ModelZoo::with_defaults();
    print_zoo("Default zoo (built-in entries)", &zoo);

    // Step 2: demonstrate a lookup.
    if let Some(entry) = zoo.get("places365/resnet18") {
        println!(
            "\nLookup example: `places365/resnet18` → task={:?}, input={}, classes={}",
            entry.task,
            format_opt_tuple(entry.input_size),
            format_opt_usize(entry.num_classes),
        );
    } else {
        println!("\nLookup example: `places365/resnet18` not found.");
    }

    // Step 3: register a custom entry. `ModelEntry` fields are `&'static str`,
    // so the registration uses string literals.
    zoo.register(ModelEntry {
        id: "demo/custom-classifier",
        name: "User-supplied demo classifier",
        task: PipelineTask::Custom,
        input_size: Some((256, 256)),
        num_classes: Some(10),
        notes: "Registered at runtime via ModelZoo::register().",
    });
    print_zoo("After ModelZoo::register(...)", &zoo);

    // Step 4: illustrate the `ModelZoo::new` empty path.
    let empty = ModelZoo::new();
    print_zoo("Fresh ModelZoo::new()", &empty);

    println!(
        "\nTip: the zoo is a discovery surface. Pipelines load weights from a \
         user-supplied `.onnx` path — e.g. `SceneClassifier::load_with_config(\
         path, DeviceType::auto(), config)`."
    );

    Ok(())
}
