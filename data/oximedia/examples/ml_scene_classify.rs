//! Sovereign ML scene classification example — top-K labels from a single frame.
//!
//! Demonstrates end-to-end usage of the `oximedia::ml` scene-classification
//! pipeline. With no arguments the example runs in **dry-run** mode:
//!
//!   1. Builds an [`ImagePreprocessor`] with ImageNet-style normalisation.
//!   2. Preprocesses a synthetic 224×224 RGB gradient.
//!   3. Prints the tensor shape, min/max, and the default [`PipelineInfo`].
//!   4. Lists the default [`ModelZoo`] entries.
//!
//! With arguments, the example loads a real ONNX scene classifier and runs
//! inference on the synthetic image (a PNG decoder is intentionally not
//! pulled in — the example stays zero-extra-deps). If a sibling
//! `<model>.labels.txt` file exists next to the `.onnx` file, human-readable
//! labels are loaded from it (one label per line); otherwise numeric
//! `class_<n>` labels are used.
//!
//! # Usage
//!
//! ```bash
//! # Dry run (no inference, always exits 0):
//! cargo run -p oximedia --example ml_scene_classify --features ml-scene-classifier
//!
//! # Real inference (requires the `ml-onnx` feature plus an .onnx file):
//! cargo run -p oximedia --example ml_scene_classify \
//!     --features "ml-scene-classifier ml-onnx" -- /path/to/places365.onnx
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use oximedia::prelude::*;
use oximedia_ml::pipelines::{SceneClassifier, SceneClassifierConfig, SceneImage};

/// Build a 224×224 RGB gradient: R sweeps horizontally, G sweeps vertically,
/// B is a constant mid-tone. Deterministic and dependency-free.
fn build_synthetic_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = 64u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Locate an optional `<model>.labels.txt` sibling of `model_path` and
/// return it as a `Vec<String>` (one label per non-empty line). Returns
/// `None` if the file does not exist; returns an error only on I/O
/// failures while reading an existing file.
fn load_sibling_labels(
    model_path: &Path,
) -> Result<Option<Vec<String>>, Box<dyn std::error::Error>> {
    let mut candidate = PathBuf::from(model_path);
    let stem = candidate
        .file_stem()
        .map(std::ffi::OsStr::to_os_string)
        .ok_or("model path has no file stem")?;
    let mut labels_name = stem;
    labels_name.push(".labels.txt");
    candidate.set_file_name(labels_name);
    if !candidate.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&candidate)?;
    let labels: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    Ok(Some(labels))
}

/// Compute `(min, max)` of an f32 slice without panicking.
fn tensor_min_max(tensor: &[f32]) -> (f32, f32) {
    tensor
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
            (mn.min(*v), mx.max(*v))
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia — Sovereign ML Scene Classification");
    println!("============================================\n");

    // Step 1: synthetic 224×224 RGB image.
    let width = 224u32;
    let height = 224u32;
    let pixels = build_synthetic_image(width, height);
    println!(
        "Built synthetic {width}x{height} RGB image ({} bytes).",
        pixels.len()
    );

    // Step 2: ImageNet-style preprocessing.
    let preprocessor = ImagePreprocessor::new(width, height).with_imagenet_normalization();
    let tensor = preprocessor.process_u8_rgb(&pixels, width, height)?;
    let (min_v, max_v) = tensor_min_max(&tensor);
    println!(
        "Preprocessed into {} f32 values, logical shape = {:?}, min={:.4}, max={:.4}",
        tensor.len(),
        preprocessor.batch_shape(),
        min_v,
        max_v
    );

    // Step 3: device probe.
    let device = DeviceType::auto();
    println!("Auto-selected device: {device}");

    // Step 4: pipeline info (always printed, even in dry-run).
    let default_info = PipelineInfo {
        id: "scene-classifier/places365",
        name: "Scene Classifier",
        task: PipelineTask::SceneClassification,
        input_size: Some((width, height)),
    };
    println!(
        "\nDefault pipeline: id={} name={} task={:?} input={:?}",
        default_info.id, default_info.name, default_info.task, default_info.input_size
    );

    // Step 5: optional real-inference path.
    let args: Vec<String> = std::env::args().collect();
    if let Some(model_arg) = args.get(1) {
        let model_path = PathBuf::from(model_arg);
        println!("\nLoading ONNX model from: {}", model_path.display());

        let sibling_labels = load_sibling_labels(&model_path)?;
        let labels_for_config = sibling_labels.clone();
        let mut config = SceneClassifierConfig {
            input_size: (width, height),
            labels: labels_for_config,
            ..SceneClassifierConfig::default()
        };
        if let Some(ref labels) = sibling_labels {
            println!(
                "Loaded {} sibling labels from <model>.labels.txt",
                labels.len()
            );
        } else {
            println!("No sibling <model>.labels.txt found; numeric labels will be used.");
            // Generate numeric fallback labels for a generous class count; the
            // pipeline will only touch the ones it needs.
            config.labels = Some((0..1024).map(|i| format!("class_{i}")).collect());
        }

        match SceneClassifier::load_with_config(&model_path, device, config) {
            Ok(classifier) => {
                let info = classifier.info();
                println!(
                    "Loaded pipeline '{}' (input size {:?}).",
                    info.name, info.input_size
                );
                let image = SceneImage::new(pixels, width, height)?;
                match classifier.run(image) {
                    Ok(predictions) => {
                        println!("Top {} predictions:", predictions.len());
                        for (rank, pred) in predictions.iter().enumerate() {
                            let label = pred.label.as_deref().unwrap_or("<unlabelled>");
                            println!(
                                "  {:>2}. class={:>4}  score={:.4}  label={}",
                                rank + 1,
                                pred.class_index,
                                pred.score,
                                label
                            );
                        }
                    }
                    Err(MlError::FeatureDisabled(feat)) => {
                        println!(
                            "Inference disabled (missing feature '{feat}'). \
                             Rebuild with --features \"ml-scene-classifier ml-onnx\"."
                        );
                    }
                    Err(e) => {
                        eprintln!("Inference failed: {e}");
                    }
                }
            }
            Err(MlError::FeatureDisabled(feat)) => {
                println!(
                    "Model load requires the '{feat}' feature. \
                     Rebuild with --features \"ml-scene-classifier ml-onnx\"."
                );
            }
            Err(e) => {
                eprintln!("Failed to load classifier: {e}");
            }
        }
    } else {
        println!(
            "\nNo model path provided; dry-run complete. \
             Run with: cargo run -p oximedia --example ml_scene_classify \
             --features \"ml-scene-classifier ml-onnx\" -- <path-to-model.onnx>"
        );
    }

    // Step 6: preview the model zoo.
    let zoo = ModelZoo::with_defaults();
    println!("\nAvailable pipelines in the default zoo:");
    for entry in zoo.entries() {
        println!(
            "  id={:<30} name={:<40} task={:?}",
            entry.id, entry.name, entry.task
        );
    }

    Ok(())
}
