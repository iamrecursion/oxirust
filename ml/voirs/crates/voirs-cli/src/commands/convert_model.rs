//! Model format conversion utilities
//!
//! Converts models from various formats (ONNX, PyTorch) to SafeTensors format
//! for use with VoiRS.

use super::onnx_weights::{extract_onnx_weights, write_safetensors};
use crate::GlobalOptions;
use safetensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use voirs_sdk::Result;

/// Run model conversion
pub async fn run_convert_model(
    input: PathBuf,
    output: PathBuf,
    from: Option<String>,
    model_type: String,
    verify: bool,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("🔄 VoiRS Model Converter");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Input:  {}", input.display());
        println!("Output: {}", output.display());
        println!("Type:   {}", model_type);
    }

    // Validate input file
    if !input.exists() {
        return Err(voirs_sdk::VoirsError::config_error(format!(
            "Input model file not found: {}",
            input.display()
        )));
    }

    // Auto-detect format if not specified
    let source_format = from.unwrap_or_else(|| detect_format(&input));

    if !global.quiet {
        println!("Format: {} → SafeTensors", source_format);
        println!();
    }

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Convert based on format
    match source_format.as_str() {
        "onnx" => {
            if !global.quiet {
                println!("📥 Loading ONNX model...");
            }
            convert_onnx_to_safetensors(&input, &output, &model_type, global).await?;
        }
        "pytorch" | "pt" | "pth" => {
            if !global.quiet {
                println!("📥 Loading PyTorch model...");
            }
            convert_pytorch_to_safetensors(&input, &output, &model_type, global).await?;
        }
        _ => {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Unsupported format: '{}'. Supported formats: onnx, pytorch/pt/pth",
                source_format
            )));
        }
    }

    if !global.quiet {
        println!("✅ Conversion complete!");
        println!("   Output: {}", output.display());
    }

    // Verify if requested
    if verify {
        if !global.quiet {
            println!();
            println!("🔍 Verifying converted model...");
        }
        verify_conversion(&output, &model_type, global).await?;
        if !global.quiet {
            println!("✅ Verification passed!");
        }
    }

    if !global.quiet {
        println!();
        println!("🎉 Model conversion successful!");
    }

    Ok(())
}

/// Detect input format from file extension
fn detect_format(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Convert ONNX model to SafeTensors
///
/// Every constant tensor (graph initializers and `Constant` node values) is
/// copied with its original ONNX element type -- see
/// [`super::onnx_weights`]. Tensors whose element type SafeTensors cannot
/// hold (strings, complex, FP8, 4-bit) are listed and skipped.
async fn convert_onnx_to_safetensors(
    input: &Path,
    output: &Path,
    model_type: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("📥 Loading ONNX model with oxionnx...");
    }

    let model_bytes = std::fs::read(input)?;
    let base_dir = input.parent().unwrap_or_else(|| Path::new("."));
    let weights = extract_onnx_weights(&model_bytes, base_dir).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to load ONNX model: {e}"))
    })?;

    if !global.quiet {
        println!("✅ Model loaded successfully");
        println!("📊 Model information:");
        println!("   - Total nodes: {}", weights.node_count);
        println!("   - Inputs: {}", weights.input_count);
        println!("   - Outputs: {}", weights.output_count);
        println!();
        println!(
            "📊 Extracted {} constant tensors from model graph",
            weights.tensors.len()
        );
        for skipped in &weights.skipped {
            println!("   ⚠️  Skipped '{}': {}", skipped.name, skipped.reason);
        }
    }

    let tensor_count = weights.tensors.len();

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "onnx".to_string());
    metadata.insert("source_path".to_string(), input.display().to_string());
    metadata.insert("model_type".to_string(), model_type.to_string());
    metadata.insert("tensor_count".to_string(), tensor_count.to_string());
    metadata.insert(
        "converted_with".to_string(),
        "voirs-cli/oxionnx".to_string(),
    );

    if !global.quiet {
        println!("💾 Saving as SafeTensors...");
    }

    write_safetensors(&weights.tensors, metadata, output).map_err(|e| {
        voirs_sdk::VoirsError::config_error(format!("Failed to save SafeTensors: {e}"))
    })?;

    if !global.quiet {
        println!("✅ Saved to {}", output.display());
        println!("📊 Summary:");
        println!("   - Extracted {} tensors", tensor_count);
        println!("   - Model type: {}", model_type);
        println!("   - Output format: SafeTensors");
    }

    Ok(())
}

/// Convert PyTorch model to SafeTensors
async fn convert_pytorch_to_safetensors(
    input: &Path,
    output: &Path,
    _model_type: &str,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("⚠️  PyTorch .pt/.pth conversion not yet implemented in pure Rust.");
        println!("    PyTorch files use Python's pickle format which requires:");
        println!("    1. Python interpreter with PyTorch installed, OR");
        println!("    2. tch-rs crate with libtorch dependency");
        println!();
        println!("🔧 Recommended Conversion Methods:");
        println!();
        println!("   Method 1: Python script (easiest)");
        println!("   ```python");
        println!("   import torch");
        println!("   from safetensors.torch import save_file");
        println!();
        println!("   # Load PyTorch model");
        println!(
            "   state_dict = torch.load('{}', map_location='cpu')",
            input.display()
        );
        println!();
        println!("   # Save as SafeTensors");
        println!("   save_file(state_dict, '{}')", output.display());
        println!("   ```");
        println!();
        println!("   Method 2: Convert to ONNX first");
        println!("   ```python");
        println!("   import torch");
        println!("   import torch.onnx");
        println!();
        println!("   model = torch.load('{}').eval()", input.display());
        println!("   dummy_input = torch.randn(1, 80, 100)  # Adjust shape");
        println!("   torch.onnx.export(model, dummy_input, 'model.onnx')");
        println!("   ```");
        println!("   Then: voirs convert-model model.onnx output.safetensors");
        println!();
        println!("   Method 3: Use tch-rs (requires libtorch)");
        println!("   Add to Cargo.toml: tch = \"0.15\"");
        println!("   Requires: libtorch C++ library installed");
    }

    Err(voirs_sdk::VoirsError::config_error(
        "PyTorch conversion requires Python script or tch-rs. See output above for methods.",
    ))
}

/// Verify converted model
async fn verify_conversion(output: &Path, model_type: &str, global: &GlobalOptions) -> Result<()> {
    if !global.quiet {
        println!("   Checking file exists...");
    }

    // Check if output file exists
    let metadata_path = output.with_extension("json");
    if !metadata_path.exists() {
        return Err(voirs_sdk::VoirsError::config_error(
            "Converted model metadata file not found",
        ));
    }

    if !global.quiet {
        println!("   Loading metadata...");
    }

    // Load and verify metadata
    let metadata_content = std::fs::read_to_string(&metadata_path)?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;

    // Check model type matches
    if let Some(mt) = metadata.get("model_type").and_then(|v| v.as_str()) {
        if mt != model_type && !global.quiet {
            println!(
                "   ⚠️  Model type mismatch: expected '{}', found '{}'",
                model_type, mt
            );
        }
    }

    if !global.quiet {
        println!("   Model type: {}", model_type);
        println!(
            "   Source format: {}",
            metadata
                .get("source_format")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        );
    }

    // Load and verify SafeTensors file
    if !global.quiet {
        println!("   Loading SafeTensors file...");
    }

    // Read SafeTensors file
    let safetensors_data = std::fs::read(output)?;

    // Parse SafeTensors format
    match safetensors::SafeTensors::deserialize(&safetensors_data) {
        Ok(tensors) => {
            if !global.quiet {
                println!("   ✅ SafeTensors format valid");
                println!("   Tensors found: {}", tensors.names().len());
                println!();

                // Show tensor information
                println!("   Tensor Details:");
                for name in tensors.names() {
                    if let Ok(tensor_view) = tensors.tensor(name) {
                        let shape = tensor_view.shape();
                        let dtype = tensor_view.dtype();
                        println!("   - {}: shape={:?}, dtype={:?}", name, shape, dtype);
                    }
                }

                // Model type specific validation
                println!();
                println!("   Model Type Validation:");
                match model_type {
                    "acoustic" => {
                        println!("   Checking for acoustic model tensors...");
                        let expected_tensors = vec!["encoder", "decoder", "mel_linear"];
                        check_expected_tensors(&tensors, &expected_tensors);
                    }
                    "vocoder" => {
                        println!("   Checking for vocoder model tensors...");
                        let expected_tensors = vec!["upsample", "resblock", "conv_post"];
                        check_expected_tensors(&tensors, &expected_tensors);
                    }
                    "g2p" => {
                        println!("   Checking for G2P model tensors...");
                        let expected_tensors = vec!["embedding", "transformer"];
                        check_expected_tensors(&tensors, &expected_tensors);
                    }
                    _ => {
                        println!("   Generic model - skipping specific tensor checks");
                    }
                }
            }
            Ok(())
        }
        Err(e) => Err(voirs_sdk::VoirsError::config_error(format!(
            "Failed to load SafeTensors: {}",
            e
        ))),
    }
}

/// Helper function to check for expected tensors
fn check_expected_tensors(tensors: &safetensors::SafeTensors, expected: &[&str]) {
    let names = tensors.names();
    let mut found_count = 0;

    for &expected_name in expected {
        let found = names.iter().any(|name| name.contains(expected_name));
        if found {
            println!("   ✅ Found tensor matching '{}'", expected_name);
            found_count += 1;
        } else {
            println!("   ⚠️  No tensor matching '{}'", expected_name);
        }
    }

    if found_count > 0 {
        println!(
            "   Model appears valid ({}/{} expected patterns found)",
            found_count,
            expected.len()
        );
    } else {
        println!("   ⚠️  Model may not match expected type (no standard tensors found)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("model.onnx")), "onnx");
        assert_eq!(detect_format(Path::new("model.pt")), "pt");
        assert_eq!(detect_format(Path::new("model.pth")), "pth");
        assert_eq!(detect_format(Path::new("model")), "unknown");
    }
}
