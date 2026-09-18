//! HuggingFace format converter
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::converters::{MLFrameworkConverter, SafeTensorsConverter};
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use std::fs::File;
use std::path::Path;

/// HuggingFace format converter
pub struct HuggingFaceConverter;

impl MLFrameworkConverter for HuggingFaceConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // HuggingFace models typically use safetensors + config.json
        let config_path = path.with_extension("json");
        let weights_path = path.with_extension("safetensors");

        // Save config
        let config = serde_json::json!({
            "architectures": [model.metadata.architecture],
            "model_type": "custom",
            "torch_dtype": "float32",
            "_name_or_path": model.metadata.model_name,
            "transformers_version": "4.30.0",
            "config": model.config
        });

        let config_file = File::create(&config_path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(config_file, &config)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        // Save weights in SafeTensors format
        let safetensors_converter = SafeTensorsConverter;
        safetensors_converter.save_model(model, &weights_path)
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let config_path = path.with_extension("json");
        let weights_path = path.with_extension("safetensors");

        // Load config
        let config_file = File::open(&config_path).map_err(IoError::Io)?;
        let config: serde_json::Value = serde_json::from_reader(config_file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        // Load weights
        let safetensors_converter = SafeTensorsConverter;
        let mut model = safetensors_converter.load_model(&weights_path)?;

        // Update with HuggingFace-specific metadata
        model.metadata.framework = "HuggingFace".to_string();
        if let Some(name) = config.get("_name_or_path").and_then(|v| v.as_str()) {
            model.metadata.model_name = Some(name.to_string());
        }
        if let Some(arch) = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
        {
            model.metadata.architecture = Some(arch.to_string());
        }
        if let Some(hf_config) = config.get("config") {
            model.config = serde_json::from_value(hf_config.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }

        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        // Use SafeTensors format for individual tensors
        let safetensors_converter = SafeTensorsConverter;
        safetensors_converter.save_tensor(tensor, path)
    }

    fn load_tensor(&self, path: &Path) -> Result<MLTensor> {
        // Use SafeTensors format for individual tensors
        let safetensors_converter = SafeTensorsConverter;
        safetensors_converter.load_tensor(path)
    }
}
