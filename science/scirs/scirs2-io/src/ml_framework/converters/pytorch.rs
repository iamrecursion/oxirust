//! PyTorch format converter
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::converters::MLFrameworkConverter;
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use crate::ml_framework::utils::{python_dict_to_tensor, tensor_to_python_dict};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// PyTorch format converter
pub struct PyTorchConverter;

impl MLFrameworkConverter for PyTorchConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // Save in a PyTorch-compatible format (simplified)
        let mut state_dict = HashMap::new();

        for (name, tensor) in &model.weights {
            state_dict.insert(name.clone(), tensor_to_python_dict(tensor)?);
        }

        let model_dict = serde_json::json!({
            "state_dict": state_dict,
            "metadata": model.metadata,
            "config": model.config,
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &model_dict)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(IoError::Io)?;
        let model_dict: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let mut model = MLModel::new(MLFramework::PyTorch);

        if let Some(metadata) = model_dict.get("metadata") {
            model.metadata = serde_json::from_value(metadata.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }

        if let Some(config) = model_dict.get("config") {
            model.config = serde_json::from_value(config.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }

        if let Some(state_dict) = model_dict.get("state_dict").and_then(|v| v.as_object()) {
            for (name, tensor_data) in state_dict {
                let tensor = python_dict_to_tensor(tensor_data)?;
                model.weights.insert(name.clone(), tensor);
            }
        }

        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_dict = tensor_to_python_dict(tensor)?;
        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &tensor_dict)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_tensor(&self, path: &Path) -> Result<MLTensor> {
        let file = File::open(path).map_err(IoError::Io)?;
        let tensor_dict: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        python_dict_to_tensor(&tensor_dict)
    }
}
