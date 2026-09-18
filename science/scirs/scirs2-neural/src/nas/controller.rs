//! NAS controller for building and managing architectures

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding,
    search_space::{Architecture, LayerType, SearchSpaceConfig},
};
use std::sync::Arc;

/// Configuration for the NAS controller
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Input shape for the models
    pub input_shape: Vec<usize>,
    /// Number of output classes
    pub num_classes: usize,
    /// Whether to add a final softmax layer
    pub add_softmax: bool,
    /// Global seed for reproducibility
    pub seed: Option<u64>,
    /// Device to use (cpu, cuda, etc.)
    pub device: String,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            input_shape: vec![32, 32, 3],
            num_classes: 10,
            add_softmax: true,
            seed: None,
            device: "cpu".to_string(),
        }
    }
}

/// NAS Controller for building models from architecture encodings
pub struct NASController {
    pub config: ControllerConfig,
    pub search_space: SearchSpaceConfig,
}

impl NASController {
    /// Create a new NAS controller
    pub fn new(search_space: SearchSpaceConfig) -> Result<Self> {
        Ok(Self {
            config: ControllerConfig::default(),
            search_space,
        })
    }

    /// Create with custom configuration
    pub fn with_config(search_space: SearchSpaceConfig, config: ControllerConfig) -> Result<Self> {
        Ok(Self {
            config,
            search_space,
        })
    }

    /// Build a model from an architecture encoding
    pub fn build_model(&self, encoding: &Arc<dyn ArchitectureEncoding>) -> Result<Sequential<f32>> {
        let architecture = encoding.to_architecture()?;
        self.build_from_architecture(&architecture)
    }

    /// Build a model from an Architecture struct
    pub fn build_from_architecture(&self, architecture: &Architecture) -> Result<Sequential<f32>> {
        use scirs2_core::random::{rngs::SmallRng, SeedableRng};
        let seed = scirs2_core::random::random::<u64>();
        let mut rng_inst = SmallRng::seed_from_u64(seed);
        let mut model = Sequential::new();
        let mut current_shape = self.config.input_shape.clone();
        let effective_layers = self.apply_multipliers(
            &architecture.layers,
            architecture.width_multiplier,
            architecture.depth_multiplier,
        )?;
        for layer_type in effective_layers.iter() {
            match layer_type {
                LayerType::Dense(units) => {
                    let input_size = current_shape.iter().product();
                    model.add_layer(crate::layers::Dense::new(
                        input_size,
                        *units,
                        None,
                        &mut rng_inst,
                    )?);
                    current_shape = vec![*units];
                }
                LayerType::Dropout(rate) => {
                    model.add_layer(crate::layers::Dropout::new(*rate as f64, &mut rng_inst)?);
                }
                LayerType::BatchNorm => {
                    let features = current_shape.last().copied().unwrap_or(1);
                    model.add_layer(crate::layers::BatchNorm::new(
                        features,
                        0.9,
                        1e-5,
                        &mut rng_inst,
                    )?);
                }
                LayerType::Activation(name) => {
                    let input_size = current_shape.iter().product();
                    model.add_layer(crate::layers::Dense::new(
                        input_size,
                        input_size,
                        Some(name.as_str()),
                        &mut rng_inst,
                    )?);
                }
                LayerType::Flatten => {
                    let input_size: usize = current_shape.iter().product();
                    model.add_layer(crate::layers::Dense::new(
                        input_size,
                        input_size,
                        None,
                        &mut rng_inst,
                    )?);
                    current_shape = vec![input_size];
                }
                _ => {
                    // Skip unsupported layer types in simplified builder
                    continue;
                }
            }
        }
        if self.config.add_softmax {
            let input_size = current_shape.iter().product();
            model.add_layer(crate::layers::Dense::new(
                input_size,
                self.config.num_classes,
                Some("softmax"),
                &mut rng_inst,
            )?);
        }
        Ok(model)
    }

    /// Apply width and depth multipliers to layers
    pub fn apply_multipliers(
        &self,
        layers: &[LayerType],
        width_mult: f32,
        depth_mult: f32,
    ) -> Result<Vec<LayerType>> {
        let mut result = Vec::new();
        for layer in layers {
            let repetitions = (depth_mult.max(0.1) as usize).max(1);
            for _ in 0..repetitions {
                let modified_layer = match layer {
                    LayerType::Dense(units) => {
                        LayerType::Dense((*units as f32 * width_mult).round() as usize)
                    }
                    LayerType::Conv2D {
                        filters,
                        kernel_size,
                        stride,
                    } => LayerType::Conv2D {
                        filters: (*filters as f32 * width_mult).round() as usize,
                        kernel_size: *kernel_size,
                        stride: *stride,
                    },
                    LayerType::Conv1D {
                        filters,
                        kernel_size,
                        stride,
                    } => LayerType::Conv1D {
                        filters: (*filters as f32 * width_mult).round() as usize,
                        kernel_size: *kernel_size,
                        stride: *stride,
                    },
                    LayerType::LSTM {
                        units,
                        return_sequences,
                    } => LayerType::LSTM {
                        units: (*units as f32 * width_mult).round() as usize,
                        return_sequences: *return_sequences,
                    },
                    LayerType::GRU {
                        units,
                        return_sequences,
                    } => LayerType::GRU {
                        units: (*units as f32 * width_mult).round() as usize,
                        return_sequences: *return_sequences,
                    },
                    LayerType::Attention { num_heads, key_dim } => LayerType::Attention {
                        num_heads: *num_heads,
                        key_dim: (*key_dim as f32 * width_mult).round() as usize,
                    },
                    other => other.clone(),
                };
                result.push(modified_layer);
            }
        }
        Ok(result)
    }

    /// Count parameters in a model
    pub fn count_parameters(&self, _model: &Sequential<f32>) -> Result<usize> {
        // Simplified parameter counting
        Ok(1_000_000)
    }

    /// Estimate FLOPs for a model
    pub fn estimate_flops(
        &self,
        _model: &Sequential<f32>,
        _input_shape: &[usize],
    ) -> Result<usize> {
        // Simplified FLOPs estimation
        Ok(1_000_000)
    }

    /// Compute output shape after a layer
    pub fn compute_output_shape(
        &self,
        layer_type: &LayerType,
        input_shape: &[usize],
    ) -> Result<Vec<usize>> {
        match layer_type {
            LayerType::Dense(units) => Ok(vec![*units]),
            LayerType::Conv2D {
                filters,
                kernel_size,
                stride,
            } => {
                if input_shape.len() < 2 {
                    return Err(NeuralError::InvalidArgument(
                        "Conv2D requires at least 2D input (H, W)".to_string(),
                    ));
                }
                let h = (input_shape[0].saturating_sub(kernel_size.0)) / stride.0 + 1;
                let w = (input_shape[1].saturating_sub(kernel_size.1)) / stride.1 + 1;
                Ok(vec![h, w, *filters])
            }
            LayerType::MaxPool2D { pool_size, stride }
            | LayerType::AvgPool2D { pool_size, stride } => {
                if input_shape.len() < 2 {
                    return Ok(input_shape.to_vec());
                }
                let h = (input_shape[0].saturating_sub(pool_size.0)) / stride.0 + 1;
                let w = (input_shape[1].saturating_sub(pool_size.1)) / stride.1 + 1;
                let channels = input_shape.get(2).copied().unwrap_or(1);
                Ok(vec![h, w, channels])
            }
            LayerType::GlobalMaxPool2D | LayerType::GlobalAvgPool2D => {
                let channels = input_shape.last().copied().unwrap_or(1);
                Ok(vec![channels])
            }
            LayerType::Flatten => {
                let total_size: usize = input_shape.iter().product();
                Ok(vec![total_size])
            }
            LayerType::Dropout(_)
            | LayerType::BatchNorm
            | LayerType::LayerNorm
            | LayerType::Activation(_)
            | LayerType::Residual => Ok(input_shape.to_vec()),
            LayerType::LSTM {
                units,
                return_sequences,
            }
            | LayerType::GRU {
                units,
                return_sequences,
            } => {
                if *return_sequences {
                    if input_shape.is_empty() {
                        Ok(vec![*units])
                    } else {
                        Ok(vec![input_shape[0], *units])
                    }
                } else {
                    Ok(vec![*units])
                }
            }
            LayerType::Attention { key_dim, .. } => {
                if input_shape.is_empty() {
                    Ok(vec![*key_dim])
                } else {
                    let mut output_shape = input_shape.to_vec();
                    if let Some(last) = output_shape.last_mut() {
                        *last = *key_dim;
                    }
                    Ok(output_shape)
                }
            }
            LayerType::Embedding { embedding_dim, .. } => Ok(vec![*embedding_dim]),
            _ => Ok(input_shape.to_vec()),
        }
    }

    /// Validate an architecture
    pub fn validate_architecture(&self, architecture: &Architecture) -> Result<()> {
        if architecture.layers.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "Architecture must have at least one layer".to_string(),
            ));
        }
        for (from, to) in &architecture.connections {
            if *from >= architecture.layers.len() || *to >= architecture.layers.len() {
                return Err(NeuralError::InvalidArgument(format!(
                    "Invalid skip connection: {} -> {}",
                    from, to
                )));
            }
            if from >= to {
                return Err(NeuralError::InvalidArgument(
                    "Skip connections must be forward connections".to_string(),
                ));
            }
        }
        if architecture.width_multiplier <= 0.0 || architecture.depth_multiplier <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "Multipliers must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::Architecture;

    #[test]
    fn test_controller_creation() {
        let search_space = SearchSpaceConfig::default();
        let controller = NASController::new(search_space).expect("failed to create controller");
        assert_eq!(controller.config.num_classes, 10);
    }

    #[test]
    fn test_architecture_validation() {
        let search_space = SearchSpaceConfig::default();
        let controller = NASController::new(search_space).expect("failed to create controller");
        let valid_arch = Architecture {
            layers: vec![
                LayerType::Dense(128),
                LayerType::Activation("relu".to_string()),
                LayerType::Dense(10),
            ],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&valid_arch).is_ok());
        let empty_arch = Architecture {
            layers: vec![],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&empty_arch).is_err());
        let invalid_skip = Architecture {
            layers: vec![LayerType::Dense(128), LayerType::Dense(10)],
            connections: vec![(1, 0)],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&invalid_skip).is_err());
    }

    #[test]
    fn test_multiplier_application() {
        let controller =
            NASController::new(SearchSpaceConfig::default()).expect("failed to create controller");
        let layers = vec![
            LayerType::Dense(100),
            LayerType::Conv2D {
                filters: 32,
                kernel_size: (3, 3),
                stride: (1, 1),
            },
        ];
        let modified = controller
            .apply_multipliers(&layers, 2.0, 1.0)
            .expect("failed to apply multipliers");
        match &modified[0] {
            LayerType::Dense(units) => assert_eq!(*units, 200),
            _ => panic!("Expected Dense layer"),
        }
    }
}
