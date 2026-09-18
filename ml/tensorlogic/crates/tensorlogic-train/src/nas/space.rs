//! Neural architecture search space definitions.
//!
//! Defines [`ArchSearchSpace`], [`Architecture`], and [`LayerSpec`] — the primitives
//! that describe which architectures are valid candidates during NAS.

use std::collections::HashMap;

use crate::error::{TrainError, TrainResult};
use crate::hyperparameter::{HyperparamConfig, HyperparamValue};

// ─── LayerSpec ──────────────────────────────────────────────────────────────

/// Specification for a single layer in a neural architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    /// Operation type (e.g. "linear", "conv", "attention").
    pub op: String,
    /// Width (number of output units / channels).
    pub width: usize,
    /// Non-linearity applied after the operation (e.g. "relu", "gelu").
    pub activation: String,
}

// ─── Architecture ───────────────────────────────────────────────────────────

/// A concrete neural architecture represented as an ordered sequence of layers.
#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    /// Ordered layer specifications (input → output).
    pub layers: Vec<LayerSpec>,
}

impl Architecture {
    /// Proxy parameter count: sum of width_i × width_{i+1} over consecutive pairs.
    ///
    /// Returns 0 if fewer than 2 layers.
    pub fn param_count(&self) -> usize {
        if self.layers.len() < 2 {
            return 0;
        }
        self.layers
            .windows(2)
            .map(|w| w[0].width * w[1].width)
            .sum()
    }

    /// Depth (number of layers).
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Encode this architecture as a [`HyperparamConfig`].
    ///
    /// Keys: `depth` (Int), `layer_{i}_op` (String), `layer_{i}_width` (Int),
    /// `layer_{i}_activation` (String).
    pub fn to_config(&self) -> HyperparamConfig {
        let mut m: HashMap<String, HyperparamValue> = HashMap::new();
        m.insert(
            "depth".to_string(),
            HyperparamValue::Int(self.layers.len() as i64),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            m.insert(
                format!("layer_{i}_op"),
                HyperparamValue::String(layer.op.clone()),
            );
            m.insert(
                format!("layer_{i}_width"),
                HyperparamValue::Int(layer.width as i64),
            );
            m.insert(
                format!("layer_{i}_activation"),
                HyperparamValue::String(layer.activation.clone()),
            );
        }
        m
    }

    /// Reconstruct an [`Architecture`] from a [`HyperparamConfig`] created by [`Architecture::to_config`].
    ///
    /// `max_depth` is used only for bounds-checking the encoded depth.
    pub fn from_config(cfg: &HyperparamConfig, max_depth: usize) -> TrainResult<Self> {
        let depth = cfg.get("depth").and_then(|v| v.as_int()).ok_or_else(|| {
            TrainError::InvalidParameter("config missing 'depth' Int key".to_string())
        })?;

        if depth < 1 {
            return Err(TrainError::InvalidParameter(format!(
                "decoded depth {depth} must be ≥ 1"
            )));
        }
        if depth as usize > max_depth {
            return Err(TrainError::InvalidParameter(format!(
                "decoded depth {depth} exceeds max_depth {max_depth}"
            )));
        }

        let mut layers = Vec::with_capacity(depth as usize);
        for i in 0..depth as usize {
            let op = cfg
                .get(&format!("layer_{i}_op"))
                .and_then(|v| v.as_string())
                .ok_or_else(|| {
                    TrainError::InvalidParameter(format!(
                        "config missing 'layer_{i}_op' String key"
                    ))
                })?
                .to_string();

            let width = cfg
                .get(&format!("layer_{i}_width"))
                .and_then(|v| v.as_int())
                .ok_or_else(|| {
                    TrainError::InvalidParameter(format!(
                        "config missing 'layer_{i}_width' Int key"
                    ))
                })?;

            if width < 1 {
                return Err(TrainError::InvalidParameter(format!(
                    "layer {i} width {width} must be ≥ 1"
                )));
            }

            let activation = cfg
                .get(&format!("layer_{i}_activation"))
                .and_then(|v| v.as_string())
                .ok_or_else(|| {
                    TrainError::InvalidParameter(format!(
                        "config missing 'layer_{i}_activation' String key"
                    ))
                })?
                .to_string();

            layers.push(LayerSpec {
                op,
                width: width as usize,
                activation,
            });
        }

        Ok(Architecture { layers })
    }
}

// ─── ArchSearchSpace ────────────────────────────────────────────────────────

/// Defines the search space over neural architectures.
///
/// Constrains depth range, layer width choices, activation functions, and
/// operation types that can appear in any sampled architecture.
#[derive(Debug, Clone)]
pub struct ArchSearchSpace {
    /// Minimum number of layers (inclusive, ≥ 1).
    pub min_depth: usize,
    /// Maximum number of layers (inclusive, ≥ min_depth).
    pub max_depth: usize,
    /// Allowed width (hidden-unit count) options per layer.
    pub width_options: Vec<usize>,
    /// Allowed activation function names per layer.
    pub activation_options: Vec<String>,
    /// Allowed operation type names per layer.
    pub op_options: Vec<String>,
}

impl ArchSearchSpace {
    /// Construct a validated [`ArchSearchSpace`].
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidParameter`] if:
    /// - `min_depth` < 1
    /// - `max_depth` < `min_depth`
    /// - any of the option vecs is empty
    pub fn new(
        min_depth: usize,
        max_depth: usize,
        width_options: Vec<usize>,
        activation_options: Vec<String>,
        op_options: Vec<String>,
    ) -> TrainResult<Self> {
        if min_depth < 1 {
            return Err(TrainError::InvalidParameter(
                "min_depth must be ≥ 1".to_string(),
            ));
        }
        if max_depth < min_depth {
            return Err(TrainError::InvalidParameter(format!(
                "max_depth ({max_depth}) must be ≥ min_depth ({min_depth})"
            )));
        }
        if width_options.is_empty() {
            return Err(TrainError::InvalidParameter(
                "width_options must be non-empty".to_string(),
            ));
        }
        if activation_options.is_empty() {
            return Err(TrainError::InvalidParameter(
                "activation_options must be non-empty".to_string(),
            ));
        }
        if op_options.is_empty() {
            return Err(TrainError::InvalidParameter(
                "op_options must be non-empty".to_string(),
            ));
        }
        Ok(Self {
            min_depth,
            max_depth,
            width_options,
            activation_options,
            op_options,
        })
    }
}
