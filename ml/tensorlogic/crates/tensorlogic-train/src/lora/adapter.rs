//! Multi-layer LoRA adapter managing named LoRA layers.

use indexmap::IndexMap;

use super::config::LoraConfig;
use super::error::{LoraError, LoraResult};
use super::layer::LoraLayer;

/// Per-layer statistics included in the adapter summary.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub name: String,
    pub d: usize,
    pub k: usize,
    pub rank: usize,
    pub trainable_params: usize,
    pub total_params: usize,
    pub compression_ratio: f64,
    pub merged: bool,
}

/// Summary of the entire LoRA adapter.
#[derive(Debug, Clone)]
pub struct LoraAdapterSummary {
    pub layers: Vec<LayerStats>,
    pub total_trainable: usize,
    pub total_params: usize,
}

/// Manages multiple named [`LoraLayer`]s that share a single [`LoraConfig`].
pub struct LoraAdapter {
    config: LoraConfig,
    layers: IndexMap<String, LoraLayer>,
}

impl LoraAdapter {
    pub fn new(config: LoraConfig) -> Self {
        Self {
            config,
            layers: IndexMap::new(),
        }
    }

    /// Wrap `base_weight` in a new LoRA layer registered under `name`.
    pub fn add_layer(&mut self, name: &str, base_weight: Vec<Vec<f64>>) -> LoraResult<()> {
        let layer = LoraLayer::new(base_weight, self.config.clone())?;
        self.layers.insert(name.to_string(), layer);
        Ok(())
    }

    /// Forward pass through the named layer.
    pub fn forward(&mut self, name: &str, input: &[Vec<f64>]) -> LoraResult<Vec<Vec<f64>>> {
        let layer = self
            .layers
            .get_mut(name)
            .ok_or_else(|| LoraError::DimensionMismatch {
                expected: format!("layer '{name}' exists"),
                got: "not found".into(),
            })?;
        layer.forward(input)
    }

    /// Merge all layers.
    pub fn merge_all(&mut self) -> LoraResult<()> {
        for layer in self.layers.values_mut() {
            if !layer.merged {
                layer.merge()?;
            }
        }
        Ok(())
    }

    /// Unmerge all layers.
    pub fn unmerge_all(&mut self) -> LoraResult<()> {
        for layer in self.layers.values_mut() {
            if layer.merged {
                layer.unmerge()?;
            }
        }
        Ok(())
    }

    /// Sum of trainable params across all layers.
    pub fn total_trainable_params(&self) -> usize {
        self.layers.values().map(|l| l.trainable_params()).sum()
    }

    /// Build a summary with per-layer statistics.
    pub fn summary(&self) -> LoraAdapterSummary {
        let mut layers = Vec::with_capacity(self.layers.len());
        for (name, layer) in &self.layers {
            let d = layer.base_weight.len();
            let k = layer.base_weight[0].len();
            layers.push(LayerStats {
                name: name.clone(),
                d,
                k,
                rank: layer.config.rank,
                trainable_params: layer.trainable_params(),
                total_params: layer.total_params(),
                compression_ratio: layer.compression_ratio(),
                merged: layer.merged,
            });
        }
        let total_trainable = layers.iter().map(|s| s.trainable_params).sum();
        let total_params = layers.iter().map(|s| s.total_params).sum();
        LoraAdapterSummary {
            layers,
            total_trainable,
            total_params,
        }
    }
}
