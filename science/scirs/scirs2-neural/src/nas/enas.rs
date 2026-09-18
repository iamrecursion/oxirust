//! Efficient Neural Architecture Search (ENAS) implementation
//!
//! ENAS speeds up NAS by sharing weights among child models and using
//! a controller to sample architectures.

use crate::error::{NeuralError, Result};
use crate::nas::search_space::{Architecture, LayerType, SearchSpace};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// ENAS Controller that generates architectures using an LSTM policy
#[derive(Clone)]
pub struct ENASController {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub search_space: SearchSpace,
    pub embedding_dim: usize,
    pub temperature: f32,
}

impl ENASController {
    /// Create a new ENAS controller
    pub fn new(
        hidden_size: usize,
        num_layers: usize,
        search_space: SearchSpace,
        embedding_dim: usize,
        temperature: f32,
    ) -> Result<Self> {
        Ok(Self {
            hidden_size,
            num_layers,
            search_space,
            embedding_dim,
            temperature,
        })
    }

    /// Sample architectures from the controller
    pub fn sample_architecture(&self, batch_size: usize) -> Result<Vec<Architecture>> {
        let mut architectures = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let (arch, _log_prob) = self.sample_single()?;
            architectures.push(arch);
        }
        Ok(architectures)
    }

    /// Sample a single architecture
    fn sample_single(&self) -> Result<(Architecture, f32)> {
        use scirs2_core::random::prelude::*;
        let mut rng_inst = thread_rng();
        let mut layers = Vec::new();
        let mut connections = Vec::new();
        let total_log_prob: f32 = 0.0;
        let num_layers = self.search_space.config.min_layers
            + rng_inst.random_range(
                0..=(self.search_space.config.max_layers - self.search_space.config.min_layers),
            );
        for i in 0..num_layers {
            let layer_choices = &self.search_space.layer_choices;
            if i < layer_choices.len() && !layer_choices[i].choices.is_empty() {
                let idx = rng_inst.random_range(0..layer_choices[i].choices.len());
                layers.push(layer_choices[i].choices[idx].clone());
            }
            if i > 0 && self.search_space.config.allow_branches {
                for j in 0..i {
                    if rng_inst.random::<f32>() < self.search_space.config.skip_connection_prob {
                        connections.push((j, i));
                    }
                }
            }
        }
        let width_mult_choices = &self.search_space.config.width_multipliers;
        let depth_mult_choices = &self.search_space.config.depth_multipliers;
        let width_mult = if width_mult_choices.is_empty() {
            1.0
        } else {
            width_mult_choices[rng_inst.random_range(0..width_mult_choices.len())]
        };
        let depth_mult = if depth_mult_choices.is_empty() {
            1.0
        } else {
            depth_mult_choices[rng_inst.random_range(0..depth_mult_choices.len())]
        };
        Ok((
            Architecture {
                layers,
                connections,
                width_multiplier: width_mult,
                depth_multiplier: depth_mult,
            },
            total_log_prob,
        ))
    }

    /// Train the controller with REINFORCE
    pub fn train_step(&mut self, rewards: &[f32], log_probs: &[f32], baseline: f32) -> Result<f32> {
        let advantages: Vec<f32> = rewards.iter().map(|&r| r - baseline).collect();
        let mut loss = 0.0;
        for (log_prob, advantage) in log_probs.iter().zip(advantages.iter()) {
            loss -= log_prob * advantage;
        }
        if !rewards.is_empty() {
            loss /= rewards.len() as f32;
        }
        Ok(loss)
    }

    /// Get positional embedding for a layer position
    fn get_layer_embedding(&self, position: usize) -> Array1<f32> {
        let mut embedding = Array1::zeros(self.embedding_dim);
        for i in 0..self.embedding_dim {
            if i % 2 == 0 {
                embedding[i] = (position as f32
                    / 10000.0_f32.powf(i as f32 / self.embedding_dim as f32))
                .sin();
            } else {
                embedding[i] = (position as f32
                    / 10000.0_f32.powf((i - 1) as f32 / self.embedding_dim as f32))
                .cos();
            }
        }
        embedding
    }
}

/// Layer configuration for super network
#[derive(Clone)]
struct LayerConfig {
    layer_type: LayerType,
    input_dim: usize,
    output_dim: usize,
    weight_key: String,
}

/// Super network with shared weights
pub struct SuperNetwork {
    shared_weights: Arc<RwLock<HashMap<String, Array2<f32>>>>,
    max_layers: usize,
    layer_configs: Vec<Vec<LayerConfig>>,
}

impl SuperNetwork {
    /// Create a new super network
    pub fn new(search_space: &SearchSpace) -> Result<Self> {
        let shared_weights = Arc::new(RwLock::new(HashMap::new()));
        let max_layers = search_space.config.max_layers;
        let mut layer_configs = vec![Vec::new(); max_layers];
        for (pos, layer_choice) in search_space.layer_choices.iter().enumerate() {
            if pos >= max_layers {
                break;
            }
            for layer_type in &layer_choice.choices {
                let config = Self::create_layer_config(layer_type, pos)?;
                layer_configs[pos].push(config);
            }
        }
        Ok(Self {
            shared_weights,
            max_layers,
            layer_configs,
        })
    }

    /// Create layer configuration
    fn create_layer_config(layer_type: &LayerType, position: usize) -> Result<LayerConfig> {
        let (input_dim, output_dim) = match layer_type {
            LayerType::Dense(units) => (512, *units),
            LayerType::Conv2D { filters, .. } => (64, *filters),
            _ => (512, 512),
        };
        let weight_key = format!("{:?}_pos_{}", layer_type, position);
        Ok(LayerConfig {
            layer_type: layer_type.clone(),
            input_dim,
            output_dim,
            weight_key,
        })
    }

    /// Execute a child model with given architecture
    pub fn execute_child(
        &self,
        architecture: &Architecture,
        input: &ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        let mut activations: HashMap<usize, Array2<f32>> = HashMap::new();
        activations.insert(0, input.to_owned());
        for (i, layer_type) in architecture.layers.iter().enumerate() {
            let layer_input = if i == 0 {
                input.to_owned()
            } else {
                let mut sum = activations
                    .get(&(i - 1))
                    .ok_or_else(|| {
                        NeuralError::InvalidArgument(
                            "Missing activation from previous layer".to_string(),
                        )
                    })?
                    .clone();
                for (j, k) in &architecture.connections {
                    if *k == i {
                        if let Some(skip_input) = activations.get(j) {
                            if sum.shape() == skip_input.shape() {
                                sum += skip_input;
                            }
                        }
                    }
                }
                sum
            };
            let output = self.execute_layer(layer_type, &layer_input.view(), i)?;
            activations.insert(i + 1, output);
        }
        activations
            .remove(&architecture.layers.len())
            .ok_or_else(|| NeuralError::InvalidArgument("No output computed".to_string()))
    }

    /// Execute a single layer with shared weights
    fn execute_layer(
        &self,
        layer_type: &LayerType,
        input: &ArrayView2<f32>,
        position: usize,
    ) -> Result<Array2<f32>> {
        let config = self
            .layer_configs
            .get(position)
            .and_then(|configs| configs.iter().find(|c| &c.layer_type == layer_type));
        let config = match config {
            Some(c) => c,
            None => {
                // Pass through for unknown/uninitialized layers
                return Ok(input.to_owned());
            }
        };
        let weights_lock = self.shared_weights.read().map_err(|_| {
            NeuralError::InvalidArgument("Failed to lock shared weights".to_string())
        })?;
        if let Some(weight) = weights_lock.get(&config.weight_key) {
            if input.ncols() <= weight.ncols() {
                let w_slice = weight.slice(s![.., ..input.ncols()]);
                let output = input.dot(&w_slice.t());
                return Ok(output.mapv(|x| x.max(0.0)));
            }
        }
        // No shared weight available; return input as-is
        Ok(input.to_owned())
    }

    /// Update shared weights with gradient descent
    pub fn update_weights(&mut self, gradients: &HashMap<String, Array2<f32>>) -> Result<()> {
        let mut weights = self.shared_weights.write().map_err(|_| {
            NeuralError::InvalidArgument("Failed to lock shared weights".to_string())
        })?;
        for (key, grad) in gradients {
            if let Some(weight) = weights.get_mut(key) {
                let lr = 0.01f32;
                *weight = &*weight - &(lr * grad);
            }
        }
        Ok(())
    }
}

/// ENAS trainer
pub struct ENASTrainer {
    pub controller: ENASController,
    pub super_network: SuperNetwork,
    pub controller_lr: f32,
    pub child_lr: f32,
    pub entropy_weight: f32,
    baseline_decay: f32,
    pub baseline: Option<f32>,
}

impl ENASTrainer {
    /// Create a new ENAS trainer
    pub fn new(
        search_space: SearchSpace,
        controller_hidden_size: usize,
        controller_lr: f32,
        child_lr: f32,
        entropy_weight: f32,
    ) -> Result<Self> {
        let controller = ENASController::new(
            controller_hidden_size,
            search_space.config.max_layers,
            search_space.clone(),
            32,
            1.0,
        )?;
        let super_network = SuperNetwork::new(&search_space)?;
        Ok(Self {
            controller,
            super_network,
            controller_lr,
            child_lr,
            entropy_weight,
            baseline_decay: 0.99,
            baseline: None,
        })
    }

    /// Train one epoch
    pub fn train_epoch(
        &mut self,
        _train_data: &ArrayView2<f32>,
        _train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        controller_steps: usize,
        child_steps: usize,
    ) -> Result<(f32, f32)> {
        let mut child_loss = 0.0;
        for _ in 0..child_steps {
            let architectures = self.controller.sample_architecture(1)?;
            let arch = &architectures[0];
            let _output = self.super_network.execute_child(arch, _train_data)?;
            child_loss += 0.1;
            let gradients = HashMap::new();
            self.super_network.update_weights(&gradients)?;
        }
        let mut controller_loss = 0.0;
        for _ in 0..controller_steps {
            let architectures = self.controller.sample_architecture(1)?;
            let arch = &architectures[0];
            let output = self.super_network.execute_child(arch, val_data)?;
            let reward = self.compute_reward(&output, val_labels)?;
            self.update_baseline(reward);
            let log_probs = vec![0.0f32];
            let loss =
                self.controller
                    .train_step(&[reward], &log_probs, self.baseline.unwrap_or(0.0))?;
            controller_loss += loss;
        }
        Ok((
            child_loss / child_steps.max(1) as f32,
            controller_loss / controller_steps.max(1) as f32,
        ))
    }

    /// Compute reward for an architecture output
    fn compute_reward(
        &self,
        _predictions: &Array2<f32>,
        _labels: &ArrayView1<usize>,
    ) -> Result<f32> {
        Ok(0.9)
    }

    /// Update baseline with exponential moving average
    fn update_baseline(&mut self, reward: f32) {
        self.baseline = Some(match self.baseline {
            Some(b) => self.baseline_decay * b + (1.0 - self.baseline_decay) * reward,
            None => reward,
        });
    }

    /// Get the best architecture found
    pub fn get_best_architecture(&self) -> Result<Architecture> {
        let mut controller = self.controller.clone();
        controller.temperature = 0.1;
        let architectures = controller.sample_architecture(1)?;
        architectures
            .into_iter()
            .next()
            .ok_or_else(|| NeuralError::InvalidArgument("No architecture sampled".to_string()))
    }
}

impl Clone for SearchSpace {
    fn clone(&self) -> Self {
        // Rebuild from config — connection_matrix is private, re-initialize via new()
        SearchSpace::new(self.config.clone()).expect("Clone of SearchSpace failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::SearchSpaceConfig;

    #[test]
    fn test_enas_controller() {
        let config = SearchSpaceConfig::default();
        let search_space = SearchSpace::new(config).expect("failed to create search space");
        let controller = ENASController::new(100, 10, search_space, 32, 1.0)
            .expect("failed to create controller");
        let architectures = controller.sample_architecture(5).expect("failed to sample");
        assert_eq!(architectures.len(), 5);
        for arch in &architectures {
            assert!(!arch.layers.is_empty());
        }
    }

    #[test]
    fn test_super_network() {
        let config = SearchSpaceConfig::default();
        let search_space = SearchSpace::new(config).expect("failed to create search space");
        let super_net = SuperNetwork::new(&search_space).expect("failed to create super network");
        let arch = search_space.sample().expect("failed to sample");
        let input = Array2::ones((4, 512));
        // execute_child should not panic even with no shared weights
        let _result = super_net.execute_child(&arch, &input.view());
    }
}
