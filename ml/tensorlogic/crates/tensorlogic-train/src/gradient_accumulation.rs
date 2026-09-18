//! Gradient accumulation for effective large-batch training.
//!
//! Enables training with larger effective batch sizes than memory allows
//! by accumulating gradients over multiple micro-batches before applying
//! an optimizer update.

use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during gradient accumulation.
#[derive(Debug, Error)]
pub enum AccumulationError {
    /// Gradient shape mismatch when accumulating into a buffer.
    #[error("Gradient shape mismatch for '{name}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Parameter name (may be empty for unnamed buffers).
        name: String,
        /// Expected shape of the gradient.
        expected: Vec<usize>,
        /// Actual shape of the gradient provided.
        got: Vec<usize>,
    },
    /// Attempted to retrieve gradients when none have been accumulated.
    #[error("No gradients accumulated")]
    NoGradients,
    /// The accumulator has already reached the configured number of micro-batches.
    #[error("Accumulator already full ({0} micro-batches)")]
    AccumulatorFull(usize),
}

/// Configuration for gradient accumulation.
#[derive(Debug, Clone)]
pub struct AccumulationConfig {
    /// Number of micro-batches to accumulate before triggering an update.
    pub accumulation_steps: usize,
    /// Whether to normalize (average) gradients across micro-batches.
    pub normalize: bool,
    /// Maximum gradient norm for clipping (None = no clipping).
    pub max_grad_norm: Option<f64>,
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        AccumulationConfig {
            accumulation_steps: 4,
            normalize: true,
            max_grad_norm: None,
        }
    }
}

impl AccumulationConfig {
    /// Create a new config with the given number of accumulation steps.
    /// Clamps to a minimum of 1.
    pub fn new(steps: usize) -> Self {
        AccumulationConfig {
            accumulation_steps: steps.max(1),
            ..Default::default()
        }
    }

    /// Set whether to normalize (average) gradients.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set maximum gradient norm for clipping.
    pub fn with_max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = Some(norm);
        self
    }

    /// Compute the effective batch size given the micro-batch size.
    pub fn effective_batch_size(&self, micro_batch_size: usize) -> usize {
        micro_batch_size * self.accumulation_steps
    }
}

/// A single gradient buffer for one parameter.
#[derive(Debug, Clone)]
pub struct GradientBuffer {
    /// Accumulated gradient values (flattened).
    pub data: Vec<f64>,
    /// Shape of the gradient tensor.
    pub shape: Vec<usize>,
    /// Number of micro-batches accumulated so far.
    pub accumulated_count: usize,
}

impl GradientBuffer {
    /// Create a new gradient buffer initialized to zeros.
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        GradientBuffer {
            data: vec![0.0; size],
            shape,
            accumulated_count: 0,
        }
    }

    /// Add a micro-batch gradient to the buffer.
    pub fn accumulate(&mut self, grad: &[f64]) -> Result<(), AccumulationError> {
        if grad.len() != self.data.len() {
            return Err(AccumulationError::ShapeMismatch {
                name: String::new(),
                expected: self.shape.clone(),
                got: vec![grad.len()],
            });
        }
        for (acc, &g) in self.data.iter_mut().zip(grad.iter()) {
            *acc += g;
        }
        self.accumulated_count += 1;
        Ok(())
    }

    /// Get the accumulated gradient, optionally normalized by the count.
    pub fn get(&self, normalize: bool) -> Vec<f64> {
        if normalize && self.accumulated_count > 0 {
            let scale = 1.0 / self.accumulated_count as f64;
            self.data.iter().map(|&v| v * scale).collect()
        } else {
            self.data.clone()
        }
    }

    /// Compute the L2 norm of the accumulated gradient.
    pub fn l2_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Reset the buffer to zeros.
    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.accumulated_count = 0;
    }
}

/// Gradient accumulator managing multiple parameter gradients.
///
/// Provides micro-batching support by accumulating gradients across
/// multiple forward/backward passes before applying an optimizer step.
pub struct GradientAccumulator {
    config: AccumulationConfig,
    buffers: HashMap<String, GradientBuffer>,
    total_micro_batches: usize,
    total_updates: usize,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator with the given configuration.
    pub fn new(config: AccumulationConfig) -> Self {
        GradientAccumulator {
            config,
            buffers: HashMap::new(),
            total_micro_batches: 0,
            total_updates: 0,
        }
    }

    /// Register a parameter with its gradient shape.
    ///
    /// If the parameter is already registered, this is a no-op.
    pub fn register(&mut self, name: impl Into<String>, shape: Vec<usize>) {
        let name = name.into();
        self.buffers
            .entry(name)
            .or_insert_with(|| GradientBuffer::new(shape));
    }

    /// Accumulate a gradient for a named parameter.
    ///
    /// Returns an error if the parameter has not been registered or if
    /// the gradient size does not match the registered shape.
    pub fn accumulate(&mut self, name: &str, grad: &[f64]) -> Result<(), AccumulationError> {
        if let Some(buf) = self.buffers.get_mut(name) {
            if buf.accumulated_count >= self.config.accumulation_steps {
                return Err(AccumulationError::AccumulatorFull(
                    self.config.accumulation_steps,
                ));
            }
            buf.accumulate(grad).map_err(|e| match e {
                AccumulationError::ShapeMismatch { expected, got, .. } => {
                    AccumulationError::ShapeMismatch {
                        name: name.to_string(),
                        expected,
                        got,
                    }
                }
                other => other,
            })
        } else {
            Err(AccumulationError::NoGradients)
        }
    }

    /// Check if enough micro-batches have been accumulated to trigger an update.
    pub fn should_update(&self) -> bool {
        self.buffers
            .values()
            .any(|b| b.accumulated_count >= self.config.accumulation_steps)
    }

    /// Get all accumulated gradients, optionally normalized and clipped.
    pub fn get_gradients(&self) -> Result<HashMap<String, Vec<f64>>, AccumulationError> {
        if self.buffers.is_empty() {
            return Err(AccumulationError::NoGradients);
        }
        let mut grads: HashMap<String, Vec<f64>> = self
            .buffers
            .iter()
            .map(|(name, buf)| (name.clone(), buf.get(self.config.normalize)))
            .collect();

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_grad_norm {
            let total_norm: f64 = grads
                .values()
                .flat_map(|g| g.iter())
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            if total_norm > max_norm {
                let scale = max_norm / total_norm;
                for grad in grads.values_mut() {
                    for v in grad.iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }
        Ok(grads)
    }

    /// Reset all buffers after an update step.
    pub fn reset(&mut self) {
        for buf in self.buffers.values_mut() {
            buf.reset();
        }
        self.total_updates += 1;
    }

    /// Accumulate a full micro-batch of gradients, returning `true` if an
    /// update should now be applied.
    pub fn step(
        &mut self,
        gradients: &HashMap<String, Vec<f64>>,
    ) -> Result<bool, AccumulationError> {
        for (name, grad) in gradients {
            self.accumulate(name, grad)?;
        }
        self.total_micro_batches += 1;
        Ok(self.should_update())
    }

    /// Get statistics about the accumulation state.
    pub fn stats(&self) -> AccumulationStats {
        AccumulationStats {
            total_micro_batches: self.total_micro_batches,
            total_updates: self.total_updates,
            accumulation_steps: self.config.accumulation_steps,
            registered_params: self.buffers.len(),
            total_param_elements: self.buffers.values().map(|b| b.data.len()).sum(),
        }
    }
}

/// Statistics from gradient accumulation.
#[derive(Debug, Clone)]
pub struct AccumulationStats {
    /// Total number of micro-batches processed.
    pub total_micro_batches: usize,
    /// Total number of optimizer updates applied.
    pub total_updates: usize,
    /// Configured number of accumulation steps.
    pub accumulation_steps: usize,
    /// Number of registered parameters.
    pub registered_params: usize,
    /// Total number of scalar gradient elements across all parameters.
    pub total_param_elements: usize,
}

impl AccumulationStats {
    /// The effective batch size multiplier (same as accumulation_steps).
    pub fn effective_batch_multiplier(&self) -> usize {
        self.accumulation_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AccumulationConfig::default();
        assert_eq!(config.accumulation_steps, 4);
        assert!(config.normalize);
        assert!(config.max_grad_norm.is_none());
    }

    #[test]
    fn test_config_effective_batch_size() {
        let config = AccumulationConfig::new(4);
        assert_eq!(config.effective_batch_size(32), 128);
    }

    #[test]
    fn test_buffer_new() {
        let buf = GradientBuffer::new(vec![3, 4]);
        assert_eq!(buf.data.len(), 12);
        assert!(buf.data.iter().all(|&v| v == 0.0));
        assert_eq!(buf.accumulated_count, 0);
    }

    #[test]
    fn test_buffer_accumulate() {
        let mut buf = GradientBuffer::new(vec![3]);
        let grad = vec![1.0, 2.0, 3.0];
        buf.accumulate(&grad).expect("accumulate should succeed");
        assert_eq!(buf.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.accumulated_count, 1);

        buf.accumulate(&grad)
            .expect("second accumulate should succeed");
        assert_eq!(buf.data, vec![2.0, 4.0, 6.0]);
        assert_eq!(buf.accumulated_count, 2);
    }

    #[test]
    fn test_buffer_accumulate_shape_mismatch() {
        let mut buf = GradientBuffer::new(vec![3]);
        let grad = vec![1.0, 2.0];
        let result = buf.accumulate(&grad);
        assert!(result.is_err());
        match result {
            Err(AccumulationError::ShapeMismatch { .. }) => {}
            _ => panic!("expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_buffer_get_normalized() {
        let mut buf = GradientBuffer::new(vec![2]);
        buf.accumulate(&[2.0, 4.0]).expect("accumulate");
        buf.accumulate(&[6.0, 8.0]).expect("accumulate");
        let normalized = buf.get(true);
        assert_eq!(normalized, vec![4.0, 6.0]); // (2+6)/2=4, (4+8)/2=6
    }

    #[test]
    fn test_buffer_get_unnormalized() {
        let mut buf = GradientBuffer::new(vec![2]);
        buf.accumulate(&[2.0, 4.0]).expect("accumulate");
        buf.accumulate(&[6.0, 8.0]).expect("accumulate");
        let raw = buf.get(false);
        assert_eq!(raw, vec![8.0, 12.0]); // 2+6=8, 4+8=12
    }

    #[test]
    fn test_buffer_l2_norm() {
        let mut buf = GradientBuffer::new(vec![2]);
        buf.accumulate(&[3.0, 4.0]).expect("accumulate");
        let norm = buf.l2_norm();
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_buffer_reset() {
        let mut buf = GradientBuffer::new(vec![3]);
        buf.accumulate(&[1.0, 2.0, 3.0]).expect("accumulate");
        assert_eq!(buf.accumulated_count, 1);
        buf.reset();
        assert!(buf.data.iter().all(|&v| v == 0.0));
        assert_eq!(buf.accumulated_count, 0);
    }

    #[test]
    fn test_accumulator_register() {
        let mut acc = GradientAccumulator::new(AccumulationConfig::default());
        acc.register("weight", vec![3, 4]);
        assert_eq!(acc.buffers.len(), 1);
        assert!(acc.buffers.contains_key("weight"));
    }

    #[test]
    fn test_accumulator_accumulate() {
        let mut acc = GradientAccumulator::new(AccumulationConfig::default());
        acc.register("w", vec![2]);
        acc.accumulate("w", &[1.0, 2.0])
            .expect("accumulate should succeed");
        let buf = acc.buffers.get("w").expect("buffer should exist");
        assert_eq!(buf.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_accumulator_should_update() {
        let config = AccumulationConfig::new(2);
        let mut acc = GradientAccumulator::new(config);
        acc.register("w", vec![2]);
        assert!(!acc.should_update());
        acc.accumulate("w", &[1.0, 1.0]).expect("accumulate");
        assert!(!acc.should_update());
        acc.accumulate("w", &[1.0, 1.0]).expect("accumulate");
        assert!(acc.should_update());
    }

    #[test]
    fn test_accumulator_get_gradients() {
        let config = AccumulationConfig::new(2).with_normalize(true);
        let mut acc = GradientAccumulator::new(config);
        acc.register("w", vec![2]);
        acc.accumulate("w", &[2.0, 4.0]).expect("accumulate");
        acc.accumulate("w", &[6.0, 8.0]).expect("accumulate");
        let grads = acc.get_gradients().expect("get_gradients");
        let w_grad = grads.get("w").expect("w gradient");
        assert_eq!(w_grad, &vec![4.0, 6.0]);
    }

    #[test]
    fn test_accumulator_grad_clipping() {
        let config = AccumulationConfig::new(1)
            .with_normalize(false)
            .with_max_grad_norm(5.0);
        let mut acc = GradientAccumulator::new(config);
        acc.register("w", vec![2]);
        // gradient [30, 40] has norm 50, clip to 5 => scale by 5/50 = 0.1
        acc.accumulate("w", &[30.0, 40.0]).expect("accumulate");
        let grads = acc.get_gradients().expect("get_gradients");
        let w_grad = grads.get("w").expect("w gradient");
        assert!((w_grad[0] - 3.0).abs() < 1e-10);
        assert!((w_grad[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_accumulator_reset() {
        let config = AccumulationConfig::new(2);
        let mut acc = GradientAccumulator::new(config);
        acc.register("w", vec![2]);
        acc.accumulate("w", &[1.0, 2.0]).expect("accumulate");
        acc.reset();
        let buf = acc.buffers.get("w").expect("buffer");
        assert!(buf.data.iter().all(|&v| v == 0.0));
        assert_eq!(buf.accumulated_count, 0);
        assert_eq!(acc.total_updates, 1);
    }

    #[test]
    fn test_accumulator_step() {
        let config = AccumulationConfig::new(2);
        let mut acc = GradientAccumulator::new(config);
        acc.register("w", vec![2]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), vec![1.0, 1.0]);

        let should = acc.step(&grads).expect("step 1");
        assert!(!should);
        let should = acc.step(&grads).expect("step 2");
        assert!(should);
    }

    #[test]
    fn test_accumulator_stats() {
        let config = AccumulationConfig::new(3);
        let mut acc = GradientAccumulator::new(config);
        acc.register("a", vec![2, 3]);
        acc.register("b", vec![4]);

        let stats = acc.stats();
        assert_eq!(stats.total_micro_batches, 0);
        assert_eq!(stats.total_updates, 0);
        assert_eq!(stats.accumulation_steps, 3);
        assert_eq!(stats.registered_params, 2);
        assert_eq!(stats.total_param_elements, 10); // 6 + 4
        assert_eq!(stats.effective_batch_multiplier(), 3);
    }

    #[test]
    fn test_accumulator_empty_no_gradients() {
        let acc = GradientAccumulator::new(AccumulationConfig::default());
        let result = acc.get_gradients();
        assert!(result.is_err());
        match result {
            Err(AccumulationError::NoGradients) => {}
            _ => panic!("expected NoGradients error"),
        }
    }
}
