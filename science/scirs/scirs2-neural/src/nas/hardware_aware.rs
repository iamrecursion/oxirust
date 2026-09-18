//! Hardware-Aware Neural Architecture Search
//!
//! This module provides hardware-aware optimization capabilities that consider
//! deployment constraints such as latency, memory usage, energy consumption,
//! and platform-specific optimizations.

use crate::error::Result;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding,
    search_space::{Architecture, LayerType},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Hardware platform types
#[derive(Debug, Clone, PartialEq)]
pub enum HardwarePlatform {
    /// Desktop/Server CPU
    CPU,
    /// NVIDIA GPU with CUDA
    GPU,
    /// Mobile ARM CPU
    MobileARM,
    /// Edge TPU
    EdgeTPU,
    /// FPGA
    FPGA,
    /// Custom accelerator
    Custom(String),
}

/// Quantization support levels
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationSupport {
    None,
    Int8,
    Float16,
    Mixed,
    Custom(Vec<String>),
}

/// Hardware constraints for deployment
#[derive(Debug, Clone)]
pub struct HardwareConstraints {
    pub platform: HardwarePlatform,
    pub max_latency_ms: Option<f64>,
    pub max_memory_mb: Option<f64>,
    pub max_energy_mj: Option<f64>,
    pub max_model_size_mb: Option<f64>,
    pub min_throughput: Option<f64>,
    pub compute_units: usize,
    pub memory_bandwidth_gbps: Option<f64>,
    pub quantization_support: QuantizationSupport,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            platform: HardwarePlatform::CPU,
            max_latency_ms: Some(100.0),
            max_memory_mb: Some(512.0),
            max_energy_mj: Some(100.0),
            max_model_size_mb: Some(50.0),
            min_throughput: Some(10.0),
            compute_units: 4,
            memory_bandwidth_gbps: Some(25.6),
            quantization_support: QuantizationSupport::Int8,
        }
    }
}

/// Latency prediction model
pub struct LatencyPredictor {
    operation_latencies: HashMap<String, f64>,
    memory_costs: HashMap<String, f64>,
    parallelization_factors: HashMap<String, f64>,
    platform: HardwarePlatform,
}

impl LatencyPredictor {
    /// Create a new latency predictor for a platform
    pub fn new(platform: HardwarePlatform) -> Self {
        let mut predictor = Self {
            operation_latencies: HashMap::new(),
            memory_costs: HashMap::new(),
            parallelization_factors: HashMap::new(),
            platform,
        };
        predictor.initialize_platform_characteristics();
        predictor
    }

    fn initialize_platform_characteristics(&mut self) {
        let (dense_lat, conv_lat, pool_lat, act_lat, bn_lat, wl_cost, as_cost, dense_par, conv_par) =
            match self.platform {
                HardwarePlatform::CPU => (0.1, 0.5, 0.05, 0.01, 0.02, 0.001, 0.0005, 0.8, 0.9),
                HardwarePlatform::GPU => {
                    (0.02, 0.1, 0.01, 0.005, 0.01, 0.0001, 0.00005, 0.95, 0.98)
                }
                HardwarePlatform::MobileARM => (0.3, 1.0, 0.1, 0.02, 0.05, 0.002, 0.001, 0.6, 0.7),
                HardwarePlatform::EdgeTPU => {
                    (0.05, 0.2, 0.02, 0.01, 0.015, 0.0002, 0.0001, 0.9, 0.95)
                }
                _ => (0.2, 0.8, 0.08, 0.015, 0.04, 0.0015, 0.00075, 0.75, 0.85),
            };
        self.operation_latencies
            .insert("dense".to_string(), dense_lat);
        self.operation_latencies
            .insert("conv2d".to_string(), conv_lat);
        self.operation_latencies
            .insert("pooling".to_string(), pool_lat);
        self.operation_latencies
            .insert("activation".to_string(), act_lat);
        self.operation_latencies
            .insert("batchnorm".to_string(), bn_lat);
        self.memory_costs.insert("weight_load".to_string(), wl_cost);
        self.memory_costs
            .insert("activation_store".to_string(), as_cost);
        self.parallelization_factors
            .insert("dense".to_string(), dense_par);
        self.parallelization_factors
            .insert("conv2d".to_string(), conv_par);
    }

    /// Predict latency for an architecture
    pub fn predict_latency(
        &self,
        architecture: &Architecture,
        input_shape: &[usize],
    ) -> Result<f64> {
        let mut total_latency = 0.0;
        let mut current_shape = input_shape.to_vec();
        for layer in &architecture.layers {
            let layer_latency = self.predict_layer_latency(layer, &current_shape)?;
            total_latency += layer_latency;
            current_shape = self.compute_output_shape(layer, &current_shape)?;
        }
        total_latency += self.compute_initialization_overhead(architecture)?;
        Ok(total_latency)
    }

    fn predict_layer_latency(&self, layer: &LayerType, input_shape: &[usize]) -> Result<f64> {
        let base_latency = match layer {
            LayerType::Dense(units) => {
                let input_size: usize = input_shape.iter().product();
                let ops = input_size * units;
                let base = self
                    .operation_latencies
                    .get("dense")
                    .copied()
                    .unwrap_or(0.1);
                base * (ops as f64 / 1e6)
            }
            LayerType::Conv2D {
                filters,
                kernel_size,
                stride,
            } => {
                if input_shape.len() < 3 {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Conv2D requires 3D input".to_string(),
                    ));
                }
                let h = input_shape[0];
                let w = input_shape[1];
                let c = input_shape[2];
                let output_h = (h.saturating_sub(kernel_size.0)) / stride.0 + 1;
                let output_w = (w.saturating_sub(kernel_size.1)) / stride.1 + 1;
                let ops = output_h * output_w * filters * kernel_size.0 * kernel_size.1 * c;
                let base = self
                    .operation_latencies
                    .get("conv2d")
                    .copied()
                    .unwrap_or(0.5);
                base * (ops as f64 / 1e6)
            }
            LayerType::MaxPool2D { pool_size, stride }
            | LayerType::AvgPool2D { pool_size, stride } => {
                if input_shape.len() < 3 {
                    return Ok(0.0);
                }
                let h = input_shape[0];
                let w = input_shape[1];
                let c = input_shape[2];
                let output_h = (h.saturating_sub(pool_size.0)) / stride.0 + 1;
                let output_w = (w.saturating_sub(pool_size.1)) / stride.1 + 1;
                let ops = output_h * output_w * c * pool_size.0 * pool_size.1;
                let base = self
                    .operation_latencies
                    .get("pooling")
                    .copied()
                    .unwrap_or(0.05);
                base * (ops as f64 / 1e6)
            }
            LayerType::Activation(_) => {
                let ops: usize = input_shape.iter().product();
                let base = self
                    .operation_latencies
                    .get("activation")
                    .copied()
                    .unwrap_or(0.01);
                base * (ops as f64 / 1e6)
            }
            LayerType::BatchNorm | LayerType::LayerNorm => {
                let ops: usize = input_shape.iter().product();
                let base = self
                    .operation_latencies
                    .get("batchnorm")
                    .copied()
                    .unwrap_or(0.02);
                base * (ops as f64 / 1e6)
            }
            LayerType::Dropout(_) => 0.001,
            _ => 0.01,
        };
        let parallelization = match layer {
            LayerType::Dense(_) => self
                .parallelization_factors
                .get("dense")
                .copied()
                .unwrap_or(1.0),
            LayerType::Conv2D { .. } => self
                .parallelization_factors
                .get("conv2d")
                .copied()
                .unwrap_or(1.0),
            _ => 1.0,
        };
        Ok(base_latency * parallelization)
    }

    fn compute_output_shape(&self, layer: &LayerType, input_shape: &[usize]) -> Result<Vec<usize>> {
        match layer {
            LayerType::Dense(units) => Ok(vec![*units]),
            LayerType::Conv2D {
                filters,
                kernel_size,
                stride,
            } => {
                if input_shape.len() < 2 {
                    return Ok(input_shape.to_vec());
                }
                let h = (input_shape[0].saturating_sub(kernel_size.0)) / stride.0 + 1;
                let w = (input_shape[1].saturating_sub(kernel_size.1)) / stride.1 + 1;
                Ok(vec![h, w, *filters])
            }
            LayerType::MaxPool2D { pool_size, stride }
            | LayerType::AvgPool2D { pool_size, stride } => {
                if input_shape.len() < 3 {
                    return Ok(input_shape.to_vec());
                }
                let h = (input_shape[0].saturating_sub(pool_size.0)) / stride.0 + 1;
                let w = (input_shape[1].saturating_sub(pool_size.1)) / stride.1 + 1;
                Ok(vec![h, w, input_shape[2]])
            }
            LayerType::Flatten => {
                let total: usize = input_shape.iter().product();
                Ok(vec![total])
            }
            _ => Ok(input_shape.to_vec()),
        }
    }

    fn compute_initialization_overhead(&self, architecture: &Architecture) -> Result<f64> {
        let num_layers = architecture.layers.len() as f64;
        let base_overhead = match self.platform {
            HardwarePlatform::CPU => 5.0,
            HardwarePlatform::GPU => 20.0,
            HardwarePlatform::MobileARM => 10.0,
            HardwarePlatform::EdgeTPU => 15.0,
            _ => 8.0,
        };
        Ok(base_overhead + num_layers * 0.5)
    }
}

/// Memory usage predictor
pub struct MemoryPredictor {
    platform: HardwarePlatform,
}

impl MemoryPredictor {
    pub fn new(platform: HardwarePlatform) -> Self {
        Self { platform }
    }

    pub fn predict_memory_usage(
        &self,
        architecture: &Architecture,
        input_shape: &[usize],
    ) -> Result<f64> {
        let mut total_memory = 0.0;
        total_memory += self.compute_weights_memory(architecture)?;
        let mut current_shape = input_shape.to_vec();
        let mut max_activation_memory = 0.0f64;
        for layer in &architecture.layers {
            let activation_memory = self.compute_activation_memory(&current_shape)?;
            max_activation_memory = max_activation_memory.max(activation_memory);
            current_shape = self.compute_output_shape(layer, &current_shape);
        }
        total_memory += max_activation_memory;
        total_memory += self.compute_memory_overhead()?;
        Ok(total_memory)
    }

    fn compute_weights_memory(&self, architecture: &Architecture) -> Result<f64> {
        let mut weights_memory = 0.0;
        for layer in &architecture.layers {
            let layer_params = match layer {
                LayerType::Dense(units) => 1024 * units + units,
                LayerType::Conv2D {
                    filters,
                    kernel_size,
                    ..
                } => filters * kernel_size.0 * kernel_size.1 * 64 + filters,
                LayerType::BatchNorm => 128 * 4,
                LayerType::Embedding {
                    vocab_size,
                    embedding_dim,
                } => vocab_size * embedding_dim,
                _ => 0,
            };
            weights_memory += layer_params as f64 * 4.0;
        }
        Ok(weights_memory / (1024.0 * 1024.0))
    }

    fn compute_activation_memory(&self, shape: &[usize]) -> Result<f64> {
        let elements: usize = shape.iter().product();
        Ok(elements as f64 * 4.0 / (1024.0 * 1024.0))
    }

    fn compute_memory_overhead(&self) -> Result<f64> {
        let overhead_mb = match self.platform {
            HardwarePlatform::CPU => 50.0,
            HardwarePlatform::GPU => 100.0,
            HardwarePlatform::MobileARM => 25.0,
            HardwarePlatform::EdgeTPU => 30.0,
            _ => 40.0,
        };
        Ok(overhead_mb)
    }

    fn compute_output_shape(&self, layer: &LayerType, input_shape: &[usize]) -> Vec<usize> {
        match layer {
            LayerType::Dense(units) => vec![*units],
            _ => input_shape.to_vec(),
        }
    }
}

/// Energy consumption predictor
pub struct EnergyPredictor {
    power_characteristics: HashMap<String, f64>,
    platform: HardwarePlatform,
}

impl EnergyPredictor {
    pub fn new(platform: HardwarePlatform) -> Self {
        let mut predictor = Self {
            power_characteristics: HashMap::new(),
            platform,
        };
        predictor.initialize_power_characteristics();
        predictor
    }

    fn initialize_power_characteristics(&mut self) {
        let (dense_pw, conv_pw, mem_pw) = match self.platform {
            HardwarePlatform::CPU => (100.0, 150.0, 50.0),
            HardwarePlatform::GPU => (200.0, 300.0, 80.0),
            HardwarePlatform::MobileARM => (30.0, 50.0, 15.0),
            HardwarePlatform::EdgeTPU => (75.0, 100.0, 30.0),
            _ => (100.0, 150.0, 50.0),
        };
        self.power_characteristics
            .insert("dense".to_string(), dense_pw);
        self.power_characteristics
            .insert("conv2d".to_string(), conv_pw);
        self.power_characteristics
            .insert("memory".to_string(), mem_pw);
    }

    pub fn predict_energy(&self, architecture: &Architecture, latency_ms: f64) -> Result<f64> {
        let mut total_energy = 0.0;
        for layer in &architecture.layers {
            total_energy += self.compute_layer_energy(layer)?;
        }
        let static_power = self.get_static_power();
        total_energy += static_power * latency_ms;
        Ok(total_energy)
    }

    fn compute_layer_energy(&self, layer: &LayerType) -> Result<f64> {
        let power = match layer {
            LayerType::Dense(_) => self
                .power_characteristics
                .get("dense")
                .copied()
                .unwrap_or(100.0),
            LayerType::Conv2D { .. } => self
                .power_characteristics
                .get("conv2d")
                .copied()
                .unwrap_or(150.0),
            _ => 10.0,
        };
        Ok(power * 1.0)
    }

    fn get_static_power(&self) -> f64 {
        match self.platform {
            HardwarePlatform::CPU => 1000.0,
            HardwarePlatform::GPU => 5000.0,
            HardwarePlatform::MobileARM => 500.0,
            HardwarePlatform::EdgeTPU => 2000.0,
            _ => 1500.0,
        }
    }
}

/// Hardware-aware search implementation
pub struct HardwareAwareSearch {
    pub constraints: HardwareConstraints,
    latency_predictor: LatencyPredictor,
    memory_predictor: MemoryPredictor,
    energy_predictor: EnergyPredictor,
    constraint_weights: HashMap<String, f64>,
    violation_history: Vec<HashMap<String, f64>>,
}

impl HardwareAwareSearch {
    /// Create a new hardware-aware search
    pub fn new(constraints: HardwareConstraints) -> Self {
        let latency_predictor = LatencyPredictor::new(constraints.platform.clone());
        let memory_predictor = MemoryPredictor::new(constraints.platform.clone());
        let energy_predictor = EnergyPredictor::new(constraints.platform.clone());
        let mut constraint_weights = HashMap::new();
        constraint_weights.insert("latency".to_string(), 0.3);
        constraint_weights.insert("memory".to_string(), 0.25);
        constraint_weights.insert("energy".to_string(), 0.2);
        constraint_weights.insert("model_size".to_string(), 0.15);
        constraint_weights.insert("throughput".to_string(), 0.1);
        Self {
            constraints,
            latency_predictor,
            memory_predictor,
            energy_predictor,
            constraint_weights,
            violation_history: Vec::new(),
        }
    }

    /// Evaluate hardware constraints for an architecture
    pub fn evaluate_constraints(
        &mut self,
        architecture: &Arc<dyn ArchitectureEncoding>,
        input_shape: &[usize],
    ) -> Result<HashMap<String, f64>> {
        let arch = architecture.to_architecture()?;
        let mut violations = HashMap::new();
        let predicted_latency = self.latency_predictor.predict_latency(&arch, input_shape)?;
        if let Some(max_latency) = self.constraints.max_latency_ms {
            let violation = (predicted_latency - max_latency).max(0.0) / max_latency.max(1e-9);
            violations.insert("latency".to_string(), violation);
        }
        let predicted_memory = self
            .memory_predictor
            .predict_memory_usage(&arch, input_shape)?;
        if let Some(max_memory) = self.constraints.max_memory_mb {
            let violation = (predicted_memory - max_memory).max(0.0) / max_memory.max(1e-9);
            violations.insert("memory".to_string(), violation);
        }
        let predicted_energy = self
            .energy_predictor
            .predict_energy(&arch, predicted_latency)?;
        if let Some(max_energy) = self.constraints.max_energy_mj {
            let violation = (predicted_energy - max_energy).max(0.0) / max_energy.max(1e-9);
            violations.insert("energy".to_string(), violation);
        }
        let model_size = self.estimate_model_size(&arch)?;
        if let Some(max_size) = self.constraints.max_model_size_mb {
            let violation = (model_size - max_size).max(0.0) / max_size.max(1e-9);
            violations.insert("model_size".to_string(), violation);
        }
        let throughput = 1000.0 / predicted_latency.max(1e-9);
        if let Some(min_throughput) = self.constraints.min_throughput {
            let violation = (min_throughput - throughput).max(0.0) / min_throughput.max(1e-9);
            violations.insert("throughput".to_string(), violation);
        }
        self.violation_history.push(violations.clone());
        Ok(violations)
    }

    /// Compute weighted constraint violation score
    pub fn compute_constraint_score(&self, violations: &HashMap<String, f64>) -> f64 {
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        for (constraint, &violation) in violations {
            if let Some(&weight) = self.constraint_weights.get(constraint) {
                weighted_score += violation * weight;
                total_weight += weight;
            }
        }
        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        }
    }

    /// Check if architecture satisfies all constraints
    pub fn satisfies_constraints(&self, violations: &HashMap<String, f64>) -> bool {
        violations.values().all(|&v| v <= 0.0)
    }

    /// Get constraint satisfaction rate
    pub fn get_satisfaction_rate(&self) -> f64 {
        if self.violation_history.is_empty() {
            return 1.0;
        }
        let satisfied = self
            .violation_history
            .iter()
            .filter(|v| self.satisfies_constraints(v))
            .count();
        satisfied as f64 / self.violation_history.len() as f64
    }

    /// Generate optimization suggestions
    pub fn generate_optimization_suggestions(
        &self,
        violations: &HashMap<String, f64>,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        for (constraint, &violation) in violations {
            if violation > 0.1 {
                match constraint.as_str() {
                    "latency" => {
                        suggestions.push("Consider reducing model depth or width".to_string());
                        suggestions.push("Use depthwise separable convolutions".to_string());
                    }
                    "memory" => {
                        suggestions.push("Reduce batch size or model parameters".to_string());
                        suggestions.push("Use gradient checkpointing".to_string());
                    }
                    "energy" => {
                        suggestions.push("Reduce computational complexity".to_string());
                    }
                    "model_size" => {
                        suggestions.push("Apply model compression techniques".to_string());
                    }
                    "throughput" => {
                        suggestions.push("Optimize for batch processing".to_string());
                    }
                    _ => {}
                }
            }
        }
        suggestions
    }

    fn estimate_model_size(&self, architecture: &Architecture) -> Result<f64> {
        let mut total_params = 0usize;
        for layer in &architecture.layers {
            let layer_params = match layer {
                LayerType::Dense(units) => 1024 * units + units,
                LayerType::Conv2D {
                    filters,
                    kernel_size,
                    ..
                } => filters * kernel_size.0 * kernel_size.1 * 64 + filters,
                _ => 0,
            };
            total_params += layer_params;
        }
        Ok(total_params as f64 * 4.0 / (1024.0 * 1024.0))
    }

    /// Update constraint weights based on violation history
    pub fn adapt_constraint_weights(&mut self) {
        if self.violation_history.len() < 10 {
            return;
        }
        let recent = &self.violation_history[self.violation_history.len() - 10..];
        let mut avg_violations: HashMap<String, f64> = HashMap::new();
        for violations in recent {
            for (constraint, &violation) in violations {
                *avg_violations.entry(constraint.clone()).or_insert(0.0) += violation / 10.0;
            }
        }
        for (constraint, avg_violation) in avg_violations {
            if avg_violation > 0.1 {
                if let Some(weight) = self.constraint_weights.get_mut(&constraint) {
                    *weight = (*weight * 1.1).min(0.5);
                }
            }
        }
        let total_weight: f64 = self.constraint_weights.values().sum();
        if total_weight > 0.0 {
            for weight in self.constraint_weights.values_mut() {
                *weight /= total_weight;
            }
        }
    }

    /// Get platform characteristics summary
    pub fn get_platform_summary(&self) -> String {
        format!(
            "Platform: {:?}\nMax Latency: {:?} ms\nMax Memory: {:?} MB\nMax Energy: {:?} mJ\nCompute Units: {}",
            self.constraints.platform,
            self.constraints.max_latency_ms,
            self.constraints.max_memory_mb,
            self.constraints.max_energy_mj,
            self.constraints.compute_units
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::Architecture;

    #[test]
    fn test_hardware_constraints_default() {
        let constraints = HardwareConstraints::default();
        assert_eq!(constraints.platform, HardwarePlatform::CPU);
        assert!(constraints.max_latency_ms.is_some());
    }

    #[test]
    fn test_latency_predictor() {
        let predictor = LatencyPredictor::new(HardwarePlatform::CPU);
        let architecture = Architecture {
            layers: vec![
                LayerType::Dense(128),
                LayerType::Activation("relu".to_string()),
                LayerType::Dense(10),
            ],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        let latency = predictor
            .predict_latency(&architecture, &[784])
            .expect("failed to predict latency");
        assert!(latency >= 0.0);
    }

    #[test]
    fn test_memory_predictor() {
        let predictor = MemoryPredictor::new(HardwarePlatform::GPU);
        let architecture = Architecture {
            layers: vec![LayerType::Dense(256), LayerType::Dense(10)],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        let memory = predictor
            .predict_memory_usage(&architecture, &[784])
            .expect("failed to predict memory");
        assert!(memory > 0.0);
    }

    #[test]
    fn test_hardware_aware_search() {
        let constraints = HardwareConstraints::default();
        let mut search = HardwareAwareSearch::new(constraints);
        let arch: Arc<dyn crate::nas::ArchitectureEncoding> =
            Arc::new(crate::nas::architecture_encoding::SequentialEncoding::new(
                vec![LayerType::Dense(128), LayerType::Dense(10)],
            ));
        let violations = search
            .evaluate_constraints(&arch, &[784])
            .expect("failed to evaluate constraints");
        let score = search.compute_constraint_score(&violations);
        assert!(score >= 0.0);
    }
}
