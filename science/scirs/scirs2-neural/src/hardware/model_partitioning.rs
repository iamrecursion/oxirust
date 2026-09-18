//! Hardware-Aware Model Partitioning
//!
//! This module provides automatic partitioning of neural network models across
//! heterogeneous hardware accelerators for optimal performance and efficiency.

use crate::error::Result;
use crate::hardware::{Accelerator, AcceleratorType};
use crate::models::sequential::Sequential;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
/// Model partitioning strategy
#[derive(Debug, Clone, PartialEq)]
pub enum PartitioningStrategy {
    /// Minimize total execution time
    MinLatency,
    /// Minimize energy consumption
    MinEnergy,
    /// Maximize throughput
    MaxThroughput,
    /// Balance latency and energy
    Balanced {
        latency_weight: f32,
        energy_weight: f32,
    },
    /// Custom objective function
    Custom { objective: String },
}
/// Model partition representation
#[derive(Debug, Clone)]
pub struct ModelPartition {
    /// Partition ID
    pub id: usize,
    /// Layer indices included in this partition
    pub layer_indices: Vec<usize>,
    /// Target device for this partition
    pub target_device: AcceleratorType,
    /// Device ID
    pub device_id: usize,
    /// Estimated execution time in microseconds
    pub estimated_latency_us: f64,
    /// Estimated energy consumption in millijoules
    pub estimated_energy_mj: f64,
    /// Memory requirements in bytes
    pub memory_required: usize,
    /// Communication cost with other partitions
    pub communication_cost: f64,
/// Layer profiling information
pub struct LayerProfile {
    /// Layer index
    pub layer_index: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Input shape
    pub inputshape: Vec<usize>,
    /// Output shape
    pub outputshape: Vec<usize>,
    /// Number of parameters
    pub parameters: usize,
    /// FLOPs required
    pub flops: u64,
    /// Memory footprint in bytes
    pub memory_footprint: usize,
    /// Performance on different devices
    pub device_performance: HashMap<(AcceleratorType, usize), LayerPerformance>,
/// Layer types for partitioning
pub enum LayerType {
    Dense {
        units: usize,
    Conv2D {
        filters: usize,
        kernel_size: (usize, usize),
    Conv1D {
        kernel_size: usize,
    LSTM {
    GRU {
    Attention {
        heads: usize,
        dims: usize,
    BatchNorm,
    Dropout {
        rate: f32,
    Activation {
        function: String,
    Pooling {
        pool_type: String,
    Reshape,
    Custom {
        name: String,
/// Performance metrics for a layer on specific device
pub struct LayerPerformance {
    /// Execution time in microseconds
    pub execution_time_us: f64,
    /// Energy consumption in millijoules
    pub energy_consumption_mj: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Achievable throughput in samples/second
    pub throughput: f64,
    /// Accuracy impact (if any)
    pub accuracy_impact: f32,
/// Communication cost between partitions
pub struct CommunicationCost {
    /// Source partition
    pub src_partition: usize,
    /// Destination partition
    pub dst_partition: usize,
    /// Data transfer size in bytes
    pub transfer_size: usize,
    /// Transfer latency in microseconds
    pub latency_us: f64,
    /// Transfer energy in millijoules
    pub energy_mj: f64,
    /// Bandwidth utilization
    pub bandwidth_utilization: f32,
/// Hardware-aware model partitioner
pub struct ModelPartitioner {
    /// Available hardware devices
    devices: Vec<Arc<dyn Accelerator>>,
    /// Device performance profiles
    device_profiles: HashMap<(AcceleratorType, usize), DeviceProfile>,
    /// Communication topology
    communication_topology: CommunicationTopology,
    /// Partitioning strategy
    strategy: PartitioningStrategy,
    /// Constraints
    constraints: PartitioningConstraints,
impl ModelPartitioner {
    /// Create a new model partitioner
    pub fn new(devices: Vec<Arc<dyn Accelerator>>, strategy: PartitioningStrategy) -> Self {
        let device_profiles = Self::build_device_profiles(&_devices);
        let communication_topology = CommunicationTopology::build_from_devices(&_devices);
        Self {
            devices,
            device_profiles,
            communication_topology,
            strategy,
            constraints: PartitioningConstraints::default(),
        }
    }
    /// Set partitioning constraints
    pub fn set_constraints(&mut self, constraints: PartitioningConstraints) {
        self.constraints = constraints;
    /// Profile a model for partitioning
    pub fn profile_model(&self, model: &Sequential<f32>) -> Result<Vec<LayerProfile>> {
        let mut profiles = Vec::new();
        // Mock layer profiling - in practice, would analyze actual model layers
        for i in 0..10 {
            // Assume 10 layers for example
            let layer_type = match i % 4 {
                0 => LayerType::Dense { units: 512 },
                1 => LayerType::Conv2D {
                    filters: 64,
                    kernel_size: (3, 3),
                },
                2 => LayerType::LSTM { units: 256 },
                3 => LayerType::Attention {
                    heads: 8,
                    dims: 512_ => LayerType::Dense { units: 256 },
            };
            let mut device_performance = HashMap::new();
            // Profile performance on each device
            for device in &self.devices {
                let perf = self.estimate_layer_performance(&layer_type, device)?;
                device_performance.insert(
                    (device.accelerator_type(), 0), // Assume device ID 0
                    perf,
                );
            }
            profiles.push(LayerProfile {
                layer_index: i,
                layer_type,
                inputshape: vec![1, 224, 224, 3], // Example input
                outputshape: vec![1, 512],        // Example output
                parameters: 1000000,               // 1M parameters
                flops: 2000000,                    // 2M FLOPs
                memory_footprint: 4000000,         // 4MB
                device_performance,
            });
        Ok(profiles)
    /// Partition a model based on profiling information
    pub fn partition_model(&self, profiles: &[LayerProfile]) -> Result<Vec<ModelPartition>> {
        match self.strategy {
            PartitioningStrategy::MinLatency => self.partition_min_latency(profiles),
            PartitioningStrategy::MinEnergy => self.partition_min_energy(profiles),
            PartitioningStrategy::MaxThroughput => self.partition_max_throughput(profiles),
            PartitioningStrategy::Balanced {
                latency_weight,
                energy_weight,
            } => self.partition_balanced(profiles, latency_weight, energy_weight),
            PartitioningStrategy::Custom { .. } => Err(crate::error::NeuralError::NotImplemented(
                "Custom partitioning strategy not implemented".to_string(),
            )),
    /// Partition to minimize latency
    fn partition_min_latency(&self, profiles: &[LayerProfile]) -> Result<Vec<ModelPartition>> {
        let mut partitions = Vec::new();
        let mut current_partition_layers = Vec::new();
        let mut current_device = None;
        for profile in profiles {
            // Find device with minimum latency for this layer
            let best_device = profile
                .device_performance
                .iter()
                .min_by(|(_, perf1), (_, perf2)| {
                    perf1
                        .execution_time_us
                        .partial_cmp(&perf2.execution_time_us)
                        .expect("Operation failed")
                })
                .map(|((device_type, device_id)_)| (*device_type, *device_id));
            if let Some((device_type, device_id)) = best_device {
                // Check if we should continue current partition or start new one
                if current_device.is_none() || current_device == Some((device_type, device_id)) {
                    current_partition_layers.push(profile.layer_index);
                    current_device = Some((device_type, device_id));
                } else {
                    // Finalize current partition
                    if !current_partition_layers.is_empty() {
                        let partition = self.create_partition(
                            partitions.len(),
                            current_partition_layers.clone(),
                            current_device.expect("Operation failed"),
                            profiles,
                        )?;
                        partitions.push(partition);
                    }
                    // Start new partition
                    current_partition_layers = vec![profile.layer_index];
                }
        // Finalize last partition
        if !current_partition_layers.is_empty() {
            let partition = self.create_partition(
                partitions.len(),
                current_partition_layers,
                current_device.expect("Operation failed"),
                profiles,
            )?;
            partitions.push(partition);
        Ok(partitions)
    /// Partition to minimize energy consumption
    fn partition_min_energy(&self, profiles: &[LayerProfile]) -> Result<Vec<ModelPartition>> {
            // Find device with minimum energy for this layer
                        .energy_consumption_mj
                        .partial_cmp(&perf2.energy_consumption_mj)
    /// Partition to maximize throughput
    fn partition_max_throughput(&self, profiles: &[LayerProfile]) -> Result<Vec<ModelPartition>> {
        // For throughput maximization, try to use all devices in parallel
        let num_devices = self.devices.len();
        // Distribute layers among devices
        for (device_idx, device) in self.devices.iter().enumerate() {
            let device_type = device.accelerator_type();
            let device_id = 0; // Simplified
            let assigned_layers: Vec<usize> = profiles
                .enumerate()
                .filter(|(idx_)| idx % num_devices == device_idx)
                .map(|(_, profile)| profile.layer_index)
                .collect();
            if !assigned_layers.is_empty() {
                let partition = self.create_partition(
                    partitions.len(),
                    assigned_layers,
                    (device_type, device_id),
                    profiles,
                )?;
                partitions.push(partition);
    /// Balanced partitioning considering both latency and energy
    fn partition_balanced(
        &self,
        profiles: &[LayerProfile],
    ) -> Result<Vec<ModelPartition>> {
            // Find device with best weighted score
                    let score1 = latency_weight * perf1.execution_time_us as f32
                        + energy_weight * perf1.energy_consumption_mj as f32;
                    let score2 = latency_weight * perf2.execution_time_us as f32
                        + energy_weight * perf2.energy_consumption_mj as f32;
                    score1.partial_cmp(&score2).expect("Operation failed")
    /// Create a partition from layer indices and device assignment
    fn create_partition(
        partition_id: usize,
        layer_indices: Vec<usize>,
        (device_type, device_id): (AcceleratorType, usize),
    ) -> Result<ModelPartition> {
        let mut total_latency = 0.0;
        let mut total_energy = 0.0;
        let mut total_memory = 0;
        for &layer_idx in &layer_indices {
            if let Some(profile) = profiles.get(layer_idx) {
                if let Some(perf) = profile.device_performance.get(&(device_type, device_id)) {
                    total_latency += perf.execution_time_us;
                    total_energy += perf.energy_consumption_mj;
                    total_memory += perf.memory_usage;
        // Estimate communication cost (simplified)
        let communication_cost = if partition_id > 0 {
            self.estimate_communication_cost(partition_id, &layer_indices, profiles)
        } else {
            0.0
        };
        Ok(ModelPartition {
            id: partition_id,
            layer_indices,
            target_device: device_type,
            device_id,
            estimated_latency_us: total_latency,
            estimated_energy_mj: total_energy,
            memory_required: total_memory,
            communication_cost,
        })
    /// Estimate layer performance on a device
    fn estimate_layer_performance(
        layer_type: &LayerType,
        device: &Arc<dyn Accelerator>,
    ) -> Result<LayerPerformance> {
        let capabilities = device.capabilities();
        // Simplified performance estimation based on layer type and device capabilities
        let (base_time, base_energy) = match layer_type {
            LayerType::Dense { units } => {
                let ops = units * units; // Simplified
                let time = ops as f64 / (capabilities.peak_tflops_fp32 as f64 * 1e12) * 1e6; // microseconds
                let energy = time * 0.1; // Simplified energy model
                (time, energy)
            LayerType::Conv2D {
                filters,
                kernel_size,
            } => {
                let ops = filters * kernel_size.0 * kernel_size.1 * 224 * 224; // Simplified
                let time = ops as f64 / (capabilities.peak_tflops_fp32 as f64 * 1e12) * 1e6;
                let energy = time * 0.15; // Conv is more energy intensive
            LayerType::LSTM { units } => {
                let ops = units * units * 4; // 4 gates
                let energy = time * 0.2; // LSTM is memory intensive
            _ => (100.0, 10.0), // Default values
        // Device-specific adjustments
        let (adjusted_time, adjusted_energy) = match device.accelerator_type() {
            AcceleratorType::CPU => (base_time * 2.0, base_energy * 0.5),
            AcceleratorType::CUDA => (base_time * 0.3, base_energy * 1.5),
            AcceleratorType::TPU => (base_time * 0.2, base_energy * 0.8),
            AcceleratorType::FPGA => (base_time * 0.4, base_energy * 0.3, _ => (base_time, base_energy),
        Ok(LayerPerformance {
            execution_time_us: adjusted_time,
            energy_consumption_mj: adjusted_energy,
            memory_usage: 1000000,                 // 1MB default
            throughput: 1.0 / adjusted_time * 1e6, // samples/second
            accuracy_impact: 0.0,
    /// Estimate communication cost between partitions
    fn estimate_communication_cost(
        layer_indices: &[usize],
    ) -> f64 {
        // Simplified communication cost based on data transfer size
        if let Some(first_layer) = layer_indices.first() {
            if let Some(profile) = profiles.get(*first_layer) {
                let transfer_size = profile.inputshape.iter().product::<usize>() * 4; // bytes
                let bandwidth = 10e9; // 10 GB/s assumed
                return transfer_size as f64 / bandwidth * 1e6; // microseconds
        0.0
    /// Build device profiles
    fn build_device_profiles(
        devices: &[Arc<dyn Accelerator>],
    ) -> HashMap<(AcceleratorType, usize), DeviceProfile> {
        let mut profiles = HashMap::new();
        for (device_id, device) in devices.iter().enumerate() {
            let capabilities = device.capabilities();
            let profile = DeviceProfile {
                device_type: device.accelerator_type(),
                device_id,
                peak_compute_tflops: capabilities.peak_tflops_fp32,
                memory_bandwidth_gbps: capabilities.memory_bandwidth,
                memory_capacity_gb: capabilities.total_memory as f32 / 1e9,
                power_efficiency_tops_per_watt: 100.0, // Placeholder
                supported_precisions: vec!["fp32".to_string(), "fp16".to_string()],
            profiles.insert((device.accelerator_type(), device_id), profile);
        profiles
    /// Optimize partitions using genetic algorithm
    pub fn optimize_partitions(&self, profiles: &[LayerProfile]) -> Result<Vec<ModelPartition>> {
        // Simplified genetic algorithm for partition optimization
        let population_size = 50;
        let generations = 100;
        // Generate initial population
        let mut population = Vec::new();
        for _ in 0..population_size {
            let partitions = self.generate_random_partitioning(profiles)?;
            population.push(partitions);
        // Evolve population
        for _generation in 0..generations {
            // Evaluate fitness
            let mut fitness_scores: Vec<(f64, usize)> = population
                .map(|(idx, partitions)| {
                    let score = self.evaluate_partitioning_fitness(partitions);
                    (score, idx)
            // Sort by fitness (lower is better)
            fitness_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("Operation failed"));
            // Keep best half
            let mut new_population = Vec::new();
            for i in 0..population_size / 2 {
                new_population.push(population[fitness_scores[i].1].clone());
            // Generate offspring from crossover and mutation
            while new_population.len() < population_size {
                let parent1_idx = fitness_scores[0].1;
                let parent2_idx = fitness_scores[1].1;
                let offspring =
                    self.crossover_partitions(&population[parent1_idx], &population[parent2_idx]);
                new_population.push(offspring);
            population = new_population;
        // Return best solution
        let fitness_scores: Vec<(f64, usize)> = population
            .iter()
            .enumerate()
            .map(|(idx, partitions)| {
                let score = self.evaluate_partitioning_fitness(partitions);
                (score, idx)
            })
            .collect();
        let best_idx = fitness_scores
            .min_by(|a, b| a.0.partial_cmp(&b.0).expect("Operation failed"))
            .map(|(_, idx)| *idx)
            .unwrap_or(0);
        Ok(population[best_idx].clone())
    /// Generate random partitioning
    fn generate_random_partitioning(
        // Simple random assignment
        self.partition_min_latency(profiles) // Placeholder
    /// Evaluate fitness of a partitioning
    fn evaluate_partitioning_fitness(&self, partitions: &[ModelPartition]) -> f64 {
            PartitioningStrategy::MinLatency => partitions
                .map(|p| p.estimated_latency_us)
                .sum::<f64>(),
            PartitioningStrategy::MinEnergy => partitions
                .map(|p| p.estimated_energy_mj)
            } => partitions
                .map(|p| {
                    latency_weight as f64 * p.estimated_latency_us
                        + energy_weight as f64 * p.estimated_energy_mj
            _ => 0.0,
    /// Crossover operation for genetic algorithm
    fn crossover_partitions(
        parent1: &[ModelPartition],
        parent2: &[ModelPartition],
    ) -> Vec<ModelPartition> {
        // Simple crossover - take random partitions from each parent
        let mut offspring = Vec::new();
        let crossover_point = parent1.len() / 2;
        for i in 0..parent1.len().min(parent2.len()) {
            if i < crossover_point {
                offspring.push(parent1[i].clone());
            } else {
                offspring.push(parent2[i].clone());
        offspring
/// Device performance profile
pub struct DeviceProfile {
    pub device_type: AcceleratorType,
    pub peak_compute_tflops: f32,
    pub memory_bandwidth_gbps: f32,
    pub memory_capacity_gb: f32,
    pub power_efficiency_tops_per_watt: f32,
    pub supported_precisions: Vec<String>,
/// Communication topology between devices
pub struct CommunicationTopology {
    /// Bandwidth matrix (device_id -> device_id -> bandwidth in GB/s)
    pub bandwidth_matrix: BTreeMap<(usize, usize), f32>,
    /// Latency matrix (device_id -> device_id -> latency in microseconds)
    pub latency_matrix: BTreeMap<(usize, usize), f64>,
impl CommunicationTopology {
    /// Build communication topology from devices
    fn build_from_devices(devices: &[Arc<dyn Accelerator>]) -> Self {
        let mut bandwidth_matrix = BTreeMap::new();
        let mut latency_matrix = BTreeMap::new();
        // Build all-to-all connectivity with estimated values
        for i in 0.._devices.len() {
            for j in 0.._devices.len() {
                if i == j {
                    // Same device - no communication cost
                    bandwidth_matrix.insert((i, j), f32::INFINITY);
                    latency_matrix.insert((i, j), 0.0);
                    // Inter-device communication
                    let bandwidth = 10.0; // 10 GB/s default
                    let latency = 100.0; // 100 microseconds default
                    bandwidth_matrix.insert((i, j), bandwidth);
                    latency_matrix.insert((i, j), latency);
            bandwidth_matrix,
            latency_matrix,
/// Partitioning constraints
pub struct PartitioningConstraints {
    /// Maximum number of partitions
    pub max_partitions: Option<usize>,
    /// Memory limit per device
    pub memory_limits: HashMap<(AcceleratorType, usize), usize>,
    /// Latency constraint in microseconds
    pub max_latency_us: Option<f64>,
    /// Energy budget in millijoules
    pub energy_budget_mj: Option<f64>,
    /// Required _devices (must be used)
    pub required_devices: Vec<(AcceleratorType, usize)>,
    /// Forbidden _devices
    pub forbidden_devices: Vec<(AcceleratorType, usize)>,
impl Default for PartitioningConstraints {
    fn default() -> Self {
            max_partitions: None,
            memory_limits: HashMap::new(),
            max_latency_us: None,
            energy_budget_mj: None,
            required_devices: Vec::new(),
            forbidden_devices: Vec::new(),
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::accelerator::{AcceleratorFactory, CPUAccelerator};
    #[test]
    fn test_model_partitioner_creation() {
        let cpu1 = AcceleratorFactory::create(AcceleratorType::CPU).expect("Operation failed");
        let cpu2 = AcceleratorFactory::create(AcceleratorType::CPU).expect("Operation failed");
        let partitioner = ModelPartitioner::new(vec![cpu1, cpu2], PartitioningStrategy::MinLatency);
        assert_eq!(partitioner.devices.len(), 2);
    fn test_layer_profiling() {
        let cpu = AcceleratorFactory::create(AcceleratorType::CPU).expect("Operation failed");
        let partitioner = ModelPartitioner::new(vec![cpu], PartitioningStrategy::MinLatency);
        let model = Sequential::new(); // Mock model
        let profiles = partitioner.profile_model(&model).expect("Operation failed");
        assert!(!profiles.is_empty());
        assert!(profiles[0].device_performance.len() > 0);
    fn test_partitioning_strategies() {
        let model = Sequential::new();
        let partitions = partitioner.partition_model(&profiles).expect("Operation failed");
        assert!(!partitions.is_empty());
        assert!(partitions.iter().all(|p| !p.layer_indices.is_empty()));
