//! Load balancing strategies for distributed integration
//!
//! This module provides various load balancing strategies for distributing
//! work across compute nodes in the distributed integration system.

use crate::common::IntegrateFloat;
use crate::distributed::types::{
    ChunkId, DistributedError, DistributedResult, JobId, LoadBalancingStrategy, NodeId, NodeInfo,
    WorkChunk,
};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Load balancer for distributing work chunks across nodes
pub struct LoadBalancer<F: IntegrateFloat> {
    /// Current strategy
    strategy: RwLock<LoadBalancingStrategy>,
    /// Node performance history
    node_performance: RwLock<HashMap<NodeId, NodePerformance>>,
    /// Work assignment history
    assignment_history: Mutex<VecDeque<Assignment>>,
    /// Round-robin counter
    round_robin_counter: AtomicUsize,
    /// Configuration
    config: LoadBalancerConfig,
    /// Phantom for float type
    _phantom: std::marker::PhantomData<F>,
}

/// Configuration for the load balancer
#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    /// Maximum history entries to keep
    pub max_history: usize,
    /// Minimum samples before adapting
    pub min_samples_for_adaptation: usize,
    /// Performance smoothing factor (EMA alpha)
    pub smoothing_factor: f64,
    /// Imbalance threshold for triggering rebalancing
    pub imbalance_threshold: f64,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Work stealing threshold (fraction of work to steal)
    pub work_stealing_threshold: f64,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            min_samples_for_adaptation: 10,
            smoothing_factor: 0.3,
            imbalance_threshold: 0.3,
            enable_work_stealing: true,
            work_stealing_threshold: 0.5,
        }
    }
}

/// Performance metrics for a node
#[derive(Debug, Clone)]
pub struct NodePerformance {
    /// Node ID
    pub node_id: NodeId,
    /// Average processing time per unit of estimated cost
    pub avg_time_per_cost: f64,
    /// Standard deviation of processing times
    pub time_stddev: f64,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Total processing time
    pub total_time: Duration,
    /// Number of failures
    pub failures: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Current load (number of pending chunks)
    pub current_load: usize,
    /// Recent processing times for variance calculation
    recent_times: VecDeque<f64>,
}

impl NodePerformance {
    /// Create new performance metrics
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            avg_time_per_cost: 1.0,
            time_stddev: 0.0,
            chunks_processed: 0,
            total_time: Duration::ZERO,
            failures: 0,
            success_rate: 1.0,
            current_load: 0,
            recent_times: VecDeque::with_capacity(100),
        }
    }

    /// Update performance with a new sample
    pub fn update(&mut self, processing_time: Duration, estimated_cost: f64, success: bool) {
        if success {
            let time_per_cost = processing_time.as_secs_f64() / estimated_cost.max(0.001);

            // Update EMA of time per cost
            if self.chunks_processed == 0 {
                self.avg_time_per_cost = time_per_cost;
            } else {
                let alpha = 0.3;
                self.avg_time_per_cost =
                    alpha * time_per_cost + (1.0 - alpha) * self.avg_time_per_cost;
            }

            // Track recent times for variance
            self.recent_times.push_back(time_per_cost);
            if self.recent_times.len() > 100 {
                self.recent_times.pop_front();
            }

            // Update variance
            if self.recent_times.len() >= 2 {
                let mean: f64 =
                    self.recent_times.iter().sum::<f64>() / self.recent_times.len() as f64;
                let variance: f64 = self
                    .recent_times
                    .iter()
                    .map(|t| (t - mean).powi(2))
                    .sum::<f64>()
                    / self.recent_times.len() as f64;
                self.time_stddev = variance.sqrt();
            }

            self.chunks_processed += 1;
            self.total_time += processing_time;
        } else {
            self.failures += 1;
        }

        // Update success rate
        let total_attempts = self.chunks_processed + self.failures;
        if total_attempts > 0 {
            self.success_rate = self.chunks_processed as f64 / total_attempts as f64;
        }
    }

    /// Get expected processing time for a given cost
    pub fn expected_time(&self, estimated_cost: f64) -> Duration {
        Duration::from_secs_f64(self.avg_time_per_cost * estimated_cost)
    }

    /// Calculate node score for assignment (higher is better)
    pub fn assignment_score(&self, estimated_cost: f64) -> f64 {
        // Factors: speed, reliability, current load
        let speed_score = 1.0 / (self.avg_time_per_cost + 0.001);
        let reliability_score = self.success_rate;
        let load_penalty = 1.0 / (1.0 + self.current_load as f64);

        speed_score * reliability_score * load_penalty
    }
}

/// Record of a work assignment
#[derive(Debug, Clone)]
struct Assignment {
    /// Chunk ID
    chunk_id: ChunkId,
    /// Assigned node
    node_id: NodeId,
    /// Timestamp
    timestamp: Instant,
    /// Estimated cost
    estimated_cost: f64,
}

impl<F: IntegrateFloat> LoadBalancer<F> {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy, config: LoadBalancerConfig) -> Self {
        Self {
            strategy: RwLock::new(strategy),
            node_performance: RwLock::new(HashMap::new()),
            assignment_history: Mutex::new(VecDeque::new()),
            round_robin_counter: AtomicUsize::new(0),
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Register a new node
    pub fn register_node(&self, node_id: NodeId) -> DistributedResult<()> {
        match self.node_performance.write() {
            Ok(mut perf) => {
                perf.insert(node_id, NodePerformance::new(node_id));
                Ok(())
            }
            Err(_) => Err(DistributedError::ConfigError(
                "Failed to register node".to_string(),
            )),
        }
    }

    /// Deregister a node
    pub fn deregister_node(&self, node_id: NodeId) -> DistributedResult<()> {
        match self.node_performance.write() {
            Ok(mut perf) => {
                perf.remove(&node_id);
                Ok(())
            }
            Err(_) => Err(DistributedError::ConfigError(
                "Failed to deregister node".to_string(),
            )),
        }
    }

    /// Get current strategy
    pub fn get_strategy(&self) -> LoadBalancingStrategy {
        match self.strategy.read() {
            Ok(s) => *s,
            Err(_) => LoadBalancingStrategy::RoundRobin,
        }
    }

    /// Set strategy
    pub fn set_strategy(&self, strategy: LoadBalancingStrategy) {
        if let Ok(mut s) = self.strategy.write() {
            *s = strategy;
        }
    }

    /// Assign a work chunk to a node
    pub fn assign_chunk(
        &self,
        chunk: &WorkChunk<F>,
        available_nodes: &[NodeInfo],
    ) -> DistributedResult<NodeId> {
        if available_nodes.is_empty() {
            return Err(DistributedError::ResourceExhausted(
                "No available nodes".to_string(),
            ));
        }

        let strategy = self.get_strategy();
        let node_id = match strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_assignment(available_nodes)?,
            LoadBalancingStrategy::CapabilityBased => {
                self.capability_based_assignment(chunk, available_nodes)?
            }
            LoadBalancingStrategy::WorkStealing => {
                self.work_stealing_assignment(chunk, available_nodes)?
            }
            LoadBalancingStrategy::Adaptive => self.adaptive_assignment(chunk, available_nodes)?,
            LoadBalancingStrategy::LocalityAware => {
                self.locality_aware_assignment(chunk, available_nodes)?
            }
        };

        // Record assignment
        self.record_assignment(chunk.id, node_id, chunk.estimated_cost);

        // Update current load
        if let Ok(mut perf) = self.node_performance.write() {
            if let Some(p) = perf.get_mut(&node_id) {
                p.current_load += 1;
            }
        }

        Ok(node_id)
    }

    /// Round-robin assignment
    fn round_robin_assignment(&self, nodes: &[NodeInfo]) -> DistributedResult<NodeId> {
        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % nodes.len();
        Ok(nodes[idx].id)
    }

    /// Capability-based assignment
    fn capability_based_assignment(
        &self,
        chunk: &WorkChunk<F>,
        nodes: &[NodeInfo],
    ) -> DistributedResult<NodeId> {
        // Score nodes by capabilities
        let best_node = nodes
            .iter()
            .max_by(|a, b| {
                let score_a = Self::capability_score(a, chunk.estimated_cost);
                let score_b = Self::capability_score(b, chunk.estimated_cost);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| DistributedError::ResourceExhausted("No suitable node".to_string()))?;

        Ok(best_node.id)
    }

    /// Calculate capability score for a node
    fn capability_score(node: &NodeInfo, estimated_cost: f64) -> f64 {
        let cpu_score = node.capabilities.cpu_cores as f64;
        let memory_score = (node.capabilities.memory_bytes as f64 / 1e9).min(32.0) / 32.0;
        let gpu_bonus = if node.capabilities.has_gpu { 5.0 } else { 0.0 };
        let latency_penalty = (node.capabilities.latency_us as f64 / 10000.0).min(1.0);

        (cpu_score + memory_score + gpu_bonus) * (1.0 - latency_penalty * 0.1)
    }

    /// Work-stealing-aware assignment
    fn work_stealing_assignment(
        &self,
        chunk: &WorkChunk<F>,
        nodes: &[NodeInfo],
    ) -> DistributedResult<NodeId> {
        // Find node with lowest current load, considering performance
        match self.node_performance.read() {
            Ok(perf) => {
                let best_node = nodes
                    .iter()
                    .min_by(|a, b| {
                        let load_a = perf.get(&a.id).map(|p| p.current_load).unwrap_or(0);
                        let load_b = perf.get(&b.id).map(|p| p.current_load).unwrap_or(0);
                        load_a.cmp(&load_b)
                    })
                    .ok_or_else(|| {
                        DistributedError::ResourceExhausted("No suitable node".to_string())
                    })?;

                Ok(best_node.id)
            }
            Err(_) => self.round_robin_assignment(nodes),
        }
    }

    /// Adaptive assignment based on performance history
    fn adaptive_assignment(
        &self,
        chunk: &WorkChunk<F>,
        nodes: &[NodeInfo],
    ) -> DistributedResult<NodeId> {
        match self.node_performance.read() {
            Ok(perf) => {
                // Check if we have enough samples for adaptation
                let total_samples: usize = perf.values().map(|p| p.chunks_processed).sum();

                if total_samples < self.config.min_samples_for_adaptation {
                    // Not enough data, use round-robin
                    return self.round_robin_assignment(nodes);
                }

                // Score each node
                let best_node = nodes
                    .iter()
                    .max_by(|a, b| {
                        let score_a = perf
                            .get(&a.id)
                            .map(|p| p.assignment_score(chunk.estimated_cost))
                            .unwrap_or(0.0);
                        let score_b = perf
                            .get(&b.id)
                            .map(|p| p.assignment_score(chunk.estimated_cost))
                            .unwrap_or(0.0);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| {
                        DistributedError::ResourceExhausted("No suitable node".to_string())
                    })?;

                Ok(best_node.id)
            }
            Err(_) => self.round_robin_assignment(nodes),
        }
    }

    /// Locality-aware assignment (keeps related chunks together)
    fn locality_aware_assignment(
        &self,
        chunk: &WorkChunk<F>,
        nodes: &[NodeInfo],
    ) -> DistributedResult<NodeId> {
        // For now, use job ID modulo to keep related chunks on same nodes
        let job_mod = chunk.job_id.value() as usize % nodes.len();
        let chunk_mod = chunk.id.value() as usize % nodes.len();

        // Combine job and chunk locality
        let idx = (job_mod + chunk_mod) % nodes.len();
        Ok(nodes[idx].id)
    }

    /// Record an assignment
    fn record_assignment(&self, chunk_id: ChunkId, node_id: NodeId, estimated_cost: f64) {
        if let Ok(mut history) = self.assignment_history.lock() {
            history.push_back(Assignment {
                chunk_id,
                node_id,
                timestamp: Instant::now(),
                estimated_cost,
            });

            // Trim history
            while history.len() > self.config.max_history {
                history.pop_front();
            }
        }
    }

    /// Report chunk completion
    pub fn report_completion(
        &self,
        node_id: NodeId,
        estimated_cost: f64,
        processing_time: Duration,
        success: bool,
    ) {
        if let Ok(mut perf) = self.node_performance.write() {
            if let Some(p) = perf.get_mut(&node_id) {
                p.update(processing_time, estimated_cost, success);
                if p.current_load > 0 {
                    p.current_load -= 1;
                }
            }
        }
    }

    /// Get current load distribution
    pub fn get_load_distribution(&self) -> HashMap<NodeId, usize> {
        match self.node_performance.read() {
            Ok(perf) => perf.iter().map(|(id, p)| (*id, p.current_load)).collect(),
            Err(_) => HashMap::new(),
        }
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalancing(&self) -> bool {
        match self.node_performance.read() {
            Ok(perf) => {
                if perf.is_empty() {
                    return false;
                }

                let loads: Vec<f64> = perf.values().map(|p| p.current_load as f64).collect();

                if loads.is_empty() {
                    return false;
                }

                let mean = loads.iter().sum::<f64>() / loads.len() as f64;
                if mean <= 0.0 {
                    return false;
                }

                let max_deviation = loads
                    .iter()
                    .map(|l| (l - mean).abs() / mean)
                    .fold(0.0_f64, f64::max);

                max_deviation > self.config.imbalance_threshold
            }
            Err(_) => false,
        }
    }

    /// Get nodes with excess work (candidates for work stealing)
    pub fn get_overloaded_nodes(&self) -> Vec<(NodeId, usize)> {
        match self.node_performance.read() {
            Ok(perf) => {
                let loads: Vec<_> = perf.iter().map(|(id, p)| (*id, p.current_load)).collect();

                if loads.is_empty() {
                    return Vec::new();
                }

                let mean_load: f64 =
                    loads.iter().map(|(_, l)| *l as f64).sum::<f64>() / loads.len() as f64;
                let threshold = (mean_load * (1.0 + self.config.imbalance_threshold)) as usize;

                loads
                    .into_iter()
                    .filter(|(_, load)| *load > threshold)
                    .collect()
            }
            Err(_) => Vec::new(),
        }
    }

    /// Get nodes with room for more work
    pub fn get_underloaded_nodes(&self) -> Vec<(NodeId, usize)> {
        match self.node_performance.read() {
            Ok(perf) => {
                let loads: Vec<_> = perf.iter().map(|(id, p)| (*id, p.current_load)).collect();

                if loads.is_empty() {
                    return Vec::new();
                }

                let mean_load: f64 =
                    loads.iter().map(|(_, l)| *l as f64).sum::<f64>() / loads.len() as f64;
                let threshold = (mean_load * (1.0 - self.config.imbalance_threshold)) as usize;

                loads
                    .into_iter()
                    .filter(|(_, load)| *load < threshold)
                    .collect()
            }
            Err(_) => Vec::new(),
        }
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> LoadBalancerStatistics {
        match self.node_performance.read() {
            Ok(perf) => {
                let node_count = perf.len();
                let total_chunks: usize = perf.values().map(|p| p.chunks_processed).sum();
                let total_failures: usize = perf.values().map(|p| p.failures).sum();

                let loads: Vec<f64> = perf.values().map(|p| p.current_load as f64).collect();
                let load_variance = if !loads.is_empty() {
                    let mean = loads.iter().sum::<f64>() / loads.len() as f64;
                    loads.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / loads.len() as f64
                } else {
                    0.0
                };

                LoadBalancerStatistics {
                    node_count,
                    total_chunks_assigned: total_chunks,
                    total_failures,
                    load_variance,
                    current_strategy: self.get_strategy(),
                }
            }
            Err(_) => LoadBalancerStatistics::default(),
        }
    }
}

/// Statistics about load balancer performance
#[derive(Debug, Clone, Default)]
pub struct LoadBalancerStatistics {
    /// Number of registered nodes
    pub node_count: usize,
    /// Total chunks assigned
    pub total_chunks_assigned: usize,
    /// Total failures
    pub total_failures: usize,
    /// Current load variance
    pub load_variance: f64,
    /// Current strategy
    pub current_strategy: LoadBalancingStrategy,
}

#[allow(clippy::derivable_impls)]
impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Work chunk distributor for initial distribution
pub struct ChunkDistributor<F: IntegrateFloat> {
    /// Job ID
    job_id: JobId,
    /// Next chunk ID
    next_chunk_id: AtomicUsize,
    /// Phantom for float type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: IntegrateFloat> ChunkDistributor<F> {
    /// Create a new chunk distributor
    pub fn new(job_id: JobId) -> Self {
        Self {
            job_id,
            next_chunk_id: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create work chunks from a time interval
    pub fn create_chunks(
        &self,
        t_span: (F, F),
        initial_state: Array1<F>,
        num_chunks: usize,
    ) -> Vec<WorkChunk<F>> {
        let t_start = t_span.0;
        let t_end = t_span.1;
        let dt = (t_end - t_start) / F::from(num_chunks).unwrap_or(F::one());

        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let chunk_t_start = t_start + dt * F::from(i).unwrap_or(F::zero());
            let chunk_t_end = if i == num_chunks - 1 {
                t_end
            } else {
                t_start + dt * F::from(i + 1).unwrap_or(F::one())
            };

            let chunk_id = ChunkId::new(self.next_chunk_id.fetch_add(1, Ordering::SeqCst) as u64);

            // Initial state for first chunk, placeholder for others
            // (will be filled in by boundary exchange)
            let state = if i == 0 {
                initial_state.clone()
            } else {
                Array1::zeros(initial_state.len())
            };

            chunks.push(WorkChunk::new(
                chunk_id,
                self.job_id,
                (chunk_t_start, chunk_t_end),
                state,
            ));
        }

        chunks
    }

    /// Subdivide a chunk into smaller chunks
    pub fn subdivide_chunk(&self, chunk: &WorkChunk<F>, num_parts: usize) -> Vec<WorkChunk<F>> {
        let (t_start, t_end) = chunk.time_interval;
        let dt = (t_end - t_start) / F::from(num_parts).unwrap_or(F::one());

        let mut sub_chunks = Vec::with_capacity(num_parts);

        for i in 0..num_parts {
            let sub_t_start = t_start + dt * F::from(i).unwrap_or(F::zero());
            let sub_t_end = if i == num_parts - 1 {
                t_end
            } else {
                t_start + dt * F::from(i + 1).unwrap_or(F::one())
            };

            let sub_chunk_id =
                ChunkId::new(self.next_chunk_id.fetch_add(1, Ordering::SeqCst) as u64);

            let state = if i == 0 {
                chunk.initial_state.clone()
            } else {
                Array1::zeros(chunk.initial_state.len())
            };

            let mut sub_chunk =
                WorkChunk::new(sub_chunk_id, chunk.job_id, (sub_t_start, sub_t_end), state);

            sub_chunk.priority = chunk.priority;
            sub_chunks.push(sub_chunk);
        }

        sub_chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::types::NodeCapabilities;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn create_test_nodes(n: usize) -> Vec<NodeInfo> {
        (0..n)
            .map(|i| {
                let addr =
                    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080 + i as u16);
                let mut info = NodeInfo::new(NodeId::new(i as u64), addr);
                info.capabilities = NodeCapabilities::default();
                info
            })
            .collect()
    }

    #[test]
    fn test_round_robin_assignment() {
        let balancer: LoadBalancer<f64> = LoadBalancer::new(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancerConfig::default(),
        );

        let nodes = create_test_nodes(3);

        // Register nodes
        for node in &nodes {
            balancer.register_node(node.id).expect("Failed to register");
        }

        let chunk = WorkChunk::new(ChunkId::new(1), JobId::new(1), (0.0, 1.0), Array1::zeros(3));

        // Should cycle through nodes
        let assignments: Vec<_> = (0..6)
            .map(|_| {
                balancer
                    .assign_chunk(&chunk, &nodes)
                    .expect("Assignment failed")
            })
            .collect();

        // Check round-robin pattern
        for i in 0..3 {
            assert_eq!(assignments[i], assignments[i + 3]);
        }
    }

    #[test]
    fn test_performance_update() {
        let mut perf = NodePerformance::new(NodeId::new(1));

        perf.update(Duration::from_millis(100), 1.0, true);
        assert_eq!(perf.chunks_processed, 1);
        assert!(perf.success_rate > 0.9);

        perf.update(Duration::from_millis(50), 1.0, false);
        assert_eq!(perf.failures, 1);
        assert!(perf.success_rate < 1.0);
    }

    #[test]
    fn test_chunk_distributor() {
        let distributor: ChunkDistributor<f64> = ChunkDistributor::new(JobId::new(1));

        let chunks = distributor.create_chunks((0.0, 10.0), Array1::from_vec(vec![1.0, 2.0]), 5);

        assert_eq!(chunks.len(), 5);
        assert!((chunks[0].time_interval.0 - 0.0).abs() < 1e-10);
        assert!((chunks[4].time_interval.1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_distribution() {
        let balancer: LoadBalancer<f64> = LoadBalancer::new(
            LoadBalancingStrategy::Adaptive,
            LoadBalancerConfig::default(),
        );

        let nodes = create_test_nodes(3);
        for node in &nodes {
            balancer.register_node(node.id).expect("Failed to register");
        }

        // Simulate assignments
        for i in 0..10 {
            let chunk =
                WorkChunk::new(ChunkId::new(i), JobId::new(1), (0.0, 1.0), Array1::zeros(3));
            let _ = balancer.assign_chunk(&chunk, &nodes);
        }

        let distribution = balancer.get_load_distribution();
        assert_eq!(distribution.len(), 3);
    }
}
