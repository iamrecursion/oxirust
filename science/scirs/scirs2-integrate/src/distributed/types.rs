//! Core types for distributed computing in numerical integration
//!
//! This module provides the fundamental types and abstractions for
//! distributed computation of ODEs and numerical integration across
//! multiple compute nodes.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Unique identifier for a compute node in the distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Unique identifier for a distributed computation job
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

impl JobId {
    /// Create a new job ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Unique identifier for a work chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkId(pub u64);

impl ChunkId {
    /// Create a new chunk ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Status of a compute node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is available for work
    Available,
    /// Node is currently processing work
    Busy,
    /// Node has failed and is unavailable
    Failed,
    /// Node is in maintenance mode
    Maintenance,
    /// Node is starting up
    Initializing,
    /// Node is shutting down
    ShuttingDown,
}

/// Capabilities and resources of a compute node
#[derive(Debug, Clone)]
pub struct NodeCapabilities {
    /// Number of CPU cores available
    pub cpu_cores: usize,
    /// Available memory in bytes
    pub memory_bytes: usize,
    /// Whether GPU acceleration is available
    pub has_gpu: bool,
    /// GPU memory in bytes (if available)
    pub gpu_memory_bytes: Option<usize>,
    /// Network bandwidth in bytes per second
    pub network_bandwidth: usize,
    /// Latency to coordinator in microseconds
    pub latency_us: u64,
    /// Supported floating-point precisions
    pub supported_precisions: Vec<FloatPrecision>,
    /// SIMD capabilities
    pub simd_capabilities: SimdCapability,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_bytes: 1024 * 1024 * 1024, // 1 GB
            has_gpu: false,
            gpu_memory_bytes: None,
            network_bandwidth: 100 * 1024 * 1024, // 100 MB/s
            latency_us: 1000,                     // 1ms
            supported_precisions: vec![FloatPrecision::F32, FloatPrecision::F64],
            simd_capabilities: SimdCapability::default(),
        }
    }
}

/// Floating-point precision options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatPrecision {
    /// 16-bit floating point (half)
    F16,
    /// 32-bit floating point (single)
    F32,
    /// 64-bit floating point (double)
    F64,
}

/// SIMD capability information
#[derive(Debug, Clone, Default)]
pub struct SimdCapability {
    /// SSE support
    pub has_sse: bool,
    /// SSE2 support
    pub has_sse2: bool,
    /// AVX support
    pub has_avx: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// AVX-512 support
    pub has_avx512: bool,
    /// NEON support (ARM)
    pub has_neon: bool,
}

/// Information about a compute node
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Unique identifier for this node
    pub id: NodeId,
    /// Network address of the node
    pub address: SocketAddr,
    /// Current status of the node
    pub status: NodeStatus,
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    /// Last heartbeat timestamp
    pub last_heartbeat: Instant,
    /// Number of jobs completed
    pub jobs_completed: usize,
    /// Average job duration
    pub average_job_duration: Duration,
}

impl NodeInfo {
    /// Create a new node info with default capabilities
    pub fn new(id: NodeId, address: SocketAddr) -> Self {
        Self {
            id,
            address,
            status: NodeStatus::Initializing,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Instant::now(),
            jobs_completed: 0,
            average_job_duration: Duration::ZERO,
        }
    }

    /// Check if the node is healthy (recent heartbeat)
    pub fn is_healthy(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() < timeout
            && self.status != NodeStatus::Failed
            && self.status != NodeStatus::ShuttingDown
    }

    /// Calculate the node's processing score for load balancing
    pub fn processing_score(&self) -> f64 {
        let base_score = self.capabilities.cpu_cores as f64;
        let gpu_bonus = if self.capabilities.has_gpu { 10.0 } else { 0.0 };
        let latency_penalty = (self.capabilities.latency_us as f64 / 1000.0).min(5.0);

        base_score + gpu_bonus - latency_penalty
    }
}

/// A chunk of work to be processed by a node
#[derive(Debug, Clone)]
pub struct WorkChunk<F: IntegrateFloat> {
    /// Unique identifier for this chunk
    pub id: ChunkId,
    /// Job this chunk belongs to
    pub job_id: JobId,
    /// Time interval for this chunk [t_start, t_end]
    pub time_interval: (F, F),
    /// Initial state for this chunk
    pub initial_state: Array1<F>,
    /// Boundary conditions from adjacent chunks
    pub boundary_conditions: BoundaryConditions<F>,
    /// Priority level (higher = more urgent)
    pub priority: u32,
    /// Estimated computational cost
    pub estimated_cost: f64,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Maximum allowed retries
    pub max_retries: u32,
}

impl<F: IntegrateFloat> WorkChunk<F> {
    /// Create a new work chunk
    pub fn new(
        id: ChunkId,
        job_id: JobId,
        time_interval: (F, F),
        initial_state: Array1<F>,
    ) -> Self {
        let estimated_cost = Self::estimate_cost(&time_interval, initial_state.len());
        Self {
            id,
            job_id,
            time_interval,
            initial_state,
            boundary_conditions: BoundaryConditions::default(),
            priority: 0,
            estimated_cost,
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Estimate computational cost based on interval and state size
    fn estimate_cost(time_interval: &(F, F), state_size: usize) -> f64 {
        let dt = (time_interval.1 - time_interval.0).to_f64().unwrap_or(1.0);
        dt * state_size as f64
    }

    /// Check if this chunk can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Boundary conditions for inter-chunk communication
#[derive(Debug, Clone)]
pub struct BoundaryConditions<F: IntegrateFloat> {
    /// Left boundary values (from previous chunk)
    pub left_boundary: Option<BoundaryData<F>>,
    /// Right boundary values (from next chunk)
    pub right_boundary: Option<BoundaryData<F>>,
    /// Ghost cells for finite difference methods
    pub ghost_cells: Vec<F>,
    /// Coupling information for multi-physics
    pub coupling_data: HashMap<String, Array1<F>>,
}

impl<F: IntegrateFloat> Default for BoundaryConditions<F> {
    fn default() -> Self {
        Self {
            left_boundary: None,
            right_boundary: None,
            ghost_cells: Vec::new(),
            coupling_data: HashMap::new(),
        }
    }
}

/// Data at a boundary between chunks
#[derive(Debug, Clone)]
pub struct BoundaryData<F: IntegrateFloat> {
    /// Time at which boundary data is valid
    pub time: F,
    /// State values at boundary
    pub state: Array1<F>,
    /// Derivative values at boundary (for higher-order continuity)
    pub derivative: Option<Array1<F>>,
    /// Source chunk ID
    pub source_chunk: ChunkId,
}

/// Result of processing a work chunk
#[derive(Debug, Clone)]
pub struct ChunkResult<F: IntegrateFloat> {
    /// Chunk that was processed
    pub chunk_id: ChunkId,
    /// Node that processed the chunk
    pub node_id: NodeId,
    /// Time points in the solution
    pub time_points: Vec<F>,
    /// Solution states at each time point
    pub states: Vec<Array1<F>>,
    /// Final state for continuity with next chunk
    pub final_state: Array1<F>,
    /// Final derivative for higher-order continuity
    pub final_derivative: Option<Array1<F>>,
    /// Error estimate for this chunk
    pub error_estimate: F,
    /// Processing duration
    pub processing_time: Duration,
    /// Memory usage in bytes
    pub memory_used: usize,
    /// Status of the result
    pub status: ChunkResultStatus,
}

/// Status of a chunk result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkResultStatus {
    /// Chunk was processed successfully
    Success,
    /// Chunk processing failed
    Failed,
    /// Chunk needs to be reprocessed (e.g., tolerance not met)
    NeedsRefinement,
    /// Chunk was cancelled
    Cancelled,
}

/// Configuration for distributed computation
#[derive(Debug, Clone)]
pub struct DistributedConfig<F: IntegrateFloat> {
    /// Minimum chunk size (time interval)
    pub min_chunk_size: F,
    /// Maximum chunk size
    pub max_chunk_size: F,
    /// Target number of chunks per node
    pub chunks_per_node: usize,
    /// Tolerance for solution accuracy
    pub tolerance: F,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Enable checkpointing
    pub checkpointing_enabled: bool,
    /// Checkpoint interval (number of chunks)
    pub checkpoint_interval: usize,
    /// Communication timeout
    pub communication_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Maximum retries for failed chunks
    pub max_retries: u32,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Fault tolerance mode
    pub fault_tolerance: FaultToleranceMode,
}

impl<F: IntegrateFloat> Default for DistributedConfig<F> {
    fn default() -> Self {
        Self {
            min_chunk_size: F::from(0.001).unwrap_or(F::epsilon()),
            max_chunk_size: F::from(1.0).unwrap_or(F::one()),
            chunks_per_node: 4,
            tolerance: F::from(1e-6).unwrap_or(F::epsilon()),
            max_iterations: 1000,
            checkpointing_enabled: true,
            checkpoint_interval: 10,
            communication_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(5),
            max_retries: 3,
            load_balancing: LoadBalancingStrategy::Adaptive,
            fault_tolerance: FaultToleranceMode::Standard,
        }
    }
}

/// Load balancing strategies for distributing work
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Distribute based on node capabilities
    CapabilityBased,
    /// Dynamic work stealing
    WorkStealing,
    /// Adaptive strategy that adjusts based on performance
    Adaptive,
    /// Minimize communication by keeping related chunks together
    LocalityAware,
}

/// Fault tolerance modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultToleranceMode {
    /// No fault tolerance (fastest but risky)
    None,
    /// Standard fault tolerance with retries
    Standard,
    /// High availability with replication
    HighAvailability,
    /// Checkpoint-based recovery
    CheckpointRecovery,
}

/// Message types for inter-node communication
#[derive(Debug, Clone)]
pub enum DistributedMessage<F: IntegrateFloat> {
    /// Heartbeat message
    Heartbeat {
        node_id: NodeId,
        status: NodeStatus,
        timestamp: u64,
    },
    /// Work assignment message
    WorkAssignment {
        chunk: WorkChunk<F>,
        deadline: Option<Duration>,
    },
    /// Work result message
    WorkResult { result: ChunkResult<F> },
    /// Boundary data exchange
    BoundaryExchange {
        source_chunk: ChunkId,
        target_chunk: ChunkId,
        boundary_data: BoundaryData<F>,
    },
    /// Checkpoint request
    CheckpointRequest { job_id: JobId, checkpoint_id: u64 },
    /// Checkpoint data
    CheckpointData {
        job_id: JobId,
        checkpoint_id: u64,
        node_id: NodeId,
        data: Vec<u8>,
    },
    /// Node registration
    NodeRegister {
        node_id: NodeId,
        address: SocketAddr,
        capabilities: NodeCapabilities,
    },
    /// Node deregistration
    NodeDeregister { node_id: NodeId, reason: String },
    /// Job cancellation
    JobCancel { job_id: JobId, reason: String },
    /// Synchronization barrier
    SyncBarrier { barrier_id: u64, node_id: NodeId },
    /// Acknowledgment
    Ack { message_id: u64, status: AckStatus },
}

/// Acknowledgment status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckStatus {
    /// Message received and processed
    Ok,
    /// Message received but processing failed
    Error,
    /// Message not understood
    Unknown,
}

/// Metrics for monitoring distributed computation
#[derive(Debug, Clone, Default)]
pub struct DistributedMetrics {
    /// Total number of chunks processed
    pub chunks_processed: usize,
    /// Number of chunks failed
    pub chunks_failed: usize,
    /// Number of chunks retried
    pub chunks_retried: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Total communication time
    pub total_communication_time: Duration,
    /// Average chunk processing time
    pub average_chunk_time: Duration,
    /// Load balance efficiency (0.0 to 1.0)
    pub load_balance_efficiency: f64,
    /// Network bytes sent
    pub bytes_sent: usize,
    /// Network bytes received
    pub bytes_received: usize,
    /// Number of checkpoints created
    pub checkpoints_created: usize,
    /// Number of recoveries from failures
    pub recoveries: usize,
}

impl DistributedMetrics {
    /// Update load balance efficiency
    pub fn update_load_balance(&mut self, node_loads: &[f64]) {
        if node_loads.is_empty() {
            self.load_balance_efficiency = 1.0;
            return;
        }

        let mean_load: f64 = node_loads.iter().sum::<f64>() / node_loads.len() as f64;
        if mean_load <= 0.0 {
            self.load_balance_efficiency = 1.0;
            return;
        }

        let variance: f64 = node_loads
            .iter()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f64>()
            / node_loads.len() as f64;

        let cv = variance.sqrt() / mean_load; // Coefficient of variation
        self.load_balance_efficiency = (1.0 - cv.min(1.0)).max(0.0);
    }
}

/// Error types specific to distributed computing
#[derive(Debug, Clone)]
pub enum DistributedError {
    /// Node communication failure
    CommunicationError(String),
    /// Node timeout
    NodeTimeout(NodeId),
    /// Node failure
    NodeFailure(NodeId, String),
    /// Chunk processing error
    ChunkError(ChunkId, String),
    /// Synchronization error
    SyncError(String),
    /// Checkpoint error
    CheckpointError(String),
    /// Configuration error
    ConfigError(String),
    /// Resource exhaustion
    ResourceExhausted(String),
}

impl std::fmt::Display for DistributedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommunicationError(msg) => write!(f, "Communication error: {}", msg),
            Self::NodeTimeout(id) => write!(f, "Node {} timed out", id),
            Self::NodeFailure(id, msg) => write!(f, "Node {} failed: {}", id, msg),
            Self::ChunkError(id, msg) => write!(f, "Chunk {:?} error: {}", id, msg),
            Self::SyncError(msg) => write!(f, "Synchronization error: {}", msg),
            Self::CheckpointError(msg) => write!(f, "Checkpoint error: {}", msg),
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
        }
    }
}

impl std::error::Error for DistributedError {}

impl From<DistributedError> for IntegrateError {
    fn from(err: DistributedError) -> Self {
        IntegrateError::ComputationError(err.to_string())
    }
}

/// Result type for distributed operations
pub type DistributedResult<T> = std::result::Result<T, DistributedError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_node_id_display() {
        let id = NodeId::new(42);
        assert_eq!(format!("{}", id), "Node(42)");
    }

    #[test]
    fn test_node_info_health_check() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080);
        let mut node = NodeInfo::new(NodeId::new(1), addr);
        node.status = NodeStatus::Available;

        assert!(node.is_healthy(Duration::from_secs(60)));

        // Simulate old heartbeat
        node.last_heartbeat = Instant::now() - Duration::from_secs(120);
        assert!(!node.is_healthy(Duration::from_secs(60)));
    }

    #[test]
    fn test_work_chunk_retry() {
        let chunk: WorkChunk<f64> =
            WorkChunk::new(ChunkId::new(1), JobId::new(1), (0.0, 1.0), Array1::zeros(3));

        assert!(chunk.can_retry());
        let mut chunk = chunk;
        for _ in 0..3 {
            chunk.increment_retry();
        }
        assert!(!chunk.can_retry());
    }

    #[test]
    fn test_distributed_metrics_load_balance() {
        let mut metrics = DistributedMetrics::default();

        // Perfect balance
        metrics.update_load_balance(&[1.0, 1.0, 1.0, 1.0]);
        assert!((metrics.load_balance_efficiency - 1.0).abs() < 0.01);

        // Imbalanced
        metrics.update_load_balance(&[0.1, 0.1, 0.1, 3.7]);
        assert!(metrics.load_balance_efficiency < 0.5);
    }
}
