//! Distributed computing support for scirs2-integrate
//!
//! This module provides distributed computing capabilities for large-scale
//! ODE integration problems, enabling computation across multiple nodes
//! with load balancing, fault tolerance, and checkpointing.
//!
//! # Overview
//!
//! The distributed module enables solving large ODE systems by:
//! - Splitting the time domain into chunks
//! - Distributing chunks across multiple compute nodes
//! - Exchanging boundary conditions between adjacent chunks
//! - Providing fault tolerance through checkpointing
//! - Balancing load dynamically across nodes
//!
//! # Architecture
//!
//! ```text
//! +-------------------+
//! | DistributedSolver |
//! +-------------------+
//!         |
//!         v
//! +-------------------+     +----------------+
//! |   NodeManager     |---->|  LoadBalancer  |
//! +-------------------+     +----------------+
//!         |                        |
//!         v                        v
//! +-------------------+     +----------------+
//! |   ComputeNode     |     | WorkChunks     |
//! +-------------------+     +----------------+
//!         |
//!         v
//! +-------------------+     +----------------+
//! |  Communication    |---->| Checkpointing  |
//! +-------------------+     +----------------+
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use scirs2_integrate::distributed::{
//!     DistributedODESolver, DistributedODESolverBuilder, NodeInfo, NodeId,
//!     DistributedConfig, FaultToleranceMode,
//! };
//! use scirs2_core::ndarray::{array, ArrayView1, Array1};
//! use std::time::Duration;
//!
//! // Create a distributed solver
//! let solver = DistributedODESolverBuilder::new()
//!     .tolerance(1e-6)
//!     .chunks_per_node(4)
//!     .with_checkpointing(10)
//!     .fault_tolerance(FaultToleranceMode::Standard)
//!     .timeout(Duration::from_secs(300))
//!     .build()
//!     .expect("Failed to create solver");
//!
//! // Register compute nodes
//! // (In a real scenario, nodes would be on different machines)
//! let node1 = NodeInfo::new(
//!     NodeId::new(1),
//!     "127.0.0.1:8080".parse().unwrap()
//! );
//! solver.register_node(node1).expect("Failed to register node");
//!
//! // Define the ODE: y' = -y (exponential decay)
//! let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
//!     array![-y[0]]
//! };
//!
//! // Solve the ODE
//! let result = solver.solve(f, (0.0, 10.0), array![1.0], None)
//!     .expect("Solve failed");
//!
//! println!("Final value: {:?}", result.final_state());
//! println!("Chunks processed: {}", result.chunks_processed);
//! println!("Nodes used: {}", result.nodes_used);
//! ```
//!
//! # Features
//!
//! ## Multi-Node Computation
//!
//! The solver distributes work across multiple compute nodes:
//! - Automatic chunk generation based on time domain
//! - Work assignment using various load balancing strategies
//! - Support for heterogeneous node capabilities
//!
//! ## Communication Primitives
//!
//! Efficient inter-node communication:
//! - Message passing for work distribution
//! - Boundary data exchange between adjacent chunks
//! - Synchronization barriers for coordinated operations
//!
//! ## Load Balancing
//!
//! Multiple load balancing strategies:
//! - **Round Robin**: Simple cyclic distribution
//! - **Capability Based**: Assignment based on node resources
//! - **Work Stealing**: Dynamic load redistribution
//! - **Adaptive**: Performance-based strategy selection
//! - **Locality Aware**: Minimizes communication overhead
//!
//! ## Fault Tolerance
//!
//! Robust handling of failures:
//! - Automatic detection of node failures
//! - Chunk retry with exponential backoff
//! - Checkpoint-based recovery
//! - High availability mode with replication
//!
//! ## Checkpointing
//!
//! Periodic state saving for recovery:
//! - Configurable checkpoint intervals
//! - Disk persistence for durability
//! - Validation to ensure data integrity
//! - Efficient incremental checkpointing

pub mod checkpointing;
pub mod communication;
pub mod load_balancing;
pub mod node;
pub mod solver;
pub mod types;

// Re-export main types
pub use checkpointing::{
    Checkpoint, CheckpointConfig, CheckpointGlobalState, CheckpointManager, CheckpointStatistics,
    ChunkCheckpoint, FaultToleranceCoordinator, RecoveryAction,
};

pub use communication::{
    deserialize_boundary_data, serialize_boundary_data, BoundaryExchanger, Communicator,
    MessageChannel, SyncBarrier,
};

pub use load_balancing::{
    ChunkDistributor, LoadBalancer, LoadBalancerConfig, LoadBalancerStatistics, NodePerformance,
};

pub use node::{ComputeNode, NodeBuilder, NodeManager, ResourceMonitor};

pub use solver::{DistributedODEResult, DistributedODESolver, DistributedODESolverBuilder};

pub use types::{
    AckStatus, BoundaryConditions, BoundaryData, ChunkId, ChunkResult, ChunkResultStatus,
    DistributedConfig, DistributedError, DistributedMessage, DistributedMetrics, DistributedResult,
    FaultToleranceMode, FloatPrecision, JobId, LoadBalancingStrategy, NodeCapabilities, NodeId,
    NodeInfo, NodeStatus, SimdCapability, WorkChunk,
};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, ArrayView1};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn test_address(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    #[test]
    fn test_integration_simple_ode() {
        // Create solver
        let config = DistributedConfig::<f64>::default();
        let solver = DistributedODESolver::new(config).expect("Failed to create solver");

        // Register nodes
        for i in 0..2 {
            let mut node = NodeInfo::new(NodeId::new(i), test_address(8080 + i as u16));
            node.status = NodeStatus::Available;
            solver.register_node(node).expect("Failed to register");
        }

        // Define ODE: y' = -y
        let f = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

        // Solve
        let result = solver.solve(f, (0.0, 1.0), array![1.0], None);
        assert!(result.is_ok());

        let result = result.expect("Solve failed");
        assert!(!result.is_empty());

        // Check accuracy
        let expected = (-1.0_f64).exp();
        let actual = result.final_state().expect("No final state")[0];
        assert!((actual - expected).abs() < 0.05);
    }

    #[test]
    fn test_integration_with_checkpointing() {
        let mut config = DistributedConfig::<f64>::default();
        config.checkpointing_enabled = true;
        config.checkpoint_interval = 2;

        let solver = DistributedODESolver::new(config).expect("Failed to create solver");

        // Register a node
        let mut node = NodeInfo::new(NodeId::new(0), test_address(9000));
        node.status = NodeStatus::Available;
        solver.register_node(node).expect("Failed to register");

        // Solve a longer problem
        let f = |_t: f64, y: ArrayView1<f64>| array![-y[0]];
        let result = solver.solve(f, (0.0, 2.0), array![1.0], None);

        assert!(result.is_ok());
    }

    #[test]
    fn test_node_manager_integration() {
        use std::time::Duration;

        let manager = NodeManager::new(Duration::from_secs(30));

        // Register multiple nodes
        for i in 0..5 {
            let addr = test_address(7000 + i);
            let caps = NodeCapabilities::default();
            let result = manager.register_node(addr, caps);
            assert!(result.is_ok());
        }

        assert_eq!(manager.available_node_count(), 5);

        // Test best node selection
        let best = manager.select_best_node(1.0);
        assert!(best.is_some());
    }

    #[test]
    fn test_load_balancer_integration() {
        use std::time::Duration;

        let balancer: LoadBalancer<f64> = LoadBalancer::new(
            LoadBalancingStrategy::Adaptive,
            LoadBalancerConfig::default(),
        );

        // Register nodes
        for i in 0..3 {
            balancer
                .register_node(NodeId::new(i))
                .expect("Failed to register");
        }

        // Create test nodes
        let nodes: Vec<NodeInfo> = (0..3)
            .map(|i| {
                let mut node = NodeInfo::new(NodeId::new(i), test_address(6000 + i as u16));
                node.status = NodeStatus::Available;
                node
            })
            .collect();

        // Assign multiple chunks
        for i in 0..10 {
            let chunk = WorkChunk::new(ChunkId::new(i), JobId::new(1), (0.0, 1.0), array![1.0]);
            let result = balancer.assign_chunk(&chunk, &nodes);
            assert!(result.is_ok());
        }

        // Check statistics
        let stats = balancer.get_statistics();
        assert_eq!(stats.node_count, 3);
    }

    #[test]
    fn test_boundary_exchanger_integration() {
        use std::time::Duration;

        let exchanger: BoundaryExchanger<f64> = BoundaryExchanger::new(Duration::from_secs(5));

        let target = ChunkId::new(2);
        let source = ChunkId::new(1);

        // Request boundary
        exchanger
            .request_boundary(target, source)
            .expect("Request failed");

        // Receive boundary data
        let data = BoundaryData {
            time: 1.0,
            state: array![1.0, 2.0, 3.0],
            derivative: Some(array![-1.0, -2.0, -3.0]),
            source_chunk: source,
        };

        exchanger
            .receive_boundary(target, source, data.clone())
            .expect("Receive failed");

        // Get boundary conditions
        let bc = exchanger.build_boundary_conditions(target, Some(source), None);
        assert!(bc.left_boundary.is_some());
    }

    #[test]
    fn test_checkpoint_manager_integration() {
        use std::path::PathBuf;

        let path = std::env::temp_dir().join(format!("scirs_dist_test_{}", std::process::id()));
        let mut config = CheckpointConfig::default();
        config.persist_to_disk = false;

        let manager: CheckpointManager<f64> =
            CheckpointManager::new(path.clone(), config).expect("Failed to create manager");

        let job_id = JobId::new(1);

        // Create checkpoint with results
        let chunk_result = ChunkResult {
            chunk_id: ChunkId::new(1),
            node_id: NodeId::new(1),
            time_points: vec![0.0, 0.5, 1.0],
            states: vec![array![1.0], array![0.5], array![0.25]],
            final_state: array![0.25],
            final_derivative: Some(array![-0.25]),
            error_estimate: 1e-6,
            processing_time: std::time::Duration::from_millis(100),
            memory_used: 1024,
            status: ChunkResultStatus::Success,
        };

        let global_state = CheckpointGlobalState {
            iteration: 1,
            chunks_completed: 1,
            chunks_remaining: 5,
            current_time: 1.0,
            error_estimate: 1e-6,
        };

        let cp_id = manager
            .create_checkpoint(
                job_id,
                vec![chunk_result],
                vec![ChunkId::new(2)],
                global_state,
            )
            .expect("Failed to create checkpoint");

        assert!(cp_id > 0);

        // Restore checkpoint
        let restored = manager.restore(job_id, None).expect("Failed to restore");
        assert_eq!(restored.global_state.chunks_completed, 1);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_communicator_integration() {
        use std::sync::Arc;
        use std::time::Duration;

        let channel: Arc<MessageChannel<f64>> =
            Arc::new(MessageChannel::new(Duration::from_secs(5)));
        let exchanger: Arc<BoundaryExchanger<f64>> =
            Arc::new(BoundaryExchanger::new(Duration::from_secs(5)));

        let comm = Communicator::new(NodeId::new(1), channel, exchanger);

        // Add peers
        comm.add_peer(NodeId::new(2)).expect("Failed to add peer");
        comm.add_peer(NodeId::new(3)).expect("Failed to add peer");

        let peers = comm.get_peers();
        assert_eq!(peers.len(), 2);

        // Create barrier
        let barrier_id = comm.create_barrier(3).expect("Failed to create barrier");
        assert!(barrier_id > 0);
    }
}
