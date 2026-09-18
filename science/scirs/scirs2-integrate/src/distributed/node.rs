//! Node management for distributed computing
//!
//! This module provides abstractions for compute nodes in the distributed
//! integration system, including node lifecycle management, resource tracking,
//! and health monitoring.

use crate::common::IntegrateFloat;
use crate::distributed::types::{
    ChunkResult, DistributedError, DistributedResult, NodeCapabilities, NodeId, NodeInfo,
    NodeStatus, SimdCapability, WorkChunk,
};
use crate::error::IntegrateResult;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Manager for compute nodes in the distributed system
pub struct NodeManager {
    /// Registered nodes
    nodes: RwLock<HashMap<NodeId, NodeInfo>>,
    /// Next node ID to assign
    next_node_id: AtomicU64,
    /// Node health check timeout
    health_check_timeout: Duration,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Health monitor handle
    health_monitor: Mutex<Option<thread::JoinHandle<()>>>,
    /// Node failure callbacks
    failure_callbacks: RwLock<Vec<Arc<dyn Fn(NodeId) + Send + Sync>>>,
}

impl NodeManager {
    /// Create a new node manager
    pub fn new(health_check_timeout: Duration) -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            next_node_id: AtomicU64::new(1),
            health_check_timeout,
            shutdown: AtomicBool::new(false),
            health_monitor: Mutex::new(None),
            failure_callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Start the health monitoring background thread
    pub fn start_health_monitoring(&self) -> IntegrateResult<()> {
        let nodes = unsafe { &*(&self.nodes as *const RwLock<HashMap<NodeId, NodeInfo>>) };
        let timeout = self.health_check_timeout;
        let shutdown = unsafe { &*(&self.shutdown as *const AtomicBool) };
        let callbacks = unsafe {
            &*(&self.failure_callbacks as *const RwLock<Vec<Arc<dyn Fn(NodeId) + Send + Sync>>>)
        };

        // Create references for the thread
        let nodes_ptr = nodes as *const RwLock<HashMap<NodeId, NodeInfo>> as usize;
        let shutdown_ptr = shutdown as *const AtomicBool as usize;
        let callbacks_ptr =
            callbacks as *const RwLock<Vec<Arc<dyn Fn(NodeId) + Send + Sync>>> as usize;

        let handle = thread::spawn(move || {
            let nodes = unsafe { &*(nodes_ptr as *const RwLock<HashMap<NodeId, NodeInfo>>) };
            let shutdown = unsafe { &*(shutdown_ptr as *const AtomicBool) };
            let callbacks = unsafe {
                &*(callbacks_ptr as *const RwLock<Vec<Arc<dyn Fn(NodeId) + Send + Sync>>>)
            };

            while !shutdown.load(Ordering::Relaxed) {
                // Check node health
                let failed_nodes = {
                    let mut nodes_write = match nodes.write() {
                        Ok(guard) => guard,
                        Err(_) => continue,
                    };

                    let mut failed = Vec::new();
                    for (id, info) in nodes_write.iter_mut() {
                        if !info.is_healthy(timeout) && info.status != NodeStatus::Failed {
                            info.status = NodeStatus::Failed;
                            failed.push(*id);
                        }
                    }
                    failed
                };

                // Invoke failure callbacks
                if !failed_nodes.is_empty() {
                    if let Ok(cbs) = callbacks.read() {
                        for node_id in &failed_nodes {
                            for cb in cbs.iter() {
                                cb(*node_id);
                            }
                        }
                    }
                }

                thread::sleep(Duration::from_secs(1));
            }
        });

        if let Ok(mut monitor) = self.health_monitor.lock() {
            *monitor = Some(handle);
        }

        Ok(())
    }

    /// Stop health monitoring
    pub fn stop_health_monitoring(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Ok(mut monitor) = self.health_monitor.lock() {
            if let Some(handle) = monitor.take() {
                let _ = handle.join();
            }
        }
    }

    /// Register a new node
    pub fn register_node(
        &self,
        address: SocketAddr,
        capabilities: NodeCapabilities,
    ) -> DistributedResult<NodeId> {
        let node_id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::SeqCst));

        let mut node_info = NodeInfo::new(node_id, address);
        node_info.capabilities = capabilities;
        node_info.status = NodeStatus::Available;

        match self.nodes.write() {
            Ok(mut nodes) => {
                nodes.insert(node_id, node_info);
                Ok(node_id)
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to acquire nodes lock".to_string(),
            )),
        }
    }

    /// Deregister a node
    pub fn deregister_node(&self, node_id: NodeId) -> DistributedResult<()> {
        match self.nodes.write() {
            Ok(mut nodes) => {
                nodes.remove(&node_id);
                Ok(())
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to acquire nodes lock".to_string(),
            )),
        }
    }

    /// Update node heartbeat
    pub fn update_heartbeat(&self, node_id: NodeId) -> DistributedResult<()> {
        match self.nodes.write() {
            Ok(mut nodes) => {
                if let Some(node) = nodes.get_mut(&node_id) {
                    node.last_heartbeat = Instant::now();
                    if node.status == NodeStatus::Failed {
                        node.status = NodeStatus::Available;
                    }
                    Ok(())
                } else {
                    Err(DistributedError::NodeFailure(
                        node_id,
                        "Node not found".to_string(),
                    ))
                }
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to acquire nodes lock".to_string(),
            )),
        }
    }

    /// Update node status
    pub fn update_status(&self, node_id: NodeId, status: NodeStatus) -> DistributedResult<()> {
        match self.nodes.write() {
            Ok(mut nodes) => {
                if let Some(node) = nodes.get_mut(&node_id) {
                    node.status = status;
                    Ok(())
                } else {
                    Err(DistributedError::NodeFailure(
                        node_id,
                        "Node not found".to_string(),
                    ))
                }
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to acquire nodes lock".to_string(),
            )),
        }
    }

    /// Get list of available nodes
    pub fn get_available_nodes(&self) -> Vec<NodeInfo> {
        match self.nodes.read() {
            Ok(nodes) => nodes
                .values()
                .filter(|n| n.status == NodeStatus::Available)
                .cloned()
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Get all registered nodes
    pub fn get_all_nodes(&self) -> Vec<NodeInfo> {
        match self.nodes.read() {
            Ok(nodes) => nodes.values().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<NodeInfo> {
        match self.nodes.read() {
            Ok(nodes) => nodes.get(&node_id).cloned(),
            Err(_) => None,
        }
    }

    /// Get number of available nodes
    pub fn available_node_count(&self) -> usize {
        self.get_available_nodes().len()
    }

    /// Register a failure callback
    pub fn on_node_failure<F>(&self, callback: F)
    where
        F: Fn(NodeId) + Send + Sync + 'static,
    {
        if let Ok(mut callbacks) = self.failure_callbacks.write() {
            callbacks.push(Arc::new(callback));
        }
    }

    /// Record job completion for a node
    pub fn record_job_completion(
        &self,
        node_id: NodeId,
        duration: Duration,
    ) -> DistributedResult<()> {
        match self.nodes.write() {
            Ok(mut nodes) => {
                if let Some(node) = nodes.get_mut(&node_id) {
                    let total_time = node.average_job_duration * node.jobs_completed as u32;
                    node.jobs_completed += 1;
                    node.average_job_duration =
                        (total_time + duration) / node.jobs_completed as u32;
                    Ok(())
                } else {
                    Err(DistributedError::NodeFailure(
                        node_id,
                        "Node not found".to_string(),
                    ))
                }
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to acquire nodes lock".to_string(),
            )),
        }
    }

    /// Select best node for a given workload
    pub fn select_best_node(&self, estimated_cost: f64) -> Option<NodeId> {
        match self.nodes.read() {
            Ok(nodes) => nodes
                .values()
                .filter(|n| n.status == NodeStatus::Available)
                .max_by(|a, b| {
                    a.processing_score()
                        .partial_cmp(&b.processing_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|n| n.id),
            Err(_) => None,
        }
    }
}

impl Drop for NodeManager {
    fn drop(&mut self) {
        self.stop_health_monitoring();
    }
}

/// A compute node that can process work chunks
pub struct ComputeNode<F: IntegrateFloat> {
    /// Node information
    info: NodeInfo,
    /// Current work queue
    work_queue: Mutex<Vec<WorkChunk<F>>>,
    /// Results buffer
    results: Mutex<Vec<ChunkResult<F>>>,
    /// Processing thread handles
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// ODE solver function
    solver_fn: Arc<dyn Fn(&WorkChunk<F>) -> IntegrateResult<ChunkResult<F>> + Send + Sync>,
}

impl<F: IntegrateFloat> ComputeNode<F> {
    /// Create a new compute node
    pub fn new<S>(info: NodeInfo, solver_fn: S) -> Self
    where
        S: Fn(&WorkChunk<F>) -> IntegrateResult<ChunkResult<F>> + Send + Sync + 'static,
    {
        Self {
            info,
            work_queue: Mutex::new(Vec::new()),
            results: Mutex::new(Vec::new()),
            workers: Mutex::new(Vec::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
            solver_fn: Arc::new(solver_fn),
        }
    }

    /// Get node ID
    pub fn id(&self) -> NodeId {
        self.info.id
    }

    /// Get node status
    pub fn status(&self) -> NodeStatus {
        self.info.status
    }

    /// Submit a work chunk
    pub fn submit_work(&self, chunk: WorkChunk<F>) -> DistributedResult<()> {
        match self.work_queue.lock() {
            Ok(mut queue) => {
                queue.push(chunk);
                Ok(())
            }
            Err(_) => Err(DistributedError::ResourceExhausted(
                "Failed to acquire work queue lock".to_string(),
            )),
        }
    }

    /// Process all queued work
    pub fn process_all(&self) -> DistributedResult<Vec<ChunkResult<F>>> {
        let chunks = {
            match self.work_queue.lock() {
                Ok(mut queue) => std::mem::take(&mut *queue),
                Err(_) => {
                    return Err(DistributedError::ResourceExhausted(
                        "Failed to acquire work queue lock".to_string(),
                    ))
                }
            }
        };

        let mut results = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            match (self.solver_fn)(&chunk) {
                Ok(result) => results.push(result),
                Err(e) => {
                    return Err(DistributedError::ChunkError(
                        chunk.id,
                        format!("Solver error: {}", e),
                    ))
                }
            }
        }

        Ok(results)
    }

    /// Get pending work count
    pub fn pending_work_count(&self) -> usize {
        match self.work_queue.lock() {
            Ok(queue) => queue.len(),
            Err(_) => 0,
        }
    }

    /// Collect completed results
    pub fn collect_results(&self) -> Vec<ChunkResult<F>> {
        match self.results.lock() {
            Ok(mut results) => std::mem::take(&mut *results),
            Err(_) => Vec::new(),
        }
    }

    /// Shutdown the node
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

/// Builder for creating compute nodes with detected capabilities
pub struct NodeBuilder {
    address: SocketAddr,
    capabilities: Option<NodeCapabilities>,
}

impl NodeBuilder {
    /// Create a new node builder
    pub fn new(address: SocketAddr) -> Self {
        Self {
            address,
            capabilities: None,
        }
    }

    /// Set custom capabilities
    pub fn with_capabilities(mut self, capabilities: NodeCapabilities) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    /// Auto-detect capabilities
    pub fn detect_capabilities(mut self) -> Self {
        self.capabilities = Some(Self::detect_system_capabilities());
        self
    }

    /// Detect system capabilities
    fn detect_system_capabilities() -> NodeCapabilities {
        let cpu_cores = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Estimate available memory (simplified)
        #[cfg(target_pointer_width = "32")]
        let memory_bytes = 512 * 1024 * 1024; // 512MB default for 32-bit
        #[cfg(target_pointer_width = "64")]
        let memory_bytes = 8usize * 1024 * 1024 * 1024; // 8GB default for 64-bit

        // Detect SIMD capabilities
        let simd_capabilities = Self::detect_simd();

        NodeCapabilities {
            cpu_cores,
            memory_bytes,
            has_gpu: false, // Would need GPU detection library
            gpu_memory_bytes: None,
            network_bandwidth: 1024 * 1024 * 1024, // 1 Gbps
            latency_us: 100,
            supported_precisions: vec![
                crate::distributed::types::FloatPrecision::F32,
                crate::distributed::types::FloatPrecision::F64,
            ],
            simd_capabilities,
        }
    }

    /// Detect SIMD capabilities
    fn detect_simd() -> SimdCapability {
        SimdCapability {
            has_sse: cfg!(target_feature = "sse"),
            has_sse2: cfg!(target_feature = "sse2"),
            has_avx: cfg!(target_feature = "avx"),
            has_avx2: cfg!(target_feature = "avx2"),
            has_avx512: cfg!(target_feature = "avx512f"),
            has_neon: cfg!(target_feature = "neon"),
        }
    }

    /// Build the node info
    pub fn build(self, node_id: NodeId) -> NodeInfo {
        let capabilities = self
            .capabilities
            .unwrap_or_else(Self::detect_system_capabilities);
        let mut info = NodeInfo::new(node_id, self.address);
        info.capabilities = capabilities;
        info.status = NodeStatus::Available;
        info
    }
}

/// Resource monitor for tracking node resource usage
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// CPU usage (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Memory usage (0.0 to 1.0)
    pub memory_usage: f64,
    /// Network usage (bytes/sec)
    pub network_usage: usize,
    /// GPU usage (0.0 to 1.0), if available
    pub gpu_usage: Option<f64>,
    /// Last update time
    pub last_update: Instant,
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            network_usage: 0,
            gpu_usage: None,
            last_update: Instant::now(),
        }
    }
}

impl ResourceMonitor {
    /// Update resource usage (simplified implementation)
    pub fn update(&mut self) {
        // In a real implementation, this would query system resources
        self.last_update = Instant::now();
    }

    /// Check if resources are available
    pub fn has_available_resources(&self, required_memory_fraction: f64) -> bool {
        self.memory_usage + required_memory_fraction <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_address() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080)
    }

    #[test]
    fn test_node_manager_registration() {
        let manager = NodeManager::new(Duration::from_secs(30));

        let node_id = manager
            .register_node(test_address(), NodeCapabilities::default())
            .expect("Failed to register node");

        assert_eq!(manager.available_node_count(), 1);

        let node = manager.get_node(node_id);
        assert!(node.is_some());
        assert_eq!(node.map(|n| n.id), Some(node_id));
    }

    #[test]
    fn test_node_manager_deregistration() {
        let manager = NodeManager::new(Duration::from_secs(30));

        let node_id = manager
            .register_node(test_address(), NodeCapabilities::default())
            .expect("Failed to register node");

        assert_eq!(manager.available_node_count(), 1);

        manager
            .deregister_node(node_id)
            .expect("Failed to deregister node");
        assert_eq!(manager.available_node_count(), 0);
    }

    #[test]
    fn test_node_manager_heartbeat() {
        let manager = NodeManager::new(Duration::from_secs(30));

        let node_id = manager
            .register_node(test_address(), NodeCapabilities::default())
            .expect("Failed to register node");

        manager
            .update_heartbeat(node_id)
            .expect("Failed to update heartbeat");

        let node = manager.get_node(node_id).expect("Node not found");
        assert!(node.is_healthy(Duration::from_secs(60)));
    }

    #[test]
    fn test_node_builder() {
        let addr = test_address();
        let node_info = NodeBuilder::new(addr)
            .detect_capabilities()
            .build(NodeId::new(1));

        assert_eq!(node_info.id, NodeId::new(1));
        assert_eq!(node_info.address, addr);
        assert!(node_info.capabilities.cpu_cores > 0);
    }

    #[test]
    fn test_resource_monitor() {
        let mut monitor = ResourceMonitor::default();
        assert!(monitor.has_available_resources(0.5));

        monitor.memory_usage = 0.8;
        assert!(!monitor.has_available_resources(0.3));
    }
}
