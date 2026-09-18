//! Load balancing algorithms for distributed systems
//!
//! This module provides various load balancing strategies to distribute
//! computational workloads efficiently across cluster nodes.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Load balancing strategy
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections strategy
    LeastConnections,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Resource-aware balancing
    ResourceAware,
    /// Latency-based balancing
    LatencyBased,
}

/// Node load information
#[derive(Debug, Clone)]
pub struct NodeLoad {
    pub nodeid: String,
    pub address: SocketAddr,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub active_connections: usize,
    pub average_latency: f64,
    pub weight: f64,
    pub last_updated: Instant,
}

impl NodeLoad {
    /// Create new node load info
    pub fn new(nodeid: String, address: SocketAddr) -> Self {
        Self {
            nodeid,
            address,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            active_connections: 0,
            average_latency: 0.0,
            weight: 1.0,
            last_updated: Instant::now(),
        }
    }

    /// Calculate overall load score (0.0 = no load, 1.0 = maximum load)
    pub fn load_score(&self) -> f64 {
        let cpu_score = self.cpu_utilization;
        let memory_score = self.memory_utilization;
        let connection_score = (self.active_connections as f64) / 1000.0; // Normalize to 1000 max connections
        let latency_score = self.average_latency / 1000.0; // Normalize to 1000ms

        (cpu_score + memory_score + connection_score + latency_score) / 4.0
    }

    /// Check if node is available for new tasks
    pub fn is_available(&self) -> bool {
        self.cpu_utilization < 0.9 && self.memory_utilization < 0.9
    }

    /// Update load metrics
    pub fn update_metrics(&mut self, cpu: f64, memory: f64, connections: usize, latency: f64) {
        self.cpu_utilization = cpu;
        self.memory_utilization = memory;
        self.active_connections = connections;
        self.average_latency = latency;
        self.last_updated = Instant::now();
    }
}

/// Task assignment result
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    pub taskid: String,
    pub nodeid: String,
    pub node_address: SocketAddr,
    pub assigned_at: Instant,
}

/// Load balancer implementation
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    nodes: Arc<Mutex<HashMap<String, NodeLoad>>>,
    round_robin_index: Arc<Mutex<usize>>,
    assignment_history: Arc<Mutex<VecDeque<TaskAssignment>>>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            nodes: Arc::new(Mutex::new(HashMap::new())),
            round_robin_index: Arc::new(Mutex::new(0)),
            assignment_history: Arc::new(Mutex::new(VecDeque::with_capacity(10000))),
        }
    }

    /// Register a node for load balancing
    pub fn register_node(&self, nodeload: NodeLoad) -> CoreResult<()> {
        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;
        nodes.insert(nodeload.nodeid.clone(), nodeload);
        Ok(())
    }

    /// Update node load metrics
    pub fn update_nodeload(
        &self,
        nodeid: &str,
        cpu: f64,
        memory: f64,
        connections: usize,
        latency: f64,
    ) -> CoreResult<()> {
        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        if let Some(node) = nodes.get_mut(nodeid) {
            node.update_metrics(cpu, memory, connections, latency);
        } else {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Unknown node: {nodeid}"
            ))));
        }
        Ok(())
    }

    /// Assign a task to the best available node
    pub fn assign_task(&self, taskid: String) -> CoreResult<TaskAssignment> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let availablenodes: Vec<_> = nodes
            .values()
            .filter(|node| node.is_available())
            .cloned()
            .collect();

        if availablenodes.is_empty() {
            return Err(CoreError::InvalidState(ErrorContext::new(
                "No available nodes for task assignment".to_string(),
            )));
        }

        let selected_node = match &self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&availablenodes)?,
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(&availablenodes)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.select_weighted_round_robin(&availablenodes)?
            }
            LoadBalancingStrategy::ResourceAware => self.select_resource_aware(&availablenodes),
            LoadBalancingStrategy::LatencyBased => self.select_latencybased(&availablenodes),
        };

        let assignment = TaskAssignment {
            taskid,
            nodeid: selected_node.nodeid.clone(),
            node_address: selected_node.address,
            assigned_at: Instant::now(),
        };

        // Record assignment history
        let mut history = self.assignment_history.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire history lock".to_string(),
            ))
        })?;
        history.push_back(assignment.clone());
        if history.len() > 10000 {
            history.pop_front();
        }

        Ok(assignment)
    }

    fn select_round_robin(&self, nodes: &[NodeLoad]) -> CoreResult<NodeLoad> {
        let mut index = self.round_robin_index.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire index lock".to_string(),
            ))
        })?;

        let selected = nodes[*index % nodes.len()].clone();
        *index += 1;
        Ok(selected)
    }

    fn select_least_connections(&self, nodes: &[NodeLoad]) -> NodeLoad {
        nodes
            .iter()
            .min_by_key(|node| node.active_connections)
            .expect("Operation failed")
            .clone()
    }

    fn select_weighted_round_robin(&self, nodes: &[NodeLoad]) -> CoreResult<NodeLoad> {
        // Simple weighted selection based on inverse load score
        let weights: Vec<f64> = nodes.iter()
            .map(|node| 1.0 / (node.load_score() + 0.1)) // Add small value to avoid division by zero
            .collect();

        let total_weight: f64 = weights.iter().sum();
        let mut cumulative_weight = 0.0;
        let target = total_weight * 0.5; // Select middle-weighted node for simplicity

        for (i, weight) in weights.iter().enumerate() {
            cumulative_weight += weight;
            if cumulative_weight >= target {
                return Ok(nodes[i].clone());
            }
        }

        Ok(nodes[0].clone()) // Fallback
    }

    fn select_resource_aware(&self, nodes: &[NodeLoad]) -> NodeLoad {
        nodes
            .iter()
            .min_by(|a, b| {
                a.load_score()
                    .partial_cmp(&b.load_score())
                    .expect("Operation failed")
            })
            .expect("Operation failed")
            .clone()
    }

    fn select_latencybased(&self, nodes: &[NodeLoad]) -> NodeLoad {
        nodes
            .iter()
            .min_by(|a, b| {
                a.average_latency
                    .partial_cmp(&b.average_latency)
                    .expect("Operation failed")
            })
            .expect("Operation failed")
            .clone()
    }

    /// Get load balancing statistics
    pub fn get_statistics(&self) -> CoreResult<LoadBalancingStats> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let history = self.assignment_history.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire history lock".to_string(),
            ))
        })?;

        let total_nodes = nodes.len();
        let availablenodes = nodes.values().filter(|node| node.is_available()).count();
        let total_assignments = history.len();

        // Calculate assignment distribution
        let mut assignment_counts: HashMap<String, usize> = HashMap::new();
        for assignment in history.iter() {
            *assignment_counts
                .entry(assignment.nodeid.clone())
                .or_insert(0) += 1;
        }

        let average_load = if !nodes.is_empty() {
            nodes.values().map(|node| node.load_score()).sum::<f64>() / nodes.len() as f64
        } else {
            0.0
        };

        Ok(LoadBalancingStats {
            total_nodes,
            availablenodes,
            total_assignments,
            assignment_distribution: assignment_counts,
            average_load,
            strategy: self.strategy.clone(),
        })
    }

    /// Get all node loads
    pub fn get_all_nodes(&self) -> CoreResult<Vec<NodeLoad>> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;
        Ok(nodes.values().cloned().collect())
    }
}

/// Load balancing statistics
#[derive(Debug)]
pub struct LoadBalancingStats {
    pub total_nodes: usize,
    pub availablenodes: usize,
    pub total_assignments: usize,
    pub assignment_distribution: HashMap<String, usize>,
    pub average_load: f64,
    pub strategy: LoadBalancingStrategy,
}

impl LoadBalancingStats {
    /// Calculate load distribution balance (0.0 = perfect balance, 1.0 = maximum imbalance)
    pub fn balance_score(&self) -> f64 {
        if self.assignment_distribution.is_empty() || self.total_assignments == 0 {
            return 0.0;
        }

        let average_assignments =
            self.total_assignments as f64 / self.assignment_distribution.len() as f64;
        let variance: f64 = self
            .assignment_distribution
            .values()
            .map(|&count| {
                let diff = count as f64 - average_assignments;
                diff * diff
            })
            .sum::<f64>()
            / self.assignment_distribution.len() as f64;

        (variance.sqrt() / average_assignments).min(1.0)
    }

    /// Check if load is well balanced
    pub fn is_well_balanced(&self) -> bool {
        self.balance_score() < 0.2 // Less than 20% imbalance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_nodeload_creation() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeLoad::new("node1".to_string(), address);

        assert_eq!(node.nodeid, "node1");
        assert_eq!(node.address, address);
        assert!(node.is_available());
        assert_eq!(node.load_score(), 0.0);
    }

    #[test]
    fn test_nodeload_update() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut node = NodeLoad::new("node1".to_string(), address);

        node.update_metrics(0.5, 0.6, 10, 50.0);
        assert_eq!(node.cpu_utilization, 0.5);
        assert_eq!(node.memory_utilization, 0.6);
        assert_eq!(node.active_connections, 10);
        assert_eq!(node.average_latency, 50.0);
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let address1 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node1 = NodeLoad::new("node1".to_string(), address1);

        let address2 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8081);
        let node2 = NodeLoad::new("node2".to_string(), address2);

        assert!(balancer.register_node(node1).is_ok());
        assert!(balancer.register_node(node2).is_ok());

        let assignment1 = balancer
            .assign_task("task1".to_string())
            .expect("Operation failed");
        let assignment2 = balancer
            .assign_task("task2".to_string())
            .expect("Operation failed");

        // Should alternate between nodes
        assert_ne!(assignment1.nodeid, assignment2.nodeid);
    }

    #[test]
    fn test_load_balancing_stats() {
        let mut stats = LoadBalancingStats {
            total_nodes: 3,
            availablenodes: 3,
            total_assignments: 100,
            assignment_distribution: HashMap::new(),
            average_load: 0.5,
            strategy: LoadBalancingStrategy::RoundRobin,
        };

        // Perfect balance
        stats
            .assignment_distribution
            .insert("node1".to_string(), 33);
        stats
            .assignment_distribution
            .insert("node2".to_string(), 33);
        stats
            .assignment_distribution
            .insert("node3".to_string(), 34);

        assert!(stats.is_well_balanced());
        assert!(stats.balance_score() < 0.1);
    }
}
