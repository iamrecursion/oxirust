//! Routing path solver for finding optimal routing paths.

use super::connection::{ConnectionManager, ConnectionType, RoutingPriority};
use std::collections::{HashMap, HashSet, VecDeque};

/// Represents a routing path from source to destination
#[derive(Debug, Clone)]
pub struct RoutingPath {
    /// Sequence of nodes in the path (input/output indices)
    pub nodes: Vec<usize>,
    /// Total gain through the path (in dB)
    pub total_gain_db: f32,
    /// Connection types used in the path
    pub connection_types: Vec<ConnectionType>,
    /// Priority of the path (minimum priority of all connections)
    pub priority: RoutingPriority,
}

impl RoutingPath {
    /// Create a new routing path
    #[must_use]
    pub fn new(nodes: Vec<usize>) -> Self {
        Self {
            nodes,
            total_gain_db: 0.0,
            connection_types: Vec::new(),
            priority: RoutingPriority::Normal,
        }
    }

    /// Get the source of this path
    #[must_use]
    pub fn source(&self) -> Option<usize> {
        self.nodes.first().copied()
    }

    /// Get the destination of this path
    #[must_use]
    pub fn destination(&self) -> Option<usize> {
        self.nodes.last().copied()
    }

    /// Get the number of hops in this path
    #[must_use]
    pub fn hop_count(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }
}

/// Routing path solver for finding optimal paths
pub struct RoutingPathSolver {
    /// Connection manager
    connection_manager: ConnectionManager,
}

impl RoutingPathSolver {
    /// Create a new routing path solver
    #[must_use]
    pub fn new(connection_manager: ConnectionManager) -> Self {
        Self { connection_manager }
    }

    /// Find all paths from source to destination
    #[must_use]
    pub fn find_paths(&self, source: usize, destination: usize) -> Vec<RoutingPath> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = Vec::new();

        self.find_paths_recursive(
            source,
            destination,
            &mut visited,
            &mut current_path,
            &mut paths,
        );

        paths
    }

    /// Recursive helper for finding paths
    fn find_paths_recursive(
        &self,
        current: usize,
        destination: usize,
        visited: &mut HashSet<usize>,
        current_path: &mut Vec<usize>,
        paths: &mut Vec<RoutingPath>,
    ) {
        visited.insert(current);
        current_path.push(current);

        if current == destination {
            // Found a complete path
            let mut path = RoutingPath::new(current_path.clone());
            self.calculate_path_metrics(&mut path);
            paths.push(path);
        } else {
            // Continue searching
            let connections = self.connection_manager.get_connections_from_source(current);
            for conn in connections {
                if conn.active && !visited.contains(&conn.destination) {
                    self.find_paths_recursive(
                        conn.destination,
                        destination,
                        visited,
                        current_path,
                        paths,
                    );
                }
            }
        }

        current_path.pop();
        visited.remove(&current);
    }

    /// Calculate metrics for a path
    fn calculate_path_metrics(&self, path: &mut RoutingPath) {
        let mut total_gain = 0.0;
        let mut connection_types = Vec::new();
        let mut min_priority = RoutingPriority::Critical;

        for i in 0..path.nodes.len().saturating_sub(1) {
            let source = path.nodes[i];
            let dest = path.nodes[i + 1];

            // Find the connection between these nodes
            let connections = self.connection_manager.get_connections_from_source(source);
            for conn in connections {
                if conn.destination == dest && conn.active {
                    total_gain += conn.gain_db;
                    connection_types.push(conn.connection_type);

                    // Update minimum priority
                    if (conn.priority as u8) < (min_priority as u8) {
                        min_priority = conn.priority;
                    }
                    break;
                }
            }
        }

        path.total_gain_db = total_gain;
        path.connection_types = connection_types;
        path.priority = min_priority;
    }

    /// Find the shortest path from source to destination (fewest hops)
    #[must_use]
    pub fn find_shortest_path(&self, source: usize, destination: usize) -> Option<RoutingPath> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent_map: HashMap<usize, usize> = HashMap::new();

        queue.push_back(source);
        visited.insert(source);

        while let Some(current) = queue.pop_front() {
            if current == destination {
                // Reconstruct path
                let mut path_nodes = Vec::new();
                let mut node = destination;
                path_nodes.push(node);

                while let Some(&parent) = parent_map.get(&node) {
                    path_nodes.push(parent);
                    node = parent;
                }

                path_nodes.reverse();
                let mut path = RoutingPath::new(path_nodes);
                self.calculate_path_metrics(&mut path);
                return Some(path);
            }

            let connections = self.connection_manager.get_connections_from_source(current);
            for conn in connections {
                if conn.active && !visited.contains(&conn.destination) {
                    visited.insert(conn.destination);
                    parent_map.insert(conn.destination, current);
                    queue.push_back(conn.destination);
                }
            }
        }

        None
    }

    /// Detect if there are any routing loops
    #[must_use]
    pub fn detect_loops(&self) -> Vec<Vec<usize>> {
        let mut loops = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        // Get all unique nodes
        let active_connections = self.connection_manager.get_active_connections();
        let mut nodes = HashSet::new();
        for conn in &active_connections {
            nodes.insert(conn.source);
            nodes.insert(conn.destination);
        }

        for &node in &nodes {
            if !visited.contains(&node) {
                self.detect_loops_dfs(node, &mut visited, &mut rec_stack, &mut path, &mut loops);
            }
        }

        loops
    }

    /// DFS helper for loop detection
    fn detect_loops_dfs(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        rec_stack: &mut HashSet<usize>,
        path: &mut Vec<usize>,
        loops: &mut Vec<Vec<usize>>,
    ) {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);

        let connections = self.connection_manager.get_connections_from_source(node);
        for conn in connections {
            if !conn.active {
                continue;
            }

            if !visited.contains(&conn.destination) {
                self.detect_loops_dfs(conn.destination, visited, rec_stack, path, loops);
            } else if rec_stack.contains(&conn.destination) {
                // Found a loop
                if let Some(loop_start) = path.iter().position(|&n| n == conn.destination) {
                    let loop_path = path[loop_start..].to_vec();
                    loops.push(loop_path);
                }
            }
        }

        path.pop();
        rec_stack.remove(&node);
    }

    /// Check if routing from source to destination would create a loop
    #[must_use]
    pub fn would_create_loop(&self, source: usize, destination: usize) -> bool {
        // Check if there's already a path from destination to source
        self.find_shortest_path(destination, source).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_path() {
        let path = RoutingPath::new(vec![0, 1, 2, 3]);
        assert_eq!(path.source(), Some(0));
        assert_eq!(path.destination(), Some(3));
        assert_eq!(path.hop_count(), 3);
    }

    #[test]
    fn test_find_shortest_path() {
        let mut manager = ConnectionManager::new();

        // Create a simple path: 0 -> 1 -> 2
        manager.add_connection(0, 1, ConnectionType::Analog);
        manager.add_connection(1, 2, ConnectionType::Analog);

        let solver = RoutingPathSolver::new(manager);
        let path = solver.find_shortest_path(0, 2);

        assert!(path.is_some());
        let path = path.expect("should succeed in test");
        assert_eq!(path.nodes, vec![0, 1, 2]);
        assert_eq!(path.hop_count(), 2);
    }

    #[test]
    fn test_no_path() {
        let manager = ConnectionManager::new();
        let solver = RoutingPathSolver::new(manager);

        let path = solver.find_shortest_path(0, 2);
        assert!(path.is_none());
    }

    #[test]
    fn test_loop_detection() {
        let mut manager = ConnectionManager::new();

        // Create a loop: 0 -> 1 -> 2 -> 0
        manager.add_connection(0, 1, ConnectionType::Analog);
        manager.add_connection(1, 2, ConnectionType::Analog);
        manager.add_connection(2, 0, ConnectionType::Analog);

        let solver = RoutingPathSolver::new(manager);
        let loops = solver.detect_loops();

        assert!(!loops.is_empty());
    }

    #[test]
    fn test_would_create_loop() {
        let mut manager = ConnectionManager::new();

        // Create path: 0 -> 1 -> 2
        manager.add_connection(0, 1, ConnectionType::Analog);
        manager.add_connection(1, 2, ConnectionType::Analog);

        let solver = RoutingPathSolver::new(manager);

        // Connecting 2 -> 0 would create a loop
        assert!(solver.would_create_loop(2, 0));

        // Connecting 2 -> 3 would not
        assert!(!solver.would_create_loop(2, 3));
    }

    #[test]
    fn test_find_all_paths() {
        let mut manager = ConnectionManager::new();

        // Create multiple paths from 0 to 2
        // Path 1: 0 -> 2 (direct)
        // Path 2: 0 -> 1 -> 2
        manager.add_connection(0, 2, ConnectionType::Analog);
        manager.add_connection(0, 1, ConnectionType::Analog);
        manager.add_connection(1, 2, ConnectionType::Analog);

        let solver = RoutingPathSolver::new(manager);
        let paths = solver.find_paths(0, 2);

        assert_eq!(paths.len(), 2);

        // One path should be direct, one through node 1
        let has_direct = paths.iter().any(|p| p.nodes == vec![0, 2]);
        let has_indirect = paths.iter().any(|p| p.nodes == vec![0, 1, 2]);

        assert!(has_direct);
        assert!(has_indirect);
    }
}
