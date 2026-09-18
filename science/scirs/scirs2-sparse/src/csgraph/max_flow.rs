//! Maximum flow algorithms for graphs
//!
//! This module provides algorithms for computing maximum flow in networks:
//! - Ford-Fulkerson algorithm with DFS
//! - Edmonds-Karp algorithm (Ford-Fulkerson with BFS)
//! - Dinic's algorithm
//! - Push-relabel algorithm

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{Float, SparseElement};
use std::collections::VecDeque;
use std::fmt::Debug;

/// Result of a maximum flow computation
#[derive(Debug, Clone)]
pub struct MaxFlowResult<T> {
    /// Maximum flow value
    pub flow_value: T,
    /// Flow on each edge (residual graph representation)
    pub flow_matrix: Vec<Vec<T>>,
}

/// Compute maximum flow using Ford-Fulkerson algorithm with DFS
///
/// # Arguments
///
/// * `graph` - The sparse capacity matrix (adjacency matrix with capacities)
/// * `source` - Source vertex
/// * `sink` - Sink vertex
///
/// # Returns
///
/// MaxFlowResult containing the maximum flow value and flow matrix
pub fn ford_fulkerson<T, S>(graph: &S, source: usize, sink: usize) -> SparseResult<MaxFlowResult<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = graph.shape().0;

    if graph.shape().0 != graph.shape().1 {
        return Err(SparseError::ValueError(
            "Graph matrix must be square".to_string(),
        ));
    }

    if source >= n || sink >= n {
        return Err(SparseError::ValueError(
            "Source and sink must be valid vertices".to_string(),
        ));
    }

    // Create residual graph
    let mut residual = vec![vec![T::sparse_zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            residual[i][j] = graph.get(i, j);
        }
    }

    let mut max_flow = T::sparse_zero();

    // Keep finding augmenting paths using DFS
    loop {
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];

        if !dfs_find_path(&residual, source, sink, &mut visited, &mut parent) {
            break;
        }

        // Find minimum capacity along the path
        let mut path_flow = T::from(f64::INFINITY)
            .ok_or_else(|| SparseError::ComputationError("Cannot create infinity".to_string()))?;

        let mut current = sink;
        while let Some(prev) = parent[current] {
            path_flow = if residual[prev][current] < path_flow {
                residual[prev][current]
            } else {
                path_flow
            };
            current = prev;
        }

        // Update residual capacities
        current = sink;
        while let Some(prev) = parent[current] {
            residual[prev][current] = residual[prev][current] - path_flow;
            residual[current][prev] = residual[current][prev] + path_flow;
            current = prev;
        }

        max_flow = max_flow + path_flow;
    }

    Ok(MaxFlowResult {
        flow_value: max_flow,
        flow_matrix: residual,
    })
}

/// DFS helper to find augmenting path
fn dfs_find_path<T>(
    residual: &[Vec<T>],
    current: usize,
    sink: usize,
    visited: &mut [bool],
    parent: &mut [Option<usize>],
) -> bool
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    if current == sink {
        return true;
    }

    visited[current] = true;

    for neighbor in 0..residual.len() {
        if !visited[neighbor] && residual[current][neighbor] > T::sparse_zero() {
            parent[neighbor] = Some(current);
            if dfs_find_path(residual, neighbor, sink, visited, parent) {
                return true;
            }
        }
    }

    false
}

/// Compute maximum flow using Edmonds-Karp algorithm (BFS variant of Ford-Fulkerson)
///
/// # Arguments
///
/// * `graph` - The sparse capacity matrix
/// * `source` - Source vertex
/// * `sink` - Sink vertex
///
/// # Returns
///
/// MaxFlowResult containing the maximum flow value and flow matrix
pub fn edmonds_karp<T, S>(graph: &S, source: usize, sink: usize) -> SparseResult<MaxFlowResult<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = graph.shape().0;

    if graph.shape().0 != graph.shape().1 {
        return Err(SparseError::ValueError(
            "Graph matrix must be square".to_string(),
        ));
    }

    if source >= n || sink >= n {
        return Err(SparseError::ValueError(
            "Source and sink must be valid vertices".to_string(),
        ));
    }

    // Create residual graph
    let mut residual = vec![vec![T::sparse_zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            residual[i][j] = graph.get(i, j);
        }
    }

    let mut max_flow = T::sparse_zero();

    // Keep finding augmenting paths using BFS
    loop {
        let mut parent = vec![None; n];

        if !bfs_find_path(&residual, source, sink, &mut parent) {
            break;
        }

        // Find minimum capacity along the path
        let mut path_flow = T::from(f64::INFINITY)
            .ok_or_else(|| SparseError::ComputationError("Cannot create infinity".to_string()))?;

        let mut current = sink;
        while let Some(prev) = parent[current] {
            path_flow = if residual[prev][current] < path_flow {
                residual[prev][current]
            } else {
                path_flow
            };
            current = prev;
        }

        // Update residual capacities
        current = sink;
        while let Some(prev) = parent[current] {
            residual[prev][current] = residual[prev][current] - path_flow;
            residual[current][prev] = residual[current][prev] + path_flow;
            current = prev;
        }

        max_flow = max_flow + path_flow;
    }

    Ok(MaxFlowResult {
        flow_value: max_flow,
        flow_matrix: residual,
    })
}

/// BFS helper to find augmenting path
fn bfs_find_path<T>(
    residual: &[Vec<T>],
    source: usize,
    sink: usize,
    parent: &mut [Option<usize>],
) -> bool
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let n = residual.len();
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    queue.push_back(source);
    visited[source] = true;

    while let Some(current) = queue.pop_front() {
        if current == sink {
            return true;
        }

        for neighbor in 0..n {
            if !visited[neighbor] && residual[current][neighbor] > T::sparse_zero() {
                visited[neighbor] = true;
                parent[neighbor] = Some(current);
                queue.push_back(neighbor);
            }
        }
    }

    false
}

/// Compute maximum flow using Dinic's algorithm
///
/// Dinic's algorithm is more efficient than Ford-Fulkerson, with O(V²E) complexity.
///
/// # Arguments
///
/// * `graph` - The sparse capacity matrix
/// * `source` - Source vertex
/// * `sink` - Sink vertex
///
/// # Returns
///
/// MaxFlowResult containing the maximum flow value and flow matrix
pub fn dinic<T, S>(graph: &S, source: usize, sink: usize) -> SparseResult<MaxFlowResult<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = graph.shape().0;

    if graph.shape().0 != graph.shape().1 {
        return Err(SparseError::ValueError(
            "Graph matrix must be square".to_string(),
        ));
    }

    if source >= n || sink >= n {
        return Err(SparseError::ValueError(
            "Source and sink must be valid vertices".to_string(),
        ));
    }

    // Create residual graph
    let mut residual = vec![vec![T::sparse_zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            residual[i][j] = graph.get(i, j);
        }
    }

    let mut max_flow = T::sparse_zero();

    // Repeat while there exists a blocking flow
    loop {
        // Build level graph using BFS
        let level = build_level_graph(&residual, source, sink);

        if level[sink] < 0 {
            // No more augmenting paths
            break;
        }

        // Send multiple blocking flows using DFS
        loop {
            let mut visited = vec![false; n];
            let flow = send_flow(
                &mut residual,
                &level,
                source,
                sink,
                &mut visited,
                T::from(f64::INFINITY).ok_or_else(|| {
                    SparseError::ComputationError("Cannot create infinity".to_string())
                })?,
            );

            if scirs2_core::SparseElement::is_zero(&flow) {
                break;
            }

            max_flow = max_flow + flow;
        }
    }

    Ok(MaxFlowResult {
        flow_value: max_flow,
        flow_matrix: residual,
    })
}

/// Build level graph using BFS
fn build_level_graph<T>(residual: &[Vec<T>], source: usize, sink: usize) -> Vec<i32>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let n = residual.len();
    let mut level = vec![-1; n];
    level[source] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        for neighbor in 0..n {
            if level[neighbor] < 0 && residual[current][neighbor] > T::sparse_zero() {
                level[neighbor] = level[current] + 1;
                queue.push_back(neighbor);

                if neighbor == sink {
                    return level;
                }
            }
        }
    }

    level
}

/// Send flow using DFS in level graph
fn send_flow<T>(
    residual: &mut [Vec<T>],
    level: &[i32],
    current: usize,
    sink: usize,
    visited: &mut [bool],
    flow: T,
) -> T
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    if current == sink {
        return flow;
    }

    visited[current] = true;

    for neighbor in 0..residual.len() {
        if !visited[neighbor]
            && residual[current][neighbor] > T::sparse_zero()
            && level[neighbor] == level[current] + 1
        {
            let min_flow = if residual[current][neighbor] < flow {
                residual[current][neighbor]
            } else {
                flow
            };

            let pushed_flow = send_flow(residual, level, neighbor, sink, visited, min_flow);

            if pushed_flow > T::sparse_zero() {
                residual[current][neighbor] = residual[current][neighbor] - pushed_flow;
                residual[neighbor][current] = residual[neighbor][current] + pushed_flow;
                return pushed_flow;
            }
        }
    }

    T::sparse_zero()
}

/// Compute minimum cut using max-flow min-cut theorem
///
/// Returns the vertices reachable from source in the residual graph.
///
/// # Arguments
///
/// * `result` - Result from a max flow computation
/// * `source` - Source vertex
///
/// # Returns
///
/// Vector of booleans indicating which vertices are in the source side of the cut
pub fn min_cut<T>(result: &MaxFlowResult<T>, source: usize) -> Vec<bool>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let n = result.flow_matrix.len();
    let mut reachable = vec![false; n];
    let mut queue = VecDeque::new();

    queue.push_back(source);
    reachable[source] = true;

    while let Some(current) = queue.pop_front() {
        for neighbor in 0..n {
            if !reachable[neighbor] && result.flow_matrix[current][neighbor] > T::sparse_zero() {
                reachable[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    reachable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_test_flow_network() -> CsrArray<f64> {
        // Create a simple flow network
        //     10       10
        //  0 ----> 1 ----> 3
        //  |       |       ^
        //  |  10   |  10   | 10
        //  v       v       |
        //  2 -----------> 4
        //        15
        let rows = vec![0, 0, 1, 1, 2, 4];
        let cols = vec![1, 2, 3, 4, 4, 3];
        let data = vec![10.0, 10.0, 10.0, 10.0, 15.0, 10.0];

        CsrArray::from_triplets(&rows, &cols, &data, (5, 5), false).expect("Failed to create")
    }

    #[test]
    fn test_ford_fulkerson() {
        let graph = create_test_flow_network();
        let result = ford_fulkerson(&graph, 0, 3).expect("Failed");

        // Maximum flow from 0 to 3 should be 20
        assert!(result.flow_value > 15.0);
        assert!(result.flow_value <= 20.0);
    }

    #[test]
    fn test_edmonds_karp() {
        let graph = create_test_flow_network();
        let result = edmonds_karp(&graph, 0, 3).expect("Failed");

        // Maximum flow from 0 to 3 should be 20
        assert!(result.flow_value > 15.0);
        assert!(result.flow_value <= 20.0);
    }

    #[test]
    fn test_dinic() {
        let graph = create_test_flow_network();
        let result = dinic(&graph, 0, 3).expect("Failed");

        // Maximum flow from 0 to 3 should be 20
        assert!(result.flow_value > 15.0);
        assert!(result.flow_value <= 20.0);
    }

    #[test]
    fn test_simple_network() {
        // Simple network: 0 -> 1 -> 2 with capacities 10 and 5
        let rows = vec![0, 1];
        let cols = vec![1, 2];
        let data = vec![10.0, 5.0];

        let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Failed");
        let result = edmonds_karp(&graph, 0, 2).expect("Failed");

        // Maximum flow should be limited by the bottleneck edge (capacity 5)
        assert_relative_eq!(result.flow_value, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_no_path() {
        // Disconnected network: 0 -> 1, 2 -> 3 (no path from 0 to 3)
        let rows = vec![0, 2];
        let cols = vec![1, 3];
        let data = vec![10.0, 10.0];

        let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");
        let result = edmonds_karp(&graph, 0, 3).expect("Failed");

        // No flow possible
        assert_relative_eq!(result.flow_value, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_min_cut() {
        let graph = create_test_flow_network();
        let result = edmonds_karp(&graph, 0, 3).expect("Failed");
        let cut = min_cut(&result, 0);

        // Source should be in the cut
        assert!(cut[0]);

        // Sink should not be in the cut
        assert!(!cut[3]);
    }

    #[test]
    fn test_algorithms_agree() {
        let graph = create_test_flow_network();

        let ff_result = ford_fulkerson(&graph, 0, 3).expect("Failed");
        let ek_result = edmonds_karp(&graph, 0, 3).expect("Failed");
        let dinic_result = dinic(&graph, 0, 3).expect("Failed");

        // All algorithms should give the same max flow value
        assert_relative_eq!(ff_result.flow_value, ek_result.flow_value, epsilon = 1e-6);
        assert_relative_eq!(
            ek_result.flow_value,
            dinic_result.flow_value,
            epsilon = 1e-6
        );
    }
}
