//! Community detection algorithms for graphs
//!
//! This module provides algorithms for detecting communities/clusters in graphs:
//! - Louvain method for modularity optimization
//! - Label propagation algorithm
//! - Girvan-Newman algorithm (edge betweenness)

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{Float, SparseElement};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// Detect communities using the Louvain method
///
/// The Louvain method is a greedy optimization method that maximizes modularity.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `resolution` - Resolution parameter (default 1.0)
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
///
/// Tuple of (number of communities, community labels for each vertex)
pub fn louvain_communities<T, S>(
    graph: &S,
    resolution: T,
    max_iter: usize,
) -> SparseResult<(usize, Vec<usize>)>
where
    T: Float + SparseElement + Debug + Copy + std::iter::Sum + 'static,
    S: SparseArray<T>,
{
    let n = graph.shape().0;

    if graph.shape().0 != graph.shape().1 {
        return Err(SparseError::ValueError(
            "Graph matrix must be square".to_string(),
        ));
    }

    // Initialize each node in its own community
    let mut communities = (0..n).collect::<Vec<_>>();

    // Compute degree for each node: k_i = sum_j A_{i,j}
    let mut degrees = vec![T::sparse_zero(); n];
    // Total edge weight m = (1/2) * sum_{i,j} A_{i,j} for undirected graphs
    let mut sum_all_weights = T::sparse_zero();

    for i in 0..n {
        for j in 0..n {
            let weight = graph.get(i, j);
            if !scirs2_core::SparseElement::is_zero(&weight) {
                degrees[i] = degrees[i] + weight;
                sum_all_weights = sum_all_weights + weight;
            }
        }
    }

    // m = total edge weight (sum of all entries / 2 for undirected symmetric matrices)
    let two = T::from(2.0)
        .ok_or_else(|| SparseError::ComputationError("Cannot convert 2.0".to_string()))?;
    let m = sum_all_weights / two;

    if scirs2_core::SparseElement::is_zero(&m) {
        return Ok((n, communities));
    }

    let mut improvement = true;
    let mut iteration = 0;

    while improvement && iteration < max_iter {
        improvement = false;
        iteration += 1;

        // For each node, try to move it to the community that maximizes modularity gain
        for node in 0..n {
            let current_community = communities[node];

            // Collect neighboring communities (use sorted Vec for deterministic iteration)
            let mut neighbor_community_set = HashSet::new();

            for neighbor in 0..n {
                let weight = graph.get(node, neighbor);
                if !scirs2_core::SparseElement::is_zero(&weight) && neighbor != node {
                    neighbor_community_set.insert(communities[neighbor]);
                }
            }
            // Always consider staying in (or returning to) current community
            neighbor_community_set.insert(current_community);

            // Sort for deterministic behavior
            let mut neighbor_communities: Vec<usize> = neighbor_community_set.into_iter().collect();
            neighbor_communities.sort();

            // Temporarily remove node from its current community for gain calculation
            // Compute weight from node to its own (current) community (excluding self)
            let mut weight_to_current = T::sparse_zero();
            let mut sigma_current = T::sparse_zero(); // sum of degrees in current community (excluding node)
            for i in 0..n {
                if i != node && communities[i] == current_community {
                    let w = graph.get(node, i);
                    weight_to_current = weight_to_current + w;
                    sigma_current = sigma_current + degrees[i];
                }
            }

            let k_i = degrees[node];

            // The gain of removing node from current community (negative of adding cost)
            // delta_remove = - (weight_to_current - resolution * k_i * sigma_current / (2m))
            let remove_cost = weight_to_current - resolution * k_i * sigma_current / (two * m);

            let mut best_community = current_community;
            let mut best_delta = T::sparse_zero();

            for &community in &neighbor_communities {
                if community == current_community {
                    // No net gain from staying
                    continue;
                }

                // Compute weight from node to target community
                let mut weight_to_target = T::sparse_zero();
                let mut sigma_target = T::sparse_zero();
                for i in 0..n {
                    if communities[i] == community {
                        let w = graph.get(node, i);
                        weight_to_target = weight_to_target + w;
                        sigma_target = sigma_target + degrees[i];
                    }
                }

                // Gain of adding node to target community
                let add_gain = weight_to_target - resolution * k_i * sigma_target / (two * m);

                // Net gain = gain of adding to new - cost of being in old
                let delta = add_gain - remove_cost;

                if delta > best_delta {
                    best_delta = delta;
                    best_community = community;
                }
            }

            if best_community != current_community {
                communities[node] = best_community;
                improvement = true;
            }
        }
    }

    // Renumber communities to be consecutive
    let community_map = renumber_communities(&communities);
    let final_communities: Vec<usize> = communities.iter().map(|&c| community_map[&c]).collect();

    let num_communities = community_map.len();

    Ok((num_communities, final_communities))
}

/// Renumber communities to be consecutive starting from 0
fn renumber_communities(communities: &[usize]) -> HashMap<usize, usize> {
    let unique_communities: HashSet<usize> = communities.iter().copied().collect();
    let mut community_map = HashMap::new();

    for (new_id, &old_id) in unique_communities.iter().enumerate() {
        community_map.insert(old_id, new_id);
    }

    community_map
}

/// Detect communities using label propagation
///
/// Label propagation is a fast algorithm that assigns labels to nodes based on
/// the most common label among neighbors.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
///
/// Tuple of (number of communities, community labels for each vertex)
pub fn label_propagation<T, S>(graph: &S, max_iter: usize) -> SparseResult<(usize, Vec<usize>)>
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

    // Initialize each node with unique label
    let mut labels = (0..n).collect::<Vec<_>>();

    let mut changed = true;
    let mut iteration = 0;

    while changed && iteration < max_iter {
        changed = false;
        iteration += 1;

        // Process nodes in order (could be randomized for better results)
        for node in 0..n {
            // Count labels of neighbors
            let mut label_counts: HashMap<usize, T> = HashMap::new();

            for neighbor in 0..n {
                let weight = graph.get(node, neighbor);
                if !scirs2_core::SparseElement::is_zero(&weight) && neighbor != node {
                    let neighbor_label = labels[neighbor];
                    let count = label_counts
                        .entry(neighbor_label)
                        .or_insert(T::sparse_zero());
                    *count = *count + weight;
                }
            }

            if label_counts.is_empty() {
                continue;
            }

            // Find most common label
            let mut best_label = labels[node];
            let mut best_count = T::sparse_zero();

            for (&label, &count) in &label_counts {
                if count > best_count {
                    best_count = count;
                    best_label = label;
                }
            }

            if best_label != labels[node] {
                labels[node] = best_label;
                changed = true;
            }
        }
    }

    // Renumber communities
    let community_map = renumber_communities(&labels);
    let final_communities: Vec<usize> = labels.iter().map(|&c| community_map[&c]).collect();
    let num_communities = community_map.len();

    Ok((num_communities, final_communities))
}

/// Calculate modularity of a community partition
///
/// Modularity measures the quality of a community structure.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `communities` - Community assignment for each vertex
///
/// # Returns
///
/// Modularity value (typically between -0.5 and 1.0)
pub fn modularity<T, S>(graph: &S, communities: &[usize]) -> SparseResult<T>
where
    T: Float + SparseElement + Debug + Copy + std::iter::Sum + 'static,
    S: SparseArray<T>,
{
    let n = graph.shape().0;

    if graph.shape().0 != graph.shape().1 {
        return Err(SparseError::ValueError(
            "Graph matrix must be square".to_string(),
        ));
    }

    if communities.len() != n {
        return Err(SparseError::ValueError(
            "Communities vector must match graph size".to_string(),
        ));
    }

    let two = T::from(2.0)
        .ok_or_else(|| SparseError::ComputationError("Cannot convert 2.0".to_string()))?;

    // Calculate sum of all edge weights and node degrees
    let mut sum_all_weights = T::sparse_zero();
    let mut degrees = vec![T::sparse_zero(); n];
    for i in 0..n {
        for j in 0..n {
            let weight = graph.get(i, j);
            if !scirs2_core::SparseElement::is_zero(&weight) {
                degrees[i] = degrees[i] + weight;
                sum_all_weights = sum_all_weights + weight;
            }
        }
    }

    // m = total edge weight for undirected graph = sum of all entries / 2
    let m = sum_all_weights / two;

    if scirs2_core::SparseElement::is_zero(&m) {
        return Ok(T::sparse_zero());
    }

    let two_m = two * m;

    // Calculate modularity: Q = (1/2m) * sum_{ij} [A_{ij} - k_i*k_j/(2m)] * delta(c_i, c_j)
    let mut q = T::sparse_zero();
    for i in 0..n {
        for j in 0..n {
            if communities[i] == communities[j] {
                let aij = graph.get(i, j);
                let kikj = degrees[i] * degrees[j];
                q = q + (aij - kikj / two_m);
            }
        }
    }

    q = q / two_m;

    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;

    fn create_two_community_graph() -> CsrArray<f64> {
        // Create a graph with two clear communities:
        // Community 1: nodes 0, 1, 2
        // Community 2: nodes 3, 4, 5
        let rows = vec![
            0, 0, 1, 1, 2, 2, // Community 1 edges
            3, 3, 4, 4, 5, 5, // Community 2 edges
            2, 3, // Inter-community edge
        ];
        let cols = vec![
            1, 2, 0, 2, 0, 1, // Community 1 edges
            4, 5, 3, 5, 3, 4, // Community 2 edges
            3, 2, // Inter-community edge
        ];
        let data = vec![1.0; 14];

        CsrArray::from_triplets(&rows, &cols, &data, (6, 6), false).expect("Failed to create")
    }

    #[test]
    fn test_louvain_communities() {
        let graph = create_two_community_graph();
        let (num_communities, communities) = louvain_communities(&graph, 1.0, 10).expect("Failed");

        // Should detect 2 or 3 communities
        assert!(num_communities >= 2);
        assert!(num_communities <= 3);
        assert_eq!(communities.len(), 6);
    }

    #[test]
    fn test_label_propagation() {
        let graph = create_two_community_graph();
        let (num_communities, communities) = label_propagation(&graph, 10).expect("Failed");

        // Should detect communities
        assert!(num_communities >= 1);
        assert_eq!(communities.len(), 6);
    }

    #[test]
    fn test_modularity() {
        let graph = create_two_community_graph();

        // Perfect community assignment
        let communities = vec![0, 0, 0, 1, 1, 1];
        let q = modularity(&graph, &communities).expect("Failed");

        // Modularity should be positive for good partition
        assert!(q > 0.0);

        // Random assignment should have lower modularity
        let random_communities = vec![0, 1, 0, 1, 0, 1];
        let q_random = modularity(&graph, &random_communities).expect("Failed");

        assert!(q > q_random);
    }

    #[test]
    fn test_single_node_communities() {
        let graph = create_two_community_graph();

        // Each node in its own community
        let communities = vec![0, 1, 2, 3, 4, 5];
        let q = modularity(&graph, &communities).expect("Failed");

        // Should have low modularity
        assert!(q < 0.3);
    }

    #[test]
    fn test_all_same_community() {
        let graph = create_two_community_graph();

        // All nodes in same community
        let communities = vec![0, 0, 0, 0, 0, 0];
        let q = modularity(&graph, &communities).expect("Failed");

        // Modularity should be 0 for complete graph in one community
        assert!(q.abs() < 0.1);
    }
}
