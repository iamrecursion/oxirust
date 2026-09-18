//! Centrality measures for graphs
//!
//! This module provides various centrality measures including:
//! - Betweenness centrality
//! - Closeness centrality
//! - Eigenvector centrality
//! - PageRank

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{Float, SparseElement};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Compute betweenness centrality for all vertices
///
/// Betweenness centrality measures the extent to which a vertex lies on paths between other vertices.
/// Uses Brandes' algorithm for efficient computation.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `directed` - Whether the graph is directed
/// * `normalized` - Whether to normalize the centrality values
///
/// # Returns
///
/// Vector of betweenness centrality values for each vertex
pub fn betweenness_centrality<T, S>(
    graph: &S,
    directed: bool,
    normalized: bool,
) -> SparseResult<Vec<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = graph.shape().0;

    // Validate square matrix
    if graph.shape().0 != graph.shape().1 {
        return Err(SparseError::ValueError(
            "Graph matrix must be square".to_string(),
        ));
    }

    // Initialize centrality scores
    let mut centrality = vec![T::sparse_zero(); n];

    // Brandes' algorithm
    for s in 0..n {
        // Single source shortest paths
        let mut stack = Vec::new();
        let mut paths = vec![Vec::new(); n];
        let mut sigma = vec![T::sparse_zero(); n];
        sigma[s] = T::sparse_one();
        let mut distance = vec![
            T::from(-1.0).ok_or_else(|| {
                SparseError::ComputationError("Cannot convert -1.0".to_string())
            })?;
            n
        ];
        distance[s] = T::sparse_zero();

        let mut queue = VecDeque::new();
        queue.push_back(s);

        // BFS to find shortest paths
        while let Some(v) = queue.pop_front() {
            stack.push(v);

            // Get neighbors of v
            for col in 0..n {
                let weight = graph.get(v, col);
                if !scirs2_core::SparseElement::is_zero(&weight) {
                    // Found an edge
                    let w = col;

                    // First time seeing w?
                    if distance[w] < T::sparse_zero() {
                        distance[w] = distance[v] + T::sparse_one();
                        queue.push_back(w);
                    }

                    // Is this a shortest path?
                    if (distance[w] - distance[v] - T::sparse_one()).abs()
                        < T::from(1e-10).ok_or_else(|| {
                            SparseError::ComputationError("Cannot convert 1e-10".to_string())
                        })?
                    {
                        sigma[w] = sigma[w] + sigma[v];
                        paths[w].push(v);
                    }
                }
            }
        }

        // Accumulation phase
        let mut delta = vec![T::sparse_zero(); n];
        while let Some(w) = stack.pop() {
            for &v in &paths[w] {
                let coeff = (sigma[v] / sigma[w]) * (T::sparse_one() + delta[w]);
                delta[v] = delta[v] + coeff;
            }

            if w != s {
                centrality[w] = centrality[w] + delta[w];
            }
        }
    }

    // Normalization
    if normalized {
        let scale = if directed {
            T::from((n - 1) * (n - 2)).ok_or_else(|| {
                SparseError::ComputationError("Cannot compute normalization scale".to_string())
            })?
        } else {
            T::from((n - 1) * (n - 2) / 2).ok_or_else(|| {
                SparseError::ComputationError("Cannot compute normalization scale".to_string())
            })?
        };

        if scale > T::sparse_zero() {
            for cb in &mut centrality {
                *cb = *cb / scale;
            }
        }
    } else if !directed {
        // For undirected graphs, divide by 2 (each path counted twice)
        let two = T::from(2.0)
            .ok_or_else(|| SparseError::ComputationError("Cannot convert 2.0".to_string()))?;
        for cb in &mut centrality {
            *cb = *cb / two;
        }
    }

    Ok(centrality)
}

/// Compute closeness centrality for all vertices
///
/// Closeness centrality measures how close a vertex is to all other vertices.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `normalized` - Whether to normalize the centrality values
///
/// # Returns
///
/// Vector of closeness centrality values for each vertex
pub fn closeness_centrality<T, S>(graph: &S, normalized: bool) -> SparseResult<Vec<T>>
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

    let mut centrality = vec![T::sparse_zero(); n];

    // For each vertex, compute sum of distances to all other reachable vertices
    for s in 0..n {
        let mut distance = vec![
            T::from(-1.0).ok_or_else(|| {
                SparseError::ComputationError("Cannot convert -1.0".to_string())
            })?;
            n
        ];
        distance[s] = T::sparse_zero();

        let mut queue = VecDeque::new();
        queue.push_back(s);

        // BFS to find distances
        while let Some(v) = queue.pop_front() {
            for col in 0..n {
                let weight = graph.get(v, col);
                if !scirs2_core::SparseElement::is_zero(&weight) {
                    let w = col;
                    if distance[w] < T::sparse_zero() {
                        distance[w] = distance[v] + T::sparse_one();
                        queue.push_back(w);
                    }
                }
            }
        }

        // Sum distances to reachable vertices
        let mut sum_dist = T::sparse_zero();
        let mut reachable_count = 0;
        for (i, &d) in distance.iter().enumerate() {
            if i != s && d >= T::sparse_zero() {
                sum_dist = sum_dist + d;
                reachable_count += 1;
            }
        }

        // Compute centrality
        if reachable_count > 0 && sum_dist > T::sparse_zero() {
            centrality[s] = T::from(reachable_count).ok_or_else(|| {
                SparseError::ComputationError("Cannot convert reachable_count".to_string())
            })? / sum_dist;

            if normalized {
                let scale = T::from(n - 1).ok_or_else(|| {
                    SparseError::ComputationError("Cannot compute scale".to_string())
                })?;
                centrality[s] = centrality[s] * scale;
            }
        }
    }

    Ok(centrality)
}

/// Compute eigenvector centrality using power iteration
///
/// Eigenvector centrality measures the influence of a vertex in a network.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// Vector of eigenvector centrality values for each vertex
pub fn eigenvector_centrality<T, S>(graph: &S, max_iter: usize, tol: T) -> SparseResult<Vec<T>>
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

    // Initialize with uniform distribution
    let mut x = vec![
        T::sparse_one()
            / T::from(n).ok_or_else(|| {
                SparseError::ComputationError("Cannot convert n".to_string())
            })?;
        n
    ];

    // Power iteration
    for _ in 0..max_iter {
        let mut x_new = vec![T::sparse_zero(); n];

        // Matrix-vector multiplication: x_new = A * x
        for i in 0..n {
            let mut sum = T::sparse_zero();
            for j in 0..n {
                let weight = graph.get(i, j);
                if !scirs2_core::SparseElement::is_zero(&weight) {
                    sum = sum + weight * x[j];
                }
            }
            x_new[i] = sum;
        }

        // Normalize
        let norm: T = x_new.iter().map(|&v| v * v).sum();
        let norm = norm.sqrt();

        if norm < tol {
            return Err(SparseError::ComputationError(
                "Power iteration failed to converge".to_string(),
            ));
        }

        for v in &mut x_new {
            *v = *v / norm;
        }

        // Check convergence
        let diff: T = x.iter().zip(&x_new).map(|(&a, &b)| (a - b) * (a - b)).sum();
        let diff = diff.sqrt();

        x = x_new;

        if diff < tol {
            break;
        }
    }

    Ok(x)
}

/// Compute PageRank scores
///
/// PageRank is a link analysis algorithm that measures the importance of vertices.
///
/// # Arguments
///
/// * `graph` - The sparse adjacency matrix representing the graph
/// * `damping` - Damping factor (typically 0.85)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// Vector of PageRank scores for each vertex
pub fn pagerank<T, S>(graph: &S, damping: T, max_iter: usize, tol: T) -> SparseResult<Vec<T>>
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

    // Compute out-degrees
    let mut out_degree = vec![T::sparse_zero(); n];
    for i in 0..n {
        let mut deg = T::sparse_zero();
        for j in 0..n {
            let weight = graph.get(i, j);
            if !scirs2_core::SparseElement::is_zero(&weight) {
                deg = deg + T::sparse_one();
            }
        }
        out_degree[i] = deg;
    }

    // Initialize PageRank uniformly
    let init_value = T::sparse_one()
        / T::from(n)
            .ok_or_else(|| SparseError::ComputationError("Cannot convert n".to_string()))?;
    let mut pr = vec![init_value; n];

    let teleport = (T::sparse_one() - damping)
        / T::from(n)
            .ok_or_else(|| SparseError::ComputationError("Cannot convert n".to_string()))?;

    // Iterate until convergence
    for _ in 0..max_iter {
        let mut pr_new = vec![teleport; n];

        for i in 0..n {
            if out_degree[i] > T::sparse_zero() {
                let contrib = damping * pr[i] / out_degree[i];

                for j in 0..n {
                    let weight = graph.get(i, j);
                    if !scirs2_core::SparseElement::is_zero(&weight) {
                        pr_new[j] = pr_new[j] + contrib;
                    }
                }
            }
        }

        // Check convergence
        let diff: T = pr.iter().zip(&pr_new).map(|(&a, &b)| (a - b).abs()).sum();

        pr = pr_new;

        if diff < tol {
            break;
        }
    }

    Ok(pr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_test_graph() -> CsrArray<f64> {
        // Simple 4-vertex graph
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
        let data = vec![1.0; 8];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed to create")
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = create_test_graph();
        let centrality = betweenness_centrality(&graph, true, true).expect("Failed");

        assert_eq!(centrality.len(), 4);
        // All vertices should have some centrality
        for &c in &centrality {
            assert!(c >= 0.0);
        }
    }

    #[test]
    fn test_closeness_centrality() {
        let graph = create_test_graph();
        let centrality = closeness_centrality(&graph, true).expect("Failed");

        assert_eq!(centrality.len(), 4);
        for &c in &centrality {
            assert!(c >= 0.0);
        }
    }

    #[test]
    fn test_eigenvector_centrality() {
        let graph = create_test_graph();
        let centrality = eigenvector_centrality(&graph, 100, 1e-6).expect("Failed");

        assert_eq!(centrality.len(), 4);

        // Sum of squares should be approximately 1 (normalized)
        let sum_sq: f64 = centrality.iter().map(|&x| x * x).sum();
        assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_pagerank() {
        let graph = create_test_graph();
        let pr = pagerank(&graph, 0.85, 100, 1e-6).expect("Failed");

        assert_eq!(pr.len(), 4);

        // Sum should be approximately 1.0
        let sum: f64 = pr.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // All scores should be positive
        for &score in &pr {
            assert!(score > 0.0);
        }
    }

    #[test]
    fn test_simple_line_graph() {
        // Create a simple line graph: 0 - 1 - 2 - 3
        let rows = vec![0, 1, 1, 2, 2, 3];
        let cols = vec![1, 0, 2, 1, 3, 2];
        let data = vec![1.0; 6];

        let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

        let centrality = betweenness_centrality(&graph, false, true).expect("Failed");

        // Middle vertices should have higher betweenness
        assert!(centrality[1] > centrality[0]);
        assert!(centrality[2] > centrality[3]);
    }
}
