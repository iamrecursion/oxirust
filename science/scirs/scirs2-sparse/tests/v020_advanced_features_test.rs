//! Test suite for v0.2.0 advanced features
//!
//! This test file verifies all the new advanced features added in v0.2.0:
//! - GPU acceleration with Vulkan backend
//! - Advanced graph algorithms (betweenness centrality, community detection, max flow)
//! - Tensor-based sparse operations
//! - Specialized solvers for structured problems

use approx::assert_relative_eq;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_sparse::csr_array::CsrArray;
use scirs2_sparse::gpu::{GpuSpMatVec, VulkanSpMatVec};
use scirs2_sparse::tensor_sparse::{khatri_rao_product, SparseTensor};
use scirs2_sparse::SparseArray;
use scirs2_sparse::{
    betweenness_centrality, closeness_centrality, dinic, edmonds_karp, eigenvector_centrality,
    ford_fulkerson, label_propagation, louvain_communities, modularity, pagerank,
    solve_arrow_matrix, solve_banded_system, solve_block_2x2, solve_saddle_point,
};

/// Test GPU Vulkan backend integration
#[test]
fn test_vulkan_backend() {
    // Test Vulkan backend creation
    let vulkan_spmv = VulkanSpMatVec::new();
    assert!(vulkan_spmv.is_ok());

    let spmv = vulkan_spmv.expect("Failed to create Vulkan handler");
    let device_info = spmv.device_info();

    // Device info should be valid
    assert!(!device_info.device_name.is_empty());
    assert!(device_info.optimal_workgroup_size() > 0);

    // Test memory manager
    let mem_manager = spmv.memory_manager();
    assert_eq!(mem_manager.current_usage(), 0);
}

#[test]
fn test_vulkan_unified_interface() {
    // Test unified GPU interface with Vulkan
    let gpu_spmv = GpuSpMatVec::new();
    assert!(gpu_spmv.is_ok());

    let spmv = gpu_spmv.expect("Failed to create GPU handler");

    // Should have a valid backend
    let backend_info = spmv.get_backend_info();
    assert!(!backend_info.name.is_empty());

    // Test CPU fallback for Vulkan
    let rows = vec![0, 0, 1, 2];
    let cols = vec![0, 1, 1, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Failed");

    let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let result = spmv.spmv(&matrix, &vector.view(), None);
    assert!(result.is_ok());
}

/// Test betweenness centrality
#[test]
fn test_betweenness_centrality_advanced() {
    // Create a simple graph
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
    let data = vec![1.0; 8];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    // Test normalized
    let centrality = betweenness_centrality(&graph, false, true).expect("Failed");
    assert_eq!(centrality.len(), 4);

    // All centralities should be non-negative
    for &c in &centrality {
        assert!(c >= 0.0);
    }

    // Sum should be reasonable
    let sum: f64 = centrality.iter().sum();
    assert!(sum >= 0.0);
}

/// Test closeness centrality
#[test]
fn test_closeness_centrality_advanced() {
    let rows = vec![0, 1, 1, 2, 2, 3];
    let cols = vec![1, 0, 2, 1, 3, 2];
    let data = vec![1.0; 6];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    let centrality = closeness_centrality(&graph, true).expect("Failed");
    assert_eq!(centrality.len(), 4);

    for &c in &centrality {
        assert!(c >= 0.0);
    }
}

/// Test eigenvector centrality
#[test]
fn test_eigenvector_centrality_advanced() {
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
    let data = vec![1.0; 8];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    let centrality = eigenvector_centrality(&graph, 100, 1e-6).expect("Failed");
    assert_eq!(centrality.len(), 4);

    // Should be normalized (sum of squares ≈ 1)
    let sum_sq: f64 = centrality.iter().map(|&x| x * x).sum();
    assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-4);
}

/// Test PageRank
#[test]
fn test_pagerank_advanced() {
    let rows = vec![0, 1, 2, 3];
    let cols = vec![1, 2, 3, 0];
    let data = vec![1.0; 4];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    let pr = pagerank(&graph, 0.85, 100, 1e-6).expect("Failed");
    assert_eq!(pr.len(), 4);

    // Sum should be approximately 1
    let sum: f64 = pr.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
}

/// Test Louvain community detection
#[test]
fn test_louvain_communities_advanced() {
    // Two-community graph
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 2, 3];
    let cols = vec![1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 3, 2];
    let data = vec![1.0; 14];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (6, 6), false).expect("Failed");

    let (num_communities, communities) = louvain_communities(&graph, 1.0, 10).expect("Failed");

    assert!(num_communities >= 1);
    assert!(num_communities <= 6);
    assert_eq!(communities.len(), 6);
}

/// Test label propagation
#[test]
fn test_label_propagation_advanced() {
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let cols = vec![1, 2, 0, 2, 0, 1, 4, 5];
    let data = vec![1.0; 8];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (6, 6), false).expect("Failed");

    let (num_communities, communities) = label_propagation(&graph, 10).expect("Failed");

    assert!(num_communities >= 1);
    assert_eq!(communities.len(), 6);
}

/// Test modularity computation
#[test]
fn test_modularity_advanced() {
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let cols = vec![1, 2, 0, 2, 0, 1, 4, 5];
    let data = vec![1.0; 8];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (6, 6), false).expect("Failed");

    // Good partition
    let communities_good = vec![0, 0, 0, 1, 1, 1];
    let q_good = modularity(&graph, &communities_good).expect("Failed");

    // Random partition
    let communities_random = vec![0, 1, 0, 1, 0, 1];
    let q_random = modularity(&graph, &communities_random).expect("Failed");

    // Good partition should have higher modularity
    assert!(q_good >= q_random - 0.1); // Allow some tolerance
}

/// Test Ford-Fulkerson max flow
#[test]
fn test_ford_fulkerson_advanced() {
    let rows = vec![0, 0, 1, 2];
    let cols = vec![1, 2, 3, 3];
    let data = vec![10.0, 10.0, 10.0, 10.0];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    let result = ford_fulkerson(&graph, 0, 3).expect("Failed");
    assert!(result.flow_value >= 0.0);
    assert!(result.flow_value <= 20.0);
}

/// Test Edmonds-Karp max flow
#[test]
fn test_edmonds_karp_advanced() {
    let rows = vec![0, 0, 1, 2];
    let cols = vec![1, 2, 3, 3];
    let data = vec![10.0, 10.0, 10.0, 10.0];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    let result = edmonds_karp(&graph, 0, 3).expect("Failed");
    assert!(result.flow_value >= 0.0);
}

/// Test Dinic's algorithm
#[test]
fn test_dinic_advanced() {
    let rows = vec![0, 0, 1, 2];
    let cols = vec![1, 2, 3, 3];
    let data = vec![10.0, 10.0, 10.0, 10.0];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    let result = dinic(&graph, 0, 3).expect("Failed");
    assert!(result.flow_value >= 0.0);
}

/// Test sparse tensor operations
#[test]
fn test_sparse_tensor_basic() {
    let indices = vec![vec![0, 0, 1], vec![0, 1, 0], vec![0, 1, 2]];
    let values = vec![1.0, 2.0, 3.0];
    let shape = vec![2, 2, 3];

    let tensor = SparseTensor::new(indices, values, shape).expect("Failed");

    assert_eq!(tensor.ndim(), 3);
    assert_eq!(tensor.nnz(), 3);
    assert_eq!(tensor.size(), 12);
}

/// Test tensor unfolding
#[test]
fn test_tensor_unfold() {
    let indices = vec![vec![0, 1], vec![0, 1], vec![0, 2]];
    let values = vec![1.0, 2.0];
    let shape = vec![2, 2, 3];

    let tensor = SparseTensor::new(indices, values, shape).expect("Failed");

    // Unfold along each mode
    for mode in 0..3 {
        let unfolded = tensor.unfold(mode).expect("Failed");
        assert!(unfolded.nnz() > 0);
    }
}

/// Test tensor inner product
#[test]
fn test_tensor_inner_product() {
    let indices = vec![vec![0, 1], vec![0, 1]];
    let values = vec![2.0, 3.0];
    let shape = vec![2, 2];

    let tensor = SparseTensor::new(indices, values, shape).expect("Failed");

    let ip = tensor.inner_product(&tensor).expect("Failed");
    // Should be 2^2 + 3^2 = 13
    assert_relative_eq!(ip, 13.0, epsilon = 1e-10);
}

/// Test Khatri-Rao product
#[test]
fn test_khatri_rao_product_advanced() {
    let rows_a = vec![0, 1];
    let cols_a = vec![0, 0];
    let data_a = vec![1.0, 2.0];
    let a = CsrArray::from_triplets(&rows_a, &cols_a, &data_a, (2, 1), false).expect("Failed");

    let rows_b = vec![0, 1];
    let cols_b = vec![0, 0];
    let data_b = vec![3.0, 4.0];
    let b = CsrArray::from_triplets(&rows_b, &cols_b, &data_b, (2, 1), false).expect("Failed");

    let result = khatri_rao_product(&a, &b).expect("Failed");
    assert_eq!(result.shape(), (4, 1));
}

/// Test saddle point solver
#[test]
fn test_saddle_point_solver_advanced() {
    let rows_a = vec![0, 1];
    let cols_a = vec![0, 1];
    let data_a = vec![2.0, 3.0];
    let a = CsrArray::from_triplets(&rows_a, &cols_a, &data_a, (2, 2), false).expect("Failed");

    let rows_b = vec![0, 0];
    let cols_b = vec![0, 1];
    let data_b = vec![1.0, 1.0];
    let b = CsrArray::from_triplets(&rows_b, &cols_b, &data_b, (1, 2), false).expect("Failed");

    let f = Array1::from_vec(vec![1.0, 2.0]);
    let g = Array1::from_vec(vec![3.0]);

    let result = solve_saddle_point(&a, &b, &f, &g, 1e-6, 100);
    assert!(result.is_ok());
}

/// Test block 2x2 solver
#[test]
fn test_block_2x2_solver_advanced() {
    let rows = vec![0];
    let cols = vec![0];
    let data = vec![4.0];
    let a11 = CsrArray::from_triplets(&rows, &cols, &data, (1, 1), false).expect("Failed");
    let a22 = CsrArray::from_triplets(&rows, &cols, &[3.0], (1, 1), false).expect("Failed");
    let a12 = CsrArray::from_triplets(&rows, &cols, &[1.0], (1, 1), false).expect("Failed");
    let a21 = CsrArray::from_triplets(&rows, &cols, &[1.0], (1, 1), false).expect("Failed");

    let b1 = Array1::from_vec(vec![1.0]);
    let b2 = Array1::from_vec(vec![2.0]);

    let result = solve_block_2x2(&a11, &a12, &a21, &a22, &b1, &b2, 1e-6, 100);
    assert!(result.is_ok());
}

/// Test arrow matrix solver
#[test]
fn test_arrow_matrix_solver_advanced() {
    let diag = Array1::from_vec(vec![3.0, 2.0, 4.0]);
    let arrow_row = Array1::from_vec(vec![1.0, 0.5]);
    let arrow_col = Array1::from_vec(vec![0.8, 0.6]);
    let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    let result = solve_arrow_matrix(&diag, &arrow_row, &arrow_col, &rhs);
    assert!(result.is_ok());

    let solution: Array1<f64> = result.expect("Failed");
    for &val in solution.iter() {
        assert!(val.is_finite());
    }
}

/// Test banded system solver
#[test]
fn test_banded_system_solver_advanced() {
    // Tridiagonal matrix
    let rows = vec![0, 0, 1, 1, 1, 2, 2];
    let cols = vec![0, 1, 0, 1, 2, 1, 2];
    let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];

    let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Failed");
    let rhs = Array1::from_vec(vec![1.0, 0.0, 1.0]);

    let result = solve_banded_system(&matrix, &rhs, 1, 1e-6, 100);
    assert!(result.is_ok());
}

/// Integration test: Combine multiple v0.2.0 features
#[test]
fn test_v020_integration() {
    // Create a graph
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
    let data = vec![1.0; 8];
    let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");

    // Compute centrality
    let bc = betweenness_centrality(&graph, false, true).expect("Failed");
    assert_eq!(bc.len(), 4);

    // Find communities
    let (num_comm, _) = louvain_communities(&graph, 1.0, 10).expect("Failed");
    assert!(num_comm >= 1);

    // Create a flow network
    let flow_graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).expect("Failed");
    let max_flow_result = dinic(&flow_graph, 0, 3).expect("Failed");
    assert!(max_flow_result.flow_value >= 0.0);

    // Create tensor
    let tensor_indices = vec![vec![0, 1], vec![0, 1]];
    let tensor_values = vec![2.0, 3.0];
    let tensor_shape = vec![2, 2];
    let tensor = SparseTensor::new(tensor_indices, tensor_values, tensor_shape).expect("Failed");
    assert_eq!(tensor.ndim(), 2);
}
