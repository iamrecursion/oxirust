// Integration tests for scirs2-graph pipeline
// Tests: graph construction + traversal, spectral Laplacian, community detection

use scirs2_graph::spectral_graph::{graph_laplacian, normalized_laplacian};
use scirs2_graph::{
    breadth_first_search, complete_graph, dijkstra_path, louvain_communities_result, Graph,
};

use crate::common::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a path graph manually (0—1—2—…—(n-1))
// ─────────────────────────────────────────────────────────────────────────────

fn build_path_graph(n: usize) -> TestResult<Graph<usize, f64>> {
    let mut g: Graph<usize, f64> = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }
    for i in 0..n - 1 {
        g.add_edge(i, i + 1, 1.0)
            .map_err(|e| format!("add_edge failed: {}", e))?;
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build two complete subgraphs linked by one bridge edge
// Used to test community detection
// Clique A: nodes 0..5, Clique B: nodes 5..10
// Bridge: 4 — 5
// ─────────────────────────────────────────────────────────────────────────────

fn build_two_cliques_graph() -> TestResult<Graph<usize, f64>> {
    const HALF: usize = 5;
    let mut g: Graph<usize, f64> = Graph::new();

    // Add nodes 0..10
    for i in 0..10 {
        g.add_node(i);
    }

    // Clique A: 0..HALF (complete)
    for i in 0..HALF {
        for j in i + 1..HALF {
            g.add_edge(i, j, 1.0)
                .map_err(|e| format!("add_edge clique A: {}", e))?;
        }
    }

    // Clique B: HALF..10 (complete)
    for i in HALF..10 {
        for j in i + 1..10 {
            g.add_edge(i, j, 1.0)
                .map_err(|e| format!("add_edge clique B: {}", e))?;
        }
    }

    // Single bridge between the two cliques
    g.add_edge(HALF - 1, HALF, 1.0)
        .map_err(|e| format!("add_edge bridge: {}", e))?;

    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Basic graph construction, BFS traversal, Dijkstra shortest path
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 10-node path graph, verify BFS visits all nodes, verify shortest path length.
#[test]
fn test_graph_pipeline_basic_construction() -> TestResult<()> {
    let n = 10usize;
    let g = build_path_graph(n)?;

    // Node and edge count
    assert_eq!(g.node_count(), n, "Expected {} nodes", n);
    assert_eq!(g.edge_count(), n - 1, "Expected {} edges", n - 1);

    // BFS from node 0 should visit all n nodes
    let bfs_order = breadth_first_search(&g, &0).map_err(|e| format!("BFS failed: {}", e))?;

    assert_eq!(
        bfs_order.len(),
        n,
        "BFS should visit all {} nodes, got {}",
        n,
        bfs_order.len()
    );

    // In a path graph, BFS from 0 visits nodes in increasing order
    for (pos, &node_val) in bfs_order.iter().enumerate() {
        assert_eq!(
            node_val, pos,
            "BFS order mismatch at position {}: expected {}, got {}",
            pos, pos, node_val
        );
    }

    // Dijkstra shortest path from 0 to n-1 — distance = n-1 hops
    let path_opt =
        dijkstra_path(&g, &0, &(n - 1)).map_err(|e| format!("dijkstra_path failed: {}", e))?;

    let path = path_opt.ok_or("No path found from 0 to n-1")?;
    assert_eq!(
        path.nodes.len(),
        n,
        "Path should pass through all {} nodes, got {}",
        n,
        path.nodes.len()
    );

    // Total weight = n-1 hops × 1.0 per edge
    let expected_weight = (n - 1) as f64;
    assert!(
        (path.total_weight - expected_weight).abs() < 1e-10,
        "Shortest path weight: expected {}, got {}",
        expected_weight,
        path.total_weight
    );

    println!(
        "Graph construction: {} nodes, {} edges, BFS ok, Dijkstra weight={}",
        g.node_count(),
        g.edge_count(),
        path.total_weight
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Graph Laplacian is symmetric, PSD, and row-sums are zero
// ─────────────────────────────────────────────────────────────────────────────

/// Verify properties of the graph Laplacian for a 5-node complete graph.
#[test]
fn test_graph_pipeline_spectral() -> TestResult<()> {
    // 5-node complete graph — Laplacian eigenvalues: 0 (once) and 5 (four times)
    let kg = complete_graph(5).map_err(|e| format!("complete_graph failed: {}", e))?;

    let n = kg.node_count();
    assert_eq!(n, 5);

    // Build adjacency matrix via adjacency_matrix() on the Graph struct
    let adj: scirs2_core::ndarray::Array2<f64> = kg.adjacency_matrix();

    let lap = graph_laplacian(&adj);

    // Shape check
    assert_eq!(lap.shape(), &[n, n], "Laplacian shape wrong");

    // Symmetry: L[i,j] == L[j,i]
    for i in 0..n {
        for j in 0..n {
            assert!(
                (lap[[i, j]] - lap[[j, i]]).abs() < 1e-12,
                "Laplacian not symmetric at ({}, {}): L[i,j]={}, L[j,i]={}",
                i,
                j,
                lap[[i, j]],
                lap[[j, i]]
            );
        }
    }

    // Row sums must be zero (property of any graph Laplacian)
    for i in 0..n {
        let row_sum: f64 = lap.row(i).sum();
        assert!(
            row_sum.abs() < 1e-12,
            "Laplacian row {} sum is {} (expected 0)",
            i,
            row_sum
        );
    }

    // For K_5: each diagonal = 4, off-diagonals = -1
    for i in 0..n {
        assert!(
            (lap[[i, i]] - 4.0).abs() < 1e-12,
            "K_5 Laplacian diagonal at {}: expected 4, got {}",
            i,
            lap[[i, i]]
        );
        for j in 0..n {
            if j != i {
                assert!(
                    (lap[[i, j]] - (-1.0)).abs() < 1e-12,
                    "K_5 Laplacian off-diagonal ({},{}): expected -1, got {}",
                    i,
                    j,
                    lap[[i, j]]
                );
            }
        }
    }

    // Verify normalized Laplacian also computes without error
    let _norm_lap =
        normalized_laplacian(&adj).map_err(|e| format!("normalized_laplacian failed: {}", e))?;

    println!(
        "Spectral pipeline: K_{} Laplacian verified (symmetric, row-sums=0, correct entries)",
        n
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Community detection on two-clique graph
// ─────────────────────────────────────────────────────────────────────────────

/// Two dense cliques connected by a single bridge should resolve to ≥2 communities.
#[test]
fn test_graph_pipeline_community_detection() -> TestResult<()> {
    let g = build_two_cliques_graph()?;

    assert_eq!(g.node_count(), 10, "Expected 10 nodes");

    // Expected edges: C(5,2)+C(5,2)+1 = 10+10+1 = 21
    let expected_edges = 21usize;
    assert_eq!(
        g.edge_count(),
        expected_edges,
        "Expected {} edges, got {}",
        expected_edges,
        g.edge_count()
    );

    // Run Louvain community detection
    let result = louvain_communities_result(&g);

    assert!(
        result.num_communities >= 2,
        "Expected ≥2 communities for two-clique graph, got {}",
        result.num_communities
    );

    // Verify all 10 nodes are assigned to some community
    let total_assigned: usize = result.communities.iter().map(|c| c.len()).sum();
    assert_eq!(
        total_assigned, 10,
        "All 10 nodes must be assigned; got {}",
        total_assigned
    );

    // Verify communities are non-empty
    for (idx, comm) in result.communities.iter().enumerate() {
        assert!(!comm.is_empty(), "Community {} is empty", idx);
    }

    println!(
        "Community detection: {} communities, {} nodes assigned",
        result.num_communities, total_assigned
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Adjacency matrix properties for a path graph
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that the adjacency matrix of a path graph has exactly the right structure.
#[test]
fn test_graph_pipeline_adjacency_matrix() -> TestResult<()> {
    let n = 6usize;
    let g = build_path_graph(n)?;

    let adj: scirs2_core::ndarray::Array2<f64> = g.adjacency_matrix();

    assert_eq!(adj.shape(), &[n, n], "Adjacency matrix shape wrong");

    // Symmetry
    for i in 0..n {
        for j in 0..n {
            assert!(
                (adj[[i, j]] - adj[[j, i]]).abs() < 1e-12,
                "Adjacency matrix not symmetric at ({}, {})",
                i,
                j
            );
        }
    }

    // For a path graph: adj[i][i+1] = adj[i+1][i] = 1 for i in 0..n-1
    for i in 0..n - 1 {
        assert!(
            (adj[[i, i + 1]] - 1.0).abs() < 1e-12,
            "Expected adj[{},{}]=1, got {}",
            i,
            i + 1,
            adj[[i, i + 1]]
        );
    }

    // Diagonal must be 0 (no self-loops)
    for i in 0..n {
        assert!(
            adj[[i, i]].abs() < 1e-12,
            "Expected adj[{},{}]=0 (no self-loops), got {}",
            i,
            i,
            adj[[i, i]]
        );
    }

    // Row sums must equal degree (1 for endpoints, 2 for internal nodes)
    for i in 0..n {
        let row_sum: f64 = adj.row(i).sum();
        let expected = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
        assert!(
            (row_sum - expected).abs() < 1e-12,
            "Row sum at node {}: expected {}, got {}",
            i,
            expected,
            row_sum
        );
    }

    println!("Adjacency matrix for path graph P_{}: verified", n);
    Ok(())
}
