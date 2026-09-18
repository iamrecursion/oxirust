//! Deep / cyclic mob-graph traversal tests.
//!
//! Exercises `MobGraph` over the `u64`-keyed traversal model: a 100-level
//! linear reference chain (recursive DFS is safe at this depth), a 3-node
//! cycle that must terminate and be detected, and a single-node self-loop.

use oximedia_aaf::mob_traversal::{
    detect_cycles, find_source_clips, topological_sort, traverse_dfs, MobGraph, MobNode, MobType,
};

/// A 100-level Composition → Master* → Source chain resolves end-to-end:
/// DFS visits all 100 nodes ending at the leaf, the only source clip is node
/// 99, and the acyclic chain has a topological order.
#[test]
fn deep_100_level_chain_resolves_to_leaf() {
    let nodes: Vec<MobNode> = (0..100u64)
        .map(|i| {
            let mob_type = if i == 0 {
                MobType::Composition
            } else if i == 99 {
                MobType::Source
            } else {
                MobType::Master
            };
            MobNode::new(i, mob_type, format!("mob_{i}"))
        })
        .collect();
    let edges: Vec<(u64, u64)> = (1..100u64).map(|i| (i - 1, i)).collect();

    let g = MobGraph::build(&nodes, &edges);

    let order = traverse_dfs(&g, 0);
    assert_eq!(order.len(), 100, "DFS must visit every node in the chain");
    assert_eq!(
        order.last(),
        Some(&99),
        "linear chain DFS must end at the leaf node 99"
    );

    assert_eq!(
        find_source_clips(&g, 0),
        vec![99],
        "the single source leaf is node 99"
    );

    assert!(
        topological_sort(&g).is_some(),
        "an acyclic chain must have a topological order"
    );
}

/// A 3-node cycle (0→1→2→0) must NOT hang DFS (visited-guard terminates it),
/// must be reported by cycle detection, and has no topological order.
#[test]
fn cyclic_chain_terminates_and_is_detected() {
    let nodes = vec![
        MobNode::new(0, MobType::Composition, "C"),
        MobNode::new(1, MobType::Master, "M1"),
        MobNode::new(2, MobType::Master, "M2"),
    ];
    let edges = vec![(0u64, 1u64), (1, 2), (2, 0)];

    let g = MobGraph::build(&nodes, &edges);

    let order = traverse_dfs(&g, 0);
    assert_eq!(
        order.len(),
        3,
        "DFS over a cycle must terminate and visit each node once"
    );

    assert!(
        !detect_cycles(&g).is_empty(),
        "the 0→1→2→0 cycle must be detected"
    );
    assert!(
        topological_sort(&g).is_none(),
        "a cyclic graph has no topological order"
    );
}

/// A single node with a self-loop (0→0) terminates DFS at just that node and
/// is detected as a cycle.
#[test]
fn self_loop_terminates() {
    let nodes = vec![MobNode::new(0, MobType::Composition, "Self")];
    let edges = vec![(0u64, 0u64)];

    let g = MobGraph::build(&nodes, &edges);

    assert_eq!(
        traverse_dfs(&g, 0),
        vec![0],
        "self-loop DFS must terminate at the single node"
    );
    assert!(
        !detect_cycles(&g).is_empty(),
        "a self-loop is a cycle and must be detected"
    );
}
