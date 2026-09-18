//! Correctness suite for cycle detection ([`cycle_detect`]) and graph merging
//! ([`graph_merge`]).
//!
//! These tests pin behaviour that earlier had a latent node-collision bug in
//! the `Remap` merge strategy (a non-conflicting `other` id could be silently
//! overwritten by a freshly-allocated remap id) and document the deliberate
//! parallel-edge-collapse semantics of [`CycleGraph`]'s `HashSet` adjacency.

use std::collections::HashSet;

use oximedia_graph::cycle_detect::{Cycle, CycleGraph, CycleNodeId};
use oximedia_graph::graph_merge::{
    ConflictStrategy, MergeConfig, MergeEdge, MergeError, MergeGraph, MergeNode, MergeNodeId,
};

fn cn(id: usize) -> CycleNodeId {
    CycleNodeId(id)
}

fn mn(id: usize) -> MergeNodeId {
    MergeNodeId(id)
}

fn make_node(id: usize, label: &str) -> MergeNode {
    MergeNode::new(mn(id), label)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cycle detection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cycle_detect_self_loop_in_detect_all() {
    // A single self-loop 0 -> 0 is the shortest possible cycle.
    let mut graph = CycleGraph::new();
    graph.add_edge(cn(0), cn(0));

    assert!(graph.has_cycle(), "self-loop must be detected by has_cycle");

    let result = graph.detect_all_cycles();
    assert!(
        !result.is_acyclic,
        "detect_all_cycles must report a non-acyclic graph for a self-loop"
    );
    assert!(
        !result.cycles.is_empty(),
        "at least one cycle must be reported"
    );

    // Exactly one node participates: node 0. Compare canonically so the result
    // is order-independent.
    let canon: Vec<Vec<CycleNodeId>> = result.cycles.iter().map(|c| c.canonical().nodes).collect();
    assert!(
        canon.contains(&vec![cn(0)]),
        "expected a cycle whose canonical form is [0], got {canon:?}"
    );
}

#[test]
fn cycle_detect_two_node_and_multiedge() {
    // 0 -> 1 -> 0 is a 2-node cycle.
    let mut graph = CycleGraph::new();
    graph.add_edge(cn(0), cn(1));
    graph.add_edge(cn(1), cn(0));
    assert!(graph.has_cycle());

    let result = graph.detect_all_cycles();
    assert!(!result.is_acyclic);
    let found = result
        .cycles
        .iter()
        .map(Cycle::canonical)
        .any(|c| c.nodes == vec![cn(0), cn(1)]);
    assert!(found, "expected the 2-node cycle [0, 1] (canonical)");

    // Adjacency is HashMap<_, HashSet<_>>, so parallel edges COLLAPSE: adding
    // 0 -> 1 twice leaves a single edge. This is documented behaviour.
    let mut multi = CycleGraph::new();
    multi.add_edge(cn(0), cn(1));
    multi.add_edge(cn(0), cn(1));
    assert_eq!(
        multi.edge_count(),
        1,
        "parallel edges collapse in the HashSet adjacency (edge_count stays 1)"
    );
    assert!(!multi.has_cycle(), "a single 0 -> 1 edge is acyclic");
}

#[test]
fn cycle_detect_mixed_cycle_and_tail() {
    // 0 -> 1 -> 2 -> 0 is a 3-cycle; 3 -> 0 is a tail feeding into it but is not
    // itself part of any cycle (nothing returns to 3).
    let mut graph = CycleGraph::new();
    graph.add_edge(cn(0), cn(1));
    graph.add_edge(cn(1), cn(2));
    graph.add_edge(cn(2), cn(0));
    graph.add_edge(cn(3), cn(0));

    let cyclic = graph.cyclic_nodes();
    assert!(cyclic.contains(&cn(0)), "0 is on the cycle");
    assert!(cyclic.contains(&cn(1)), "1 is on the cycle");
    assert!(cyclic.contains(&cn(2)), "2 is on the cycle");
    assert!(
        !cyclic.contains(&cn(3)),
        "3 is only a tail, not on the cycle"
    );
    assert!(!graph.is_in_cycle(cn(3)), "is_in_cycle(3) must be false");
    assert!(graph.is_in_cycle(cn(0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph merge — Remap correctness
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn merge_overlapping_ids_remap_preserves_all_nodes() {
    // g1 holds nodes 0..50 chained; g2 holds nodes 0..50 chained — a full
    // overlap. A Remap merge must keep all 100 distinct nodes and remap every
    // one of g2's 50 ids to a fresh, unique, in-range id.
    let mut g1 = MergeGraph::new();
    for i in 0..50 {
        g1.add_node(make_node(i, &format!("g1_{i}")));
    }
    for i in 0..49 {
        g1.add_edge(MergeEdge::new(mn(i), mn(i + 1)));
    }

    let mut g2 = MergeGraph::new();
    for i in 0..50 {
        g2.add_node(make_node(i, &format!("g2_{i}")));
    }
    for i in 0..49 {
        g2.add_edge(MergeEdge::new(mn(i), mn(i + 1)));
    }

    let config = MergeConfig {
        conflict_strategy: ConflictStrategy::Remap,
        ..Default::default()
    };
    let mapping = g1.merge(&g2, &config).expect("remap merge should succeed");

    assert_eq!(
        g1.node_count(),
        100,
        "full overlap must yield 50 + 50 = 100 distinct nodes (no loss)"
    );
    assert_eq!(
        mapping.remap_count(),
        50,
        "every one of g2's 50 ids conflicts and must be remapped"
    );

    // Every remapped target must be unique, in range, and actually present.
    let mut targets: HashSet<MergeNodeId> = HashSet::new();
    for (&old, &new) in mapping.mappings() {
        if old != new {
            assert!(
                targets.insert(new),
                "remapped target {new} collided with another remap"
            );
            assert!(
                g1.has_node(new),
                "remapped target {new} must exist in the merged graph"
            );
        }
    }
    assert_eq!(targets.len(), 50);

    // All of g2's edges must be present under the remapping.
    let pairs = g1.edge_pairs();
    for edge in g2.edges() {
        let from = mapping.resolve(edge.from);
        let to = mapping.resolve(edge.to);
        assert!(
            pairs.contains(&(from, to)),
            "remapped g2 edge {from} -> {to} is missing after merge"
        );
    }
}

#[test]
fn merge_mixed_overlap_does_not_collide() {
    // THE BUG-TRIP CASE.
    //
    // g1 = {0}; g2 = {0, 1}. Under Remap:
    //   * g2 node 0 conflicts with g1 node 0 -> must be remapped.
    //   * g2 node 1 does NOT conflict -> kept at id 1.
    //
    // The original code seeded the remap counter from `self.next_id()` (== 1
    // here) and did not reserve the surviving non-conflicting id 1. So the
    // remap of g2's node 0 was handed id 1 and `nodes.insert(1, ..)` could
    // silently overwrite the non-conflicting g2 node 1 (HashMap-iteration-order
    // dependent). Result: a lost node and a mapping pointing at the wrong node.
    //
    // After the fix, both the kept g2-node-1 and the remapped g2-node-0 survive
    // with distinct ids, giving 1 (g1) + 2 (g2) = 3 nodes.
    let mut g1 = MergeGraph::new();
    g1.add_node(make_node(0, "g1_zero"));

    let mut g2 = MergeGraph::new();
    g2.add_node(make_node(0, "g2_zero"));
    g2.add_node(make_node(1, "g2_one"));

    let config = MergeConfig {
        conflict_strategy: ConflictStrategy::Remap,
        ..Default::default()
    };
    let mapping = g1.merge(&g2, &config).expect("remap merge should succeed");

    assert_eq!(
        g1.node_count(),
        3,
        "g1{{0}} + g2{{0,1}} under Remap must yield 3 distinct nodes; \
         a smaller count means a node was silently overwritten"
    );

    // g2 node 1 was non-conflicting and must still be reachable at id 1, holding
    // its own label (not clobbered by the remapped node 0).
    let kept_one = g1.get_node(mn(1)).expect("g2 node 1 must survive at id 1");
    assert_eq!(
        kept_one.label, "g2_one",
        "the surviving id 1 must be g2's node 1, not the remapped node 0"
    );

    // g2 node 0 was remapped; its new id must differ from both 0 and 1 and the
    // node must be present and carry g2's label.
    let remapped_zero = mapping.resolve(mn(0));
    assert_ne!(remapped_zero, mn(0), "g2 node 0 must have been remapped");
    assert_ne!(
        remapped_zero,
        mn(1),
        "the remapped g2 node 0 must NOT land on the surviving id 1"
    );
    let remapped_node = g1
        .get_node(remapped_zero)
        .expect("remapped g2 node 0 must exist");
    assert_eq!(remapped_node.label, "g2_zero");

    // g1's own node 0 is untouched.
    assert_eq!(
        g1.get_node(mn(0)).expect("g1 node 0 survives").label,
        "g1_zero"
    );
}

#[test]
fn merge_fail_keepfirst_keepsecond_semantics() {
    // KeepFirst: on a conflicting id, the first graph's node is retained.
    {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "first"));
        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "second"));
        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::KeepFirst,
            ..Default::default()
        };
        g1.merge(&g2, &config)
            .expect("KeepFirst merge should succeed");
        assert_eq!(g1.node_count(), 1);
        assert_eq!(
            g1.get_node(mn(0)).expect("node 0 present").label,
            "first",
            "KeepFirst must retain the first graph's label"
        );
    }

    // KeepSecond: on a conflicting id, the second graph's node overwrites.
    {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "first"));
        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "second"));
        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::KeepSecond,
            ..Default::default()
        };
        g1.merge(&g2, &config)
            .expect("KeepSecond merge should succeed");
        assert_eq!(g1.node_count(), 1);
        assert_eq!(
            g1.get_node(mn(0)).expect("node 0 present").label,
            "second",
            "KeepSecond must overwrite with the second graph's label"
        );
    }

    // Fail: any conflicting id aborts the merge with MergeError::Conflict.
    {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "first"));
        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "second"));
        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::Fail,
            ..Default::default()
        };
        let result = g1.merge(&g2, &config);
        assert!(
            matches!(result, Err(MergeError::Conflict(id)) if id == mn(0)),
            "Fail strategy must return Conflict(0)"
        );
    }
}
