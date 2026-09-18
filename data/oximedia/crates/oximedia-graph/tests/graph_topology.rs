//! Integration tests for complex multi-branch graph topologies.
//!
//! These tests build real [`FilterGraph`]s through the public [`GraphBuilder`]
//! API and assert *structural* correctness (node/source/sink counts and a valid
//! topological execution order). They deliberately do NOT assert end-to-end
//! frame delivery — that is covered elsewhere; here we pin the builder's
//! topology handling for diamond and fan-out/fan-in shapes plus cycle rejection.

use std::collections::HashMap;

use oximedia_graph::filters::video::{
    MergeConfig, MergeFilter, NullSink, PassthroughFilter, SplitFilter,
};
use oximedia_graph::graph::GraphBuilder;
use oximedia_graph::node::NodeId;
use oximedia_graph::port::PortId;
use oximedia_graph::{FilterGraph, GraphError};

/// Assert that `order` is a valid topological order for the directed edges
/// `edges` (each `(from, to)`): every edge must go strictly forward in `order`,
/// and every endpoint must appear exactly once.
fn assert_valid_topo_order(order: &[NodeId], edges: &[(NodeId, NodeId)]) {
    let mut position: HashMap<NodeId, usize> = HashMap::new();
    for (idx, &id) in order.iter().enumerate() {
        assert!(
            position.insert(id, idx).is_none(),
            "node {id:?} appears more than once in execution order"
        );
    }
    for &(from, to) in edges {
        let pf = position
            .get(&from)
            .unwrap_or_else(|| panic!("edge source {from:?} missing from execution order"));
        let pt = position
            .get(&to)
            .unwrap_or_else(|| panic!("edge dest {to:?} missing from execution order"));
        assert!(
            pf < pt,
            "edge {from:?} -> {to:?} is not forward in execution order ({pf} !< {pt})"
        );
    }
}

#[test]
fn diamond_topology_builds_and_orders() {
    // source -> split(2) -> { passA, passB } -> merge(2) -> sink
    let source = PassthroughFilter::new_source(NodeId(0), "source");
    let split = SplitFilter::new(NodeId(0), "split", 2);
    let pass_a = PassthroughFilter::new(NodeId(0), "passA");
    let pass_b = PassthroughFilter::new(NodeId(0), "passB");
    let merge = MergeFilter::new(NodeId(0), "merge", MergeConfig::tiled(2, 640, 480));
    let sink = NullSink::new(NodeId(0), "sink");

    let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
    let (builder, split_id) = builder.add_node(Box::new(split));
    let (builder, a_id) = builder.add_node(Box::new(pass_a));
    let (builder, b_id) = builder.add_node(Box::new(pass_b));
    let (builder, merge_id) = builder.add_node(Box::new(merge));
    let (builder, sink_id) = builder.add_node(Box::new(sink));

    let graph: FilterGraph = builder
        .connect(source_id, PortId(0), split_id, PortId(0))
        .expect("connect source->split")
        .connect(split_id, PortId(0), a_id, PortId(0))
        .expect("connect split.0->passA")
        .connect(split_id, PortId(1), b_id, PortId(0))
        .expect("connect split.1->passB")
        .connect(a_id, PortId(0), merge_id, PortId(0))
        .expect("connect passA->merge.0")
        .connect(b_id, PortId(0), merge_id, PortId(1))
        .expect("connect passB->merge.1")
        .connect(merge_id, PortId(0), sink_id, PortId(0))
        .expect("connect merge->sink")
        .build()
        .expect("diamond graph should build");

    assert_eq!(graph.node_count(), 6, "diamond has six nodes");
    assert_eq!(graph.source_nodes().len(), 1, "exactly one source");
    assert_eq!(graph.sink_nodes().len(), 1, "exactly one sink");

    let edges = [
        (source_id, split_id),
        (split_id, a_id),
        (split_id, b_id),
        (a_id, merge_id),
        (b_id, merge_id),
        (merge_id, sink_id),
    ];
    assert_valid_topo_order(graph.execution_order(), &edges);
}

#[test]
fn fan_out_fan_in_structure() {
    // split(4) -> 4x passthrough -> merge(4)
    // A source feeds the split; a sink drains the merge so the graph validates.
    let source = PassthroughFilter::new_source(NodeId(0), "source");
    let split = SplitFilter::new(NodeId(0), "split4", 4);
    let merge = MergeFilter::new(NodeId(0), "merge4", MergeConfig::tiled(4, 1280, 720));
    let sink = NullSink::new(NodeId(0), "sink");

    let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
    let (builder, split_id) = builder.add_node(Box::new(split));
    let (builder, merge_id) = builder.add_node(Box::new(merge));
    let (mut builder_nodes, sink_id) = builder.add_node(Box::new(sink));

    // Four passthrough branches between split and merge.
    let mut branch_ids = Vec::with_capacity(4);
    for i in 0..4 {
        let (b, id) = builder_nodes.add_node(Box::new(PassthroughFilter::new(
            NodeId(0),
            format!("branch_{i}"),
        )));
        builder_nodes = b;
        branch_ids.push(id);
    }

    // source -> split
    let mut builder = builder_nodes
        .connect(source_id, PortId(0), split_id, PortId(0))
        .expect("connect source->split");

    let mut edges = vec![(source_id, split_id)];
    for (i, &branch_id) in branch_ids.iter().enumerate() {
        let port = i as u32;
        builder = builder
            .connect(split_id, PortId(port), branch_id, PortId(0))
            .expect("connect split->branch");
        builder = builder
            .connect(branch_id, PortId(0), merge_id, PortId(port))
            .expect("connect branch->merge");
        edges.push((split_id, branch_id));
        edges.push((branch_id, merge_id));
    }

    let graph = builder
        .connect(merge_id, PortId(0), sink_id, PortId(0))
        .expect("connect merge->sink")
        .build()
        .expect("fan-out/fan-in graph should build");
    edges.push((merge_id, sink_id));

    // source + split + merge + sink + 4 branches + ... wait: 1 source, 1 split,
    // 1 merge, 1 sink, 4 branches = 8. But the slice spec says 11. The slice's
    // node count of 11 assumed split(4)+4 branches+merge(4) plus extra; here we
    // add a source and sink to satisfy graph validation, giving 8 nodes.
    assert_eq!(
        graph.node_count(),
        8,
        "split + merge + sink + source + 4 branches = 8 nodes"
    );
    assert_eq!(graph.source_nodes().len(), 1);
    assert_eq!(graph.sink_nodes().len(), 1);
    assert_valid_topo_order(graph.execution_order(), &edges);
}

#[test]
fn cycle_is_rejected_by_builder() {
    // The A -> B -> A pair forms a 2-node cycle. The builder computes the
    // execution order at build() via Kahn's algorithm and must reject it with
    // GraphError::CycleDetected.
    //
    // A `source` feeds A and a `sink` drains B so that validate() (which runs
    // before the cycle check) is satisfied — exactly one source, one sink, and
    // every required input connected. That isolates the cycle as the reason the
    // build fails.
    //
    //   source -> A -> B -> A   (cycle A<->B)
    //                  B -> sink
    let source = PassthroughFilter::new_source(NodeId(0), "source");
    let node_a = PassthroughFilter::new(NodeId(0), "A");
    let node_b = PassthroughFilter::new(NodeId(0), "B");
    let sink = NullSink::new(NodeId(0), "sink");

    let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
    let (builder, a_id) = builder.add_node(Box::new(node_a));
    let (builder, b_id) = builder.add_node(Box::new(node_b));
    let (builder, sink_id) = builder.add_node(Box::new(sink));

    let result = builder
        .connect(source_id, PortId(0), a_id, PortId(0))
        .expect("connect source->A")
        .connect(a_id, PortId(0), b_id, PortId(0))
        .expect("connect A->B")
        .connect(b_id, PortId(0), a_id, PortId(0))
        .expect("connect B->A (closes the cycle; connect itself succeeds)")
        .connect(b_id, PortId(0), sink_id, PortId(0))
        .expect("connect B->sink")
        .build();

    match result {
        Err(GraphError::CycleDetected(_)) => {}
        Err(other) => panic!("expected CycleDetected, got error {other:?}"),
        Ok(_) => panic!("expected CycleDetected, but the cyclic graph built successfully"),
    }
}
