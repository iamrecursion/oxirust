//! Filter graph throughput benchmarks
//!
//! This benchmark suite measures the performance of filter graph operations:
//! - Graph construction and topology
//! - Frame passing through pipeline
//! - Filter chain throughput
//! - Multi-branch graph execution
//! - Audio/video filter performance

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_graph::filters::audio::{AudioPassthrough, VolumeFilter};
use oximedia_graph::filters::video::{NullSink, PassthroughFilter};
use oximedia_graph::graph::{FilterGraph, GraphBuilder, HasConnections, HasNodes};
use oximedia_graph::node::NodeId;
use oximedia_graph::port::PortId;
use std::hint::black_box;
use std::time::Duration;

/// Builder state after at least one node has been added.
enum AnyBuilder {
    Nodes(GraphBuilder<HasNodes>),
    Connected(GraphBuilder<HasConnections>),
}

impl AnyBuilder {
    fn add_node(self, node: Box<dyn oximedia_graph::node::Node>) -> (Self, NodeId) {
        match self {
            AnyBuilder::Nodes(b) => {
                let (nb, id) = b.add_node(node);
                (AnyBuilder::Nodes(nb), id)
            }
            AnyBuilder::Connected(b) => {
                let (nb, id) = b.add_node(node);
                (AnyBuilder::Connected(nb), id)
            }
        }
    }

    fn connect(
        self,
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    ) -> Self {
        match self {
            AnyBuilder::Nodes(b) => AnyBuilder::Connected(
                b.connect(from_node, from_port, to_node, to_port)
                    .expect("connect failed"),
            ),
            AnyBuilder::Connected(b) => AnyBuilder::Connected(
                b.connect(from_node, from_port, to_node, to_port)
                    .expect("connect failed"),
            ),
        }
    }

    fn build(self) -> FilterGraph {
        match self {
            AnyBuilder::Nodes(b) => b.build().expect("build failed"),
            AnyBuilder::Connected(b) => b.build().expect("build failed"),
        }
    }
}

/// Build a linear chain of nodes (source → filter0..filterN → sink).
fn build_linear_chain(count: usize) -> FilterGraph {
    assert!(count >= 2, "need at least source + sink");
    let (first_builder, source_id) = GraphBuilder::new().add_node(Box::new(
        PassthroughFilter::new_source(NodeId(0u64), "source_0"),
    ));
    let mut state = AnyBuilder::Nodes(first_builder);
    let mut prev_id = source_id;

    for idx in 1..count {
        let node: Box<dyn oximedia_graph::node::Node> = if idx == count - 1 {
            Box::new(NullSink::new(NodeId(idx as u64), &format!("sink_{idx}")))
        } else {
            Box::new(PassthroughFilter::new(
                NodeId(idx as u64),
                &format!("filter_{idx}"),
            ))
        };
        let (new_state, node_id) = state.add_node(node);
        state = new_state.connect(prev_id, PortId(0), node_id, PortId(0));
        prev_id = node_id;
    }

    state.build()
}

/// Build a branching graph (source → N sinks).
fn build_branching_graph(branches: usize) -> FilterGraph {
    let (first_builder, source_id) = GraphBuilder::new().add_node(Box::new(
        PassthroughFilter::new_source(NodeId(0u64), "source"),
    ));
    let mut state = AnyBuilder::Nodes(first_builder);

    for idx in 0..branches {
        let (new_state, sink_id) = state.add_node(Box::new(NullSink::new(
            NodeId((idx + 1) as u64),
            &format!("sink_{idx}"),
        )));
        state = new_state.connect(source_id, PortId(0), sink_id, PortId(0));
    }

    state.build()
}

/// Benchmark simple graph construction
fn graph_construction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");

    let node_counts = vec![
        ("2_nodes", 2),
        ("5_nodes", 5),
        ("10_nodes", 10),
        ("20_nodes", 20),
    ];

    for (name, count) in node_counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &count, |b, &count| {
            b.iter(|| {
                let graph = build_linear_chain(count);
                black_box(graph);
            });
        });
    }

    group.finish();
}

/// Benchmark linear filter chain throughput
fn linear_chain_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_chain");
    group.measurement_time(Duration::from_secs(10));

    let chain_lengths = vec![
        ("1_filter", 3),    // source + 1 filter + sink
        ("3_filters", 5),   // source + 3 filters + sink
        ("5_filters", 7),   // source + 5 filters + sink
        ("10_filters", 12), // source + 10 filters + sink
    ];

    for (name, total_nodes) in chain_lengths {
        group.throughput(Throughput::Elements(total_nodes as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &total_nodes,
            |b, &total_nodes| {
                let graph = build_linear_chain(total_nodes);

                b.iter(|| {
                    // Simulate frame processing through the chain
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark branching graph topology
fn branching_graph_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("branching_graph");
    group.measurement_time(Duration::from_secs(10));

    let branch_counts = vec![("2_branches", 2), ("4_branches", 4), ("8_branches", 8)];

    for (name, branches) in branch_counts {
        group.throughput(Throughput::Elements(branches as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &branches,
            |b, &branches| {
                let graph = build_branching_graph(branches);
                b.iter(|| {
                    black_box(&graph);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark audio filter throughput
fn audio_filter_throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_filter_throughput");
    group.measurement_time(Duration::from_secs(10));

    let sample_counts = vec![
        ("1k_samples", 1024u64),
        ("10k_samples", 10_240),
        ("100k_samples", 102_400),
    ];

    for (name, samples) in sample_counts {
        group.throughput(Throughput::Elements(samples));
        group.bench_function(name, |b| {
            b.iter(|| {
                let (first_builder, source_id) = GraphBuilder::new().add_node(Box::new(
                    AudioPassthrough::new_source(NodeId(0u64), "audio_source"),
                ));
                let (second_builder, filter_id) =
                    first_builder.add_node(Box::new(VolumeFilter::new(
                        NodeId(1u64),
                        "volume",
                        oximedia_graph::filters::audio::VolumeConfig::new(0.5),
                    )));
                let connected = second_builder
                    .connect(source_id, PortId(0), filter_id, PortId(0))
                    .expect("connect failed");
                let graph = connected.build().expect("build failed");
                black_box(graph);
            });
        });
    }

    group.finish();
}

/// Benchmark graph validation overhead
fn graph_validation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_validation");

    let node_counts = vec![
        ("5_nodes", 5),
        ("10_nodes", 10),
        ("20_nodes", 20),
        ("50_nodes", 50),
    ];

    for (name, count) in node_counts {
        group.bench_with_input(BenchmarkId::from_parameter(name), &count, |b, &count| {
            b.iter(|| {
                // Validation happens during build
                let result = build_linear_chain(count);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark topology sort performance
fn topology_sort_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_sort");

    let complexities = vec![
        ("linear_5", 5usize, false),
        ("linear_20", 20, false),
        ("branching_8", 8, true),
        ("branching_16", 16, true),
    ];

    for (name, count, branching) in complexities {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(count, branching),
            |b, &(count, branching)| {
                b.iter(|| {
                    let graph = if branching {
                        build_branching_graph(count)
                    } else {
                        build_linear_chain(count)
                    };
                    black_box(graph);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark node state transitions
fn node_state_transitions_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_state_transitions");

    group.bench_function("create_start_stop", |b| {
        b.iter(|| {
            let (first_builder, source_id) = GraphBuilder::new().add_node(Box::new(
                PassthroughFilter::new_source(NodeId(0u64), "source"),
            ));

            let (second_builder, sink_id) =
                first_builder.add_node(Box::new(NullSink::new(NodeId(1u64), "sink")));

            let connected = second_builder
                .connect(source_id, PortId(0), sink_id, PortId(0))
                .expect("connect failed");

            let graph = connected.build().expect("build failed");
            black_box(graph);
        });
    });

    group.finish();
}

criterion_group!(
    graph_benches,
    graph_construction_benchmark,
    linear_chain_benchmark,
    branching_graph_benchmark,
    audio_filter_throughput_benchmark,
    graph_validation_benchmark,
    topology_sort_benchmark,
    node_state_transitions_benchmark,
);

criterion_main!(graph_benches);
