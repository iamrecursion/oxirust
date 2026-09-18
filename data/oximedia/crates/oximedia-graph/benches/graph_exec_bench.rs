//! Throughput benchmark for graph execution ordering.
//!
//! Measures [`ProcessingGraph::execution_order`] (a deterministic, sorted
//! Kahn's-algorithm topological sort) across a range of node counts. The graph
//! is generated deterministically as a linear chain plus seeded forward-only
//! cross-edges, keeping it a DAG without any external RNG dependency (an inline
//! xorshift PRNG is used to pick forward `to` indices).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_graph::processing_graph::{GraphNode, NodeType, ProcessingGraph};

/// Tiny deterministic xorshift64 PRNG (no `rand` dependency).
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which is a fixed point of xorshift.
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform-ish value in `[0, bound)`. Returns 0 when `bound == 0`.
    fn next_below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next_u64() % bound
        }
    }
}

/// Build a deterministic DAG of `n` nodes.
///
/// Topology:
///   * node 0 is a `Source`, node `n-1` is a `Sink`, the rest are `Filter`s;
///   * a linear chain `i -> i+1` guarantees connectivity and a unique source;
///   * each node `i` also gets up to two seeded *forward* cross-edges
///     `i -> j` with `j > i`, which keeps the graph acyclic while exercising
///     the in-degree bookkeeping of Kahn's algorithm.
fn build_dag(n: u64) -> ProcessingGraph {
    let mut graph = ProcessingGraph::new();
    for i in 0..n {
        let node_type = if i == 0 {
            NodeType::Source
        } else if i + 1 == n {
            NodeType::Sink
        } else {
            NodeType::Filter
        };
        graph.add_node(GraphNode::new(i, &format!("n{i}"), node_type));
    }

    // Linear backbone.
    for i in 0..n.saturating_sub(1) {
        graph.connect(i, 0, i + 1, 0);
    }

    // Seeded forward-only cross-edges.
    let mut rng = XorShift64::new(0xC0FF_EE00_1234_5678 ^ n);
    for i in 0..n {
        // The window of valid forward targets strictly after the immediate
        // successor (which is already the backbone edge).
        let remaining = n.saturating_sub(i + 2);
        if remaining == 0 {
            continue;
        }
        for _ in 0..2 {
            let offset = rng.next_below(remaining) + 2;
            let to = i + offset;
            if to < n {
                // `connect` returns false only for missing nodes; all ids exist.
                graph.connect(i, 0, to, 0);
            }
        }
    }

    graph
}

fn bench_execution_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_execution_order");
    for &n in &[16_u64, 64, 256, 1024] {
        let graph = build_dag(n);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &graph, |b, g| {
            b.iter(|| black_box(g.execution_order()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_execution_order);
criterion_main!(benches);
