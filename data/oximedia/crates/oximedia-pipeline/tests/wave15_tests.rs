//! Wave 15 integration tests for oximedia-pipeline.
//!
//! Covers:
//!  - `test_pipeline_validation_disconnected_nodes` — validator flags a
//!    node that is not reachable from any source.
//!  - `test_pipeline_topo_sort_random_graph` — seeded xorshift-generated
//!    DAGs verified against the topological-order invariant.
//!  - `test_planner_large_dag_*` — `ExecutionPlanner` stress / scaling tests
//!    on deterministically generated 1000+ node graphs, asserting structural
//!    correctness (valid topo order, every node scheduled exactly once,
//!    dependencies precede dependents, chain/wide-layer level structure).

use std::collections::{HashMap, HashSet};

use oximedia_pipeline::execution_plan::{ExecutionPlan, ExecutionPlanner};
use oximedia_pipeline::graph::{Edge, PipelineGraph};
use oximedia_pipeline::node::{
    FilterConfig, FrameFormat, NodeId, NodeSpec, SinkConfig, SourceConfig, StreamSpec,
};
use oximedia_pipeline::validation::{PipelineValidator, ValidationError};

// ── helpers ───────────────────────────────────────────────────────────────────

fn vs() -> StreamSpec {
    StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
}

fn filter_node(name: &str) -> NodeSpec {
    NodeSpec::filter(name, FilterConfig::Hflip, vs(), vs())
}

fn source_node(name: &str) -> NodeSpec {
    NodeSpec::source(name, SourceConfig::File("x.mp4".into()), vs())
}

fn sink_node(name: &str) -> NodeSpec {
    NodeSpec::sink(name, SinkConfig::Null, vs())
}

fn add_edge(g: &mut PipelineGraph, from: NodeId, to: NodeId) {
    g.edges.push(Edge {
        from_node: from,
        from_pad: "default".into(),
        to_node: to,
        to_pad: "default".into(),
    });
}

/// Assert that `order` is a valid topological order for `graph`:
/// every edge `(u, v)` must have `u` appearing before `v`.
fn assert_topo_valid(graph: &PipelineGraph, order: &[NodeId]) {
    let pos: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    assert_eq!(
        order.len(),
        graph.nodes.len(),
        "topo order must contain all nodes"
    );
    for edge in &graph.edges {
        let fp = pos.get(&edge.from_node).copied().unwrap_or(usize::MAX);
        let tp = pos.get(&edge.to_node).copied().unwrap_or(usize::MAX);
        assert!(
            fp < tp,
            "topo invariant broken: edge from pos {fp} to pos {tp} (from={:?}, to={:?})",
            edge.from_node,
            edge.to_node
        );
    }
}

// ── Tiny seeded PRNG (32-bit xorshift) ───────────────────────────────────────

struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        // Seed must be non-zero; use 1 as fallback.
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }

    /// Return a value in `0..n` (exclusive). Panics if `n == 0`.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

/// Build a random DAG with `node_count` source-filter nodes using `rng`.
///
/// Strategy: assign each node a "layer" in `0..layer_count`, then for each
/// node in layer `k` draw `edge_density` random targets in layer `k+1..`.
/// This guarantees the graph is acyclic by construction.
fn build_random_dag(node_count: usize, rng: &mut Xorshift32) -> PipelineGraph {
    assert!(node_count >= 2);
    let mut g = PipelineGraph::new();

    // All nodes are filters/sources/sinks; we use a source for node 0,
    // sink for node N-1, and filters for the rest.
    let ids: Vec<NodeId> = (0..node_count)
        .map(|i| {
            let spec = if i == 0 {
                source_node(&format!("n{i}"))
            } else if i == node_count - 1 {
                sink_node(&format!("n{i}"))
            } else {
                filter_node(&format!("n{i}"))
            };
            g.add_node(spec)
        })
        .collect();

    // Add edges only from lower-indexed to higher-indexed nodes so the graph
    // is always a DAG.  Draw ~1-2 edges per intermediate node.
    for from_idx in 0..node_count - 1 {
        // Number of forward edges from this node (1 or 2).
        let n_edges = 1 + rng.next_usize(2);
        for _ in 0..n_edges {
            // Target must have a strictly higher index.
            let range = node_count - from_idx - 1;
            if range == 0 {
                break;
            }
            let to_idx = from_idx + 1 + rng.next_usize(range);
            add_edge(&mut g, ids[from_idx], ids[to_idx]);
        }
    }

    g
}

// ── Test 3: validator flags disconnected (unreachable) node ──────────────────

#[test]
fn test_pipeline_validation_disconnected_nodes() {
    let mut g = PipelineGraph::new();

    // Build a valid source→sink chain.
    let src = g.add_node(source_node("src"));
    let sk = g.add_node(sink_node("sink"));
    add_edge(&mut g, src, sk);

    // Add a filter that is completely disconnected (no input wired, no output
    // wired — it just floats in the graph without being connected to the
    // reachable component).
    g.add_node(filter_node("orphan_filter"));

    let report = PipelineValidator::new().validate(&g);

    // The validator must report at least one error.
    assert!(
        !report.is_valid,
        "graph with disconnected node should be invalid"
    );

    // There must be a DisconnectedNode error naming "orphan_filter".
    let has_disconnected = report.errors.iter().any(|e| match e {
        ValidationError::DisconnectedNode { node_name } => node_name == "orphan_filter",
        _ => false,
    });
    assert!(
        has_disconnected,
        "expected DisconnectedNode error for 'orphan_filter'; got errors: {:?}",
        report.errors
    );
}

// ── Test 4: topological sort is valid on 5 random DAGs of 8–12 nodes ─────────

#[test]
fn test_pipeline_topo_sort_random_graph() {
    // Five deterministic seeds producing different graph shapes.
    let seeds: [(u32, usize); 5] = [
        (0xDEAD_BEEF, 8),
        (0xCAFE_BABE, 9),
        (0x1234_5678, 10),
        (0xABCD_EF01, 11),
        (0xF00D_FACE, 12),
    ];

    for (seed, node_count) in seeds {
        let mut rng = Xorshift32::new(seed);
        let graph = build_random_dag(node_count, &mut rng);

        // The graph must be acyclic by construction, so topo sort must succeed.
        let order = graph
            .topological_sort()
            .unwrap_or_else(|e| panic!("topo sort failed for seed={seed:#010x}: {e}"));

        // (a) All node IDs appear in the order.
        assert_eq!(
            order.len(),
            graph.nodes.len(),
            "seed={seed:#010x}: topo order length {} != node count {}",
            order.len(),
            graph.nodes.len()
        );

        // (b) Every edge points forward in the sort order.
        assert_topo_valid(&graph, &order);
    }
}

// ── ExecutionPlanner large-graph stress / scaling tests ──────────────────────
//
// The existing planner tests in `src/execution_plan.rs` only exercise graphs
// of fewer than ~10 nodes.  These tests pin behaviour on 1000+ node graphs:
// the resulting plan must be a structurally valid topological schedule, every
// node must be scheduled exactly once, dependencies must precede dependents,
// and degenerate shapes (deep chain, wide single layer) must yield the correct
// level structure without panicking.
//
// All graphs are generated deterministically (seeded xorshift32 or fully
// structural construction) so the assertions are reproducible.  Only STRUCTURAL
// correctness is asserted — never wall-clock timing.

/// Flatten an `ExecutionPlan` into its scheduled node order and assert that it
/// is a valid topological order for `graph` (every edge points forward) and
/// that every node appears exactly once.
fn assert_plan_topo_valid(graph: &PipelineGraph, plan: &ExecutionPlan) {
    let order = plan.node_order();

    // Exactly one slot per node, no duplicates, no drops.
    assert_eq!(
        order.len(),
        graph.nodes.len(),
        "plan node_order length {} != graph node count {}",
        order.len(),
        graph.nodes.len()
    );
    let unique: HashSet<NodeId> = order.iter().copied().collect();
    assert_eq!(
        unique.len(),
        graph.nodes.len(),
        "plan scheduled a node more than once (unique {} != nodes {})",
        unique.len(),
        graph.nodes.len()
    );
    assert_eq!(
        unique,
        graph.nodes.keys().copied().collect::<HashSet<NodeId>>(),
        "scheduled node set differs from graph node set"
    );

    // Every edge u→v must have u scheduled strictly before v.
    let pos: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    for edge in &graph.edges {
        let fp = pos
            .get(&edge.from_node)
            .copied()
            .expect("edge source must be scheduled");
        let tp = pos
            .get(&edge.to_node)
            .copied()
            .expect("edge target must be scheduled");
        assert!(
            fp < tp,
            "plan topo invariant broken: edge scheduled from pos {fp} to pos {tp}"
        );
    }
}

/// Assert that within an `ExecutionPlan` every stage's declared dependencies
/// reference strictly-earlier stages (dependencies precede dependents) and that
/// every edge's predecessor stage is `<` its successor stage (levels respect
/// the DAG).  Stage 0 must have no dependencies.
fn assert_plan_levels_well_ordered(graph: &PipelineGraph, plan: &ExecutionPlan) {
    // Map each node to the stage (level) it was assigned to.
    let mut node_stage: HashMap<NodeId, usize> = HashMap::new();
    for stage in &plan.stages {
        for &nid in &stage.nodes {
            node_stage.insert(nid, stage.stage_id);
        }
    }

    for stage in &plan.stages {
        // Declared dependencies must all be earlier stages.
        for &dep in &stage.dependencies {
            assert!(
                dep < stage.stage_id,
                "stage {} lists dependency {} which is not strictly earlier",
                stage.stage_id,
                dep
            );
        }
        if stage.stage_id == 0 {
            assert!(
                stage.dependencies.is_empty(),
                "stage 0 must have no dependencies, got {:?}",
                stage.dependencies
            );
        }
    }

    // Every edge must cross from a lower level to a higher level.
    for edge in &graph.edges {
        let from_level = node_stage
            .get(&edge.from_node)
            .copied()
            .expect("edge source must be assigned a stage");
        let to_level = node_stage
            .get(&edge.to_node)
            .copied()
            .expect("edge target must be assigned a stage");
        assert!(
            from_level < to_level,
            "edge crosses from stage {from_level} to stage {to_level}; \
             dependent must be in a strictly later stage"
        );
    }
}

/// Build a large *layered* DAG with at least `min_nodes` nodes.
///
/// Nodes are partitioned into `layers` layers.  Edges are drawn only from a
/// node in layer `k` to a node in some layer `> k`, guaranteeing acyclicity.
/// Every non-first-layer node is given at least one incoming edge from the
/// immediately preceding layer so the graph stays connected and exercises
/// multi-level planning.  Returns the graph and the per-layer node-id lists.
fn build_layered_dag(
    min_nodes: usize,
    layers: usize,
    rng: &mut Xorshift32,
) -> (PipelineGraph, Vec<Vec<NodeId>>) {
    assert!(layers >= 2);
    let mut g = PipelineGraph::new();

    // Distribute nodes roughly evenly across layers, rounding up so the total
    // is >= min_nodes.
    let per_layer = min_nodes.div_ceil(layers);
    let mut layer_ids: Vec<Vec<NodeId>> = Vec::with_capacity(layers);

    let mut counter = 0usize;
    for layer in 0..layers {
        let mut ids = Vec::with_capacity(per_layer);
        for _ in 0..per_layer {
            let name = format!("L{layer}_n{counter}");
            let spec = if layer == 0 {
                source_node(&name)
            } else if layer == layers - 1 {
                sink_node(&name)
            } else {
                filter_node(&name)
            };
            ids.push(g.add_node(spec));
            counter += 1;
        }
        layer_ids.push(ids);
    }

    // Wire edges.  Each non-first-layer node gets >=1 parent in the previous
    // layer (keeps it connected), plus a few extra forward edges to deeper
    // layers for irregular fan-in/fan-out.
    for layer in 1..layers {
        let prev = layer - 1;
        for &to in &layer_ids[layer] {
            // Guaranteed parent from the immediately preceding layer.
            let prev_len = layer_ids[prev].len();
            let from = layer_ids[prev][rng.next_usize(prev_len)];
            add_edge(&mut g, from, to);

            // A few additional forward edges from any earlier layer.
            let extra = rng.next_usize(3);
            for _ in 0..extra {
                let src_layer = rng.next_usize(layer); // 0..layer
                let src_len = layer_ids[src_layer].len();
                let from2 = layer_ids[src_layer][rng.next_usize(src_len)];
                add_edge(&mut g, from2, to);
            }
        }
    }

    (g, layer_ids)
}

// ── Test: 1000-node DAG produces a valid topological schedule ────────────────

#[test]
fn test_planner_large_dag_valid_topo_order() {
    // Layered DAG with >= 1000 nodes across 20 layers (50 per layer).
    let mut rng = Xorshift32::new(0x5EED_1000);
    let (graph, _layers) = build_layered_dag(1000, 20, &mut rng);

    assert!(
        graph.node_count() >= 1000,
        "expected >= 1000 nodes, got {}",
        graph.node_count()
    );

    let plan = ExecutionPlanner::plan(&graph).expect("planner should handle a large valid DAG");

    // The flattened plan is a valid topological order with each node once.
    assert_plan_topo_valid(&graph, &plan);

    // The level assignment itself must be a well-ordered layering.
    assert_plan_levels_well_ordered(&graph, &plan);

    // A non-trivial layered DAG should plan into more than one stage but never
    // more stages than nodes.
    assert!(
        plan.stage_count() > 1,
        "layered DAG should have many stages"
    );
    assert!(
        plan.stage_count() <= graph.node_count(),
        "stage count {} must not exceed node count {}",
        plan.stage_count(),
        graph.node_count()
    );
}

// ── Test: every node is scheduled exactly once (no drops, no duplicates) ─────

#[test]
fn test_planner_large_dag_all_scheduled_once() {
    // Use a different seed and shape to broaden coverage.
    let mut rng = Xorshift32::new(0xABCD_2000);
    let (graph, _layers) = build_layered_dag(1200, 30, &mut rng);

    let plan = ExecutionPlanner::plan(&graph).expect("plan should succeed");

    // Count via stages directly (independent of node_order helper).
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut total = 0usize;
    for stage in &plan.stages {
        for &nid in &stage.nodes {
            assert!(
                seen.insert(nid),
                "node scheduled in more than one stage: {nid:?}"
            );
            total += 1;
        }
    }

    assert_eq!(
        total,
        graph.node_count(),
        "total scheduled nodes {} != graph node count {}",
        total,
        graph.node_count()
    );
    // Scheduled set must equal the input node set exactly.
    let input: HashSet<NodeId> = graph.nodes.keys().copied().collect();
    assert_eq!(seen, input, "scheduled node set != input node set");
}

// ── Test: fan-in / fan-out heavy graph plans with correct level ordering ─────

#[test]
fn test_planner_fan_in_fan_out_heavy() {
    // Construct a deliberately high-degree graph:
    //   layer 0:   1 source hub  (fan-out to all of layer 1)
    //   layer 1:   500 filters   (each consumes the hub; fan-in to layer-2 hub)
    //   layer 2:   1 merge hub   (fan-in from all of layer 1; fan-out to layer 3)
    //   layer 3:   500 sinks     (each consumes the layer-2 hub)
    //
    // The two hubs have in/out degree ~500 each, far beyond the existing
    // small-graph tests.
    let mut g = PipelineGraph::new();
    let width = 500usize;

    let src_hub = g.add_node(source_node("src_hub"));

    let mid: Vec<NodeId> = (0..width)
        .map(|i| g.add_node(filter_node(&format!("mid{i}"))))
        .collect();

    let merge_hub = g.add_node(filter_node("merge_hub"));

    let sinks: Vec<NodeId> = (0..width)
        .map(|i| g.add_node(sink_node(&format!("sink{i}"))))
        .collect();

    // Fan-out from source hub to every mid node (out-degree = width).
    for &m in &mid {
        add_edge(&mut g, src_hub, m);
    }
    // Fan-in from every mid node into the merge hub (in-degree = width).
    for &m in &mid {
        add_edge(&mut g, m, merge_hub);
    }
    // Fan-out from merge hub to every sink (out-degree = width).
    for &s in &sinks {
        add_edge(&mut g, merge_hub, s);
    }

    let expected_nodes = 1 + width + 1 + width;
    assert_eq!(g.node_count(), expected_nodes);

    let plan = ExecutionPlanner::plan(&g).expect("plan should succeed on high-degree graph");

    // Structural correctness: valid topo order, every node once, levels sound.
    assert_plan_topo_valid(&g, &plan);
    assert_plan_levels_well_ordered(&g, &plan);

    // Depth structure is exact for this construction:
    //   src_hub   -> level 0
    //   mid[*]    -> level 1
    //   merge_hub -> level 2
    //   sink[*]   -> level 3
    // => 4 stages total.
    assert_eq!(
        plan.stage_count(),
        4,
        "expected 4 depth levels (hub/mids/merge/sinks)"
    );

    // The two wide stages (level 1 mids and level 3 sinks) each have `width`
    // nodes and must be flagged parallel.
    let wide_stages: Vec<&_> = plan
        .stages
        .iter()
        .filter(|s| s.nodes.len() == width)
        .collect();
    assert_eq!(
        wide_stages.len(),
        2,
        "expected exactly two stages of width {width}"
    );
    for s in wide_stages {
        assert!(s.parallel, "a {width}-node stage must be parallel");
    }

    // The merge hub stage must depend on the mid stage (level 1), proving the
    // dependency precedes the dependent.
    let merge_stage = plan
        .stages
        .iter()
        .find(|s| s.nodes.contains(&merge_hub))
        .expect("merge hub must be scheduled");
    assert!(
        merge_stage.dependencies.contains(&1),
        "merge hub stage must depend on the mid layer (stage 1), got {:?}",
        merge_stage.dependencies
    );
}

// ── Test: deep chain and wide single layer yield correct level structure ─────

#[test]
fn test_planner_chain_and_wide_layer_structure() {
    // (a) A 1000-node linear chain: n0 -> n1 -> ... -> n999.
    //     Each node sits at its own depth level, so the plan must have exactly
    //     1000 sequential stages, each containing exactly one node, and none of
    //     them parallel.
    {
        let chain_len = 1000usize;
        let mut g = PipelineGraph::new();
        let ids: Vec<NodeId> = (0..chain_len)
            .map(|i| {
                let spec = if i == 0 {
                    source_node(&format!("c{i}"))
                } else if i == chain_len - 1 {
                    sink_node(&format!("c{i}"))
                } else {
                    filter_node(&format!("c{i}"))
                };
                g.add_node(spec)
            })
            .collect();
        for w in ids.windows(2) {
            add_edge(&mut g, w[0], w[1]);
        }

        let plan = ExecutionPlanner::plan(&g).expect("chain must plan without panic");

        // Liveness/structure: every node scheduled once, valid topo order.
        assert_plan_topo_valid(&g, &plan);
        assert_plan_levels_well_ordered(&g, &plan);

        // Exactly `chain_len` sequential stages, one node each, none parallel.
        assert_eq!(
            plan.stage_count(),
            chain_len,
            "1000-node chain must produce 1000 sequential stages"
        );
        for stage in &plan.stages {
            assert_eq!(stage.nodes.len(), 1, "each chain stage holds one node");
            assert!(!stage.parallel, "single-node chain stage is not parallel");
        }
        // Generous liveness bound: no stage references a future dependency.
        // (Already covered by assert_plan_levels_well_ordered, but assert the
        // chain-specific dependency edge as well.)
        for (i, stage) in plan.stages.iter().enumerate() {
            if i == 0 {
                assert!(stage.dependencies.is_empty());
            } else {
                assert!(
                    stage.dependencies.contains(&(i - 1)),
                    "chain stage {i} must depend on stage {}",
                    i - 1
                );
            }
        }
    }

    // (b) A 1000-wide single layer: 1000 isolated source nodes with no edges.
    //     All sit at depth 0, so the plan must collapse to exactly ONE stage
    //     containing all 1000 nodes, flagged parallel, with 1000 independent
    //     parallel groups (no shared ancestors).
    {
        let width = 1000usize;
        let mut g = PipelineGraph::new();
        let ids: Vec<NodeId> = (0..width)
            .map(|i| g.add_node(source_node(&format!("w{i}"))))
            .collect();
        assert_eq!(g.edge_count(), 0);

        let plan = ExecutionPlanner::plan(&g).expect("wide layer must plan without panic");

        assert_plan_topo_valid(&g, &plan);

        // Exactly one level.
        assert_eq!(
            plan.stage_count(),
            1,
            "1000 isolated sources must collapse to a single stage"
        );
        let only = &plan.stages[0];
        assert_eq!(only.nodes.len(), width, "the single stage holds all nodes");
        assert!(only.parallel, "a 1000-node stage must be parallel");
        assert!(only.dependencies.is_empty(), "stage 0 has no dependencies");

        // With no edges, no two nodes share an ancestor => 1000 groups, and the
        // plan's max parallelism equals the layer width.
        assert_eq!(
            only.parallel_groups.len(),
            width,
            "independent sources form one parallel group each"
        );
        assert_eq!(plan.max_parallelism(), width);

        // Every id is present exactly once in the single stage.
        let set: HashSet<NodeId> = only.nodes.iter().copied().collect();
        assert_eq!(set.len(), width);
        for id in &ids {
            assert!(set.contains(id));
        }
    }
}
