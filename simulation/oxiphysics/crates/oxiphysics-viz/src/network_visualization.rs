// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Network graph visualization for the OxiPhysics engine.
//!
//! Provides:
//! - `NetworkGraph` — nodes, edges, adjacency
//! - `ForceDirectedLayout` — Fruchterman-Reingold / Barnes-Hut
//! - `HierarchicalLayout` — Sugiyama layer assignment & crossing minimization
//! - `NetworkMetrics` — degree, clustering, betweenness, PageRank
//! - `CommunityDetection` — modularity, Louvain, spectral, label propagation
//! - `NetworkAnimation` — time-evolving graph with interpolation

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::Error;

mod barnes_hut;
mod community;
mod hierarchical;

#[cfg(test)]
mod regression_tests;

// ---------------------------------------------------------------------------
// NetworkGraph — core data structure
// ---------------------------------------------------------------------------

/// A node in the network graph.
#[derive(Debug, Clone)]
pub struct NetworkNode {
    /// Unique node identifier.
    pub id: usize,
    /// 2-D position for layout.
    pub position: [f64; 2],
    /// Display size (radius).
    pub size: f32,
    /// RGBA color.
    pub color: [f32; 4],
    /// Human-readable label.
    pub label: String,
    /// Optional community assignment.
    pub community: Option<usize>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl NetworkNode {
    /// Create a node at the origin.
    pub fn new(id: usize, label: impl Into<String>) -> Self {
        Self {
            id,
            position: [0.0, 0.0],
            size: 1.0,
            color: [0.4, 0.6, 1.0, 1.0],
            label: label.into(),
            community: None,
            metadata: HashMap::new(),
        }
    }

    /// Set position.
    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = [x, y];
        self
    }

    /// Set size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Set color.
    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }
}

/// A directed or undirected edge.
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    /// Source node id.
    pub source: usize,
    /// Target node id.
    pub target: usize,
    /// Edge weight.
    pub weight: f64,
    /// If true, edge is directed source→target.
    pub directed: bool,
    /// Optional label.
    pub label: Option<String>,
}

impl NetworkEdge {
    /// Create an undirected edge with unit weight.
    pub fn undirected(source: usize, target: usize) -> Self {
        Self {
            source,
            target,
            weight: 1.0,
            directed: false,
            label: None,
        }
    }

    /// Create a directed edge with given weight.
    pub fn directed(source: usize, target: usize, weight: f64) -> Self {
        Self {
            source,
            target,
            weight,
            directed: true,
            label: None,
        }
    }

    /// With label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Network graph: nodes + edges + adjacency list.
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    /// All nodes.
    pub nodes: Vec<NetworkNode>,
    /// All edges.
    pub edges: Vec<NetworkEdge>,
    /// Adjacency list: node_id → list of (neighbor_id, edge_weight).
    pub adjacency: HashMap<usize, Vec<(usize, f64)>>,
}

impl NetworkGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add a node. Returns the node id.
    pub fn add_node(&mut self, node: NetworkNode) -> usize {
        let id = node.id;
        self.adjacency.entry(id).or_default();
        self.nodes.push(node);
        id
    }

    /// Add an edge and update adjacency.
    ///
    /// The edge is stored as given: its end points are *not* checked against
    /// [`nodes`](Self::nodes), so an edge may be added before the nodes it
    /// joins.  The layouts, the metrics and the community detection treat an
    /// edge as absent while one of its end points is not a node of the graph,
    /// whereas [`bfs`](Self::bfs), [`dfs`](Self::dfs),
    /// [`neighbors`](Self::neighbors) and [`degree`](Self::degree) report the
    /// adjacency lists as they are stored.  Use
    /// [`try_add_edge`](Self::try_add_edge) to refuse such an edge instead.
    pub fn add_edge(&mut self, edge: NetworkEdge) {
        self.link(edge);
    }

    /// Add an edge after checking that both of its end points are nodes of the
    /// graph.
    ///
    /// The check scans the node list, like [`node`](Self::node).
    ///
    /// # Errors
    ///
    /// [`Error::NotFound`] naming the source (checked first) or the target if
    /// it is not a node; the graph is left unchanged.
    pub fn try_add_edge(&mut self, edge: NetworkEdge) -> crate::error::Result<()> {
        let mut has_source = false;
        let mut has_target = false;
        for node in &self.nodes {
            has_source |= node.id == edge.source;
            has_target |= node.id == edge.target;
            if has_source && has_target {
                break;
            }
        }
        if !has_source {
            return Err(Error::not_found(format!("node {}", edge.source)));
        }
        if !has_target {
            return Err(Error::not_found(format!("node {}", edge.target)));
        }
        self.link(edge);
        Ok(())
    }

    /// Record `edge` in the edge list and in the adjacency lists, without
    /// looking at its end points.
    fn link(&mut self, edge: NetworkEdge) {
        self.adjacency
            .entry(edge.source)
            .or_default()
            .push((edge.target, edge.weight));
        if !edge.directed {
            self.adjacency
                .entry(edge.target)
                .or_default()
                .push((edge.source, edge.weight));
        }
        self.edges.push(edge);
    }

    /// Return the degree of a node (number of neighbors in adjacency list).
    pub fn degree(&self, node_id: usize) -> usize {
        self.adjacency.get(&node_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Return neighbors of a node.
    pub fn neighbors(&self, node_id: usize) -> Vec<usize> {
        self.adjacency
            .get(&node_id)
            .map(|v| v.iter().map(|(n, _)| *n).collect())
            .unwrap_or_default()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if edge (u, v) exists.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adjacency
            .get(&u)
            .map(|ns| ns.iter().any(|(n, _)| *n == v))
            .unwrap_or(false)
    }

    /// BFS from start, returns visited node ids in BFS order.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(node) = queue.pop_front() {
            result.push(node);
            for neighbor in self.neighbors(node) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        result
    }

    /// DFS from start, returns visited node ids in DFS (pre-)order.
    ///
    /// Neighbours are visited in adjacency-list order.  The search keeps its
    /// own stack, so the length of the longest path is limited by memory and
    /// not by the call stack.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }
            result.push(node);
            if let Some(adjacent) = self.adjacency.get(&node) {
                // Pushed in reverse so that the first neighbour is popped
                // first, as in the recursive formulation of the search.
                for &(neighbor, _) in adjacent.iter().rev() {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        result
    }

    /// Build a simple adjacency matrix (node indices in `nodes` order).
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.nodes.len();
        let id_to_idx: HashMap<usize, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id, i))
            .collect();
        let mut mat = vec![vec![0.0f64; n]; n];
        for edge in &self.edges {
            if let (Some(&i), Some(&j)) = (id_to_idx.get(&edge.source), id_to_idx.get(&edge.target))
            {
                mat[i][j] = edge.weight;
                if !edge.directed {
                    mat[j][i] = edge.weight;
                }
            }
        }
        mat
    }

    /// Get mutable reference to node by id.
    pub fn node_mut(&mut self, id: usize) -> Option<&mut NetworkNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Get reference to node by id.
    pub fn node(&self, id: usize) -> Option<&NetworkNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

impl Default for NetworkGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ForceDirectedLayout — Fruchterman-Reingold with Barnes-Hut approximation
// ---------------------------------------------------------------------------

/// Configuration for force-directed layout.
#[derive(Debug, Clone)]
pub struct ForceDirectedConfig {
    /// Number of iterations.
    pub iterations: usize,
    /// Optimal distance between nodes (k).
    pub optimal_distance: f64,
    /// Initial temperature.
    pub temperature: f64,
    /// Cooling factor (multiplied each iteration).
    pub cooling: f64,
    /// Repulsion coefficient.
    pub repulsion: f64,
    /// Attraction coefficient.
    pub attraction: f64,
    /// Barnes-Hut opening angle (0 = exact, higher = faster approximation).
    ///
    /// [`ForceDirectedLayout::apply`] approximates the repulsion only when
    /// this is positive and the graph has more than 256 nodes; smaller graphs
    /// always use the exact sum.
    pub barnes_hut_theta: f64,
}

impl Default for ForceDirectedConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            optimal_distance: 50.0,
            temperature: 100.0,
            cooling: 0.95,
            repulsion: 1.0,
            attraction: 1.0,
            barnes_hut_theta: 0.5,
        }
    }
}

/// Graphs with more nodes than this use the Barnes-Hut approximation for the
/// repulsive forces (when its `theta` is positive); below it, building the
/// tree costs more than the exact sum over all pairs saves.
const BARNES_HUT_MIN_NODES: usize = 256;

/// Force-directed graph layout (Fruchterman-Reingold).
#[derive(Debug, Clone)]
pub struct ForceDirectedLayout {
    /// Layout configuration.
    pub config: ForceDirectedConfig,
}

impl ForceDirectedLayout {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: ForceDirectedConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ForceDirectedConfig) -> Self {
        Self { config }
    }

    /// Run layout, modifying node positions in-place.
    ///
    /// Edges with an end point that is not a node of the graph, and
    /// self-loops, exert no force.  With a positive
    /// [`barnes_hut_theta`](ForceDirectedConfig::barnes_hut_theta) and more
    /// than 256 nodes the repulsive forces are approximated on a Barnes-Hut
    /// quadtree (see [`barnes_hut_repulsion`](Self::barnes_hut_repulsion));
    /// otherwise they are summed exactly over all pairs of nodes.
    pub fn apply(&self, graph: &mut NetworkGraph) {
        let n = graph.nodes.len();
        if n == 0 {
            return;
        }
        let k = self.config.optimal_distance;
        // Initialize positions if at origin
        if graph.nodes.iter().all(|nd| nd.position == [0.0, 0.0]) {
            let side = (n as f64).sqrt() * k;
            for (i, nd) in graph.nodes.iter_mut().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                nd.position = [side * angle.cos(), side * angle.sin()];
            }
        }

        // The edges as pairs of indices into `graph.nodes`.  A node id that
        // occurs twice stands for its first occurrence.
        let mut index_of: HashMap<usize, usize> = HashMap::with_capacity(n);
        for (i, nd) in graph.nodes.iter().enumerate() {
            index_of.entry(nd.id).or_insert(i);
        }
        let links: Vec<(usize, usize)> = graph
            .edges
            .iter()
            .filter_map(|edge| {
                let source = *index_of.get(&edge.source)?;
                let target = *index_of.get(&edge.target)?;
                (source != target).then_some((source, target))
            })
            .collect();

        let mut positions: Vec<[f64; 2]> = graph.nodes.iter().map(|nd| nd.position).collect();
        let theta = self.config.barnes_hut_theta;
        let use_quadtree = theta > 0.0 && n > BARNES_HUT_MIN_NODES;
        let repulsion_strength = self.config.repulsion * k * k;
        let mut temp = self.config.temperature;

        for _iter in 0..self.config.iterations {
            // Repulsive forces
            let mut disp = if use_quadtree {
                barnes_hut::approximate_repulsion(&positions, theta, repulsion_strength)
            } else {
                barnes_hut::exact_repulsion(&positions, repulsion_strength)
            };

            // Attractive forces
            for &(source, target) in &links {
                let pu = positions[source];
                let pv = positions[target];
                let dx = pu[0] - pv[0];
                let dy = pu[1] - pv[1];
                let dist = (dx * dx + dy * dy).sqrt().max(barnes_hut::MIN_DISTANCE);
                let force = self.config.attraction * dist * dist / k;
                let fx = force * dx / dist;
                let fy = force * dy / dist;
                disp[source][0] -= fx;
                disp[source][1] -= fy;
                disp[target][0] += fx;
                disp[target][1] += fy;
            }

            // Apply displacements, clamped by temperature
            for (position, d) in positions.iter_mut().zip(&disp) {
                let dmag = (d[0] * d[0] + d[1] * d[1]).sqrt().max(0.001);
                let scale = dmag.min(temp) / dmag;
                position[0] += d[0] * scale;
                position[1] += d[1] * scale;
            }

            temp *= self.config.cooling;
        }

        for (nd, position) in graph.nodes.iter_mut().zip(positions) {
            nd.position = position;
        }
    }

    /// Barnes-Hut approximation of the repulsive forces of the layout.
    ///
    /// The nodes are inserted into a quadtree whose cells know their number of
    /// nodes and centre of mass.  A cell is treated as one body at its centre
    /// of mass when `width / distance` is below
    /// [`barnes_hut_theta`](ForceDirectedConfig::barnes_hut_theta) and is
    /// opened otherwise; leaves contribute the exact repulsion of their nodes.
    /// A cell that contains the node itself is always opened.  A `theta` of 0
    /// (or less) opens every cell and gives the exact sum over all pairs.
    ///
    /// Returns a per-node displacement map (repulsion only, no attraction).
    pub fn barnes_hut_repulsion(&self, graph: &NetworkGraph) -> HashMap<usize, [f64; 2]> {
        let k = self.config.optimal_distance;
        let positions: Vec<[f64; 2]> = graph.nodes.iter().map(|nd| nd.position).collect();
        let displacement = barnes_hut::approximate_repulsion(
            &positions,
            self.config.barnes_hut_theta,
            self.config.repulsion * k * k,
        );
        graph
            .nodes
            .iter()
            .map(|nd| nd.id)
            .zip(displacement)
            .collect()
    }
}

impl Default for ForceDirectedLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HierarchicalLayout — Sugiyama framework
// ---------------------------------------------------------------------------

/// Configuration for hierarchical layout.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Vertical layer separation.
    pub layer_separation: f64,
    /// Horizontal node separation.
    pub node_separation: f64,
    /// Number of crossing minimization passes.
    pub crossing_passes: usize,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            layer_separation: 100.0,
            node_separation: 60.0,
            crossing_passes: 4,
        }
    }
}

/// Hierarchical (Sugiyama) graph layout.
#[derive(Debug, Clone)]
pub struct HierarchicalLayout {
    /// Layout configuration.
    pub config: HierarchicalConfig,
}

impl HierarchicalLayout {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: HierarchicalConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: HierarchicalConfig) -> Self {
        Self { config }
    }

    /// Run layout, modifying node positions in-place.
    ///
    /// Step 1: Layer assignment (longest-path).
    /// Step 2: Crossing minimization (barycenter heuristic).
    /// Step 3: Coordinate assignment.
    pub fn apply(&self, graph: &mut NetworkGraph) {
        if graph.nodes.is_empty() {
            return;
        }
        let layers = self.assign_layers(graph);
        let ordered = self.minimize_crossings(graph, &layers);
        self.assign_coordinates(graph, &ordered);
    }

    /// Assign layer (rank) to each node using longest-path layering.
    ///
    /// A node goes one layer below its lowest-placed predecessor, following
    /// every edge from its `source` to its `target` whether or not it is
    /// marked `directed`.  Edges that close a cycle are reversed first, so
    /// cyclic graphs are layered too.  Self-loops and edges with an end point
    /// that is not a node are ignored.  Runs in `O(V + E)`.
    pub fn assign_layers(&self, graph: &NetworkGraph) -> HashMap<usize, usize> {
        hierarchical::assign_layers(graph)
    }

    /// Barycenter crossing minimization.
    ///
    /// Every layer is seeded in ascending id order and then reordered by
    /// alternating sweeps, down by the mean position of a node's neighbours in
    /// the layer above and up by those in the layer below; edges count in
    /// either direction.  At most
    /// [`crossing_passes`](HierarchicalConfig::crossing_passes) passes run,
    /// fewer if no crossing is left or a pass changes nothing.  The ordering
    /// with the fewest crossings is returned, so the result is never worse than
    /// the seed.  It does not depend on the iteration order of `layers`.
    ///
    /// Returns layers as ordered lists of node ids.
    pub fn minimize_crossings(
        &self,
        graph: &NetworkGraph,
        layers: &HashMap<usize, usize>,
    ) -> Vec<Vec<usize>> {
        hierarchical::minimize_crossings(graph, layers, self.config.crossing_passes)
    }

    /// Number of edge crossings in the drawing that places `ordered_layers`
    /// (as returned by [`minimize_crossings`](Self::minimize_crossings)) one
    /// below the other.
    ///
    /// Only edges between adjacent layers are counted, in either direction.
    /// Parallel edges count individually, edges that share an end point never
    /// cross, and edges with an end point that is not in `ordered_layers` are
    /// ignored.
    pub fn count_crossings(&self, graph: &NetworkGraph, ordered_layers: &[Vec<usize>]) -> usize {
        hierarchical::crossing_count(graph, ordered_layers)
    }

    /// Assign (x, y) coordinates based on ordered layers.
    pub fn assign_coordinates(&self, graph: &mut NetworkGraph, ordered_layers: &[Vec<usize>]) {
        // A node id that occurs twice stands for its first occurrence.
        let mut index_of: HashMap<usize, usize> = HashMap::with_capacity(graph.nodes.len());
        for (i, nd) in graph.nodes.iter().enumerate() {
            index_of.entry(nd.id).or_insert(i);
        }
        for (layer_idx, layer) in ordered_layers.iter().enumerate() {
            let y = layer_idx as f64 * self.config.layer_separation;
            let total_w = layer.len() as f64 * self.config.node_separation;
            let start_x = -total_w / 2.0;
            for (pos_idx, &node_id) in layer.iter().enumerate() {
                let x = start_x + pos_idx as f64 * self.config.node_separation;
                if let Some(&index) = index_of.get(&node_id) {
                    graph.nodes[index].position = [x, y];
                }
            }
        }
    }
}

impl Default for HierarchicalLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NetworkMetrics — graph analysis
// ---------------------------------------------------------------------------

/// Degree distribution of the network.
#[derive(Debug, Clone)]
pub struct DegreeDistribution {
    /// Map from degree → number of nodes with that degree.
    pub counts: HashMap<usize, usize>,
    /// Total number of nodes.
    pub total_nodes: usize,
}

impl DegreeDistribution {
    /// Probability mass function at degree k.
    pub fn pmf(&self, k: usize) -> f64 {
        if self.total_nodes == 0 {
            return 0.0;
        }
        *self.counts.get(&k).unwrap_or(&0) as f64 / self.total_nodes as f64
    }

    /// Mean degree.
    pub fn mean_degree(&self) -> f64 {
        if self.total_nodes == 0 {
            return 0.0;
        }
        let total: usize = self.counts.iter().map(|(k, c)| k * c).sum();
        total as f64 / self.total_nodes as f64
    }
}

/// Network analysis metrics.
///
/// Every metric ignores the edges that have an end point which is not a node
/// of the graph (see [`NetworkGraph::add_edge`]).
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    graph: NetworkGraph,
    /// Ids of the nodes of `graph`.
    node_ids: HashSet<usize>,
}

impl NetworkMetrics {
    /// Create from a graph (takes ownership of it).
    pub fn new(graph: NetworkGraph) -> Self {
        let node_ids = graph.nodes.iter().map(|n| n.id).collect();
        Self { graph, node_ids }
    }

    /// Neighbours of `node_id` that are nodes of the graph, in adjacency order.
    fn neighbors_of(&self, node_id: usize) -> Vec<usize> {
        self.graph
            .adjacency
            .get(&node_id)
            .map(|adjacent| {
                adjacent
                    .iter()
                    .map(|&(neighbor, _)| neighbor)
                    .filter(|neighbor| self.node_ids.contains(neighbor))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Number of entries of `node_id`'s adjacency list that are nodes.
    fn degree_of(&self, node_id: usize) -> usize {
        self.graph.adjacency.get(&node_id).map_or(0, |adjacent| {
            adjacent
                .iter()
                .filter(|(neighbor, _)| self.node_ids.contains(neighbor))
                .count()
        })
    }

    /// Compute the degree distribution.
    pub fn degree_distribution(&self) -> DegreeDistribution {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for node in &self.graph.nodes {
            let deg = self.degree_of(node.id);
            *counts.entry(deg).or_insert(0) += 1;
        }
        DegreeDistribution {
            counts,
            total_nodes: self.graph.node_count(),
        }
    }

    /// Compute the local clustering coefficient for a node.
    pub fn local_clustering_coefficient(&self, node_id: usize) -> f64 {
        let neighbors: Vec<usize> = self.neighbors_of(node_id);
        let k = neighbors.len();
        if k < 2 {
            return 0.0;
        }
        let mut triangles = 0usize;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if self.graph.has_edge(neighbors[i], neighbors[j])
                    || self.graph.has_edge(neighbors[j], neighbors[i])
                {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }

    /// Compute the global (average) clustering coefficient.
    pub fn global_clustering_coefficient(&self) -> f64 {
        let nodes = &self.graph.nodes;
        if nodes.is_empty() {
            return 0.0;
        }
        let sum: f64 = nodes
            .iter()
            .map(|n| self.local_clustering_coefficient(n.id))
            .sum();
        sum / nodes.len() as f64
    }

    /// Compute betweenness centrality with Brandes' algorithm (unweighted,
    /// BFS), normalised by `(n - 1)(n - 2)`.
    ///
    /// Edges are followed in the direction of the adjacency lists, so a
    /// directed edge counts one way and an undirected edge both ways.
    ///
    /// Returns a map from node_id → centrality score.
    pub fn betweenness_centrality(&self) -> HashMap<usize, f64> {
        // Dense indices over the distinct node ids; edges to other ids vanish.
        let mut ids: Vec<usize> = Vec::with_capacity(self.node_ids.len());
        let mut index: HashMap<usize, usize> = HashMap::with_capacity(self.node_ids.len());
        for node in &self.graph.nodes {
            index.entry(node.id).or_insert_with(|| {
                ids.push(node.id);
                ids.len() - 1
            });
        }
        let n = ids.len();
        let successors: Vec<Vec<usize>> = ids
            .iter()
            .map(|id| {
                self.graph
                    .adjacency
                    .get(id)
                    .map(|adjacent| {
                        adjacent
                            .iter()
                            .filter_map(|(neighbor, _)| index.get(neighbor).copied())
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();

        let mut centrality = vec![0.0_f64; n];
        let mut dist = vec![-1_i64; n];
        let mut sigma = vec![0.0_f64; n]; // number of shortest paths
        let mut delta = vec![0.0_f64; n];
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut stack: Vec<usize> = Vec::with_capacity(n);

        for src in 0..n {
            dist.fill(-1);
            sigma.fill(0.0);
            delta.fill(0.0);
            pred.iter_mut().for_each(Vec::clear);
            stack.clear();
            dist[src] = 0;
            sigma[src] = 1.0;
            queue.push_back(src);

            // BFS to compute shortest-path counts
            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for &w in &successors[v] {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        pred[w].push(v);
                    }
                }
            }

            // Back-propagation
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    let coeff = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    delta[v] += coeff;
                }
                if w != src {
                    centrality[w] += delta[w];
                }
            }
        }

        // Normalize
        let norm = if n > 2 {
            1.0 / ((n - 1) * (n - 2)) as f64
        } else {
            1.0
        };
        ids.into_iter()
            .zip(centrality)
            .map(|(id, score)| (id, score * norm))
            .collect()
    }

    /// Compute PageRank with damping factor `d` and `max_iter` iterations.
    pub fn pagerank(&self, d: f64, max_iter: usize, tol: f64) -> HashMap<usize, f64> {
        let ids: Vec<usize> = self.graph.nodes.iter().map(|n| n.id).collect();
        let n = ids.len();
        if n == 0 {
            return HashMap::new();
        }
        let init = 1.0 / n as f64;
        let mut pr: HashMap<usize, f64> = ids.iter().map(|&id| (id, init)).collect();

        for _iter in 0..max_iter {
            let mut new_pr: HashMap<usize, f64> =
                ids.iter().map(|&id| (id, (1.0 - d) / n as f64)).collect();
            for &v in &ids {
                let out_deg = self.degree_of(v) as f64;
                if out_deg > 0.0 {
                    let share = d * pr[&v] / out_deg;
                    for u in self.neighbors_of(v) {
                        *new_pr.entry(u).or_insert(0.0) += share;
                    }
                } else {
                    // Dangling node: distribute to all
                    let share = d * pr[&v] / n as f64;
                    for &u in &ids {
                        *new_pr.entry(u).or_insert(0.0) += share;
                    }
                }
            }
            // Check convergence
            let diff: f64 = ids.iter().map(|id| (new_pr[id] - pr[id]).abs()).sum();
            pr = new_pr;
            if diff < tol {
                break;
            }
        }
        pr
    }

    /// Network density: the fraction of the `n * (n - 1)` possible arcs
    /// between distinct nodes that are present.
    ///
    /// An undirected edge accounts for both of its arcs and a directed edge
    /// for one, so a complete undirected graph has density 1 and a directed
    /// 3-cycle has density 1/2.  Parallel edges count once; self-loops and
    /// edges with an end point that is not a node are ignored.  Graphs with
    /// fewer than two nodes have density 0.
    pub fn density(&self) -> f64 {
        let n = self.node_ids.len();
        if n < 2 {
            return 0.0;
        }
        let mut arcs: HashSet<(usize, usize)> = HashSet::new();
        for edge in &self.graph.edges {
            let (source, target) = (edge.source, edge.target);
            if source == target
                || !self.node_ids.contains(&source)
                || !self.node_ids.contains(&target)
            {
                continue;
            }
            arcs.insert((source, target));
            if !edge.directed {
                arcs.insert((target, source));
            }
        }
        arcs.len() as f64 / (n as f64 * (n - 1) as f64)
    }

    /// Average shortest path length (BFS, unweighted). Returns None if graph is disconnected.
    pub fn average_shortest_path_length(&self) -> Option<f64> {
        let ids: Vec<usize> = self.graph.nodes.iter().map(|n| n.id).collect();
        let n = ids.len();
        if n == 0 {
            return None;
        }
        let mut total = 0u64;
        let mut count = 0u64;
        for &src in &ids {
            let mut dist: HashMap<usize, usize> = HashMap::new();
            let mut queue = VecDeque::new();
            dist.insert(src, 0);
            queue.push_back(src);
            while let Some(v) = queue.pop_front() {
                for u in self.neighbors_of(v) {
                    if !dist.contains_key(&u) {
                        dist.insert(u, dist[&v] + 1);
                        queue.push_back(u);
                    }
                }
            }
            if dist.len() < n {
                return None; // disconnected
            }
            for (&_v, &d) in &dist {
                total += d as u64;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(total as f64 / count as f64)
        }
    }
}

// ---------------------------------------------------------------------------
// CommunityDetection
// ---------------------------------------------------------------------------

/// Result of community detection.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Map from node_id → community_id.
    pub membership: HashMap<usize, usize>,
    /// Modularity score.
    pub modularity: f64,
}

impl CommunityResult {
    /// Number of distinct communities.
    pub fn num_communities(&self) -> usize {
        let set: HashSet<usize> = self.membership.values().copied().collect();
        set.len()
    }

    /// Return members of community `c`.
    pub fn community_members(&self, c: usize) -> Vec<usize> {
        self.membership
            .iter()
            .filter(|(_, v)| **v == c)
            .map(|(k, _)| *k)
            .collect()
    }
}

/// Community detection algorithms.
///
/// The algorithms work on the undirected, weighted graph underlying the
/// [`NetworkGraph`]:
///
/// - An edge counts with its weight for both of its end points, whether or not
///   it is marked `directed`; parallel edges add up and a self-loop of weight
///   `w` adds `2w` to the strength of its node.
/// - Edges whose end points are not nodes of the graph, and edges whose weight
///   is not a finite positive number, are ignored.
/// - With `k_i` the strength (weighted degree) of node `i` and `m` the total
///   edge weight (`2m = sum_i k_i`), the modularity of a partition is
///   `Q = sum_c [L_c / m - (D_c / 2m)^2]`, where `L_c` is the weight of the
///   edges inside community `c` and `D_c` the summed strength of its nodes.
#[derive(Debug, Clone)]
pub struct CommunityDetection {
    graph: NetworkGraph,
}

impl CommunityDetection {
    /// Create from a graph.
    pub fn new(graph: NetworkGraph) -> Self {
        Self { graph }
    }

    /// Compute the weighted modularity `Q` of `membership` in `O(n + m)`.
    ///
    /// A node that `membership` does not mention forms a community of its own.
    /// A graph without edge weight has `Q = 0`.  See the type-level
    /// documentation for how edges are interpreted.
    pub fn modularity(&self, membership: &HashMap<usize, usize>) -> f64 {
        community::modularity(&self.graph, membership)
    }

    /// Louvain modularity maximisation.
    ///
    /// Repeats two phases until modularity stops improving: every node moves
    /// to the neighbouring community with the largest modularity gain (an
    /// `O(degree)` update instead of a fresh modularity evaluation), then the
    /// communities are collapsed into single nodes of a coarser graph.
    /// Deterministic: nodes are visited in insertion order and ties keep the
    /// current community.  Communities are numbered `0..` in order of first
    /// appearance among the nodes.
    pub fn greedy_modularity(&self) -> CommunityResult {
        community::greedy_modularity(&self.graph)
    }

    /// Label propagation community detection.
    ///
    /// Every node starts with a label of its own.  In each sweep the nodes
    /// are visited in ascending id order and each takes the label that carries
    /// the largest total edge weight among its neighbours (labels already
    /// updated in the same sweep count); ties go to the smallest label, which
    /// is that of the smallest id.  It stops after `max_iter` sweeps or as
    /// soon as a sweep changes nothing.  Deterministic.  Communities are
    /// numbered `0..` in order of first appearance among the nodes.
    ///
    /// Label propagation is quick but crude: a label can flood weakly
    /// separated communities.  [`greedy_modularity`](Self::greedy_modularity)
    /// optimises the modularity directly.
    pub fn label_propagation(&self, max_iter: usize) -> CommunityResult {
        community::label_propagation(&self.graph, max_iter)
    }

    /// Spectral clustering into `k` communities by recursive bisection.
    ///
    /// The Fiedler vector of the weighted graph Laplacian `L = D - A` (the
    /// eigenvector of its second-smallest eigenvalue) orders the nodes along
    /// the sparsest cut of the graph.  It is computed by restarted Lanczos
    /// iteration with the constant vector (eigenvalue `0`) deflated, which
    /// converges far faster than a power iteration on `c I - L` when the graph
    /// has several communities.  The nodes are split at the prefix of that
    /// order with the lowest conductance (the sign change of the Fiedler
    /// vector is one of the candidates), and single nodes are then moved to
    /// the other side while that lowers the conductance.  For `k > 2` the
    /// largest community is bisected again, using the Fiedler vector of the
    /// subgraph it induces, until `k` communities exist.
    ///
    /// `k >= n` yields one community per node and `k == 0` (or an empty
    /// graph) an empty result.  Directed edges are treated as undirected, and
    /// a graph without edges is halved by node index.  Deterministic.
    pub fn spectral_clustering(&self, k: usize) -> CommunityResult {
        community::spectral_clustering(&self.graph, k)
    }
}

// ---------------------------------------------------------------------------
// NetworkAnimation — time-evolving graph
// ---------------------------------------------------------------------------

/// An event in the network timeline.
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// A node appears at the given time.
    NodeAppear {
        /// Index of the node that appears.
        node_id: usize,
        /// Time at which the node appears.
        time: f64,
    },
    /// A node disappears at the given time.
    NodeDisappear {
        /// Index of the node that disappears.
        node_id: usize,
        /// Time at which the node disappears.
        time: f64,
    },
    /// An edge appears between source and target at the given time.
    EdgeAppear {
        /// Source node index.
        source: usize,
        /// Target node index.
        target: usize,
        /// Time at which the edge appears.
        time: f64,
    },
    /// An edge disappears at the given time.
    EdgeDisappear {
        /// Source node index.
        source: usize,
        /// Target node index.
        target: usize,
        /// Time at which the edge disappears.
        time: f64,
    },
    /// A node moves to a new position over a duration.
    NodeMove {
        /// Index of the node to move.
        node_id: usize,
        /// World-space 2-D target position.
        target_position: [f64; 2],
        /// Time at which the movement begins.
        start_time: f64,
        /// Duration of the movement animation.
        duration: f64,
    },
}

impl NetworkEvent {
    /// Time at which the event takes effect: its `time`, or the `start_time`
    /// of a [`NodeMove`](NetworkEvent::NodeMove).
    pub fn time(&self) -> f64 {
        match self {
            NetworkEvent::NodeAppear { time, .. }
            | NetworkEvent::NodeDisappear { time, .. }
            | NetworkEvent::EdgeAppear { time, .. }
            | NetworkEvent::EdgeDisappear { time, .. } => *time,
            NetworkEvent::NodeMove { start_time, .. } => *start_time,
        }
    }
}

/// Snapshot of the network at a single frame.
#[derive(Debug, Clone)]
pub struct NetworkFrame {
    /// Time of this frame.
    pub time: f64,
    /// Active node ids.
    pub active_nodes: HashSet<usize>,
    /// Active edges as (source, target) pairs.
    pub active_edges: Vec<(usize, usize)>,
    /// Node positions at this frame.
    pub node_positions: HashMap<usize, [f64; 2]>,
}

/// Network animation: time-evolving graph with interpolation.
#[derive(Debug, Clone)]
pub struct NetworkAnimation {
    /// Base graph (contains all possible nodes/edges).
    pub base_graph: NetworkGraph,
    /// Timeline events.
    pub events: Vec<NetworkEvent>,
    /// Total animation duration.
    pub total_duration: f64,
    /// Frames per second for sampling.
    pub fps: f64,
}

impl NetworkAnimation {
    /// Create a new animation from a base graph.
    pub fn new(base_graph: NetworkGraph, total_duration: f64, fps: f64) -> Self {
        Self {
            base_graph,
            events: Vec::new(),
            total_duration,
            fps,
        }
    }

    /// Add an event.
    pub fn add_event(&mut self, event: NetworkEvent) {
        self.events.push(event);
    }

    /// Compute the network frame at time `t`.
    ///
    /// The events that have happened by `t` (their time is at most `t`) are
    /// applied in order of time, whatever the order in which they were added;
    /// events at the same time are applied in the order they were added.
    ///
    /// * A node is active from its latest `NodeAppear` on until a later
    ///   `NodeDisappear`; an edge likewise with `EdgeAppear` and
    ///   `EdgeDisappear`, which match the exact `(source, target)` pair.  Every
    ///   active edge is listed once, in the order of its (re-)appearance.
    /// * A `NodeMove` interpolates linearly from the position the node has
    ///   when the move is applied (the base position, or where earlier moves
    ///   left it) to its target, so consecutive moves are continuous.  A move
    ///   of no (or negative) duration jumps.
    pub fn frame_at(&self, t: f64) -> NetworkFrame {
        self.frame_from_timeline(&self.timeline(), t)
    }

    /// The events ordered by time (stable, so ties keep their insertion order).
    fn timeline(&self) -> Vec<&NetworkEvent> {
        let mut timeline: Vec<&NetworkEvent> = self.events.iter().collect();
        timeline.sort_by(|a, b| a.time().total_cmp(&b.time()));
        timeline
    }

    /// The frame at `t` for a `timeline` as returned by [`Self::timeline`].
    fn frame_from_timeline(&self, timeline: &[&NetworkEvent], t: f64) -> NetworkFrame {
        let mut active_nodes: HashSet<usize> = HashSet::new();
        // Active edges with the sequence number of their appearance.
        let mut active_edges: HashMap<(usize, usize), usize> = HashMap::new();
        let mut appearances = 0_usize;
        let mut node_positions: HashMap<usize, [f64; 2]> = HashMap::new();

        // Initialize from base graph
        for node in &self.base_graph.nodes {
            node_positions.insert(node.id, node.position);
        }

        // Process events up to time t
        for event in timeline {
            // Written so that an event at a time that is not a number never
            // happens.
            let has_happened = t >= event.time();
            if !has_happened {
                continue;
            }
            match event {
                NetworkEvent::NodeAppear { node_id, .. } => {
                    active_nodes.insert(*node_id);
                }
                NetworkEvent::NodeDisappear { node_id, .. } => {
                    active_nodes.remove(node_id);
                }
                NetworkEvent::EdgeAppear { source, target, .. } => {
                    active_edges.entry((*source, *target)).or_insert_with(|| {
                        appearances += 1;
                        appearances
                    });
                }
                NetworkEvent::EdgeDisappear { source, target, .. } => {
                    active_edges.remove(&(*source, *target));
                }
                NetworkEvent::NodeMove {
                    node_id,
                    target_position,
                    start_time,
                    duration,
                } => {
                    let alpha = if *duration > 0.0 {
                        ((t - start_time) / duration).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    if let Some(pos) = node_positions.get_mut(node_id) {
                        let start = *pos;
                        pos[0] = start[0] * (1.0 - alpha) + target_position[0] * alpha;
                        pos[1] = start[1] * (1.0 - alpha) + target_position[1] * alpha;
                    }
                }
            }
        }

        let mut edges: Vec<((usize, usize), usize)> = active_edges.into_iter().collect();
        edges.sort_unstable_by_key(|&(_, appearance)| appearance);
        NetworkFrame {
            time: t,
            active_nodes,
            active_edges: edges.into_iter().map(|(edge, _)| edge).collect(),
            node_positions,
        }
    }

    /// Sample the animation into `n_frames` evenly-spaced frames.
    pub fn sample_frames(&self, n_frames: usize) -> Vec<NetworkFrame> {
        if n_frames == 0 {
            return Vec::new();
        }
        let timeline = self.timeline();
        (0..n_frames)
            .map(|i| {
                let t = self.total_duration * i as f64 / (n_frames - 1).max(1) as f64;
                self.frame_from_timeline(&timeline, t)
            })
            .collect()
    }

    /// Interpolate node position between two times using linear interpolation.
    pub fn interpolate_position(&self, node_id: usize, t0: f64, t1: f64, alpha: f64) -> [f64; 2] {
        let timeline = self.timeline();
        let f0 = self.frame_from_timeline(&timeline, t0);
        let f1 = self.frame_from_timeline(&timeline, t1);
        let p0 = f0
            .node_positions
            .get(&node_id)
            .copied()
            .unwrap_or([0.0, 0.0]);
        let p1 = f1
            .node_positions
            .get(&node_id)
            .copied()
            .unwrap_or([0.0, 0.0]);
        [
            p0[0] * (1.0 - alpha) + p1[0] * alpha,
            p0[1] * (1.0 - alpha) + p1[1] * alpha,
        ]
    }

    /// Return how many events are in the timeline.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle() -> NetworkGraph {
        let mut g = NetworkGraph::new();
        g.add_node(NetworkNode::new(0, "A"));
        g.add_node(NetworkNode::new(1, "B"));
        g.add_node(NetworkNode::new(2, "C"));
        g.add_edge(NetworkEdge::undirected(0, 1));
        g.add_edge(NetworkEdge::undirected(1, 2));
        g.add_edge(NetworkEdge::undirected(0, 2));
        g
    }

    fn make_chain(n: usize) -> NetworkGraph {
        let mut g = NetworkGraph::new();
        for i in 0..n {
            g.add_node(NetworkNode::new(i, format!("N{}", i)));
        }
        for i in 0..n - 1 {
            g.add_edge(NetworkEdge::undirected(i, i + 1));
        }
        g
    }

    // --- NetworkGraph ---

    #[test]
    fn test_graph_node_count() {
        let g = make_triangle();
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_graph_edge_count() {
        let g = make_triangle();
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_graph_degree() {
        let g = make_triangle();
        assert_eq!(g.degree(0), 2);
    }

    #[test]
    fn test_graph_has_edge() {
        let g = make_triangle();
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(0, 99));
    }

    #[test]
    fn test_graph_neighbors() {
        let g = make_triangle();
        let mut nb = g.neighbors(0);
        nb.sort_unstable();
        assert_eq!(nb, vec![1, 2]);
    }

    #[test]
    fn test_graph_bfs() {
        let g = make_chain(5);
        let visited = g.bfs(0);
        assert_eq!(visited.len(), 5);
        assert_eq!(visited[0], 0);
    }

    #[test]
    fn test_graph_dfs() {
        let g = make_chain(5);
        let visited = g.dfs(0);
        assert_eq!(visited.len(), 5);
    }

    #[test]
    fn test_graph_adjacency_matrix() {
        let g = make_triangle();
        let mat = g.adjacency_matrix();
        assert_eq!(mat.len(), 3);
        // Symmetric
        for (i, row) in mat.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!((val - mat[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_graph_node_mut() {
        let mut g = make_triangle();
        g.node_mut(0).unwrap().position = [5.0, 5.0];
        assert_eq!(g.node(0).unwrap().position, [5.0, 5.0]);
    }

    #[test]
    fn test_graph_directed_edge() {
        let mut g = NetworkGraph::new();
        g.add_node(NetworkNode::new(0, "A"));
        g.add_node(NetworkNode::new(1, "B"));
        g.add_edge(NetworkEdge::directed(0, 1, 2.0));
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0));
    }

    #[test]
    fn test_graph_default() {
        let g = NetworkGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_node_builder() {
        let n = NetworkNode::new(5, "test")
            .with_position(1.0, 2.0)
            .with_size(3.0)
            .with_color(1.0, 0.0, 0.0, 1.0);
        assert_eq!(n.position, [1.0, 2.0]);
        assert_eq!(n.size, 3.0);
    }

    #[test]
    fn test_edge_label() {
        let e = NetworkEdge::undirected(0, 1).with_label("link");
        assert_eq!(e.label.unwrap(), "link");
    }

    // --- ForceDirectedLayout ---

    #[test]
    fn test_force_directed_applies() {
        let mut g = make_triangle();
        let layout = ForceDirectedLayout::new();
        layout.apply(&mut g);
        // Positions should have moved
        let not_all_zero = g.nodes.iter().any(|n| n.position != [0.0, 0.0]);
        assert!(not_all_zero);
    }

    #[test]
    fn test_force_directed_empty() {
        let mut g = NetworkGraph::new();
        let layout = ForceDirectedLayout::new();
        layout.apply(&mut g); // Should not panic
    }

    #[test]
    fn test_force_directed_single_node() {
        let mut g = NetworkGraph::new();
        g.add_node(NetworkNode::new(0, "A"));
        let layout = ForceDirectedLayout::new();
        layout.apply(&mut g);
    }

    #[test]
    fn test_force_directed_config() {
        let config = ForceDirectedConfig {
            iterations: 10,
            optimal_distance: 30.0,
            ..ForceDirectedConfig::default()
        };
        let mut g = make_chain(4);
        let layout = ForceDirectedLayout::with_config(config);
        layout.apply(&mut g);
    }

    #[test]
    fn test_force_directed_default() {
        let layout = ForceDirectedLayout::default();
        assert_eq!(layout.config.iterations, 200);
    }

    #[test]
    fn test_barnes_hut_repulsion() {
        let g = make_triangle();
        let layout = ForceDirectedLayout::new();
        let disp = layout.barnes_hut_repulsion(&g);
        assert_eq!(disp.len(), 3);
    }

    // --- HierarchicalLayout ---

    #[test]
    fn test_hierarchical_applies() {
        let mut g = NetworkGraph::new();
        g.add_node(NetworkNode::new(0, "root"));
        g.add_node(NetworkNode::new(1, "child1"));
        g.add_node(NetworkNode::new(2, "child2"));
        g.add_edge(NetworkEdge::directed(0, 1, 1.0));
        g.add_edge(NetworkEdge::directed(0, 2, 1.0));
        let layout = HierarchicalLayout::new();
        layout.apply(&mut g);
        // Root should be at a different y than children
        let root_y = g.node(0).unwrap().position[1];
        let child_y = g.node(1).unwrap().position[1];
        assert!((root_y - child_y).abs() > 1.0);
    }

    #[test]
    fn test_hierarchical_empty() {
        let mut g = NetworkGraph::new();
        let layout = HierarchicalLayout::new();
        layout.apply(&mut g);
    }

    #[test]
    fn test_hierarchical_assign_layers() {
        let g = make_chain(4);
        let layout = HierarchicalLayout::new();
        let layers = layout.assign_layers(&g);
        assert_eq!(layers.len(), 4);
    }

    #[test]
    fn test_hierarchical_minimize_crossings() {
        let g = make_chain(3);
        let layout = HierarchicalLayout::new();
        let layers = layout.assign_layers(&g);
        let ordered = layout.minimize_crossings(&g, &layers);
        assert!(!ordered.is_empty());
    }

    #[test]
    fn test_hierarchical_default() {
        let layout = HierarchicalLayout::default();
        assert_eq!(layout.config.layer_separation, 100.0);
    }

    // --- NetworkMetrics ---

    #[test]
    fn test_metrics_degree_distribution() {
        let g = make_triangle();
        let m = NetworkMetrics::new(g);
        let dd = m.degree_distribution();
        assert_eq!(dd.total_nodes, 3);
        assert!((dd.mean_degree() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_clustering_triangle() {
        let g = make_triangle();
        let m = NetworkMetrics::new(g);
        let cc = m.local_clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_global_clustering() {
        let g = make_triangle();
        let m = NetworkMetrics::new(g);
        let gc = m.global_clustering_coefficient();
        assert!((gc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_betweenness() {
        let g = make_chain(5);
        let m = NetworkMetrics::new(g);
        let bc = m.betweenness_centrality();
        assert_eq!(bc.len(), 5);
        // Node 2 (middle) should have highest centrality
        assert!(bc[&2] >= bc[&0]);
    }

    #[test]
    fn test_metrics_pagerank_sums_to_one() {
        let g = make_triangle();
        let m = NetworkMetrics::new(g);
        let pr = m.pagerank(0.85, 100, 1e-8);
        let total: f64 = pr.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_density() {
        let g = make_triangle();
        let m = NetworkMetrics::new(g);
        // Each of the 3 undirected edges is two arcs: 6 of 3 * 2 possible.
        assert!((m.density() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_metrics_density_directed_cycle() {
        let mut g = NetworkGraph::new();
        for i in 0..3 {
            g.add_node(NetworkNode::new(i, format!("N{i}")));
        }
        g.add_edge(NetworkEdge::directed(0, 1, 1.0));
        g.add_edge(NetworkEdge::directed(1, 2, 1.0));
        g.add_edge(NetworkEdge::directed(2, 0, 1.0));
        let m = NetworkMetrics::new(g);
        // 3 arcs of 3 * 2 possible.
        assert!((m.density() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_metrics_avg_path_chain() {
        let g = make_chain(3);
        let m = NetworkMetrics::new(g);
        let apl = m.average_shortest_path_length();
        assert!(apl.is_some());
    }

    #[test]
    fn test_degree_distribution_pmf() {
        let g = make_triangle();
        let m = NetworkMetrics::new(g);
        let dd = m.degree_distribution();
        let p = dd.pmf(2);
        assert!((p - 1.0).abs() < 1e-10);
    }

    // --- CommunityDetection ---

    #[test]
    fn test_community_label_propagation() {
        let g = make_triangle();
        let cd = CommunityDetection::new(g);
        let result = cd.label_propagation(20);
        assert_eq!(result.membership.len(), 3);
    }

    #[test]
    fn test_community_num_communities() {
        let g = make_chain(6);
        let cd = CommunityDetection::new(g);
        let result = cd.label_propagation(10);
        assert!(result.num_communities() >= 1);
    }

    #[test]
    fn test_community_greedy() {
        let g = make_triangle();
        let cd = CommunityDetection::new(g);
        let result = cd.greedy_modularity();
        assert_eq!(result.membership.len(), 3);
        // A triangle has no community structure: Louvain keeps it whole, and
        // one community holding every edge has Q = 1 - (2m / 2m)^2 = 0.
        assert_eq!(result.num_communities(), 1);
        assert!(result.modularity.abs() < 1e-12, "Q = {}", result.modularity);
    }

    #[test]
    fn test_community_spectral_k2() {
        let g = make_chain(4);
        let cd = CommunityDetection::new(g);
        let result = cd.spectral_clustering(2);
        assert_eq!(result.membership.len(), 4);
        // The Fiedler vector of a path is antisymmetric, so the sparsest cut
        // is the middle edge.
        assert_eq!(result.num_communities(), 2);
        assert_eq!(result.membership[&0], result.membership[&1]);
        assert_eq!(result.membership[&2], result.membership[&3]);
        assert_ne!(result.membership[&1], result.membership[&2]);
        // m = 3, strengths 1, 2, 2, 1: Q = 2 * (1 / 3 - (3 / 6)^2) = 1 / 6.
        assert!(
            (result.modularity - 1.0 / 6.0).abs() < 1e-12,
            "Q = {}",
            result.modularity
        );
    }

    #[test]
    fn test_community_spectral_k_geq_n() {
        let g = make_triangle();
        let cd = CommunityDetection::new(g);
        let result = cd.spectral_clustering(10);
        assert_eq!(result.membership.len(), 3);
    }

    #[test]
    fn test_community_members() {
        let g = make_triangle();
        let cd = CommunityDetection::new(g);
        let result = cd.label_propagation(10);
        let comm = result.membership[&0];
        let members = result.community_members(comm);
        assert!(members.contains(&0));
    }

    #[test]
    fn test_community_modularity_range() {
        let g = make_triangle();
        let cd = CommunityDetection::new(g);
        let result = cd.greedy_modularity();
        // Modularity is bounded [-0.5, 1.0]
        assert!(result.modularity >= -0.5 && result.modularity <= 1.0);
    }

    // --- NetworkAnimation ---

    #[test]
    fn test_animation_frame_at_start() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::NodeAppear {
            node_id: 0,
            time: 0.0,
        });
        anim.add_event(NetworkEvent::NodeAppear {
            node_id: 1,
            time: 1.0,
        });
        let f = anim.frame_at(0.5);
        assert!(f.active_nodes.contains(&0));
        assert!(!f.active_nodes.contains(&1));
    }

    #[test]
    fn test_animation_edge_appear() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::EdgeAppear {
            source: 0,
            target: 1,
            time: 2.0,
        });
        let f = anim.frame_at(3.0);
        assert!(f.active_edges.contains(&(0, 1)));
    }

    #[test]
    fn test_animation_node_disappear() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::NodeAppear {
            node_id: 0,
            time: 0.0,
        });
        anim.add_event(NetworkEvent::NodeDisappear {
            node_id: 0,
            time: 5.0,
        });
        let f0 = anim.frame_at(3.0);
        let f1 = anim.frame_at(7.0);
        assert!(f0.active_nodes.contains(&0));
        assert!(!f1.active_nodes.contains(&0));
    }

    #[test]
    fn test_animation_edge_disappear() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::EdgeAppear {
            source: 0,
            target: 1,
            time: 0.0,
        });
        anim.add_event(NetworkEvent::EdgeDisappear {
            source: 0,
            target: 1,
            time: 5.0,
        });
        let f0 = anim.frame_at(3.0);
        let f1 = anim.frame_at(7.0);
        assert!(f0.active_edges.contains(&(0, 1)));
        assert!(!f1.active_edges.contains(&(0, 1)));
    }

    #[test]
    fn test_animation_node_move() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::NodeMove {
            node_id: 0,
            target_position: [100.0, 100.0],
            start_time: 0.0,
            duration: 10.0,
        });
        let f = anim.frame_at(10.0);
        let pos = f.node_positions[&0];
        assert!((pos[0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_animation_sample_frames() {
        let g = make_triangle();
        let anim = NetworkAnimation::new(g, 10.0, 30.0);
        let frames = anim.sample_frames(5);
        assert_eq!(frames.len(), 5);
        assert!((frames[0].time).abs() < 1e-10);
    }

    #[test]
    fn test_animation_interpolate_position() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::NodeMove {
            node_id: 0,
            target_position: [100.0, 0.0],
            start_time: 0.0,
            duration: 10.0,
        });
        let p = anim.interpolate_position(0, 0.0, 10.0, 0.5);
        assert!(p[0].is_finite());
    }

    #[test]
    fn test_animation_event_count() {
        let g = make_triangle();
        let mut anim = NetworkAnimation::new(g, 10.0, 30.0);
        anim.add_event(NetworkEvent::NodeAppear {
            node_id: 0,
            time: 0.0,
        });
        assert_eq!(anim.event_count(), 1);
    }

    #[test]
    fn test_animation_sample_empty() {
        let g = make_triangle();
        let anim = NetworkAnimation::new(g, 10.0, 30.0);
        let frames = anim.sample_frames(0);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_force_directed_chain() {
        let mut g = make_chain(6);
        let config = ForceDirectedConfig {
            iterations: 50,
            ..Default::default()
        };
        let layout = ForceDirectedLayout::with_config(config);
        layout.apply(&mut g);
        // All positions should be finite
        for n in &g.nodes {
            assert!(n.position[0].is_finite());
            assert!(n.position[1].is_finite());
        }
    }
}
