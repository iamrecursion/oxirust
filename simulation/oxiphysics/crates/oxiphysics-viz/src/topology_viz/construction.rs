// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Combinatorial construction algorithms backing [`super::ReebGraph`],
//! [`super::ContourTree`] and [`super::MorseSmaleComplex`].
//!
//! All algorithms use one strict total order on the vertices — the scalar
//! value compared with [`f64::total_cmp`], ties broken by vertex index
//! (simulation of simplicity).  This makes every construction deterministic
//! even on plateaus or with NaN values.
//!
//! - [`reeb_graph_from_triangles`]: online Reeb graph by triangle gluing
//!   (Pascucci, Scorzelli, Bremer, Mascarenhas, "Robust on-line computation
//!   of Reeb graphs: simplicity and speed", SIGGRAPH 2007).
//! - [`contour_tree_from_graph`]: join/split tree merge (Carr, Snoeyink,
//!   Axen, "Computing contour trees in all dimensions", 2003).
//! - [`steepest_terminals`]: memoised steepest ascent / descent.

use std::collections::{HashMap, HashSet, VecDeque};

use super::{
    ContourTree, ContourTreeArc, ContourTreeNode, CriticalPointType, ReebGraph, uf_find, uf_union,
};

/// Vertex order by `(value, index)` and the inverse permutation (`rank`).
///
/// Returns `(order, rank)` with `order[rank[v]] == v`.
pub(super) fn simulation_of_simplicity_order(values: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));
    let mut rank = vec![0usize; values.len()];
    for (r, &v) in order.iter().enumerate() {
        rank[v] = r;
    }
    (order, rank)
}

/// Classify a node of an augmented (vertex-level) graph from its degrees.
///
/// A node with no arcs at all (isolated vertex) is reported as a minimum.
fn classify_by_degree(down: usize, up: usize) -> CriticalPointType {
    if down == 0 {
        CriticalPointType::Minimum
    } else if up == 0 {
        CriticalPointType::Maximum
    } else if down == 1 && up == 1 {
        CriticalPointType::Regular
    } else {
        CriticalPointType::Saddle
    }
}

// ---------------------------------------------------------------------------
// Reeb graph by triangle gluing
// ---------------------------------------------------------------------------

/// Arc of the augmented Reeb graph under construction.
#[derive(Debug, Clone)]
struct GluingArc {
    /// Lower endpoint (mesh vertex).
    bottom: usize,
    /// Upper endpoint (mesh vertex).
    top: usize,
    /// Mesh edges whose monotone image path runs through this arc.
    labels: HashSet<usize>,
}

/// Augmented Reeb graph whose nodes are all mesh vertices.
///
/// Initially it is the 1-skeleton of the mesh, every edge oriented from its
/// lower to its upper endpoint.  Gluing a triangle identifies the image of
/// its long edge with the image of its two short edges, which is exactly the
/// level-set identification the triangle induces.  After all triangles are
/// glued the graph is the augmented Reeb graph.
struct ReebGluing {
    rank: Vec<usize>,
    arcs: Vec<GluingArc>,
    /// Arcs leaving each node upwards (alive arcs only).
    up: Vec<Vec<usize>>,
    /// Arcs arriving at each node from below (alive arcs only).
    down: Vec<Vec<usize>>,
}

impl ReebGluing {
    fn new(rank: Vec<usize>) -> Self {
        let n = rank.len();
        Self {
            rank,
            arcs: Vec::new(),
            up: vec![Vec::new(); n],
            down: vec![Vec::new(); n],
        }
    }

    /// Order the two endpoints of an edge by rank.
    fn oriented(&self, a: usize, b: usize) -> (usize, usize) {
        if self.rank[a] < self.rank[b] {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Add the arc for mesh edge `label` between vertices `a` and `b`.
    fn add_edge_arc(&mut self, a: usize, b: usize, label: usize) {
        let (bottom, top) = self.oriented(a, b);
        let idx = self.arcs.len();
        let mut labels = HashSet::new();
        labels.insert(label);
        self.arcs.push(GluingArc {
            bottom,
            top,
            labels,
        });
        self.up[bottom].push(idx);
        self.down[top].push(idx);
    }

    /// The upward arc at `node` on the image path of mesh edge `label`.
    fn arc_with_label(&self, node: usize, label: usize) -> Option<usize> {
        self.up[node]
            .iter()
            .copied()
            .find(|&a| self.arcs[a].labels.contains(&label))
    }

    /// Copy the labels of arc `from` into arc `into`.
    fn copy_labels(&mut self, into: usize, from: usize) {
        let extra: Vec<usize> = self.arcs[from].labels.iter().copied().collect();
        self.arcs[into].labels.extend(extra);
    }

    /// Merge arc `kill` into arc `keep` (both span the same node pair).
    fn merge_parallel(&mut self, keep: usize, kill: usize) {
        let moved = std::mem::take(&mut self.arcs[kill].labels);
        self.arcs[keep].labels.extend(moved);
        let (bottom, top) = (self.arcs[kill].bottom, self.arcs[kill].top);
        self.up[bottom].retain(|&a| a != kill);
        self.down[top].retain(|&a| a != kill);
    }

    /// Move the lower endpoint of `arc` up to `new_bottom`.
    fn raise_bottom(&mut self, arc: usize, new_bottom: usize) {
        let old = self.arcs[arc].bottom;
        self.up[old].retain(|&a| a != arc);
        self.up[new_bottom].push(arc);
        self.arcs[arc].bottom = new_bottom;
    }

    /// Glue the image path of edge `e02` (from `v0` to `v2`) with the
    /// concatenated image paths of `e01` (`v0`→`v1`) and `e12` (`v1`→`v2`).
    ///
    /// The vertices must satisfy `rank[v0] < rank[v1] < rank[v2]`.
    fn glue_triangle(&mut self, v: [usize; 3], e02: usize, e01: usize, e12: usize) {
        let [v0, v1, v2] = v;
        let mut node = v0;
        while node != v2 {
            let label_b = if self.rank[node] < self.rank[v1] {
                e01
            } else {
                e12
            };
            let (Some(a), Some(b)) = (
                self.arc_with_label(node, e02),
                self.arc_with_label(node, label_b),
            ) else {
                // Unreachable for consistent input; stop instead of looping.
                return;
            };
            let top_a = self.arcs[a].top;
            if a == b {
                node = top_a;
                continue;
            }
            let top_b = self.arcs[b].top;
            if top_a == top_b {
                self.merge_parallel(a, b);
                node = top_a;
            } else if self.rank[top_a] < self.rank[top_b] {
                // `a` is the shorter arc: identify it with the lower part of
                // `b`, which now starts at `top_a`.
                self.copy_labels(a, b);
                self.raise_bottom(b, top_a);
                node = top_a;
            } else {
                self.copy_labels(b, a);
                self.raise_bottom(a, top_b);
                node = top_b;
            }
        }
    }
}

/// Build the (contracted) Reeb graph of a piecewise-linear scalar field on a
/// triangulated 2-complex.
///
/// Triangles referencing a vertex index `>= vertices.len()` are skipped;
/// degenerate triangles (repeated index) only contribute their edge.
pub(super) fn reeb_graph_from_triangles(
    vertices: &[(f64, f64, f64, f64)],
    triangles: &[(usize, usize, usize)],
) -> ReebGraph {
    let n = vertices.len();
    let mut graph = ReebGraph::new();
    if n == 0 {
        return graph;
    }
    let values: Vec<f64> = vertices.iter().map(|v| v.3).collect();
    let (order, rank) = simulation_of_simplicity_order(&values);
    let mut gluing = ReebGluing::new(rank);

    // Unique edges, keyed by (min index, max index), in deterministic order.
    let mut edge_ids: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_id = |gluing: &mut ReebGluing, a: usize, b: usize| -> usize {
        let key = (a.min(b), a.max(b));
        if let Some(&id) = edge_ids.get(&key) {
            return id;
        }
        let id = edge_ids.len();
        edge_ids.insert(key, id);
        gluing.add_edge_arc(a, b, id);
        id
    };

    let mut valid_triangles: Vec<[usize; 3]> = Vec::new();
    for &(a, b, c) in triangles {
        if a >= n || b >= n || c >= n {
            continue;
        }
        if a == b && b == c {
            continue;
        }
        if a == b || b == c || a == c {
            // Degenerate triangle: just the segment between its two
            // distinct vertices.
            let (p, q) = if a == b { (a, c) } else { (a, b) };
            edge_id(&mut gluing, p, q);
            continue;
        }
        let mut tri = [a, b, c];
        tri.sort_by_key(|&v| gluing.rank[v]);
        edge_id(&mut gluing, tri[0], tri[1]);
        edge_id(&mut gluing, tri[1], tri[2]);
        edge_id(&mut gluing, tri[0], tri[2]);
        valid_triangles.push(tri);
    }

    for tri in valid_triangles {
        let key = |p: usize, q: usize| (p.min(q), p.max(q));
        let (Some(&e01), Some(&e12), Some(&e02)) = (
            edge_ids.get(&key(tri[0], tri[1])),
            edge_ids.get(&key(tri[1], tri[2])),
            edge_ids.get(&key(tri[0], tri[2])),
        ) else {
            continue;
        };
        gluing.glue_triangle(tri, e02, e01, e12);
    }

    // Contract regular nodes (one arc down, one arc up).
    let types: Vec<CriticalPointType> = (0..n)
        .map(|v| classify_by_degree(gluing.down[v].len(), gluing.up[v].len()))
        .collect();
    let mut node_of: Vec<Option<usize>> = vec![None; n];
    for &v in &order {
        if types[v] != CriticalPointType::Regular {
            let (x, y, z, f) = vertices[v];
            node_of[v] = Some(graph.add_node_with_position(f, types[v], [x, y, z]));
        }
    }
    for &v in &order {
        let Some(src) = node_of[v] else {
            continue;
        };
        for &arc in &gluing.up[v] {
            let mut t = gluing.arcs[arc].top;
            // Each step strictly increases the rank, so this terminates.
            while node_of[t].is_none() {
                match gluing.up[t].first() {
                    Some(&next) => t = gluing.arcs[next].top,
                    None => break,
                }
            }
            if let Some(dst) = node_of[t] {
                graph.add_edge(src, dst);
            }
        }
    }
    graph
}

// ---------------------------------------------------------------------------
// Contour tree (Carr–Snoeyink–Axen)
// ---------------------------------------------------------------------------

/// Augmented merge tree of a sweep in `order`.
///
/// Returns `(parent, children)`: `parent[v]` is the vertex processed later
/// that first joins v's component, `children[v]` holds the *head* (last
/// processed vertex) of every component merged at `v`.
fn merge_tree(order: &[usize], nbrs: &[Vec<usize>]) -> (Vec<Option<usize>>, Vec<Vec<usize>>) {
    let n = nbrs.len();
    let mut pos = vec![0usize; n];
    for (i, &v) in order.iter().enumerate() {
        pos[v] = i;
    }
    let mut uf_parent: Vec<usize> = (0..n).collect();
    let mut uf_rank = vec![0u32; n];
    let mut head: Vec<usize> = (0..n).collect();
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];

    for &v in order {
        let mut roots: Vec<usize> = Vec::new();
        for &u in &nbrs[v] {
            if pos[u] < pos[v] {
                let r = uf_find(&mut uf_parent, u);
                if !roots.contains(&r) {
                    roots.push(r);
                }
            }
        }
        for &r in &roots {
            let h = head[r];
            parent[h] = Some(v);
            children[v].push(h);
            uf_union(&mut uf_parent, &mut uf_rank, v, r);
        }
        let root = uf_find(&mut uf_parent, v);
        head[root] = v;
    }
    (parent, children)
}

/// Replace the first occurrence of `old` in `list` with `new`.
fn replace_in(list: &mut [usize], old: usize, new: usize) {
    if let Some(slot) = list.iter_mut().find(|x| **x == old) {
        *slot = new;
    }
}

/// Build the contour tree of a scalar function on a graph.
///
/// Returns an empty tree if `adjacency.len() != values.len()`.  Neighbour
/// indices out of range and self loops are ignored; the adjacency is
/// symmetrised.
pub(super) fn contour_tree_from_graph(values: &[f64], adjacency: &[Vec<usize>]) -> ContourTree {
    let n = values.len();
    if n == 0 || adjacency.len() != n {
        return ContourTree::new();
    }
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (v, list) in adjacency.iter().enumerate() {
        for &u in list {
            if u < n && u != v {
                nbrs[v].push(u);
                nbrs[u].push(v);
            }
        }
    }
    for list in &mut nbrs {
        list.sort_unstable();
        list.dedup();
    }

    let (order, _rank) = simulation_of_simplicity_order(values);
    // Join tree: sweep upwards; parent is higher, children lower.
    let (mut jt_up, mut jt_down) = merge_tree(&order, &nbrs);
    // Split tree: sweep downwards; parent is lower, children higher.
    let rev: Vec<usize> = order.iter().rev().copied().collect();
    let (mut st_down, mut st_up) = merge_tree(&rev, &nbrs);

    let mut ct_up: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut ct_down: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut removed = vec![false; n];

    let is_lower_leaf = |jt_down: &[Vec<usize>], st_up: &[Vec<usize>], v: usize| {
        jt_down[v].is_empty() && st_up[v].len() == 1
    };
    let is_upper_leaf = |jt_down: &[Vec<usize>], st_up: &[Vec<usize>], v: usize| {
        st_up[v].is_empty() && jt_down[v].len() == 1
    };

    let mut queue: VecDeque<usize> = order
        .iter()
        .copied()
        .filter(|&v| is_lower_leaf(&jt_down, &st_up, v) || is_upper_leaf(&jt_down, &st_up, v))
        .collect();

    while let Some(v) = queue.pop_front() {
        if removed[v] {
            continue;
        }
        let mut touched: Vec<usize> = Vec::new();
        if is_lower_leaf(&jt_down, &st_up, v) {
            let (Some(w), Some(&s_up)) = (jt_up[v], st_up[v].first()) else {
                continue;
            };
            ct_up[v].push(w);
            ct_down[w].push(v);
            // Remove v from the join tree (it is a leaf there).
            jt_down[w].retain(|&x| x != v);
            jt_up[v] = None;
            // Splice v out of the split tree.
            let s_down = st_down[v];
            st_down[s_up] = s_down;
            if let Some(d) = s_down {
                replace_in(&mut st_up[d], v, s_up);
                touched.push(d);
            }
            st_up[v].clear();
            st_down[v] = None;
            touched.push(w);
            touched.push(s_up);
        } else if is_upper_leaf(&jt_down, &st_up, v) {
            let (Some(d), Some(&j_down)) = (st_down[v], jt_down[v].first()) else {
                continue;
            };
            ct_up[d].push(v);
            ct_down[v].push(d);
            // Remove v from the split tree (it is a leaf there).
            st_up[d].retain(|&x| x != v);
            st_down[v] = None;
            // Splice v out of the join tree.
            let j_up = jt_up[v];
            jt_up[j_down] = j_up;
            if let Some(u) = j_up {
                replace_in(&mut jt_down[u], v, j_down);
                touched.push(u);
            }
            jt_down[v].clear();
            jt_up[v] = None;
            touched.push(d);
            touched.push(j_down);
        } else {
            continue;
        }
        removed[v] = true;
        for t in touched {
            if !removed[t]
                && (is_lower_leaf(&jt_down, &st_up, t) || is_upper_leaf(&jt_down, &st_up, t))
            {
                queue.push_back(t);
            }
        }
    }

    // Contract regular vertices into arcs.
    let mut ct = ContourTree::new();
    let types: Vec<CriticalPointType> = (0..n)
        .map(|v| classify_by_degree(ct_down[v].len(), ct_up[v].len()))
        .collect();
    let mut node_of: Vec<Option<usize>> = vec![None; n];
    for &v in &order {
        if types[v] != CriticalPointType::Regular {
            node_of[v] = Some(ct.nodes.len());
            ct.nodes.push(ContourTreeNode {
                vertex: v,
                value: values[v],
                critical_type: types[v],
            });
        }
    }
    for &v in &order {
        let Some(lower) = node_of[v] else {
            continue;
        };
        for &first in &ct_up[v] {
            let mut t = first;
            let mut regular_count = 0usize;
            while node_of[t].is_none() {
                match ct_up[t].first() {
                    Some(&next) => {
                        regular_count += 1;
                        t = next;
                    }
                    None => break,
                }
            }
            if let Some(upper) = node_of[t] {
                ct.arcs.push(ContourTreeArc {
                    lower,
                    upper,
                    regular_count,
                });
            }
        }
    }
    ct
}

// ---------------------------------------------------------------------------
// Steepest ascent / descent
// ---------------------------------------------------------------------------

/// Terminal vertex of steepest ascent (`ascending == true`) or descent from
/// every vertex, memoised in O(V + E).
///
/// The steepest neighbour is the one with the largest (resp. smallest) value
/// under the simulation-of-simplicity order; a vertex with no higher (resp.
/// lower) neighbour is its own terminal.  Vertices without an adjacency entry
/// and out-of-range neighbour indices are treated as having no neighbours.
pub(super) fn steepest_terminals(
    values: &[f64],
    adjacency: &[Vec<usize>],
    ascending: bool,
) -> Vec<usize> {
    let n = values.len();
    let (order, rank) = simulation_of_simplicity_order(values);
    let mut terminal: Vec<usize> = (0..n).collect();
    let visit: Vec<usize> = if ascending {
        order.iter().rev().copied().collect()
    } else {
        order
    };
    for v in visit {
        let neighbours = adjacency.get(v).map(Vec::as_slice).unwrap_or(&[]);
        let best = neighbours
            .iter()
            .copied()
            .filter(|&u| u < n)
            .filter(|&u| {
                if ascending {
                    rank[u] > rank[v]
                } else {
                    rank[u] < rank[v]
                }
            })
            .max_by_key(|&u| if ascending { rank[u] } else { n - rank[u] });
        if let Some(b) = best {
            terminal[v] = terminal[b];
        }
    }
    terminal
}
