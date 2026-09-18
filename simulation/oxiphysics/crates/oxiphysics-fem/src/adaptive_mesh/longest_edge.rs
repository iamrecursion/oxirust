// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Conforming longest-edge bisection (Rivara) of simplicial meshes.
//!
//! Bisecting a simplex through the midpoint of one of its edges is only valid for a
//! mesh when *every* simplex that shares that edge is bisected through the same
//! midpoint; otherwise the midpoint is a hanging node on the unsplit neighbours and
//! the mesh is no longer conforming.  [`LongestEdgeMesh`] therefore never bisects a
//! single simplex in isolation: it bisects the whole *star* of an edge (all simplices
//! that contain it) at once, which keeps the mesh conforming after every single step.
//!
//! Which edge is bisected follows Rivara's longest-edge propagation path.  To refine
//! a simplex `t` the engine starts at its longest edge `e`.  If some simplex in the
//! star of `e` has an even longer longest edge `e'`, bisecting `e` would split that
//! neighbour through a poor (non-longest) edge, so the path moves on to `e'`, and so
//! on.  Edges are compared by length and, for equal lengths, by their node-index
//! pair, a strict total order, so the path grows strictly and ends at a *terminal*
//! edge: one that is the longest edge of every simplex in its star.  The star of the
//! terminal edge is bisected and the path is recomputed until `t` itself has been
//! bisected: through its own longest edge once that edge is terminal, or earlier as a
//! member of the star of another terminal edge that a different refinement chose.
//!
//! Every bisection is only performed when all edges it creates are strictly shorter
//! than the edge it removes.  That makes the refinement finite for any input (a
//! multiset of edge lengths that decreases in the Dershowitz-Manna order cannot
//! decrease for ever) and turns degenerate geometry — coincident nodes, non-finite
//! coordinates, edges too short to be split in floating point — into a refusal to
//! refine instead of an endless loop.
//!
//! The same code serves triangles (`N = 3`) and tetrahedra (`N = 4`).  Children are
//! made by replacing one end node of the bisected edge by the midpoint *in place*, so
//! the vertex order, and with it the orientation of every simplex, is preserved.

use std::collections::HashMap;

/// An undirected edge as the ordered pair `(min, max)` of its end-node indices.
pub(crate) type EdgeKey = (usize, usize);

fn edge_key(a: usize, b: usize) -> EdgeKey {
    if a <= b { (a, b) } else { (b, a) }
}

/// All `N·(N−1)/2` edges of a simplex.
fn cell_edges<const N: usize>(cell: &[usize; N]) -> impl Iterator<Item = EdgeKey> + '_ {
    (0..N).flat_map(move |i| ((i + 1)..N).map(move |j| edge_key(cell[i], cell[j])))
}

/// Squared distance between two points; symmetric in its arguments bit for bit.
fn distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Position of an edge in the total order that decides which edge is "longest".
#[derive(Clone, Copy)]
struct EdgeRank {
    /// Squared length of the edge.
    length_squared: f64,
    /// Node-index pair, the tie-break between edges of equal length.
    key: EdgeKey,
}

impl EdgeRank {
    /// `true` when `self` ranks strictly above `other`.
    fn exceeds(&self, other: &EdgeRank) -> bool {
        self.length_squared > other.length_squared
            || (self.length_squared == other.length_squared && self.key > other.key)
    }
}

/// A simplicial mesh refined by conforming longest-edge bisection.
///
/// `N` is the number of vertices of a simplex (`3` for triangles, `4` for
/// tetrahedra) and `P` the per-simplex payload (polynomial degree, refinement level,
/// ...) that every child inherits from its parent through `inherit`.
pub(crate) struct LongestEdgeMesh<const N: usize, P> {
    /// Node coordinates; refinement appends the new midpoints.
    nodes: Vec<[f64; 3]>,
    /// Simplices.  A bisected simplex keeps its slot for its first child and the
    /// second child is appended, so untouched simplices never move.
    cells: Vec<[usize; N]>,
    /// Payload of every slot of `cells`.
    payload: Vec<P>,
    /// Payload of a child, derived from the payload of its parent.
    inherit: fn(&P) -> P,
    /// Slots of the simplices that contain an edge, in ascending slot order.
    stars: HashMap<EdgeKey, Vec<usize>>,
    /// `true` for every slot whose simplex was produced by a bisection.
    replaced: Vec<bool>,
    /// Number of simplices bisected so far.
    bisections: usize,
}

impl<const N: usize, P: Clone> LongestEdgeMesh<N, P> {
    /// Build the refinement state of a mesh.
    ///
    /// `payload` must hold one entry per simplex.
    pub(crate) fn new(
        nodes: Vec<[f64; 3]>,
        cells: Vec<[usize; N]>,
        payload: Vec<P>,
        inherit: fn(&P) -> P,
    ) -> Self {
        let mut stars: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
        for (slot, cell) in cells.iter().enumerate() {
            for key in cell_edges(cell) {
                let star = stars.entry(key).or_default();
                // A simplex with a repeated vertex lists the same edge twice.
                if star.last() != Some(&slot) {
                    star.push(slot);
                }
            }
        }
        let replaced = vec![false; cells.len()];
        Self {
            nodes,
            cells,
            payload,
            inherit,
            stars,
            replaced,
            bisections: 0,
        }
    }

    /// Number of simplices bisected so far, including those that were bisected only
    /// to keep the mesh conforming.
    pub(crate) fn bisections(&self) -> usize {
        self.bisections
    }

    /// Hand the refined mesh back as `(nodes, simplices, payload)`.
    pub(crate) fn into_parts(self) -> (Vec<[f64; 3]>, Vec<[usize; N]>, Vec<P>) {
        (self.nodes, self.cells, self.payload)
    }

    /// Bisect the simplex in slot `target` through its longest edge, together with
    /// every simplex that has to be bisected for the mesh to stay conforming.
    ///
    /// Returns `true` when the simplex has been bisected (also when an earlier call
    /// had already done so) and `false` when it cannot be: the slot does not exist, a
    /// simplex on the propagation path is degenerate (non-finite coordinates, a node
    /// index outside the node list, coincident nodes), or an edge is too short to be
    /// halved in floating point.  The mesh is conforming in either case.
    pub(crate) fn refine(&mut self, target: usize) -> bool {
        if target >= self.cells.len() {
            return false;
        }
        while !self.replaced[target] {
            let Some(edge) = self.terminal_edge(target) else {
                return false;
            };
            if !self.bisect_star(edge) {
                return false;
            }
        }
        true
    }

    /// Rank of the longest edge of the simplex in `slot`; `None` when the simplex
    /// has a node index outside the node list, a non-finite coordinate or a
    /// zero-length edge.
    fn longest_edge_of(&self, slot: usize) -> Option<EdgeRank> {
        let cell = self.cells.get(slot)?;
        let mut longest: Option<EdgeRank> = None;
        for i in 0..N {
            for j in (i + 1)..N {
                let a = *self.nodes.get(cell[i])?;
                let b = *self.nodes.get(cell[j])?;
                let length_squared = distance_squared(a, b);
                if !(length_squared.is_finite() && length_squared > 0.0) {
                    return None;
                }
                let rank = EdgeRank {
                    length_squared,
                    key: edge_key(cell[i], cell[j]),
                };
                if longest.as_ref().is_none_or(|best| rank.exceeds(best)) {
                    longest = Some(rank);
                }
            }
        }
        longest
    }

    /// End of the longest-edge propagation path that starts at the longest edge of
    /// the simplex in `target`; `None` when a simplex on the path is degenerate.
    fn terminal_edge(&self, target: usize) -> Option<EdgeKey> {
        let mut current = self.longest_edge_of(target)?;
        loop {
            let star = self.stars.get(&current.key)?;
            let mut longer: Option<EdgeRank> = None;
            for &slot in star {
                let candidate = self.longest_edge_of(slot)?;
                if candidate.exceeds(&current)
                    && longer.as_ref().is_none_or(|best| candidate.exceeds(best))
                {
                    longer = Some(candidate);
                }
            }
            match longer {
                Some(next) => current = next,
                None => return Some(current.key),
            }
        }
    }

    /// Bisect every simplex that contains `edge` through one shared midpoint.
    ///
    /// Nothing is changed (and `false` is returned) unless every edge that the
    /// bisection creates is strictly shorter than `edge`.
    fn bisect_star(&mut self, edge: EdgeKey) -> bool {
        let (a, b) = edge;
        let Some(slots) = self.stars.get(&edge).cloned() else {
            return false;
        };
        let (Some(&node_a), Some(&node_b)) = (self.nodes.get(a), self.nodes.get(b)) else {
            return false;
        };
        let midpoint = [
            0.5 * (node_a[0] + node_b[0]),
            0.5 * (node_a[1] + node_b[1]),
            0.5 * (node_a[2] + node_b[2]),
        ];
        if midpoint == node_a || midpoint == node_b {
            return false;
        }
        let removed = distance_squared(node_a, node_b);
        let halves_shrink = distance_squared(node_a, midpoint) < removed
            && distance_squared(midpoint, node_b) < removed;
        if !halves_shrink {
            return false;
        }
        for &slot in &slots {
            let Some(cell) = self.cells.get(slot) else {
                return false;
            };
            for &vertex in cell.iter().filter(|&&v| v != a && v != b) {
                let Some(&node) = self.nodes.get(vertex) else {
                    return false;
                };
                let shrinks = distance_squared(midpoint, node) < removed;
                if !shrinks {
                    return false;
                }
            }
        }

        let mid_index = self.nodes.len();
        self.nodes.push(midpoint);
        self.stars.remove(&edge);
        for slot in slots {
            self.split_cell(slot, a, b, mid_index);
        }
        true
    }

    /// Replace the simplex in `slot` by its two halves at `mid_index`, the midpoint
    /// of its edge `(a, b)`, and keep the edge stars up to date.
    fn split_cell(&mut self, slot: usize, a: usize, b: usize, mid_index: usize) {
        let cell = self.cells[slot];
        // The bisected edge's own star is dropped by the caller; every other edge of
        // the old simplex loses this slot.
        for key in cell_edges(&cell).filter(|&key| key != edge_key(a, b)) {
            if let Some(star) = self.stars.get_mut(&key) {
                star.retain(|&s| s != slot);
                if star.is_empty() {
                    self.stars.remove(&key);
                }
            }
        }

        // Replacing a vertex by the midpoint in place keeps the orientation.
        let mut first = cell;
        let mut second = cell;
        for (position, &vertex) in cell.iter().enumerate() {
            if vertex == b {
                first[position] = mid_index;
            } else if vertex == a {
                second[position] = mid_index;
            }
        }

        let child_payload = (self.inherit)(&self.payload[slot]);
        let second_slot = self.cells.len();
        self.cells[slot] = first;
        self.cells.push(second);
        self.payload[slot] = child_payload.clone();
        self.payload.push(child_payload);
        self.replaced[slot] = true;
        self.replaced.push(true);
        self.bisections += 1;

        for key in cell_edges(&first) {
            let star = self.stars.entry(key).or_default();
            if let Err(position) = star.binary_search(&slot) {
                star.insert(position, slot);
            }
        }
        // `second_slot` is larger than every slot in use, so appending keeps the
        // stars sorted.
        for key in cell_edges(&second) {
            self.stars.entry(key).or_default().push(second_slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_mesh(nodes: Vec<[f64; 3]>, cells: Vec<[usize; 3]>) -> LongestEdgeMesh<3, usize> {
        let payload = vec![0_usize; cells.len()];
        LongestEdgeMesh::new(nodes, cells, payload, |level| level + 1)
    }

    /// Check that the stars list exactly the simplices that contain each edge.
    fn assert_stars_are_consistent<const N: usize, P: Clone>(mesh: &LongestEdgeMesh<N, P>) {
        let mut expected: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
        for (slot, cell) in mesh.cells.iter().enumerate() {
            for key in cell_edges(cell) {
                expected.entry(key).or_default().push(slot);
            }
        }
        assert_eq!(mesh.stars, expected, "edge stars went out of sync");
        assert_eq!(mesh.payload.len(), mesh.cells.len());
        assert_eq!(mesh.replaced.len(), mesh.cells.len());
    }

    #[test]
    fn edge_rank_is_a_strict_total_order_with_index_tie_break() {
        let short = EdgeRank {
            length_squared: 1.0,
            key: (5, 9),
        };
        let long = EdgeRank {
            length_squared: 2.0,
            key: (0, 1),
        };
        let tied_low = EdgeRank {
            length_squared: 2.0,
            key: (0, 1),
        };
        let tied_high = EdgeRank {
            length_squared: 2.0,
            key: (0, 2),
        };
        assert!(long.exceeds(&short) && !short.exceeds(&long));
        assert!(tied_high.exceeds(&tied_low) && !tied_low.exceeds(&tied_high));
        assert!(!tied_low.exceeds(&long) && !long.exceeds(&tied_low));
    }

    #[test]
    fn single_triangle_is_bisected_through_its_longest_edge() {
        // Longest edge is (1, 2), between (2, 0) and (0, 1).
        let nodes = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut mesh = triangle_mesh(nodes, vec![[0, 1, 2]]);
        assert!(mesh.refine(0));
        assert_eq!(mesh.bisections(), 1);
        let (nodes, cells, payload) = mesh.into_parts();
        assert_eq!(nodes.len(), 4);
        assert_eq!(nodes[3], [1.0, 0.5, 0.0]);
        // The first child keeps the parent's slot: end node 2 is replaced by the
        // midpoint; the second child replaces end node 1 and is appended.
        assert_eq!(cells, vec![[0, 1, 3], [0, 3, 2]]);
        // Both children inherit from the parent.
        assert_eq!(payload, vec![1, 1]);
    }

    #[test]
    fn neighbour_with_a_longer_edge_is_bisected_first() {
        // T0 = (0,1,2) has its longest edge (1,2); the neighbour T1 = (1,3,2) across
        // that edge has the longer edge (2,3), so the path must continue to it.
        let nodes = vec![
            [0.0, 0.0, 0.0], // 0
            [2.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0], // 2
            [3.0, 3.0, 0.0], // 3
        ];
        let mut mesh = triangle_mesh(nodes, vec![[0, 1, 2], [1, 3, 2]]);
        assert!(mesh.refine(0));
        assert_stars_are_consistent(&mesh);
        // T1 is cut through (2,3) and then through the shared edge (1,2): three
        // triangles; T0 is cut through (1,2): two.  Five in total, two new nodes.
        assert_eq!(mesh.bisections(), 3);
        let (nodes, cells, _) = mesh.into_parts();
        assert_eq!(cells.len(), 5);
        assert_eq!(nodes.len(), 6);
    }

    #[test]
    fn already_replaced_and_out_of_range_targets_are_handled() {
        let nodes = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut mesh = triangle_mesh(nodes, vec![[0, 1, 2]]);
        assert!(!mesh.refine(7));
        assert!(mesh.refine(0));
        // The slot now holds a child, which counts as refined for this session.
        assert!(mesh.refine(0));
        assert_eq!(mesh.bisections(), 1);
    }

    #[test]
    fn degenerate_simplices_are_refused_without_looping() {
        // Two coincident nodes: a zero-length edge.
        let nodes = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut mesh = triangle_mesh(nodes, vec![[0, 1, 2]]);
        assert!(!mesh.refine(0));
        assert_eq!(mesh.bisections(), 0);

        // A non-finite coordinate.
        let nodes = vec![[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut mesh = triangle_mesh(nodes, vec![[0, 1, 2]]);
        assert!(!mesh.refine(0));

        // A node index outside the node list.
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut mesh = triangle_mesh(nodes, vec![[0, 1, 5]]);
        assert!(!mesh.refine(0));

        // A repeated vertex.
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut mesh = triangle_mesh(nodes, vec![[0, 0, 1]]);
        assert!(!mesh.refine(0));
        assert_eq!(mesh.bisections(), 0);
    }

    #[test]
    fn an_edge_too_short_to_halve_is_refused() {
        // The end nodes are adjacent doubles: their midpoint rounds onto one of them,
        // so the segment cannot be bisected into two shorter segments.
        let x = 1.0_f64;
        let next = f64::from_bits(x.to_bits() + 1);
        let nodes = vec![[x, 0.0, 0.0], [next, 0.0, 0.0]];
        let mut mesh: LongestEdgeMesh<2, u8> =
            LongestEdgeMesh::new(nodes, vec![[0, 1]], vec![0], |l| l.saturating_add(1));
        assert!(!mesh.refine(0));
        assert_eq!(mesh.bisections(), 0);
        assert_eq!(mesh.cells, vec![[0, 1]]);
        assert_eq!(mesh.nodes.len(), 2);
    }

    #[test]
    fn tetrahedron_bisection_replaces_the_opposite_vertices_in_place() {
        // The longest edge of this tetrahedron is (0, 1).
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.3, 0.8],
        ];
        let payload = vec![0_u8];
        let mut mesh: LongestEdgeMesh<4, u8> =
            LongestEdgeMesh::new(nodes, vec![[0, 1, 2, 3]], payload, |l| l.saturating_add(1));
        assert!(mesh.refine(0));
        assert_stars_are_consistent(&mesh);
        let (nodes, cells, levels) = mesh.into_parts();
        assert_eq!(nodes.len(), 5);
        assert_eq!(nodes[4], [1.0, 0.0, 0.0]);
        assert_eq!(cells, vec![[0, 4, 2, 3], [4, 1, 2, 3]]);
        assert_eq!(levels, vec![1, 1]);
    }

    #[test]
    fn the_whole_edge_star_is_bisected_through_one_midpoint() {
        // Three tetrahedra around the common longest edge (0, 1): one midpoint, six
        // children, no duplicated nodes.
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, -0.5, 0.8],
            [2.0, -0.5, -0.8],
        ];
        let cells = vec![[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]];
        let payload = vec![0_u8; cells.len()];
        let mut mesh: LongestEdgeMesh<4, u8> =
            LongestEdgeMesh::new(nodes, cells, payload, |l| l.saturating_add(1));
        assert!(mesh.refine(1));
        assert_stars_are_consistent(&mesh);
        assert_eq!(mesh.bisections(), 3);
        let (nodes, cells, levels) = mesh.into_parts();
        assert_eq!(nodes.len(), 6);
        assert_eq!(cells.len(), 6);
        assert!(levels.iter().all(|&l| l == 1));
        // No tetrahedron still contains the removed edge (0, 1).
        for cell in &cells {
            assert!(!(cell.contains(&0) && cell.contains(&1)), "{cell:?}");
        }
    }

    #[test]
    fn refinement_state_stays_consistent_over_many_bisections() {
        // A fan of triangles around node 0 with irregular edge lengths: refine every
        // slot in turn and check the star bookkeeping after each call.
        let mut nodes = vec![[0.0, 0.0, 0.0]];
        for k in 0..9 {
            let angle = 0.7 * k as f64;
            let radius = 1.0 + 0.3 * (k % 4) as f64;
            nodes.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
        }
        let cells: Vec<[usize; 3]> = (1..9).map(|k| [0, k, k + 1]).collect();
        let count = cells.len();
        let mut mesh = triangle_mesh(nodes, cells);
        for slot in 0..count {
            assert!(mesh.refine(slot), "slot {slot}");
            assert_stars_are_consistent(&mesh);
        }
        assert!(mesh.bisections() >= count);
    }
}
