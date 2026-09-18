// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Repulsive forces of the Fruchterman-Reingold layout: the exact `O(n^2)` sum
//! and its Barnes-Hut approximation on a quadtree.
//!
//! # The force
//!
//! Two nodes at distance `d` repel each other along their connecting line with
//! a force of magnitude `strength / d`, where `strength` is `repulsion * k^2`
//! for the optimal distance `k`.  The distance is never taken below
//! [`MIN_DISTANCE`], so nodes that are very close do not produce huge forces.
//! Nodes at exactly the same position do not produce any force: the direction
//! between them is undefined.
//!
//! # Barnes-Hut
//!
//! The nodes are inserted into a quadtree.  Every cell knows how many nodes it
//! holds and where their centre of mass is.  The repulsion on one node is
//! computed by walking the tree from the root:
//!
//! * a leaf contributes the exact repulsion of every node it holds;
//! * an inner cell is replaced by a single body (its mass at its centre of
//!   mass) when it is small compared to its distance, `width / d < theta`, and
//!   is opened otherwise.
//!
//! A cell that contains the node itself is always opened, so a node is never
//! repelled by its own mass (for `theta >= 1 / sqrt(2)` the distance test alone
//! could accept such a cell).  With `theta = 0` every inner cell is opened and
//! the result is the exact sum up to the order of the additions; a larger
//! `theta` trades accuracy for speed, as the walk then touches `O(log n)` cells
//! per node instead of `n`.
//!
//! The tree lives in an arena (`Vec<Cell>`) and is walked with an explicit
//! stack, so neither building nor querying recurses.  Nodes at identical
//! positions share a leaf (they could never be separated by subdividing), and
//! the subdivision stops at [`MAX_DEPTH`] levels for the same reason with
//! distinct but extremely close nodes.

/// Smallest distance at which a repulsion is evaluated.
pub(super) const MIN_DISTANCE: f64 = 0.01;

/// Deepest tree level.  Nodes that are still not separated there (they are
/// closer than `2^-48` of the extent of the layout) share a leaf.
const MAX_DEPTH: usize = 48;

/// Marks "no cell" and "no body".
const NONE: usize = usize::MAX;

/// Index of the root cell.
const ROOT: usize = 0;

/// Repulsion on a node at `at` from `mass` unit nodes concentrated at `source`.
fn repulsion_from(at: [f64; 2], source: [f64; 2], mass: f64, strength: f64) -> [f64; 2] {
    let dx = at[0] - source[0];
    let dy = at[1] - source[1];
    let dist = (dx * dx + dy * dy).sqrt().max(MIN_DISTANCE);
    let force = strength * mass / dist;
    [force * dx / dist, force * dy / dist]
}

/// Exact repulsion on every node: the sum over all other nodes, `O(n^2)`.
pub(super) fn exact_repulsion(positions: &[[f64; 2]], strength: f64) -> Vec<[f64; 2]> {
    let mut displacement = vec![[0.0_f64; 2]; positions.len()];
    for (i, &at) in positions.iter().enumerate() {
        for (j, &other) in positions.iter().enumerate().skip(i + 1) {
            let [fx, fy] = repulsion_from(at, other, 1.0, strength);
            displacement[i][0] += fx;
            displacement[i][1] += fy;
            displacement[j][0] -= fx;
            displacement[j][1] -= fy;
        }
    }
    displacement
}

/// Barnes-Hut repulsion on every node with opening angle `theta`.
///
/// A `theta` that is not positive (or not a number) opens every cell and gives
/// the exact sum.
pub(super) fn approximate_repulsion(
    positions: &[[f64; 2]],
    theta: f64,
    strength: f64,
) -> Vec<[f64; 2]> {
    let tree = QuadTree::build(positions);
    let theta_squared = if theta > 0.0 { theta * theta } else { 0.0 };
    let mut stack: Vec<(usize, bool)> = Vec::new();
    (0..positions.len())
        .map(|body| tree.repulsion_on(body, theta_squared, strength, &mut stack))
        .collect()
}

/// One square cell of the quadtree.
#[derive(Debug, Clone)]
struct Cell {
    /// Centre of the cell.
    center: [f64; 2],
    /// Half of the side length.
    half: f64,
    /// Child cells by quadrant ([`Cell::quadrant_of`]); all [`NONE`] in a leaf.
    children: [usize; 4],
    /// First node of the leaf's bucket ([`NONE`] for an inner cell or an empty
    /// tree); the others follow through `QuadTree::next_body`.
    first_body: usize,
    /// Number of nodes in the cell (each has unit mass).
    mass: f64,
    /// Sum of the positions of the nodes in the cell.
    moment: [f64; 2],
}

impl Cell {
    fn new(center: [f64; 2], half: f64) -> Self {
        Self {
            center,
            half,
            children: [NONE; 4],
            first_body: NONE,
            mass: 0.0,
            moment: [0.0; 2],
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.iter().all(|&child| child == NONE)
    }

    /// Quadrant of `at`: bit 0 is set for `x >= center.x`, bit 1 for
    /// `y >= center.y`.  Points on a dividing line belong to the upper side.
    fn quadrant_of(&self, at: [f64; 2]) -> usize {
        usize::from(at[0] >= self.center[0]) | (usize::from(at[1] >= self.center[1]) << 1)
    }

    /// Centre and half side length of the child in `quadrant`.
    fn child_geometry(&self, quadrant: usize) -> ([f64; 2], f64) {
        let quarter = 0.5 * self.half;
        let dx = if quadrant & 1 == 0 { -quarter } else { quarter };
        let dy = if quadrant & 2 == 0 { -quarter } else { quarter };
        ([self.center[0] + dx, self.center[1] + dy], quarter)
    }
}

/// Quadtree over a set of node positions.
struct QuadTree<'a> {
    positions: &'a [[f64; 2]],
    /// Cell arena.  A child is always created after its parent, so a child's
    /// index is larger than its parent's.
    cells: Vec<Cell>,
    /// `next_body[b]` is the node that follows `b` in its leaf bucket.
    next_body: Vec<usize>,
}

impl<'a> QuadTree<'a> {
    /// Build the tree over `positions` (node `i` is `positions[i]`).
    fn build(positions: &'a [[f64; 2]]) -> Self {
        let (center, half) = bounding_square(positions);
        let mut tree = Self {
            positions,
            cells: vec![Cell::new(center, half)],
            next_body: vec![NONE; positions.len()],
        };
        for body in 0..positions.len() {
            tree.insert(body);
        }
        tree.accumulate();
        tree
    }

    /// Append a child to `parent` in `quadrant` and return its index.
    fn new_child(&mut self, parent: usize, quadrant: usize) -> usize {
        let (center, half) = self.cells[parent].child_geometry(quadrant);
        self.cells.push(Cell::new(center, half));
        let child = self.cells.len() - 1;
        self.cells[parent].children[quadrant] = child;
        child
    }

    /// Insert node `body` into the tree (topology only; the masses are
    /// filled in by [`QuadTree::accumulate`]).
    fn insert(&mut self, body: usize) {
        let at = self.positions[body];
        let mut index = ROOT;
        let mut depth = 0_usize;
        loop {
            if self.cells[index].is_leaf() {
                let head = self.cells[index].first_body;
                if head == NONE {
                    self.cells[index].first_body = body;
                    return;
                }
                if depth >= MAX_DEPTH || self.positions[head] == at {
                    self.next_body[body] = head;
                    self.cells[index].first_body = body;
                    return;
                }
                // The bucket's nodes sit at one position (the depth limit and
                // the comparison above guarantee it), so they all move into
                // the same child.  The cell is an inner cell from now on and
                // the loop places `body` again.
                let quadrant = self.cells[index].quadrant_of(self.positions[head]);
                let child = self.new_child(index, quadrant);
                self.cells[child].first_body = head;
                self.cells[index].first_body = NONE;
                continue;
            }
            let quadrant = self.cells[index].quadrant_of(at);
            let child = self.cells[index].children[quadrant];
            if child == NONE {
                let child = self.new_child(index, quadrant);
                self.cells[child].first_body = body;
                return;
            }
            index = child;
            depth += 1;
        }
    }

    /// Fill in the mass and the position sum of every cell, children before
    /// parents.
    fn accumulate(&mut self) {
        for index in (0..self.cells.len()).rev() {
            let mut mass = 0.0_f64;
            let mut moment = [0.0_f64; 2];
            if self.cells[index].is_leaf() {
                let mut body = self.cells[index].first_body;
                while body != NONE {
                    mass += 1.0;
                    moment[0] += self.positions[body][0];
                    moment[1] += self.positions[body][1];
                    body = self.next_body[body];
                }
            } else {
                for &child in &self.cells[index].children {
                    if child != NONE {
                        let cell = &self.cells[child];
                        mass += cell.mass;
                        moment[0] += cell.moment[0];
                        moment[1] += cell.moment[1];
                    }
                }
            }
            self.cells[index].mass = mass;
            self.cells[index].moment = moment;
        }
    }

    /// Repulsion on node `body`.  `stack` is scratch space reused between
    /// calls; an entry is a cell and whether it contains `body`.
    fn repulsion_on(
        &self,
        body: usize,
        theta_squared: f64,
        strength: f64,
        stack: &mut Vec<(usize, bool)>,
    ) -> [f64; 2] {
        let at = self.positions[body];
        let mut force = [0.0_f64; 2];
        stack.clear();
        stack.push((ROOT, true));
        while let Some((index, holds_body)) = stack.pop() {
            let cell = &self.cells[index];
            if cell.is_leaf() {
                let mut other = cell.first_body;
                while other != NONE {
                    if other != body {
                        let [fx, fy] = repulsion_from(at, self.positions[other], 1.0, strength);
                        force[0] += fx;
                        force[1] += fy;
                    }
                    other = self.next_body[other];
                }
                continue;
            }
            if !holds_body {
                let center_of_mass = [cell.moment[0] / cell.mass, cell.moment[1] / cell.mass];
                let dx = at[0] - center_of_mass[0];
                let dy = at[1] - center_of_mass[1];
                let width = 2.0 * cell.half;
                // width / d < theta, without the square root and the division.
                if width * width < theta_squared * (dx * dx + dy * dy) {
                    let [fx, fy] = repulsion_from(at, center_of_mass, cell.mass, strength);
                    force[0] += fx;
                    force[1] += fy;
                    continue;
                }
            }
            let own_quadrant = cell.quadrant_of(at);
            for (quadrant, &child) in cell.children.iter().enumerate() {
                if child != NONE {
                    stack.push((child, holds_body && quadrant == own_quadrant));
                }
            }
        }
        force
    }
}

/// Centre and half side length of a square that contains all `positions`.
///
/// Non-finite coordinates are ignored where possible; if no sensible square
/// results, the unit square around the origin is used.  The walk never relies
/// on the square containing every node, only on all nodes being placed by the
/// same quadrant rule, so a degenerate square costs accuracy at worst.
fn bounding_square(positions: &[[f64; 2]]) -> ([f64; 2], f64) {
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for position in positions {
        for ((low, high), &coordinate) in low.iter_mut().zip(high.iter_mut()).zip(position) {
            *low = low.min(coordinate);
            *high = high.max(coordinate);
        }
    }
    let center = [0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])];
    let extent = (high[0] - low[0]).max(high[1] - low[1]);
    // Nodes that all coincide have no extent; any positive size will do.
    let half = if extent > 0.0 { 0.5 * extent } else { 0.5 };
    if center.iter().all(|c| c.is_finite()) && half.is_finite() {
        (center, half)
    } else {
        ([0.0, 0.0], 1.0)
    }
}

#[cfg(test)]
mod tests;
