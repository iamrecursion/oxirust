// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Delaunay triangulation of scattered 2-D points, for the background integration
//! cells of meshfree (EFG) analyses.
//!
//! The triangulation is built by incremental Bowyer–Watson insertion in its
//! "vertex at infinity" form: besides the real triangles, every edge of the convex
//! hull carries a *ghost* triangle `(v, u, ∞)` on its outer side, so that the
//! triangulation is a closed surface and a point outside the hull needs no special
//! case.  A real triangle is in conflict with a new point when the point lies
//! strictly inside its circumcircle; a ghost triangle when the point lies strictly
//! outside its hull edge (or on the open edge).  The conflicting triangles form a
//! star-shaped cavity that is re-triangulated as a fan around the new point.
//!
//! Orientation and in-circle tests use the exact adaptive predicates of
//! [`oxiphysics_core::exact_predicates`]: collinear and cocircular configurations
//! (regular grids, points on the hull, repeated coordinates) are decided correctly,
//! and the triangles always tile the complete convex hull of the points, with no
//! sliver lost to a finite "super triangle".

use std::collections::{HashMap, HashSet};

use oxiphysics_core::exact_predicates::{Orientation, incircle, orient2d};

use super::functions::Point2;
use super::types::TriCell;

/// Index of the vertex at infinity (it is never looked up in the point list).
const GHOST: usize = usize::MAX;

/// `true` if `p`, known to be collinear with `a` and `b`, lies strictly between
/// them.  `a` and `b` are distinct.
fn strictly_between(a: Point2, b: Point2, p: Point2) -> bool {
    let axis = if (a[0] - b[0]).abs() >= (a[1] - b[1]).abs() {
        0
    } else {
        1
    };
    let (low, high) = if a[axis] <= b[axis] {
        (a[axis], b[axis])
    } else {
        (b[axis], a[axis])
    };
    low < p[axis] && p[axis] < high
}

/// A triangulation with ghost triangles, stored as triangle slots plus a map from
/// every directed edge to the slot of the triangle that owns it.
struct Mesh<'a> {
    points: &'a [Point2],
    triangles: Vec<[usize; 3]>,
    alive: Vec<bool>,
    free_slots: Vec<usize>,
    owner: HashMap<(usize, usize), usize>,
}

impl<'a> Mesh<'a> {
    /// A mesh made of the counter-clockwise triangle `seed` and its three ghosts.
    fn new(points: &'a [Point2], seed: [usize; 3]) -> Self {
        let mut mesh = Self {
            points,
            triangles: Vec::new(),
            alive: Vec::new(),
            free_slots: Vec::new(),
            owner: HashMap::new(),
        };
        let [a, b, c] = seed;
        mesh.add([a, b, c]);
        mesh.add([b, a, GHOST]);
        mesh.add([c, b, GHOST]);
        mesh.add([a, c, GHOST]);
        mesh
    }

    /// Store a triangle and register its three directed edges.
    fn add(&mut self, triangle: [usize; 3]) -> usize {
        let slot = match self.free_slots.pop() {
            Some(slot) => {
                self.triangles[slot] = triangle;
                self.alive[slot] = true;
                slot
            }
            None => {
                self.triangles.push(triangle);
                self.alive.push(true);
                self.triangles.len() - 1
            }
        };
        for edge in 0..3 {
            self.owner
                .insert((triangle[edge], triangle[(edge + 1) % 3]), slot);
        }
        slot
    }

    /// Delete a triangle and release its directed edges.
    fn remove(&mut self, slot: usize) {
        let triangle = self.triangles[slot];
        for edge in 0..3 {
            let key = (triangle[edge], triangle[(edge + 1) % 3]);
            if self.owner.get(&key) == Some(&slot) {
                self.owner.remove(&key);
            }
        }
        self.alive[slot] = false;
        self.free_slots.push(slot);
    }

    /// The triangle across the directed edge `from -> to` of a triangle.
    fn across(&self, from: usize, to: usize) -> Option<usize> {
        self.owner.get(&(to, from)).copied()
    }

    /// Whether `p` is in conflict with `triangle`: strictly inside the circumcircle
    /// of a real triangle, strictly outside the hull edge of a ghost (or on its
    /// open segment, where the real triangle behind it is in conflict as well).
    fn conflicts(&self, triangle: [usize; 3], p: Point2) -> bool {
        match triangle.iter().position(|&vertex| vertex == GHOST) {
            Some(ghost) => {
                // rotate to (x, y, ghost): the hull edge is y -> x with the interior on
                // its left, so the outside is on the left of x -> y
                let x = self.points[triangle[(ghost + 1) % 3]];
                let y = self.points[triangle[(ghost + 2) % 3]];
                match orient2d(x, y, p) {
                    Orientation::Positive => true,
                    Orientation::Negative => false,
                    Orientation::Degenerate => strictly_between(x, y, p),
                }
            }
            None => {
                incircle(
                    self.points[triangle[0]],
                    self.points[triangle[1]],
                    self.points[triangle[2]],
                    p,
                ) == Orientation::Positive
            }
        }
    }

    /// A triangle in conflict with `p`, found by a visibility walk from `hint`.
    ///
    /// Returns `None` if no triangle is in conflict, which is the case exactly when
    /// `p` coincides with a vertex.
    fn locate(&self, p: Point2, hint: usize) -> Option<usize> {
        let mut current = hint;
        for _ in 0..self.triangles.len() + 8 {
            let triangle = self.triangles[current];
            if self.conflicts(triangle, p) {
                return Some(current);
            }
            if let Some(ghost) = triangle.iter().position(|&vertex| vertex == GHOST) {
                // outside the ghost's reach: continue in the real triangle behind it
                let x = triangle[(ghost + 1) % 3];
                let y = triangle[(ghost + 2) % 3];
                match self.across(x, y) {
                    Some(next) => current = next,
                    None => break,
                }
                continue;
            }
            let mut next = None;
            for edge in 0..3 {
                let from = triangle[edge];
                let to = triangle[(edge + 1) % 3];
                if orient2d(self.points[from], self.points[to], p) == Orientation::Negative {
                    next = self.across(from, to);
                    break;
                }
            }
            match next {
                Some(slot) => current = slot,
                // `p` lies in the closed triangle without being in its open circumdisk:
                // it is one of the triangle's vertices
                None => return None,
            }
        }
        // The walk is bounded to stay safe on any input; a scan always finds a conflict
        // if there is one.
        (0..self.triangles.len())
            .find(|&slot| self.alive[slot] && self.conflicts(self.triangles[slot], p))
    }

    /// Insert the point `index`; returns the slot of a triangle created by the
    /// insertion (a good start for the next walk), or `hint` if the point
    /// duplicates a vertex.
    fn insert(&mut self, index: usize, hint: usize) -> usize {
        let p = self.points[index];
        let Some(seed) = self.locate(p, hint) else {
            return hint;
        };
        // the cavity: all triangles in conflict, connected through shared edges
        let mut cavity = vec![seed];
        let mut in_cavity: HashSet<usize> = HashSet::from([seed]);
        let mut stack = vec![seed];
        while let Some(slot) = stack.pop() {
            let triangle = self.triangles[slot];
            for edge in 0..3 {
                let Some(neighbour) = self.across(triangle[edge], triangle[(edge + 1) % 3]) else {
                    continue;
                };
                if !in_cavity.contains(&neighbour) && self.conflicts(self.triangles[neighbour], p) {
                    in_cavity.insert(neighbour);
                    cavity.push(neighbour);
                    stack.push(neighbour);
                }
            }
        }
        // its boundary: the directed edges whose twin belongs to a surviving triangle
        let mut boundary = Vec::new();
        for &slot in &cavity {
            let triangle = self.triangles[slot];
            for edge in 0..3 {
                let from = triangle[edge];
                let to = triangle[(edge + 1) % 3];
                let survives = self
                    .across(from, to)
                    .is_none_or(|neighbour| !in_cavity.contains(&neighbour));
                if survives {
                    boundary.push((from, to));
                }
            }
        }
        for &slot in &cavity {
            self.remove(slot);
        }
        let mut newest = hint;
        for (from, to) in boundary {
            newest = self.add([from, to, index]);
        }
        newest
    }

    /// The real triangles, counter-clockwise.
    fn into_triangles(self) -> Vec<[usize; 3]> {
        self.triangles
            .into_iter()
            .zip(self.alive)
            .filter(|(triangle, alive)| *alive && !triangle.contains(&GHOST))
            .map(|(triangle, _)| triangle)
            .collect()
    }
}

/// Delaunay triangulation of a 2-D point set.
///
/// Returns counter-clockwise index triples into `points`.  The triangles tile the
/// convex hull of the points exactly (no triangle is missing at the hull, including
/// collinear points on it) and no point lies strictly inside the circumcircle of a
/// triangle (cocircular points, as on a regular grid, are triangulated in one of the
/// equivalent ways).  Orientation and in-circle decisions are exact.
///
/// Non-finite points and repeated points are ignored (a repeated point is not
/// duplicated into the triangulation; only its first occurrence is used).  Fewer
/// than three distinct points, or points that are all collinear, give no triangle.
///
/// The cost is `O(n log n)` for typical inputs.
pub fn delaunay_triangulation(points: &[Point2]) -> Vec<[usize; 3]> {
    // distinct finite points, in lexicographic order so that consecutive insertions
    // are spatially close
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let mut order: Vec<usize> = (0..points.len())
        .filter(|&index| {
            let [x, y] = points[index];
            // -0.0 and 0.0 are the same coordinate (and the same point)
            let canonical = |v: f64| if v == 0.0 { 0.0 } else { v };
            x.is_finite()
                && y.is_finite()
                && seen.insert((canonical(x).to_bits(), canonical(y).to_bits()))
        })
        .collect();
    order.sort_by(|&a, &b| {
        points[a][0]
            .total_cmp(&points[b][0])
            .then(points[a][1].total_cmp(&points[b][1]))
    });
    if order.len() < 3 {
        return Vec::new();
    }

    // seed triangle: the first two points and the first point off their line
    let (first, second) = (order[0], order[1]);
    let Some(third_position) = order.iter().skip(2).position(|&candidate| {
        orient2d(points[first], points[second], points[candidate]) != Orientation::Degenerate
    }) else {
        return Vec::new();
    };
    let third_position = third_position + 2;
    let third = order[third_position];
    let seed = if orient2d(points[first], points[second], points[third]) == Orientation::Positive {
        [first, second, third]
    } else {
        [first, third, second]
    };

    let mut mesh = Mesh::new(points, seed);
    let mut hint = 0;
    for (position, &index) in order.iter().enumerate() {
        if position < 2 || position == third_position {
            continue;
        }
        hint = mesh.insert(index, hint);
    }
    mesh.into_triangles()
}

/// Delaunay background cells of a 2-D point set.
///
/// One counter-clockwise [`TriCell`] per Delaunay triangle of `nodes` (see
/// [`delaunay_triangulation`]): the cells tile the convex hull of the nodes without
/// overlap or gap, so `Σ area` is the hull area, which is what the background
/// integration of an EFG analysis needs.  Fewer than three distinct nodes, or
/// collinear nodes, give no cell.
pub fn generate_delaunay_cells(nodes: &[Point2]) -> Vec<TriCell> {
    delaunay_triangulation(nodes)
        .into_iter()
        .map(|[a, b, c]| TriCell {
            vertices: [nodes[a], nodes[b], nodes[c]],
        })
        .collect()
}
