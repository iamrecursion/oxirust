// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Bar-and-hinge discretisation of a [`CreasePattern`].
//!
//! Follows the bar-and-hinge model of Liu & Paulino ("Nonlinear mechanics of
//! non-rigid origami: an efficient computational approach", Proc. R. Soc. A
//! 473, 2017):
//!
//! * every panel edge and every panel diagonal becomes an (in-plane rigid)
//!   **bar** whose rest length is measured in the flat crease pattern;
//! * every interior fan diagonal of a polygonal panel (four or more vertices)
//!   becomes a **panel hinge** with rest angle zero, which keeps the panel
//!   flat;
//! * every [`FoldLine`](super::FoldLine) shared by exactly two panels becomes a
//!   **fold hinge** whose signed dihedral angle is driven towards its
//!   commanded fold angle.
//!
//! The constraints are solved with (X)PBD projections using the analytic
//! dihedral-angle gradients of Schenk & Guest / Liu & Paulino.

use std::collections::HashSet;
use std::f64::consts::PI;

use super::{CreasePattern, OrigamiVertex, vec3_cross, vec3_dot, vec3_len, vec3_sub};

/// Wrap an angle to the interval `(−π, π]`.
pub fn wrap_angle(angle: f64) -> f64 {
    let two_pi = 2.0_f64 * PI;
    let mut a = angle % two_pi;
    if a > PI {
        a -= two_pi;
    } else if a <= -PI {
        a += two_pi;
    }
    a
}

/// Signed fold angle of the hinge `(j, k)` with wing vertices `i` and `l`.
///
/// `i` must be the wing lying to the **left** of the directed hinge `j → k`
/// (seen from the reference `+z` side of the flat sheet) and `l` the wing on
/// the right.  The result lies in `(−π, π]`: `0` for a flat hinge, positive
/// for a valley fold (both wings rise towards `+z`) and negative for a
/// mountain fold.  Returns `0` for degenerate (collinear) wings.
pub fn signed_fold_angle(xi: [f64; 3], xj: [f64; 3], xk: [f64; 3], xl: [f64; 3]) -> f64 {
    let r_ij = vec3_sub(xi, xj);
    let r_kj = vec3_sub(xk, xj);
    let r_kl = vec3_sub(xk, xl);
    let m = vec3_cross(r_ij, r_kj);
    let n = vec3_cross(r_kj, r_kl);
    let axis_len = vec3_len(r_kj);
    if vec3_len(m) < 1.0e-15_f64 || vec3_len(n) < 1.0e-15_f64 || axis_len < 1.0e-15_f64 {
        return 0.0_f64;
    }
    let cos_part = vec3_dot(m, n);
    let sin_part = vec3_dot(vec3_cross(m, n), r_kj) / axis_len;
    // atan2(sin, cos) is the Liu–Paulino rotation angle θ (π when flat);
    // the fold angle is ψ = π − θ.
    wrap_angle(PI - sin_part.atan2(cos_part))
}

/// Gradient of [`signed_fold_angle`] with respect to `(x_i, x_j, x_k, x_l)`.
///
/// Uses the closed-form expressions of Liu & Paulino (2017), eq. (9)–(12).
/// Returns `None` when a wing is collinear with the hinge axis.
pub fn signed_fold_angle_gradient(
    xi: [f64; 3],
    xj: [f64; 3],
    xk: [f64; 3],
    xl: [f64; 3],
) -> Option<[[f64; 3]; 4]> {
    let r_ij = vec3_sub(xi, xj);
    let r_kj = vec3_sub(xk, xj);
    let r_kl = vec3_sub(xk, xl);
    let m = vec3_cross(r_ij, r_kj);
    let n = vec3_cross(r_kj, r_kl);
    let m_sq = vec3_dot(m, m);
    let n_sq = vec3_dot(n, n);
    let axis_sq = vec3_dot(r_kj, r_kj);
    if m_sq < 1.0e-30_f64 || n_sq < 1.0e-30_f64 || axis_sq < 1.0e-30_f64 {
        return None;
    }
    let axis_len = axis_sq.sqrt();
    // Gradients of the Liu–Paulino angle θ; ψ = π − θ flips every sign.
    let theta_i = [
        axis_len / m_sq * m[0],
        axis_len / m_sq * m[1],
        axis_len / m_sq * m[2],
    ];
    let theta_l = [
        -axis_len / n_sq * n[0],
        -axis_len / n_sq * n[1],
        -axis_len / n_sq * n[2],
    ];
    let a = vec3_dot(r_ij, r_kj) / axis_sq;
    let b = vec3_dot(r_kl, r_kj) / axis_sq;
    let mut grad = [[0.0_f64; 3]; 4];
    for d in 0..3 {
        let theta_j = (a - 1.0_f64) * theta_i[d] - b * theta_l[d];
        let theta_k = (b - 1.0_f64) * theta_l[d] - a * theta_i[d];
        grad[0][d] = -theta_i[d];
        grad[1][d] = -theta_j;
        grad[2][d] = -theta_k;
        grad[3][d] = -theta_l[d];
    }
    Some(grad)
}

/// An in-plane rigid bar between two vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    /// First vertex index.
    pub a: usize,
    /// Second vertex index.
    pub b: usize,
    /// Rest length measured in the flat crease pattern (m).
    pub rest_length: f64,
}

/// A four-vertex dihedral hinge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hinge {
    /// Wing vertex to the left of the directed axis `axis_start → axis_end`.
    pub wing_left: usize,
    /// Start vertex of the hinge axis.
    pub axis_start: usize,
    /// End vertex of the hinge axis.
    pub axis_end: usize,
    /// Wing vertex to the right of the directed axis.
    pub wing_right: usize,
    /// Fold line this hinge represents (`None` for a panel-flatness hinge).
    pub fold_line: Option<usize>,
}

impl Hinge {
    fn indices(&self) -> [usize; 4] {
        [
            self.wing_left,
            self.axis_start,
            self.axis_end,
            self.wing_right,
        ]
    }

    /// Current signed fold angle of the hinge (see [`signed_fold_angle`]).
    pub fn angle(&self, vertices: &[OrigamiVertex]) -> Option<f64> {
        let p = self.positions(vertices)?;
        Some(signed_fold_angle(p[0], p[1], p[2], p[3]))
    }

    fn positions(&self, vertices: &[OrigamiVertex]) -> Option<[[f64; 3]; 4]> {
        let idx = self.indices();
        let mut out = [[0.0_f64; 3]; 4];
        for (slot, &vi) in out.iter_mut().zip(idx.iter()) {
            *slot = vertices.get(vi)?.position;
        }
        Some(out)
    }
}

/// Bar-and-hinge model built from the topology of a [`CreasePattern`].
#[derive(Debug, Clone, Default)]
pub struct BarHingeModel {
    /// Bars (panel edges and diagonals).
    pub bars: Vec<Bar>,
    /// Panel-flatness hinges (rest angle 0, rigid).
    pub panel_hinges: Vec<Hinge>,
    /// Fold hinges, one per fold line shared by two panels.
    pub fold_hinges: Vec<Hinge>,
    /// For each fold line, the index into `fold_hinges` (if it has a hinge).
    pub fold_line_hinge: Vec<Option<usize>>,
    vertex_count: usize,
    fold_line_count: usize,
    panel_count: usize,
}

/// Flat-pattern 2-D cross product `(b − a) × (p − a)`.
fn flat_side(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Orient a hinge so that `wing_left` lies to the left of the axis in the
/// flat pattern.  Returns `None` if both wings lie on the same side or on
/// the axis.
fn oriented_hinge(
    vertices: &[OrigamiVertex],
    axis: (usize, usize),
    wings: (usize, usize),
    fold_line: Option<usize>,
) -> Option<Hinge> {
    let pa = vertices.get(axis.0)?.flat_pos;
    let pb = vertices.get(axis.1)?.flat_pos;
    let s0 = flat_side(pa, pb, vertices.get(wings.0)?.flat_pos);
    let s1 = flat_side(pa, pb, vertices.get(wings.1)?.flat_pos);
    let (left, right) = if s0 > 0.0_f64 && s1 < 0.0_f64 {
        (wings.0, wings.1)
    } else if s0 < 0.0_f64 && s1 > 0.0_f64 {
        (wings.1, wings.0)
    } else {
        return None;
    };
    Some(Hinge {
        wing_left: left,
        axis_start: axis.0,
        axis_end: axis.1,
        wing_right: right,
        fold_line,
    })
}

/// Whether `a` and `b` are consecutive (cyclically) in `panel`.
fn panel_has_edge(panel: &[usize], a: usize, b: usize) -> bool {
    let n = panel.len();
    (0..n).any(|i| {
        let p = panel[i];
        let q = panel[(i + 1) % n];
        (p == a && q == b) || (p == b && q == a)
    })
}

/// Wing vertex of `panel` for the hinge `(a, b)`: the panel vertex farthest
/// from the hinge line in the flat pattern.
fn panel_wing(vertices: &[OrigamiVertex], panel: &[usize], a: usize, b: usize) -> Option<usize> {
    let pa = vertices.get(a)?.flat_pos;
    let pb = vertices.get(b)?.flat_pos;
    let mut best: Option<(usize, f64)> = None;
    for &v in panel {
        if v == a || v == b {
            continue;
        }
        let d = flat_side(pa, pb, vertices.get(v)?.flat_pos).abs();
        if best.is_none_or(|(_, bd)| d > bd) {
            best = Some((v, d));
        }
    }
    best.filter(|&(_, d)| d > 1.0e-14_f64).map(|(v, _)| v)
}

fn flat_distance(vertices: &[OrigamiVertex], a: usize, b: usize) -> Option<f64> {
    let pa = vertices.get(a)?.flat_pos;
    let pb = vertices.get(b)?.flat_pos;
    Some(((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt())
}

impl BarHingeModel {
    /// Build the bar-and-hinge model of `pattern`.
    ///
    /// Panels with fewer than three valid, distinct vertices are ignored.
    /// Fold lines that are not shared by exactly two panels (boundary edges)
    /// receive no hinge; their dihedral angle is undefined.
    pub fn from_pattern(pattern: &CreasePattern) -> Self {
        let vertices = &pattern.vertices;
        let nv = vertices.len();
        let valid_panels: Vec<&Vec<usize>> = pattern
            .panels
            .iter()
            .filter(|p| {
                let distinct: HashSet<usize> = p.iter().copied().collect();
                p.len() >= 3 && distinct.len() == p.len() && p.iter().all(|&v| v < nv)
            })
            .collect();

        let mut bar_keys: HashSet<(usize, usize)> = HashSet::new();
        let mut bars = Vec::new();
        let mut panel_hinges = Vec::new();
        for panel in &valid_panels {
            let n = panel.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let (a, b) = (panel[i].min(panel[j]), panel[i].max(panel[j]));
                    if bar_keys.insert((a, b))
                        && let Some(len) = flat_distance(vertices, a, b)
                    {
                        bars.push(Bar {
                            a,
                            b,
                            rest_length: len,
                        });
                    }
                }
            }
            // Fan triangulation from panel[0]; each interior diagonal
            // (panel[0], panel[t]) is a flatness hinge.
            for t in 2..n.saturating_sub(1) {
                if let Some(h) = oriented_hinge(
                    vertices,
                    (panel[0], panel[t]),
                    (panel[t - 1], panel[t + 1]),
                    None,
                ) {
                    panel_hinges.push(h);
                }
            }
        }

        let mut fold_hinges = Vec::new();
        let mut fold_line_hinge = Vec::with_capacity(pattern.fold_lines.len());
        for (fi, fl) in pattern.fold_lines.iter().enumerate() {
            let (a, b) = (fl.vertex_a, fl.vertex_b);
            let adjacent: Vec<&&Vec<usize>> = valid_panels
                .iter()
                .filter(|p| panel_has_edge(p, a, b))
                .collect();
            let hinge = if adjacent.len() == 2 {
                match (
                    panel_wing(vertices, adjacent[0], a, b),
                    panel_wing(vertices, adjacent[1], a, b),
                ) {
                    (Some(w0), Some(w1)) => oriented_hinge(vertices, (a, b), (w0, w1), Some(fi)),
                    _ => None,
                }
            } else {
                None
            };
            match hinge {
                Some(h) => {
                    fold_line_hinge.push(Some(fold_hinges.len()));
                    fold_hinges.push(h);
                }
                None => fold_line_hinge.push(None),
            }
        }

        Self {
            bars,
            panel_hinges,
            fold_hinges,
            fold_line_hinge,
            vertex_count: nv,
            fold_line_count: pattern.fold_lines.len(),
            panel_count: pattern.panels.len(),
        }
    }

    /// Whether this model was built from a pattern with the same topology
    /// sizes as `pattern` (vertex, fold-line and panel counts).
    pub fn matches(&self, pattern: &CreasePattern) -> bool {
        self.vertex_count == pattern.vertices.len()
            && self.fold_line_count == pattern.fold_lines.len()
            && self.panel_count == pattern.panels.len()
    }

    /// Signed geometric fold angle of fold line `fold_line`, if it has a hinge.
    pub fn fold_line_angle(&self, vertices: &[OrigamiVertex], fold_line: usize) -> Option<f64> {
        let hi = (*self.fold_line_hinge.get(fold_line)?)?;
        self.fold_hinges.get(hi)?.angle(vertices)
    }

    /// Largest absolute bar-length error `| |x_a − x_b| − L₀ |` (m).
    pub fn max_bar_error(&self, vertices: &[OrigamiVertex]) -> f64 {
        self.bars
            .iter()
            .filter_map(|bar| {
                let pa = vertices.get(bar.a)?.position;
                let pb = vertices.get(bar.b)?.position;
                Some((vec3_len(vec3_sub(pb, pa)) - bar.rest_length).abs())
            })
            .fold(0.0_f64, f64::max)
    }

    /// One Gauss–Seidel sweep of rigid bar projections.
    pub(crate) fn project_bars(&self, vertices: &mut [OrigamiVertex]) {
        for bar in &self.bars {
            project_bar(vertices, bar);
        }
    }

    /// Repeat rigid bar sweeps until every bar error is below `tolerance`
    /// or `max_sweeps` sweeps have been performed.
    pub(crate) fn enforce_rigidity(
        &self,
        vertices: &mut [OrigamiVertex],
        tolerance: f64,
        max_sweeps: usize,
    ) {
        for _ in 0..max_sweeps {
            if self.max_bar_error(vertices) <= tolerance {
                break;
            }
            self.project_panel_hinges(vertices);
            self.project_bars(vertices);
        }
    }

    /// One sweep of rigid panel-flatness projections.
    pub(crate) fn project_panel_hinges(&self, vertices: &mut [OrigamiVertex]) {
        for h in &self.panel_hinges {
            let mut lambda = 0.0_f64;
            project_hinge(vertices, h, 0.0_f64, 0.0_f64, 0.0_f64, &mut lambda);
        }
    }
}

/// Inverse-mass weight of a vertex in the projections (0 when pinned).
fn vertex_weight(v: &OrigamiVertex) -> f64 {
    if v.is_pinned {
        0.0_f64
    } else {
        v.inv_mass.max(0.0_f64)
    }
}

fn project_bar(vertices: &mut [OrigamiVertex], bar: &Bar) {
    let (Some(va), Some(vb)) = (vertices.get(bar.a), vertices.get(bar.b)) else {
        return;
    };
    let wa = vertex_weight(va);
    let wb = vertex_weight(vb);
    let w_sum = wa + wb;
    if w_sum <= 0.0_f64 {
        return;
    }
    let d = vec3_sub(va.position, vb.position);
    let len = vec3_len(d);
    if len < 1.0e-15_f64 {
        return;
    }
    let c = len - bar.rest_length;
    let n = [d[0] / len, d[1] / len, d[2] / len];
    let s = c / w_sum;
    if let Some(v) = vertices.get_mut(bar.a) {
        for (p, ni) in v.position.iter_mut().zip(n.iter()) {
            *p -= wa * s * ni;
        }
    }
    if let Some(v) = vertices.get_mut(bar.b) {
        for (p, ni) in v.position.iter_mut().zip(n.iter()) {
            *p += wb * s * ni;
        }
    }
}

/// XPBD projection of a hinge towards `target` (radians).
///
/// * `compliance_tilde` — XPBD compliance divided by `dt²` (0 = rigid).
/// * `regularisation`   — extra term added to the denominator (damped
///   least-squares step; 0 = full Gauss–Newton step).
/// * `lambda`           — accumulated Lagrange multiplier for this step.
pub(crate) fn project_hinge(
    vertices: &mut [OrigamiVertex],
    hinge: &Hinge,
    target: f64,
    compliance_tilde: f64,
    regularisation: f64,
    lambda: &mut f64,
) {
    let idx = hinge.indices();
    let Some(p) = hinge.positions(vertices) else {
        return;
    };
    let Some(grad) = signed_fold_angle_gradient(p[0], p[1], p[2], p[3]) else {
        return;
    };
    let psi = signed_fold_angle(p[0], p[1], p[2], p[3]);
    let c = wrap_angle(psi - target);
    let mut denom = compliance_tilde + regularisation;
    let mut weights = [0.0_f64; 4];
    for (q, &vi) in idx.iter().enumerate() {
        let w = vertices.get(vi).map_or(0.0_f64, vertex_weight);
        weights[q] = w;
        denom += w * vec3_dot(grad[q], grad[q]);
    }
    if denom < 1.0e-30_f64 || weights.iter().all(|&w| w <= 0.0_f64) {
        return;
    }
    let d_lambda = (-c - compliance_tilde * *lambda) / denom;
    *lambda += d_lambda;
    for (q, &vi) in idx.iter().enumerate() {
        if weights[q] <= 0.0_f64 {
            continue;
        }
        if let Some(v) = vertices.get_mut(vi) {
            for (pos, g) in v.position.iter_mut().zip(grad[q].iter()) {
                *pos += weights[q] * g * d_lambda;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wings(alpha: f64) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        // Hinge along +x, left wing at +y, right wing at −y; both wings
        // rotate up (towards +z) by alpha/2 → valley fold of alpha.
        let h = alpha * 0.5_f64;
        (
            [0.5_f64, h.cos(), h.sin()],
            [0.0_f64; 3],
            [1.0_f64, 0.0_f64, 0.0_f64],
            [0.5_f64, -h.cos(), h.sin()],
        )
    }

    #[test]
    fn signed_angle_flat_is_zero_and_valley_positive() {
        let (i, j, k, l) = wings(0.0_f64);
        assert!(signed_fold_angle(i, j, k, l).abs() < 1e-12_f64);
        for &alpha in &[0.3_f64, 1.2_f64, 2.9_f64, -0.7_f64, -2.5_f64] {
            let (i, j, k, l) = wings(alpha);
            let psi = signed_fold_angle(i, j, k, l);
            assert!((psi - alpha).abs() < 1e-12_f64, "alpha={alpha} psi={psi}");
        }
    }

    #[test]
    fn signed_angle_gradient_matches_finite_difference() {
        for &alpha in &[0.4_f64, -1.1_f64, 2.2_f64] {
            let (i, j, k, l) = wings(alpha);
            // Perturb to a generic configuration.
            let pts = [
                [i[0] + 0.13_f64, i[1], i[2] - 0.05_f64],
                [j[0], j[1] + 0.07_f64, j[2]],
                [k[0] + 0.2_f64, k[1], k[2] + 0.03_f64],
                [l[0] - 0.11_f64, l[1], l[2]],
            ];
            let g = signed_fold_angle_gradient(pts[0], pts[1], pts[2], pts[3])
                .expect("non-degenerate hinge");
            let h = 1e-6_f64;
            for q in 0..4 {
                for d in 0..3 {
                    let mut plus = pts;
                    let mut minus = pts;
                    plus[q][d] += h;
                    minus[q][d] -= h;
                    let fd = (signed_fold_angle(plus[0], plus[1], plus[2], plus[3])
                        - signed_fold_angle(minus[0], minus[1], minus[2], minus[3]))
                        / (2.0_f64 * h);
                    assert!(
                        (fd - g[q][d]).abs() < 1e-6_f64,
                        "q={q} d={d} fd={fd} analytic={}",
                        g[q][d]
                    );
                }
            }
        }
    }

    #[test]
    fn wrap_angle_range() {
        assert!((wrap_angle(3.0_f64 * PI) - PI).abs() < 1e-12_f64);
        assert!((wrap_angle(-3.0_f64 * PI) - PI).abs() < 1e-12_f64);
        assert!((wrap_angle(0.5_f64) - 0.5_f64).abs() < 1e-15_f64);
        assert!((wrap_angle(2.0_f64 * PI - 0.1_f64) + 0.1_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn model_from_two_triangles_has_one_fold_hinge() {
        let mut p = CreasePattern::new();
        p.add_vertex(
            [0.0_f64, 0.0_f64],
            super::super::MountainValley::Boundary,
            1.0_f64,
        );
        p.add_vertex(
            [1.0_f64, 0.0_f64],
            super::super::MountainValley::Boundary,
            1.0_f64,
        );
        p.add_vertex(
            [0.5_f64, 1.0_f64],
            super::super::MountainValley::Boundary,
            1.0_f64,
        );
        p.add_vertex(
            [0.5_f64, -1.0_f64],
            super::super::MountainValley::Boundary,
            1.0_f64,
        );
        p.add_fold_line(0, 1, super::super::MountainValley::Valley, 1.0_f64);
        p.add_panel(vec![0, 1, 2]);
        p.add_panel(vec![1, 0, 3]);
        let model = BarHingeModel::from_pattern(&p);
        assert_eq!(model.bars.len(), 5);
        assert_eq!(model.fold_hinges.len(), 1);
        assert!(model.panel_hinges.is_empty());
        let h = model.fold_hinges[0];
        assert_eq!(h.wing_left, 2);
        assert_eq!(h.wing_right, 3);
        assert!(model.matches(&p));
    }

    #[test]
    fn quad_panel_gets_flatness_hinge_and_both_diagonals() {
        let mut p = CreasePattern::new();
        for pos in [[0.0_f64, 0.0_f64], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
            p.add_vertex(pos, super::super::MountainValley::Boundary, 1.0_f64);
        }
        p.add_panel(vec![0, 1, 2, 3]);
        let model = BarHingeModel::from_pattern(&p);
        assert_eq!(model.bars.len(), 6);
        assert_eq!(model.panel_hinges.len(), 1);
    }
}
