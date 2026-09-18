//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::f64::consts::PI;

use super::types::{BranchingDecision, Crack, CrackSegment, LevelSet, MixedModeSif, SubTriangle};

/// Heaviside enrichment: returns +1 if `point` is on the left side of the
/// crack and -1 if on the right side.
///
/// The side is decided by the signed distance to the whole crack polyline
/// ([`signed_distance_to_crack`]), so the two faces of a kinked crack are
/// separated correctly.  A point exactly on the crack counts as the left side.
pub fn heaviside_enrichment(point: [f64; 3], crack: &Crack) -> f64 {
    if signed_distance_to_crack(point, crack) >= 0.0 {
        1.0
    } else {
        -1.0
    }
}
/// Four crack-tip enrichment functions evaluated at local polar coordinates
/// `(r, theta)` relative to the crack tip.
pub fn crack_tip_enrichment(r: f64, theta: f64) -> [f64; 4] {
    let sr = r.max(0.0).sqrt();
    let ht = theta / 2.0;
    [
        sr * ht.sin(),
        sr * ht.cos(),
        sr * ht.sin() * theta.sin(),
        sr * ht.cos() * theta.sin(),
    ]
}
/// Gradients of the crack-tip enrichment functions in polar coordinates.
///
/// Returns `d/dr [F1..F4]` and `d/dtheta [F1..F4]` as two arrays.
pub fn crack_tip_enrichment_gradients(r: f64, theta: f64) -> ([f64; 4], [f64; 4]) {
    let r_safe = r.max(1e-30);
    let sr = r_safe.sqrt();
    let inv_sr = 1.0 / (2.0 * sr);
    let ht = theta / 2.0;
    let dr = [
        inv_sr * ht.sin(),
        inv_sr * ht.cos(),
        inv_sr * ht.sin() * theta.sin(),
        inv_sr * ht.cos() * theta.sin(),
    ];
    let dt = [
        sr * 0.5 * ht.cos(),
        -sr * 0.5 * ht.sin(),
        sr * (0.5 * ht.cos() * theta.sin() + ht.sin() * theta.cos()),
        sr * (-0.5 * ht.sin() * theta.sin() + ht.cos() * theta.cos()),
    ];
    (dr, dt)
}
/// Signed distance from `point` to the infinite line through `line_start`
/// and `line_end` (positive on the left / counter-clockwise side).
pub fn signed_distance_to_line(point: [f64; 3], line_start: [f64; 3], line_end: [f64; 3]) -> f64 {
    let dx = line_end[0] - line_start[0];
    let dy = line_end[1] - line_start[1];
    let _dz = line_end[2] - line_start[2];
    let line_len = (dx * dx + dy * dy + _dz * _dz).sqrt().max(f64::EPSILON);
    let px = point[0] - line_start[0];
    let py = point[1] - line_start[1];
    let cross_z = dx * py - dy * px;
    cross_z / line_len
}
/// Squared planar length below which a crack segment has no usable direction.
const DEGENERATE_SEGMENT_SQ: f64 = 1e-30;

/// Squared length of a segment projected onto the x-y plane.
fn planar_length_sq(seg: &CrackSegment) -> f64 {
    let dx = seg.end[0] - seg.start[0];
    let dy = seg.end[1] - seg.start[1];
    dx * dx + dy * dy
}
/// Unit left normal of a segment in the x-y plane, or `None` for a
/// (numerically) zero-length segment.
fn segment_left_normal(seg: &CrackSegment) -> Option<[f64; 2]> {
    let len_sq = planar_length_sq(seg);
    if len_sq <= DEGENERATE_SEGMENT_SQ {
        return None;
    }
    let len = len_sq.sqrt();
    Some([
        -(seg.end[1] - seg.start[1]) / len,
        (seg.end[0] - seg.start[0]) / len,
    ])
}
/// Whether `current` starts where `previous` ends (in the x-y plane, up to a
/// relative tolerance of 1e-9 of the longer segment), i.e. both meet at a
/// vertex of the same polyline.
fn segments_joined(previous: &CrackSegment, current: &CrackSegment) -> bool {
    let dx = previous.end[0] - current.start[0];
    let dy = previous.end[1] - current.start[1];
    let scale_sq = planar_length_sq(previous).max(planar_length_sq(current));
    dx * dx + dy * dy <= 1e-18 * scale_sq
}
/// Closest point of the crack polyline to `point` in the x-y plane, skipping
/// zero-length segments.
///
/// Returns `(distance, segment_index, parameter_t, left_normal)` of the nearest
/// segment, with `t` clamped to \[0, 1\]; ties keep the earlier segment.
fn nearest_segment_planar(
    point: [f64; 3],
    segments: &[CrackSegment],
) -> Option<(f64, usize, f64, [f64; 2])> {
    let mut best: Option<(f64, usize, f64, [f64; 2])> = None;
    for (idx, seg) in segments.iter().enumerate() {
        let Some(normal) = segment_left_normal(seg) else {
            continue;
        };
        let abx = seg.end[0] - seg.start[0];
        let aby = seg.end[1] - seg.start[1];
        let apx = point[0] - seg.start[0];
        let apy = point[1] - seg.start[1];
        let t = ((apx * abx + apy * aby) / (abx * abx + aby * aby)).clamp(0.0, 1.0);
        let dist = (apx - t * abx).hypot(apy - t * aby);
        let closer = match best {
            Some((best_dist, ..)) => dist < best_dist,
            None => true,
        };
        if closer {
            best = Some((dist, idx, t, normal));
        }
    }
    best
}
/// Left normal of the next non-degenerate segment after `idx` if it continues
/// the polyline at the end vertex of segment `idx`.
fn next_joined_normal(segments: &[CrackSegment], idx: usize) -> Option<[f64; 2]> {
    let current = segments.get(idx)?;
    let next = segments
        .get(idx + 1..)?
        .iter()
        .find(|seg| segment_left_normal(seg).is_some())?;
    if segments_joined(current, next) {
        segment_left_normal(next)
    } else {
        None
    }
}
/// Left normal of the last non-degenerate segment before `idx` if it continues
/// into the start vertex of segment `idx`.
fn previous_joined_normal(segments: &[CrackSegment], idx: usize) -> Option<[f64; 2]> {
    let current = segments.get(idx)?;
    let previous = segments
        .get(..idx)?
        .iter()
        .rev()
        .find(|seg| segment_left_normal(seg).is_some())?;
    if segments_joined(previous, current) {
        segment_left_normal(previous)
    } else {
        None
    }
}
/// Signed distance to the line through the crack tip along its growth
/// direction (used for cracks without any usable segment).
fn signed_distance_to_growth_line(point: [f64; 3], crack: &Crack) -> f64 {
    let dir = crack.growth_direction;
    let len = dir[0].hypot(dir[1]);
    if len <= f64::EPSILON {
        return 0.0;
    }
    (dir[0] * (point[1] - crack.tip[1]) - dir[1] * (point[0] - crack.tip[0])) / len
}
/// Signed distance from `point` to the crack polyline (the level set `phi`),
/// positive on the left of the crack direction and negative on the right.
///
/// The distance and the sides are evaluated in the x-y plane, like
/// [`signed_distance_to_line`], and the whole polyline is used (not just the
/// last segment):
///
/// * In the interior of a segment the value is the signed perpendicular
///   distance to that segment.
/// * Where the nearest point is a vertex shared by two segments, the magnitude is
///   the distance to the vertex and the sign comes from the sum of the two unit
///   left normals (the bisector normal), which is the same on both sides of the
///   bisector and therefore keeps the two crack faces apart at a kink.  If the
///   two segments fold back onto each other the sign of the nearest segment is
///   used.
/// * Where the nearest point is the crack tip or the crack start, the value is
///   the signed distance to the infinite extension of that end segment, which is
///   the usual extension of the XFEM level set `phi` beyond the crack ends.
///
/// Zero-length segments are ignored.  A crack without any usable segment is
/// replaced by the line through the tip along `growth_direction`.
pub fn signed_distance_to_crack(point: [f64; 3], crack: &Crack) -> f64 {
    let segments = &crack.segments;
    let Some((dist, idx, t, normal)) = nearest_segment_planar(point, segments) else {
        return signed_distance_to_growth_line(point, crack);
    };
    let Some(seg) = segments.get(idx) else {
        return signed_distance_to_growth_line(point, crack);
    };
    let perpendicular =
        (point[0] - seg.start[0]) * normal[0] + (point[1] - seg.start[1]) * normal[1];
    let vertex_neighbour = if t >= 1.0 {
        next_joined_normal(segments, idx).map(|other| (other, seg.end))
    } else if t <= 0.0 {
        previous_joined_normal(segments, idx).map(|other| (other, seg.start))
    } else {
        None
    };
    let Some((other, vertex)) = vertex_neighbour else {
        return perpendicular;
    };
    let bisector = [normal[0] + other[0], normal[1] + other[1]];
    if bisector[0].hypot(bisector[1]) <= 1e-12 {
        return perpendicular;
    }
    let side = (point[0] - vertex[0]) * bisector[0] + (point[1] - vertex[1]) * bisector[1];
    if side >= 0.0 { dist } else { -dist }
}
/// Compute polar coordinates `(r, theta)` of `point` relative to the crack
/// `tip`, where the crack grows in `crack_dir`.
pub fn polar_coords_from_tip(point: [f64; 3], tip: [f64; 3], crack_dir: [f64; 3]) -> (f64, f64) {
    let dx = point[0] - tip[0];
    let dy = point[1] - tip[1];
    let dz = point[2] - tip[2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < f64::EPSILON {
        return (0.0, 0.0);
    }
    let cdx = crack_dir[0];
    let cdy = crack_dir[1];
    let cdz = crack_dir[2];
    let clen = (cdx * cdx + cdy * cdy + cdz * cdz).sqrt().max(f64::EPSILON);
    let cx = cdx / clen;
    let cy = cdy / clen;
    let cz = cdz / clen;
    let ux = dx / r;
    let uy = dy / r;
    let uz = dz / r;
    let cos_t = (cx * ux + cy * uy + cz * uz).clamp(-1.0, 1.0);
    let cross_z = cx * uy - cy * ux;
    let _cross_y = cz * ux - cx * uz;
    let _cross_x = cy * uz - cz * uy;
    let theta = cross_z.atan2(cos_t);
    (r, theta)
}
/// Compute the nearest point on a crack to a given point.
///
/// Returns (distance, segment_index, parameter_t) where t in \[0,1\]
/// parameterizes the nearest point on the closest segment.
pub fn nearest_point_on_crack(point: [f64; 3], crack: &Crack) -> (f64, usize, f64) {
    let mut best_dist = f64::MAX;
    let mut best_seg = 0;
    let mut best_t = 0.0;
    for (idx, seg) in crack.segments.iter().enumerate() {
        let ab = [
            seg.end[0] - seg.start[0],
            seg.end[1] - seg.start[1],
            seg.end[2] - seg.start[2],
        ];
        let ap = [
            point[0] - seg.start[0],
            point[1] - seg.start[1],
            point[2] - seg.start[2],
        ];
        let ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let t = if ab_sq > 1e-30 {
            let dot = ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2];
            (dot / ab_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let nearest = [
            seg.start[0] + t * ab[0],
            seg.start[1] + t * ab[1],
            seg.start[2] + t * ab[2],
        ];
        let diff = [
            point[0] - nearest[0],
            point[1] - nearest[1],
            point[2] - nearest[2],
        ];
        let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
        if dist < best_dist {
            best_dist = dist;
            best_seg = idx;
            best_t = t;
        }
    }
    (best_dist, best_seg, best_t)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::xfem::*;
    #[test]
    fn crack_segment_length() {
        let seg = CrackSegment {
            start: [0.0, 0.0, 0.0],
            end: [3.0, 4.0, 0.0],
        };
        assert!((seg.length() - 5.0).abs() < 1e-12);
    }
    #[test]
    fn crack_propagate_tip() {
        let mut crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        crack.propagate(2.0, [1.0, 0.0, 0.0]);
        assert!((crack.tip[0] - 3.0).abs() < 1e-12);
        assert!(crack.tip[1].abs() < 1e-12);
        assert_eq!(crack.segments.len(), 2);
        assert!((crack.length() - 3.0).abs() < 1e-12);
    }
    #[test]
    fn crack_tip_enrichment_at_r1_theta0() {
        let f = crack_tip_enrichment(1.0, 0.0);
        assert!(f[0].abs() < 1e-12);
        assert!((f[1] - 1.0).abs() < 1e-12);
        assert!(f[2].abs() < 1e-12);
        assert!(f[3].abs() < 1e-12);
    }
    #[test]
    fn heaviside_enrichment_opposite_sides() {
        let crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let above = heaviside_enrichment([0.5, 1.0, 0.0], &crack);
        let below = heaviside_enrichment([0.5, -1.0, 0.0], &crack);
        assert!(above > 0.0);
        assert!(below < 0.0);
        assert!((above + below).abs() < 1e-12);
    }
    #[test]
    fn polar_coords_from_tip_directly_ahead() {
        let tip = [1.0, 0.0, 0.0];
        let crack_dir = [1.0, 0.0, 0.0];
        let point = [2.0, 0.0, 0.0];
        let (r, theta) = polar_coords_from_tip(point, tip, crack_dir);
        assert!((r - 1.0).abs() < 1e-12);
        assert!(theta.abs() < 1e-12);
    }
    #[test]
    fn signed_distance_to_line_basic() {
        let sd = signed_distance_to_line([0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(sd > 0.0);
        let sd2 = signed_distance_to_line([0.0, -2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(sd2 < 0.0);
    }
    #[test]
    fn plastic_zone_size_positive() {
        let sif = SifCalculator::new(200e9, 0.3);
        let rp = sif.plastic_zone_size(50e6, 250e6);
        assert!(rp > 0.0);
    }
    #[test]
    fn ki_from_displacement_plane_strain_matches_closed_form() {
        let (e, nu) = (200e9, 0.3);
        let sif = SifCalculator::new(e, nu);
        let (u_y, r) = (2.0e-6, 1.0e-3);
        let ki = sif.ki_from_displacement(u_y, r, true);
        let expected = e / (4.0 * (1.0 - nu * nu)) * u_y * (2.0 * PI / r).sqrt();
        assert!(
            (ki / expected - 1.0).abs() < 1e-12,
            "K_I = {ki:e} vs {expected:e}"
        );
    }
    #[test]
    fn tip_distance_check() {
        let crack = Crack::new([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        let d = crack.tip_distance([3.0, 4.0, 5.0]);
        assert!((d - 5.0).abs() < 1e-12);
    }
    #[test]
    fn test_crack_tip_enrichment_gradients() {
        let (dr, dt) = crack_tip_enrichment_gradients(1.0, 0.0);
        assert!((dr[1] - 0.5).abs() < 1e-12, "dF2/dr = {}", dr[1]);
        assert!((dt[0] - 0.5).abs() < 1e-12, "dF1/dtheta = {}", dt[0]);
    }
    #[test]
    fn test_level_set_from_crack() {
        let crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let nodes = vec![
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ];
        let ls = LevelSet::from_crack(&nodes, &crack);
        assert!(ls.phi[0] > 0.0, "above should have positive phi");
        assert!(ls.phi[1] < 0.0, "below should have negative phi");
        assert!(ls.psi[0] < 0.0, "behind tip should have negative psi");
        assert!(ls.psi[2] > 0.0, "ahead of tip should have positive psi");
    }
    #[test]
    fn test_level_set_needs_tip_enrichment() {
        let crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let nodes = vec![[1.0, 0.1, 0.0], [5.0, 0.0, 0.0]];
        let ls = LevelSet::from_crack(&nodes, &crack);
        assert!(
            ls.needs_tip_enrichment(0, 0.5),
            "near tip should need enrichment"
        );
        assert!(
            !ls.needs_tip_enrichment(1, 0.5),
            "far from tip should not need enrichment"
        );
    }
    #[test]
    fn test_level_set_zero_crossing() {
        let crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let nodes = vec![[0.5, 1.0, 0.0], [0.5, -1.0, 0.0]];
        let ls = LevelSet::from_crack(&nodes, &crack);
        let xi = ls.zero_crossing(0, 1);
        assert!(xi.is_some(), "should have a zero crossing");
        let xi_val = xi.unwrap();
        assert!(
            xi_val > 0.0 && xi_val < 1.0,
            "xi = {xi_val} should be in (0,1)"
        );
    }
    #[test]
    fn test_level_set_no_crossing() {
        let crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let nodes = vec![[0.5, 1.0, 0.0], [0.5, 2.0, 0.0]];
        let ls = LevelSet::from_crack(&nodes, &crack);
        assert!(
            ls.zero_crossing(0, 1).is_none(),
            "same side should have no crossing"
        );
    }
    #[test]
    fn test_level_set_interpolate() {
        let ls = LevelSet {
            phi: vec![1.0, -1.0],
            psi: vec![0.0; 2],
        };
        let val = ls.interpolate_phi(0, 1, 0.5);
        assert!((val).abs() < 1e-12, "midpoint should be zero");
    }
    #[test]
    fn test_blending_ramp() {
        let shape_fns = vec![0.25, 0.25, 0.25, 0.25];
        let enriched = vec![true, true, false, false];
        let r = BlendingFunction::ramp(&shape_fns, &enriched);
        assert!((r - 0.5).abs() < 1e-12, "ramp = {r}");
    }
    #[test]
    fn test_blending_ramp_all_enriched() {
        let shape_fns = vec![0.25, 0.25, 0.25, 0.25];
        let enriched = vec![true, true, true, true];
        let r = BlendingFunction::ramp(&shape_fns, &enriched);
        assert!((r - 1.0).abs() < 1e-12, "all enriched should give ramp = 1");
    }
    #[test]
    fn test_shifted_enrichment() {
        let f_values = vec![1.5, 2.0, 0.5];
        let f_at_nodes = vec![1.0, 1.0, 1.0];
        let shifted = BlendingFunction::shifted_enrichment(&f_values, &f_at_nodes);
        assert!((shifted[0] - 0.5).abs() < 1e-12);
        assert!((shifted[1] - 1.0).abs() < 1e-12);
        assert!((shifted[2] - (-0.5)).abs() < 1e-12);
    }
    #[test]
    fn test_xfem_sif_from_interaction_integral() {
        let extractor = XfemSifExtractor::new(200e9, 0.3);
        let (ki, kii) = extractor.sif_from_interaction_integral(1e-3, 5e-4);
        assert!(ki > 0.0);
        assert!(kii > 0.0);
        let e_prime = 200e9 / (1.0 - 0.3 * 0.3);
        assert!((ki - 1e-3 * e_prime / 2.0).abs() < 1e-3);
    }
    #[test]
    fn test_xfem_ki_from_cod() {
        let extractor = XfemSifExtractor::new(200e9, 0.3);
        let ki = extractor.ki_from_cod(1e-6, 0.001);
        assert!(ki > 0.0);
        assert!(ki.is_finite());
    }
    #[test]
    fn test_xfem_j_from_sif() {
        let extractor = XfemSifExtractor::new(200e9, 0.3);
        let j = extractor.j_from_sif(30e6, 0.0);
        let e_prime = 200e9 / (1.0 - 0.3 * 0.3);
        let expected = 30e6 * 30e6 / e_prime;
        assert!((j - expected).abs() / expected < 1e-10);
    }
    #[test]
    fn test_nearest_point_on_crack() {
        let crack = Crack::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        let point = [1.0, 1.0, 0.0];
        let (dist, seg_idx, t) = nearest_point_on_crack(point, &crack);
        assert!((dist - 1.0).abs() < 1e-12, "distance = {dist}");
        assert_eq!(seg_idx, 0);
        assert!((t - 0.5).abs() < 1e-12, "t = {t}");
    }
    #[test]
    fn test_nearest_point_on_crack_multi_segment() {
        let mut crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        crack.propagate(1.0, [0.0, 1.0, 0.0]);
        let point = [1.0, 0.5, 1.0];
        let (dist, seg_idx, _t) = nearest_point_on_crack(point, &crack);
        assert_eq!(seg_idx, 1, "should be closest to segment 1");
        assert!((dist - 1.0).abs() < 1e-12, "distance = {dist}");
    }
    #[test]
    fn test_level_set_update_after_propagation() {
        let crack = Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let nodes = vec![[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let mut ls = LevelSet::from_crack(&nodes, &crack);
        assert!(ls.psi[0] > 0.0);
        ls.update_after_propagation(&nodes, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(
            (ls.psi[0]).abs() < 1e-12,
            "node at new tip should have psi=0"
        );
        assert!(ls.psi[1] < 0.0, "node behind new tip should have psi<0");
    }
}
/// Heaviside-enriched shape function contribution.
///
/// For a node I with standard shape function N_I and Heaviside enrichment:
/// u^h(x) += N_I(x) * (H(x) - H(x_I)) * a_I
///
/// where H is the Heaviside function and a_I is the enrichment DOF.
pub fn heaviside_enriched_shape(
    shape_fn: f64,
    heaviside_at_point: f64,
    heaviside_at_node: f64,
) -> f64 {
    shape_fn * (heaviside_at_point - heaviside_at_node)
}
/// Crack-tip enriched shape functions: N_I * \[F_k(x) - F_k(x_I)\] for k=1..4.
///
/// This is the shifted (partition-of-unity) form for crack-tip enrichment.
pub fn crack_tip_enriched_shapes(
    shape_fn: f64,
    r: f64,
    theta: f64,
    r_node: f64,
    theta_node: f64,
) -> [f64; 4] {
    let f_x = crack_tip_enrichment(r, theta);
    let f_xi = crack_tip_enrichment(r_node, theta_node);
    [
        shape_fn * (f_x[0] - f_xi[0]),
        shape_fn * (f_x[1] - f_xi[1]),
        shape_fn * (f_x[2] - f_xi[2]),
        shape_fn * (f_x[3] - f_xi[3]),
    ]
}
/// Maximum circumferential stress criterion for crack growth direction.
///
/// The crack propagates in the direction θ_c where σ_θθ is maximum.
/// Standard formula (Erdogan–Sih criterion):
///   θ_c = 2 * atan((KI - sqrt(KI² + 8·KII²)) / (4·KII))
///
/// For pure mode I (KII = 0) the crack grows straight ahead (θ_c = 0).
/// For pure mode II (KI = 0) the crack deflects at ≈ ±70.5°.
pub fn max_hoop_stress_angle(ki: f64, kii: f64) -> f64 {
    if kii.abs() < 1e-30 {
        return 0.0;
    }
    if ki.abs() < 1e-30 {
        if kii > 0.0 {
            return -70.5_f64.to_radians();
        }
        return 70.5_f64.to_radians();
    }
    let discriminant = ki * ki + 8.0 * kii * kii;

    2.0 * ((ki - discriminant.sqrt()) / (4.0 * kii)).atan()
}
/// Minimum strain energy density criterion for crack growth.
///
/// S(θ) = a11 KI^2 + 2 a12 KI KII + a22 KII^2
/// where aij depend on elastic constants and angle θ.
///
/// Returns the angle θ that minimizes S (crack propagation direction).
pub fn min_strain_energy_density_angle(ki: f64, kii: f64, nu: f64) -> f64 {
    let n_pts = 360;
    let mut min_s = f64::MAX;
    let mut theta_min = 0.0;
    for i in 0..n_pts {
        let theta = (i as f64 - n_pts as f64 / 2.0) * PI / n_pts as f64;
        let ct = theta.cos();
        let st = theta.sin();
        let ht = (theta / 2.0).cos();
        let ht2 = (theta / 2.0).sin();
        let k = 3.0 - 4.0 * nu;
        let a11 = (1.0 + ct) * (k - ct);
        let a12 = st * (2.0 * ct - k + 1.0);
        let a22 = (k + 1.0) * (1.0 - ct) + (1.0 + ct) * (3.0 * ct - 1.0);
        let s = a11 * ki * ki + 2.0 * a12 * ki * kii + a22 * kii * kii;
        let _ = (ht, ht2);
        if s < min_s {
            min_s = s;
            theta_min = theta;
        }
    }
    theta_min
}
/// Paris law crack growth rate: da/dN = C * (ΔK)^m
///
/// # Arguments
/// * `delta_k` - stress intensity factor range ΔK = K_max - K_min (Pa√m)
/// * `c` - Paris constant
/// * `m` - Paris exponent
///
/// Returns da/dN in meters per cycle.
pub fn paris_law(delta_k: f64, c: f64, m: f64) -> f64 {
    if delta_k <= 0.0 {
        return 0.0;
    }
    c * delta_k.powf(m)
}
/// Walker equation (modified Paris law for R-ratio effects):
/// da/dN = C * (ΔK / (1-R)^(1-n))^m
pub fn walker_law(delta_k: f64, r_ratio: f64, c: f64, m: f64, n_exp: f64) -> f64 {
    if delta_k <= 0.0 {
        return 0.0;
    }
    let r_factor = (1.0 - r_ratio.clamp(-1.0, 0.999)).powf(1.0 - n_exp);
    let k_eff = delta_k / r_factor;
    c * k_eff.powf(m)
}
#[cfg(test)]
mod tests_extended {
    use super::*;
    use crate::xfem::*;
    #[test]
    fn test_heaviside_enriched_shape_shifted() {
        let h_at_node = 1.0;
        let h_at_point = 1.0;
        let n_i = 0.25;
        let v = heaviside_enriched_shape(n_i, h_at_point, h_at_node);
        assert!(
            (v).abs() < 1e-12,
            "shifted enrichment at node should be zero: {v}"
        );
    }
    #[test]
    fn test_heaviside_enriched_shape_across_crack() {
        let n_i = 0.5;
        let v = heaviside_enriched_shape(n_i, -1.0, 1.0);
        assert!((v - (-1.0)).abs() < 1e-12, "expected -1.0, got {v}");
    }
    #[test]
    fn test_crack_tip_enriched_shapes_at_node() {
        let r_node = 1.0;
        let theta_node = 0.0;
        let shapes = crack_tip_enriched_shapes(0.25, r_node, theta_node, r_node, theta_node);
        for (i, &s) in shapes.iter().enumerate() {
            assert!(s.abs() < 1e-12, "shape[{i}] at node = {s}");
        }
    }
    #[test]
    fn test_max_hoop_stress_pure_mode_i() {
        let theta = max_hoop_stress_angle(1.0e6, 0.0);
        assert!(
            theta.abs() < 1e-10,
            "pure mode I should grow straight: theta = {theta}"
        );
    }
    #[test]
    fn test_max_hoop_stress_pure_mode_ii() {
        let theta = max_hoop_stress_angle(0.0, 1.0e6);
        assert!(
            theta.abs() > 0.5,
            "pure mode II should deflect: theta = {theta}"
        );
    }
    #[test]
    fn test_paris_law_zero_delta_k() {
        let da_dn = paris_law(0.0, 1e-12, 3.0);
        assert_eq!(da_dn, 0.0);
    }
    #[test]
    fn test_paris_law_positive_rate() {
        let da_dn = paris_law(10e6, 1e-12, 3.0);
        assert!(da_dn > 0.0, "Paris law should give positive rate: {da_dn}");
    }
    #[test]
    fn test_paris_law_scaling() {
        let m = 3.0;
        let c = 1e-12;
        let r1 = paris_law(10e6, c, m);
        let r2 = paris_law(20e6, c, m);
        let ratio = r2 / r1;
        assert!((ratio - 8.0).abs() < 1e-8, "ratio = {ratio}");
    }
    #[test]
    fn test_walker_law_r_zero_equals_paris() {
        let delta_k = 10e6;
        let c = 1e-12;
        let m = 3.0;
        let paris = paris_law(delta_k, c, m);
        let walker = walker_law(delta_k, 0.0, c, m, 0.5);
        assert!(
            (walker - paris).abs() / paris < 1e-10,
            "Walker @ R=0: {walker}, Paris: {paris}"
        );
    }
    #[test]
    fn test_crack_front_3d_straight() {
        let front = CrackFront3D::new_straight([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5);
        assert_eq!(front.n_points(), 5);
        assert!(
            (front.arc_length() - 1.0).abs() < 1e-10,
            "arc = {}",
            front.arc_length()
        );
    }
    #[test]
    fn test_crack_front_3d_propagate() {
        let mut front = CrackFront3D::new_straight([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 3);
        let da = 0.5;
        front.propagate_uniform(da);
        for p in &front.front_points {
            assert!((p[0] - da).abs() < 1e-12, "x should be {da}, got {}", p[0]);
        }
    }
    #[test]
    fn test_multi_crack_min_distance() {
        let mut system = MultiCrackSystem::new();
        system.add_crack(Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        system.add_crack(Crack::new([5.0, 0.0, 0.0], [6.0, 0.0, 0.0]));
        let d = system.min_tip_distance();
        assert!((d - 5.0).abs() < 1e-10, "min distance = {d}");
    }
    #[test]
    fn test_multi_crack_interaction() {
        let mut system = MultiCrackSystem::new();
        system.add_crack(Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        system.add_crack(Crack::new([1.0, 0.0, 0.0], [1.1, 0.0, 0.0]));
        assert!(
            system.are_interacting(0.5),
            "cracks 0.1 apart should interact within 0.5"
        );
        assert!(
            !system.are_interacting(0.05),
            "cracks 0.1 apart should not interact within 0.05"
        );
    }
    #[test]
    fn test_multi_crack_total_length() {
        let mut system = MultiCrackSystem::new();
        system.add_crack(Crack::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        system.add_crack(Crack::new([5.0, 0.0, 0.0], [8.0, 0.0, 0.0]));
        let total = system.total_length();
        assert!((total - 4.0).abs() < 1e-10, "total length = {total}");
    }
    #[test]
    fn test_min_strain_energy_density_pure_mode_i() {
        let theta = min_strain_energy_density_angle(1.0e6, 0.0, 0.3);
        assert!(theta.abs() < 0.1, "pure mode I: theta = {theta}");
    }
    #[test]
    fn test_multi_crack_default() {
        let sys = MultiCrackSystem::default();
        assert_eq!(sys.cracks.len(), 0);
        assert_eq!(sys.total_length(), 0.0);
        assert!(sys.min_tip_distance().is_infinite());
    }
}
/// Check whether a crack should bifurcate based on the stress intensity factors.
///
/// Uses the bifurcation criterion of Ramulu & Kobayashi (1985): bifurcation
/// occurs when the ratio |K_II / K_I| exceeds a threshold and the crack-tip
/// velocity exceeds a fraction of the Rayleigh wave speed.
///
/// # Arguments
/// * `k1` – Mode-I SIF
/// * `k2` – Mode-II SIF
/// * `crack_speed` – current crack-tip velocity (m/s)
/// * `rayleigh_speed` – Rayleigh wave speed (m/s)
/// * `bifurcation_threshold` – |K_II/K_I| ratio threshold (typically 0.25–0.35)
pub fn crack_bifurcation_criterion(
    k1: f64,
    k2: f64,
    crack_speed: f64,
    rayleigh_speed: f64,
    bifurcation_threshold: f64,
) -> BranchingDecision {
    if k1.abs() < f64::EPSILON {
        return BranchingDecision::Continue;
    }
    let ratio = (k2 / k1).abs();
    let speed_ratio = crack_speed / rayleigh_speed.max(f64::EPSILON);
    if ratio > bifurcation_threshold && speed_ratio > 0.4 {
        let base_angle = (k2 / k1).atan();
        BranchingDecision::Bifurcate {
            angle1: base_angle + PI / 6.0,
            angle2: base_angle - PI / 6.0,
        }
    } else {
        BranchingDecision::Continue
    }
}
/// Compute the bifurcation angle using the maximum hoop stress criterion
/// (Erdogan-Sih, 1963) for a branching crack.
///
/// Returns the angle θ* (radians) at which the circumferential stress σ_θθ
/// is maximum.
pub fn bifurcation_angle_max_hoop(k1: f64, k2: f64) -> f64 {
    if k2.abs() < f64::EPSILON {
        return 0.0;
    }
    let disc = (k1 * k1 + 8.0 * k2 * k2).sqrt();
    2.0 * ((k1 - disc) / (4.0 * k2)).atan()
}
/// Compute mixed-mode SIF via the one-step Modified Crack Closure Integral.
///
/// # Arguments
/// * `delta_a` – virtual crack extension length (m)
/// * `f_close_i` – closure force in mode I direction at crack tip node (N)
/// * `f_close_ii` – closure force in mode II direction (N)
/// * `f_close_iii` – closure force in mode III direction (N)
/// * `displacement_jump_i` – displacement jump (m) behind tip in mode I
/// * `displacement_jump_ii` – displacement jump in mode II
/// * `displacement_jump_iii` – displacement jump in mode III
/// * `e_prime` – plane strain modulus E / (1 - ν²)
/// * `shear_modulus` – μ = E / (2(1+ν))
pub fn mcci_mixed_mode(
    delta_a: f64,
    f_close_i: f64,
    f_close_ii: f64,
    f_close_iii: f64,
    displacement_jump_i: f64,
    displacement_jump_ii: f64,
    displacement_jump_iii: f64,
    e_prime: f64,
    shear_modulus: f64,
) -> MixedModeSif {
    let g1 = f_close_i * displacement_jump_i / (2.0 * delta_a.max(f64::EPSILON));
    let g2 = f_close_ii * displacement_jump_ii / (2.0 * delta_a.max(f64::EPSILON));
    let g3 = f_close_iii * displacement_jump_iii / (2.0 * delta_a.max(f64::EPSILON));
    MixedModeSif {
        k1: (g1.abs() * e_prime).sqrt().copysign(f_close_i),
        k2: (g2.abs() * e_prime).sqrt().copysign(f_close_ii),
        k3: (g3.abs() * 2.0 * shear_modulus)
            .sqrt()
            .copysign(f_close_iii),
    }
}
#[inline]
pub(super) fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
#[cfg(test)]
mod tests_ext {
    use super::*;
    use crate::xfem::*;
    #[test]
    fn test_phantom_node_activation() {
        let mut reg = PhantomNodeRegistry::new();
        let idx = reg.register(42);
        assert_eq!(reg.active_count(), 0);
        reg.activate(idx);
        assert_eq!(reg.active_count(), 1);
    }
    #[test]
    fn test_phantom_node_jump_energy_zero() {
        let mut reg = PhantomNodeRegistry::new();
        let idx = reg.register(0);
        reg.activate(idx);
        assert!(reg.jump_energy().abs() < 1e-15);
    }
    #[test]
    fn test_phantom_node_displacement_jump() {
        let mut p = PhantomNode::new(5);
        p.dofs = [1.0, 2.0, 3.0];
        let j = p.displacement_jump();
        assert!((j[0] - 2.0).abs() < 1e-12);
        assert!((j[1] - 4.0).abs() < 1e-12);
        assert!((j[2] - 6.0).abs() < 1e-12);
    }
    #[test]
    fn test_bifurcation_continue_slow_crack() {
        let result = crack_bifurcation_criterion(1e6, 5e5, 100.0, 3000.0, 0.3);
        assert_eq!(result, BranchingDecision::Continue);
    }
    #[test]
    fn test_bifurcation_occurs_fast_mixed_mode() {
        let result = crack_bifurcation_criterion(1e6, 5e5, 1500.0, 3000.0, 0.3);
        assert!(matches!(result, BranchingDecision::Bifurcate { .. }));
    }
    #[test]
    fn test_bifurcation_angle_pure_mode_i() {
        let angle = bifurcation_angle_max_hoop(1.0e6, 0.0);
        assert!(angle.abs() < 1e-12, "pure mode I → zero angle");
    }
    #[test]
    fn test_bifurcation_angle_nonzero_mode_ii() {
        let angle = bifurcation_angle_max_hoop(1.0e6, 1.0e5);
        assert!(angle.abs() > 1e-6, "mixed mode → non-zero branch angle");
    }
    #[test]
    fn test_xfem_fem_coupling_ramp() {
        let zone = XfemFemCouplingZone::linear_ramp(10, 5);
        assert_eq!(zone.blend_nodes.len(), 5);
        assert!(zone.ramp_at(0).abs() < 1e-12, "start weight = 0");
        assert!((zone.ramp_at(4) - 1.0).abs() < 1e-12, "end weight = 1");
    }
    #[test]
    fn test_xfem_fem_blend_values() {
        let zone = XfemFemCouplingZone::linear_ramp(0, 3);
        let v = zone.blend(1, 10.0, 0.0);
        assert!((v - 5.0).abs() < 1e-10, "blend at midpoint = 5.0, got {v}");
    }
    #[test]
    fn test_mcci_pure_mode_i() {
        let e_prime = 200e9_f64;
        let mu = 77e9_f64;
        let sif = mcci_mixed_mode(0.001, 1000.0, 0.0, 0.0, 0.002, 0.0, 0.0, e_prime, mu);
        let expected_k1 = (1000.0 * e_prime).sqrt();
        assert!(
            (sif.k1 - expected_k1).abs() / expected_k1 < 1e-8,
            "K_I mismatch: {} vs {}",
            sif.k1,
            expected_k1
        );
        assert!(sif.k2.abs() < 1e-3, "K_II should be zero");
    }
    #[test]
    fn test_mcci_effective_sif() {
        let sif = MixedModeSif {
            k1: 3.0,
            k2: 4.0,
            k3: 0.0,
        };
        let k_eff = sif.effective(0.3);
        assert!((k_eff - 5.0).abs() < 1e-10, "K_eff = {k_eff}");
    }
    #[test]
    fn test_crack_front_3d_arc_length() {
        let mut front = CrackFrontEnrichment::new();
        front.push_node([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        front.push_node([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        front.push_node([3.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((front.total_arc_length() - 3.0).abs() < 1e-10);
        assert_eq!(front.len(), 3);
    }
    #[test]
    fn test_crack_front_3d_interpolation() {
        let mut front = CrackFrontEnrichment::new();
        front.push_node([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        front.push_node([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let mid = front.position_at(1.0).expect("must return value");
        assert!((mid[0] - 1.0).abs() < 1e-10, "x mid = {}", mid[0]);
    }
    #[test]
    fn test_crack_front_branch_functions_at_origin() {
        let bf = CrackFrontNode::branch_functions(0.0, 0.0);
        for v in bf {
            assert!(v.abs() < 1e-15);
        }
    }
    #[test]
    fn test_crack_front_branch_functions_nonzero() {
        let r = 1.0;
        let theta = PI / 2.0;
        let bf = CrackFrontNode::branch_functions(r, theta);
        let expected = (PI / 4.0).sin();
        assert!((bf[0] - expected).abs() < 1e-10, "bf[0] = {}", bf[0]);
    }
    #[test]
    fn test_mixed_mode_sif_energy_release_rate() {
        let sif = MixedModeSif {
            k1: 1e6,
            k2: 0.0,
            k3: 0.0,
        };
        let e_prime = 200e9_f64;
        let mu = 77e9_f64;
        let g = sif.energy_release_rate(e_prime, mu);
        let expected = 1e12 / e_prime;
        assert!((g - expected).abs() / expected < 1e-8, "G = {g}");
    }
    #[test]
    fn test_phantom_registry_inactive_not_counted() {
        let mut reg = PhantomNodeRegistry::new();
        reg.register(0);
        reg.register(1);
        reg.activate(0);
        assert_eq!(reg.active_count(), 1);
    }
}
/// Fixed-capacity convex polygon produced by cutting a triangle with a line.
///
/// A triangle cut by a line leaves at most four vertices on either side.
struct CutPolygon {
    points: [[f64; 2]; 4],
    len: usize,
}
impl CutPolygon {
    fn new() -> Self {
        Self {
            points: [[0.0; 2]; 4],
            len: 0,
        }
    }
    fn push(&mut self, point: [f64; 2]) {
        if let Some(slot) = self.points.get_mut(self.len) {
            *slot = point;
            self.len += 1;
        }
    }
    /// Fan-triangulate the polygon from its first vertex, dropping triangles
    /// whose area does not exceed `min_area`.
    fn fan_into(&self, side: i8, min_area: f64, out: &mut Vec<SubTriangle>) {
        if self.len < 3 {
            return;
        }
        for i in 1..self.len - 1 {
            let tri = SubTriangle::new([self.points[0], self.points[i], self.points[i + 1]], side);
            if tri.area() > min_area {
                out.push(tri);
            }
        }
    }
}
/// Decompose a triangular element cut by a straight line (the crack) into
/// sub-triangles on each side.
///
/// The input triangle has vertices in reference coordinates (xi, eta).  The
/// interface is the zero contour of the linear interpolation of the nodal
/// level-set values `phi`.  Both sides are built by walking the three edges in
/// order: a vertex joins the positive polygon when `phi > 0`, the negative
/// polygon when `phi < 0` and both when `phi == 0`, and an edge with a strict
/// sign change adds its crossing point to both polygons.  Every polygon is a
/// convex fan of at most four vertices.
///
/// A triangle cut away from its vertices yields three sub-triangles (one on
/// the side that owns a single vertex and two on the other side); an interface
/// through one vertex yields two; a triangle that is not cut, or that only
/// touches the interface at a vertex or along an edge, is returned as a single
/// triangle on its own side.  A triangle with all `phi == 0` is returned as one
/// triangle with side +1.  Sub-triangles with a vanishing area are dropped.
pub fn sub_triangulate_triangle(vertices: &[[f64; 2]; 3], phi: &[f64; 3]) -> Vec<SubTriangle> {
    let parent = SubTriangle::new(*vertices, 1);
    if phi.iter().all(|&value| value == 0.0) {
        return vec![parent];
    }
    let mut positive = CutPolygon::new();
    let mut negative = CutPolygon::new();
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 0)] {
        let (phi_i, phi_j) = (phi[i], phi[j]);
        if phi_i > 0.0 {
            positive.push(vertices[i]);
        } else if phi_i < 0.0 {
            negative.push(vertices[i]);
        } else {
            positive.push(vertices[i]);
            negative.push(vertices[i]);
        }
        if (phi_i > 0.0 && phi_j < 0.0) || (phi_i < 0.0 && phi_j > 0.0) {
            let t = phi_i / (phi_i - phi_j);
            let crossing = [
                vertices[i][0] + t * (vertices[j][0] - vertices[i][0]),
                vertices[i][1] + t * (vertices[j][1] - vertices[i][1]),
            ];
            positive.push(crossing);
            negative.push(crossing);
        }
    }
    let min_area = 1e-14 * parent.area();
    let mut tris = Vec::with_capacity(3);
    positive.fan_into(1, min_area, &mut tris);
    negative.fan_into(-1, min_area, &mut tris);
    if tris.is_empty() {
        let side = if phi.iter().sum::<f64>() >= 0.0 {
            1
        } else {
            -1
        };
        return vec![SubTriangle::new(*vertices, side)];
    }
    tris
}
/// Compute the global XFEM DOF count for a problem.
///
/// Standard DOFs: n_nodes * n_dof_per_node
/// Heaviside DOFs: n_heaviside_nodes * n_dof_per_node
/// Tip DOFs: n_tip_nodes * 4 * n_dof_per_node (4 branch functions per spatial DOF)
pub fn total_xfem_dofs(
    n_nodes: usize,
    n_dof_per_node: usize,
    n_heaviside_nodes: usize,
    n_tip_nodes: usize,
) -> usize {
    let standard_dofs = n_nodes * n_dof_per_node;
    let heaviside_dofs = n_heaviside_nodes * n_dof_per_node;
    let tip_dofs = n_tip_nodes * 4 * n_dof_per_node;
    standard_dofs + heaviside_dofs + tip_dofs
}
/// Build the enriched DOF index map for a set of nodes.
///
/// Returns a vector where entry `i` gives the global DOF offset for the
/// enrichment DOFs of node `i`.  Returns `None` if the node is not enriched.
pub fn build_enriched_dof_map(
    n_nodes: usize,
    n_dof_per_node: usize,
    heaviside_nodes: &[usize],
    tip_nodes: &[usize],
) -> Vec<Option<usize>> {
    let mut map = vec![None; n_nodes];
    let mut offset = n_nodes * n_dof_per_node;
    for &node in heaviside_nodes {
        if node < n_nodes {
            map[node] = Some(offset);
            offset += n_dof_per_node;
        }
    }
    for &node in tip_nodes {
        if node < n_nodes {
            map[node] = Some(offset);
            offset += 4 * n_dof_per_node;
        }
    }
    map
}
/// Locate the crack tip position by finding the node whose level set
/// `psi` value is closest to zero from the negative side (behind the tip)
/// and whose `phi` is smallest in magnitude (on the crack line).
///
/// Returns `(node_index, tip_position)` where `tip_position` is an
/// interpolated estimate.
pub fn locate_crack_tip_from_level_set(
    nodes: &[[f64; 3]],
    ls: &LevelSet,
) -> Option<(usize, [f64; 3])> {
    if nodes.is_empty() || ls.phi.len() != nodes.len() {
        return None;
    }
    let best = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| ls.psi[*i] <= 0.0)
        .min_by(|(i, _), (j, _)| {
            let dist_i = ls.psi[*i].abs();
            let dist_j = ls.psi[*j].abs();
            dist_i
                .partial_cmp(&dist_j)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    best.map(|(idx, _)| (idx, nodes[idx]))
}
/// Compute the crack mouth opening displacement (CMOD) from nodal displacements.
///
/// The CMOD is the relative normal displacement between the upper and lower
/// crack faces at the crack mouth (crack opening point).
///
/// # Arguments
/// * `u_upper` – displacement of the upper crack face node (3 components)
/// * `u_lower` – displacement of the lower crack face node (3 components)
/// * `crack_normal` – unit normal to the crack plane
///
/// Returns the scalar opening displacement (projection onto crack normal).
pub fn crack_mouth_opening_displacement(
    u_upper: [f64; 3],
    u_lower: [f64; 3],
    crack_normal: [f64; 3],
) -> f64 {
    let delta = [
        u_upper[0] - u_lower[0],
        u_upper[1] - u_lower[1],
        u_upper[2] - u_lower[2],
    ];
    delta[0] * crack_normal[0] + delta[1] * crack_normal[1] + delta[2] * crack_normal[2]
}
/// Compute the crack tip opening displacement (CTOD) at distance `r` behind
/// the crack tip using the Irwin formula.
///
/// CTOD = 4 * K_I / E' * sqrt(r / (2 π))   (plane strain)
///
/// where E' = E / (1 - ν²).
pub fn ctod_irwin(k1: f64, young_modulus: f64, poisson: f64, r: f64) -> f64 {
    if r < 0.0 {
        return 0.0;
    }
    let e_prime = young_modulus / (1.0 - poisson * poisson).max(f64::EPSILON);
    4.0 * k1 / e_prime * (r / (2.0 * PI)).sqrt()
}
/// Numerically integrate the interaction integral over a circular domain.
///
/// The interaction integral I is computed by sampling the integrand at
/// `n_pts` points around the crack tip at radius `r_domain`.
///
/// This is a simplified version that assumes the integrand is provided as a
/// function value at uniformly spaced angles.
///
/// # Arguments
/// * `integrand_values` – values of the interaction integrand at `n_pts`
///   uniformly spaced angles from 0 to 2π
/// * `r_domain` – radius of the integration domain (m)
///
/// Returns the approximate interaction integral value.
pub fn interaction_integral_trapezoidal(integrand_values: &[f64], r_domain: f64) -> f64 {
    let n = integrand_values.len();
    if n < 2 {
        return 0.0;
    }
    let dtheta = 2.0 * PI / n as f64;
    let sum: f64 = integrand_values.iter().sum();
    sum * dtheta * r_domain
}
