// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Capsule-capsule contact manifold generation.
//!
//! A capsule is every point within a radius of its axis segment, so two
//! capsules overlap exactly when their axis segments are closer than the sum of
//! the radii. [`CapsuleCapsuleManifold`] finds the closest pair of axis points
//! (re-solving the parameter on the first segment whenever the second one is
//! clamped to an end, see
//! [`closest_points_on_segments`])
//! and turns it into one contact. Capsules whose axes are (nearly) parallel
//! rest on each other along a whole stretch of their axes, so for them the
//! manifold holds two contacts, one at each end of the stretch where the axes
//! overlap.
//!
//! The contact normal points from capsule B to capsule A. `pos_a` lies on the
//! surface of A and `pos_b` on the surface of B, and `pos_a - pos_b` is
//! `-depth * normal` for a penetrating contact.

use super::{ContactPoint, add3, cross3, dot3, norm3, scale3, sub3};
use crate::contact_manifold::closest_points_on_segments;

/// Axis length below which a capsule counts as a sphere (a single point).
const MIN_AXIS_LENGTH: f64 = 1e-5;

/// Axis distance below which the contact normal cannot be taken from the
/// vector between the closest axis points and is derived from the axes instead.
const MIN_AXIS_SEPARATION: f64 = 1e-10;

/// Smallest `|cos|` of the angle between the axes for which two capsules count
/// as parallel (about 1.8 degrees).
const PARALLEL_COSINE: f64 = 0.9995;

/// Smallest distance between the two contacts of a parallel pair, as a fraction
/// of the sum of the radii. A shorter overlap of the axes is a single contact.
const MIN_END_SEPARATION: f64 = 1e-2;

/// Relative slack (of the sum of the radii) when comparing contact depths.
const DEPTH_TOLERANCE: f64 = 1e-9;

/// Capsule-capsule contact manifold generator.
///
/// Computes the closest points of the two axis segments and generates one
/// contact for them, or two contacts when the axes are parallel and overlap.
#[derive(Debug, Clone)]
pub struct CapsuleCapsuleManifold;

impl CapsuleCapsuleManifold {
    /// Compute capsule-capsule contact.
    ///
    /// Returns an empty vector when the capsules are separated, one contact
    /// when they cross or touch at a single place, and two contacts when the
    /// axes are parallel (`|cos| >= 0.9995`) and overlap: they are then placed
    /// at the two ends of the overlapping stretch.
    ///
    /// # Arguments
    /// * `pa_start`, `pa_end` – endpoints of capsule A segment
    /// * `ra` – radius of capsule A
    /// * `pb_start`, `pb_end` – endpoints of capsule B segment
    /// * `rb` – radius of capsule B
    pub fn compute(
        pa_start: [f64; 3],
        pa_end: [f64; 3],
        ra: f64,
        pb_start: [f64; 3],
        pb_end: [f64; 3],
        rb: f64,
    ) -> Vec<ContactPoint> {
        let axes = Axes {
            a_start: pa_start,
            a_dir: sub3(pa_end, pa_start),
            b_start: pb_start,
            b_dir: sub3(pb_end, pb_start),
            ra,
            rb,
        };
        let (_s, _t, ca, cb) = closest_points_on_segments(pa_start, pa_end, pb_start, pb_end);
        let Some(closest) = axes.contact_between(ca, cb) else {
            return Vec::new();
        };

        if let Some(ends) = axes.parallel_end_contacts() {
            let deepest_end = ends[0].depth.max(ends[1].depth);
            // Parallel axes are equally deep along the whole overlap. If the
            // axes cross closer than at the ends, the crossing is the contact.
            if closest.depth <= deepest_end + DEPTH_TOLERANCE * (ra + rb) {
                return ends.to_vec();
            }
        }
        vec![closest]
    }
}

/// The two capsule axes and radii of one query.
struct Axes {
    a_start: [f64; 3],
    a_dir: [f64; 3],
    b_start: [f64; 3],
    b_dir: [f64; 3],
    ra: f64,
    rb: f64,
}

impl Axes {
    /// Contact between the capsules for the axis points `ca` (on A) and `cb`
    /// (on B), or `None` when they are at least `ra + rb` apart.
    fn contact_between(&self, ca: [f64; 3], cb: [f64; 3]) -> Option<ContactPoint> {
        let diff = sub3(ca, cb);
        let distance = norm3(diff);
        let total_radius = self.ra + self.rb;
        if distance >= total_radius {
            return None;
        }
        let normal = if distance > MIN_AXIS_SEPARATION {
            scale3(diff, 1.0 / distance)
        } else {
            self.crossing_normal()
        };
        let depth = total_radius - distance;
        let pos_a = sub3(ca, scale3(normal, self.ra));
        let pos_b = add3(cb, scale3(normal, self.rb));
        Some(ContactPoint::new(pos_a, pos_b, normal, depth))
    }

    /// Contact normal for axes that touch or cross, where the closest axis
    /// points coincide and give no direction: the common perpendicular of the
    /// two axes, or any perpendicular of the axis when they are parallel.
    fn crossing_normal(&self) -> [f64; 3] {
        let a_len = norm3(self.a_dir);
        let b_len = norm3(self.b_dir);
        let common = cross3(self.a_dir, self.b_dir);
        let common_len = norm3(common);
        if common_len > 1e-12 * a_len * b_len && common_len > 0.0 {
            return scale3(common, 1.0 / common_len);
        }
        let axis = if a_len >= b_len {
            self.a_dir
        } else {
            self.b_dir
        };
        perpendicular(axis)
    }

    /// The two contacts of capsules with parallel axes, at the ends of the
    /// stretch where the axes overlap, or `None` when the axes are not
    /// parallel, the overlap is too short, or either end does not penetrate.
    fn parallel_end_contacts(&self) -> Option<[ContactPoint; 2]> {
        let a_len = norm3(self.a_dir);
        let b_len = norm3(self.b_dir);
        if a_len <= MIN_AXIS_LENGTH || b_len <= MIN_AXIS_LENGTH {
            return None;
        }
        let a_unit = scale3(self.a_dir, 1.0 / a_len);
        if dot3(a_unit, self.b_dir).abs() < PARALLEL_COSINE * b_len {
            return None;
        }

        // Positions along the axis of A (measured from its start) of the
        // projected endpoints of B; the overlap is their span clipped to A.
        let b_first = dot3(sub3(self.b_start, self.a_start), a_unit);
        let b_second = b_first + dot3(self.b_dir, a_unit);
        let overlap_start = b_first.min(b_second).max(0.0);
        let overlap_end = b_first.max(b_second).min(a_len);
        if overlap_end - overlap_start <= MIN_END_SEPARATION * (self.ra + self.rb) {
            return None;
        }

        let end_contact = |along: f64| -> Option<ContactPoint> {
            let ca = add3(self.a_start, scale3(a_unit, along));
            let t = (dot3(sub3(ca, self.b_start), self.b_dir) / (b_len * b_len)).clamp(0.0, 1.0);
            let cb = add3(self.b_start, scale3(self.b_dir, t));
            self.contact_between(ca, cb)
        };
        Some([end_contact(overlap_start)?, end_contact(overlap_end)?])
    }
}

/// A unit vector perpendicular to `v` (`[0, 1, 0]` when `v` vanishes).
fn perpendicular(v: [f64; 3]) -> [f64; 3] {
    let length = norm3(v);
    if length <= 0.0 {
        return [0.0, 1.0, 0.0];
    }
    // Cross with the world axis that is least aligned with `v`.
    let helper = if v[0].abs() <= v[1].abs() && v[0].abs() <= v[2].abs() {
        [1.0, 0.0, 0.0]
    } else if v[1].abs() <= v[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let perp = cross3(v, helper);
    scale3(perp, 1.0 / norm3(perp))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        norm3(sub3(a, b)) <= tol
    }

    /// Distance between two segments by a method that shares no code with the
    /// production path: the minimum over the four endpoint-to-segment
    /// distances and, when both parameters fall inside the segments, the
    /// closest points of the two infinite lines.
    fn reference_distance(a0: [f64; 3], a1: [f64; 3], b0: [f64; 3], b1: [f64; 3]) -> f64 {
        fn point_segment(p: [f64; 3], s0: [f64; 3], s1: [f64; 3]) -> f64 {
            let d = sub3(s1, s0);
            let len_sq = dot3(d, d);
            let t = if len_sq > 0.0 {
                (dot3(sub3(p, s0), d) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            norm3(sub3(p, add3(s0, scale3(d, t))))
        }
        let mut best = point_segment(a0, b0, b1)
            .min(point_segment(a1, b0, b1))
            .min(point_segment(b0, a0, a1))
            .min(point_segment(b1, a0, a1));
        let da = sub3(a1, a0);
        let db = sub3(b1, b0);
        let r = sub3(a0, b0);
        let (a, b, c) = (dot3(da, da), dot3(da, db), dot3(db, db));
        let (d, e) = (dot3(da, r), dot3(db, r));
        let denom = a * c - b * b;
        if denom > 1e-12 * a * c {
            let s = (b * e - c * d) / denom;
            let t = (a * e - b * d) / denom;
            if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t) {
                let pa = add3(a0, scale3(da, s));
                let pb = add3(b0, scale3(db, t));
                best = best.min(norm3(sub3(pa, pb)));
            }
        }
        best
    }

    /// Deterministic xorshift generator for the randomized tests.
    struct TestRng(u64);

    impl TestRng {
        fn next_unit(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next_unit()
        }

        fn point(&mut self, extent: f64) -> [f64; 3] {
            [
                self.range(-extent, extent),
                self.range(-extent, extent),
                self.range(-extent, extent),
            ]
        }
    }

    #[test]
    fn clamped_segment_parameter_is_re_solved() {
        // The infinite lines are closest at t = -0.5 on B, so t clamps to 0 and
        // s must be re-solved from 0.2 to 0.3: the true axis distance is 1, not
        // sqrt(2), and the capsules (radii sum 1.2) overlap by 0.2.
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            0.6,
            [3.0, 1.0, 0.0],
            [5.0, 3.0, 0.0],
            0.6,
        );
        assert_eq!(contacts.len(), 1, "{contacts:?}");
        let contact = &contacts[0];
        assert!(
            (contact.depth - 0.2).abs() < 1e-12,
            "depth {}",
            contact.depth
        );
        assert!(close(contact.normal, [0.0, -1.0, 0.0], 1e-12));
        assert!(close(contact.pos_a, [3.0, 0.6, 0.0], 1e-12));
        assert!(close(contact.pos_b, [3.0, 0.4, 0.0], 1e-12));
    }

    #[test]
    fn clamping_the_other_end_is_re_solved_too() {
        // Mirror image: B ends at t = 1.5 beyond its end, so t clamps to 1.
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            0.6,
            [5.0, 3.0, 0.0],
            [3.0, 1.0, 0.0],
            0.6,
        );
        assert_eq!(contacts.len(), 1, "{contacts:?}");
        assert!((contacts[0].depth - 0.2).abs() < 1e-12);
        assert!(close(contacts[0].normal, [0.0, -1.0, 0.0], 1e-12));
    }

    #[test]
    fn capsules_just_out_of_reach_have_no_contact() {
        // Axis distance 1, radii sum 0.9.
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            0.45,
            [3.0, 1.0, 0.0],
            [5.0, 3.0, 0.0],
            0.45,
        );
        assert!(contacts.is_empty(), "{contacts:?}");
    }

    #[test]
    fn contact_matches_the_reference_distance_for_random_capsules() {
        let mut rng = TestRng(0x5eed_cafe_f00d_0001);
        let mut hits = 0;
        let mut misses = 0;
        for _ in 0..4000 {
            let a0 = rng.point(2.0);
            let a1 = rng.point(2.0);
            let b0 = rng.point(2.0);
            let b1 = rng.point(2.0);
            let ra = rng.range(0.05, 0.8);
            let rb = rng.range(0.05, 0.8);
            let distance = reference_distance(a0, a1, b0, b1);
            let total = ra + rb;
            // Stay away from the touching configuration.
            if (distance - total).abs() < 1e-6 {
                continue;
            }
            let contacts = CapsuleCapsuleManifold::compute(a0, a1, ra, b0, b1, rb);
            if distance < total {
                hits += 1;
                assert!(
                    !contacts.is_empty(),
                    "missed contact, distance {distance} < {total}"
                );
                let deepest = contacts.iter().map(|c| c.depth).fold(0.0_f64, f64::max);
                assert!(
                    (deepest - (total - distance)).abs() < 1e-8,
                    "depth {deepest} != {}",
                    total - distance
                );
                for c in &contacts {
                    assert!((norm3(c.normal) - 1.0).abs() < 1e-9);
                    assert!(
                        (norm3(sub3(c.pos_a, c.pos_b)) - c.depth).abs() < 1e-9,
                        "pos_a - pos_b must span the depth"
                    );
                    assert!(dot3(sub3(c.pos_a, c.pos_b), c.normal) < 0.0);
                }
            } else {
                misses += 1;
                assert!(
                    contacts.is_empty(),
                    "spurious contact, distance {distance} >= {total}: {contacts:?}"
                );
            }
        }
        assert!(hits > 200 && misses > 200, "hits {hits}, misses {misses}");
    }

    #[test]
    fn parallel_capsules_get_two_contacts_at_the_ends_of_the_overlap() {
        // B lies inside the span of A, 0.5 above it.
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            0.4,
            [1.0, 0.5, 0.0],
            [3.0, 0.5, 0.0],
            0.4,
        );
        assert_eq!(contacts.len(), 2, "{contacts:?}");
        let mut xs: Vec<f64> = contacts
            .iter()
            .map(|c| 0.5 * (c.pos_a[0] + c.pos_b[0]))
            .collect();
        xs.sort_by(f64::total_cmp);
        assert!(
            (xs[0] - 1.0).abs() < 1e-12 && (xs[1] - 3.0).abs() < 1e-12,
            "{xs:?}"
        );
        for c in &contacts {
            assert!((c.depth - 0.3).abs() < 1e-12, "depth {}", c.depth);
            assert!(close(c.normal, [0.0, -1.0, 0.0], 1e-12));
        }
    }

    #[test]
    fn parallel_capsules_with_partial_overlap_clip_to_both_segments() {
        // A spans [0, 4], B spans [3, 6] (reversed direction): overlap [3, 4].
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            0.5,
            [6.0, 0.0, 0.7],
            [3.0, 0.0, 0.7],
            0.5,
        );
        assert_eq!(contacts.len(), 2, "{contacts:?}");
        let mut xs: Vec<f64> = contacts.iter().map(|c| c.pos_a[0]).collect();
        xs.sort_by(f64::total_cmp);
        assert!(
            (xs[0] - 3.0).abs() < 1e-12 && (xs[1] - 4.0).abs() < 1e-12,
            "{xs:?}"
        );
        for c in &contacts {
            assert!((c.depth - 0.3).abs() < 1e-12);
            assert!(close(c.normal, [0.0, 0.0, -1.0], 1e-12));
        }
    }

    #[test]
    fn nearly_parallel_capsule_gets_the_end_contacts_with_their_own_depths() {
        // B is tilted by 1 degree; both ends of the overlap [1, 3] penetrate,
        // the lower one deeper.
        let tilt = 1.0_f64.to_radians();
        let rise = 2.0 * tilt.tan();
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            0.4,
            [1.0, 0.5, 0.0],
            [3.0, 0.5 + rise, 0.0],
            0.4,
        );
        assert_eq!(contacts.len(), 2, "{contacts:?}");
        let (deep, shallow) = if contacts[0].depth > contacts[1].depth {
            (&contacts[0], &contacts[1])
        } else {
            (&contacts[1], &contacts[0])
        };
        assert!((deep.depth - 0.3).abs() < 1e-9, "deep {}", deep.depth);
        assert!(
            (shallow.depth - (0.3 - rise)).abs() < 1e-3,
            "shallow {}",
            shallow.depth
        );
    }

    #[test]
    fn crossing_axes_far_below_the_ends_keep_the_single_deepest_contact() {
        // Axes 1.5 degrees apart (still parallel by the manifold's tolerance)
        // and 0.1 apart where they cross at the middle, but about 0.2 apart at
        // the ends: the crossing is the deepest point and the end contacts
        // would hide it.
        let tilt = 1.5_f64.to_radians();
        let (s, c) = tilt.sin_cos();
        let contacts = CapsuleCapsuleManifold::compute(
            [-5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            0.6,
            [-5.0 * c, -5.0 * s, 0.1],
            [5.0 * c, 5.0 * s, 0.1],
            0.6,
        );
        assert_eq!(contacts.len(), 1, "{contacts:?}");
        assert!(
            (contacts[0].depth - 1.1).abs() < 1e-9,
            "depth {}",
            contacts[0].depth
        );
    }

    #[test]
    fn collinear_capsules_end_to_end_have_one_contact_along_the_axis() {
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.3,
            [2.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            0.3,
        );
        assert_eq!(contacts.len(), 1, "{contacts:?}");
        assert!((contacts[0].depth - 0.1).abs() < 1e-12);
        assert!(close(contacts[0].normal, [-1.0, 0.0, 0.0], 1e-12));
    }

    #[test]
    fn perpendicular_crossing_axes_use_the_common_perpendicular() {
        let contacts = CapsuleCapsuleManifold::compute(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            0.25,
        );
        assert_eq!(contacts.len(), 1);
        let c = &contacts[0];
        assert!((c.depth - 0.75).abs() < 1e-12);
        assert!((norm3(c.normal) - 1.0).abs() < 1e-12);
        assert!(dot3(c.normal, [1.0, 0.0, 0.0]).abs() < 1e-12);
        assert!(dot3(c.normal, [0.0, 0.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn coincident_axes_get_a_normal_perpendicular_to_the_axis() {
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            0.4,
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            0.4,
        );
        assert_eq!(contacts.len(), 2, "{contacts:?}");
        for c in &contacts {
            assert!((c.depth - 0.8).abs() < 1e-12);
            assert!((norm3(c.normal) - 1.0).abs() < 1e-12);
            assert!(dot3(c.normal, [1.0, 0.0, 0.0]).abs() < 1e-12);
        }
    }

    #[test]
    fn degenerate_capsules_behave_like_spheres() {
        // Two point capsules.
        let contacts = CapsuleCapsuleManifold::compute(
            [0.0; 3],
            [0.0; 3],
            0.5,
            [0.8, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            0.5,
        );
        assert_eq!(contacts.len(), 1);
        assert!((contacts[0].depth - 0.2).abs() < 1e-12);
        assert!(close(contacts[0].normal, [-1.0, 0.0, 0.0], 1e-12));
        // A point capsule against a segment.
        let contacts = CapsuleCapsuleManifold::compute(
            [1.0, 0.7, 0.0],
            [1.0, 0.7, 0.0],
            0.5,
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
        );
        assert_eq!(contacts.len(), 1);
        assert!((contacts[0].depth - 0.3).abs() < 1e-12);
        assert!(close(contacts[0].normal, [0.0, 1.0, 0.0], 1e-12));
    }

    #[test]
    fn identical_point_capsules_have_a_unit_normal() {
        let contacts =
            CapsuleCapsuleManifold::compute([1.0; 3], [1.0; 3], 0.5, [1.0; 3], [1.0; 3], 0.5);
        assert_eq!(contacts.len(), 1);
        assert!((norm3(contacts[0].normal) - 1.0).abs() < 1e-12);
        assert!((contacts[0].depth - 1.0).abs() < 1e-12);
    }
}
