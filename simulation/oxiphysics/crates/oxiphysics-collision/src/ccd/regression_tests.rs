// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the continuous queries of [`super`]: the approach bound
//! of [`time_of_impact`] that follows the rotation of the bodies, its iteration
//! budget, and the lower-bound GJK distance behind the conservative advancement
//! of the support-function API.

use super::*;
use oxiphysics_core::math::{UnitQuaternion, Vector4};
use oxiphysics_geometry::{BoxShape, Capsule, Compound, Sphere};
use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

type Point = [f64; 3];

fn position(x: Real, y: Real, z: Real) -> Transform {
    Transform::from_position(Vec3::new(x, y, z))
}

fn rotated(x: Real, y: Real, z: Real, axis: Point, angle: Real) -> Transform {
    Transform::from_axis_angle(
        Vec3::new(x, y, z),
        Vec3::new(axis[0], axis[1], axis[2]),
        angle,
    )
}

fn cube(half: Real) -> BoxShape {
    BoxShape::new(Vec3::new(half, half, half))
}

fn bar(half_length: Real, half_thickness: Real) -> BoxShape {
    BoxShape::new(Vec3::new(half_length, half_thickness, half_thickness))
}

/// Two moving shapes with the reference queries used to check the results.
struct Scene<'a> {
    shape_a: &'a dyn Shape,
    a_start: Transform,
    a_end: Transform,
    shape_b: &'a dyn Shape,
    b_start: Transform,
    b_end: Transform,
}

impl<'a> Scene<'a> {
    fn new(
        shape_a: &'a dyn Shape,
        a_start: Transform,
        a_end: Transform,
        shape_b: &'a dyn Shape,
        b_start: Transform,
        b_end: Transform,
    ) -> Self {
        Self {
            shape_a,
            a_start,
            a_end,
            shape_b,
            b_start,
            b_end,
        }
    }

    fn pair(&self) -> SweptPair<'_> {
        SweptPair {
            shape_a: self.shape_a,
            a_start: &self.a_start,
            a_end: &self.a_end,
            shape_b: self.shape_b,
            b_start: &self.b_start,
            b_end: &self.b_end,
        }
    }

    fn toi(&self) -> Option<ToiResult> {
        time_of_impact(
            self.shape_a,
            &self.a_start,
            &self.a_end,
            self.shape_b,
            &self.b_start,
            &self.b_end,
        )
    }

    /// Gap between the shapes at `t`, `None` while they overlap.
    fn gap(&self, t: Real) -> Option<Real> {
        match self.pair().query_at(t) {
            GjkResult::Intersecting(_) => None,
            GjkResult::Separated { distance, .. } => Some(distance),
        }
    }

    /// The scene and the gap around `t`, for failure messages.
    fn describe(&self, t: Real) -> String {
        let scan: Vec<String> = (-6..=6)
            .map(|k| {
                let at = (t + k as Real * 2.5e-4).clamp(0.0, 1.0);
                format!("{at:.5}: {:?}", self.gap(at))
            })
            .collect();
        format!(
            "A {:?} {:?} -> {:?}\nB {:?} {:?} -> {:?}\ngap scan:\n{}",
            self.shape_a,
            self.a_start,
            self.a_end,
            self.shape_b,
            self.b_start,
            self.b_end,
            scan.join("\n")
        )
    }

    /// First of `samples + 1` evenly spaced times at which the shapes overlap.
    fn first_overlap(&self, samples: usize) -> Option<Real> {
        (0..=samples)
            .map(|k| k as Real / samples as Real)
            .find(|&t| self.gap(t).is_none())
    }

    /// Check a time-of-impact result against dense sampling of the motion:
    /// a reported contact must be a real one and may not lie after the first
    /// sampled overlap, and no sampled overlap may be missed.
    fn check(&self, result: Option<&ToiResult>, label: &str) {
        const SAMPLES: usize = 800;
        let first_overlap = self.first_overlap(SAMPLES);
        match result {
            None => assert!(
                first_overlap.is_none() || first_overlap == Some(0.0),
                "{label}: no impact reported although the shapes overlap at t = {first_overlap:?}"
            ),
            Some(hit) => {
                assert!((0.0..=1.0).contains(&hit.toi), "{label}: toi {}", hit.toi);
                assert_ne!(
                    first_overlap,
                    Some(0.0),
                    "{label}: shapes overlap at the start, which is not an impact"
                );
                if let Some(overlap) = first_overlap {
                    assert!(
                        hit.toi <= overlap + 1e-9,
                        "{label}: toi {} lies after the first overlap at {overlap}\n{}",
                        hit.toi,
                        self.describe(overlap)
                    );
                }
                if let Some(gap) = self.gap(hit.toi) {
                    assert!(
                        gap < 1e-5,
                        "{label}: gap {gap} at the reported toi {}",
                        hit.toi
                    );
                }
                assert!(
                    (hit.normal.norm() - 1.0).abs() < 1e-9,
                    "{label}: normal {:?} is not a unit vector",
                    hit.normal
                );
            }
        }
    }
}

/// Deterministic xorshift generator for the randomized tests.
struct TestRng(u64);

impl TestRng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn point(&mut self, extent: f64) -> Point {
        [
            self.range(-extent, extent),
            self.range(-extent, extent),
            self.range(-extent, extent),
        ]
    }

    fn direction(&mut self) -> Point {
        loop {
            let p = self.point(1.0);
            let l = dot3(p, p);
            if l > 0.01 && l <= 1.0 {
                return norm3(p);
            }
        }
    }

    fn index(&mut self, bound: usize) -> usize {
        ((self.unit() * bound as f64) as usize).min(bound - 1)
    }
}

// ---------------------------------------------------------------------------
// F0069: the approach bound of time_of_impact follows the rotation
// ---------------------------------------------------------------------------

#[test]
fn slow_approach_of_unit_cubes_reports_the_hit() {
    // Gap 0.05, relative speed 0.1 per step: the old constant angular bound
    // (a quarter of the radii sum, about 0.31 here) made every step 4 times too
    // short and the 32 iterations ran out before the gap closed.
    let a = cube(0.5);
    let b = cube(0.5);
    let scene = Scene::new(
        &a,
        position(0.0, 0.0, 0.0),
        position(0.1, 0.0, 0.0),
        &b,
        position(1.05, 0.0, 0.0),
        position(1.05, 0.0, 0.0),
    );
    let hit = scene.toi().expect("the cubes touch at t = 0.5");
    assert!(
        hit.toi <= 0.5 && hit.toi > 0.5 - 1e-5,
        "toi {} should be just before 0.5",
        hit.toi
    );
    assert!(
        (hit.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-4,
        "{:?}",
        hit.normal
    );
    assert!((hit.witness_a.x - 0.55).abs() < 1e-5 && (hit.witness_b.x - 0.55).abs() < 1e-5);
    // Without rotation the bound is tight: two steps are enough.
    let quick = time_of_impact_with_budget(&scene.pair(), 2).expect("two iterations suffice");
    assert_eq!(quick.toi, hit.toi);
    scene.check(Some(&hit), "slow approach");
}

#[test]
fn spinning_bar_with_zero_translation_hits_the_box_in_its_sweep() {
    // The bar turns by 90 degrees about z through its centre while nothing
    // translates; the box sits where the tip of the bar swings through.
    let bar = bar(1.0, 0.05);
    let target = cube(0.5);
    let scene = Scene::new(
        &bar,
        position(0.0, 0.0, 0.0),
        rotated(0.0, 0.0, 0.0, [0.0, 0.0, 1.0], FRAC_PI_2),
        &target,
        position(0.0, 1.0, 0.0),
        position(0.0, 1.0, 0.0),
    );
    assert!(scene.gap(0.0).is_some() && scene.gap(1.0).is_none());
    let hit = scene.toi().expect("the bar sweeps into the box");
    scene.check(Some(&hit), "spinning bar");
    // The bar reaches the lower right corner of the box (0.5, 0.5) at about 45
    // degrees, thickness included a little earlier.
    assert!(hit.toi > 0.3 && hit.toi < 0.5, "toi {}", hit.toi);
    // The shapes are separated just before the reported time.
    let before = scene.gap(hit.toi - 1e-3).expect("apart before the contact");
    assert!(before > 0.0);
    // Contact normal points from the bar toward the box (up and to the left).
    assert!(hit.normal.y > 0.0, "{:?}", hit.normal);
}

#[test]
fn rotation_that_stops_short_of_the_box_is_a_miss() {
    let bar = bar(1.0, 0.05);
    let target = cube(0.5);
    let scene = Scene::new(
        &bar,
        position(0.0, 0.0, 0.0),
        rotated(0.0, 0.0, 0.0, [0.0, 0.0, 1.0], 20.0_f64.to_radians()),
        &target,
        position(0.0, 1.0, 0.0),
        position(0.0, 1.0, 0.0),
    );
    assert!(scene.toi().is_none());
    scene.check(None, "short rotation");
}

#[test]
fn bodies_at_rest_report_no_impact() {
    let a = cube(0.5);
    let b = cube(0.5);
    let scene = Scene::new(
        &a,
        position(0.0, 0.0, 0.0),
        position(0.0, 0.0, 0.0),
        &b,
        position(1.0000001, 0.0, 0.0),
        position(1.0000001, 0.0, 0.0),
    );
    assert!(scene.toi().is_none());
}

#[test]
fn shapes_overlapping_at_the_start_are_not_an_impact() {
    let a = cube(0.5);
    let b = cube(0.5);
    let scene = Scene::new(
        &a,
        position(0.0, 0.0, 0.0),
        position(3.0, 0.0, 0.0),
        &b,
        position(0.4, 0.0, 0.0),
        position(0.4, 0.0, 0.0),
    );
    assert!(scene.toi().is_none());
}

#[test]
fn contact_normal_comes_from_the_closest_points_not_the_centres() {
    // B is offset sideways, so the centre-to-centre direction is tilted while
    // the faces that meet are perpendicular to x.
    let a = cube(0.5);
    let b = cube(0.5);
    let scene = Scene::new(
        &a,
        position(0.0, 0.0, 0.0),
        position(2.0, 0.0, 0.0),
        &b,
        position(2.0, 0.6, 0.0),
        position(2.0, 0.6, 0.0),
    );
    let hit = scene.toi().expect("the cubes meet face to face");
    assert!(hit.toi <= 0.5 && hit.toi > 0.5 - 1e-5, "toi {}", hit.toi);
    assert!(hit.normal.x > 0.9999, "{:?}", hit.normal);
    assert!((hit.witness_b.x - hit.witness_a.x).abs() < 1e-5);
    scene.check(Some(&hit), "offset cubes");
}

#[test]
fn off_centre_shape_swinging_about_the_body_origin_is_caught() {
    // A sphere of radius 0.5 whose centre is 2 away from the body origin turns
    // by 90 degrees about z and sweeps an arc of radius 2. A bound built from
    // the half extents of its box (0.87) would let it tunnel through the target
    // on the arc.
    let mut swinging = Compound::new(Vec::new());
    swinging.children.push((
        position(2.0, 0.0, 0.0),
        Arc::new(Sphere::new(0.5)) as Arc<dyn Shape>,
    ));
    let target = Sphere::new(0.3);
    let scene = Scene::new(
        &swinging,
        position(0.0, 0.0, 0.0),
        rotated(0.0, 0.0, 0.0, [0.0, 0.0, 1.0], FRAC_PI_2),
        &target,
        position(2f64.sqrt(), 2f64.sqrt(), 0.0),
        position(2f64.sqrt(), 2f64.sqrt(), 0.0),
    );
    let hit = scene.toi().expect("the sphere swings into the target");
    scene.check(Some(&hit), "swinging sphere");
    // Centres are 0.8 apart when the swinging sphere has turned by
    // 45 degrees - 2 asin(0.8 / 4) = 21.9 degrees.
    let expected = (45.0 - 2.0 * (0.8_f64 / 4.0).asin().to_degrees()) / 90.0;
    assert!(
        (hit.toi - expected).abs() < 1e-3,
        "toi {} vs {expected}",
        hit.toi
    );
}

#[test]
fn half_turn_through_a_box_that_is_clear_at_both_ends_is_found() {
    // The bar is along x at the start and (turned by 180 degrees) at the end,
    // clear of the box both times; midway it points along y through the box.
    let bar = bar(1.0, 0.05);
    let target = cube(0.2);
    let scene = Scene::new(
        &bar,
        position(0.0, 0.0, 0.0),
        rotated(0.0, 0.0, 0.0, [0.0, 0.0, 1.0], PI),
        &target,
        position(0.0, 0.8, 0.0),
        position(0.0, 0.8, 0.0),
    );
    assert!(scene.gap(0.0).is_some() && scene.gap(1.0).is_some());
    assert!(
        scene.gap(0.5).is_none(),
        "the bar must pass through the box"
    );
    let reference = scene.first_overlap(4000).expect("overlap midway");
    let hit = scene.toi().expect("the sweep passes through the box");
    scene.check(Some(&hit), "half turn");
    assert!(
        (hit.toi - reference).abs() < 1e-3,
        "{} vs {reference}",
        hit.toi
    );

    // With almost no iterations to spend, the exhaustion fallback still finds
    // the first contact instead of reporting no impact.
    for budget in [1, 2, 3] {
        let late = time_of_impact_with_budget(&scene.pair(), budget)
            .unwrap_or_else(|| panic!("budget {budget}: the pass-through was lost"));
        assert!(
            (late.toi - hit.toi).abs() < 1e-4,
            "budget {budget}: {} vs {}",
            late.toi,
            hit.toi
        );
        scene.check(Some(&late), "half turn, small budget");
    }
}

#[test]
fn budget_exhaustion_without_any_overlap_reports_no_impact() {
    // The tip of the bar swings on a circle of radius ~1.0 past a box whose
    // nearest point is 1.17 from the pivot: the bar needs about 4 steps to get
    // through the interval, so a small budget runs out while the gap is open.
    let bar = bar(1.0, 0.05);
    let target = cube(0.2);
    let scene = Scene::new(
        &bar,
        position(0.0, 0.0, 0.0),
        rotated(0.0, 0.0, 0.0, [0.0, 0.0, 1.0], FRAC_PI_2),
        &target,
        position(1.3, 0.6, 0.0),
        position(1.3, 0.6, 0.0),
    );
    assert!(scene.first_overlap(4000).is_none());
    for budget in [1, 2, 3, MAX_CCD_ITERATIONS] {
        assert!(
            time_of_impact_with_budget(&scene.pair(), budget).is_none(),
            "budget {budget}"
        );
    }
}

#[test]
fn exhausted_budget_bisects_a_hit_at_the_end_of_the_step() {
    // Two cubes closing at 4 per step from a gap of 3.5: the first step lands
    // on the contact, a budget of one still bisects the overlap at the end.
    let a = cube(0.5);
    let b = cube(0.5);
    let scene = Scene::new(
        &a,
        position(0.0, 0.0, 0.0),
        position(4.0, 0.0, 0.0),
        &b,
        position(4.5, 0.0, 0.0),
        position(4.5, 0.0, 0.0),
    );
    let exact = 3.5 / 4.0;
    for budget in [1, 2, MAX_CCD_ITERATIONS] {
        let hit = time_of_impact_with_budget(&scene.pair(), budget).expect("hit");
        assert!(
            hit.toi <= exact && exact - hit.toi < 1e-5,
            "budget {budget}: toi {} vs {exact}",
            hit.toi
        );
    }
}

#[test]
fn rotation_reach_uses_the_distance_from_the_rotation_axis() {
    // A thin bar turned about its own long axis hardly moves: the reach is the
    // swing of its corners around that axis, not of its tips around the origin.
    let thin = bar(1.0, 0.05);
    let about_own_axis = rotated(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], FRAC_PI_2);
    let reach = rotation_reach(&thin, &position(0.0, 0.0, 0.0), &about_own_axis);
    let expected = FRAC_PI_2 * (0.05_f64 * 2f64.sqrt());
    assert!((reach - expected).abs() < 1e-12, "{reach} vs {expected}");
    // Turned about a perpendicular axis the tips swing on a circle of radius ~1.
    let broadside = rotated(0.0, 0.0, 0.0, [0.0, 0.0, 1.0], FRAC_PI_2);
    let reach = rotation_reach(&thin, &position(0.0, 0.0, 0.0), &broadside);
    let expected = FRAC_PI_2 * (1.0_f64 * 1.0 + 0.05 * 0.05).sqrt();
    assert!((reach - expected).abs() < 1e-12, "{reach} vs {expected}");
    // No rotation, no reach.
    let still = rotation_reach(&thin, &position(0.0, 0.0, 0.0), &position(5.0, 0.0, 0.0));
    assert_eq!(still, 0.0);
}

#[test]
fn rotation_reach_is_measured_in_body_coordinates() {
    // The bar lies along world z (local x turned by 90 degrees about y) and the
    // world rotation is about z: the bar turns about its own long axis. The
    // axis of the relative rotation in body coordinates is local x.
    let thin = bar(1.0, 0.05);
    let start = rotated(0.0, 0.0, 0.0, [0.0, 1.0, 0.0], FRAC_PI_2);
    let end = Transform::new(
        Vec3::zeros(),
        Transform::from_axis_angle(Vec3::zeros(), Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2).rotation
            * start.rotation,
    );
    let reach = rotation_reach(&thin, &start, &end);
    let expected = FRAC_PI_2 * (0.05_f64 * 2f64.sqrt());
    assert!((reach - expected).abs() < 1e-9, "{reach} vs {expected}");
}

#[test]
fn rotation_reach_bounds_and_matches_the_speed_of_every_corner() {
    let mut rng = TestRng(0x9e37_79b9_7f4a_7c15);
    for case in 0..200 {
        let axis_start = rng.direction();
        let axis_end = rng.direction();
        let start = rotated(0.0, 0.0, 0.0, axis_start, rng.range(-PI, PI));
        let end = rotated(0.0, 0.0, 0.0, axis_end, rng.range(-PI, PI));
        let half = [
            rng.range(0.1, 1.0),
            rng.range(0.1, 1.0),
            rng.range(0.1, 1.0),
        ];
        let shape = BoxShape::new(Vec3::new(half[0], half[1], half[2]));
        let reach = rotation_reach(&shape, &start, &end);
        let dt = 1e-5;
        let mut fastest: f64 = 0.0;
        for k in 0..5 {
            let t = k as f64 / 5.0 + 0.07;
            let before = interpolate_transform(&start, &end, t).rotation;
            let after = interpolate_transform(&start, &end, t + dt).rotation;
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        let corner = Vec3::new(sx * half[0], sy * half[1], sz * half[2]);
                        let moved = (after * corner - before * corner).norm();
                        fastest = fastest.max(moved / dt);
                    }
                }
            }
        }
        assert!(
            fastest <= reach * (1.0 + 1e-6) + 1e-9,
            "case {case}: a corner moves at {fastest} > reach {reach}"
        );
        // The bound is the speed of the farthest corner, so it is not loose.
        assert!(
            fastest >= reach * (1.0 - 1e-3) - 1e-9,
            "case {case}: reach {reach} overestimates the fastest corner {fastest}"
        );
    }
}

fn unit_quaternion(x: Real, y: Real, z: Real, w: Real) -> UnitQuaternion<Real> {
    UnitQuaternion::new_unchecked(Vector4::new(x, y, z, w).into())
}

#[test]
fn boxes_that_make_the_boolean_gjk_cycle_are_still_separated() {
    // Two boxes 5 cm apart at t = 0.16375 of this motion. The boolean GJK search
    // behind `Gjk::query` cycles on this pose (and on a third of the poses
    // around it) and reports an overlap, which used to end the advancement with
    // a contact that is 5 cm away from the box.
    let a = BoxShape::new(Vec3::new(
        0.5534889658557196,
        0.4575209654543864,
        0.5906233102470289,
    ));
    let b = BoxShape::new(Vec3::new(
        0.6768498634904083,
        0.603573047117394,
        0.42833248894973697,
    ));
    let a_start = Transform::new(
        Vec3::new(1.1114619362174447, -0.6054764456935344, -1.8113787917232025),
        unit_quaternion(
            0.14684047788807378,
            0.063036380525063,
            -0.2799488675996692,
            0.9466218465224455,
        ),
    );
    let a_end = Transform::new(
        Vec3::new(-1.3287020872822946, -3.2795886393096194, 0.9602770436694468),
        unit_quaternion(
            0.34143561213686063,
            -0.7846524393283535,
            -0.4971013388474116,
            0.14364028382352556,
        ),
    );
    let b_pose = Transform::new(
        Vec3::new(0.5733001993630653, -0.8906354849400366, 0.26200218994569546),
        unit_quaternion(
            -0.22250429463830823,
            -0.5343377747627323,
            0.6367187707557345,
            0.5094744235939886,
        ),
    );
    let scene = Scene::new(&a, a_start, a_end, &b, b_pose.clone(), b_pose);

    let gap = scene
        .gap(0.16375)
        .expect("the boxes are apart at t = 0.16375");
    assert!(gap > 0.054 && gap < 0.057, "gap {gap}");
    // Neighbouring poses around it are all separated, with a gap that shrinks
    // smoothly.
    let gaps: Vec<Real> = (0..400)
        .map(|k| 0.16 + 1e-5 * k as Real)
        .map(|t| {
            scene
                .gap(t)
                .unwrap_or_else(|| panic!("overlap reported at t = {t}"))
        })
        .collect();
    assert!(gaps.windows(2).all(|w| w[1] < w[0] + 1e-9));

    // A contact, if there is one, is a real one (the check requires the boxes to
    // touch at the reported time) and not the bogus one at 0.1645, where they
    // are still 5 cm apart.
    let result = scene.toi();
    scene.check(result.as_ref(), "cycling boolean GJK");
    if let Some(hit) = result {
        assert!(hit.toi > 0.166, "toi {}", hit.toi);
    }
}

fn random_shape(rng: &mut TestRng) -> Box<dyn Shape> {
    match rng.index(3) {
        0 => Box::new(Sphere::new(rng.range(0.3, 0.8))),
        1 => Box::new(BoxShape::new(Vec3::new(
            rng.range(0.2, 0.7),
            rng.range(0.2, 0.7),
            rng.range(0.2, 0.7),
        ))),
        _ => Box::new(Capsule::new(rng.range(0.2, 0.4), rng.range(0.3, 0.7))),
    }
}

fn random_orientation(rng: &mut TestRng, at: Point) -> Transform {
    let axis = rng.direction();
    rotated(at[0], at[1], at[2], axis, rng.range(-PI, PI))
}

/// A pose near `base`: moved by up to `travel` and turned to a random orientation.
fn random_end(rng: &mut TestRng, base: &Transform, travel: f64) -> Transform {
    let offset = rng.point(travel);
    let at = [
        base.position.x + offset[0],
        base.position.y + offset[1],
        base.position.z + offset[2],
    ];
    random_orientation(rng, at)
}

#[test]
fn time_of_impact_agrees_with_dense_sampling_for_random_motions() {
    let mut rng = TestRng(0x1234_5678_9abc_def1);
    let (mut hits, mut misses) = (0, 0);
    for case in 0..500 {
        let shape_a = random_shape(&mut rng);
        let shape_b = random_shape(&mut rng);
        let at_a = rng.point(1.5);
        let a_start = random_orientation(&mut rng, at_a);
        let a_end = random_end(&mut rng, &a_start, 2.5);
        let at_b = rng.point(1.5);
        let b_start = random_orientation(&mut rng, at_b);
        let b_end = if rng.unit() < 0.5 {
            b_start.clone()
        } else {
            random_end(&mut rng, &b_start, 2.5)
        };
        let scene = Scene::new(&*shape_a, a_start, a_end, &*shape_b, b_start, b_end);
        if scene.gap(0.0).is_none() {
            continue;
        }
        let result = scene.toi();
        scene.check(result.as_ref(), &format!("random case {case}"));
        if result.is_some() {
            hits += 1;
        } else {
            misses += 1;
        }
    }
    assert!(hits >= 30 && misses >= 30, "hits {hits}, misses {misses}");
}

// ---------------------------------------------------------------------------
// F0070: gjk_distance_approx is a lower bound and conservative advancement
// does not overshoot
// ---------------------------------------------------------------------------

fn sphere_support(centre: Point, radius: f64) -> impl Fn(Point) -> Point {
    move |d| add3(centre, scale3(norm3(d), radius))
}

fn box_support(centre: Point, half: Point) -> impl Fn(Point) -> Point {
    move |d| {
        [
            centre[0] + if d[0] >= 0.0 { half[0] } else { -half[0] },
            centre[1] + if d[1] >= 0.0 { half[1] } else { -half[1] },
            centre[2] + if d[2] >= 0.0 { half[2] } else { -half[2] },
        ]
    }
}

/// Capsule: segment `p0`-`p1` grown by `radius`.
fn capsule_support_at(p0: Point, p1: Point, radius: f64) -> impl Fn(Point) -> Point {
    move |d| capsule_support(p0, p1, radius, d)
}

fn distance_point_segment(p: Point, s0: Point, s1: Point) -> f64 {
    len3(sub3(p, closest_point_on_segment(s0, s1, p)))
}

/// Distance between two segments (endpoint-to-segment candidates plus the
/// interior pair of the infinite lines).
fn distance_segment_segment(a0: Point, a1: Point, b0: Point, b1: Point) -> f64 {
    let mut best = distance_point_segment(a0, b0, b1)
        .min(distance_point_segment(a1, b0, b1))
        .min(distance_point_segment(b0, a0, a1))
        .min(distance_point_segment(b1, a0, a1));
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
            best = best.min(len3(sub3(pa, pb)));
        }
    }
    best
}

fn distance_point_box(p: Point, centre: Point, half: Point) -> f64 {
    let mut sum = 0.0;
    for k in 0..3 {
        let outside = ((p[k] - centre[k]).abs() - half[k]).max(0.0);
        sum += outside * outside;
    }
    sum.sqrt()
}

fn distance_box_box(c1: Point, h1: Point, c2: Point, h2: Point) -> f64 {
    let mut sum = 0.0;
    for k in 0..3 {
        let outside = ((c1[k] - c2[k]).abs() - h1[k] - h2[k]).max(0.0);
        sum += outside * outside;
    }
    sum.sqrt()
}

/// `approx` must be a lower bound of `exact` and (for shapes that converge) close
/// to it.
fn assert_lower_bound(approx: f64, exact: f64, label: &str) {
    assert!(
        approx <= exact + 1e-9,
        "{label}: distance {approx} overestimates the true distance {exact}"
    );
    assert!(
        exact - approx <= 1e-7 * exact.max(1.0),
        "{label}: distance {approx} is far below the true distance {exact}"
    );
}

#[test]
fn diagonal_spheres_give_the_true_distance_not_the_first_support_estimate() {
    // The first support query of the Minkowski difference lies at distance
    // sqrt(34) = 5.83; the true gap of the unit spheres is 5 sqrt(2) - 2.
    let d = gjk_distance_approx(
        sphere_support([0.0; 3], 1.0),
        sphere_support([5.0, 5.0, 0.0], 1.0),
    );
    let exact = 5.0 * 2f64.sqrt() - 2.0;
    assert!((5.0710..=5.0712).contains(&d), "{d}");
    assert!(d <= exact + 1e-12, "{d} > {exact}");
    assert!(exact - d < 1e-8);
}

#[test]
fn gjk_distance_is_zero_for_touching_and_overlapping_shapes() {
    let touching = gjk_distance_approx(
        sphere_support([0.0; 3], 1.0),
        sphere_support([2.0, 0.0, 0.0], 1.0),
    );
    assert!(touching < 1e-6, "{touching}");
    let overlapping = gjk_distance_approx(
        box_support([0.0; 3], [1.0; 3]),
        box_support([0.5, 0.5, 0.5], [1.0; 3]),
    );
    assert_eq!(overlapping, 0.0);
    let nested = gjk_distance_approx(
        sphere_support([0.0; 3], 2.0),
        sphere_support([0.1, 0.0, 0.0], 0.5),
    );
    assert_eq!(nested, 0.0);
}

#[test]
fn gjk_distance_is_a_lower_bound_for_random_shape_pairs() {
    let mut rng = TestRng(0xdead_beef_0bad_f00d);
    for case in 0..400 {
        let c1 = rng.point(3.0);
        let c2 = rng.point(3.0);
        let (r1, r2) = (rng.range(0.2, 1.0), rng.range(0.2, 1.0));
        let (h1, h2) = (
            [
                rng.range(0.2, 1.0),
                rng.range(0.2, 1.0),
                rng.range(0.2, 1.0),
            ],
            [
                rng.range(0.2, 1.0),
                rng.range(0.2, 1.0),
                rng.range(0.2, 1.0),
            ],
        );
        let (s1, s2) = (
            add3(c1, scale3(rng.direction(), 0.8)),
            add3(c2, scale3(rng.direction(), 0.8)),
        );

        let sphere_sphere = (len3(sub3(c1, c2)) - r1 - r2).max(0.0);
        assert_lower_bound(
            gjk_distance_approx(sphere_support(c1, r1), sphere_support(c2, r2)),
            sphere_sphere,
            &format!("case {case}: sphere-sphere"),
        );

        let box_box = distance_box_box(c1, h1, c2, h2);
        assert_lower_bound(
            gjk_distance_approx(box_support(c1, h1), box_support(c2, h2)),
            box_box,
            &format!("case {case}: box-box"),
        );

        let box_sphere = (distance_point_box(c2, c1, h1) - r2).max(0.0);
        assert_lower_bound(
            gjk_distance_approx(box_support(c1, h1), sphere_support(c2, r2)),
            box_sphere,
            &format!("case {case}: box-sphere"),
        );

        let capsule_sphere = (distance_point_segment(c2, c1, s1) - r1 - r2).max(0.0);
        assert_lower_bound(
            gjk_distance_approx(capsule_support_at(c1, s1, r1), sphere_support(c2, r2)),
            capsule_sphere,
            &format!("case {case}: capsule-sphere"),
        );

        let capsule_capsule = (distance_segment_segment(c1, s1, c2, s2) - r1 - r2).max(0.0);
        assert_lower_bound(
            gjk_distance_approx(
                capsule_support_at(c1, s1, r1),
                capsule_support_at(c2, s2, r2),
            ),
            capsule_capsule,
            &format!("case {case}: capsule-capsule"),
        );
    }
}

#[test]
fn fast_sphere_does_not_tunnel_through_a_thin_box() {
    // Sphere of radius 0.1 closing at 100 per step on a box 0.01 thick. An
    // overestimated distance made the steps overshoot the box entirely.
    let support_a = ConservativeAdvancement::sphere_support(0.1);
    let support_b = ConservativeAdvancement::box_support([1.0, 0.005, 1.0]);
    let start = [-1.0, 5.0, -0.7];
    let velocity = [2.0, -100.0, 1.4];
    // The sphere centre reaches y = 0.005 + 0.1 above the box at this time (the
    // sphere stays over the box, its footprint x in -0.9, z in -0.63).
    let exact = (5.0 - 0.105) / 100.0;

    let toi = conservative_advancement(
        &support_a, &support_b, start, velocity, [0.0; 3], [0.0; 3], 1000.0, 64,
    )
    .expect("the sphere reaches the box");
    assert!(
        toi <= exact + 1e-12,
        "toi {toi} is past the contact at {exact}"
    );
    assert!(
        exact - toi < 2e-6,
        "toi {toi} is far before the contact at {exact}"
    );

    let solver = ConservativeAdvancement::new();
    let toi = solver
        .compute_toi(
            &support_a, &support_b, start, velocity, 0.1, [0.0; 3], [0.0; 3], 1.5,
        )
        .expect("the sphere reaches the box");
    assert!(
        toi <= exact + 1e-12,
        "toi {toi} is past the contact at {exact}"
    );
    assert!(
        exact - toi < 2e-5,
        "toi {toi} is far before the contact at {exact}"
    );
}

#[test]
fn conservative_advancement_matches_the_analytic_sphere_toi() {
    let mut rng = TestRng(0x0123_4567_89ab_cdef);
    let (mut hits, mut misses) = (0, 0);
    for case in 0..400 {
        let centre_a = rng.point(3.0);
        let to_b = rng.direction();
        let distance = rng.range(2.0, 6.0);
        let centre_b = add3(centre_a, scale3(to_b, distance));
        let vel_a = rng.point(2.0);
        // The relative motion points roughly at A (B approaches A) with a random
        // sideways component, so that about half of the pairs touch.
        let closing = rng.range(0.5, 1.5) * distance;
        let sideways = scale3(rng.direction(), rng.range(0.0, 0.5) * distance);
        let vel_b = add3(vel_a, add3(scale3(to_b, -closing), sideways));
        let (ra, rb) = (rng.range(0.2, 0.8), rng.range(0.2, 0.8));
        let relative = sub3(vel_b, vel_a);
        let offset = sub3(centre_b, centre_a);
        // Smallest centre distance during the step, for ruling out borderline cases.
        let along = (-dot3(offset, relative) / dot3(relative, relative)).clamp(0.0, 1.0);
        let closest = len3(add3(offset, scale3(relative, along))) - ra - rb;
        if closest.abs() < 1e-3 || len3(relative) < 1.0 || len3(sub3(centre_b, centre_a)) < ra + rb
        {
            continue;
        }
        let exact = sphere_sphere_toi(centre_a, vel_a, ra, centre_b, vel_b, rb);
        let found = conservative_advancement(
            sphere_support([0.0; 3], ra),
            sphere_support([0.0; 3], rb),
            centre_a,
            vel_a,
            centre_b,
            vel_b,
            1000.0,
            256,
        );
        match (exact, found) {
            (Some(exact), Some(found)) => {
                hits += 1;
                assert!(
                    found <= exact + 1e-9,
                    "case {case}: {found} is past the contact at {exact}"
                );
                assert!(
                    exact - found < 2e-3,
                    "case {case}: {found} is far before {exact}"
                );
            }
            (None, None) => misses += 1,
            (exact, found) => panic!("case {case}: analytic {exact:?} but advancement {found:?}"),
        }
    }
    assert!(hits >= 30 && misses >= 30, "hits {hits}, misses {misses}");
}

#[test]
fn capsule_capsule_toi_stops_at_the_contact() {
    // Two vertical capsules of radius 0.3, 5 apart; A moves 10 toward B.
    let toi = capsule_capsule_toi(
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [10.0, 0.0, 0.0],
        0.3,
        [5.0, -1.0, 0.0],
        [5.0, 1.0, 0.0],
        [0.0; 3],
        0.3,
        64,
    )
    .expect("the capsules meet");
    let exact = (5.0 - 0.6) / 10.0;
    assert!(toi <= exact + 1e-12, "{toi} > {exact}");
    assert!(exact - toi < 2e-5, "{toi} vs {exact}");
}
