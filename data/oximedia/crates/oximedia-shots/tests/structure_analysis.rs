//! Ground-truth tests for the structural analysis layers built on top of a
//! shot list: continuity checking, scene segmentation, and editing rhythm.
//!
//! Each test constructs `Shot` values by hand (via [`Shot::new`] plus field
//! mutation) so the inputs to the analyzers are exactly known, then asserts
//! the documented behaviour of:
//! - [`oximedia_shots::continuity::ContinuityChecker`]
//! - [`oximedia_shots::scene::SceneDetector`]
//! - [`oximedia_shots::pattern::RhythmAnalyzer`]

use oximedia_core::types::{Rational, Timestamp};
use oximedia_shots::continuity::{ContinuityChecker, IssueType, Severity};
use oximedia_shots::pattern::RhythmAnalyzer;
use oximedia_shots::scene::SceneDetector;
use oximedia_shots::types::{
    CameraAngle, CameraMovement, CoverageType, MovementType, Shot, ShotType, TransitionType,
};

/// 30 fps timebase shared by every synthetic shot.
fn tb() -> Rational {
    Rational::new(1, 30)
}

/// Build a shot spanning `dur_frames` frames starting at frame 0.
///
/// All classification fields keep their `Shot::new` defaults
/// (MediumShot / EyeLevel / Master / Cut, no movements); callers mutate the
/// fields relevant to the property under test.
fn make_shot(id: u64, dur_frames: i64) -> Shot {
    Shot::new(
        id,
        Timestamp::new(0, tb()),
        Timestamp::new(dur_frames, tb()),
    )
}

/// Attach a single full-shot camera movement to `shot`.
fn add_movement(shot: &mut Shot, movement_type: MovementType, confidence: f32, speed: f32) {
    shot.movements.push(CameraMovement {
        movement_type,
        start: 0.0,
        end: 1.0,
        confidence,
        speed,
    });
}

/// Test 6 — two consecutive shots with identical type/angle/coverage trip a
/// single `JumpCut` issue, reported against the *second* shot (id 2) at
/// Medium severity.
#[test]
fn jump_cut_detected_for_identical_consecutive_shots() {
    let checker = ContinuityChecker::new();

    let mut a = make_shot(1, 30);
    a.shot_type = ShotType::CloseUp;
    a.angle = CameraAngle::EyeLevel;
    a.coverage = CoverageType::Single;

    let mut b = make_shot(2, 30);
    b.shot_type = ShotType::CloseUp;
    b.angle = CameraAngle::EyeLevel;
    b.coverage = CoverageType::Single;

    let issues = checker.check_continuity(&[a, b]);

    let jump_cuts: Vec<_> = issues
        .iter()
        .filter(|i| i.issue_type == IssueType::JumpCut)
        .collect();
    assert_eq!(jump_cuts.len(), 1, "expected exactly one jump-cut issue");
    assert_eq!(jump_cuts[0].shot_id, 2, "jump cut must be tagged on shot 2");
    assert_eq!(
        jump_cuts[0].severity,
        Severity::Medium,
        "jump cut severity must be Medium"
    );
}

/// Test 7 — opposed pan directions (PanRight then PanLeft) flag exactly one
/// `CrossingTheLine` issue at High severity; same-direction pans flag none.
#[test]
fn axis_crossing_detected_for_opposed_pans_only() {
    let checker = ContinuityChecker::new();

    // Opposed: PanRight -> PanLeft.
    let mut a = make_shot(1, 30);
    add_movement(&mut a, MovementType::PanRight, 0.9, 0.8);
    let mut b = make_shot(2, 30);
    add_movement(&mut b, MovementType::PanLeft, 0.9, 0.8);

    let issues = checker.check_continuity(&[a, b]);
    let crossings: Vec<_> = issues
        .iter()
        .filter(|i| i.issue_type == IssueType::CrossingTheLine)
        .collect();
    assert_eq!(
        crossings.len(),
        1,
        "opposed pans must flag one axis crossing"
    );
    assert_eq!(
        crossings[0].severity,
        Severity::High,
        "axis crossing severity must be High"
    );

    // Same direction: PanRight -> PanRight => no crossing.
    let mut c = make_shot(3, 30);
    add_movement(&mut c, MovementType::PanRight, 0.9, 0.8);
    let mut d = make_shot(4, 30);
    add_movement(&mut d, MovementType::PanRight, 0.9, 0.8);

    let same_dir_issues = checker.check_continuity(&[c, d]);
    assert!(
        same_dir_issues
            .iter()
            .all(|i| i.issue_type != IssueType::CrossingTheLine),
        "same-direction pans must not flag an axis crossing"
    );
}

/// Test 8 — three shots that differ in type, angle, AND coverage (and carry
/// no movements) produce no continuity issues at all.
#[test]
fn distinct_shots_without_movement_have_no_issues() {
    let checker = ContinuityChecker::new();

    let mut a = make_shot(1, 30);
    a.shot_type = ShotType::CloseUp;
    a.angle = CameraAngle::High;
    a.coverage = CoverageType::Single;

    let mut b = make_shot(2, 30);
    b.shot_type = ShotType::LongShot;
    b.angle = CameraAngle::Low;
    b.coverage = CoverageType::TwoShot;

    let mut c = make_shot(3, 30);
    c.shot_type = ShotType::MediumShot;
    c.angle = CameraAngle::EyeLevel;
    c.coverage = CoverageType::Master;

    let issues = checker.check_continuity(&[a, b, c]);
    assert!(
        issues.is_empty(),
        "fully distinct, movement-free shots must yield no issues, got {issues:?}"
    );
}

/// Test 9 — a `[Cut, FadeToBlack, FadeFromBlack]` transition list splits into
/// at least two scenes, and the scenes form a partition of the shot list
/// (every shot belongs to exactly one scene).
#[test]
fn fade_transitions_partition_shots_into_scenes() {
    let detector = SceneDetector::new();

    let mut s0 = make_shot(0, 60);
    s0.transition = TransitionType::Cut;
    let mut s1 = make_shot(1, 60);
    s1.transition = TransitionType::FadeToBlack;
    let mut s2 = make_shot(2, 60);
    s2.transition = TransitionType::FadeFromBlack;

    let shots = [s0, s1, s2];
    let scenes = detector.detect_scenes(&shots);

    assert!(
        scenes.len() >= 2,
        "fade transitions should yield at least 2 scenes, got {}",
        scenes.len()
    );

    let total_in_scenes: usize = scenes.iter().map(|sc| sc.shots.len()).sum();
    assert_eq!(
        total_in_scenes,
        shots.len(),
        "scenes must partition all {} shots, summed to {total_in_scenes}",
        shots.len()
    );
}

/// Test 10 — a boundary needs BOTH coverage and shot-type to change: a shot
/// that changes both starts a new scene (2 scenes), while a shot that changes
/// only coverage stays in the same scene (1 scene).
#[test]
fn scene_boundary_requires_both_coverage_and_type_change() {
    let detector = SceneDetector::new();

    // Both coverage AND type change between shot 0 and shot 1 => 2 scenes.
    let mut a0 = make_shot(0, 60);
    a0.shot_type = ShotType::CloseUp;
    a0.coverage = CoverageType::Single;
    a0.transition = TransitionType::Cut;
    let mut a1 = make_shot(1, 60);
    a1.shot_type = ShotType::LongShot;
    a1.coverage = CoverageType::Master;
    a1.transition = TransitionType::Cut;

    let both_changed = detector.detect_scenes(&[a0, a1]);
    assert_eq!(
        both_changed.len(),
        2,
        "changing both coverage and type must create a boundary (2 scenes)"
    );

    // Only coverage changes (type stays CloseUp) => single scene.
    let mut b0 = make_shot(0, 60);
    b0.shot_type = ShotType::CloseUp;
    b0.coverage = CoverageType::Single;
    b0.transition = TransitionType::Cut;
    let mut b1 = make_shot(1, 60);
    b1.shot_type = ShotType::CloseUp;
    b1.coverage = CoverageType::Master;
    b1.transition = TransitionType::Cut;

    let coverage_only = detector.detect_scenes(&[b0, b1]);
    assert_eq!(
        coverage_only.len(),
        1,
        "changing coverage alone must NOT create a boundary (1 scene)"
    );
}

/// Test 11 — rhythm on shot durations `[4, 1, 1, 4]` seconds (encoded as
/// frame counts at 1/30). Asserts the exact accel/decel counts and the
/// closed-form `beat` and `regularity` values.
#[test]
fn rhythm_known_durations_produce_exact_metrics() {
    let analyzer = RhythmAnalyzer::new();

    // 4s = 120 frames, 1s = 30 frames at a 1/30 timebase.
    let durations_frames = [120i64, 30, 30, 120];
    let shots: Vec<Shot> = durations_frames
        .iter()
        .enumerate()
        .map(|(i, &df)| make_shot(i as u64, df))
        .collect();

    let r = analyzer.analyze(&shots);

    assert_eq!(r.accelerations, 1, "expected exactly one acceleration");
    assert_eq!(r.decelerations, 1, "expected exactly one deceleration");

    // beat = shot_count / total_seconds = 4 / 10.0 = 0.4.
    let total_seconds = (120.0 + 30.0 + 30.0 + 120.0) / 30.0;
    let expected_beat = 4.0 / total_seconds;
    assert!(
        (r.beat - expected_beat).abs() < 1e-9,
        "beat {} should equal {expected_beat}",
        r.beat
    );

    // regularity = 1 / (1 + variance).  mean = 2.5s, variance = 2.25.
    let seconds = [4.0_f64, 1.0, 1.0, 4.0];
    let mean = seconds.iter().sum::<f64>() / seconds.len() as f64;
    let variance =
        seconds.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / seconds.len() as f64;
    let expected_regularity = (1.0 / (1.0 + variance)) as f32;
    assert!(
        (r.regularity - expected_regularity).abs() < 1e-6,
        "regularity {} should equal {expected_regularity}",
        r.regularity
    );
}

/// A tiny deterministic SplitMix64 PRNG.
///
/// Used to drive the property test below without adding any dependency: the
/// `oximedia-shots` crate does not list `rand` as a direct dependency, and
/// integration tests may only name direct dependencies, so a self-contained
/// seeded generator keeps the test reproducible and dependency-free while
/// still exercising hundreds of randomly-shaped duration vectors.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[lo, hi]` (inclusive).
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        debug_assert!(hi >= lo);
        let span = hi - lo + 1;
        lo + self.next_u64() % span
    }
}

/// Test 12 — property test: across many randomly-sized vectors of positive
/// shot durations, the rhythm invariants always hold:
/// `0.0 <= regularity <= 1.0`, `beat > 0`, and
/// `accelerations + decelerations <= len - 1`.
#[test]
fn rhythm_invariants_hold_for_random_durations() {
    let analyzer = RhythmAnalyzer::new();
    let mut rng = SplitMix64::new(0xC0FF_EE12_3456_789A);

    for _ in 0..500 {
        // 3..=50 shots, each with a strictly positive frame count (1..=300).
        let n = rng.range(3, 50) as usize;
        let shots: Vec<Shot> = (0..n)
            .map(|i| {
                let frames = rng.range(1, 300) as i64;
                make_shot(i as u64, frames)
            })
            .collect();

        let r = analyzer.analyze(&shots);

        assert!(
            (0.0..=1.0).contains(&r.regularity),
            "regularity {} out of [0,1] for n={n}",
            r.regularity
        );
        assert!(r.beat > 0.0, "beat {} must be positive for n={n}", r.beat);
        // Equivalent to `accelerations + decelerations <= n - 1` (n >= 3 here),
        // written as `< n` to satisfy clippy::int_plus_one.
        assert!(
            r.accelerations + r.decelerations < n,
            "accel({}) + decel({}) must be < n={n}",
            r.accelerations,
            r.decelerations
        );
    }
}
