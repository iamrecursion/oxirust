// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`FrameInterpolator`]: cubic Hermite interpolation must
//! use slopes at the keyframes (a smoothstep between the values stops the motion
//! at every keyframe).

use super::types::{FrameInterpolator, InterpolationMethod};

fn hermite() -> FrameInterpolator {
    FrameInterpolator::new(InterpolationMethod::CubicHermite)
}

#[test]
fn constant_velocity_motion_is_reproduced_exactly_with_explicit_velocities() {
    // x(t) = 2 t sampled at t = 0 and t = 1, velocity 2 at both ends.  At
    // alpha = 1/4 every Hermite weight is a dyadic rational, so the result is
    // exact: 0.5.
    let mut fi = hermite();
    fi.update_with_velocities(vec![0.0], vec![2.0], 0.0, vec![2.0], vec![2.0], 1.0);
    assert_eq!(fi.interpolate(0.25), vec![0.5]);
    assert_eq!(fi.interpolate(0.5), vec![1.0]);
    assert_eq!(fi.interpolate(0.75), vec![1.5]);
}

#[test]
fn constant_velocity_motion_is_reproduced_exactly_from_three_frames() {
    // Frames of x(t) = 2 t at t = 0, 1, 2 (Catmull-Rom start slope from the
    // frame before `prev`, secant at the end).  The smoothstep that the old
    // implementation computed gives 2.3125 here.
    let mut fi = hermite();
    fi.update(vec![0.0], 0.0, vec![2.0], 1.0);
    fi.update(vec![2.0], 1.0, vec![4.0], 2.0);
    assert_eq!(fi.interpolate(1.25), vec![2.5]);
    assert_eq!(fi.interpolate(1.5), vec![3.0]);
}

#[test]
fn two_frames_without_history_interpolate_along_the_straight_line() {
    let mut fi = hermite();
    fi.update(vec![0.0, 10.0], 0.0, vec![2.0, 6.0], 1.0);
    assert_eq!(fi.interpolate(0.25), vec![0.5, 9.0]);
    assert_eq!(fi.interpolate(0.5), vec![1.0, 8.0]);
}

#[test]
fn cubic_motion_is_reproduced_exactly_from_true_velocities() {
    // x(t) = t^3 on [1, 2]: x = 1, 8 and x' = 3 t^2 = 3, 12.  A cubic Hermite
    // curve through values and slopes reproduces any cubic; at t = 1.25 the
    // exact value is 125/64 = 1.953125 (the old smoothstep gives 2.09375).
    let mut fi = hermite();
    fi.update_with_velocities(vec![1.0], vec![3.0], 1.0, vec![8.0], vec![12.0], 2.0);
    assert_eq!(fi.interpolate(1.25), vec![1.953125]);
    assert_eq!(fi.interpolate(1.5), vec![3.375]);
    assert_eq!(fi.interpolate(1.75), vec![5.359375]);
}

#[test]
fn explicit_velocities_are_scaled_by_the_segment_duration() {
    // x(t) = 4 t^2 on [3, 5]: x = 36, 100; x' = 8 t = 24, 40.
    let mut fi = hermite();
    fi.update_with_velocities(vec![36.0], vec![24.0], 3.0, vec![100.0], vec![40.0], 5.0);
    for (t, expected) in [
        (3.0, 36.0),
        (3.5, 49.0),
        (4.0, 64.0),
        (4.5, 81.0),
        (5.0, 100.0),
    ] {
        let value = fi.interpolate(t)[0];
        assert!(
            (value - expected).abs() < 1e-12,
            "t = {t}: {value} vs {expected}"
        );
    }
}

#[test]
fn hermite_passes_through_both_keyframes() {
    let mut fi = hermite();
    fi.update(vec![1.5, -2.0], 4.0, vec![7.25, 3.0], 6.0);
    fi.update(vec![7.25, 3.0], 6.0, vec![-1.0, 0.5], 9.0);
    assert_eq!(fi.interpolate(6.0), vec![7.25, 3.0]);
    assert_eq!(fi.interpolate(9.0), vec![-1.0, 0.5]);
    // Outside the segment the parameter is clamped.
    assert_eq!(fi.interpolate(0.0), vec![7.25, 3.0]);
    assert_eq!(fi.interpolate(100.0), vec![-1.0, 0.5]);
}

#[test]
fn hermite_has_a_non_zero_velocity_at_the_keyframes() {
    // The smoothstep has zero slope at both ends; with the straight-line
    // tangents the curve moves at the keyframe speed immediately.
    let mut fi = hermite();
    fi.update(vec![0.0], 0.0, vec![1.0], 1.0);
    let eps = 1e-6;
    let start_speed = (fi.interpolate(eps)[0] - fi.interpolate(0.0)[0]) / eps;
    let end_speed = (fi.interpolate(1.0)[0] - fi.interpolate(1.0 - eps)[0]) / eps;
    assert!(
        (start_speed - 1.0).abs() < 1e-3,
        "start speed {start_speed}"
    );
    assert!((end_speed - 1.0).abs() < 1e-3, "end speed {end_speed}");
}

#[test]
fn a_sliding_window_gives_a_continuous_velocity_across_keyframes() {
    // Samples of x(t) = sin(t) at t = 0, 1, 2, 3 fed as a sliding window.
    let samples: Vec<(f64, f64)> = (0..4).map(|k| (k as f64, (k as f64).sin())).collect();
    let mut fi = hermite();
    fi.update(
        vec![samples[0].1],
        samples[0].0,
        vec![samples[1].1],
        samples[1].0,
    );
    fi.update(
        vec![samples[1].1],
        samples[1].0,
        vec![samples[2].1],
        samples[2].0,
    );
    let eps = 1e-6;
    // Speed at the end of the segment [1, 2], estimated from inside the segment.
    let end_of_first = (fi.interpolate(2.0)[0] - fi.interpolate(2.0 - eps)[0]) / eps;
    fi.update(
        vec![samples[2].1],
        samples[2].0,
        vec![samples[3].1],
        samples[3].0,
    );
    let start_of_next = (fi.interpolate(2.0 + eps)[0] - fi.interpolate(2.0)[0]) / eps;
    // Start slope of [2, 3] is the Catmull-Rom slope through t = 1 and t = 3.
    let catmull = (samples[3].1 - samples[1].1) / 2.0;
    assert!(
        (start_of_next - catmull).abs() < 1e-4,
        "{start_of_next} vs {catmull}"
    );
    // The end slope of [1, 2] was the secant of that segment; the Catmull-Rom
    // start slope is a better estimate of cos(2) than the secant.
    let true_speed = 2.0_f64.cos();
    assert!((start_of_next - true_speed).abs() < (end_of_first - true_speed).abs());
}

#[test]
fn non_contiguous_updates_do_not_reuse_stale_history() {
    let mut fi = hermite();
    fi.update(vec![100.0], 0.0, vec![200.0], 1.0);
    // A jump in time: the old frames must not influence the new segment.
    fi.update(vec![0.0], 5.0, vec![2.0], 6.0);
    assert_eq!(fi.interpolate(5.25), vec![0.5]);
    // Time running backwards is not a sliding window either.
    fi.update(vec![0.0], 4.0, vec![2.0], 5.0);
    assert_eq!(fi.interpolate(4.25), vec![0.5]);
}

#[test]
fn plain_update_discards_velocities_from_an_earlier_call() {
    let mut fi = hermite();
    fi.update_with_velocities(vec![0.0], vec![50.0], 0.0, vec![2.0], vec![50.0], 1.0);
    assert!(
        fi.interpolate(0.25)[0] > 1.0,
        "explicit slopes are followed"
    );
    fi.update(vec![0.0], 0.0, vec![2.0], 1.0);
    assert_eq!(fi.interpolate(0.25), vec![0.5]);
}

#[test]
fn components_without_a_velocity_fall_back_to_finite_differences() {
    let mut fi = hermite();
    fi.update_with_velocities(
        vec![0.0, 0.0],
        vec![2.0],
        0.0,
        vec![2.0, 4.0],
        vec![2.0],
        1.0,
    );
    // First component: exact slopes (constant velocity 2); second: secant.
    assert_eq!(fi.interpolate(0.25), vec![0.5, 1.0]);
}

#[test]
fn degenerate_time_span_returns_the_next_frame() {
    let mut fi = hermite();
    fi.update(vec![1.0], 3.0, vec![9.0], 3.0);
    assert_eq!(fi.interpolate(3.0), vec![9.0]);
}

#[test]
fn linear_and_nearest_are_unchanged() {
    let mut linear = FrameInterpolator::new(InterpolationMethod::Linear);
    linear.update(vec![0.0, 0.0], 0.0, vec![2.0, 4.0], 1.0);
    assert_eq!(linear.interpolate(0.25), vec![0.5, 1.0]);
    let mut nearest = FrameInterpolator::new(InterpolationMethod::Nearest);
    nearest.update(vec![0.0], 0.0, vec![1.0], 1.0);
    assert_eq!(nearest.interpolate(0.4), vec![0.0]);
    assert_eq!(nearest.interpolate(0.6), vec![1.0]);
}

#[test]
fn interpolation_error_of_exact_cubic_data_vanishes() {
    let mut fi = hermite();
    fi.update_with_velocities(vec![1.0], vec![3.0], 1.0, vec![8.0], vec![12.0], 2.0);
    let error = fi.interpolation_error(&[1.953125], 1.25);
    assert!(error < 1e-15, "error {error}");
}
