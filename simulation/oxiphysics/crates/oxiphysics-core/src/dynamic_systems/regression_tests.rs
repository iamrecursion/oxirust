// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for `dynamic_systems`: the memory-free correlation sum,
//! the Theiler window of the Rosenstein estimator and the Hilbert-transform
//! instantaneous phase.

use super::*;
use std::f64::consts::{PI, TAU};

// ── helpers ──────────────────────────────────────────────────────────────────

/// `n` points of the Hénon attractor (classical parameters) after a transient.
fn henon_points(n: usize) -> Vec<Vec<f64>> {
    let map = HenonMap::new(1.4, 0.3);
    let (mut x, mut y) = (0.1, 0.1);
    for _ in 0..200 {
        (x, y) = map.apply(x, y);
    }
    (0..n)
        .map(|_| {
            (x, y) = map.apply(x, y);
            vec![x, y]
        })
        .collect()
}

/// The original algorithm: store every pairwise distance, sort, count.
fn correlation_dimension_reference(data: &[Vec<f64>], n_radii: usize) -> Vec<(f64, f64)> {
    let mut dists = Vec::new();
    for (i, a) in data.iter().enumerate() {
        for b in &data[i + 1..] {
            dists.push(squared_distance(a, b).sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let r_min = dists[0] * 1.1;
    let r_max = dists[dists.len() - 1] * 0.9;
    let total = dists.len() as f64;
    let mut out = Vec::new();
    for k in 0..n_radii {
        let r = r_min * (r_max / r_min).powf(k as f64 / n_radii.saturating_sub(1).max(1) as f64);
        let count = dists.partition_point(|&d| d <= r);
        let c = count as f64 / total;
        if c > 0.0 {
            out.push((r.ln(), c.ln()));
        }
    }
    out
}

/// Lorenz orbit (`σ = 10, ρ = 28, β = 8/3`) sampled every `dt` with RK4
/// sub-steps, after discarding a transient.
fn lorenz_orbit(n: usize, dt: f64) -> Vec<Vec<f64>> {
    let lorenz = LorenzSystem::standard();
    let step = |s: &[f64]| -> Vec<f64> {
        let h = dt / 4.0;
        let mut state = s.to_vec();
        for _ in 0..4 {
            let k1 = lorenz.rhs(&state);
            let s2: Vec<f64> = state
                .iter()
                .zip(k1.iter())
                .map(|(a, k)| a + 0.5 * h * k)
                .collect();
            let k2 = lorenz.rhs(&s2);
            let s3: Vec<f64> = state
                .iter()
                .zip(k2.iter())
                .map(|(a, k)| a + 0.5 * h * k)
                .collect();
            let k3 = lorenz.rhs(&s3);
            let s4: Vec<f64> = state
                .iter()
                .zip(k3.iter())
                .map(|(a, k)| a + h * k)
                .collect();
            let k4 = lorenz.rhs(&s4);
            state = (0..3)
                .map(|i| state[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
                .collect();
        }
        state
    };
    let mut state = vec![1.0, 1.0, 1.0];
    for _ in 0..1000 {
        state = step(&state);
    }
    (0..n)
        .map(|_| {
            state = step(&state);
            state.clone()
        })
        .collect()
}

/// Least-squares slope of `y(t)` over `t ∈ [first, last]`.
fn slope_over(curve: &[f64], first: usize, last: usize) -> f64 {
    let pairs: Vec<(f64, f64)> = (first..=last).map(|t| (t as f64, curve[t])).collect();
    linear_slope(&pairs)
}

/// Independent brute-force version of the Rosenstein curve: nearest neighbour
/// with `|i − j| >= window` among the points that can be followed for the full
/// `steps`, mean of `ln ‖x_{i+t} − x_{j+t}‖`.
fn rosenstein_reference(data: &[Vec<f64>], horizon: usize, window: usize) -> Vec<f64> {
    let n = data.len();
    let steps = horizon.min(n / 2);
    let followable = n - steps + 1;
    let window = window.max(1);
    let dist = |a: usize, b: usize| squared_distance(&data[a], &data[b]).sqrt();
    let mut sums = vec![0.0; steps];
    let mut pairs = 0usize;
    for i in 0..followable {
        let mut best: Option<(usize, f64)> = None;
        for j in 0..followable {
            if i.abs_diff(j) < window {
                continue;
            }
            let d = dist(i, j);
            if best.is_none_or(|(_, b)| d < b) {
                best = Some((j, d));
            }
        }
        if let Some((j, _)) = best {
            pairs += 1;
            for (t, sum) in sums.iter_mut().enumerate() {
                *sum += dist(i + t, j + t).ln();
            }
        }
    }
    sums.iter().map(|s| s / pairs as f64).collect()
}

/// Deterministic points in `[0, 1)²` (additive recurrence, equidistributed).
fn kronecker_square(n: usize) -> Vec<Vec<f64>> {
    let (a, b) = (0.754_877_666_246_693, 0.569_840_290_998_053);
    (0..n)
        .map(|i| {
            let k = i as f64 + 1.0;
            vec![(0.5 + a * k).fract(), (0.5 + b * k).fract()]
        })
        .collect()
}

// ── correlation_dimension ────────────────────────────────────────────────────

#[test]
fn test_correlation_dimension_matches_sorted_distance_reference() {
    let data = henon_points(500);
    let analysis = AttractorAnalysis::new(data.clone());
    for n_radii in [1usize, 2, 12, 20] {
        let fast = analysis.correlation_dimension(n_radii);
        let reference = correlation_dimension_reference(&data, n_radii);
        assert_eq!(fast.len(), reference.len(), "n_radii = {n_radii}");
        for (k, ((lr, lc), (rr, rc))) in fast.iter().zip(reference.iter()).enumerate() {
            assert!((lr - rr).abs() < 1e-12, "ln r[{k}]: {lr} vs {rr}");
            assert!((lc - rc).abs() < 1e-12, "ln C[{k}]: {lc} vs {rc}");
        }
    }
}

#[test]
fn test_correlation_dimension_of_henon_attractor_is_about_one_point_two() {
    let analysis = AttractorAnalysis::new(henon_points(2500));
    let pairs = analysis.correlation_dimension(40);
    // Scaling region well inside the extreme radii.
    let scaling: Vec<(f64, f64)> = pairs
        .iter()
        .copied()
        .filter(|(ln_r, _)| (-4.5..=-2.0).contains(ln_r))
        .collect();
    assert!(
        scaling.len() >= 8,
        "only {} radii in the scaling region",
        scaling.len()
    );
    let slope = linear_slope(&scaling);
    assert!(
        (slope - 1.22).abs() < 0.12,
        "D2 estimate {slope}, expected ≈ 1.22"
    );
}

#[test]
fn test_correlation_dimension_of_a_square_is_two_and_of_a_line_is_one() {
    let square = AttractorAnalysis::new(kronecker_square(2000));
    let in_range = |pairs: Vec<(f64, f64)>| -> Vec<(f64, f64)> {
        pairs
            .into_iter()
            .filter(|(ln_r, _)| (-3.5..=-2.3).contains(ln_r))
            .collect()
    };
    let scaling = in_range(square.correlation_dimension(60));
    assert!(
        scaling.len() >= 5,
        "only {} radii in the scaling region",
        scaling.len()
    );
    let slope = linear_slope(&scaling);
    assert!((slope - 2.0).abs() < 0.2, "square: D2 estimate {slope}");

    let line = AttractorAnalysis::new(
        (0..1500)
            .map(|i| vec![(0.5 + 0.618_033_988_749_895 * (i as f64 + 1.0)).fract()])
            .collect(),
    );
    let scaling = in_range(line.correlation_dimension(60));
    assert!(
        scaling.len() >= 5,
        "only {} radii in the scaling region",
        scaling.len()
    );
    let slope = linear_slope(&scaling);
    assert!((slope - 1.0).abs() < 0.1, "line: D2 estimate {slope}");
}

#[test]
fn test_correlation_dimension_ignores_coincident_points() {
    // Distances: 0 (×2), 1 (×2), 2 (×2), 3 (×4), 4 (×2), 6, 7 (×2); the
    // smallest positive one is 1, so r_min = 1.1 and r_max = 6.3.  Using the
    // zero distance as d_min gave r_min = 0, and ln 0 = -inf in the output.
    let analysis = AttractorAnalysis::new(vec![
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![3.0],
        vec![3.0],
        vec![7.0],
    ]);
    let pairs = analysis.correlation_dimension(5);
    assert_eq!(pairs.len(), 5);
    assert!(pairs.iter().all(|(r, c)| r.is_finite() && c.is_finite()));
    let (ln_r0, ln_c0) = pairs[0];
    assert!((ln_r0 - 1.1_f64.ln()).abs() < 1e-12);
    // Within r = 1.1: the two coincident pairs and the two pairs at distance 1.
    assert!((ln_c0 - (4.0_f64 / 15.0).ln()).abs() < 1e-12);
    let (ln_r4, ln_c4) = pairs[4];
    assert!((ln_r4 - 6.3_f64.ln()).abs() < 1e-12);
    // Everything but the pairs at distance 7 lies within r = 6.3.
    assert!((ln_c4 - (13.0_f64 / 15.0).ln()).abs() < 1e-12);
}

#[test]
fn test_correlation_dimension_degenerate_inputs() {
    let empty = AttractorAnalysis::new(vec![]);
    assert!(empty.correlation_dimension(10).is_empty());
    let single = AttractorAnalysis::new(vec![vec![1.0, 2.0]]);
    assert!(single.correlation_dimension(10).is_empty());
    // Two points: r_min = 1.1 d > r_max = 0.9 d.
    let pair = AttractorAnalysis::new(vec![vec![0.0], vec![1.0]]);
    assert!(pair.correlation_dimension(10).is_empty());
    // All points identical: no distance carries a scale.
    let same = AttractorAnalysis::new(vec![vec![1.0, 1.0]; 8]);
    assert!(same.correlation_dimension(10).is_empty());
    let line = AttractorAnalysis::new((0..20).map(|i| vec![i as f64]).collect());
    assert!(line.correlation_dimension(0).is_empty());
    // A single radius is r_min.
    let one = line.correlation_dimension(1);
    assert_eq!(one.len(), 1);
    assert!((one[0].0 - 1.1_f64.ln()).abs() < 1e-12);
}

#[test]
fn test_correlation_dimension_skips_pairs_with_nan_coordinates() {
    let mut data: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
    data.push(vec![f64::NAN]);
    let with_nan = AttractorAnalysis::new(data).correlation_dimension(8);
    let clean =
        AttractorAnalysis::new((0..30).map(|i| vec![i as f64]).collect()).correlation_dimension(8);
    assert_eq!(with_nan.len(), clean.len());
    for ((lr, lc), (cr, cc)) in with_nan.iter().zip(clean.iter()) {
        assert!((lr - cr).abs() < 1e-12 && (lc - cc).abs() < 1e-12);
    }
}

#[test]
fn test_correlation_dimension_theiler_window_excludes_adjacent_pairs() {
    let analysis = AttractorAnalysis::new((0..100).map(|i| vec![i as f64]).collect());
    // Window 0 and 1: every pair; d_min = 1, 99 pairs at distance 1 of 4950.
    for window in [0usize, 1] {
        let pairs = analysis.correlation_dimension_theiler(6, window);
        assert!((pairs[0].0 - 1.1_f64.ln()).abs() < 1e-12);
        assert!((pairs[0].1 - (99.0_f64 / 4950.0).ln()).abs() < 1e-12);
    }
    // Window 5: only |i − j| >= 5 remain: 95 pairs at distance 5 of
    // Σ_{d=5}^{99} (100 − d) = 4560.
    let pairs = analysis.correlation_dimension_theiler(6, 5);
    assert!((pairs[0].0 - 5.5_f64.ln()).abs() < 1e-12);
    assert!((pairs[0].1 - (95.0_f64 / 4560.0).ln()).abs() < 1e-12);
    // A window as long as the series leaves no pair.
    assert!(analysis.correlation_dimension_theiler(6, 100).is_empty());
    assert!(
        analysis
            .correlation_dimension_theiler(6, usize::MAX)
            .is_empty()
    );
}

// ── rosenstein_lyapunov ──────────────────────────────────────────────────────

#[test]
fn test_rosenstein_matches_brute_force_reference_for_every_window() {
    // Deterministic pseudo-random 2-D data (LCG).
    let mut state = 12_345_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let data: Vec<Vec<f64>> = (0..80).map(|_| vec![next(), next()]).collect();
    let analysis = AttractorAnalysis::new(data.clone());
    for (horizon, window) in [
        (6usize, 0usize),
        (6, 1),
        (6, 7),
        (15, 3),
        (200, 10),
        (1, 25),
    ] {
        let got = analysis.rosenstein_lyapunov(horizon, window);
        let want = rosenstein_reference(&data, horizon, window);
        assert_eq!(got.len(), want.len(), "horizon {horizon}, window {window}");
        for (t, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-12,
                "horizon {horizon}, window {window}, t = {t}: {g} vs {w}"
            );
        }
    }
}

#[test]
fn test_rosenstein_theiler_window_changes_the_neighbour_choice() {
    // One revolution of a circle sampled at 200 points: without a window the
    // nearest neighbour of a point is its orbit neighbour, which stays one
    // step away forever (a flat curve at ln of the step length).
    let step_length = 2.0 * (PI / 200.0).sin();
    let data: Vec<Vec<f64>> = (0..200)
        .map(|i| {
            let a = TAU * i as f64 / 200.0;
            vec![a.cos(), a.sin()]
        })
        .collect();
    let analysis = AttractorAnalysis::new(data);
    let unprotected = analysis.rosenstein_lyapunov(20, 1);
    assert_eq!(unprotected.len(), 20);
    for (t, y) in unprotected.iter().enumerate() {
        assert!(
            (y - step_length.ln()).abs() < 1e-9,
            "t = {t}: {y} should be ln(step) = {}",
            step_length.ln()
        );
    }
    // With a window of 10 samples the nearest neighbour is 10 steps away.
    let protected = analysis.rosenstein_lyapunov(20, 10);
    let ten_steps = 2.0 * (10.0 * PI / 200.0).sin();
    for (t, y) in protected.iter().enumerate() {
        assert!(
            (y - ten_steps.ln()).abs() < 1e-9,
            "t = {t}: {y} should be ln(10 steps) = {}",
            ten_steps.ln()
        );
    }
}

#[test]
fn test_rosenstein_degenerate_inputs() {
    let data: Vec<Vec<f64>> = (0..30).map(|i| vec![(i as f64 * 0.7).sin()]).collect();
    let analysis = AttractorAnalysis::new(data);
    assert!(analysis.rosenstein_lyapunov(0, 1).is_empty());
    assert!(
        AttractorAnalysis::new(vec![])
            .rosenstein_lyapunov(5, 1)
            .is_empty()
    );
    assert!(
        AttractorAnalysis::new(vec![vec![1.0]])
            .rosenstein_lyapunov(5, 1)
            .is_empty()
    );
    // A window that excludes every candidate gives no curve, not a curve of zeros.
    assert!(analysis.rosenstein_lyapunov(10, 30).is_empty());
    assert!(analysis.rosenstein_lyapunov(10, usize::MAX).is_empty());
    // Identical points: neighbours exist but all distances vanish, so no entry
    // can be evaluated and none is reported as 0.
    let constant = AttractorAnalysis::new(vec![vec![2.0, 2.0]; 12]);
    let curve = constant.rosenstein_lyapunov(4, 1);
    assert_eq!(curve.len(), 4);
    assert!(curve.iter().all(|y| y.is_nan()));
}

#[test]
fn test_default_theiler_window_is_first_autocorrelation_zero_crossing() {
    // Period 40: the autocorrelation first reaches zero at a quarter period.
    let sine: Vec<Vec<f64>> = (0..400)
        .map(|i| vec![(TAU * i as f64 / 40.0).sin(), 0.0])
        .collect();
    let window = AttractorAnalysis::new(sine).default_theiler_window();
    assert!((10..=11).contains(&window), "window = {window}");
    // Only coordinate 0 matters.
    let swapped: Vec<Vec<f64>> = (0..400)
        .map(|i| vec![0.0, (TAU * i as f64 / 40.0).sin()])
        .collect();
    assert_eq!(
        AttractorAnalysis::new(swapped).default_theiler_window(),
        200
    );
    // White-noise-like data decorrelates immediately.
    let mut state = 99_u64;
    let noise: Vec<Vec<f64>> = (0..500)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            vec![(state >> 11) as f64 / (1u64 << 53) as f64]
        })
        .collect();
    assert!(AttractorAnalysis::new(noise).default_theiler_window() <= 20);
    // No zero crossing below n / 2: a constant signal has no decorrelation
    // time, and a three-point series has no lag below n / 2 = 1 to test.
    let constant = AttractorAnalysis::new(vec![vec![1.0]; 40]);
    assert_eq!(constant.default_theiler_window(), 20);
    let three = AttractorAnalysis::new(vec![vec![1.0], vec![2.0], vec![4.0]]);
    assert_eq!(three.default_theiler_window(), 1);
    assert_eq!(AttractorAnalysis::new(vec![]).default_theiler_window(), 0);
}

#[test]
fn test_rosenstein_auto_uses_the_default_window() {
    let data: Vec<Vec<f64>> = (0..300)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t.sin(), (1.3 * t).cos()]
        })
        .collect();
    let analysis = AttractorAnalysis::new(data);
    let window = analysis.default_theiler_window();
    assert!(window > 1);
    assert_eq!(
        analysis.rosenstein_lyapunov_auto(12),
        analysis.rosenstein_lyapunov(12, window)
    );
}

#[test]
fn test_rosenstein_recovers_the_lorenz_exponent() {
    // σ = 10, ρ = 28, β = 8/3 sampled every dt = 0.01: λ₁ ≈ 0.906.  The curve
    // is not a straight line over its whole length (initial alignment of the
    // neighbour pair, saturation at the attractor size), so the slope is
    // fitted over 0.5 ≤ t ≤ 2.0.
    let dt = 0.01;
    let analysis = AttractorAnalysis::new(lorenz_orbit(5000, dt));
    let curve = analysis.rosenstein_lyapunov(250, 100);
    assert_eq!(curve.len(), 250);
    assert!(curve.iter().all(|y| y.is_finite()));
    let exponent = slope_over(&curve, 50, 200) / dt;
    assert!(
        (exponent - 0.9).abs() < 0.15,
        "Lorenz λ₁ estimate {exponent}, expected ≈ 0.9"
    );
    // The curve rises monotonically over the fitted range.
    assert!(
        curve[200] > curve[50] + 1.0,
        "{} -> {}",
        curve[50],
        curve[200]
    );
}

#[test]
fn test_rosenstein_theiler_window_matters_for_a_finely_sampled_flow() {
    // dt = 0.002: consecutive samples are only |v|·dt ≈ 0.06 apart, closer than
    // the neighbouring sheets of the attractor, so without a window every
    // nearest neighbour is the point's own orbit neighbour.  That pair moves
    // along the flow and its distance never grows.
    let dt = 0.002;
    let analysis = AttractorAnalysis::new(lorenz_orbit(5000, dt));
    let naive = slope_over(&analysis.rosenstein_lyapunov(150, 1), 20, 100) / dt;
    let windowed = slope_over(&analysis.rosenstein_lyapunov(150, 250), 20, 100) / dt;
    assert!(naive.abs() < 0.3, "no window: slope {naive} should be ≈ 0");
    assert!(windowed > 0.45, "window of 250 samples: slope {windowed}");
    assert!(windowed > 3.0 * naive.abs());
}

#[test]
fn test_rosenstein_recovers_the_logistic_map_exponent() {
    // Fully chaotic logistic map: λ = ln 2 per iteration, delay embedding m = 2.
    let map = LogisticMap::new(4.0);
    let orbit = map.orbit_1d(0.3, 3200);
    let data: Vec<Vec<f64>> = orbit[200..3200]
        .windows(2)
        .map(|w| vec![w[0], w[1]])
        .collect();
    let analysis = AttractorAnalysis::new(data);
    let curve = analysis.rosenstein_lyapunov_auto(8);
    assert_eq!(curve.len(), 8);
    let exponent = slope_over(&curve, 0, 4);
    assert!(
        (exponent - 2.0_f64.ln()).abs() < 0.1,
        "logistic λ estimate {exponent}, expected ln 2 = {}",
        2.0_f64.ln()
    );
}

// ── instantaneous_phase ──────────────────────────────────────────────────────

#[test]
fn test_instantaneous_phase_of_a_cosine_is_an_exact_ramp() {
    // 5 Hz cosine with phase 0.7, 1024 samples over 1 s (bin-centred).
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|k| (TAU * 5.0 * k as f64 / n as f64 + 0.7).cos())
        .collect();
    let phase = SynchronizationAnalysis::instantaneous_phase(&signal);
    assert_eq!(phase.len(), n);
    let slope = TAU * 5.0 / n as f64;
    for k in 100..924 {
        let step = phase[k + 1] - phase[k];
        assert!(step > 0.0, "phase must be monotone, step {k} = {step}");
        assert!(
            (step - slope).abs() < 0.01 * slope,
            "slope at {k}: {step}, expected {slope}"
        );
    }
    // The unwrapped phase is the true phase 0.7 + 2π·5·t, not its principal value.
    for (k, p) in phase.iter().enumerate() {
        let expected = 0.7 + slope * k as f64;
        assert!(
            (p - expected).abs() < 1e-8,
            "phase[{k}] = {p}, expected {expected}"
        );
    }
    assert!(
        phase[n - 1] > 30.0,
        "phase must be unwrapped: {}",
        phase[n - 1]
    );
}

#[test]
fn test_instantaneous_phase_of_a_sine_with_a_non_power_of_two_length() {
    // 1000 samples at fs = 1000 Hz of a 5 Hz sine; the FFT length is padded to 1024.
    let fs = 1000.0;
    let signal: Vec<f64> = (0..1000)
        .map(|k| (TAU * 5.0 * k as f64 / fs).sin())
        .collect();
    let phase = SynchronizationAnalysis::instantaneous_phase(&signal);
    assert_eq!(phase.len(), 1000);
    let slope = TAU * 5.0 / fs;
    for k in 100..900 {
        let step = phase[k + 1] - phase[k];
        assert!(step > 0.0, "phase must be monotone, step {k} = {step}");
    }
    let mean_slope = (phase[900] - phase[100]) / 800.0;
    assert!(
        (mean_slope - slope).abs() < 0.01 * slope,
        "slope over samples 100..900: {mean_slope}, expected {slope}"
    );
    // sin(θ) = cos(θ − π/2): the phase is 2π·5·t − π/2 up to edge effects.
    let mid_offset = phase[500] - slope * 500.0;
    assert!((mid_offset + PI / 2.0).abs() < 0.05, "offset {mid_offset}");
}

#[test]
fn test_instantaneous_phase_ignores_the_mean() {
    let n = 512;
    let base: Vec<f64> = (0..n)
        .map(|k| (TAU * 8.0 * k as f64 / n as f64 + 1.0).cos())
        .collect();
    let shifted: Vec<f64> = base.iter().map(|v| v + 3.0).collect();
    let a = SynchronizationAnalysis::instantaneous_phase(&base);
    let b = SynchronizationAnalysis::instantaneous_phase(&shifted);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-9, "{x} vs {y}");
    }
}

#[test]
fn test_instantaneous_phase_short_and_constant_inputs() {
    assert!(SynchronizationAnalysis::instantaneous_phase(&[]).is_empty());
    assert_eq!(
        SynchronizationAnalysis::instantaneous_phase(&[4.2]),
        vec![0.0]
    );
    let two = SynchronizationAnalysis::instantaneous_phase(&[1.0, -1.0]);
    assert_eq!(two.len(), 2);
    assert!(two.iter().all(|p| p.is_finite()));
    let constant = SynchronizationAnalysis::instantaneous_phase(&[2.5; 64]);
    assert!(
        constant.iter().all(|&p| p == 0.0),
        "no oscillation, no phase"
    );
}

#[test]
fn test_instantaneous_phase_feeds_phase_synchronisation_index() {
    let n = 1024;
    let first: Vec<f64> = (0..n)
        .map(|k| (TAU * 6.0 * k as f64 / n as f64).sin())
        .collect();
    let locked: Vec<f64> = (0..n)
        .map(|k| 0.4 * (TAU * 6.0 * k as f64 / n as f64 + 1.1).sin())
        .collect();
    let drifting: Vec<f64> = (0..n)
        .map(|k| (TAU * 9.3 * k as f64 / n as f64).sin())
        .collect();
    let phi_a = SynchronizationAnalysis::instantaneous_phase(&first);
    let phi_locked = SynchronizationAnalysis::instantaneous_phase(&locked);
    let phi_drifting = SynchronizationAnalysis::instantaneous_phase(&drifting);
    let r_locked = SynchronizationAnalysis::phase_synchronization_index(&phi_a, &phi_locked);
    assert!(
        r_locked > 0.99,
        "same frequency, constant lag: R = {r_locked}"
    );
    let r_drifting = SynchronizationAnalysis::phase_synchronization_index(&phi_a, &phi_drifting);
    assert!(r_drifting < 0.2, "different frequencies: R = {r_drifting}");
}
