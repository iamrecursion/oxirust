// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for recurrence quantification analysis: the minimum line
//! length of DET and the exclusion of the main diagonal (line of identity).

use super::*;
use std::f64::consts::TAU;

/// Recurrence plot with exactly the given recurrent cells `(row, column)`.
fn plot_from_cells(size: usize, cells: &[(usize, usize)]) -> RecurrencePlot {
    let mut matrix = vec![vec![false; size]; size];
    for &(i, j) in cells {
        matrix[i][j] = true;
    }
    RecurrencePlot {
        size,
        threshold: 0.0,
        matrix,
    }
}

/// Main diagonal of an `n × n` plot.
fn identity_cells(n: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| (i, i)).collect()
}

/// An 8 × 8 plot with the identity line, a symmetric pair of 3-point lines
/// (`k = ±2`), a symmetric pair of 2-point lines (`k = ±4`) and two symmetric
/// isolated points (`k = ±6`): 12 recurrent points off the main diagonal.
fn structured_plot() -> RecurrencePlot {
    let mut cells = identity_cells(8);
    for &(i, j) in &[(0, 2), (1, 3), (2, 4), (0, 4), (1, 5), (0, 6)] {
        cells.push((i, j));
        cells.push((j, i));
    }
    plot_from_cells(8, &cells)
}

/// Deterministic uniform noise in `[0, 1)`.
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

#[test]
fn test_determinism_honours_min_line() {
    let rp = structured_plot();
    // Line lengths off the main diagonal: 3, 3, 2, 2, 1, 1 → 12 points.
    assert!((determinism_with_min_line(&rp, 1) - 12.0 / 12.0).abs() < 1e-15);
    assert!((determinism_with_min_line(&rp, 2) - 10.0 / 12.0).abs() < 1e-15);
    assert!((determinism_with_min_line(&rp, 3) - 6.0 / 12.0).abs() < 1e-15);
    assert_eq!(determinism_with_min_line(&rp, 4), 0.0);
    // The parameter-free `determinism` keeps its documented min_line of 2.
    assert!((determinism(&rp) - 10.0 / 12.0).abs() < 1e-15);
    // RecurrenceAnalysis::compute reports the DET of the min_line it is given
    // (it used to hard-code 2 whatever min_line was).
    for (min_line, expected) in [(1usize, 1.0), (2, 10.0 / 12.0), (3, 0.5), (4, 0.0)] {
        let rqa = RecurrenceAnalysis::compute(&rp, min_line);
        assert!(
            (rqa.determinism - expected).abs() < 1e-15,
            "min_line {min_line}: DET = {}, expected {expected}",
            rqa.determinism
        );
        assert_eq!(rqa.determinism, determinism_with_min_line(&rp, min_line));
    }
}

#[test]
fn test_diagonal_statistics_exclude_the_line_of_identity() {
    let rp = structured_plot();
    let rqa = RecurrenceAnalysis::compute(&rp, 2);
    // Lines ≥ 2 off the main diagonal: 3, 3, 2, 2 (the 8-point identity line
    // used to make L_max = n for every signal).
    assert_eq!(rqa.max_diagonal_length, 3);
    assert!((rqa.mean_diagonal_length - 2.5).abs() < 1e-15);
    let p_three = 0.5_f64;
    let expected_entropy = -2.0 * p_three * p_three.ln();
    assert!(
        (rqa.diagonal_entropy - expected_entropy).abs() < 1e-12,
        "entropy {} vs {expected_entropy}",
        rqa.diagonal_entropy
    );
    // RR still counts the whole plot: 8 identity + 12 other points of 64.
    assert!((rqa.recurrence_rate - 20.0 / 64.0).abs() < 1e-15);
}

#[test]
fn test_laminarity_and_trapping_time_keep_the_whole_plot_as_reference() {
    // Column 0 holds the identity point and two off-diagonal points, a vertical
    // line of 3; LAM = (points on vertical lines) / (all recurrent points).
    let mut cells = identity_cells(4);
    cells.push((1, 0));
    cells.push((2, 0));
    let rp = plot_from_cells(4, &cells);
    let rqa = RecurrenceAnalysis::compute(&rp, 2);
    assert!(
        (rqa.laminarity - 3.0 / 6.0).abs() < 1e-15,
        "LAM = {}",
        rqa.laminarity
    );
    assert!(
        (rqa.trapping_time - 3.0).abs() < 1e-15,
        "TT = {}",
        rqa.trapping_time
    );
    // The two off-diagonal points are not on diagonal lines of length 2.
    assert_eq!(rqa.determinism, 0.0);
    assert_eq!(rqa.max_diagonal_length, 0);
}

#[test]
fn test_constant_signal_rqa_values() {
    // Every cell recurrent: off-diagonal lines of 7, 6, …, 1 points, twice.
    let rp = RecurrencePlot::new(&[1.0; 8], 0.01);
    let rqa = RecurrenceAnalysis::compute(&rp, 2);
    assert_eq!(rqa.max_diagonal_length, 7);
    assert!((rqa.recurrence_rate - 1.0).abs() < 1e-15);
    // Lines ≥ 2: lengths 2..=7 twice = 54 points of the 56 off-diagonal ones.
    assert!((rqa.determinism - 54.0 / 56.0).abs() < 1e-15);
    assert!((rqa.mean_diagonal_length - 4.5).abs() < 1e-15);
    // Eight full columns of 8.
    assert!((rqa.laminarity - 1.0).abs() < 1e-15);
    assert!((rqa.trapping_time - 8.0).abs() < 1e-15);
}

#[test]
fn test_sine_wave_longest_diagonal_is_shorter_than_the_series() {
    let n = 200;
    let signal: Vec<f64> = (0..n).map(|i| (TAU * i as f64 / 25.0).sin()).collect();
    let rp = RecurrencePlot::new(&signal, 0.1);
    let rqa = RecurrenceAnalysis::compute(&rp, 2);
    assert!(
        rqa.max_diagonal_length < n,
        "L_max = {}",
        rqa.max_diagonal_length
    );
    // A periodic signal recurs along the diagonals k = ±25, ±50, …: the
    // longest line has n − 25 points (a sine sampled at an integer number of
    // points per period returns exactly).
    assert_eq!(rqa.max_diagonal_length, n - 25);
    // The time-reversed branch of each period (θ' = π − θ) falls half a sample
    // off the grid, so some recurrent points on the mirror bands are isolated:
    // DET = 2680/3960 = 67/99.  (Reference values from an independent numpy
    // implementation of the Marwan et al. definitions.)
    assert!(
        (rqa.determinism - 0.676_767_676_767_676_8).abs() < 1e-12,
        "DET = {}",
        rqa.determinism
    );
    assert!(
        (rqa.recurrence_rate - 0.104).abs() < 1e-12,
        "RR = {}",
        rqa.recurrence_rate
    );
}

#[test]
fn test_white_noise_determinism_drops_with_min_line() {
    let signal = noise(300, 7);
    let rp = RecurrencePlot::new(&signal, 0.15);
    let det_2 = RecurrenceAnalysis::compute(&rp, 2).determinism;
    let det_5 = RecurrenceAnalysis::compute(&rp, 5).determinism;
    assert!(
        det_5 < det_2,
        "DET(5) = {det_5} must be below DET(2) = {det_2}"
    );
    assert!(
        det_5 < 0.5 * det_2,
        "DET(5) = {det_5} should be far below DET(2) = {det_2}"
    );
    // Noise is not deterministic: long lines are rare.
    assert!(det_5 < 0.1, "DET(5) = {det_5}");
    let rqa = RecurrenceAnalysis::compute(&rp, 2);
    assert!(
        rqa.max_diagonal_length < 30,
        "L_max = {}",
        rqa.max_diagonal_length
    );
}

#[test]
fn test_min_line_zero_counts_as_one() {
    let rp = structured_plot();
    let zero = RecurrenceAnalysis::compute(&rp, 0);
    let one = RecurrenceAnalysis::compute(&rp, 1);
    assert_eq!(zero.determinism, one.determinism);
    assert_eq!(zero.mean_diagonal_length, one.mean_diagonal_length);
    assert_eq!(zero.max_diagonal_length, one.max_diagonal_length);
    assert_eq!(zero.diagonal_entropy, one.diagonal_entropy);
    assert_eq!(zero.laminarity, one.laminarity);
    assert_eq!(zero.trapping_time, one.trapping_time);
    // No zero-length "lines" dilute the mean: (3 + 3 + 2 + 2 + 1 + 1) / 6 = 2.
    assert!((zero.mean_diagonal_length - 2.0).abs() < 1e-15);
    assert_eq!(
        determinism_with_min_line(&rp, 0),
        determinism_with_min_line(&rp, 1)
    );
}

#[test]
fn test_determinism_without_off_diagonal_recurrence() {
    // Only the identity line is recurrent: nothing is deterministic.
    let identity_only = plot_from_cells(5, &identity_cells(5));
    assert_eq!(determinism(&identity_only), 0.0);
    let rqa = RecurrenceAnalysis::compute(&identity_only, 2);
    assert_eq!(rqa.determinism, 0.0);
    assert_eq!(rqa.max_diagonal_length, 0);
    assert_eq!(rqa.mean_diagonal_length, 0.0);
    assert_eq!(rqa.diagonal_entropy, 0.0);
    // A plot whose main diagonal is not recurrent (as for a negative threshold)
    // still counts its off-diagonal points: a full 3-point line at k = 1.
    let no_identity = plot_from_cells(5, &[(0, 1), (1, 2), (2, 3), (4, 0)]);
    assert!((determinism(&no_identity) - 3.0 / 4.0).abs() < 1e-15);
    // Nothing recurrent at all, and the empty plot.
    let empty_plot = plot_from_cells(5, &[]);
    assert_eq!(determinism(&empty_plot), 0.0);
    let none = RecurrencePlot {
        size: 0,
        threshold: 0.1,
        matrix: Vec::new(),
    };
    let rqa = RecurrenceAnalysis::compute(&none, 2);
    assert_eq!(rqa.recurrence_rate, 0.0);
    assert_eq!(rqa.determinism, 0.0);
    assert_eq!(rqa.laminarity, 0.0);
    assert_eq!(rqa.max_diagonal_length, 0);
}
