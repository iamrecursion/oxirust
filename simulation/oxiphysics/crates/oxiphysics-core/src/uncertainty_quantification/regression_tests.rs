// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the internal random number generator and the
//! resampling, shuffling and permutation code that is built on it.
//!
//! The raw state of a 2^64-modulus LCG has low bits of period `2^k`; taking
//! `% n` of it made bootstrap resamples of a power-of-two sample exact
//! permutations (a zero-width confidence interval) and restricted
//! Fisher-Yates shuffles to a fraction of all permutations.

use super::*;
use std::cell::RefCell;

/// Pearson chi-square statistic of `counts` against equal expected counts.
fn chi_square_uniform(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    let expected = total as f64 / counts.len() as f64;
    counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum()
}

/// Upper 0.1 % point of the chi-square distribution with `dof` degrees of
/// freedom (Wilson-Hilferty approximation, within a few percent of the exact
/// table value for `dof >= 1`).
fn chi_square_critical(dof: usize) -> f64 {
    let k = dof as f64;
    let z_999 = 3.090_232_306_167_813;
    let t = 2.0 / (9.0 * k);
    k * (1.0 - t + z_999 * t.sqrt()).powi(3)
}

/// `n!`
fn factorial_usize(n: usize) -> usize {
    (1..=n).product()
}

/// Rank of the permutation `perm` of `0..perm.len()` in lexicographic order
/// (Lehmer code in the factorial number system).
fn permutation_rank(perm: &[usize]) -> usize {
    let n = perm.len();
    let mut rank = 0;
    for i in 0..n {
        let smaller_after = perm[i + 1..].iter().filter(|&&v| v < perm[i]).count();
        rank = rank * (n - i) + smaller_after;
    }
    rank
}

#[test]
fn generator_low_bits_do_not_alternate_or_repeat() {
    let mut rng = UqRng::new(7);
    let draws: Vec<u64> = (0..10_001).map(|_| rng.next_u64()).collect();
    // The raw LCG's lowest bit alternates strictly (fraction 1.0); random bits
    // flip half of the time.
    let flips = draws
        .windows(2)
        .filter(|w| (w[0] & 1) != (w[1] & 1))
        .count();
    let fraction = flips as f64 / 10_000.0;
    assert!((fraction - 0.5).abs() < 0.03, "flip fraction {fraction}");
    // The raw LCG's low byte repeats with period 256.
    let low_bytes: Vec<u64> = draws.iter().map(|d| d & 0xff).collect();
    assert_ne!(low_bytes[0..256], low_bytes[256..512]);
    assert_ne!(low_bytes[256..512], low_bytes[512..768]);
}

#[test]
fn generator_is_reproducible_and_seed_dependent() {
    let mut a = UqRng::new(42);
    let mut b = UqRng::new(42);
    let mut c = UqRng::new(43);
    let first_a: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
    let first_b: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
    let first_c: Vec<u64> = (0..16).map(|_| c.next_u64()).collect();
    assert_eq!(first_a, first_b);
    assert_ne!(first_a, first_c);
}

#[test]
fn next_f64_is_in_the_unit_interval_with_mean_one_half() {
    let mut rng = UqRng::new(5);
    let n = 100_000;
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
    let m = mean(&values);
    // Standard error of the mean of n uniforms is 1/sqrt(12 n) = 9.1e-4.
    assert!((m - 0.5).abs() < 5.0e-3, "mean {m}");
}

#[test]
fn next_below_is_uniform_for_small_ranges() {
    for &n in &[2usize, 3, 5, 10, 64] {
        let mut rng = UqRng::new(2024);
        let mut counts = vec![0usize; n];
        for _ in 0..50_000 {
            let v = rng.next_below(n);
            assert!(v < n);
            counts[v] += 1;
        }
        let stat = chi_square_uniform(&counts);
        let limit = chi_square_critical(n - 1);
        assert!(stat < limit, "n = {n}: chi2 {stat} >= {limit}");
    }
}

#[test]
fn next_below_degenerate_ranges() {
    let mut rng = UqRng::new(1);
    assert_eq!(rng.next_below(0), 0);
    for _ in 0..100 {
        assert_eq!(rng.next_below(1), 0);
    }
}

#[test]
fn next_below_is_unbiased_when_the_rejection_step_is_active() {
    // For a range of about three quarters of 2^bits a large share of the raw
    // words fall in the biased zone and are rejected and redrawn.
    let n = usize::MAX / 4 * 3;
    let mut rng = UqRng::new(11);
    let mut quartiles = [0usize; 4];
    for _ in 0..40_000 {
        let v = rng.next_below(n);
        assert!(v < n);
        quartiles[(v / (n / 4)).min(3)] += 1;
    }
    let stat = chi_square_uniform(&quartiles);
    assert!(
        stat < chi_square_critical(3),
        "chi2 {stat} over {quartiles:?}"
    );
}

#[test]
fn shuffle_is_uniform_over_all_six_permutations_of_three_elements() {
    // With the raw LCG only three of the six permutations ever occurred
    // (chi-square statistic of about 1e5).
    let mut rng = UqRng::new(42);
    let trials = 100_000;
    let mut counts = [0usize; 6];
    for _ in 0..trials {
        let mut items = [0usize, 1, 2];
        rng.shuffle(&mut items);
        counts[permutation_rank(&items)] += 1;
    }
    let stat = chi_square_uniform(&counts);
    // 5 degrees of freedom: the 0.1 % critical value is 20.515.
    assert!(stat < 20.515, "chi2 {stat} over {counts:?}");
}

#[test]
fn shuffle_is_uniform_over_all_24_permutations_of_four_elements() {
    let mut rng = UqRng::new(8);
    let trials = 240_000;
    let mut counts = vec![0usize; factorial_usize(4)];
    for _ in 0..trials {
        let mut items = [0usize, 1, 2, 3];
        rng.shuffle(&mut items);
        counts[permutation_rank(&items)] += 1;
    }
    let stat = chi_square_uniform(&counts);
    let limit = chi_square_critical(23);
    assert!(stat < limit, "chi2 {stat} >= {limit} over {counts:?}");
}

#[test]
fn shuffle_keeps_the_elements_and_accepts_tiny_slices() {
    let mut rng = UqRng::new(3);
    let mut empty: [u32; 0] = [];
    rng.shuffle(&mut empty);
    let mut single = [9u32];
    rng.shuffle(&mut single);
    assert_eq!(single, [9]);
    let mut items: Vec<usize> = (0..100).collect();
    rng.shuffle(&mut items);
    assert_ne!(items, (0..100).collect::<Vec<_>>());
    items.sort_unstable();
    assert_eq!(items, (0..100).collect::<Vec<_>>());
}

#[test]
fn latin_hypercube_stratum_order_is_uniform_across_adjacent_seeds() {
    // Three samples in one dimension: the order in which the strata are
    // visited is the shuffled permutation of 0..3.
    let mut counts = [0usize; 6];
    for seed in 1..=60_000u64 {
        let lhs = latin_hypercube_sample(3, 1, seed);
        let strata: Vec<usize> = (0..3).map(|i| (lhs.get(i, 0) * 3.0) as usize).collect();
        counts[permutation_rank(&strata)] += 1;
    }
    let stat = chi_square_uniform(&counts);
    assert!(stat < 20.515, "chi2 {stat} over {counts:?}");
}

#[test]
fn latin_hypercube_keeps_one_point_per_stratum_in_every_dimension() {
    let n = 64;
    let lhs = latin_hypercube_sample(n, 3, 17);
    for j in 0..3 {
        let mut hit = vec![0usize; n];
        for i in 0..n {
            let v = lhs.get(i, j);
            assert!((0.0..1.0).contains(&v));
            hit[(v * n as f64) as usize] += 1;
        }
        assert!(hit.iter().all(|&c| c == 1), "dimension {j}: {hit:?}");
    }
}

#[test]
fn morris_trajectories_perturb_the_dimensions_in_uniformly_random_order() {
    let n_dims = 4;
    let r = 12_000;
    let calls: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::new());
    let model = |x: &[f64]| {
        calls.borrow_mut().push(x.to_vec());
        x.iter().sum::<f64>()
    };
    let bounds = vec![(0.0, 1.0); n_dims];
    let _ = morris_sensitivity(n_dims, r, 0.25, &model, &bounds, 99);
    let calls = calls.borrow();
    assert_eq!(calls.len(), r * (n_dims + 1));
    let mut counts = vec![0usize; factorial_usize(n_dims)];
    for trajectory in calls.chunks(n_dims + 1) {
        // Each step changes exactly one coordinate: the dimension it belongs to.
        let order: Vec<usize> = trajectory
            .windows(2)
            .map(|w| {
                let changed: Vec<usize> = (0..n_dims).filter(|&d| w[0][d] != w[1][d]).collect();
                assert_eq!(changed.len(), 1, "one coordinate per step");
                changed[0]
            })
            .collect();
        counts[permutation_rank(&order)] += 1;
    }
    let stat = chi_square_uniform(&counts);
    let limit = chi_square_critical(counts.len() - 1);
    assert!(stat < limit, "chi2 {stat} >= {limit} over {counts:?}");
}

#[test]
fn bootstrap_ci_of_a_power_of_two_sample_has_the_normal_width() {
    // Data 0..64: standard error of the mean sqrt((64^2 - 1) / 12) / 8 = 2.31,
    // so a 95 % interval is about 9.05 wide.  The raw LCG gave exact
    // permutations of the data in every resample: width 0.
    let data: Vec<f64> = (0..64).map(f64::from).collect();
    let (lo, hi) = bootstrap_ci(&data, &mean, 2_000, 0.05, 1);
    assert!(hi - lo > 1.0);
    assert!(
        (7.5..10.5).contains(&(hi - lo)),
        "95% interval [{lo}, {hi}] has width {}",
        hi - lo
    );
    assert!(lo < 31.5 && 31.5 < hi);
}

#[test]
fn bootstrap_ci_width_tracks_the_sample_size() {
    for &n in &[32usize, 100, 128] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sd = std_dev(&data);
        let expected = 2.0 * 1.959_964 * sd / (n as f64).sqrt();
        let (lo, hi) = bootstrap_ci(&data, &mean, 2_000, 0.05, 3);
        let ratio = (hi - lo) / expected;
        assert!(
            (0.85..1.15).contains(&ratio),
            "n = {n}: width ratio {ratio}"
        );
    }
}

#[test]
fn bootstrap_ci_is_reproducible_and_seed_dependent() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
    let first = bootstrap_ci(&data, &mean, 500, 0.1, 21);
    let second = bootstrap_ci(&data, &mean, 500, 0.1, 21);
    let other = bootstrap_ci(&data, &mean, 500, 0.1, 22);
    assert_eq!(first, second);
    assert_ne!(first, other);
}

#[test]
fn bootstrap_ci_accepts_empty_data_and_zero_resamples() {
    // Resamples of an empty sample are empty; the statistic sees `&[]`.
    assert_eq!(bootstrap_ci(&[], &mean, 10, 0.05, 1), (0.0, 0.0));
    assert_eq!(bootstrap_ci(&[1.0, 2.0], &mean, 0, 0.05, 1), (0.0, 0.0));
}
