// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the lossless run-length encoding, the bitwise snapshot
//! diffs and the stratified Latin-hypercube design.

use super::*;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn table_with(field: &str, first: Vec<f64>, second: Vec<f64>) -> SnapshotTable {
    let mut table = SnapshotTable::new();
    let mut s0 = Snapshot::new(0.0);
    s0.set_field(field, first);
    let mut s1 = Snapshot::new(1.0);
    s1.set_field(field, second);
    table.insert(s0);
    table.insert(s1);
    table
}

/// Deterministic 64-bit generator (xorshift) so no dev-dependency is needed.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

// --- run-length encoding ---------------------------------------------------

#[test]
fn rle_keeps_values_below_epsilon_apart() {
    let data = [1e-20, 3e-20, 3e-20];
    let encoded = rle_encode(&data);
    assert_eq!(encoded, vec![(1e-20, 1), (3e-20, 2)]);
    assert_eq!(bits(&rle_decode(&encoded)), bits(&data));
}

#[test]
fn rle_round_trip_is_bit_exact_for_special_values() {
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0000);
    let nan_b = f64::from_bits(0x7ff8_0000_0000_0001);
    let data = [
        0.0,
        -0.0,
        -0.0,
        nan_a,
        nan_a,
        nan_b,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        5e-324,
        5e-324,
        1.0,
        1.0 + f64::EPSILON,
    ];
    let encoded = rle_encode(&data);
    // 0.0 | -0.0 x2 | nan_a x2 | nan_b | inf | -inf | MIN_POSITIVE | 5e-324 x2 | 1.0 | 1+eps
    let counts: Vec<usize> = encoded.iter().map(|&(_, count)| count).collect();
    assert_eq!(counts, vec![1, 2, 2, 1, 1, 1, 1, 2, 1, 1]);
    assert_eq!(bits(&rle_decode(&encoded)), bits(&data));
}

#[test]
fn rle_round_trip_of_pseudo_random_tiny_values_is_exact() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let data: Vec<f64> = (0..2000)
        .map(|_| {
            // Few distinct values so that real runs occur, all far below EPSILON.
            let bucket = rng.next_u64() % 5;
            (bucket as f64 + 1.0) * 1e-21
        })
        .collect();
    let encoded = rle_encode(&data);
    assert!(encoded.len() < data.len());
    assert_eq!(bits(&rle_decode(&encoded)), bits(&data));
}

#[test]
fn vtk_compression_rle_is_lossless_below_its_old_threshold() {
    let data = [1e-20, 3e-20, 3e-20, 0.0, 5e-16];
    let runs = crate::VtkCompression::rle_encode_f64(&data);
    assert_eq!(runs.len(), 4);
    assert_eq!(runs[1], (3e-20, 2));
    let decoded = crate::VtkCompression::rle_decode_f64(&runs);
    assert_eq!(bits(&decoded), bits(&data));
}

#[test]
fn compressed_field_of_a_snapshot_round_trips_bit_exactly() {
    let field = vec![1e-20, 3e-20, 3e-20, -0.0, 0.0];
    let mut table = SnapshotTable::new();
    let mut snap = Snapshot::new(0.0);
    snap.set_field("tiny", field.clone());
    table.insert(snap);
    table.compress_field(0, "tiny");
    let decoded = table
        .decompress_field(0, "tiny")
        .expect("field was compressed");
    assert_eq!(bits(&decoded), bits(&field));
}

// --- snapshot diffs --------------------------------------------------------

#[test]
fn compute_diff_detects_a_change_below_epsilon() {
    let mut table = table_with("u", vec![1e-20, 2.0, 1e-20], vec![3e-20, 2.0, 1e-20]);
    assert_eq!(table.compute_diff("u", 1), 1);
    assert_eq!(table.diffs[0].indices, vec![0]);
    assert_eq!(table.diffs[0].new_values[0].to_bits(), 3e-20_f64.to_bits());
}

#[test]
fn compute_diff_detects_changes_to_and_from_nan_but_not_nan_to_same_nan() {
    let mut table = table_with(
        "u",
        vec![1.0, f64::NAN, f64::NAN, 4.0],
        vec![f64::NAN, 2.0, f64::NAN, 4.0],
    );
    assert_eq!(table.compute_diff("u", 1), 2);
    let diff = &table.diffs[0];
    assert_eq!(diff.indices, vec![0, 1]);
    assert!(diff.new_values[0].is_nan());
    assert_eq!(diff.new_values[1], 2.0);
}

#[test]
fn compute_diff_distinguishes_negative_zero() {
    let mut table = table_with("u", vec![0.0, 1.0], vec![-0.0, 1.0]);
    assert_eq!(table.compute_diff("u", 1), 1);
    assert_eq!(table.diffs[0].indices, vec![0]);
    assert!(table.diffs[0].new_values[0].is_sign_negative());
}

#[test]
fn compute_diff_records_elements_past_the_previous_length() {
    let mut table = table_with("u", vec![1.0, 2.0], vec![1.0, 5.0, 7.0, 8.0]);
    assert_eq!(table.compute_diff("u", 1), 3);
    let diff = &table.diffs[0];
    assert_eq!(diff.indices, vec![1, 2, 3]);
    assert_eq!(diff.new_values, vec![5.0, 7.0, 8.0]);

    // A slice cannot grow, so the tail is ignored there ...
    let mut fixed = vec![1.0, 2.0];
    SnapshotTable::apply_diff(&mut fixed, diff);
    assert_eq!(fixed, vec![1.0, 5.0]);
    // ... while the resizing variant reproduces the new field.
    let mut growable = vec![1.0, 2.0];
    SnapshotTable::apply_diff_resizing(&mut growable, diff);
    assert_eq!(growable, vec![1.0, 5.0, 7.0, 8.0]);
}

#[test]
fn compute_diff_of_a_field_missing_in_the_previous_snapshot_lists_every_element() {
    let mut table = SnapshotTable::new();
    table.insert(Snapshot::new(0.0));
    let mut later = Snapshot::new(1.0);
    later.set_field("p", vec![0.0, 0.0, 9.0]);
    table.insert(later);
    assert_eq!(table.compute_diff("p", 1), 3);
    assert_eq!(table.diffs[0].indices, vec![0, 1, 2]);
}

#[test]
fn diff_applied_to_the_previous_field_reproduces_the_next_one_bit_for_bit() {
    let mut rng = XorShift(0x1234_5678_9ABC_DEF1);
    let previous: Vec<f64> = (0..64).map(|i| (i as f64 + 1.0) * 1e-19).collect();
    let mut next = previous.clone();
    for slot in next.iter_mut() {
        match rng.next_u64() % 6 {
            0 => *slot *= 1.5,
            1 => *slot = f64::NAN,
            2 => *slot = -0.0,
            _ => {}
        }
    }
    next.extend([1e-300, f64::INFINITY]);
    let mut table = table_with("u", previous.clone(), next.clone());
    let changed = table.compute_diff("u", 1);
    assert!(changed >= 2);
    let mut rebuilt = previous;
    SnapshotTable::apply_diff_resizing(&mut rebuilt, &table.diffs[0]);
    assert_eq!(bits(&rebuilt), bits(&next));
}

#[test]
fn apply_diff_resizing_ignores_indices_that_cannot_belong_to_the_base() {
    let diff = DiffEntry {
        time: 0.0,
        field: "u".to_string(),
        indices: vec![usize::MAX, 1],
        new_values: vec![9.0, 7.0],
    };
    let mut base = vec![0.0, 0.0];
    SnapshotTable::apply_diff_resizing(&mut base, &diff);
    assert_eq!(base, vec![0.0, 7.0]);
}

// --- Latin-hypercube design --------------------------------------------------

#[test]
fn latin_hypercube_over_the_unit_range_visits_every_stratum_midpoint() {
    let mut sweep = ParameterSweep::new();
    sweep.add_param("x", vec![0.0, 1.0]);
    let samples = sweep.latin_hypercube_sample(10);
    assert_eq!(samples.len(), 10);
    let mut xs: Vec<f64> = samples.iter().map(|s| s["x"]).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    for (i, x) in xs.iter().enumerate() {
        let expected = (i as f64 + 0.5) / 10.0;
        assert!(
            (x - expected).abs() < 1e-12,
            "sample {i}: {x} vs {expected}"
        );
    }
}

#[test]
fn latin_hypercube_interpolates_between_multiple_knots() {
    let mut sweep = ParameterSweep::new();
    sweep.add_param("x", vec![0.0, 1.0, 4.0]);
    let mut xs: Vec<f64> = sweep
        .latin_hypercube_sample(4)
        .iter()
        .map(|s| s["x"])
        .collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    // Stratum midpoints 1/8, 3/8, 5/8, 7/8 map to knot positions 0.25, 0.75, 1.25, 1.75.
    let expected = [0.25, 0.75, 1.75, 3.25];
    for (x, e) in xs.iter().zip(expected) {
        assert!((x - e).abs() < 1e-12, "{x} vs {e}");
    }
}

#[test]
fn latin_hypercube_uses_every_stratum_of_every_dimension_once() {
    let mut sweep = ParameterSweep::new();
    sweep.add_param("a", vec![0.0, 10.0]);
    sweep.add_param("b", vec![-1.0, 1.0, 5.0]);
    sweep.add_param("c", vec![3.0]);
    let n = 8;
    let samples = sweep.latin_hypercube_sample_seeded(n, 42);
    assert_eq!(samples.len(), n);

    let mut a_strata: Vec<usize> = samples
        .iter()
        .map(|s| (s["a"] / 10.0 * n as f64).floor() as usize)
        .collect();
    a_strata.sort_unstable();
    assert_eq!(a_strata, (0..n).collect::<Vec<_>>());

    // "b" has knots at u = 0, 0.5, 1; invert the piecewise-linear map.
    let mut b_strata: Vec<usize> = samples
        .iter()
        .map(|s| {
            let v = s["b"];
            let u = if v <= 1.0 {
                (v + 1.0) / 2.0 * 0.5
            } else {
                0.5 + (v - 1.0) / 4.0 * 0.5
            };
            (u * n as f64).floor() as usize
        })
        .collect();
    b_strata.sort_unstable();
    assert_eq!(b_strata, (0..n).collect::<Vec<_>>());

    assert!(samples.iter().all(|s| s["c"] == 3.0));
}

#[test]
fn latin_hypercube_shuffles_each_dimension_independently() {
    let mut sweep = ParameterSweep::new();
    for name in ["x", "y", "z"] {
        sweep.add_param(name, vec![0.0, 1.0]);
    }
    let samples = sweep.latin_hypercube_sample_seeded(32, 7);
    let column = |name: &str| -> Vec<u64> { samples.iter().map(|s| s[name].to_bits()).collect() };
    assert_ne!(column("x"), column("y"));
    assert_ne!(column("y"), column("z"));
    assert_ne!(column("x"), column("z"));
}

#[test]
fn latin_hypercube_seed_controls_the_design() {
    let mut sweep = ParameterSweep::new();
    sweep.add_param("x", vec![0.0, 1.0]);
    sweep.add_param("y", vec![0.0, 1.0]);
    let first = sweep.latin_hypercube_sample_seeded(16, 1);
    assert_eq!(first, sweep.latin_hypercube_sample_seeded(16, 1));
    assert_ne!(first, sweep.latin_hypercube_sample_seeded(16, 2));
    assert_eq!(
        sweep.latin_hypercube_sample(16),
        sweep.latin_hypercube_sample_seeded(16, 0)
    );
}

#[test]
fn latin_hypercube_handles_single_samples_and_missing_candidates() {
    let mut sweep = ParameterSweep::new();
    sweep.add_param("x", vec![0.0, 10.0]);
    sweep.add_param("empty", Vec::new());
    let samples = sweep.latin_hypercube_sample(1);
    assert_eq!(samples.len(), 1);
    assert!((samples[0]["x"] - 5.0).abs() < 1e-12);
    assert!(!samples[0].contains_key("empty"));
    assert!(sweep.latin_hypercube_sample(0).is_empty());
    assert!(ParameterSweep::new().latin_hypercube_sample(4).is_empty());
}
