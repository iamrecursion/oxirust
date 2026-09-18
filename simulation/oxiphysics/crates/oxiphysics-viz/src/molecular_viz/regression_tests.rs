//! Regression tests for the molecular visualization fixes: the normalisation
//! of the radial distribution function.

use super::*;
use std::f64::consts::PI;

/// Deterministic 64-bit LCG (Knuth's MMIX constants) returning values in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Volume of the spherical shell between `r_lo` and `r_hi`.
fn shell_volume(r_lo: f64, r_hi: f64) -> f64 {
    4.0 / 3.0 * PI * (r_hi.powi(3) - r_lo.powi(3))
}

#[test]
fn rdf_of_an_ideal_gas_tends_to_one() {
    // 2000 uniformly random atoms in a 20 A cube; the reference atoms A are
    // those at least r_max away from every wall, so their shells are complete.
    let box_len = 20.0;
    let n_atoms = 2000usize;
    let r_max = 5.0;
    let mut rng = Lcg(0x5eed_1234_abcd_ef01);
    let positions: Vec<[f64; 3]> = (0..n_atoms)
        .map(|_| {
            [
                rng.next_unit() * box_len,
                rng.next_unit() * box_len,
                rng.next_unit() * box_len,
            ]
        })
        .collect();
    let interior = |p: &[f64; 3]| p.iter().all(|c| (r_max..=box_len - r_max).contains(c));
    let indices_a: Vec<usize> = (0..n_atoms).filter(|&i| interior(&positions[i])).collect();
    let indices_b: Vec<usize> = (0..n_atoms).collect();
    assert!(
        indices_a.len() > 150,
        "only {} reference atoms",
        indices_a.len()
    );

    let number_density = n_atoms as f64 / box_len.powi(3);
    let rdf = RadialDistribution::compute(
        &positions,
        &indices_a,
        &indices_b,
        r_max,
        50,
        number_density,
        "gas-gas",
    );
    assert_eq!(rdf.r.len(), 50);
    assert_eq!(rdf.g_r.len(), 50);
    let shell: Vec<f64> = rdf
        .r
        .iter()
        .zip(&rdf.g_r)
        .filter(|&(&r, _)| (2.0..=5.0).contains(&r))
        .map(|(_, &g)| g)
        .collect();
    assert!(shell.len() >= 25, "{} bins between 2 and 5 A", shell.len());
    let mean = shell.iter().sum::<f64>() / shell.len() as f64;
    // N_B g(r) used to be ~1 / N_B = 0.0005 here.
    assert!((mean - 1.0).abs() < 0.05, "mean g(r) = {mean}");
}

/// A ring of 12 B atoms at distance 1.55 A around `centre`, each pair of
/// neighbours on the ring more than 0.4 A apart (so no extra pairs fall in
/// the first bins).
fn ring_around(centre: [f64; 3]) -> Vec<[f64; 3]> {
    (0..12)
        .map(|k| {
            let phi = 2.0 * PI * k as f64 / 12.0;
            [
                centre[0] + 1.55 * phi.cos(),
                centre[1] + 1.55 * phi.sin(),
                centre[2],
            ]
        })
        .collect()
}

#[test]
fn rdf_normalises_by_the_number_of_reference_atoms() {
    // One reference atom with 12 neighbours in the bin [1.5, 1.6) A.
    let density = 0.05;
    let r_max = 3.0;
    let mut positions = vec![[0.0, 0.0, 0.0]];
    positions.extend(ring_around([0.0, 0.0, 0.0]));
    let b: Vec<usize> = (1..=12).collect();
    let one = RadialDistribution::compute(&positions, &[0], &b, r_max, 30, density, "A-B");
    let expected = 12.0 / (density * shell_volume(1.5, 1.6));
    assert!(
        (one.g_r[15] - expected).abs() < 1e-9 * expected,
        "g = {} vs {expected}",
        one.g_r[15]
    );
    // Every other bin is empty.
    assert!(
        one.g_r
            .iter()
            .enumerate()
            .all(|(i, &g)| i == 15 || g == 0.0)
    );

    // A second, distant reference atom with its own 12 neighbours doubles the
    // pair count and N_A, so g(r) is unchanged (it used to shrink with N_A * N_B).
    positions.push([100.0, 0.0, 0.0]);
    positions.extend(ring_around([100.0, 0.0, 0.0]));
    let b: Vec<usize> = (1..positions.len()).filter(|&i| i != 13).collect();
    let two = RadialDistribution::compute(&positions, &[0, 13], &b, r_max, 30, density, "A-B");
    assert!(
        (two.g_r[15] - expected).abs() < 1e-9 * expected,
        "g = {} vs {expected}",
        two.g_r[15]
    );
}

#[test]
fn rdf_skips_self_pairs_and_unknown_indices() {
    let positions = vec![[0.0, 0.0, 0.0], [2.05, 0.0, 0.0]];
    let density = 0.1;
    // A = B = both atoms: the two ordered pairs (0,1) and (1,0) at 2.05 A.
    let same = RadialDistribution::compute(&positions, &[0, 1], &[0, 1], 4.0, 40, density, "X-X");
    let expected = 2.0 / (2.0 * density * shell_volume(2.0, 2.1));
    assert!(
        (same.g_r[20] - expected).abs() < 1e-9 * expected,
        "g = {}",
        same.g_r[20]
    );
    assert!(
        same.g_r
            .iter()
            .enumerate()
            .all(|(i, &g)| i == 20 || g == 0.0)
    );
    // Indices outside the position list are ignored and do not count as
    // reference atoms.
    let with_unknown =
        RadialDistribution::compute(&positions, &[0, 99], &[1, 77], 4.0, 40, density, "X-Y");
    let expected = 1.0 / (density * shell_volume(2.0, 2.1));
    assert!((with_unknown.g_r[20] - expected).abs() < 1e-9 * expected);
}

#[test]
fn rdf_counts_only_pairs_closer_than_r_max() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
    ];
    // r_max = 2: the pair at exactly 2.0 and the one at 2.5 are outside.
    let rdf = RadialDistribution::compute(&positions, &[0], &[1, 2, 3], 2.0, 8, 0.1, "A-B");
    let counted: f64 = rdf
        .g_r
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            let r_lo = i as f64 * rdf.dr;
            g * 0.1 * shell_volume(r_lo, r_lo + rdf.dr)
        })
        .sum();
    assert!((counted - 1.0).abs() < 1e-9, "{counted} pairs counted");
}

#[test]
fn rdf_without_bins_or_range_is_empty() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    for (r_max, n_bins) in [
        (5.0, 0),
        (0.0, 10),
        (-1.0, 10),
        (f64::NAN, 10),
        (f64::INFINITY, 10),
    ] {
        let rdf = RadialDistribution::compute(&positions, &[0], &[1], r_max, n_bins, 0.03, "A-B");
        assert!(
            rdf.r.is_empty() && rdf.g_r.is_empty(),
            "({r_max}, {n_bins})"
        );
        assert_eq!(rdf.dr, 0.0);
        assert_eq!(rdf.first_peak_r(), None);
        assert_eq!(rdf.pair_label, "A-B");
    }
}

#[test]
fn rdf_with_no_usable_density_is_zero() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    for density in [0.0, -0.1, f64::NAN] {
        let rdf = RadialDistribution::compute(&positions, &[0], &[1], 3.0, 12, density, "A-B");
        assert!(rdf.g_r.iter().all(|&g| g == 0.0), "density {density}");
    }
}
