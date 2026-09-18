//! Regression tests for the statistical-mechanics fixes: ground-state shifted
//! Boltzmann weights (no underflow or overflow of the partition function) and
//! the FFT based Green-Kubo autocorrelation.
//!
//! Reference values were computed independently with 60-digit decimal
//! arithmetic.

use super::*;

/// Asserts `|actual - expected| <= rel * |expected| + abs`.
fn assert_close(actual: f64, expected: f64, rel: f64, abs: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs() + abs,
        "{what}: got {actual}, expected {expected}"
    );
}

/// `ln(1 + e^-1)`.
const LN_1_PLUS_E_INV: f64 = 0.313_261_687_518_222_8;
/// `1 / (1 + e)`: the excited-state probability of a gap-1 two-level system at `T = 1`.
const P_EXCITED: f64 = 0.268_941_421_369_995_1;

// ---------------------------------------------------------------------------
// PartitionFunction
// ---------------------------------------------------------------------------

#[test]
fn free_energy_is_finite_when_the_plain_partition_function_underflows() {
    // exp(-1000) is below the smallest f64, so the plain sum is 0 and the old
    // free energy was +inf, the average energy 0 and the heat capacity 0.
    let pf = PartitionFunction::canonical(vec![1000.0, 1001.0], vec![1.0, 1.0], 1.0);
    assert_eq!(pf.canonical_z(), 0.0, "Z itself is not representable");
    assert_close(pf.free_energy(), 1000.0 - LN_1_PLUS_E_INV, 1e-14, 0.0, "F");
    assert_close(
        pf.ln_canonical_z(),
        -1000.0 + LN_1_PLUS_E_INV,
        1e-14,
        0.0,
        "ln Z",
    );
    assert_close(pf.average_energy(), 1000.0 + P_EXCITED, 1e-14, 0.0, "<E>");
    assert_close(
        pf.heat_capacity(),
        (1.0 - P_EXCITED) * P_EXCITED,
        1e-13,
        0.0,
        "C_V",
    );
    assert_close(pf.entropy(), LN_1_PLUS_E_INV + P_EXCITED, 1e-13, 0.0, "S");
}

#[test]
fn thermodynamics_is_finite_when_the_plain_partition_function_overflows() {
    // exp(1001) overflows, so the plain sum was inf and <E>, S and C_V were NaN.
    let pf = PartitionFunction::canonical(vec![-1000.0, -1001.0], vec![1.0, 1.0], 1.0);
    assert!(
        pf.canonical_z().is_infinite(),
        "Z itself is not representable"
    );
    assert_close(
        pf.ln_canonical_z(),
        1001.0 + LN_1_PLUS_E_INV,
        1e-14,
        0.0,
        "ln Z",
    );
    assert_close(pf.free_energy(), -1001.0 - LN_1_PLUS_E_INV, 1e-14, 0.0, "F");
    assert_close(pf.average_energy(), -1001.0 + P_EXCITED, 1e-14, 0.0, "<E>");
    assert_close(
        pf.heat_capacity(),
        (1.0 - P_EXCITED) * P_EXCITED,
        1e-12,
        0.0,
        "C_V",
    );
    let entropy = pf.entropy();
    assert_close(entropy, LN_1_PLUS_E_INV + P_EXCITED, 1e-12, 0.0, "S");
    // The Gibbs entropy of the two occupation probabilities.
    let ground = 1.0 - P_EXCITED;
    let gibbs = -ground * ground.ln() - P_EXCITED * P_EXCITED.ln();
    assert_close(entropy, gibbs, 1e-12, 0.0, "S against -sum p ln p");
}

#[test]
fn thermodynamics_is_invariant_under_a_common_energy_offset() {
    // Shifting every level by 1000 shifts F and <E> by 1000 and leaves S and
    // C_V unchanged; only the shifted evaluation can show this.
    let levels = vec![0.0, 1.0, 2.5];
    let degeneracies = vec![1.0, 3.0, 2.0];
    let shifted_levels: Vec<f64> = levels.iter().map(|e| e + 1000.0).collect();
    for temperature in [0.4, 1.0, 7.5] {
        let base = PartitionFunction::canonical(levels.clone(), degeneracies.clone(), temperature);
        let moved =
            PartitionFunction::canonical(shifted_levels.clone(), degeneracies.clone(), temperature);
        assert_close(
            moved.free_energy() - base.free_energy(),
            1000.0,
            0.0,
            1e-11,
            "F offset",
        );
        assert_close(
            moved.average_energy() - base.average_energy(),
            1000.0,
            0.0,
            1e-11,
            "<E> offset",
        );
        assert_close(moved.entropy(), base.entropy(), 1e-12, 1e-12, "S");
        assert_close(
            moved.heat_capacity(),
            base.heat_capacity(),
            1e-11,
            1e-13,
            "C_V",
        );
    }
}

#[test]
fn degenerate_levels_far_from_zero_match_the_closed_form() {
    // g = [2, 3] at E = [1000, 1001], T = 2:
    // Z' = 2 + 3 exp(-1/2) = 3.8195919791379003.
    let pf = PartitionFunction::canonical(vec![1000.0, 1001.0], vec![2.0, 3.0], 2.0);
    assert_close(pf.free_energy(), 997.319_712_789_660_6, 1e-14, 0.0, "F");
    assert_close(
        pf.average_energy(),
        1_000.476_383_862_223,
        1e-14,
        0.0,
        "<E>",
    );
    assert_close(
        pf.heat_capacity(),
        0.062_360_569_509_125_04,
        1e-12,
        0.0,
        "C_V",
    );
    assert_close(pf.entropy(), 1.578_335_536_281_223_6, 1e-12, 0.0, "S");
}

#[test]
fn an_empty_ground_level_does_not_bring_the_underflow_back() {
    // The lowest level has zero degeneracy; the shift must be taken over the
    // populated levels, otherwise exp(-1000) underflows again.
    let pf = PartitionFunction::canonical(vec![0.0, 1000.0], vec![0.0, 1.0], 1.0);
    assert_eq!(pf.free_energy(), 1000.0);
    assert_eq!(pf.average_energy(), 1000.0);
    assert_eq!(pf.heat_capacity(), 0.0);
    assert_eq!(pf.entropy(), 0.0);
    assert_eq!(pf.ln_canonical_z(), -1000.0);
}

#[test]
fn a_spectrum_without_populated_levels_keeps_the_documented_fallbacks() {
    let pf = PartitionFunction::canonical(vec![0.0, 1.0], vec![0.0, 0.0], 1.0);
    assert_eq!(pf.canonical_z(), 0.0);
    assert_eq!(pf.free_energy(), f64::INFINITY);
    assert_eq!(pf.average_energy(), 0.0);
    assert_eq!(pf.heat_capacity(), 0.0);
    assert_eq!(pf.ln_canonical_z(), f64::NEG_INFINITY);
    let none = PartitionFunction::canonical(vec![], vec![], 1.0);
    assert_eq!(none.canonical_z(), 0.0);
    assert_eq!(none.free_energy(), f64::INFINITY);
}

#[test]
fn ordinary_spectra_agree_with_the_plain_sums() {
    // The shift must not change anything in the well-conditioned regime.
    let energies = [0.0, 1.0, 2.0, 3.5];
    let degeneracies = [1.0, 2.0, 3.0, 1.0];
    let temperature = 1.5;
    let pf = PartitionFunction::canonical(energies.to_vec(), degeneracies.to_vec(), temperature);
    let beta = 1.0 / temperature;
    let z: f64 = energies
        .iter()
        .zip(&degeneracies)
        .map(|(e, g)| g * (-beta * e).exp())
        .sum();
    let e_sum: f64 = energies
        .iter()
        .zip(&degeneracies)
        .map(|(e, g)| g * e * (-beta * e).exp())
        .sum();
    let e2_sum: f64 = energies
        .iter()
        .zip(&degeneracies)
        .map(|(e, g)| g * e * e * (-beta * e).exp())
        .sum();
    let mean = e_sum / z;
    assert_close(pf.canonical_z(), z, 1e-14, 0.0, "Z");
    assert_close(pf.free_energy(), -temperature * z.ln(), 1e-13, 1e-14, "F");
    assert_close(pf.average_energy(), mean, 1e-13, 1e-14, "<E>");
    assert_close(
        pf.heat_capacity(),
        (e2_sum / z - mean * mean) / (temperature * temperature),
        1e-12,
        1e-14,
        "C_V",
    );
    assert_close(
        pf.entropy(),
        (mean + temperature * z.ln()) / temperature,
        1e-12,
        1e-14,
        "S",
    );
}

#[test]
fn a_ground_state_at_a_subnormal_temperature_keeps_unit_weight() {
    // 1 / 1e-320 overflows to inf; the ground level must stay at weight g, not
    // become NaN through 0 * inf.
    let pf = PartitionFunction::canonical(vec![0.0, 1.0], vec![2.0, 1.0], 1.0e-320);
    assert_eq!(pf.canonical_z(), 2.0);
    // F = -T ln 2 is a subnormal number.
    assert!(pf.free_energy().abs() < 1.0e-319);
    assert_eq!(pf.average_energy(), 0.0);
    assert_eq!(pf.heat_capacity(), 0.0);
}

#[test]
fn grand_canonical_z_matches_the_plain_sum_for_ordinary_spectra() {
    let energies = vec![0.0, 1.0, 2.0];
    let degeneracies = vec![1.0, 2.0, 1.0];
    let particles = vec![0.0, 1.0, 2.0];
    let (temperature, mu) = (1.5, 0.5);
    let pf = PartitionFunction::grand_canonical(
        energies.clone(),
        degeneracies.clone(),
        particles.clone(),
        temperature,
        mu,
    );
    let beta = 1.0 / temperature;
    let xi: f64 = energies
        .iter()
        .zip(&degeneracies)
        .zip(&particles)
        .map(|((e, g), n)| g * (-beta * (e - mu * n)).exp())
        .sum();
    assert_close(pf.grand_canonical_z(), xi, 1e-14, 0.0, "Xi");
    assert_close(pf.ln_grand_canonical_z(), xi.ln(), 1e-14, 1e-14, "ln Xi");
}

#[test]
fn grand_canonical_logarithm_survives_overflow_and_underflow() {
    // E - mu N = -1000 and -2000: Xi = e^1000 + e^2000 overflows.
    let overflowing = PartitionFunction::grand_canonical(
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        1.0,
        1000.0,
    );
    assert!(overflowing.grand_canonical_z().is_infinite());
    assert_close(
        overflowing.ln_grand_canonical_z(),
        2000.0,
        1e-14,
        1e-12,
        "ln Xi",
    );
    // E - mu N = 1000 and 1001: Xi = e^-1000 (1 + e^-1) underflows to 0.
    let underflowing =
        PartitionFunction::grand_canonical(vec![1000.0, 1001.0], vec![1.0, 1.0], vec![], 1.0, 0.7);
    assert_eq!(underflowing.grand_canonical_z(), 0.0);
    assert_close(
        underflowing.ln_grand_canonical_z(),
        -1000.0 + LN_1_PLUS_E_INV,
        1e-14,
        0.0,
        "ln Xi",
    );
}

#[test]
fn non_positive_temperature_keeps_the_documented_fallbacks() {
    let pf = PartitionFunction::canonical(vec![0.0, 1.0], vec![1.0, 1.0], 0.0);
    assert_eq!(pf.canonical_z(), 0.0);
    assert_eq!(pf.grand_canonical_z(), 0.0);
    assert_eq!(pf.free_energy(), f64::INFINITY);
    assert_eq!(pf.average_energy(), 0.0);
    assert_eq!(pf.entropy(), 0.0);
    assert_eq!(pf.heat_capacity(), 0.0);
    assert_eq!(pf.ln_canonical_z(), f64::NEG_INFINITY);
    assert_eq!(pf.ln_grand_canonical_z(), f64::NEG_INFINITY);
}

// ---------------------------------------------------------------------------
// BoltzmannDistribution
// ---------------------------------------------------------------------------

#[test]
fn boltzmann_probabilities_are_normalised_when_every_raw_weight_underflows() {
    // Raw weights exp(-1000), exp(-1001), exp(-1002) are all 0: the old
    // probabilities were all zero and the free energy +inf.
    let bd = BoltzmannDistribution::uniform(vec![1000.0, 1001.0, 1002.0], 1.0);
    assert_eq!(bd.weight(0), 0.0, "the documented raw weight is unchanged");
    let probabilities = bd.probabilities();
    assert_close(probabilities[0], 0.665_240_955_774_821_9, 1e-14, 0.0, "P_0");
    assert_close(
        probabilities[1],
        0.244_728_471_054_797_65,
        1e-14,
        0.0,
        "P_1",
    );
    assert_close(
        probabilities[2],
        0.090_030_573_170_380_46,
        1e-14,
        0.0,
        "P_2",
    );
    assert_close(probabilities.iter().sum::<f64>(), 1.0, 1e-14, 0.0, "sum P");
    assert_eq!(bd.probability(1), probabilities[1]);
    assert_close(bd.free_energy(), 999.592_394_035_555_6, 1e-14, 0.0, "F");
    assert_close(bd.mean_energy(), 1_000.424_789_617_395_6, 1e-14, 0.0, "<E>");
    assert!(bd.entropy() > 0.0 && bd.entropy().is_finite());
}

#[test]
fn boltzmann_probabilities_are_normalised_when_every_raw_weight_overflows() {
    // Raw weights exp(1000) ... are all inf: the old probabilities were NaN.
    let bd = BoltzmannDistribution::uniform(vec![-1000.0, -1001.0, -1002.0], 1.0);
    assert!(bd.weight(2).is_infinite());
    let probabilities = bd.probabilities();
    assert_close(
        probabilities[0],
        0.090_030_573_170_380_46,
        1e-14,
        0.0,
        "P_0",
    );
    assert_close(
        probabilities[1],
        0.244_728_471_054_797_65,
        1e-14,
        0.0,
        "P_1",
    );
    assert_close(probabilities[2], 0.665_240_955_774_821_9, 1e-14, 0.0, "P_2");
    assert_close(probabilities.iter().sum::<f64>(), 1.0, 1e-14, 0.0, "sum P");
    assert_close(bd.free_energy(), -1_002.407_605_964_444_4, 1e-14, 0.0, "F");
    assert_close(
        bd.mean_energy(),
        -1_001.575_210_382_604_4,
        1e-14,
        0.0,
        "<E>",
    );
}

#[test]
fn boltzmann_entropy_does_not_depend_on_a_common_energy_offset() {
    let near = BoltzmannDistribution::new(vec![0.0, 0.5, 1.5], vec![1.0, 2.0, 1.0], 0.8);
    let far = BoltzmannDistribution::new(vec![-900.0, -899.5, -898.5], vec![1.0, 2.0, 1.0], 0.8);
    assert_close(far.entropy(), near.entropy(), 1e-12, 1e-14, "S");
    assert_close(
        far.free_energy() - near.free_energy(),
        -900.0,
        0.0,
        1e-11,
        "F offset",
    );
}

#[test]
fn boltzmann_probability_of_a_missing_level_is_zero() {
    let bd = BoltzmannDistribution::uniform(vec![0.0, 1.0], 1.0);
    assert_eq!(bd.probability(2), 0.0);
    assert_eq!(bd.probability(usize::MAX), 0.0);
    let cold = BoltzmannDistribution::uniform(vec![0.0, 1.0], 0.0);
    assert_eq!(cold.probability(0), 0.0);
    assert_eq!(cold.probabilities(), vec![0.0, 0.0]);
    assert_eq!(cold.free_energy(), f64::INFINITY);
    assert_eq!(cold.mean_energy(), 0.0);
}

#[test]
fn boltzmann_ordinary_spectra_agree_with_the_plain_weights() {
    let bd = BoltzmannDistribution::new(vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0], 1.3);
    let z: f64 = (0..3).map(|i| bd.weight(i)).sum();
    assert_close(bd.partition_function(), z, 1e-14, 0.0, "Z");
    for i in 0..3 {
        assert_close(bd.probability(i), bd.weight(i) / z, 1e-14, 0.0, "P_i");
    }
    assert_close(bd.free_energy(), -1.3 * z.ln(), 1e-13, 1e-14, "F");
    let mean: f64 = (0..3).map(|i| bd.weight(i) / z * i as f64).sum();
    assert_close(bd.mean_energy(), mean, 1e-13, 1e-14, "<E>");
}

// ---------------------------------------------------------------------------
// GreenKubo::autocorrelation
// ---------------------------------------------------------------------------

/// The `O(n^2)` normalised autocorrelation that the FFT evaluation replaces.
fn direct_autocorrelation(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centred: Vec<f64> = signal.iter().map(|x| x - mean).collect();
    let norm: f64 = centred.iter().map(|x| x * x).sum();
    (0..n)
        .map(|lag| {
            let sum: f64 = (0..n - lag).map(|i| centred[i] * centred[i + lag]).sum();
            sum / norm
        })
        .collect()
}

/// `n` uniform samples in `[-1, 1)`.
fn uniform_series(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SmRng::new(seed);
    (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect()
}

/// A stationary AR(1) series `x_t = phi x_{t-1} + e_t` with uniform noise.
fn ar1_series(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let noise = uniform_series(n, seed);
    let mut x = 0.0;
    noise
        .iter()
        .map(|e| {
            x = phi * x + e;
            x
        })
        .collect()
}

fn assert_series_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (lag, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "{what}: lag {lag}: got {a}, direct sum {b}"
        );
    }
}

#[test]
fn autocorrelation_matches_the_direct_sum_on_a_random_series() {
    let signal = uniform_series(1000, 2_024);
    let fast = GreenKubo::autocorrelation(&signal);
    assert_series_close(&fast, &direct_autocorrelation(&signal), 1e-10, "n = 1000");
}

#[test]
fn autocorrelation_matches_the_direct_sum_for_correlated_signals_and_awkward_lengths() {
    for n in [2, 3, 5, 17, 100, 257, 999, 1024, 1500] {
        let signal: Vec<f64> = ar1_series(n, 0.9, n as u64)
            .iter()
            .enumerate()
            .map(|(t, x)| x + (0.05 * t as f64).sin())
            .collect();
        let fast = GreenKubo::autocorrelation(&signal);
        assert_series_close(
            &fast,
            &direct_autocorrelation(&signal),
            1e-11,
            &format!("n = {n}"),
        );
    }
}

#[test]
fn autocorrelation_is_normalised_and_bounded_by_one() {
    let signal = uniform_series(4096, 5);
    let acf = GreenKubo::autocorrelation(&signal);
    assert_eq!(acf.len(), 4096);
    assert_eq!(acf[0], 1.0);
    assert!(acf.iter().all(|c| c.abs() <= 1.0 + 1e-12));
}

#[test]
fn autocorrelation_of_degenerate_signals() {
    assert!(GreenKubo::autocorrelation(&[]).is_empty());
    // One sample and constant signals have no fluctuations to correlate.
    assert_eq!(GreenKubo::autocorrelation(&[4.0]), vec![0.0]);
    assert_eq!(GreenKubo::autocorrelation(&[2.5; 33]), vec![0.0; 33]);
    // Two samples: centred (-1, 1), so C = (1, -1/2).
    let two = GreenKubo::autocorrelation(&[1.0, 3.0]);
    assert_eq!(two.len(), 2);
    assert_eq!(two[0], 1.0);
    assert!((two[1] + 0.5).abs() < 1e-14, "C(1) = {}", two[1]);
}

#[test]
fn autocorrelation_of_a_long_series_is_log_linear() {
    // 2^17 + 123 samples, an AR(1) process with correlation phi^lag: the direct
    // double sum would need about 8.6e9 multiply-adds for the full lag range.
    let n = (1 << 17) + 123;
    let phi = 0.95;
    let signal = ar1_series(n, phi, 77);
    let acf = GreenKubo::autocorrelation(&signal);
    assert_eq!(acf.len(), n);
    // Rounding: a handful of lags against their O(n) definition.
    let mean = signal.iter().sum::<f64>() / n as f64;
    let norm: f64 = signal.iter().map(|x| (x - mean) * (x - mean)).sum();
    for lag in [1, 2, 10, 500, n / 2, n - 1] {
        let direct: f64 = (0..n - lag)
            .map(|i| (signal[i] - mean) * (signal[i + lag] - mean))
            .sum::<f64>()
            / norm;
        assert!(
            (acf[lag] - direct).abs() < 1e-11,
            "lag {lag}: fft {}, direct {direct}",
            acf[lag]
        );
    }
    // Statistics: C(tau) = phi^tau up to a sampling error of about 0.01.
    for lag in [1_usize, 5, 20] {
        let expected = phi.powi(lag as i32);
        assert!(
            (acf[lag] - expected).abs() < 0.04,
            "lag {lag}: C = {}, expected {expected}",
            acf[lag]
        );
    }
}

// ---------------------------------------------------------------------------
// Seeding of the Ising model's random number generator
// ---------------------------------------------------------------------------

/// Largest `|r|` over the first eight draw positions of the Pearson correlation,
/// across seeds, between the draw of seed `s` and the draw of seed `s + 1`.
fn max_consecutive_seed_correlation(draws: impl Fn(u64) -> [f64; 8]) -> f64 {
    let seeds = 20_000_u64;
    let first_seed = 12_345_u64;
    let table: Vec<[f64; 8]> = (0..=seeds).map(|i| draws(first_seed + i)).collect();
    let count = seeds as usize;
    (0..8)
        .map(|k| {
            let current: Vec<f64> = table[..count].iter().map(|row| row[k]).collect();
            let next: Vec<f64> = table[1..].iter().map(|row| row[k]).collect();
            crate::statistics::correlation(&current, &next).abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn consecutive_seeds_of_the_ising_rng_give_uncorrelated_streams() {
    // Raw LCG seeding gave correlations of -0.36 to 0.89 between the draws of
    // neighbouring seeds; the standard error of an uncorrelated pair is 0.007.
    let worst = max_consecutive_seed_correlation(|seed| {
        let mut rng = SmRng::new(seed);
        std::array::from_fn(|_| rng.next_f64())
    });
    assert!(worst < 0.05, "max |r| = {worst}");
}

#[test]
fn randomised_lattices_of_consecutive_seeds_are_unrelated() {
    // The fraction of sites on which the lattices of two seeds agree is 1/2 for
    // unrelated configurations; average it over many neighbouring seed pairs.
    let mut agreeing = 0_usize;
    let mut total = 0_usize;
    for seed in 100..164_u64 {
        let mut a = IsingModel::new_2d(16, 16, 1.0, 0.0);
        let mut b = IsingModel::new_2d(16, 16, 1.0, 0.0);
        a.randomise(seed);
        b.randomise(seed + 1);
        agreeing += a.spins.iter().zip(&b.spins).filter(|(x, y)| x == y).count();
        total += a.spins.len();
    }
    let fraction = agreeing as f64 / total as f64;
    // The standard error is 0.5 / sqrt(16384) = 0.004.
    assert!((fraction - 0.5).abs() < 0.02, "agreement = {fraction}");
}
