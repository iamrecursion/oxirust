//! Regression tests for the stochastic-process fixes: the Langevin thermostat
//! noise scale, the SABR smile factor, the Hosking fractional-Gaussian-noise
//! sampler, the BAOAB force update, the Kleinman-Kramers noise amplitude, the
//! Kramers escape time and the seeding of the linear congruential generator.

use super::functions::{K_B, mean_first_passage_time};
use super::types::{FractionalBrownianMotion, KleinmanKramers, LangevinDynamics, Rng, SabrModel};
use crate::error::PhysicsError;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// LangevinDynamics::step
// ---------------------------------------------------------------------------

/// Stationary `m <v²> / (k_B T)` of a force-free particle after a burn-in.
fn equipartition_ratio(thermostat: &LangevinDynamics, dt: f64, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut vel = 0.0_f64;
    for _ in 0..10_000 {
        vel = thermostat.step(0.0, vel, 0.0, dt, &mut rng).1;
    }
    let steps = 1_000_000_u32;
    let mut sum_v2 = 0.0_f64;
    for _ in 0..steps {
        vel = thermostat.step(0.0, vel, 0.0, dt, &mut rng).1;
        sum_v2 += vel * vel;
    }
    thermostat.mass * (sum_v2 / f64::from(steps)) / (K_B * thermostat.temperature)
}

#[test]
fn langevin_step_thermalises_to_the_equipartition_temperature() {
    // gamma * dt = 0.01 and 0.05: the Euler-Maruyama scheme has the stationary
    // variance (k_B T / m) / (1 - gamma dt / 2), i.e. +0.5 % and +2.6 %.
    let thermostat = LangevinDynamics::new(1.0e-26, 1.0e10, 300.0);
    for dt in [1.0e-12, 5.0e-12] {
        let ratio = equipartition_ratio(&thermostat, dt, 20_240_611);
        let predicted = 1.0 / (1.0 - 0.5 * thermostat.friction * dt);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "dt = {dt}: m<v^2>/(k_B T) = {ratio}, expected about 1"
        );
        assert!(
            (ratio - predicted).abs() < 0.05,
            "dt = {dt}: m<v^2>/(k_B T) = {ratio}, scheme predicts {predicted}"
        );
    }
}

#[test]
fn langevin_random_kick_has_the_euler_maruyama_scale() {
    // From rest and without force the new velocity is exactly the random kick,
    // whose standard deviation is sqrt(2 gamma k_B T dt / m): it must not be
    // divided by sqrt(dt) once more.
    let thermostat = LangevinDynamics::new(1.0e-26, 1.0e10, 300.0);
    for dt in [1.0e-15, 1.0e-12, 5.0e-12] {
        let mut rng = Rng::new(99);
        let samples = 40_000_u32;
        let mut sum_sq = 0.0_f64;
        for _ in 0..samples {
            let kick = thermostat.step(0.0, 0.0, 0.0, dt, &mut rng).1;
            sum_sq += kick * kick;
        }
        let rms = (sum_sq / f64::from(samples)).sqrt();
        let expected = (2.0 * thermostat.friction * K_B * thermostat.temperature / thermostat.mass
            * dt)
            .sqrt();
        assert!(
            (rms / expected - 1.0).abs() < 0.03,
            "dt = {dt}: kick rms {rms}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// SabrModel::implied_vol_approx
// ---------------------------------------------------------------------------

/// Reference implied volatilities of Hagan et al.'s formula (2.17a), evaluated
/// in 40-digit arithmetic.
fn assert_smile(sabr: &SabrModel, t: f64, reference: &[(f64, f64)]) {
    for &(strike, expected) in reference {
        let iv = sabr.implied_vol_approx(strike, t);
        assert!(
            (iv - expected).abs() <= 1e-11 * expected.abs(),
            "K = {strike}: implied vol {iv}, reference {expected}"
        );
    }
}

#[test]
fn sabr_smile_matches_the_hagan_formula() {
    let rates = SabrModel::new(0.04, 0.2, 0.5, 0.4, -0.3);
    assert_smile(
        &rates,
        1.0,
        &[
            (0.04, 1.00695),
            (0.0399, 1.00773529863871),
            (0.0401, 1.00616723008142),
            (0.039, 1.01491899463123),
            (0.041, 0.999233960637095),
            (0.03, 1.10072140328405),
            (0.06, 0.887947411358506),
        ],
    );
    let equity = SabrModel::new(100.0, 0.2, 0.5, 0.3, -0.2);
    assert_smile(
        &equity,
        1.0,
        &[
            (100.0, 0.0201380833333333),
            (102.0, 0.0197347669099528),
            (98.0, 0.0211121242317547),
            (90.0, 0.0280228245031828),
            (120.0, 0.0285592064437594),
        ],
    );
    let lognormal = SabrModel::new(1.0, 0.3, 1.0, 0.5, 0.4);
    assert_smile(
        &lognormal,
        2.0,
        &[
            (0.8, 0.301404230190628),
            (1.0, 0.3185),
            (1.25, 0.346707817780297),
        ],
    );
}

#[test]
fn sabr_smile_is_continuous_across_the_at_the_money_branch() {
    // The smile slope here is about -7.9 per unit strike: strikes closer to the
    // forward than 1e-8 take the ATM branch, all others the general branch, and
    // the two must join without a jump.
    let sabr = SabrModel::new(0.04, 0.2, 0.5, 0.4, -0.3);
    let atm = sabr.implied_vol_approx(0.04, 1.0);
    for offset in [1.0e-4, 1.0e-6, 4.0e-8, 2.0e-8] {
        for strike in [0.04 - offset, 0.04 + offset] {
            let iv = sabr.implied_vol_approx(strike, 1.0);
            assert!(
                (iv - atm).abs() < 10.0 * offset,
                "K = {strike}: implied vol {iv} jumps away from the ATM value {atm}"
            );
        }
    }
}

#[test]
fn sabr_negative_correlation_gives_a_downward_sloping_skew() {
    let sabr = SabrModel::new(0.04, 0.2, 0.5, 0.4, -0.3);
    let low = sabr.implied_vol_approx(0.039, 1.0);
    let high = sabr.implied_vol_approx(0.041, 1.0);
    assert!(
        low > high,
        "rho < 0 needs iv(0.039) = {low} > iv(0.041) = {high}"
    );
    // Lognormal backbone (beta = 1): the correlation alone carries the skew.
    let lognormal = SabrModel::new(0.04, 0.2, 1.0, 0.4, -0.3);
    assert_smile(
        &lognormal,
        1.0,
        &[(0.039, 0.202706997806936), (0.041, 0.199688947586286)],
    );
    let low = lognormal.implied_vol_approx(0.039, 1.0);
    let high = lognormal.implied_vol_approx(0.041, 1.0);
    assert!(
        low > high,
        "beta = 1, rho < 0 needs iv(0.039) = {low} > iv(0.041) = {high}"
    );
}

#[test]
fn sabr_positive_correlation_tilts_the_smile_upwards() {
    // Lognormal backbone (beta = 1): the sign of the skew is the sign of rho.
    let lognormal = SabrModel::new(0.04, 0.2, 1.0, 0.4, 0.3);
    assert_smile(
        &lognormal,
        1.0,
        &[(0.039, 0.202037632094366), (0.041, 0.205084366114734)],
    );
    let low = lognormal.implied_vol_approx(0.039, 1.0);
    let high = lognormal.implied_vol_approx(0.041, 1.0);
    assert!(
        low < high,
        "beta = 1, rho > 0 needs iv(0.039) = {low} < iv(0.041) = {high}"
    );
    // With beta = 1/2 the CEV backbone (1 - beta = 0.5) outweighs the correlation
    // term rho * lambda = 0.12 (lambda = nu F^(1-beta) / alpha = 0.4), so even
    // rho = +0.3 leaves the smile sloping downwards, only less steeply than
    // rho = -0.3: the leading-order relative slope d ln(iv) / d ln(K / F) is
    // -(1 - beta - rho lambda) / 2.
    let skew = |rho: f64| {
        let sabr = SabrModel::new(0.04, 0.2, 0.5, 0.4, rho);
        sabr.implied_vol_approx(0.041, 1.0) - sabr.implied_vol_approx(0.039, 1.0)
    };
    let (negative, positive) = (skew(-0.3), skew(0.3));
    assert!(
        negative < positive && positive < 0.0,
        "skew(rho = -0.3) = {negative} must lie below skew(rho = +0.3) = {positive} < 0"
    );
    let expected_gap = 0.005_382_434_289;
    assert!(
        (positive - negative - expected_gap).abs() < 1e-9,
        "skew gap {} differs from the reference {expected_gap}",
        positive - negative
    );
}

#[test]
fn sabr_out_of_the_money_volatility_stays_close_to_at_the_money() {
    let sabr = SabrModel::new(100.0, 0.2, 0.5, 0.3, -0.2);
    let atm = sabr.implied_vol_approx(100.0, 1.0);
    let otm = sabr.implied_vol_approx(102.0, 1.0);
    assert!(
        (otm / atm - 1.0).abs() < 0.1,
        "OTM volatility {otm} is not within 10 % of ATM {atm}"
    );
}

// ---------------------------------------------------------------------------
// FractionalBrownianMotion::sample (Hosking / Durbin-Levinson)
// ---------------------------------------------------------------------------

/// Autocovariance of fractional Gaussian noise at an integer lag.
fn fgn_autocovariance(h: f64, dt: f64, lag: usize) -> f64 {
    let k = lag as f64;
    let two_h = 2.0 * h;
    0.5 * ((k + 1.0).powf(two_h) - 2.0 * k.powf(two_h) + (k - 1.0).abs().powf(two_h))
        * dt.powf(two_h)
}

/// `L z` for the lower Cholesky factor `L` of the Toeplitz autocovariance matrix.
fn cholesky_colouring(h: f64, dt: f64, z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let earlier_terms: f64 = l[i][..j].iter().zip(&l[j][..j]).map(|(a, b)| a * b).sum();
            let sum = fgn_autocovariance(h, dt, i - j) - earlier_terms;
            l[i][j] = if i == j { sum.sqrt() } else { sum / l[j][j] };
        }
    }
    (0..n)
        .map(|i| (0..=i).map(|j| l[i][j] * z[j]).sum())
        .collect()
}

#[test]
fn hosking_sampler_reproduces_the_cholesky_colouring_of_the_same_normals() {
    // Durbin-Levinson is the Cholesky factorisation of the Toeplitz covariance
    // in disguise: both must map the same normals to the same sample.
    for h in [0.2, 0.3, 0.5, 0.7, 0.8, 0.95] {
        let dt = 0.37;
        let fbm = FractionalBrownianMotion::new(h, dt);
        let mut rng = Rng::new(2024);
        let z: Vec<f64> = (0..32).map(|_| rng.next_normal()).collect();
        let expected = cholesky_colouring(h, dt, &z);
        let got = fbm.correlate_normals(&z);
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() <= 1e-9 * (1.0 + e.abs()),
                "H = {h}, i = {i}: sampler {g}, Cholesky {e}"
            );
        }
    }
}

#[test]
fn fbm_increments_have_the_fractional_gaussian_noise_autocorrelation() {
    // Pooled lag-k autocorrelation of 2000 independent paths of 64 increments;
    // the sampling error of each estimate is below 0.006.
    let dt = 0.01;
    let paths = 2000;
    let n = 64;
    for (h, seed) in [(0.8, 11_u64), (0.5, 12), (0.3, 13)] {
        let fbm = FractionalBrownianMotion::new(h, dt);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut lag_sums = [0.0_f64; 6];
        for _ in 0..paths {
            let x = fbm.sample(n, &mut rng);
            for (lag, sum) in lag_sums.iter_mut().enumerate() {
                *sum += x[..n - lag]
                    .iter()
                    .zip(&x[lag..])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
        }
        let gamma_hat = |lag: usize| lag_sums[lag] / (paths * (n - lag)) as f64;
        for lag in 1..=5 {
            let measured = gamma_hat(lag) / gamma_hat(0);
            let expected = fgn_autocovariance(h, dt, lag) / fgn_autocovariance(h, dt, 0);
            assert!(
                (measured - expected).abs() < 0.03,
                "H = {h}, lag {lag}: autocorrelation {measured}, expected {expected}"
            );
        }
        let variance_ratio = gamma_hat(0) / fgn_autocovariance(h, dt, 0);
        assert!(
            (variance_ratio - 1.0).abs() < 0.05,
            "H = {h}: increment variance off by a factor {variance_ratio}"
        );
    }
}

#[test]
fn fbm_partial_sums_follow_the_power_law_variance() {
    // Var[B(m dt)] = (m dt)^(2H) for fractional Brownian motion.
    let dt = 0.05;
    let paths = 4000;
    let n = 64;
    for (h, seed) in [(0.8, 21_u64), (0.3, 22)] {
        let fbm = FractionalBrownianMotion::new(h, dt);
        let mut rng = StdRng::seed_from_u64(seed);
        let checkpoints = [1_usize, 8, 32, 64];
        let mut second_moments = [0.0_f64; 4];
        for _ in 0..paths {
            let x = fbm.sample(n, &mut rng);
            for (moment, &m) in second_moments.iter_mut().zip(&checkpoints) {
                let b: f64 = x[..m].iter().sum();
                *moment += b * b;
            }
        }
        for (moment, &m) in second_moments.iter().zip(&checkpoints) {
            let measured = moment / f64::from(paths);
            let expected = (m as f64 * dt).powf(2.0 * h);
            assert!(
                (measured / expected - 1.0).abs() < 0.1,
                "H = {h}, m = {m}: Var[B] = {measured}, expected {expected}"
            );
        }
    }
}

#[test]
fn fbm_sample_handles_trivial_and_extreme_inputs() {
    let mut rng = StdRng::seed_from_u64(5);
    let fbm = FractionalBrownianMotion::new(0.7, 0.01);
    assert!(fbm.sample(0, &mut rng).is_empty());
    assert_eq!(fbm.sample(1, &mut rng).len(), 1);
    // Hurst exponents very close to 1 make the increments almost perfectly
    // correlated; the recursion must stay finite.
    let nearly_smooth = FractionalBrownianMotion::new(0.999, 0.01);
    let x = nearly_smooth.sample(512, &mut rng);
    assert_eq!(x.len(), 512);
    assert!(x.iter().all(|v| v.is_finite()));
    // H = 1/2 is white noise with variance dt: increments are independent.
    let white = FractionalBrownianMotion::new(0.5, 4.0);
    let mut state = Rng::new(31);
    let z: Vec<f64> = (0..16).map(|_| state.next_normal()).collect();
    for (value, normal) in white.correlate_normals(&z).iter().zip(&z) {
        assert!((value - 2.0 * normal).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// LangevinDynamics::step_baoab
// ---------------------------------------------------------------------------

#[test]
fn baoab_final_kick_uses_the_force_at_the_new_position() {
    // Without friction and noise (gamma = 0) BAOAB is velocity Verlet.  For the
    // harmonic force F = -x with m = 1 and dt = 0.1, from (x, v) = (1, 0):
    //   v_half = -0.05,  x_new = 1 + 2 * (-0.05) * 0.05 = 0.995,
    //   F(x_new) = -0.995,  v_new = -0.05 + (-0.995) * 0.05 = -0.09975.
    // Reusing the stale force F(1) = -1 would give v_new = -0.1.
    let verlet = LangevinDynamics::new(1.0, 0.0, 300.0);
    let mut rng = Rng::new(1);
    let (x, v, f) = verlet.step_baoab(1.0, 0.0, -1.0, |x| -x, 0.1, &mut rng);
    assert!((x - 0.995).abs() < 1e-15, "x = {x}");
    assert!((v + 0.09975).abs() < 1e-15, "v = {v}");
    assert!((f + 0.995).abs() < 1e-15, "returned force = {f}");
}

#[test]
fn baoab_evaluates_the_force_function_once_per_step_at_the_new_position() {
    let thermostat = LangevinDynamics::new(1.0, 1.0, 1.0 / K_B);
    let mut rng = Rng::new(11);
    let calls = std::cell::Cell::new(0_u32);
    let probe = std::cell::Cell::new(f64::NAN);
    let (x, _v, f) = thermostat.step_baoab(
        0.3,
        0.2,
        -0.3,
        |position| {
            calls.set(calls.get() + 1);
            probe.set(position);
            3.0 * position
        },
        0.05,
        &mut rng,
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(
        probe.get(),
        x,
        "the force is sampled at the returned position"
    );
    assert_eq!(f, 3.0 * x);
}

#[test]
fn baoab_samples_the_boltzmann_position_variance_of_a_harmonic_oscillator() {
    // m = 1, gamma = 1, k_B T = 1 and F = -x give <x^2> = k_B T / k = 1 for any
    // time step (the exact stationary variance of BAOAB on a harmonic well).
    // Evolving with the force from the previous position (the old behaviour)
    // inflates it to about 1.34 at dt = 0.5.
    let thermostat = LangevinDynamics::new(1.0, 1.0, 1.0 / K_B);
    let force = |x: f64| -x;
    let dt = 0.5;
    let mut rng = Rng::new(20_261_026);
    let (mut x, mut v) = (0.0_f64, 0.0_f64);
    let mut f = force(x);
    for _ in 0..2_000 {
        (x, v, f) = thermostat.step_baoab(x, v, f, force, dt, &mut rng);
    }
    let steps = 200_000_u32;
    let (mut sum_x2, mut sum_v2) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        (x, v, f) = thermostat.step_baoab(x, v, f, force, dt, &mut rng);
        sum_x2 += x * x;
        sum_v2 += v * v;
    }
    let mean_x2 = sum_x2 / f64::from(steps);
    let mean_v2 = sum_v2 / f64::from(steps);
    assert!(
        (mean_x2 - 1.0).abs() < 0.03,
        "<x^2> = {mean_x2}, Boltzmann value 1"
    );
    // The scheme's stationary kinetic energy is (1 - dt^2 / 4) k_B T / m.
    let predicted_v2 = 1.0 - 0.25 * dt * dt;
    assert!(
        (mean_v2 - predicted_v2).abs() < 0.03,
        "<v^2> = {mean_v2}, scheme value {predicted_v2}"
    );
}

// ---------------------------------------------------------------------------
// KleinmanKramers::step
// ---------------------------------------------------------------------------

#[test]
fn kleinman_kramers_kick_variance_is_two_gamma_kt_dt_over_mass_squared() {
    // From rest and without force v_new is the random kick alone: the friction
    // force -gamma v and its random counterpart sqrt(2 gamma kT) dW are both
    // divided by the mass, so Var[v_new] = 2 gamma kT dt / m^2.
    let (gamma, mass, k_t, dt) = (2.0, 4.0, 1.0, 0.01);
    let kk = KleinmanKramers::new(gamma, mass, k_t);
    let mut rng = StdRng::seed_from_u64(77);
    let samples = 100_000_u32;
    let mut sum_sq = 0.0_f64;
    for _ in 0..samples {
        let (_, v_new) = kk.step(0.0, 0.0, 0.0, dt, &mut rng);
        sum_sq += v_new * v_new;
    }
    let measured = sum_sq / f64::from(samples);
    let expected = 2.0 * gamma * k_t * dt / (mass * mass);
    assert!(
        (measured / expected - 1.0).abs() < 0.05,
        "Var[v_new] = {measured}, expected {expected} (the old mass-blind amplitude gives {})",
        4.0 * expected
    );
}

#[test]
fn kleinman_kramers_obeys_equipartition_for_a_heavy_particle() {
    // m = 4, gamma = 40, kT = 1: <v^2> = kT / m = 0.25.  The Euler-Maruyama
    // stationary variance is larger by 1 / (1 - gamma dt / (2 m)) = +0.5 %;
    // the old noise amplitude (off by a factor m) gave <v^2> = kT = 1.
    let (gamma, mass, k_t, dt) = (40.0, 4.0, 1.0, 1.0e-3);
    let kk = KleinmanKramers::new(gamma, mass, k_t);
    let mut rng = StdRng::seed_from_u64(2_026);
    let (mut x, mut v) = (0.0_f64, 0.0_f64);
    for _ in 0..20_000 {
        (x, v) = kk.step(x, v, 0.0, dt, &mut rng);
    }
    let steps = 2_000_000_u32;
    let mut sum_v2 = 0.0_f64;
    for _ in 0..steps {
        (x, v) = kk.step(x, v, 0.0, dt, &mut rng);
        sum_v2 += v * v;
    }
    let mean_v2 = sum_v2 / f64::from(steps);
    let equipartition = k_t / mass;
    assert!(
        (mean_v2 / equipartition - 1.0).abs() < 0.05,
        "<v^2> = {mean_v2}, equipartition value {equipartition}"
    );
}

// ---------------------------------------------------------------------------
// mean_first_passage_time (Kramers escape time)
// ---------------------------------------------------------------------------

#[test]
fn kramers_escape_time_matches_the_overdamped_formula() {
    // tau = 2 pi gamma / (omega_0 omega_b) * exp(dE / kT).
    let (barrier, k_t, gamma, omega_0, omega_b) = (2.0, 0.5, 3.0, 2.0, 1.5);
    let tau = mean_first_passage_time(barrier, k_t, gamma, omega_0, omega_b)
        .expect("valid Kramers parameters");
    let expected = 2.0 * PI * gamma / (omega_0 * omega_b) * (barrier / k_t).exp();
    assert!(
        (tau - expected).abs() <= 1e-12 * expected,
        "tau = {tau}, expected {expected}"
    );
    // Independent value: 2 pi e^4.
    assert!((tau - 2.0 * PI * 4.0_f64.exp()).abs() <= 1e-10 * tau);
}

#[test]
fn kramers_escape_time_scales_with_friction_and_thermal_energy() {
    let base = mean_first_passage_time(3.0, 1.0, 2.0, 1.0, 1.0).expect("valid");
    // Linear in the friction rate.
    let doubled_friction = mean_first_passage_time(3.0, 1.0, 4.0, 1.0, 1.0).expect("valid");
    assert!((doubled_friction / base - 2.0).abs() < 1e-12);
    // Only the ratio dE / kT enters the exponential.
    let scaled_energy = mean_first_passage_time(6.0, 2.0, 2.0, 1.0, 1.0).expect("valid");
    assert!((scaled_energy / base - 1.0).abs() < 1e-12);
    // A hotter bath escapes sooner.
    let hotter = mean_first_passage_time(3.0, 2.0, 2.0, 1.0, 1.0).expect("valid");
    assert!(hotter < base);
    // Inversely proportional to both curvature frequencies.
    let stiffer = mean_first_passage_time(3.0, 1.0, 2.0, 2.0, 4.0).expect("valid");
    assert!((base / stiffer - 8.0).abs() < 1e-12);
}

#[test]
fn kramers_escape_time_rejects_unphysical_parameters() {
    let invalid_field = |result: Result<f64, PhysicsError>| match result {
        Err(PhysicsError::InvalidInput { field, .. }) => field,
        other => panic!("expected InvalidInput, got {other:?}"),
    };
    assert_eq!(
        invalid_field(mean_first_passage_time(1.0, 0.0, 1.0, 1.0, 1.0)),
        "k_t"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(1.0, -1.0, 1.0, 1.0, 1.0)),
        "k_t"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(1.0, 1.0, 0.0, 1.0, 1.0)),
        "gamma"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(1.0, 1.0, 1.0, 0.0, 1.0)),
        "omega_0"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(1.0, 1.0, 1.0, 1.0, -2.0)),
        "omega_b"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(1.0, f64::NAN, 1.0, 1.0, 1.0)),
        "k_t"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(-0.5, 1.0, 1.0, 1.0, 1.0)),
        "barrier_height"
    );
    assert_eq!(
        invalid_field(mean_first_passage_time(f64::INFINITY, 1.0, 1.0, 1.0, 1.0)),
        "barrier_height"
    );
    // A vanishing barrier is the attempt-time limit, not an error.
    let tau = mean_first_passage_time(0.0, 1.0, 1.0, 1.0, 1.0).expect("zero barrier is valid");
    assert!((tau - 2.0 * PI).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Seeding of the linear congruential generators
// ---------------------------------------------------------------------------

/// Largest `|r|` over the first eight draw positions of the Pearson correlation,
/// across seeds, between the draw of seed `s` and the draw of seed `s + 1`.
///
/// For a LCG used raw, the stream of seed `s + 1` is the stream of seed `s` plus
/// `a^k (mod 2^64)` at step `k`, which makes these correlations large (the
/// measured values were -0.36, -0.45, 0.75 and 0.89 at steps 1 to 4).
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
fn consecutive_seeds_of_the_stochastic_rng_give_uncorrelated_streams() {
    let worst = max_consecutive_seed_correlation(|seed| {
        let mut rng = Rng::new(seed);
        std::array::from_fn(|_| rng.next_f64())
    });
    // The standard error is 1 / sqrt(20000) = 0.007.
    assert!(worst < 0.05, "max |r| = {worst}");
}

#[test]
fn consecutive_seeds_of_the_statistics_rng_give_uncorrelated_streams() {
    let worst = max_consecutive_seed_correlation(|seed| {
        let mut rng = crate::StatRng::new(seed);
        std::array::from_fn(|_| rng.next_f64())
    });
    assert!(worst < 0.05, "max |r| = {worst}");
}

#[test]
fn consecutive_seeds_of_the_sampling_lcg_give_uncorrelated_streams() {
    let worst = max_consecutive_seed_correlation(|seed| {
        let mut rng = crate::sampling::Lcg::new(seed);
        std::array::from_fn(|_| rng.next_f64())
    });
    assert!(worst < 0.05, "max |r| = {worst}");
}

#[test]
fn monte_carlo_gbm_paths_with_consecutive_seeds_are_independent() {
    // `monte_carlo_gbm_paths` seeds path i with seed + i.  The terminal values of
    // paths i and i + 1 must be uncorrelated; with raw LCG seeding they followed
    // each other closely.
    use super::types::GeometricBrownianMotion;
    let gbm = GeometricBrownianMotion::new(0.05, 0.2);
    let paths = 4_000_u64;
    let steps = 4;
    let terminal: Vec<f64> = (0..=paths)
        .map(|i| {
            let path = gbm.simulate_exact(100.0, 1.0, 1.0 / steps as f64, 777 + i);
            path[path.len() - 1].ln()
        })
        .collect();
    let count = paths as usize;
    let r = crate::statistics::correlation(&terminal[..count], &terminal[1..]);
    // The standard error is 1 / sqrt(4000) = 0.016.
    assert!(r.abs() < 0.06, "correlation of neighbouring seeds = {r}");
}

#[test]
fn rng_streams_of_distinct_seeds_differ_and_are_reproducible() {
    let draw = |seed: u64| -> Vec<u64> {
        let mut rng = Rng::new(seed);
        (0..4).map(|_| rng.next_u64()).collect()
    };
    assert_eq!(draw(0), draw(0));
    assert_ne!(draw(0), draw(1));
    assert_ne!(draw(1), draw(2));
    // Seed 0 is an ordinary seed, not a degenerate all-zero state.
    assert!(draw(0).iter().all(|&v| v != 0));
}

// ---------------------------------------------------------------------------
// Rng::next_poisson
// ---------------------------------------------------------------------------

#[test]
fn next_poisson_with_a_large_mean_is_not_capped_near_745() {
    // Knuth's loop stopped once the running product underflowed, returning about
    // 745 for every mean above that.
    let mut rng = Rng::new(31);
    let n = 40_000;
    let draws: Vec<f64> = (0..n).map(|_| rng.next_poisson(1000.0) as f64).collect();
    let mean = draws.iter().sum::<f64>() / f64::from(n);
    let variance = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f64::from(n - 1);
    assert!((mean - 1000.0).abs() < 10.0, "mean = {mean}");
    assert!((variance - 1000.0).abs() < 50.0, "variance = {variance}");
}

#[test]
fn next_poisson_degenerate_means_return_zero() {
    let mut rng = Rng::new(2);
    assert_eq!(rng.next_poisson(0.0), 0);
    assert_eq!(rng.next_poisson(-4.0), 0);
    assert_eq!(rng.next_poisson(f64::NAN), 0);
    assert_eq!(rng.next_poisson(f64::INFINITY), 0);
    // A small mean still follows the Poisson law: P(0) = exp(-0.5).
    let n = 20_000;
    let zeros = (0..n).filter(|_| rng.next_poisson(0.5) == 0).count();
    let fraction = zeros as f64 / f64::from(n);
    assert!(
        (fraction - (-0.5_f64).exp()).abs() < 0.02,
        "P(0) = {fraction}"
    );
}
