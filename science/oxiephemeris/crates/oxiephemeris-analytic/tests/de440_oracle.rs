//! End-to-end oracle: the truncated analytic theories against JPL
//! DE440 (ICRS). This measures the *whole* error budget at once —
//! series truncation + the theories' own DE200-era fit + the
//! dynamical-ecliptic→FK5(≈ICRS) frame tie.
//!
//! Two families of checks:
//!
//! * **Barycentric** states carry a ~9.7e-6 au common-mode offset: the
//!   solar-system barycenter itself moved between DE200 (no TNOs, few
//!   asteroids) and DE440. Every VSOP87E body shows the same floor —
//!   the Sun..Mars maxima below are all ≈ 9.7e-6 au.
//! * **Geocentric differences** (body − Earth) cancel that common mode
//!   and are what a fallback ephemeris is consumed as; the inner
//!   planets land at the few-e-6-au (sub-arcsecond) level.
//!
//! Requires `data/de440/linux_p1550p2650.440`; skips cleanly when it is
//! absent. One test function so the DE440 binary is read once.

use std::path::PathBuf;

use oxiephemeris_analytic::{geocentric_moon_km, state_au, VsopBody};
use oxiephemeris_de::{Body, DeFile, Series};

fn de440() -> Option<Vec<u8>> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(e) => panic!("read failed: {e}"),
    }
}

fn norm3(v: &[f64]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The nine `(analytic, DE)` body pairs.
const PAIRS: [(VsopBody, Body); 9] = [
    (VsopBody::Sun, Body::Sun),
    (VsopBody::Mercury, Body::Mercury),
    (VsopBody::Venus, Body::Venus),
    (VsopBody::Earth, Body::Earth),
    (VsopBody::Mars, Body::Mars),
    (VsopBody::Jupiter, Body::Jupiter),
    (VsopBody::Saturn, Body::Saturn),
    (VsopBody::Uranus, Body::Uranus),
    (VsopBody::Neptune, Body::Neptune),
];

/// Epochs: `count` samples from `start` (TDB JD), even stride to `stop`.
fn epochs(start: f64, stop: f64, count: usize) -> impl Iterator<Item = (f64, f64)> {
    #[allow(clippy::cast_precision_loss)] // small counts
    let stride = (stop - start) / (count as f64 - 1.0);
    #[allow(clippy::cast_precision_loss)]
    (0..count).map(move |k| (start + stride * k as f64, 0.0))
}

/// Worst |Δposition| (au) and |Δvelocity| (au/day) of one barycentric
/// pair over the epoch set.
fn measure_pair(
    de: &DeFile<'_>,
    au_km: f64,
    pair: (VsopBody, Body),
    epoch_set: impl Iterator<Item = (f64, f64)>,
) -> (f64, f64) {
    let mut worst = (0.0f64, 0.0f64);
    for jd in epoch_set {
        let ours = state_au(pair.0, jd);
        let theirs = match de.state_km(pair.1, jd) {
            Ok(s) => s,
            Err(e) => panic!("DE440 {:?} at {jd:?}: {e}", pair.1),
        };
        let mut dp = [0.0; 3];
        let mut dv = [0.0; 3];
        for i in 0..3 {
            dp[i] = ours[i] - theirs[i] / au_km;
            dv[i] = ours[i + 3] - theirs[i + 3] / au_km;
        }
        worst.0 = worst.0.max(norm3(&dp));
        worst.1 = worst.1.max(norm3(&dv));
    }
    worst
}

/// Worst |Δ(body − Earth)| position (au) of one pair over the epoch
/// set — the common-mode barycenter offset cancels here.
fn measure_geocentric(
    de: &DeFile<'_>,
    au_km: f64,
    pair: (VsopBody, Body),
    epoch_set: impl Iterator<Item = (f64, f64)>,
) -> f64 {
    let mut worst = 0.0f64;
    for jd in epoch_set {
        let ours = state_au(pair.0, jd);
        let ours_earth = state_au(VsopBody::Earth, jd);
        let theirs = match de.state_km(pair.1, jd) {
            Ok(s) => s,
            Err(e) => panic!("DE440 {:?} at {jd:?}: {e}", pair.1),
        };
        let theirs_earth = match de.state_km(Body::Earth, jd) {
            Ok(s) => s,
            Err(e) => panic!("DE440 Earth at {jd:?}: {e}"),
        };
        let mut dp = [0.0; 3];
        for i in 0..3 {
            dp[i] = (ours[i] - ours_earth[i]) - (theirs[i] - theirs_earth[i]) / au_km;
        }
        worst = worst.max(norm3(&dp));
    }
    worst
}

/// Worst geocentric-Moon |Δr| (km) and |Δv| (km/day) over the epoch set.
fn measure_moon(de: &DeFile<'_>, epoch_set: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    let mut worst = (0.0f64, 0.0f64);
    for jd in epoch_set {
        let ours = geocentric_moon_km(jd);
        let theirs = match de.series_state(Series::Moon, jd) {
            Ok(s) => s,
            Err(e) => panic!("DE440 Moon at {jd:?}: {e}"),
        };
        let dp = [
            ours[0] - theirs.value[0],
            ours[1] - theirs.value[1],
            ours[2] - theirs.value[2],
        ];
        let dv = [
            ours[3] - theirs.rate[0],
            ours[4] - theirs.rate[1],
            ours[5] - theirs.rate[2],
        ];
        worst.0 = worst.0.max(norm3(&dp));
        worst.1 = worst.1.max(norm3(&dv));
    }
    worst
}

/// 1900-01-01 and 2100-01-01 TDB.
const JD_1900: f64 = 2_415_020.5;
const JD_2100: f64 = 2_488_069.5;

#[test]
fn analytic_theories_against_de440() {
    let Some(bytes) = de440() else { return };
    let de = match DeFile::parse(&bytes) {
        Ok(d) => d,
        Err(e) => panic!("parse failed: {e}"),
    };
    let Some(au_km) = de.constant("AU") else {
        panic!("DE440 header lacks AU")
    };

    // Barycentric, 1900–2100 (84 epochs). Gates ≈ 2× the measured
    // maxima (printed): Sun..Mars all sit on the ~9.7e-6 au common
    // barycenter offset; the giants add their DE200-era orbit drift
    // (Uranus and Neptune were pre-Voyager orbits in DE200).
    let gates: [(f64, f64); 9] = [
        (2e-5, 3e-9),   // Sun      (measured 9.67e-6, 1.30e-9)
        (2e-5, 2e-8),   // Mercury  (measured 9.67e-6, 5.65e-9)
        (2e-5, 1e-8),   // Venus    (measured 9.66e-6, 4.02e-9)
        (2e-5, 2e-8),   // Earth    (measured 9.69e-6, 5.39e-9)
        (2e-5, 2e-8),   // Mars     (measured 9.74e-6, 9.99e-9)
        (4e-5, 3e-8),   // Jupiter  (measured 1.58e-5, 1.27e-8)
        (4e-5, 2e-8),   // Saturn   (measured 1.94e-5, 8.35e-9)
        (2.5e-4, 1e-7), // Uranus   (measured 1.25e-4, 4.53e-8)
        (7e-4, 1e-7),   // Neptune  (measured 3.31e-4, 4.11e-8)
    ];
    for (pair, (pos_gate, vel_gate)) in PAIRS.iter().zip(&gates) {
        let (dp, dv) = measure_pair(&de, au_km, *pair, epochs(JD_1900, JD_2100, 84));
        println!(
            "{:?}: barycentric max |Δr| {dp:.3e} au, |Δv| {dv:.3e} au/day (1900–2100)",
            pair.0
        );
        assert!(dp < *pos_gate, "{:?} position {dp} au", pair.0);
        assert!(dv < *vel_gate, "{:?} velocity {dv} au/day", pair.0);
    }

    // Geocentric differences, 1900–2100: the barycenter offset cancels.
    // Gates ≈ 2× measured (Earth − Earth is identically zero). In arc
    // terms: apparent Sun good to ~0.04″, Mercury/Venus ~0.1″ at
    // closest, Neptune (a pre-Voyager DE200 orbit) ~2.3″.
    let geo_gates: [f64; 9] = [
        4e-7,   // Sun      (measured 1.91e-7)
        4e-7,   // Mercury  (measured 1.98e-7)
        5e-7,   // Venus    (measured 2.20e-7)
        1e-12,  // Earth − Earth ≡ 0
        2e-6,   // Mars     (measured 8.07e-7)
        2e-5,   // Jupiter  (measured 9.24e-6)
        3e-5,   // Saturn   (measured 1.43e-5)
        2.5e-4, // Uranus   (measured 1.29e-4)
        7e-4,   // Neptune  (measured 3.32e-4)
    ];
    for (pair, gate) in PAIRS.iter().zip(&geo_gates) {
        let dp = measure_geocentric(&de, au_km, *pair, epochs(JD_1900, JD_2100, 84));
        println!("{:?}: geocentric max |Δr| {dp:.3e} au (1900–2100)", pair.0);
        assert!(dp < *gate, "{:?} geocentric {dp} au", pair.0);
    }

    // ELP2000-82B geocentric Moon, 1900–2100: dominated by the secular
    // tidal-acceleration difference between the DE200-era fit and
    // DE440 (~1″/cy² in mean longitude, i.e. km-level at the ends).
    let (dp, dv) = measure_moon(&de, epochs(JD_1900, JD_2100, 84));
    println!("Moon: max |Δr| {dp:.3e} km, |Δv| {dv:.3e} km/day (1900–2100)");
    assert!(dp < 6.0, "Moon position {dp} km (measured 2.67)");
    assert!(dv < 1.5, "Moon velocity {dv} km/day (measured 0.64)");

    // Long-span sanity, 1700–2200 (Moon) and 1650–2400 (planets,
    // geocentric): the secular drifts grow but stay bounded
    // (measured: Moon 14.6 km / 3.45 km/day; planets ≤ 3.7× their
    // 1900–2100 maxima, all far inside the 10× allowance).
    let (dp, dv) = measure_moon(&de, epochs(2_341_972.5, 2_524_593.5, 41));
    println!("Moon: max |Δr| {dp:.3e} km, |Δv| {dv:.3e} km/day (1700–2200)");
    assert!(dp < 40.0, "long-span Moon position {dp} km");
    assert!(dv < 8.0, "long-span Moon velocity {dv} km/day");
    for (pair, gate) in PAIRS.iter().zip(&geo_gates) {
        let dp = measure_geocentric(&de, au_km, *pair, epochs(2_323_708.5, 2_597_641.5, 41));
        println!("{:?}: geocentric max |Δr| {dp:.3e} au (1650–2400)", pair.0);
        assert!(
            dp < 10.0 * gate,
            "{:?} long-span geocentric {dp} au",
            pair.0
        );
    }
}
