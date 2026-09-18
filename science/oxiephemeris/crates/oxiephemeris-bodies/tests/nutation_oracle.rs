//! Oracle tests: the library's FULL and truncated IAU `2000A_R06`
//! nutation series against a runtime evaluation of the IERS tables.
//!
//! Parses the complete series from the IERS Conventions (2010) electronic
//! tables `data/iers/tab5.3a.txt` (nutation in longitude, 1320 + 38 terms)
//! and `data/iers/tab5.3b.txt` (nutation in obliquity, 1037 + 19 terms),
//! evaluates it independently here (series form: TN36 eq. 5.35; arguments:
//! TN36 eqs. 5.43/5.44), and requires
//!
//! - the generated tables behind `nutation_iau2000a` to carry exactly the
//!   term counts of the IERS files (1320 + 38 / 1037 + 19),
//! - `|nutation_iau2000a - runtime tables| < 1e-9 arcsec` in both
//!   components at 401 epochs over 1800-2200 (same data, so only
//!   summation-order float noise remains),
//! - `|dpsi_trunc - dpsi_A| < 1.3 mas` and `|deps_trunc - deps_A| < 1.3
//!   mas` at monthly samples over 1995-2050, and `< 3 mas` over
//!   1900-2100 (IAU 2000B accuracy class, McCarthy & Luzum 2003), and
//! - `|full - truncated| < 2 mas` over 1995-2050 (documents the
//!   truncation error; measured max ~0.64 mas in `dpsi`, ~0.35 mas in
//!   `deps`),
//!
//! printing the observed maxima.
//!
//! The data files are NOT committed (see `scripts/fetch_de440.sh`); when
//! absent the data-dependent tests print a note and pass vacuously.

use std::path::{Path, PathBuf};

use oxiephemeris_bodies::frames::{
    nutation_iau2000a, nutation_iau2000a_truncated, NUTATION_IAU2000A_TERM_COUNTS,
};

const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0);
const UAS_TO_RAD: f64 = ARCSEC_TO_RAD * 1.0e-6;
const TWO_PI: f64 = std::f64::consts::TAU;

/// One row of an IERS nutation table: sin/cos coefficients (µas or µas/cy)
/// and the 14 fundamental-argument multipliers.
struct RawTerm {
    coeff_sin: f64,
    coeff_cos: f64,
    mult: [i32; 14],
}

/// A parsed table: the `j = 0` (constant) and `j = 1` (t-linear) blocks.
struct Series {
    j0: Vec<RawTerm>,
    j1: Vec<RawTerm>,
}

/// Parses `tab5.3a.txt` / `tab5.3b.txt`. In BOTH files the first numeric
/// column after the index multiplies `sin(ARG)` and the second multiplies
/// `cos(ARG)` (tab5.3a: `A_i`, `A''_i`; tab5.3b: `B''_i`, `B_i` — note the
/// column order printed in the tab5.3b header).
fn parse_table(path: &Path) -> Result<Series, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut j0 = Vec::new();
    let mut j1 = Vec::new();
    let mut block = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("j = 0") {
            block = Some(0);
            continue;
        }
        if trimmed.starts_with("j = 1") {
            block = Some(1);
            continue;
        }
        let Some(which) = block else { continue };
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() != 17 {
            continue; // header/separator lines
        }
        let Ok(_index) = fields[0].parse::<u32>() else {
            continue;
        };
        let coeff_sin: f64 = fields[1]
            .parse()
            .map_err(|e| format!("bad sin coeff {:?}: {e}", fields[1]))?;
        let coeff_cos: f64 = fields[2]
            .parse()
            .map_err(|e| format!("bad cos coeff {:?}: {e}", fields[2]))?;
        let mut mult = [0_i32; 14];
        for (slot, raw) in mult.iter_mut().zip(&fields[3..17]) {
            *slot = raw
                .parse()
                .map_err(|e| format!("bad multiplier {raw:?}: {e}"))?;
        }
        let term = RawTerm {
            coeff_sin,
            coeff_cos,
            mult,
        };
        if which == 0 {
            j0.push(term);
        } else {
            j1.push(term);
        }
    }
    if j0.is_empty() || j1.is_empty() {
        return Err(format!(
            "{}: parsed {} j=0 / {} j=1 terms — format change?",
            path.display(),
            j0.len(),
            j1.len()
        ));
    }
    Ok(Series { j0, j1 })
}

/// Independent copy of the fundamental arguments, TN36 eqs. (5.43)/(5.44),
/// in radians (Delaunay part reduced modulo one turn).
fn fundamental_args(t: f64) -> [f64; 14] {
    const TURN_ARCSEC: f64 = 1_296_000.0;
    let el = 485_868.249_036
        + t * (1_717_915_923.217_8 + t * (31.8792 + t * (0.051_635 + t * (-0.000_244_70))));
    let elp = 1_287_104.793_048
        + t * (129_596_581.048_1 + t * (-0.5532 + t * (0.000_136 + t * (-0.000_011_49))));
    let ef = 335_779.526_232
        + t * (1_739_527_262.847_8 + t * (-12.7512 + t * (-0.001_037 + t * 0.000_004_17)));
    let ed = 1_072_260.703_692
        + t * (1_602_961_601.209 + t * (-6.3706 + t * (0.006_593 + t * (-0.000_031_69))));
    let om = 450_160.398_036
        + t * (-6_962_890.543_1 + t * (7.4722 + t * (0.007_702 + t * (-0.000_059_39))));
    [
        (el % TURN_ARCSEC) * ARCSEC_TO_RAD,
        (elp % TURN_ARCSEC) * ARCSEC_TO_RAD,
        (ef % TURN_ARCSEC) * ARCSEC_TO_RAD,
        (ed % TURN_ARCSEC) * ARCSEC_TO_RAD,
        (om % TURN_ARCSEC) * ARCSEC_TO_RAD,
        4.402_608_842 + 2_608.790_314_157_4 * t,
        3.176_146_697 + 1_021.328_554_621_1 * t,
        1.753_470_314 + 628.307_584_999_1 * t,
        6.203_480_913 + 334.061_242_67 * t,
        0.599_546_497 + 52.969_096_264_1 * t,
        0.874_016_757 + 21.329_910_496 * t,
        5.481_293_872 + 7.478_159_856_7 * t,
        5.311_886_287 + 3.813_303_563_8 * t,
        (0.024_381_75 + 0.000_005_386_91 * t) * t,
    ]
}

/// Evaluates one full series (TN36 eq. 5.35): result in µas. Works for
/// both longitude and obliquity because in both parsed tables `coeff_sin`
/// multiplies `sin(ARG)` and `coeff_cos` multiplies `cos(ARG)`.
fn evaluate_full(series: &Series, t: f64) -> f64 {
    let fa = fundamental_args(t);
    let mut total = 0.0;
    for (terms, factor) in [(&series.j0, 1.0), (&series.j1, t)] {
        for term in terms.iter().rev() {
            let mut arg = 0.0;
            for (n, a) in term.mult.iter().zip(fa.iter()) {
                if *n != 0 {
                    arg += f64::from(*n) * a;
                }
            }
            let arg = arg % TWO_PI;
            total += factor * (term.coeff_sin * arg.sin() + term.coeff_cos * arg.cos());
        }
    }
    total
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/iers")
}

/// Scans `[t0, t1]` monthly; returns max |B - A| in µas for (dpsi, deps).
fn max_deviation(psi: &Series, eps: &Series, t0: f64, t1: f64) -> (f64, f64) {
    let mut worst_psi = 0.0_f64;
    let mut worst_eps = 0.0_f64;
    let mut k = 0_i32;
    loop {
        let t = t0 + f64::from(k) / 1200.0; // monthly steps in centuries
        if t > t1 + 1e-12 {
            break;
        }
        let full_psi = evaluate_full(psi, t) * UAS_TO_RAD;
        let full_eps = evaluate_full(eps, t) * UAS_TO_RAD;
        let ours = nutation_iau2000a_truncated(t);
        worst_psi = worst_psi.max((ours.dpsi_rad - full_psi).abs());
        worst_eps = worst_eps.max((ours.deps_rad - full_eps).abs());
        k += 1;
    }
    (
        worst_psi / (1e-3 * ARCSEC_TO_RAD),
        worst_eps / (1e-3 * ARCSEC_TO_RAD),
    )
}

#[test]
fn nutation_matches_full_iau2000a_series() -> Result<(), String> {
    let psi_path = data_dir().join("tab5.3a.txt");
    let eps_path = data_dir().join("tab5.3b.txt");
    if !psi_path.is_file() || !eps_path.is_file() {
        println!(
            "SKIP: IERS tables not found under {} — run scripts/fetch_de440.sh to download them",
            data_dir().display()
        );
        return Ok(());
    }
    let psi = parse_table(&psi_path)?;
    let eps = parse_table(&eps_path)?;
    println!(
        "parsed tab5.3a: {} + {} terms; tab5.3b: {} + {} terms",
        psi.j0.len(),
        psi.j1.len(),
        eps.j0.len(),
        eps.j1.len()
    );
    // Expected sizes per the TN36 file headers.
    if psi.j0.len() != 1320 || psi.j1.len() != 38 {
        return Err(format!(
            "tab5.3a term counts changed: {} / {}",
            psi.j0.len(),
            psi.j1.len()
        ));
    }
    if eps.j0.len() != 1037 || eps.j1.len() != 19 {
        return Err(format!(
            "tab5.3b term counts changed: {} / {}",
            eps.j0.len(),
            eps.j1.len()
        ));
    }

    let (psi_narrow, eps_narrow) = max_deviation(&psi, &eps, -0.05, 0.50);
    println!(
        "max |truncated - 2000A_R06| over 1995-2050: dpsi = {psi_narrow:.4} mas, deps = {eps_narrow:.4} mas"
    );
    assert!(
        psi_narrow < 1.3,
        "dpsi deviation {psi_narrow} mas >= 1.3 mas (1995-2050)"
    );
    assert!(
        eps_narrow < 1.3,
        "deps deviation {eps_narrow} mas >= 1.3 mas (1995-2050)"
    );

    let (psi_wide, eps_wide) = max_deviation(&psi, &eps, -1.0, 1.0);
    println!(
        "max |truncated - 2000A_R06| over 1900-2100: dpsi = {psi_wide:.4} mas, deps = {eps_wide:.4} mas"
    );
    assert!(
        psi_wide < 3.0,
        "dpsi deviation {psi_wide} mas >= 3 mas (1900-2100)"
    );
    assert!(
        eps_wide < 3.0,
        "deps deviation {eps_wide} mas >= 3 mas (1900-2100)"
    );
    Ok(())
}

/// The generated full-series tables carry exactly the term counts of the
/// IERS files: tab5.3a `j = 0` / `j = 1` = 1320 / 38, tab5.3b = 1037 / 19.
/// (Runs without the data files; `full_series_matches_runtime_tables`
/// re-checks the same numbers against a fresh parse.)
#[test]
fn full_series_term_counts_match_iers_headers() {
    assert_eq!(NUTATION_IAU2000A_TERM_COUNTS, [1320, 38, 1037, 19]);
}

/// The committed generated tables evaluate identically (up to
/// summation-order float noise, bound 1e-9 arcsec) to a runtime parse and
/// evaluation of the same IERS text tables, at 401 epochs over 1800-2200.
#[test]
fn full_series_matches_runtime_tables() -> Result<(), String> {
    let psi_path = data_dir().join("tab5.3a.txt");
    let eps_path = data_dir().join("tab5.3b.txt");
    if !psi_path.is_file() || !eps_path.is_file() {
        println!(
            "SKIP: IERS tables not found under {} — run scripts/fetch_de440.sh to download them",
            data_dir().display()
        );
        return Ok(());
    }
    let psi = parse_table(&psi_path)?;
    let eps = parse_table(&eps_path)?;
    assert_eq!(
        NUTATION_IAU2000A_TERM_COUNTS,
        [psi.j0.len(), psi.j1.len(), eps.j0.len(), eps.j1.len()],
        "generated tables and parsed IERS tables disagree on term counts"
    );

    // 401 epochs, 1800-2200: t = -2.00, -1.99, ..., +2.00 centuries.
    let mut worst_psi = 0.0_f64;
    let mut worst_eps = 0.0_f64;
    for k in 0..=400_i32 {
        let t = -2.0 + f64::from(k) / 100.0;
        let full_psi = evaluate_full(&psi, t) * UAS_TO_RAD;
        let full_eps = evaluate_full(&eps, t) * UAS_TO_RAD;
        let ours = nutation_iau2000a(t);
        worst_psi = worst_psi.max((ours.dpsi_rad - full_psi).abs());
        worst_eps = worst_eps.max((ours.deps_rad - full_eps).abs());
    }
    let worst_psi_as = worst_psi / ARCSEC_TO_RAD;
    let worst_eps_as = worst_eps / ARCSEC_TO_RAD;
    println!(
        "max |full lib - runtime tables| over 1800-2200: dpsi = {worst_psi_as:.3e} as, deps = {worst_eps_as:.3e} as"
    );
    assert!(
        worst_psi_as < 1e-9,
        "dpsi deviation {worst_psi_as} as >= 1e-9 as (1800-2200)"
    );
    assert!(
        worst_eps_as < 1e-9,
        "deps deviation {worst_eps_as} as >= 1e-9 as (1800-2200)"
    );
    Ok(())
}

/// Truncation error of `nutation_iau2000a_truncated` against the full
/// in-library series over 1995-2050 (monthly sampling): < 2 mas in both
/// components. Measured: ~0.64 mas in `dpsi`, ~0.35 mas in `deps` (the
/// exact maxima are printed). Runs without the IERS data files.
#[test]
fn full_vs_truncated_below_two_mas() {
    let mut worst_psi = 0.0_f64;
    let mut worst_eps = 0.0_f64;
    let mut k = 0_i32;
    loop {
        let t = -0.05 + f64::from(k) / 1200.0; // monthly steps in centuries
        if t > 0.50 + 1e-12 {
            break;
        }
        let full = nutation_iau2000a(t);
        let trunc = nutation_iau2000a_truncated(t);
        worst_psi = worst_psi.max((trunc.dpsi_rad - full.dpsi_rad).abs());
        worst_eps = worst_eps.max((trunc.deps_rad - full.deps_rad).abs());
        k += 1;
    }
    let worst_psi_mas = worst_psi / (1e-3 * ARCSEC_TO_RAD);
    let worst_eps_mas = worst_eps / (1e-3 * ARCSEC_TO_RAD);
    println!(
        "max |truncated - full| over 1995-2050: dpsi = {worst_psi_mas:.4} mas, deps = {worst_eps_mas:.4} mas"
    );
    assert!(
        worst_psi_mas < 2.0,
        "dpsi truncation error {worst_psi_mas} mas >= 2 mas (1995-2050)"
    );
    assert!(
        worst_eps_mas < 2.0,
        "deps truncation error {worst_eps_mas} mas >= 2 mas (1995-2050)"
    );
}
