//! `delta_t_seconds` (Espenak–Meeus ΔT = TT − UT1) tests:
//!
//! 1. Transcription exactness: >= 10 checkpoints, one per piece, each
//!    re-derived independently below (arithmetic spelled out in comments)
//!    from the published polynomials and asserted to 1e-9 s.
//! 2. Sanity vs. the canon's own tabulated historical ΔT values (its
//!    Table 1/2 central values and stated standard errors), with tolerance
//!    matching those stated errors.
//! 3. Boundary continuity: every piecewise join has `|jump| < 1.0 s`
//!    (the source's own fit is a smoothed approximation, not required to be
//!    C0-continuous at the seams — this only guards against a gross
//!    transcription blunder, e.g. a misplaced decimal point or a swapped
//!    sign, which would produce a jump far larger than the sub-second seams
//!    the real polynomials have).
//!
//! Source: F. Espenak & J. Meeus, "Five Millennium Canon of Solar
//! Eclipses: −1999 to +3000", NASA TP-2009-214174, and the companion
//! "Polynomial Expressions for Delta T" page
//! (<https://eclipse.gsfc.nasa.gov/SEhelp/deltatpoly2004.html>).

use oxiephemeris_core::time::delta_t_seconds;

/// One transcription-exactness checkpoint: `(decimal_year, expected_seconds)`.
///
/// Each `expected` value below is independently re-derived from the
/// published polynomial for the piece that owns `decimal_year` (half-open
/// intervals, lower bound inclusive) — the arithmetic is spelled out per
/// entry so a bug shared between this file and `delta_t.rs` cannot cancel
/// out.
///
/// ```text
/// year = -500 falls in the [-500, 500) piece, u = y/100 = -5:
///   10583.6 - 1014.41*(-5) + 33.78311*25 - 5.952053*(-125)
///     - 0.1798452*625 + 0.022174192*(-3125) + 0.0090316521*15625
///   = 10583.6 + 5072.05 + 844.57775 + 744.006625 - 112.40325
///     - 69.29435 + 141.1195640625 = 17203.6563390625
///
/// year = 0: same piece, u = 0, only the constant term survives: 10583.6
///
/// year = 500 falls in the [500, 1600) piece, u = (y-1000)/100 = -5:
///   1574.2 - 556.01*(-5) + 71.23472*25 + 0.319781*(-125)
///     - 0.8503463*625 - 0.005050998*(-3125) + 0.0083572073*15625
///   = 1574.2 + 2780.05 + 1780.868 - 39.972625 - 531.4664375
///     + 15.7843688 + 130.5813641 = 5710.0446703125 (f64 rounding kept
///   bit-identical to the assertion below)
///
/// year = 1000: same piece, u = 0, constant term: 1574.2
/// year = 1600 falls in the [1600, 1700) piece, t = 0: 120
/// year = 1800 falls in the [1800, 1860) piece, t = 0: 13.72
/// year = 1900 falls in the [1900, 1920) piece, t = 0: -2.79
/// year = 1950 falls in the [1941, 1961) piece, t = y-1950 = 0: 29.07
/// year = 1975 falls in the [1961, 1986) piece, t = y-1975 = 0: 45.45
/// year = 2000 falls in the [1986, 2005) piece, t = y-2000 = 0: 63.86
///
/// year = 2050 falls in the [2050, 2150) piece:
///   u = (2050-1820)/100 = 2.3, u^2 = 5.29, 32*u^2 = 169.28
///   -20 + 169.28 - 0.5628*(2150-2050) = 149.28 - 56.28 = 93.0
/// ```
const TRANSCRIPTION_CHECKPOINTS: &[(f64, f64)] = &[
    (-500.0, 17_203.656_339_062_5),
    (0.0, 10_583.6),
    (500.0, 5_710.044_670_312_5),
    (1000.0, 1_574.2),
    (1600.0, 120.0),
    (1800.0, 13.72),
    (1900.0, -2.79),
    (1950.0, 29.07),
    (1975.0, 45.45),
    (2000.0, 63.86),
    (2050.0, 93.0),
];

#[test]
fn transcription_matches_published_polynomials_to_1e_minus_9_s() {
    for &(year, expected) in TRANSCRIPTION_CHECKPOINTS {
        let got = delta_t_seconds(year);
        assert!(
            (got - expected).abs() < 1e-9,
            "delta_t_seconds({year}) = {got:.12} != expected {expected:.12} \
             (|diff| = {:.3e})",
            (got - expected).abs()
        );
    }
}

/// The canon's own tabulated historical ΔT central values and standard
/// errors (Espenak & Meeus Table 1 for pre-telescopic/telescopic-era
/// entries, Table 2 for the 20th century): `(decimal_year, tabulated_dt,
/// stated_standard_error)`. `2050` is deliberately excluded here: it is not
/// a historical *tabulated* observation but the canon's own extrapolated
/// estimate, which the transcription-exactness table above already pins to
/// 1e-9 s.
const CANON_TABULATED: &[(f64, f64, f64)] = &[
    (-500.0, 17190.0, 430.0),
    (0.0, 10580.0, 260.0),
    (500.0, 5710.0, 140.0),
    (1000.0, 1570.0, 55.0),
    (1600.0, 120.0, 20.0),
    (1800.0, 14.0, 1.0),
    (1900.0, -3.0, 1.0),
    (1950.0, 29.0, 0.5),
    (1975.0, 45.5, 0.5),
    (2000.0, 63.8, 0.5),
];

#[test]
fn matches_canon_tabulated_values_within_stated_standard_error() {
    for &(year, tabulated, sigma) in CANON_TABULATED {
        let got = delta_t_seconds(year);
        let diff = (got - tabulated).abs();
        assert!(
            diff <= sigma,
            "delta_t_seconds({year}) = {got:.3} vs canon tabulated {tabulated:.3} \
             +/- {sigma:.3}: |diff| = {diff:.3} exceeds the stated standard error"
        );
    }
}

/// Piecewise-join boundaries, in ascending order (mirrors the dispatch in
/// `delta_t_seconds`).
const BOUNDARIES: &[f64] = &[
    -500.0, 500.0, 1600.0, 1700.0, 1800.0, 1860.0, 1900.0, 1920.0, 1941.0, 1961.0, 1986.0, 2005.0,
    2050.0, 2150.0,
];

/// Half-year-second offset (in years) used to sample "just below" a
/// boundary without leaving enough room for a piece's own curvature to
/// contaminate the jump measurement (each piece's derivative is of order
/// 1-30 s/century, so this epsilon contributes well under 1e-6 s of its
/// own, three orders of magnitude below the 1.0 s gate).
const EPSILON_YEARS: f64 = 1e-7;

#[test]
fn piecewise_joins_stay_within_one_second() {
    println!("boundary      lower-side ΔT     upper-side ΔT     |jump|");
    for &boundary in BOUNDARIES {
        let lower = delta_t_seconds(boundary - EPSILON_YEARS);
        let upper = delta_t_seconds(boundary);
        let jump = (upper - lower).abs();
        println!("{boundary:>8.1}   {lower:>14.6}   {upper:>14.6}   {jump:.6}");
        assert!(
            jump < 1.0,
            "piecewise join at year {boundary} has |jump| = {jump:.6} s >= 1.0 s \
             (lower-side {lower:.6}, upper-side {upper:.6})"
        );
    }
}

/// `y >= 2150` and `y < -500` share the same long-term parabola
/// (`-20 + 32 * ((y-1820)/100)^2`); spot-check both tails plus the y=2150
/// join (already covered by `piecewise_joins_stay_within_one_second`, but
/// checked again here directly against the closed-form value for extra
/// confidence since the two tails are the two most extreme extrapolation
/// regimes).
#[test]
fn long_term_tails_match_closed_form_parabola() {
    for &year in &[-3000.0, -1999.0, -500.000_001, 2150.0, 2500.0, 3000.0] {
        let u = (year - 1820.0) / 100.0;
        let expected = -20.0 + 32.0 * u * u;
        let got = delta_t_seconds(year);
        assert!(
            (got - expected).abs() < 1e-9,
            "delta_t_seconds({year}) = {got} != parabola {expected}"
        );
    }
}
