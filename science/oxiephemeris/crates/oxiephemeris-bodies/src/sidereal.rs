//! Earth-rotation angles: Earth Rotation Angle (ERA), Greenwich mean
//! sidereal time (GMST, IAU 2006), the equation of the equinoxes
//! (including the complementary terms), and Greenwich apparent sidereal
//! time (GAST).
//!
//! # Time arguments (supplied by the caller)
//!
//! Two distinct time scales drive these angles, and this module does *no*
//! time-scale conversion of its own:
//!
//! - `jd_ut1` — the epoch as a two-part Julian Date in **UT1**. UT1 is
//!   obtained from civil time as `UT1 = UTC + dUT1`, with `dUT1 =
//!   UT1−UTC` taken from IERS Earth-orientation data (e.g. the
//!   `finals2000A.all` combined series or Bulletin A); see USNO
//!   Circular 179 (Kaplan 2005), eq. 2.4. Use
//!   [`oxiephemeris_core::time`] for the UTC ↔ TAI ↔ TT legs and
//!   [`JulianDate::add_seconds`] for applying `dUT1`.
//! - `t_tt` — Julian centuries **TT** since J2000.0, `t = (JD(TT) −
//!   2451545.0) / 36525`, the same convention as every function in
//!   [`crate::frames`] (formally TDB, but TT is used in practice; the
//!   difference is negligible here — IERS TN36 §5.6.4, §5.7.2).
//!
//! # References (clean-room sources)
//!
//! - IERS Conventions (2010), IERS Technical Note 36 ("TN36"), ch. 5:
//!   eq. (5.15) for the ERA, eq. (5.32) for GMST(IAU 2006).
//! - Capitaine, Wallace & Chapront (2003), A&A 412, 567 (P03; the GMST
//!   polynomial is the accumulated precession of the equinox in right
//!   ascension, their eq. 42), with the final IAU 2006-consistent GMST
//!   expression as given by Capitaine, Wallace & Chapront (2005), A&A
//!   431, 633 (erratum) and adopted by TN36 eq. (5.32).
//! - Kaplan (2005), USNO Circular 179, §2.6.2: eq. 2.11/2.12 (ERA,
//!   including the numerically preferable split form), eq. 2.13
//!   (GMST, GAST), eq. 2.14 (equation of the equinoxes with the full
//!   list of complementary terms), eq. 2.15 (local sidereal time).
//! - IAU 2000 Resolution B1.8, note 3 (definition of UT1 through the
//!   ERA; the ERA constants originate there).

use crate::frames::{
    mean_obliquity_iau2006, nutation_iau2000a_truncated, ARCSEC_TO_RAD, TURN_ARCSEC,
};
use libm::{cos, floor, sin};
use oxiephemeris_core::angle::{normalize_0_two_pi, TWO_PI};
use oxiephemeris_core::time::{JulianDate, J2000_JD};

/// ERA at the epoch J2000.0 UT1, in revolutions: the constant term of
/// IERS TN36 eq. (5.15) / IAU 2000 Res. B1.8 note 3.
const ERA_AT_J2000_REV: f64 = 0.779_057_273_264_0;

/// ERA rate minus one revolution per day, in revolutions per UT1 day:
/// `1.00273781191135448 − 1` (IERS TN36 eq. 5.15). Keeping only the
/// excess over one revolution lets whole UT1 days be reduced exactly
/// (Circular 179, eq. 2.12).
const ERA_RATE_EXCESS_REV_PER_DAY: f64 = 0.002_737_811_911_354_48;

/// Earth Rotation Angle, radians in `[0, 2π)`, for a two-part UT1 epoch.
///
/// IERS TN36 eq. (5.15) (= IAU 2000 Resolution B1.8, note 3):
///
/// ```text
/// ERA(Tu) = 2π (0.7790572732640 + 1.00273781191135448 · Tu),
/// Tu = JD(UT1) − 2451545.0
/// ```
///
/// # Precision (why the two-part JD matters)
///
/// Evaluated naively on a one-part `f64` JD, `1.00273781191135448 · Tu`
/// is ~2.5 × 10⁶ revolutions for contemporary dates and its rounding
/// error alone reaches ~10⁻⁹ rad after reduction. Here the hi and lo
/// parts are reduced to fractional revolutions *separately* first, in
/// the split form of USNO Circular 179 eq. 2.12,
///
/// ```text
/// ERA/2π = 0.7790572732640 + 0.00273781191135448 · Tu + frac(Tu)  (mod 1)
/// ```
///
/// with `Tu = (jd.hi − 2451545.0) + jd.lo`. The subtraction
/// `jd.hi − 2451545.0` is exact for `jd.hi ∈ [J2000/2, 2·J2000]`
/// (Sterbenz's lemma; roughly years −1350 to +8700), `frac` of that
/// exact difference is again exact, and the only surviving rounding is
/// in the small products and the final sum: < 10⁻¹³ rad representation
/// error over 1900–2100 (< 10⁻¹² over the full DE440 span).
#[must_use]
pub fn era(jd_ut1: JulianDate) -> f64 {
    // Tu split: exact difference of the large part, plus the small part.
    let du_hi = jd_ut1.hi - J2000_JD;
    let du_lo = jd_ut1.lo;
    // frac(du_hi): exact (du_hi and floor(du_hi) are both representable
    // and their difference is a multiple of ulp(du_hi) below 1).
    let frac_hi = du_hi - floor(du_hi);
    // Excess rotation of the large part, reduced modulo one revolution.
    // For |du_hi| ≤ 4 × 10⁴ d the product is ≤ ~110 rev, so its 0.5-ulp
    // rounding is ≤ 8 × 10⁻¹⁵ rev ≈ 5 × 10⁻¹⁴ rad.
    let excess_hi = ERA_RATE_EXCESS_REV_PER_DAY * du_hi;
    let excess_hi_frac = excess_hi - floor(excess_hi);
    // The lo part is ≪ 1 day, so its full-rate contribution
    // (1 + excess) · du_lo needs no reduction.
    let rev =
        ERA_AT_J2000_REV + frac_hi + excess_hi_frac + du_lo + ERA_RATE_EXCESS_REV_PER_DAY * du_lo;
    normalize_0_two_pi(TWO_PI * (rev - floor(rev)))
}

/// The GMST−ERA polynomial of IERS TN36 eq. (5.32), in arcseconds: the
/// accumulated precession of the equinox in right ascension consistent
/// with IAU 2006 precession (Capitaine, Wallace & Chapront 2003, A&A
/// 412, 567, eq. 42; final form Capitaine, Wallace & Chapront 2005, A&A
/// 431, 633; also USNO Circular 179 eq. 2.13).
fn gmst_polynomial_arcsec(t_tt: f64) -> f64 {
    0.014_506
        + t_tt
            * (4_612.156_534
                + t_tt
                    * (1.391_581_7
                        + t_tt
                            * (-0.000_000_44
                                + t_tt * (-0.000_029_956 + t_tt * (-0.000_000_036_8)))))
}

/// Greenwich mean sidereal time (IAU 2006), radians in `[0, 2π)`.
///
/// IERS TN36 eq. (5.32) (Capitaine, Wallace & Chapront 2005, A&A 431,
/// 633; USNO Circular 179 eq. 2.13):
///
/// ```text
/// GMST = ERA(UT1) + 0.014506″ + 4612.156534″ t + 1.3915817″ t²
///        − 0.00000044″ t³ − 0.000029956″ t⁴ − 0.0000000368″ t⁵
/// ```
///
/// Two time scales enter deliberately: the fast term through `jd_ut1`
/// (UT1), the polynomial through `t_tt` (Julian centuries TT since
/// J2000.0, the [`crate::frames`] convention).
#[must_use]
pub fn gmst_iau2006(jd_ut1: JulianDate, t_tt: f64) -> f64 {
    normalize_0_two_pi(era(jd_ut1) + gmst_polynomial_arcsec(t_tt) * ARCSEC_TO_RAD)
}

/// Fundamental luni-solar arguments `F`, `D`, `Ω` in radians, reduced
/// modulo one turn, for `t` Julian centuries TT since J2000.0.
///
/// Developments of IERS TN36 eq. (5.43) (Simon et al. 1994; the same
/// expressions are USNO Circular 179 eq. 5.19):
///
/// ```text
/// F  = 335779.526232″ + 1739527262.8478″ t − 12.7512″ t²
///      − 0.001037″ t³ + 0.00000417″ t⁴
/// D  = 1072260.703692″ + 1602961601.2090″ t − 6.3706″ t²
///      + 0.006593″ t³ − 0.00003169″ t⁴
/// Om = 450160.398036″ − 6962890.5431″ t + 7.4722″ t²
///      + 0.007702″ t³ − 0.00005939″ t⁴
/// ```
///
/// These duplicate the `F`/`D`/`Ω` entries of the crate-private
/// `frames::nutation::delaunay_args` (not reachable from this module);
/// both are implementations of the same published development.
fn fund_args_f_d_om(t: f64) -> (f64, f64, f64) {
    let f = 335_779.526_232
        + t * (1_739_527_262.847_8 + t * (-12.7512 + t * (-0.001_037 + t * 0.000_004_17)));
    let d = 1_072_260.703_692
        + t * (1_602_961_601.209 + t * (-6.3706 + t * (0.006_593 + t * (-0.000_031_69))));
    let om = 450_160.398_036
        + t * (-6_962_890.543_1 + t * (7.4722 + t * (0.007_702 + t * (-0.000_059_39))));
    (
        (f % TURN_ARCSEC) * ARCSEC_TO_RAD,
        (d % TURN_ARCSEC) * ARCSEC_TO_RAD,
        (om % TURN_ARCSEC) * ARCSEC_TO_RAD,
    )
}

/// Complementary terms of the equation of the equinoxes, in arcseconds:
/// every non-`t` term of USNO Circular 179 eq. 2.14, as multiples
/// `(coefficient″, n_F, n_D, n_Ω)` of the fundamental arguments, largest
/// first. The single `t`-proportional term of the same equation is
/// handled separately in [`ee_complementary_arcsec`].
///
/// Circular 179 notes this series is a truncated form of the longer
/// series in the IERS Conventions (2003) — terms below its 0.5 µas
/// cut-off are omitted by the source itself.
const EE_COMPLEMENTARY_TERMS: [(f64, i8, i8, i8); 8] = [
    (0.002_640_96, 0, 0, 1),   // +0.00264096″ sin(Ω)
    (0.000_063_52, 0, 0, 2),   // +0.00006352″ sin(2Ω)
    (0.000_011_75, 2, -2, 3),  // +0.00001175″ sin(2F − 2D + 3Ω)
    (0.000_011_21, 2, -2, 1),  // +0.00001121″ sin(2F − 2D + Ω)
    (-0.000_004_55, 2, -2, 2), // −0.00000455″ sin(2F − 2D + 2Ω)
    (0.000_002_02, 2, 0, 3),   // +0.00000202″ sin(2F + 3Ω)
    (0.000_001_98, 2, 0, 1),   // +0.00000198″ sin(2F + Ω)
    (-0.000_001_72, 0, 0, 3),  // −0.00000172″ sin(3Ω)
];

/// Coefficient of the `t·sin(Ω)` complementary term of USNO Circular 179
/// eq. 2.14, arcseconds per Julian century: `−0.00000087″ t sin(Ω)`.
const EE_COMPLEMENTARY_T_SIN_OM_ARCSEC: f64 = -0.000_000_87;

/// Sum of the complementary terms of the equation of the equinoxes
/// (USNO Circular 179 eq. 2.14), in arcseconds, smallest terms first for
/// numerical hygiene.
fn ee_complementary_arcsec(t: f64) -> f64 {
    let (f, d, om) = fund_args_f_d_om(t);
    let mut sum = EE_COMPLEMENTARY_T_SIN_OM_ARCSEC * t * sin(om);
    for &(coeff, nf, nd, nom) in EE_COMPLEMENTARY_TERMS.iter().rev() {
        let arg = f64::from(nf) * f + f64::from(nd) * d + f64::from(nom) * om;
        sum += coeff * sin(arg);
    }
    sum
}

/// Equation of the equinoxes `GAST − GMST`, radians, for `t` Julian
/// centuries TT since J2000.0.
///
/// USNO Circular 179 eq. 2.14 (the full published series — the leading
/// term of TN36 eq. 5.36 plus every complementary term the Circular
/// lists):
///
/// ```text
/// EE = Δψ cos εA
///    + 0.00264096″ sin(Ω)      + 0.00006352″ sin(2Ω)
///    + 0.00001175″ sin(2F−2D+3Ω) + 0.00001121″ sin(2F−2D+Ω)
///    − 0.00000455″ sin(2F−2D+2Ω) + 0.00000202″ sin(2F+3Ω)
///    + 0.00000198″ sin(2F+Ω)     − 0.00000172″ sin(3Ω)
///    − 0.00000087″ t sin(Ω)
/// ```
///
/// `Δψ` is the nutation in longitude from
/// [`nutation_iau2000a_truncated`] and `εA` the IAU 2006 mean obliquity
/// from [`mean_obliquity_iau2006`]. The **truncated** series is a
/// deliberate choice here (the apparent-place pipeline defaults to the
/// full series, [`crate::frames::NutationModel`]): its `Δψ` deviates from
/// the full IAU `2000A_R06` series by < 1.1 mas over 1900–2100 (measured
/// in `tests/nutation_oracle.rs`), i.e. < 1.1 mas · cos εA ≈ 1 mas in EE
/// ≈ 0.07 ms of sidereal time — well inside the 1 ms Horizons LAST gate
/// of `tests/sidereal.rs`. The complementary terms account for the
/// accumulated interaction of precession and nutation on the equinox
/// (Circular 179 §2.6.2); they stay below ~3 mas, while the leading term
/// reaches ±16″ (≈ ±1.05 s of time).
#[must_use]
pub fn equation_of_equinoxes(t_tt: f64) -> f64 {
    let nut = nutation_iau2000a_truncated(t_tt);
    let eps_a = mean_obliquity_iau2006(t_tt);
    nut.dpsi_rad * cos(eps_a) + ee_complementary_arcsec(t_tt) * ARCSEC_TO_RAD
}

/// Greenwich apparent sidereal time (IAU 2006/2000A-truncated), radians
/// in `[0, 2π)`: `GAST = GMST + EE` (USNO Circular 179 eq. 2.13/2.14;
/// TN36 §5.5.7).
///
/// Same two-time-scale convention as [`gmst_iau2006`]: `jd_ut1` drives
/// the fast (rotation) term, `t_tt` the precession-nutation terms. Local
/// apparent sidereal time follows as `LAST = GAST + λ` with `λ` the
/// (TIO-corrected) east longitude (Circular 179 eq. 2.15/2.16).
#[must_use]
pub fn gast_iau2006(jd_ut1: JulianDate, t_tt: f64) -> f64 {
    normalize_0_two_pi(gmst_iau2006(jd_ut1, t_tt) + equation_of_equinoxes(t_tt))
}
