//! IAU `2000A_R06` nutation (full series and a fast truncation), the
//! classical nutation matrix, the combined GCRS → true-of-date rotation,
//! and equatorial → ecliptic rotations.
//!
//! Two evaluators of the same underlying model are provided:
//!
//! - [`nutation_iau2000a`] — the FULL IAU `2000A_R06` series (1320 + 38
//!   longitude terms, 1037 + 19 obliquity terms), generated verbatim from
//!   the IERS Conventions (2010) electronic tables by
//!   `cargo run -p xtask -- gen-nutation` (see the `nutation_2000a_*`
//!   sibling modules for provenance).
//! - [`nutation_iau2000a_truncated`] — a truncation of the same series
//!   regenerated from the same tables — see `nutation_truncated_table`
//!   for the full provenance note. It is **not** the IAU 2000B series of
//!   McCarthy & Luzum (2003, Celest. Mech. Dyn. Astron. 85, 37), but it
//!   matches the full series to better than 0.7 mas in `dpsi` and 0.4 mas
//!   in `deps` over 1995–2050, i.e. the accuracy class of IAU 2000B.
//!
//! [`NutationModel`] selects between the two; its `Default` (and hence
//! the default of every matrix builder below and of the apparent-place
//! pipeline) is the **full** series.

use super::nutation_2000a_eps_ls::{EPS_LS_J0, EPS_LS_J0_LEN, EPS_LS_J1, EPS_LS_J1_LEN};
use super::nutation_2000a_eps_pl::{EPS_PL_J0, EPS_PL_J0_LEN};
use super::nutation_2000a_psi_ls::{PSI_LS_J0, PSI_LS_J0_LEN, PSI_LS_J1, PSI_LS_J1_LEN};
use super::nutation_2000a_psi_pl::{PSI_PL_J0, PSI_PL_J0_LEN};
use super::nutation_truncated_table::{BIAS_DEPS_UAS, BIAS_DPSI_UAS, LS_TERMS, PL_TERMS};
use super::obliquity::mean_obliquity_iau2006;
use super::precession::{fw_angles_iau2006, fw_rotation};
use super::{ARCSEC_TO_RAD, TURN_ARCSEC};
use crate::math::{r1, r3, Mat3};
use libm::{cos, sin};

/// Microarcseconds to radians.
const UAS_TO_RAD: f64 = ARCSEC_TO_RAD * 1.0e-6;

/// Nutation components referred to the ecliptic of date.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nutation {
    /// Nutation in longitude `dpsi`, radians.
    pub dpsi_rad: f64,
    /// Nutation in obliquity `deps`, radians.
    pub deps_rad: f64,
}

/// Selector between the two evaluators of the IAU `2000A_R06` nutation
/// model, used by the matrix builders of this module and by the
/// apparent-place pipeline ([`crate::apparent::Options::nutation`]).
///
/// Both variants evaluate the *same* underlying model (see the module
/// docs), so the choice is purely an accuracy/speed trade-off; they agree
/// to better than 0.7 mas in `dpsi` and 0.4 mas in `deps` over 1995–2050
/// (≲ 1.1 / 0.4 mas over 1900–2100).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NutationModel {
    /// The full IAU `2000A_R06` series ([`nutation_iau2000a`]: 1320 + 38
    /// longitude and 1037 + 19 obliquity terms), good to the
    /// microarcsecond level of the published model. The default.
    #[default]
    Iau2000a,
    /// The truncated series ([`nutation_iau2000a_truncated`]: 143
    /// luni-solar + 11 planetary terms, ≈ 15× cheaper), IAU 2000B
    /// accuracy class.
    Iau2000aTruncated,
}

impl NutationModel {
    /// Evaluates the selected nutation series for `t` Julian centuries TT
    /// since J2000.0.
    #[must_use]
    pub fn evaluate(self, t: f64) -> Nutation {
        match self {
            Self::Iau2000a => nutation_iau2000a(t),
            Self::Iau2000aTruncated => nutation_iau2000a_truncated(t),
        }
    }
}

/// One term of the full-series tables, luni-solar part:
/// `([l, l', F, D, Om] multipliers, sin amplitude, cos amplitude)` with
/// amplitudes in µas (`j = 0` tables) or µas per Julian century TT
/// (`j = 1` tables).
pub(crate) type FullLsTerm = ([i8; 5], f64, f64);

/// One term of the full-series tables, planetary part: multipliers over
/// the complete 14-argument set `l, l', F, D, Om, L_Me, L_Ve, L_E, L_Ma,
/// L_J, L_Sa, L_U, L_Ne, p_A` (TN36 eqs. 5.43/5.44), plus sin and cos
/// amplitudes in µas.
pub(crate) type FullPlTerm = ([i8; 14], f64, f64);

/// Numbers of terms carried by the generated full IAU `2000A_R06` tables,
/// in source-table order: `[tab5.3a j = 0, tab5.3a j = 1, tab5.3b j = 0,
/// tab5.3b j = 1]`. Matches the term counts declared in the headers of
/// the IERS files (1320, 38, 1037, 19).
pub const NUTATION_IAU2000A_TERM_COUNTS: [usize; 4] = [
    PSI_LS_J0_LEN + PSI_PL_J0_LEN,
    PSI_LS_J1_LEN,
    EPS_LS_J0_LEN + EPS_PL_J0_LEN,
    EPS_LS_J1_LEN,
];

/// Delaunay fundamental arguments `l, l', F, D, Om` in radians, reduced
/// modulo one turn, for `t` Julian centuries TT since J2000.0.
///
/// Developments of IERS TN36 eq. (5.43) (from Simon et al. 1994), with the
/// constant terms converted from degrees to arcseconds:
///
/// ```text
/// l  = 485868.249036" + 1717915923.2178" t + 31.8792" t^2
///      + 0.051635" t^3 - 0.00024470" t^4
/// l' = 1287104.793048" + 129596581.0481" t - 0.5532" t^2
///      + 0.000136" t^3 - 0.00001149" t^4
/// F  = 335779.526232" + 1739527262.8478" t - 12.7512" t^2
///      - 0.001037" t^3 + 0.00000417" t^4
/// D  = 1072260.703692" + 1602961601.2090" t - 6.3706" t^2
///      + 0.006593" t^3 - 0.00003169" t^4
/// Om = 450160.398036" - 6962890.5431" t + 7.4722" t^2
///      + 0.007702" t^3 - 0.00005939" t^4
/// ```
pub(crate) fn delaunay_args(t: f64) -> [f64; 5] {
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
    ]
}

/// Planetary fundamental arguments `L_Me, L_Ve, L_E, L_Ma, L_J, L_Sa, L_U,
/// L_Ne, p_A` in radians (not reduced; the values stay small enough), for
/// `t` Julian centuries TT since J2000.0.
///
/// Developments of IERS TN36 eq. (5.44), in radians (mean longitudes from
/// Souchay et al. 1999 / Simon et al. 1994; the general precession in
/// longitude `p_A` from Kinoshita & Souchay 1990).
pub(crate) fn planetary_args(t: f64) -> [f64; 9] {
    [
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

/// Nutation in longitude and obliquity from a truncated IAU `2000A_R06`
/// series, for `t` Julian centuries TT since J2000.0.
///
/// Evaluates the truncated IAU `2000A_R06` series of
/// `nutation_truncated_table` (TN36 eq. 5.35, 143 luni-solar +
/// 11 planetary terms) plus constant planetary-bias offsets, in the
/// manner of — but **not identical to** — the abridged IAU 2000B model
/// (McCarthy & Luzum 2003); it reproduces the full `2000A_R06` series to
/// the IAU 2000B accuracy class ("1 mas"), see the table module for the
/// exact bounds. The result is referred to the ecliptic of date, and is
/// meant to be paired with the IAU 2006 precession of
/// [`super::precession_bias_matrix`].
#[must_use]
pub fn nutation_iau2000a_truncated(t: f64) -> Nutation {
    let fa = delaunay_args(t);
    let pa = planetary_args(t);

    // Sum smallest terms first for numerical hygiene.
    let mut dpsi_uas = BIAS_DPSI_UAS;
    let mut deps_uas = BIAS_DEPS_UAS;
    for (mult, c) in PL_TERMS.iter().rev() {
        let mut arg = 0.0;
        for (n, a) in mult.iter().zip(fa.iter().chain(pa.iter())) {
            if *n != 0 {
                arg += f64::from(*n) * a;
            }
        }
        let (s, co) = (sin(arg), cos(arg));
        dpsi_uas += c[0] * s + c[1] * co;
        deps_uas += c[2] * co + c[3] * s;
    }
    for (mult, c) in LS_TERMS.iter().rev() {
        let mut arg = 0.0;
        for (n, a) in mult.iter().zip(fa.iter()) {
            if *n != 0 {
                arg += f64::from(*n) * a;
            }
        }
        let (s, co) = (sin(arg), cos(arg));
        dpsi_uas += (c[0] + c[1] * t) * s + (c[2] + c[3] * t) * co;
        deps_uas += (c[4] + c[5] * t) * co + (c[6] + c[7] * t) * s;
    }
    Nutation {
        dpsi_rad: dpsi_uas * UAS_TO_RAD,
        deps_rad: deps_uas * UAS_TO_RAD,
    }
}

/// Sum of `s * sin(arg) + c * cos(arg)` over luni-solar full-series terms,
/// iterated in reverse (the tables are amplitude-ordered, so this sums the
/// smallest terms first for numerical hygiene). Result in the amplitude
/// unit (µas for `j = 0` tables, µas/cy for `j = 1`).
fn ls_series_sum(terms: &[FullLsTerm], fa: &[f64; 5]) -> f64 {
    let mut total = 0.0;
    for (mult, s_amp, c_amp) in terms.iter().rev() {
        let mut arg = 0.0;
        for (n, a) in mult.iter().zip(fa.iter()) {
            if *n != 0 {
                arg += f64::from(*n) * a;
            }
        }
        total += s_amp * sin(arg) + c_amp * cos(arg);
    }
    total
}

/// Sum of `s * sin(arg) + c * cos(arg)` over planetary full-series terms
/// (14-argument set), iterated in reverse; result in µas.
fn pl_series_sum(terms: &[FullPlTerm], fa: &[f64; 5], pa: &[f64; 9]) -> f64 {
    let mut total = 0.0;
    for (mult, s_amp, c_amp) in terms.iter().rev() {
        let mut arg = 0.0;
        for (n, a) in mult.iter().zip(fa.iter().chain(pa.iter())) {
            if *n != 0 {
                arg += f64::from(*n) * a;
            }
        }
        total += s_amp * sin(arg) + c_amp * cos(arg);
    }
    total
}

/// Nutation in longitude and obliquity from the FULL IAU `2000A_R06`
/// series, for `t` Julian centuries TT since J2000.0.
///
/// Evaluates the complete series of IERS TN36 eq. (5.35) — 1320 + 38
/// terms in longitude, 1037 + 19 in obliquity, from the IERS Conventions
/// (2010) electronic tables `tab5.3a.txt` / `tab5.3b.txt` (IAU
/// `2000A_R06` expression: the IAU 2000A amplitudes with the slight IAU
/// 2006 adjustments already applied) — with the fundamental arguments of
/// TN36 eqs. (5.43)/(5.44). The coefficient tables are committed
/// generated code; regenerate with `cargo run -p xtask -- gen-nutation`.
///
/// The result is referred to the ecliptic of date and is meant to be
/// paired with the IAU 2006 precession of
/// [`super::precession_bias_matrix`]. For a ~15× cheaper evaluation at
/// the milliarcsecond level (154 instead of 2414 sin/cos terms) use
/// [`nutation_iau2000a_truncated`].
#[must_use]
pub fn nutation_iau2000a(t: f64) -> Nutation {
    let fa = delaunay_args(t);
    let pa = planetary_args(t);
    // Smallest contributions first: planetary block, then the t-linear
    // luni-solar block, then the dominant j = 0 luni-solar block.
    let dpsi_uas = pl_series_sum(&PSI_PL_J0, &fa, &pa)
        + t * ls_series_sum(&PSI_LS_J1, &fa)
        + ls_series_sum(&PSI_LS_J0, &fa);
    let deps_uas = pl_series_sum(&EPS_PL_J0, &fa, &pa)
        + t * ls_series_sum(&EPS_LS_J1, &fa)
        + ls_series_sum(&EPS_LS_J0, &fa);
    Nutation {
        dpsi_rad: dpsi_uas * UAS_TO_RAD,
        deps_rad: deps_uas * UAS_TO_RAD,
    }
}

/// Classical nutation matrix `N(t)`: rotates vectors from the mean equator
/// and equinox of date to the true equator and equinox of date, with the
/// nutation components from the **default** [`NutationModel`] (the full
/// IAU `2000A_R06` series). Use [`nutation_matrix_with`] to select the
/// series explicitly.
///
/// Composition (IERS TN36 §5.4.5, fig. 5.1):
///
/// ```text
/// N(t) = R1(-(eps_A + deps)) . R3(-dpsi) . R1(eps_A)
/// ```
#[must_use]
pub fn nutation_matrix(t: f64) -> Mat3 {
    nutation_matrix_with(t, NutationModel::default())
}

/// [`nutation_matrix`] with an explicit [`NutationModel`].
#[must_use]
pub fn nutation_matrix_with(t: f64, model: NutationModel) -> Mat3 {
    let eps_a = mean_obliquity_iau2006(t);
    let nut = model.evaluate(t);
    r1(-(eps_a + nut.deps_rad))
        .mul(&r3(-nut.dpsi_rad))
        .mul(&r1(eps_a))
}

/// Combined GCRS → true equator and equinox of date matrix,
/// `NPB(t) = N(t) . PB(t)`, frame bias included, with the nutation
/// components from the **default** [`NutationModel`] (the full IAU
/// `2000A_R06` series). Use [`gcrs_to_true_of_date_with`] to select the
/// series explicitly.
///
/// Implemented as the single Fukushima–Williams 4-rotation of IERS TN36
/// §5.4.5 with the nutation components added to the last two angles:
///
/// ```text
/// NPB(t) = R1(-(eps_A + deps)) . R3(-(psi_bar + dpsi))
///          . R1(phi_bar) . R3(gamma_bar)
/// ```
///
/// which is algebraically identical to
/// `nutation_matrix(t) * precession_bias_matrix(t)` (the inner
/// `R1(eps_A) . R1(-eps_A)` of the explicit product collapses).
#[must_use]
pub fn gcrs_to_true_of_date(t: f64) -> Mat3 {
    gcrs_to_true_of_date_with(t, NutationModel::default())
}

/// [`gcrs_to_true_of_date`] with an explicit [`NutationModel`].
#[must_use]
pub fn gcrs_to_true_of_date_with(t: f64, model: NutationModel) -> Mat3 {
    let fw = fw_angles_iau2006(t);
    let nut = model.evaluate(t);
    fw_rotation(&fw, nut.dpsi_rad, nut.deps_rad)
}

/// Rotation from the **mean** equator and equinox of date to the mean
/// ecliptic and mean equinox of date: `R1(eps_A)` (rotate about the mean
/// equinox x-axis by the mean obliquity; TN36 §5.4.5, fig. 5.1).
///
/// Longitudes in the resulting frame are measured from the *mean* equinox
/// of date.
#[must_use]
pub fn mean_of_date_to_ecliptic(t: f64) -> Mat3 {
    r1(mean_obliquity_iau2006(t))
}

/// Rotation from the **true** equator and equinox of date to the ecliptic
/// and *true* equinox of date: `R1(eps_A + deps)` (rotate about the true
/// equinox x-axis by the true obliquity), with `deps` from the **default**
/// [`NutationModel`] (the full IAU `2000A_R06` series). Use
/// [`true_of_date_to_ecliptic_with`] to select the series explicitly.
///
/// Longitudes in the resulting frame are measured from the *true* equinox
/// of date; they differ from mean-equinox ecliptic longitudes by the
/// nutation in longitude `dpsi`.
#[must_use]
pub fn true_of_date_to_ecliptic(t: f64) -> Mat3 {
    true_of_date_to_ecliptic_with(t, NutationModel::default())
}

/// [`true_of_date_to_ecliptic`] with an explicit [`NutationModel`].
#[must_use]
pub fn true_of_date_to_ecliptic_with(t: f64, model: NutationModel) -> Mat3 {
    let eps_a = mean_obliquity_iau2006(t);
    let nut = model.evaluate(t);
    r1(eps_a + nut.deps_rad)
}
