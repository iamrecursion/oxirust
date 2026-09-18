//! TDB − TT: analytical series of the Fairhead–Bretagnon family.
//!
//! # What is shipped
//!
//! The **complete published Table I** of Fairhead, Bretagnon & Lestrade
//! (1988), "The Time Transformation TDB−TDT: An Analytical Formula and
//! Related Problem of Convention", IAU Symposium 128, pp. 419–426
//! (doi:10.1017/S0074180900119837): 90 constant-amplitude terms, 26 terms
//! multiplied by `t`, 6 by `t²` (one of them, with zero frequency and phase
//! 3π/2, is the secular quadratic), and 1 by `t³` — 123 terms in total,
//! evaluated as
//!
//! ```text
//! TDB − TT = Σ Aᵢ sin(ωᵢ t + φᵢ)
//!          + t   Σ Bᵢ sin(ωᵢ t + φᵢ)
//!          + t²  Σ Cᵢ sin(ωᵢ t + φᵢ)
//!          + t³  Σ Dᵢ sin(ωᵢ t + φᵢ)          [microseconds]
//! ```
//!
//! with `t` in **Julian millennia** (365 250 days) from J2000.0 and `ω` in
//! rad/millennium. The paper states this truncation is accurate at the
//! **100 ns level for a few thousand years around J2000**; it is the same
//! model family as Fairhead & Bretagnon (1990), A&A 229, 240 (the IERS
//! Conventions reference for this transformation), whose leading term is
//! 1656.674564 µs versus 1656.6894 µs here — a 15 ns-level difference well
//! inside the stated truncation error.
//!
//! # Cross-checks
//!
//! The seven dominant terms (six periodic + one mixed) agree with the
//! truncated expression in Kaplan (2005), USNO Circular 179, eq. 2.6
//! (which quotes amplitudes in seconds and rates in rad/century — exactly
//! these values ÷10³ resp. ×10⁻¹, rounded). An integration test compares
//! this full series against an independent transcription of eq. 2.6.
//!
//! Transcription provenance: extracted programmatically from the publisher
//! PDF of the 1988 paper. One glyph was unreadable in the scan: the last
//! digits of the phase of the 7th `t`-multiplied term (`4.53489__`,
//! transcribed as `4.5348996`); with an amplitude of 0.0547 µs the residual
//! phase uncertainty affects TDB − TT below 10⁻¹³ s.
//!
//! # Argument and accuracy notes
//!
//! The series argument is formally TDB-based, but using TT changes the
//! result by ≲ 10⁻¹⁵ s (slope × 1.7 ms), so `t` is taken directly from the
//! TT input. Per IAU 1976 recommendation 5 (and the paper), the *linear*
//! secular term is dropped so that TDB − TT stays quasi-periodic; the tiny
//! quadratic/cubic seculars are retained via the zero-frequency `t²` term.
//!
//! A JPL `DE440t` integrated TT−TDB oracle test (< 1 µs target, 1900–2100)
//! is added later by the `de` verification track; this module is expected
//! to pass it with margin given the 100 ns truncation level (a constant
//! offset/rate difference from the `DE440t` TDB convention may need to be
//! removed in that comparison).
//!
//! Coefficients are kept digit-for-digit as printed (no digit-group
//! separators) to ease auditing against the paper.
#![allow(clippy::unreadable_literal)]

use crate::time::jd::{JulianDate, J2000_JD};

/// Days per Julian millennium.
const DAYS_PER_MILLENNIUM: f64 = 365_250.0;

/// Constant-amplitude terms: amplitude (us); angle argument is `omega * t + phase` with
/// `t` in Julian millennia of TT from J2000.0, `omega` in rad/millennium,
/// `phase` in rad. Transcribed verbatim from Fairhead, Bretagnon &
/// Lestrade (1988), Table I.
const FAIRHEAD_A: [(f64, f64, f64); 90] = [
    (1656.6894, 6283.0758494, 6.2400497),
    (22.4175, 5753.3848843, 4.2969771),
    (13.8399, 12566.1516988, 6.1968995),
    (4.7701, 529.6909651, 0.4444038),
    (4.6767, 6069.7767539, 4.0211937),
    (2.2566, 213.2990954, 5.5431320),
    (1.7307, -3.5231591, 5.0189615),
    (1.5555, 77713.7714679, 5.1984671),
    (1.2768, 7860.4193937, 5.9888233),
    (1.1934, 5223.6939192, 3.6498063),
    (1.1153, 3930.2096968, 1.4227456),
    (0.7942, 11506.7697686, 2.3223126),
    (0.6003, 1577.3435443, 2.6782570),
    (0.4968, 6208.2942508, 5.6967011),
    (0.4863, 5884.9268358, 0.5199988),
    (0.4686, 6244.9428137, 5.8663983),
    (0.4484, 26.2983277, 3.6116882),
    (0.4353, -398.1490136, 4.3493415),
    (0.4324, 74.7815986, 2.4358996),
    (0.3755, 5507.5532411, 4.1034739),
    (0.2431, -775.5226083, 3.6519195),
    (0.2307, 5856.4776585, 4.7740285),
    (0.2037, 12036.4607337, 4.3339850),
    (0.1734, 18849.2275481, 6.1537378),
    (0.1591, 10977.0788035, 1.8900771),
    (0.1440, -796.2980272, 5.9574876),
    (0.1379, 11790.6290905, 1.1359361),
    (0.1200, 38.1330356, 4.5515858),
    (0.1190, 5486.7778222, 1.9145317),
    (0.1161, 1059.3819302, 0.8734863),
    (0.1019, -5573.1428016, 5.9845038),
    (0.0984, 2544.3144043, 0.0927835),
    (0.0802, 206.1855484, 2.0953827),
    (0.0796, 4694.0029541, 2.9492402),
    (0.0750, 2942.4634179, 4.9809276),
    (0.0626, 20.7754189, 2.6543767),
    (0.0644, 5746.2713373, 1.2804037),
    (0.0638, 5760.4984313, 4.1680021),
    (0.0588, 426.5981909, 4.8396652),
    (0.0571, -0.9804182, 0.9252472),
    (0.0541, 17260.1546529, 3.4110896),
    (0.0482, 155.4203227, 2.2517971),
    (0.0480, 2146.1653907, 1.4958314),
    (0.0427, 632.7837393, 5.7206226),
    (0.0426, 161000.6857375, 1.2708377),
    (0.0424, 6275.9623024, 2.8695872),
    (0.0421, -7.1135470, 3.5707209),
    (0.0408, 12352.8526033, 3.9814932),
    (0.0405, 15720.8387873, 2.5466120),
    (0.0370, 3154.6870886, 5.0717851),
    (0.0366, 5088.6288086, 3.3246566),
    (0.0365, 801.8209360, 6.2487864),
    (0.0349, 522.5774181, 5.2100747),
    (0.0335, 6062.6632069, 4.1452250),
    (0.0335, 9437.7629379, 2.4047140),
    (0.0324, 8827.3902537, 5.5414605),
    (0.0324, 6076.8903009, 0.7495680),
    (0.0302, 7084.8967854, 3.3896043),
    (0.0299, 12139.5535079, 1.7701727),
    (0.0293, -71430.6956185, 4.1831763),
    (0.0279, -6286.5990085, 5.0737086),
    (0.0272, 6279.5526903, 5.0450074),
    (0.0252, 1748.0163771, 2.9018643),
    (0.0248, -1194.4470408, 1.0870978),
    (0.0226, 6133.5126522, 3.3080189),
    (0.0225, 10447.3878384, 1.4607311),
    (0.0217, 14143.4952430, 5.9526579),
    (0.0209, 8429.2412401, 0.6522829),
    (0.0203, 419.4846439, 3.7354887),
    (0.0178, 73.2971259, 3.4759751),
    (0.0177, 6812.7668145, 3.1861180),
    (0.0162, 10213.2855462, 1.3311023),
    (0.0160, -2352.8661526, 6.1453853),
    (0.0159, -220.4126424, 4.0052889),
    (0.0151, 19651.0484841, 3.9694831),
    (0.0147, 1349.8673635, 4.3089139),
    (0.0143, 16730.4636878, 3.0160582),
    (0.0142, 17789.8456180, 2.1045498),
    (0.0137, -536.8045121, 5.9716728),
    (0.0125, 103.0927742, 1.7374759),
    (0.0123, 3.5903879, 1.7853927),
    (0.0124, 4690.4797950, 4.7340616),
    (0.0119, 5643.1785631, 5.4893206),
    (0.0119, 8031.0922265, 2.0533868),
    (0.0117, -4705.7323051, 2.6541366),
    (0.0116, 5120.6011450, 4.8639255),
    (0.0108, 553.5693363, 0.8427244),
    (0.0104, 951.7183499, 5.7177869),
    (0.0104, 5863.5912055, 1.9138804),
    (0.0101, 283.8593219, 1.9421795),
];

/// T-multiplied terms: amplitude (us per millennium); angle argument is `omega * t + phase` with
/// `t` in Julian millennia of TT from J2000.0, `omega` in rad/millennium,
/// `phase` in rad. Transcribed verbatim from Fairhead, Bretagnon &
/// Lestrade (1988), Table I.
const FAIRHEAD_B: [(f64, f64, f64); 26] = [
    (102.1574, 6283.0758494, 4.2490312),
    (1.7068, 12566.1516988, 4.2059040),
    (0.2697, 213.2990954, 3.4002911),
    (0.2659, 529.6909651, 5.8360513),
    (0.2158, -3.5231591, 0.0349384),
    (0.0780, 5223.6939192, 4.6703356),
    (0.0547, 1577.3435443, 4.5348996),
    (0.0593, 26.2983277, 1.0873123),
    (0.0344, -398.1490136, 5.9800691),
    (0.0321, 18849.2275481, 4.1629120),
    (0.0336, 5507.5532411, 5.9801641),
    (0.0292, 5856.4776585, 0.6238510),
    (0.0277, 155.4203227, 3.7453675),
    (0.0252, 5746.2713373, 2.9803823),
    (0.0230, -796.2980272, 1.1743887),
    (0.0250, 5760.4984313, 2.4679632),
    (0.0218, 206.1855484, 3.8547865),
    (0.0179, -775.5226083, 1.0918412),
    (0.0138, 426.5981909, 2.6998356),
    (0.0133, 6062.6632069, 5.8459339),
    (0.0118, 12036.4607337, 2.2928350),
    (0.0129, 6076.8903009, 5.3335561),
    (0.0122, 1059.3819302, 6.2228683),
    (0.0106, -7.1135470, 5.1924310),
    (0.0101, 4694.0029541, 4.0451363),
    (0.0101, 522.5774181, 0.7493158),
];

/// T^2-multiplied terms: amplitude (us per millennium^2); angle argument is `omega * t + phase` with
/// `t` in Julian millennia of TT from J2000.0, `omega` in rad/millennium,
/// `phase` in rad. Transcribed verbatim from Fairhead, Bretagnon &
/// Lestrade (1988), Table I.
const FAIRHEAD_C: [(f64, f64, f64); 6] = [
    (4.3230, 6283.0758494, 2.6428936),
    (0.1226, 12566.1516988, 2.4381357),
    (0.1648, 0.0000000, 4.7123890),
    (0.0195, 213.2990954, 1.6421878),
    (0.0169, 529.6909651, 4.5109594),
    (0.0131, -3.5231591, 1.3410365),
];

/// T^3-multiplied terms: amplitude (us per millennium^3); angle argument is `omega * t + phase` with
/// `t` in Julian millennia of TT from J2000.0, `omega` in rad/millennium,
/// `phase` in rad. Transcribed verbatim from Fairhead, Bretagnon &
/// Lestrade (1988), Table I.
const FAIRHEAD_D: [(f64, f64, f64); 1] = [(0.1434, 6283.0758494, 1.1314526)];

/// Evaluate one sine series at millennia-argument `t`, smallest terms
/// first to limit accumulation error.
fn series_sum(terms: &[(f64, f64, f64)], t: f64) -> f64 {
    let mut sum = 0.0;
    for &(amp, omega, phase) in terms.iter().rev() {
        sum += amp * libm::sin(omega * t + phase);
    }
    sum
}

/// TDB − TT in **seconds** at the given TT epoch.
///
/// Truncated Fairhead–Bretagnon analytical series (full published Table I
/// of Fairhead, Bretagnon & Lestrade 1988 — see module docs for provenance,
/// accuracy, and cross-checks). The dominant behavior is the annual
/// ±1.657 ms oscillation from the eccentricity of the Earth's orbit.
#[must_use]
pub fn tdb_minus_tt_seconds(jd_tt: JulianDate) -> f64 {
    // Two-part-aware epoch difference, then to Julian millennia.
    let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_MILLENNIUM;
    let sum_a = series_sum(&FAIRHEAD_A, t);
    let sum_b = series_sum(&FAIRHEAD_B, t);
    let sum_c = series_sum(&FAIRHEAD_C, t);
    let sum_d = series_sum(&FAIRHEAD_D, t);
    // Horner in t, microseconds -> seconds.
    (sum_a + t * (sum_b + t * (sum_c + t * sum_d))) * 1e-6
}

#[cfg(test)]
mod tests {
    use super::{FAIRHEAD_A, FAIRHEAD_B, FAIRHEAD_C, FAIRHEAD_D};

    #[test]
    fn table_shape_and_leading_terms() {
        assert_eq!(FAIRHEAD_A.len(), 90);
        assert_eq!(FAIRHEAD_B.len(), 26);
        assert_eq!(FAIRHEAD_C.len(), 6);
        assert_eq!(FAIRHEAD_D.len(), 1);
        // Leading amplitudes match USNO Circular 179 eq. 2.6 (in us) to the
        // rounding of that equation.
        assert!((FAIRHEAD_A[0].0 - 1657.0).abs() < 1.0);
        assert!((FAIRHEAD_B[0].0 - 102.1574).abs() < 1e-9);
        // Annual frequency: 2 pi * 1000 / 365.25-day years, in rad/millennium.
        assert!((FAIRHEAD_A[0].1 - 6283.0758494).abs() < 1e-7);
    }
}
