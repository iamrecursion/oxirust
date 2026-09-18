//! ΔT = TT − UT1: the Espenak–Meeus piecewise-polynomial approximation.
//!
//! # Source
//!
//! F. Espenak & J. Meeus, "Five Millennium Canon of Solar Eclipses:
//! −1999 to +3000", NASA Technical Publication TP-2009-214174 (2006),
//! Appendix; polynomial coefficients as tabulated on the companion NASA
//! Eclipse website page "Polynomial Expressions for Delta T"
//! (<https://eclipse.gsfc.nasa.gov/SEhelp/deltatpoly2004.html>). Each piece
//! below is transcribed verbatim (digit-for-digit) from that source; the
//! transcription is checked against the published polynomials by hand in
//! `tests/delta_t.rs`.
//!
//! `y` in every polynomial below is the (decimal) year; `u`/`t` are the
//! source's own per-piece rescalings of `y`, spelled out in each function's
//! doc comment.
//!
//! # Coverage
//!
//! The canon fits ΔT piecewise over −500 ≤ y < 2150 from historical eclipse
//! records, telescopic/occultation timings, and (from 1955 on) atomic-clock
//! comparisons; outside that range ([`delta_t_seconds`] for `y < -500` or
//! `y >= 2150`) the source itself falls back to a single long-term parabola
//! calibrated on the observed multi-millennial deceleration of the Earth's
//! rotation, and states explicitly that this extrapolation grows
//! increasingly uncertain far from the fitted interval — this module
//! reproduces that same fallback with no modification.
//!
//! # 1962+ caveat: prefer measured UT1 − UTC when available
//!
//! For 1962 onward, IERS routinely publishes the *measured*
//! `UT1 − UTC` (`finals2000A.all`, Bulletin A/B), from which
//! `ΔT = TT − UT1 = (TT − TAI) + (TAI − UTC) − (UT1 − UTC)` is exact to the
//! precision of the measurement — orders of magnitude better than this
//! polynomial fit, which is smoothed and, for recent years, itself an
//! extrapolation beyond the last observed leap-second epoch. Callers with
//! access to `finals2000A.all` (see `scripts/fetch_de441.sh`) **must** prefer
//! that measured series for 1962+; [`delta_t_seconds`] exists only as the
//! historical/predictive fallback (pre-1962 dates, or whenever measured
//! UT1 − UTC is unavailable), and inherits every accuracy caveat the source
//! states for its own fit and long-term extrapolation.
#![allow(clippy::unreadable_literal)]

/// ΔT = TT − UT1 in seconds, from the Espenak–Meeus polynomial
/// approximation (see the module documentation for source, coverage, and
/// the 1962+ measured-UT1 caveat).
///
/// `decimal_year` is the (proleptic, astronomical-numbering: 1 BCE = year
/// 0) decimal year, e.g. `2000.0` for 2000 Jan 1.0 or `2000.5` for
/// approximately 2000 Jul 2.
#[must_use]
pub fn delta_t_seconds(decimal_year: f64) -> f64 {
    let y = decimal_year;
    if y < -500.0 {
        long_term_parabola(y)
    } else if y < 500.0 {
        piece_neg500_to_500(y)
    } else if y < 1600.0 {
        piece_500_to_1600(y)
    } else if y < 1700.0 {
        piece_1600_to_1700(y)
    } else if y < 1800.0 {
        piece_1700_to_1800(y)
    } else if y < 1860.0 {
        piece_1800_to_1860(y)
    } else if y < 1900.0 {
        piece_1860_to_1900(y)
    } else if y < 1920.0 {
        piece_1900_to_1920(y)
    } else if y < 1941.0 {
        piece_1920_to_1941(y)
    } else if y < 1961.0 {
        piece_1941_to_1961(y)
    } else if y < 1986.0 {
        piece_1961_to_1986(y)
    } else if y < 2005.0 {
        piece_1986_to_2005(y)
    } else if y < 2050.0 {
        piece_2005_to_2050(y)
    } else if y < 2150.0 {
        piece_2050_to_2150(y)
    } else {
        long_term_parabola(y)
    }
}

/// `y < -500` or `y >= 2150`: the source's own long-term extrapolation
/// parabola, `u = (y − 1820) / 100`:
///
/// `ΔT = -20 + 32 u²`
fn long_term_parabola(y: f64) -> f64 {
    let u = (y - 1820.0) / 100.0;
    -20.0 + 32.0 * u * u
}

/// `-500 <= y < 500`, `u = y / 100`:
///
/// `ΔT = 10583.6 - 1014.41 u + 33.78311 u² - 5.952053 u³ - 0.1798452 u⁴
///        + 0.022174192 u⁵ + 0.0090316521 u⁶`
fn piece_neg500_to_500(y: f64) -> f64 {
    let u = y / 100.0;
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let u5 = u4 * u;
    let u6 = u5 * u;
    10583.6 - 1014.41 * u + 33.78311 * u2 - 5.952053 * u3 - 0.1798452 * u4
        + 0.022174192 * u5
        + 0.0090316521 * u6
}

/// `500 <= y < 1600`, `u = (y − 1000) / 100`:
///
/// `ΔT = 1574.2 - 556.01 u + 71.23472 u² + 0.319781 u³ - 0.8503463 u⁴
///        - 0.005050998 u⁵ + 0.0083572073 u⁶`
fn piece_500_to_1600(y: f64) -> f64 {
    let u = (y - 1000.0) / 100.0;
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u3 * u;
    let u5 = u4 * u;
    let u6 = u5 * u;
    1574.2 - 556.01 * u + 71.23472 * u2 + 0.319781 * u3 - 0.8503463 * u4 - 0.005050998 * u5
        + 0.0083572073 * u6
}

/// `1600 <= y < 1700`, `t = y − 1600`:
///
/// `ΔT = 120 - 0.9808 t - 0.01532 t² + t³ / 7129`
fn piece_1600_to_1700(y: f64) -> f64 {
    let t = y - 1600.0;
    let t2 = t * t;
    let t3 = t2 * t;
    120.0 - 0.9808 * t - 0.01532 * t2 + t3 / 7129.0
}

/// `1700 <= y < 1800`, `t = y − 1700`:
///
/// `ΔT = 8.83 + 0.1603 t - 0.0059285 t² + 0.00013336 t³ - t⁴ / 1174000`
fn piece_1700_to_1800(y: f64) -> f64 {
    let t = y - 1700.0;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    8.83 + 0.1603 * t - 0.0059285 * t2 + 0.00013336 * t3 - t4 / 1_174_000.0
}

/// `1800 <= y < 1860`, `t = y − 1800`:
///
/// `ΔT = 13.72 - 0.332447 t + 0.0068612 t² + 0.0041116 t³ - 0.00037436 t⁴
///        + 0.0000121272 t⁵ - 0.0000001699 t⁶ + 0.000000000875 t⁷`
fn piece_1800_to_1860(y: f64) -> f64 {
    let t = y - 1800.0;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let t6 = t5 * t;
    let t7 = t6 * t;
    13.72 - 0.332447 * t + 0.0068612 * t2 + 0.0041116 * t3 - 0.00037436 * t4 + 0.0000121272 * t5
        - 0.0000001699 * t6
        + 0.000000000875 * t7
}

/// `1860 <= y < 1900`, `t = y − 1860`:
///
/// `ΔT = 7.62 + 0.5737 t - 0.251754 t² + 0.01680668 t³ - 0.0004473624 t⁴
///        + t⁵ / 233174`
fn piece_1860_to_1900(y: f64) -> f64 {
    let t = y - 1860.0;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    7.62 + 0.5737 * t - 0.251754 * t2 + 0.01680668 * t3 - 0.0004473624 * t4 + t5 / 233_174.0
}

/// `1900 <= y < 1920`, `t = y − 1900`:
///
/// `ΔT = -2.79 + 1.494119 t - 0.0598939 t² + 0.0061966 t³ - 0.000197 t⁴`
fn piece_1900_to_1920(y: f64) -> f64 {
    let t = y - 1900.0;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    -2.79 + 1.494119 * t - 0.0598939 * t2 + 0.0061966 * t3 - 0.000197 * t4
}

/// `1920 <= y < 1941`, `t = y − 1920`:
///
/// `ΔT = 21.20 + 0.84493 t - 0.076100 t² + 0.0020936 t³`
fn piece_1920_to_1941(y: f64) -> f64 {
    let t = y - 1920.0;
    let t2 = t * t;
    let t3 = t2 * t;
    21.20 + 0.84493 * t - 0.076100 * t2 + 0.0020936 * t3
}

/// `1941 <= y < 1961`, `t = y − 1950`:
///
/// `ΔT = 29.07 + 0.407 t - t² / 233 + t³ / 2547`
fn piece_1941_to_1961(y: f64) -> f64 {
    let t = y - 1950.0;
    let t2 = t * t;
    let t3 = t2 * t;
    29.07 + 0.407 * t - t2 / 233.0 + t3 / 2547.0
}

/// `1961 <= y < 1986`, `t = y − 1975`:
///
/// `ΔT = 45.45 + 1.067 t - t² / 260 - t³ / 718`
fn piece_1961_to_1986(y: f64) -> f64 {
    let t = y - 1975.0;
    let t2 = t * t;
    let t3 = t2 * t;
    45.45 + 1.067 * t - t2 / 260.0 - t3 / 718.0
}

/// `1986 <= y < 2005`, `t = y − 2000`:
///
/// `ΔT = 63.86 + 0.3345 t - 0.060374 t² + 0.0017275 t³ + 0.000651814 t⁴
///        + 0.00002373599 t⁵`
fn piece_1986_to_2005(y: f64) -> f64 {
    let t = y - 2000.0;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    63.86 + 0.3345 * t - 0.060374 * t2 + 0.0017275 * t3 + 0.000651814 * t4 + 0.00002373599 * t5
}

/// `2005 <= y < 2050`, `t = y − 2000`:
///
/// `ΔT = 62.92 + 0.32217 t + 0.005589 t²`
fn piece_2005_to_2050(y: f64) -> f64 {
    let t = y - 2000.0;
    62.92 + 0.32217 * t + 0.005589 * t * t
}

/// `2050 <= y < 2150`:
///
/// `ΔT = -20 + 32 * ((y − 1820) / 100)² - 0.5628 * (2150 − y)`
fn piece_2050_to_2150(y: f64) -> f64 {
    let u = (y - 1820.0) / 100.0;
    -20.0 + 32.0 * u * u - 0.5628 * (2150.0 - y)
}
