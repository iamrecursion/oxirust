//! UTC leap seconds and the TAI ↔ TT ↔ UTC conversions.
//!
//! # Sources
//!
//! - Leap-second dates and offsets: IERS Bulletin C (announcements) as
//!   compiled in the IERS/BIPM "TAI − UTC" history table (also published
//!   as USNO `tai-utc.dat`). Only the post-1972 integer-second regime is
//!   supported; before 1972 UTC used rate offsets and fractional jumps,
//!   which is out of scope here.
//! - TT − TAI = 32.184 s exactly: defining offset of Terrestrial Time,
//!   IAU 1991 Resolution A4 (see also USNO Circular 179, §2.2).
//!
//! # Update procedure (documented maintenance contract)
//!
//! When IERS Bulletin C announces a new leap second effective at the end of
//! a June 30 or December 31:
//!
//! 1. Append one row `(jd, tai_minus_utc)` to [`LEAP_SECONDS`], where `jd`
//!    is the JD at 0h UTC of the **effective** date (the day *after* the
//!    inserted second: July 1 or January 1) and `tai_minus_utc` is the new
//!    cumulative offset in seconds.
//! 2. Add the matching `(year, month)` row to the self-consistency test in
//!    `tests/leap_seconds.rs`; it asserts every table JD equals
//!    `julday(Gregorian, y, m, 1, 0.0)` so a typo cannot slip in.
//!
//! No other code changes are required; the lookup scans the table.

use crate::time::jd::JulianDate;
use crate::CoreError;

/// TT − TAI in seconds (exact by definition; IAU 1991 Resolution A4).
pub const TT_MINUS_TAI_SECONDS: f64 = 32.184;

/// First instant covered by the leap-second table: 1972-01-01T00:00 UTC.
pub const UTC_LEAP_START_JD: f64 = 2_441_317.5;

/// Cumulative leap-second table: `(JD at 0h UTC of the effective date,
/// TAI − UTC in seconds)`, ascending. Source: IERS Bulletin C history (see
/// module docs, including the update procedure).
pub const LEAP_SECONDS: &[(f64, f64)] = &[
    (2_441_317.5, 10.0), // 1972-01-01
    (2_441_499.5, 11.0), // 1972-07-01
    (2_441_683.5, 12.0), // 1973-01-01
    (2_442_048.5, 13.0), // 1974-01-01
    (2_442_413.5, 14.0), // 1975-01-01
    (2_442_778.5, 15.0), // 1976-01-01
    (2_443_144.5, 16.0), // 1977-01-01
    (2_443_509.5, 17.0), // 1978-01-01
    (2_443_874.5, 18.0), // 1979-01-01
    (2_444_239.5, 19.0), // 1980-01-01
    (2_444_786.5, 20.0), // 1981-07-01
    (2_445_151.5, 21.0), // 1982-07-01
    (2_445_516.5, 22.0), // 1983-07-01
    (2_446_247.5, 23.0), // 1985-07-01
    (2_447_161.5, 24.0), // 1988-01-01
    (2_447_892.5, 25.0), // 1990-01-01
    (2_448_257.5, 26.0), // 1991-01-01
    (2_448_804.5, 27.0), // 1992-07-01
    (2_449_169.5, 28.0), // 1993-07-01
    (2_449_534.5, 29.0), // 1994-07-01
    (2_450_083.5, 30.0), // 1996-01-01
    (2_450_630.5, 31.0), // 1997-07-01
    (2_451_179.5, 32.0), // 1999-01-01
    (2_453_736.5, 33.0), // 2006-01-01
    (2_454_832.5, 34.0), // 2009-01-01
    (2_456_109.5, 35.0), // 2012-07-01
    (2_457_204.5, 36.0), // 2015-07-01
    (2_457_754.5, 37.0), // 2017-01-01
];

/// TAI − UTC in seconds at the given UTC epoch.
///
/// # Errors
///
/// - [`CoreError::InvalidInput`] for a non-finite JD.
/// - [`CoreError::DateOutOfRange`] for epochs before 1972-01-01 UTC (the
///   pre-1972 rubber-second UTC is not modeled).
pub fn tai_minus_utc(jd_utc: JulianDate) -> Result<f64, CoreError> {
    let value = jd_utc.value();
    if !value.is_finite() {
        return Err(CoreError::InvalidInput);
    }
    if value < UTC_LEAP_START_JD {
        return Err(CoreError::DateOutOfRange);
    }
    let mut offset = LEAP_SECONDS[0].1;
    for &(jd, tai_utc) in LEAP_SECONDS {
        if value >= jd {
            offset = tai_utc;
        } else {
            break;
        }
    }
    Ok(offset)
}

/// UTC → TAI.
///
/// # Errors
///
/// Propagates [`tai_minus_utc`] errors (non-finite input, pre-1972 epoch).
pub fn utc_to_tai(jd_utc: JulianDate) -> Result<JulianDate, CoreError> {
    let offset = tai_minus_utc(jd_utc)?;
    Ok(jd_utc.add_seconds(offset))
}

/// TAI → UTC, by inverse lookup: guess the offset by reading the TAI value
/// as if it were UTC, then recheck once with the implied UTC.
///
/// # Leap-second boundary convention
///
/// TAI instants that fall *inside* an inserted leap second (UTC
/// 23:59:60.xxx, which a JD cannot represent) are mapped **forward** using
/// the pre-jump offset, yielding a UTC reading in the first second of the
/// following day. Consequently `utc_to_tai(tai_to_utc(t)) ≠ t` inside that
/// one-second window; everywhere else the round trip is exact to rounding.
///
/// # Errors
///
/// - [`CoreError::InvalidInput`] for a non-finite JD.
/// - [`CoreError::DateOutOfRange`] for TAI epochs before
///   1972-01-01T00:00:10 TAI (i.e. UTC before the table starts).
pub fn tai_to_utc(jd_tai: JulianDate) -> Result<JulianDate, CoreError> {
    // First guess: TAI − UTC ≤ 37 s ≪ 6 months (the minimum table spacing),
    // so reading the TAI value as UTC selects at worst a neighboring row.
    let guess_offset = tai_minus_utc(jd_tai)?;
    let utc_guess = jd_tai.add_seconds(-guess_offset);
    // One recheck settles every case except the in-leap-second window,
    // which is handled by the documented forward-mapping convention.
    let offset = tai_minus_utc(utc_guess)?;
    Ok(jd_tai.add_seconds(-offset))
}

/// TAI → TT (adds the constant 32.184 s).
///
/// # Errors
///
/// None in practice (`Result` kept for a uniform time-scale API).
#[allow(clippy::unnecessary_wraps)]
pub fn tai_to_tt(jd_tai: JulianDate) -> Result<JulianDate, CoreError> {
    Ok(jd_tai.add_seconds(TT_MINUS_TAI_SECONDS))
}

/// TT → TAI (subtracts the constant 32.184 s).
///
/// # Errors
///
/// None in practice (`Result` kept for a uniform time-scale API).
#[allow(clippy::unnecessary_wraps)]
pub fn tt_to_tai(jd_tt: JulianDate) -> Result<JulianDate, CoreError> {
    Ok(jd_tt.add_seconds(-TT_MINUS_TAI_SECONDS))
}

/// UTC → TT.
///
/// # Errors
///
/// Propagates [`tai_minus_utc`] errors (non-finite input, pre-1972 epoch).
pub fn utc_to_tt(jd_utc: JulianDate) -> Result<JulianDate, CoreError> {
    Ok(utc_to_tai(jd_utc)?.add_seconds(TT_MINUS_TAI_SECONDS))
}

/// TT → UTC (see [`tai_to_utc`] for the leap-second boundary convention).
///
/// # Errors
///
/// Propagates [`tai_to_utc`] errors (non-finite input, out-of-range epoch).
pub fn tt_to_utc(jd_tt: JulianDate) -> Result<JulianDate, CoreError> {
    tai_to_utc(jd_tt.add_seconds(-TT_MINUS_TAI_SECONDS))
}
