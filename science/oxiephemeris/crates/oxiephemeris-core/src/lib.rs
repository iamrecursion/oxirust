//! `OxiEphemeris` core: time scales, calendars, angles, Chebyshev primitives.
//!
//! `no_std` by design (enable the `std` feature for `std::error::Error`
//! support). All algorithms are implemented from published standards and
//! papers — see CONTRIBUTING.md for the clean-room provenance statement.
//!
//! # Modules
//!
//! - [`time`] — two-part Julian Date arithmetic ([`time::JulianDate`]),
//!   proleptic Gregorian/Julian calendar conversions ([`time::julday`],
//!   [`time::revjul`]), the UTC leap-second table and TAI/TT/UTC
//!   conversions, and the TDB−TT analytical series
//!   ([`time::tdb_minus_tt_seconds`]).
//! - [`angle`] — angle unit constants and normalization helpers.
//!
//! # References
//!
//! - Urban & Seidelmann (eds.), *Explanatory Supplement to the Astronomical
//!   Almanac*, 3rd ed. (2013), ch. 15 (Richards, "Calendars").
//! - IERS Conventions (2010), IERS Technical Note 36, ch. 10 (time scales).
//! - Fairhead, Bretagnon & Lestrade (1988), IAU Symp. 128, 419–426 (TDB−TT).
//! - Kaplan (2005), USNO Circular 179 (time-scale relationships).
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod angle;
pub mod time;

/// Crate-level error type.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreError {
    /// A date outside the supported calendar range.
    DateOutOfRange,
    /// Invalid input to a conversion.
    InvalidInput,
}

impl core::fmt::Display for CoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DateOutOfRange => f.write_str("date out of the supported range"),
            Self::InvalidInput => f.write_str("invalid input to a conversion"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CoreError {}
