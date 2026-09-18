//! Time scales: two-part Julian Date, calendars, TAI/TT/UTC/TDB.
//!
//! # Layout
//!
//! - [`jd`] — the two-part [`JulianDate`] type and its arithmetic.
//! - [`calendar`] — proleptic Gregorian/Julian calendar ↔ JD conversions
//!   ([`julday`] / [`revjul`]), BCE sign-correct (astronomical year
//!   numbering: year 0 = 1 BCE, year −1 = 2 BCE).
//! - [`leap`] — the embedded UTC leap-second table and the
//!   TAI ↔ TT ↔ UTC conversions.
//! - [`tdb`] — the TDB−TT analytical series (Fairhead–Bretagnon family).
//! - [`delta_t`] — the Espenak–Meeus ΔT (TT−UT1) polynomial approximation
//!   (historical/predictive fallback; prefer measured UT1−UTC for 1962+).
//!
//! # References
//!
//! - IERS Conventions (2010), IERS Technical Note 36, ch. 10 (time scales).
//! - Kaplan (2005), USNO Circular 179, §2 (time-scale relationships).
//! - Urban & Seidelmann (eds.), *Explanatory Supplement to the Astronomical
//!   Almanac*, 3rd ed. (2013), ch. 15 (Richards, "Calendars").
//! - Fairhead, Bretagnon & Lestrade (1988), IAU Symp. 128, 419–426 (TDB−TT).
//! - Espenak & Meeus, "Five Millennium Canon of Solar Eclipses", NASA
//!   TP-2009-214174 (2006), and the companion "Polynomial Expressions for
//!   Delta T" page (ΔT = TT−UT1).

pub mod calendar;
pub mod delta_t;
pub mod jd;
pub mod leap;
pub mod tdb;

pub use calendar::{days_in_month, is_leap_year, julday, revjul, Calendar, CalendarDate};
pub use delta_t::delta_t_seconds;
pub use jd::{JulianDate, J2000_JD, SECONDS_PER_DAY};
pub use leap::{
    tai_minus_utc, tai_to_tt, tai_to_utc, tt_to_tai, tt_to_utc, utc_to_tai, utc_to_tt,
    LEAP_SECONDS, TT_MINUS_TAI_SECONDS, UTC_LEAP_START_JD,
};
pub use tdb::tdb_minus_tt_seconds;
