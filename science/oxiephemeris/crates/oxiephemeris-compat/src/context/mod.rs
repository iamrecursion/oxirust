//! [`Context`]: the instance-local replacement for Swiss Ephemeris'
//! process-global state (loaded ephemeris, topocentric observer,
//! sidereal mode). See the crate docs for the design rationale.
//!
//! This module holds the struct itself, the calendar functions
//! (`julday`/`revjul`, which need no context state at all), `sidtime`,
//! `set_topo`/`set_topo_with_eop`, `set_sid_mode`, and `get_ayanamsa`.
//! `calc`/`calc_ut` live in [`crate::context::calc`] and
//! `houses`/`houses_ex` in [`crate::houses`] (both are `impl Context`
//! blocks in their own files, kept under the 2000-line-per-file policy).

pub mod calc;

use oxiephemeris_bodies::topocentric::{Eop, Observer, TopocentricObserver};
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_core::time::{
    delta_t_seconds, is_leap_year, julday as core_julday, revjul as core_revjul, Calendar,
    CalendarDate, JulianDate, J2000_JD,
};
use oxiephemeris_core::CoreError;
use oxiephemeris_de::DeFile;

use crate::flags::{SE_GREG_CAL, SE_JUL_CAL};
use crate::{CompatError, SiderealMode};

/// Days per Julian century (same convention throughout this workspace;
/// see e.g. `oxiephemeris_bodies::frames`' "Time argument" section).
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// Radians per full turn (`2*pi`), for the `sidtime` radians-to-hours
/// conversion (`24 h = 2*pi rad`).
const TWO_PI: f64 = 2.0 * core::f64::consts::PI;

/// Instance-local replacement for Swiss Ephemeris' process-global state:
/// the loaded DE ephemeris, the topocentric observer, and the sidereal
/// (ayanamsha) mode. See the crate docs for why this crate uses an
/// explicit value instead of `swe_set_*`-style global mutation.
///
/// `'a` is the lifetime of the DE file's backing byte slice (the same
/// lifetime [`DeFile`] itself borrows).
#[derive(Debug, Clone)]
pub struct Context<'a> {
    pub(crate) de: Option<DeFile<'a>>,
    pub(crate) topo: Option<TopocentricObserver>,
    pub(crate) sidereal: Option<SiderealMode>,
}

impl Context<'static> {
    /// A fresh context: no ephemeris, no topocentric observer, no
    /// sidereal mode. The calendar entry points ([`Context::julday`],
    /// [`Context::revjul`]) work immediately; every ephemeris-dependent
    /// entry point returns [`CompatError::NoEphemeris`] until
    /// [`Context::with_ephemeris`] / [`Context::set_ephemeris`] attaches
    /// one.
    #[must_use]
    pub fn new() -> Self {
        Self {
            de: None,
            topo: None,
            sidereal: None,
        }
    }
}

impl Default for Context<'static> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Context<'a> {
    /// Builds a context around an already-parsed DE file
    /// ([`DeFile::parse`]).
    #[must_use]
    pub fn with_ephemeris(de: DeFile<'a>) -> Self {
        Self {
            de: Some(de),
            topo: None,
            sidereal: None,
        }
    }

    /// Attaches (or replaces) the DE ephemeris.
    pub fn set_ephemeris(&mut self, de: DeFile<'a>) {
        self.de = Some(de);
    }

    /// Whether a DE ephemeris is currently attached.
    #[must_use]
    pub fn has_ephemeris(&self) -> bool {
        self.de.is_some()
    }

    /// The attached ephemeris, or [`CompatError::NoEphemeris`].
    pub(crate) fn ephemeris(&self) -> Result<&DeFile<'a>, CompatError> {
        self.de.as_ref().ok_or(CompatError::NoEphemeris)
    }

    /// The current sidereal mode, defaulting to
    /// [`SiderealMode::FaganBradley`] when [`Context::set_sid_mode`] was
    /// never called — mirroring SE's own documented default.
    pub(crate) fn sidereal_mode(&self) -> SiderealMode {
        self.sidereal.unwrap_or(SiderealMode::FaganBradley)
    }

    /// Sets the topocentric observer, mirroring
    /// `swe_set_topo(geo_lon, geo_lat, altitude_above_sea)`: same
    /// parameter order and units (degrees, degrees, meters; longitude
    /// positive east). Earth-orientation parameters default to zero (see
    /// [`Eop`]'s accuracy-impact docs); use
    /// [`Context::set_topo_with_eop`] to supply measured values.
    pub fn set_topo(&mut self, lon_deg: f64, lat_deg: f64, alt_m: f64) {
        self.set_topo_with_eop(lon_deg, lat_deg, alt_m, Eop::default());
    }

    /// As [`Context::set_topo`], but also supplying Earth-orientation
    /// parameters (`UT1-UTC`, polar motion). This is beyond SE's own
    /// documented `swe_set_topo` signature — a small OxiEphemeris-specific
    /// extension for accuracy (SE instead consults its own internal EOP
    /// tables when present on disk; this crate has no such file format,
    /// so the caller supplies the numbers directly).
    pub fn set_topo_with_eop(&mut self, lon_deg: f64, lat_deg: f64, alt_m: f64, eop: Eop) {
        self.topo = Some(TopocentricObserver::new(
            Observer {
                latitude_deg: lat_deg,
                longitude_deg: lon_deg,
                height_m: alt_m,
            },
            eop,
        ));
    }

    /// Sets the sidereal (ayanamsha) mode, mirroring
    /// `swe_set_sid_mode(sid_mode, t0, ayan_t0)` — see [`SiderealMode`]
    /// for why `t0`/`ayan_t0` are folded into its `User` variant.
    pub fn set_sid_mode(&mut self, mode: SiderealMode) {
        self.sidereal = Some(mode);
    }

    /// The ayanamsha (sidereal-zodiac offset) at TT epoch `jd_et`, in
    /// **degrees**, mirroring `swe_get_ayanamsa(jd_et)`. Uses the mode
    /// set by [`Context::set_sid_mode`], or the documented Fagan/Bradley
    /// default if it was never called.
    ///
    /// # Errors
    ///
    /// [`CompatError::Core`] (wrapping [`CoreError::InvalidInput`]) if
    /// `jd_et` is not finite.
    pub fn get_ayanamsa(&self, jd_et: f64) -> Result<f64, CompatError> {
        if !jd_et.is_finite() {
            return Err(CompatError::Core(CoreError::InvalidInput));
        }
        let t = (jd_et - J2000_JD) / DAYS_PER_CENTURY;
        let kind = self.sidereal_mode().to_ayanamsha();
        let rad = oxiephemeris_astro::ayanamsha::ayanamsha_rad(kind, t);
        Ok(rad * RAD2DEG)
    }

    /// Calendar date and time of day → Julian Date, mirroring
    /// `swe_julday(year, month, day, hour, gregflag)`. Returns the plain
    /// `f64` Julian Date (SE's own convention); see
    /// [`oxiephemeris_core::time::julday`] directly for this workspace's
    /// extended-precision two-part [`JulianDate`].
    ///
    /// Needs no [`Context`] state — callable without an instance, e.g.
    /// `Context::julday(2000, 1, 1, 12.0, SE_GREG_CAL)`.
    ///
    /// # Errors
    ///
    /// [`CompatError::Core`] for an invalid `calflag`, an out-of-range
    /// `month`/`day` for the given calendar, or a non-finite `hour`.
    pub fn julday(
        year: i32,
        month: i32,
        day: i32,
        hour: f64,
        calflag: i32,
    ) -> Result<f64, CompatError> {
        let calendar = calendar_from_flag(calflag)?;
        let month = u8::try_from(month).map_err(|_| CompatError::Core(CoreError::InvalidInput))?;
        let day = u8::try_from(day).map_err(|_| CompatError::Core(CoreError::InvalidInput))?;
        let jd = core_julday(calendar, year, month, day, hour)?;
        Ok(jd.value())
    }

    /// Julian Date → calendar date and time of day, mirroring
    /// `swe_revjul(jd, gregflag)`. SE returns the four fields through
    /// output parameters (`*jyear`, `*jmonth`, `*jday`, `*jut`); this
    /// crate returns them as `(`[`CalendarDate`]`, hour)` instead.
    ///
    /// Needs no [`Context`] state — callable without an instance.
    ///
    /// # Errors
    ///
    /// [`CompatError::Core`] for an invalid `calflag`, a non-finite
    /// `jd`, or `|jd|` too large to represent as a calendar year (see
    /// [`oxiephemeris_core::time::revjul`]).
    pub fn revjul(jd: f64, calflag: i32) -> Result<(CalendarDate, f64), CompatError> {
        let calendar = calendar_from_flag(calflag)?;
        let (date, hour) = core_revjul(JulianDate::from_f64(jd), calendar)?;
        Ok((date, hour))
    }

    /// Greenwich apparent sidereal time at UT1 epoch `jd_ut`, in
    /// **hours** `[0, 24)`, mirroring `swe_sidtime(jd_ut)`. `jd_ut` is
    /// treated as UT1 directly (`dUT1 = 0`), except that when
    /// [`Context::set_topo`]/[`Context::set_topo_with_eop`] was called,
    /// its [`Eop::dut1_s`] is applied — for consistency with
    /// [`Context::calc_ut`]'s topocentric Earth-rotation chain, which
    /// uses the same correction.
    ///
    /// # Errors
    ///
    /// [`CompatError::Core`] if `jd_ut` is non-finite or too far out of
    /// range for the internal ΔT-based TT conversion (see the crate
    /// docs' "Different time-scale models" note).
    #[allow(clippy::similar_names)] // jd_ut / jd_tt are the domain-standard names
    pub fn sidtime(&self, jd_ut: f64) -> Result<f64, CompatError> {
        let dut1_s = self.topo.as_ref().map_or(0.0, |t| t.eop.dut1_s);
        let jd_ut1 = JulianDate::from_f64(jd_ut).add_seconds(dut1_s);
        let jd_tt = jd_tt_from_ut(jd_ut)?;
        let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY;
        let gast_rad = oxiephemeris_bodies::sidereal::gast_iau2006(jd_ut1, t);
        Ok(gast_rad * (24.0 / TWO_PI))
    }
}

/// Maps `SE_JUL_CAL`/`SE_GREG_CAL` to [`Calendar`].
fn calendar_from_flag(calflag: i32) -> Result<Calendar, CompatError> {
    match calflag {
        SE_JUL_CAL => Ok(Calendar::Julian),
        SE_GREG_CAL => Ok(Calendar::Gregorian),
        _ => Err(CompatError::Core(CoreError::InvalidInput)),
    }
}

/// The (Gregorian) decimal year of a UT Julian Date, for
/// [`delta_t_seconds`]. A day-granularity approximation (day-of-year /
/// days-in-year) is far finer than ΔT's own smoothness scale (seconds
/// per year), so this loses nothing the polynomial fit itself carries.
pub(crate) fn decimal_year_of(jd_ut: f64) -> Result<f64, CompatError> {
    let (date, _) = core_revjul(JulianDate::from_f64(jd_ut), Calendar::Gregorian)?;
    let jan1 = core_julday(Calendar::Gregorian, date.year, 1, 1, 0.0)?;
    let days_in_this_year = if is_leap_year(Calendar::Gregorian, date.year) {
        366.0
    } else {
        365.0
    };
    let day_of_year = jd_ut - jan1.value();
    Ok(f64::from(date.year) + day_of_year / days_in_this_year)
}

/// UT → TT via `oxiephemeris_core`'s Espenak–Meeus ΔT polynomial (see the
/// crate docs' "Different time-scale models" note — deliberately not the
/// leap-second table, so pre-1972 epochs work).
pub(crate) fn jd_tt_from_ut(jd_ut: f64) -> Result<JulianDate, CompatError> {
    let decimal_year = decimal_year_of(jd_ut)?;
    let dt_s = delta_t_seconds(decimal_year);
    Ok(JulianDate::from_f64(jd_ut).add_seconds(dt_s))
}

#[cfg(test)]
mod tests {
    use super::{calendar_from_flag, decimal_year_of, Context};
    use crate::flags::{SE_GREG_CAL, SE_JUL_CAL};
    use crate::CompatError;
    use oxiephemeris_core::time::Calendar;

    #[test]
    fn default_context_has_no_ephemeris() {
        let ctx = Context::new();
        assert!(!ctx.has_ephemeris());
        assert!(matches!(ctx.ephemeris(), Err(CompatError::NoEphemeris)));
    }

    #[test]
    fn calendar_flag_mapping() {
        assert_eq!(calendar_from_flag(SE_JUL_CAL), Ok(Calendar::Julian));
        assert_eq!(calendar_from_flag(SE_GREG_CAL), Ok(Calendar::Gregorian));
        assert!(calendar_from_flag(2).is_err());
        assert!(calendar_from_flag(-1).is_err());
    }

    #[test]
    fn julday_matches_core_j2000() {
        // 2000-01-01 12:00 TT is JD 2451545.0 (J2000.0 epoch).
        let jd = Context::julday(2000, 1, 1, 12.0, SE_GREG_CAL)
            .unwrap_or_else(|e| panic!("unexpected error: {e}"));
        assert!((jd - 2_451_545.0).abs() < 1e-9);
    }

    #[test]
    fn julday_rejects_invalid_month() {
        assert!(Context::julday(2000, 13, 1, 0.0, SE_GREG_CAL).is_err());
        assert!(Context::julday(2000, 2, 30, 0.0, SE_GREG_CAL).is_err());
    }

    #[test]
    fn julday_revjul_round_trip_bce_and_julian_calendar() {
        for (calflag, year, month, day, hour) in [
            (SE_GREG_CAL, 2024, 3, 15, 6.25),
            (SE_JUL_CAL, 1000, 2, 29, 0.0),
            (SE_GREG_CAL, -100, 6, 1, 18.0), // 101 BCE
            (SE_JUL_CAL, -500, 12, 31, 23.999),
        ] {
            let jd = Context::julday(year, month, day, hour, calflag)
                .unwrap_or_else(|e| panic!("julday failed for {year}-{month}-{day}: {e}"));
            let (date, out_hour) =
                Context::revjul(jd, calflag).unwrap_or_else(|e| panic!("revjul failed: {e}"));
            assert_eq!(date.year, year);
            assert_eq!(i32::from(date.month), month);
            assert_eq!(i32::from(date.day), day);
            assert!(
                (out_hour - hour).abs() < 1e-6,
                "hour round trip: {out_hour} vs {hour}"
            );
        }
    }

    #[test]
    fn decimal_year_of_matches_known_points() {
        assert!((decimal_year_of(2_451_545.0).unwrap_or(f64::NAN) - 2000.0).abs() < 0.01);
        // 2000 was a leap year: July 2 is close to the half-year mark.
        let mid = decimal_year_of(2_451_545.0 + 182.0).unwrap_or(f64::NAN);
        assert!((mid - 2000.5).abs() < 0.01, "mid = {mid}");
    }

    #[test]
    fn sidtime_is_in_range_and_pre_1972_works() {
        let ctx = Context::new();
        // A pre-1972 date: the leap-second table would reject this, but
        // sidtime uses the ΔT model instead and must succeed.
        let jd_ut =
            Context::julday(1950, 6, 15, 0.0, SE_GREG_CAL).unwrap_or_else(|e| panic!("{e}"));
        let hours = ctx
            .sidtime(jd_ut)
            .unwrap_or_else(|e| panic!("sidtime failed pre-1972: {e}"));
        assert!(
            (0.0..24.0).contains(&hours),
            "sidtime out of range: {hours}"
        );
    }

    #[test]
    fn get_ayanamsa_default_is_fagan_bradley() {
        let ctx = Context::new();
        let default_value = ctx
            .get_ayanamsa(2_451_545.0)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut ctx2 = Context::new();
        ctx2.set_sid_mode(crate::SiderealMode::FaganBradley);
        let explicit_value = ctx2
            .get_ayanamsa(2_451_545.0)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!((default_value - explicit_value).abs() < 1e-12);
    }

    #[test]
    fn get_ayanamsa_rejects_non_finite() {
        let ctx = Context::new();
        assert!(ctx.get_ayanamsa(f64::NAN).is_err());
        assert!(ctx.get_ayanamsa(f64::INFINITY).is_err());
    }
}
