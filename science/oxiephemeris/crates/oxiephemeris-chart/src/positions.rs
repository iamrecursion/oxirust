//! Shared body-position computation for the chart-family subcommands
//! (`chart`, `synastry`, `transit`, `progress`, `composite`).
//!
//! Every one of those commands needs the same thing: the ten
//! [`ALL_BODIES`] evaluated geocentric,
//! apparent, in the ecliptic-of-date frame, with daily speeds. This
//! module is the single place that pipeline lives, so the commands stay
//! thin and cannot drift apart in body set, frame, or option flags.
//!
//! The returned [`BodyPlace`] longitudes are the raw
//! *true-equinox-of-date* (tropical) values straight from the apparent
//! pipeline — no sidereal shift is applied here. A caller that wants
//! sidereal output subtracts the ayanamsha itself for display (see
//! `chart.rs`), exactly as before; keeping the raw value here means the
//! aspect/synastry engines, which must be sidereal-invariant, always see
//! a consistent tropical longitude.

use oxiephemeris_bodies::apparent::apparent;
use oxiephemeris_bodies::{Center, Frame, Options};
use oxiephemeris_core::time::JulianDate;
use oxiephemeris_de::DeFile;

use crate::body::ALL_BODIES;
use crate::error::ChartError;

/// One body's apparent ecliptic-of-date place plus daily speeds.
///
/// Longitudes/latitudes are radians; speeds radians per day; distance
/// AU; light-time days. `name` is the canonical display name from
/// [`ALL_BODIES`].
#[derive(Debug, Clone, Copy)]
pub struct BodyPlace {
    /// Canonical display name (e.g. `"Sun"`).
    pub name: &'static str,
    /// Apparent ecliptic longitude of date, radians.
    pub lon_rad: f64,
    /// Apparent ecliptic latitude of date, radians.
    pub lat_rad: f64,
    /// Longitude speed, radians per day (sign gives direct/retrograde).
    pub lon_speed_rad_per_day: f64,
    /// Latitude speed, radians per day.
    pub lat_speed_rad_per_day: f64,
    /// Geocentric distance, AU.
    pub distance_au: f64,
    /// One-way light time, days.
    pub light_time_days: f64,
}

/// The [`Options`] every chart-family command uses: geocentric, apparent
/// (aberration + light deflection), ecliptic-true-of-date, with speeds.
#[must_use]
pub fn chart_options() -> Options {
    let mut opts = Options::default();
    opts.center = Center::Geocentric;
    opts.frame = Frame::EclipticTrueOfDate;
    opts.aberration = true;
    opts.deflection = true;
    opts.with_speed = true;
    opts
}

/// Computes all ten [`ALL_BODIES`] at the TT
/// instant `jd_tt`, in the shared chart frame with daily speeds.
///
/// Taking `jd_tt` directly (rather than a full `ChartEpoch`) lets
/// secondary progression evaluate bodies at a progressed instant that
/// has no calendar date of its own.
///
/// # Errors
///
/// Propagates the apparent-place pipeline's errors
/// ([`ChartError::Bodies`]) for any body.
pub fn compute_bodies(de: &DeFile, jd_tt: JulianDate) -> Result<Vec<BodyPlace>, ChartError> {
    let opts = chart_options();
    let mut places = Vec::with_capacity(ALL_BODIES.len());
    for (target, name) in ALL_BODIES {
        let place = apparent(de, target, jd_tt, opts)?;
        // `with_speed: true` guarantees `Some`; `unwrap_or` (not
        // `unwrap`/`expect`) keeps this a documented, non-panicking
        // fallback rather than an assumption.
        let rates = place.rates.unwrap_or_default();
        places.push(BodyPlace {
            name,
            lon_rad: place.lon_rad,
            lat_rad: place.lat_rad,
            lon_speed_rad_per_day: rates.lon_rad_per_day,
            lat_speed_rad_per_day: rates.lat_rad_per_day,
            distance_au: place.r_au,
            light_time_days: place.light_time_days,
        });
    }
    Ok(places)
}
