//! House cusps and chart angles (Ascendant, Midheaven, Vertex, East
//! Point), with optional sidereal shift.
//!
//! `--alt` / observer height is not modelled (no dip-of-horizon term), so
//! the pure-spherical-trigonometry cusp/angle formulas take only the local
//! apparent sidereal time, the geographic latitude, and the true obliquity
//! of date.
//!
//! Sidereal output subtracts the ayanamsha referred to the *true* equinox
//! of date (mean-equinox ayanamsha + `Δψ`, see
//! [`crate::epoch::sidereal_offset_true_equinox_rad`]), because the
//! cusps/angles are themselves true-equinox-of-date quantities.

use oxiephemeris_astro::angles::{east_point, vertex, AnglesError};
use oxiephemeris_astro::ayanamsha::Ayanamsha;
use oxiephemeris_astro::houses::{cusps, HouseSystem, HousesError};
use oxiephemeris_core::angle::normalize_0_two_pi;

use crate::epoch::{sidereal_offset_true_equinox_rad, ChartEpoch};
use crate::error::ChartError;

/// Computed house cusps and chart angles. All longitudes are radians in
/// `[0, 2*pi)`, already sidereal-shifted if a `sidereal` kind was given.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_field_names)] // the `_rad` unit postfix is convention
pub struct HousesResult {
    /// Cusps 1-12, radians (see
    /// [`oxiephemeris_astro::houses::cusps`]'s indexing).
    pub cusps_rad: [f64; 12],
    /// Ascendant, radians (`== cusps_rad[0]`).
    pub ascendant_rad: f64,
    /// Midheaven, radians (`== cusps_rad[9]`).
    pub mc_rad: f64,
    /// Vertex, radians.
    pub vertex_rad: f64,
    /// East Point, radians.
    pub east_point_rad: f64,
}

/// A friendly [`ChartError::ChartGeometry`] message for a house-cusp
/// failure.
fn describe_houses_error(system: HouseSystem, err: HousesError) -> ChartError {
    let name = system.name();
    let msg = match err {
        HousesError::NonFiniteInput => format!(
            "cannot compute {name} house cusps: the sidereal time, latitude or \
             obliquity is not finite (check lat, lon, dut1 and the date)"
        ),
        HousesError::PolarLatitude => format!(
            "cannot compute {name} house cusps: latitude must be strictly between \
             -90 and 90 degrees"
        ),
        HousesError::Undefined => format!(
            "the {name} house system is undefined for this latitude and date (the \
             observer is inside the polar circle, where the quadrant systems' \
             defining semi-diurnal arcs do not exist for part of the day); try the \
             whole-sign or equal system instead"
        ),
        _ => format!("cannot compute {name} house cusps: {err:?}"),
    };
    ChartError::ChartGeometry(msg)
}

/// A friendly [`ChartError::ChartGeometry`] message for an angle failure.
fn describe_angle_error(what: &str, _err: AnglesError) -> ChartError {
    ChartError::ChartGeometry(format!(
        "cannot compute the {what}: the ecliptic exactly coincides with the reference \
         great circle for this latitude and date (a measure-zero degenerate configuration)"
    ))
}

/// Applies the sidereal shift (if any) and normalizes to `[0, 2*pi)`.
fn to_output_lon(lon_rad: f64, sidereal: Option<Ayanamsha>, t_tt: f64, dpsi_rad: f64) -> f64 {
    match sidereal {
        Some(kind) => {
            normalize_0_two_pi(lon_rad - sidereal_offset_true_equinox_rad(kind, t_tt, dpsi_rad))
        }
        None => normalize_0_two_pi(lon_rad),
    }
}

/// Computes cusps + chart angles for `system` at `epoch` and latitude
/// `phi_rad`, applying `sidereal` if given.
///
/// # Errors
///
/// [`ChartError::ChartGeometry`] for a polar-circle quadrant system,
/// non-finite input, or an exact ecliptic/reference-circle coincidence.
pub fn compute_houses(
    system: HouseSystem,
    epoch: &ChartEpoch,
    phi_rad: f64,
    sidereal: Option<Ayanamsha>,
) -> Result<HousesResult, ChartError> {
    let theta = epoch.last_rad;
    let eps = epoch.eps_true_rad;

    let raw_cusps =
        cusps(system, theta, phi_rad, eps).map_err(|e| describe_houses_error(system, e))?;
    let raw_vertex = vertex(theta, phi_rad, eps).map_err(|e| describe_angle_error("Vertex", e))?;
    let raw_east_point = east_point(theta, eps);

    let mut cusps_rad = [0.0_f64; 12];
    for (slot, raw) in cusps_rad.iter_mut().zip(raw_cusps) {
        *slot = to_output_lon(raw, sidereal, epoch.t_tt, epoch.dpsi_rad);
    }
    let vertex_rad = to_output_lon(raw_vertex, sidereal, epoch.t_tt, epoch.dpsi_rad);
    let east_point_rad = to_output_lon(raw_east_point, sidereal, epoch.t_tt, epoch.dpsi_rad);

    Ok(HousesResult {
        cusps_rad,
        ascendant_rad: cusps_rad[0],
        mc_rad: cusps_rad[9],
        vertex_rad,
        east_point_rad,
    })
}
