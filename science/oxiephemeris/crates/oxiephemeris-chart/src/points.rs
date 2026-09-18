//! Chart points and cross-aspect collection for the two-chart / time
//! charts (synastry, transit, progression, composite).
//!
//! A [`ChartPoint`] is one named ecliptic position with a daily speed —
//! the currency the cross-aspect engine ([`oxiephemeris_astro::synastry`])
//! consumes. [`natal_points`] builds the twelve-point set (ten planets +
//! Ascendant + Midheaven); [`transit_points`] builds the ten transiting
//! planets.
//!
//! These charts work in the **tropical** zodiac only: cross-aspects are
//! sidereal-invariant (a common ayanamsha cancels in every pairwise
//! separation), so only the *displayed* sign would differ under a sidereal
//! shift, and the natal chart already offers full sidereal output.

use oxiephemeris_astro::aspects::{find_aspect, AspectHit, OrbPolicy};
use oxiephemeris_astro::houses::HouseSystem;
use oxiephemeris_astro::synastry::for_each_cross_aspect;
use oxiephemeris_core::angle::normalize_0_two_pi;
use oxiephemeris_core::time::JulianDate;
use oxiephemeris_de::DeFile;

use crate::epoch::ChartEpoch;
use crate::error::ChartError;
use crate::houses::compute_houses;
use crate::positions::compute_bodies;

/// One named chart position and its daily longitude speed (radians,
/// radians/day). Angles (Ascendant/Midheaven) carry a zero speed.
#[derive(Debug, Clone, Copy)]
pub struct ChartPoint {
    /// Display name (e.g. `"Sun"`, `"ASC"`, `"MC"`).
    pub name: &'static str,
    /// Tropical ecliptic longitude of date, radians in `[0, 2*pi)`.
    pub lon_rad: f64,
    /// Longitude speed, radians per day.
    pub speed_rad_per_day: f64,
}

/// The twelve natal points — ten planets plus Ascendant and Midheaven —
/// for a resolved birth epoch and geographic latitude.
///
/// # Errors
///
/// Propagates the apparent-place and house-geometry errors.
pub fn natal_points(
    de: &DeFile,
    epoch: &ChartEpoch,
    phi_rad: f64,
    system: HouseSystem,
) -> Result<Vec<ChartPoint>, ChartError> {
    let mut points = transit_points(de, epoch)?;
    // Angles need no sidereal handling here (tropical only).
    let houses = compute_houses(system, epoch, phi_rad, None)?;
    points.push(ChartPoint {
        name: "ASC",
        lon_rad: normalize_0_two_pi(houses.ascendant_rad),
        speed_rad_per_day: 0.0,
    });
    points.push(ChartPoint {
        name: "MC",
        lon_rad: normalize_0_two_pi(houses.mc_rad),
        speed_rad_per_day: 0.0,
    });
    Ok(points)
}

/// The ten planets at an epoch's TT instant, as [`ChartPoint`]s.
///
/// # Errors
///
/// Propagates the apparent-place pipeline's errors.
pub fn transit_points(de: &DeFile, epoch: &ChartEpoch) -> Result<Vec<ChartPoint>, ChartError> {
    planet_points_at(de, epoch.jd_tt)
}

/// The ten planets at an arbitrary TT instant, as [`ChartPoint`]s. Used by
/// secondary progression, whose progressed instant has no calendar date.
///
/// # Errors
///
/// Propagates the apparent-place pipeline's errors.
pub fn planet_points_at(de: &DeFile, jd_tt: JulianDate) -> Result<Vec<ChartPoint>, ChartError> {
    let places = compute_bodies(de, jd_tt)?;
    Ok(places
        .iter()
        .map(|p| ChartPoint {
            name: p.name,
            lon_rad: normalize_0_two_pi(p.lon_rad),
            speed_rad_per_day: p.lon_speed_rad_per_day,
        })
        .collect())
}

/// The cross-aspects between two point sets, as `(a_name, b_name, hit)`
/// triples — the shape both the RDF layer and the renderers build from.
#[must_use]
pub fn collect_cross_aspects(
    a: &[ChartPoint],
    b: &[ChartPoint],
) -> Vec<(&'static str, &'static str, AspectHit)> {
    let a_pairs: Vec<(f64, f64)> = a.iter().map(|p| (p.lon_rad, p.speed_rad_per_day)).collect();
    let b_pairs: Vec<(f64, f64)> = b.iter().map(|p| (p.lon_rad, p.speed_rad_per_day)).collect();
    let mut out = Vec::new();
    for_each_cross_aspect(&a_pairs, &b_pairs, &OrbPolicy::default(), |cross| {
        out.push((a[cross.index_a].name, b[cross.index_b].name, cross.hit));
    });
    out
}

/// The aspects among one chart's own points (each unordered pair `i < j`
/// tested once).
#[must_use]
pub fn collect_internal_aspects(
    points: &[ChartPoint],
) -> Vec<(&'static str, &'static str, AspectHit)> {
    let policy = OrbPolicy::default();
    let mut out = Vec::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let (a, b) = (&points[i], &points[j]);
            if let Some(hit) = find_aspect(
                a.lon_rad,
                a.speed_rad_per_day,
                b.lon_rad,
                b.speed_rad_per_day,
                &policy,
            ) {
                out.push((a.name, b.name, hit));
            }
        }
    }
    out
}
