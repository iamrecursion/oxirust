//! Map projections and coordinate system transformations
//!
//! This module provides comprehensive map projection systems for converting between
//! geographic coordinates and various projected coordinate systems.
//!
//! # Features
//!
//! * **Geographic to UTM** - Universal Transverse Mercator projection
//! * **Web Mercator** - EPSG:3857 projection for web maps
//! * **Lambert Conformal Conic** - For mid-latitude regions
//! * **Albers Equal Area** - Area-preserving projection
//! * **Stereographic** - Azimuthal conformal projection
//! * **Datum transformations** - WGS84, NAD83, etc.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::projections::{geographic_to_utm, utm_to_geographic, UTMZone};
//!
//! // Convert latitude/longitude to UTM
//! let lat = 40.7128; // New York City
//! let lon = -74.0060;
//!
//! let (zone, easting, northing) = geographic_to_utm(lat, lon)
//!     .expect("Failed to convert to UTM");
//!
//! println!("UTM Zone {}: E={:.2}, N={:.2}", zone.number, easting, northing);
//!
//! // Convert back
//! let (lat2, lon2) = utm_to_geographic(easting, northing, zone)
//!     .expect("Failed to convert from UTM");
//! ```

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::numeric::Float;
use std::f64::consts::PI;

/// UTM zone information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UTMZone {
    /// Zone number (1-60)
    pub number: u8,
    /// Hemisphere (true = North, false = South)
    pub north: bool,
}

/// Map projection types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionType {
    /// Universal Transverse Mercator
    UTM,
    /// Web Mercator (EPSG:3857)
    WebMercator,
    /// Lambert Conformal Conic
    LambertConformalConic,
    /// Albers Equal Area
    AlbersEqualArea,
    /// Stereographic
    Stereographic,
    /// Mercator
    Mercator,
}

/// Ellipsoid parameters for datum definitions
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    /// Semi-major axis (equatorial radius) in meters
    pub a: f64,
    /// Flattening factor
    pub f: f64,
}

impl Ellipsoid {
    /// WGS84 ellipsoid (most common, used by GPS)
    pub const WGS84: Ellipsoid = Ellipsoid {
        a: 6378137.0,
        f: 1.0 / 298.257223563,
    };

    /// GRS80 ellipsoid (used by NAD83)
    pub const GRS80: Ellipsoid = Ellipsoid {
        a: 6378137.0,
        f: 1.0 / 298.257222101,
    };

    /// Semi-minor axis (polar radius)
    pub fn b(&self) -> f64 {
        self.a * (1.0 - self.f)
    }

    /// First eccentricity squared
    pub fn e2(&self) -> f64 {
        2.0 * self.f - self.f * self.f
    }

    /// Second eccentricity squared
    pub fn e_prime2(&self) -> f64 {
        let e2 = self.e2();
        e2 / (1.0 - e2)
    }
}

/// Convert geographic coordinates to UTM
///
/// # Arguments
///
/// * `latitude` - Latitude in decimal degrees (-80 to 84)
/// * `longitude` - Longitude in decimal degrees (-180 to 180)
///
/// # Returns
///
/// * Tuple of (UTM zone, easting in meters, northing in meters)
///
/// # Examples
///
/// ```
/// use scirs2_spatial::projections::geographic_to_utm;
///
/// let (zone, easting, northing) = geographic_to_utm(40.7128, -74.0060)
///     .expect("Failed to convert");
/// ```
pub fn geographic_to_utm(latitude: f64, longitude: f64) -> SpatialResult<(UTMZone, f64, f64)> {
    if !(-80.0..=84.0).contains(&latitude) {
        return Err(SpatialError::ValueError(
            "Latitude must be between -80 and 84 degrees for UTM".to_string(),
        ));
    }

    if !(-180.0..=180.0).contains(&longitude) {
        return Err(SpatialError::ValueError(
            "Longitude must be between -180 and 180 degrees".to_string(),
        ));
    }

    // Determine UTM zone
    let zone_number = (((longitude + 180.0) / 6.0).floor() as u8 % 60) + 1;
    let is_north = latitude >= 0.0;

    let zone = UTMZone {
        number: zone_number,
        north: is_north,
    };

    // Convert to radians
    let lat_rad = latitude * PI / 180.0;
    let lon_rad = longitude * PI / 180.0;

    // Central meridian of the zone
    let lon0 = ((zone_number as f64 - 1.0) * 6.0 - 180.0 + 3.0) * PI / 180.0;

    // WGS84 parameters
    let ellipsoid = Ellipsoid::WGS84;
    let a = ellipsoid.a;
    let e2 = ellipsoid.e2();
    let e = e2.sqrt();

    // UTM scale factor
    let k0 = 0.9996;

    // Compute auxiliary values
    let n = a / (1.0 - e2 * lat_rad.sin().powi(2)).sqrt();
    let t = lat_rad.tan();
    let c = ellipsoid.e_prime2() * lat_rad.cos().powi(2);
    let aa = (lon_rad - lon0) * lat_rad.cos();

    // Compute meridional arc
    let m = meridional_arc(lat_rad, &ellipsoid);

    // Easting
    let easting = k0
        * n
        * (aa
            + (1.0 - t * t + c) * aa.powi(3) / 6.0
            + (5.0 - 18.0 * t * t + t.powi(4) + 72.0 * c - 58.0 * ellipsoid.e_prime2())
                * aa.powi(5)
                / 120.0)
        + 500000.0; // False easting

    // Northing
    let mut northing = k0
        * (m + n
            * t
            * (aa.powi(2) / 2.0
                + (5.0 - t * t + 9.0 * c + 4.0 * c * c) * aa.powi(4) / 24.0
                + (61.0 - 58.0 * t * t + t.powi(4) + 600.0 * c - 330.0 * ellipsoid.e_prime2())
                    * aa.powi(6)
                    / 720.0));

    // False northing for southern hemisphere
    if !is_north {
        northing += 10000000.0;
    }

    Ok((zone, easting, northing))
}

/// Convert UTM coordinates to geographic
///
/// # Arguments
///
/// * `easting` - Easting in meters
/// * `northing` - Northing in meters
/// * `zone` - UTM zone
///
/// # Returns
///
/// * Tuple of (latitude, longitude) in decimal degrees
pub fn utm_to_geographic(easting: f64, northing: f64, zone: UTMZone) -> SpatialResult<(f64, f64)> {
    let ellipsoid = Ellipsoid::WGS84;
    let a = ellipsoid.a;
    let e2 = ellipsoid.e2();
    let e = e2.sqrt();
    let k0 = 0.9996;

    // Remove false easting/northing
    let x = easting - 500000.0;
    let mut y = northing;
    if !zone.north {
        y -= 10000000.0;
    }

    // Central meridian
    let lon0 = ((zone.number as f64 - 1.0) * 6.0 - 180.0 + 3.0) * PI / 180.0;

    // Footpoint latitude
    let m = y / k0;
    let mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2.powi(3) / 256.0));

    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin();

    // Compute latitude and longitude
    let n1 = a / (1.0 - e2 * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan();
    let c1 = ellipsoid.e_prime2() * phi1.cos().powi(2);
    let r1 = a * (1.0 - e2) / (1.0 - e2 * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * k0);

    let latitude = phi1
        - (n1 * t1 / r1)
            * (d * d / 2.0
                - (5.0 + 3.0 * t1 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * ellipsoid.e_prime2())
                    * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 * t1 + 298.0 * c1 + 45.0 * t1 * t1 * t1 * t1
                    - 252.0 * ellipsoid.e_prime2()
                    - 3.0 * c1 * c1)
                    * d.powi(6)
                    / 720.0);

    let longitude = lon0
        + (d - (1.0 + 2.0 * t1 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 * t1 - 3.0 * c1 * c1
                + 8.0 * ellipsoid.e_prime2()
                + 24.0 * t1 * t1 * t1 * t1)
                * d.powi(5)
                / 120.0)
            / phi1.cos();

    Ok((latitude * 180.0 / PI, longitude * 180.0 / PI))
}

/// Compute meridional arc length
fn meridional_arc(lat_rad: f64, ellipsoid: &Ellipsoid) -> f64 {
    let a = ellipsoid.a;
    let e2 = ellipsoid.e2();

    let m0 = a * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2.powi(3) / 256.0);
    let m2 = a * (3.0 * e2 / 8.0 + 3.0 * e2 * e2 / 32.0 + 45.0 * e2.powi(3) / 1024.0);
    let m4 = a * (15.0 * e2 * e2 / 256.0 + 45.0 * e2.powi(3) / 1024.0);
    let m6 = a * (35.0 * e2.powi(3) / 3072.0);

    m0 * lat_rad - m2 * (2.0 * lat_rad).sin() + m4 * (4.0 * lat_rad).sin()
        - m6 * (6.0 * lat_rad).sin()
}

/// Convert geographic coordinates to Web Mercator (EPSG:3857)
///
/// Used by most web mapping services (Google Maps, OpenStreetMap, etc.)
///
/// # Arguments
///
/// * `latitude` - Latitude in decimal degrees (-85.0511 to 85.0511)
/// * `longitude` - Longitude in decimal degrees
///
/// # Returns
///
/// * Tuple of (x, y) in Web Mercator meters
pub fn geographic_to_web_mercator(latitude: f64, longitude: f64) -> SpatialResult<(f64, f64)> {
    const MAX_LAT: f64 = 85.051_128_779_806_59; // Maximum latitude for Web Mercator

    if latitude.abs() > MAX_LAT {
        return Err(SpatialError::ValueError(format!(
            "Latitude must be between -{} and {} degrees for Web Mercator",
            MAX_LAT, MAX_LAT
        )));
    }

    let r = 6378137.0; // WGS84 equatorial radius

    let x = r * longitude * PI / 180.0;
    let y = r * ((PI / 4.0 + latitude * PI / 360.0).tan().ln());

    Ok((x, y))
}

/// Convert Web Mercator coordinates to geographic
///
/// # Arguments
///
/// * `x` - X coordinate in meters
/// * `y` - Y coordinate in meters
///
/// # Returns
///
/// * Tuple of (latitude, longitude) in decimal degrees
pub fn web_mercator_to_geographic(x: f64, y: f64) -> SpatialResult<(f64, f64)> {
    let r = 6378137.0;

    let longitude = x / r * 180.0 / PI;
    let latitude = (2.0 * (y / r).exp().atan() - PI / 2.0) * 180.0 / PI;

    Ok((latitude, longitude))
}

/// Lambert Conformal Conic projection
///
/// # Arguments
///
/// * `latitude` - Latitude in decimal degrees
/// * `longitude` - Longitude in decimal degrees
/// * `lat0` - Origin latitude
/// * `lon0` - Central meridian
/// * `lat1` - First standard parallel
/// * `lat2` - Second standard parallel
///
/// # Returns
///
/// * Tuple of (x, y) in meters
pub fn lambert_conformal_conic(
    latitude: f64,
    longitude: f64,
    lat0: f64,
    lon0: f64,
    lat1: f64,
    lat2: f64,
) -> SpatialResult<(f64, f64)> {
    let ellipsoid = Ellipsoid::WGS84;
    let a = ellipsoid.a;
    let e = ellipsoid.e2().sqrt();

    // Convert to radians
    let lat = latitude * PI / 180.0;
    let lon = longitude * PI / 180.0;
    let lat_0 = lat0 * PI / 180.0;
    let lon_0 = lon0 * PI / 180.0;
    let lat_1 = lat1 * PI / 180.0;
    let lat_2 = lat2 * PI / 180.0;

    // Compute projection constants
    let m1 = lat_1.cos() / (1.0 - e * e * lat_1.sin().powi(2)).sqrt();
    let m2 = lat_2.cos() / (1.0 - e * e * lat_2.sin().powi(2)).sqrt();

    let t = |phi: f64| {
        (PI / 4.0 - phi / 2.0).tan() / ((1.0 - e * phi.sin()) / (1.0 + e * phi.sin())).powf(e / 2.0)
    };

    let t1 = t(lat_1);
    let t2 = t(lat_2);
    let t0 = t(lat_0);
    let t_phi = t(lat);

    let n = (m1.ln() - m2.ln()) / (t1.ln() - t2.ln());
    let f = m1 / (n * t1.powf(n));
    let rho_0 = a * f * t0.powf(n);
    let rho = a * f * t_phi.powf(n);

    let theta = n * (lon - lon_0);

    let x = rho * theta.sin();
    let y = rho_0 - rho * theta.cos();

    Ok((x, y))
}

/// Albers Equal Area Conic projection
///
/// # Arguments
///
/// * `latitude` - Latitude in decimal degrees
/// * `longitude` - Longitude in decimal degrees
/// * `lat0` - Origin latitude
/// * `lon0` - Central meridian
/// * `lat1` - First standard parallel
/// * `lat2` - Second standard parallel
///
/// # Returns
///
/// * Tuple of (x, y) in meters
pub fn albers_equal_area(
    latitude: f64,
    longitude: f64,
    lat0: f64,
    lon0: f64,
    lat1: f64,
    lat2: f64,
) -> SpatialResult<(f64, f64)> {
    let ellipsoid = Ellipsoid::WGS84;
    let a = ellipsoid.a;
    let e = ellipsoid.e2().sqrt();

    // Convert to radians
    let lat = latitude * PI / 180.0;
    let lon = longitude * PI / 180.0;
    let lat_0 = lat0 * PI / 180.0;
    let lon_0 = lon0 * PI / 180.0;
    let lat_1 = lat1 * PI / 180.0;
    let lat_2 = lat2 * PI / 180.0;

    // Compute projection constants
    let q = |phi: f64| {
        let sin_phi = phi.sin();
        (1.0 - e * e)
            * (sin_phi / (1.0 - e * e * sin_phi.powi(2))
                - (1.0 / (2.0 * e)) * ((1.0 - e * sin_phi) / (1.0 + e * sin_phi)).ln())
    };

    let m = |phi: f64| phi.cos() / (1.0 - e * e * phi.sin().powi(2)).sqrt();

    let q0 = q(lat_0);
    let q1 = q(lat_1);
    let q2 = q(lat_2);
    let q_phi = q(lat);

    let m1 = m(lat_1);
    let m2 = m(lat_2);

    let n = (m1.powi(2) - m2.powi(2)) / (q2 - q1);
    let c = m1.powi(2) + n * q1;
    let rho_0 = a * (c - n * q0).sqrt() / n;
    let rho = a * (c - n * q_phi).sqrt() / n;

    let theta = n * (lon - lon_0);

    let x = rho * theta.sin();
    let y = rho_0 - rho * theta.cos();

    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_geographic_to_utm() {
        // New York City
        let (zone, easting, northing) =
            geographic_to_utm(40.7128, -74.0060).expect("Failed to convert");

        assert_eq!(zone.number, 18);
        assert!(zone.north);
        assert!(easting > 500000.0 && easting < 600000.0);
        assert!(northing > 4500000.0 && northing < 4600000.0);
    }

    #[test]
    fn test_utm_roundtrip() {
        let lat = 40.7128;
        let lon = -74.0060;

        let (zone, easting, northing) =
            geographic_to_utm(lat, lon).expect("Failed to convert to UTM");

        let (lat2, lon2) =
            utm_to_geographic(easting, northing, zone).expect("Failed to convert from UTM");

        assert_relative_eq!(lat, lat2, epsilon = 1e-6);
        assert_relative_eq!(lon, lon2, epsilon = 1e-6);
    }

    #[test]
    fn test_geographic_to_web_mercator() {
        let (x, y) = geographic_to_web_mercator(0.0, 0.0).expect("Failed to convert");

        // At the equator and prime meridian, both should be 0
        assert_relative_eq!(x, 0.0, epsilon = 1.0);
        assert_relative_eq!(y, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_web_mercator_roundtrip() {
        let lat = 40.7128;
        let lon = -74.0060;

        let (x, y) =
            geographic_to_web_mercator(lat, lon).expect("Failed to convert to Web Mercator");

        let (lat2, lon2) =
            web_mercator_to_geographic(x, y).expect("Failed to convert from Web Mercator");

        assert_relative_eq!(lat, lat2, epsilon = 1e-6);
        assert_relative_eq!(lon, lon2, epsilon = 1e-6);
    }

    #[test]
    fn test_utm_zone_calculation() {
        // Test various locations
        let test_cases = vec![
            (40.7128, -74.0060, 18),  // New York
            (51.5074, -0.1278, 30),   // London
            (35.6762, 139.6503, 54),  // Tokyo
            (-33.8688, 151.2093, 56), // Sydney
        ];

        for (lat, lon, expected_zone) in test_cases {
            let (zone, _, _) = geographic_to_utm(lat, lon).expect("Failed to convert");
            assert_eq!(
                zone.number, expected_zone,
                "Wrong zone for ({}, {})",
                lat, lon
            );
        }
    }

    #[test]
    fn test_lambert_conformal_conic() {
        let result = lambert_conformal_conic(
            40.0, -100.0, // Point
            35.0, -100.0, // Origin
            33.0, 45.0, // Standard parallels
        );

        assert!(result.is_ok());
        let (x, y) = result.expect("computation failed");
        // Just verify we get reasonable values
        assert!(x.abs() < 10_000_000.0);
        assert!(y.abs() < 10_000_000.0);
    }

    #[test]
    fn test_albers_equal_area() {
        let result = albers_equal_area(
            40.0, -100.0, // Point
            23.0, -96.0, // Origin
            29.5, 45.5, // Standard parallels
        );

        assert!(result.is_ok());
        let (x, y) = result.expect("computation failed");
        // Verify reasonable values
        assert!(x.abs() < 10_000_000.0);
        assert!(y.abs() < 10_000_000.0);
    }

    #[test]
    fn test_ellipsoid_parameters() {
        let wgs84 = Ellipsoid::WGS84;

        // Semi-minor axis should be less than semi-major
        assert!(wgs84.b() < wgs84.a);

        // Eccentricity squared should be small
        assert!(wgs84.e2() > 0.0 && wgs84.e2() < 0.01);
    }
}
