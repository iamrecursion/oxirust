//! GPS coordinate extraction, conversion, and display for geotagged media.
//!
//! This module provides tools for working with geographic metadata embedded
//! in media files (primarily EXIF GPS data in photos and XMP geotags).
//!
//! # Features
//!
//! - **Coordinate parsing**: Extract GPS coordinates from EXIF/XMP metadata
//! - **DMS/Decimal conversion**: Convert between degrees-minutes-seconds and decimal degrees
//! - **Distance calculation**: Haversine distance between two coordinates
//! - **Display formatting**: Human-readable coordinate strings
//! - **Bounding box**: Geographic region queries
//! - **Metadata integration**: Read/write coordinates from/to `Metadata` containers
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::geotag::{GeoCoordinate, DmsCoordinate};
//!
//! // Create from decimal degrees
//! let coord = GeoCoordinate::new(40.7128, -74.0060);
//! assert!((coord.latitude() - 40.7128).abs() < 0.0001);
//!
//! // Convert to DMS
//! let dms = coord.to_dms();
//! assert_eq!(dms.lat_degrees, 40);
//! assert_eq!(dms.lat_direction, 'N');
//!
//! // Display
//! let display = coord.to_display_string();
//! assert!(display.contains("40"));
//! ```

use crate::{Error, Metadata, MetadataValue};
use std::f64::consts::PI;

/// Earth's mean radius in kilometers.
const EARTH_RADIUS_KM: f64 = 6371.0;

/// A geographic coordinate in decimal degrees (WGS84).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoordinate {
    /// Latitude in decimal degrees (-90 to +90, positive = North).
    latitude: f64,
    /// Longitude in decimal degrees (-180 to +180, positive = East).
    longitude: f64,
    /// Altitude in meters above sea level (optional).
    altitude: Option<f64>,
}

impl GeoCoordinate {
    /// Create a new coordinate from decimal degrees.
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude: None,
        }
    }

    /// Create with altitude.
    pub fn with_altitude(mut self, altitude: f64) -> Self {
        self.altitude = Some(altitude);
        self
    }

    /// Latitude in decimal degrees.
    pub fn latitude(&self) -> f64 {
        self.latitude
    }

    /// Longitude in decimal degrees.
    pub fn longitude(&self) -> f64 {
        self.longitude
    }

    /// Altitude in meters (if known).
    pub fn altitude(&self) -> Option<f64> {
        self.altitude
    }

    /// Whether the location is in the Northern hemisphere.
    pub fn is_north(&self) -> bool {
        self.latitude >= 0.0
    }

    /// Whether the location is in the Eastern hemisphere.
    pub fn is_east(&self) -> bool {
        self.longitude >= 0.0
    }

    /// Convert to DMS (Degrees-Minutes-Seconds) representation.
    pub fn to_dms(&self) -> DmsCoordinate {
        let (lat_d, lat_m, lat_s) = decimal_to_dms(self.latitude.abs());
        let (lon_d, lon_m, lon_s) = decimal_to_dms(self.longitude.abs());

        DmsCoordinate {
            lat_degrees: lat_d,
            lat_minutes: lat_m,
            lat_seconds: lat_s,
            lat_direction: if self.latitude >= 0.0 { 'N' } else { 'S' },
            lon_degrees: lon_d,
            lon_minutes: lon_m,
            lon_seconds: lon_s,
            lon_direction: if self.longitude >= 0.0 { 'E' } else { 'W' },
            altitude: self.altitude,
        }
    }

    /// Create from DMS representation.
    pub fn from_dms(dms: &DmsCoordinate) -> Self {
        let lat = dms_to_decimal(dms.lat_degrees, dms.lat_minutes, dms.lat_seconds);
        let lat = if dms.lat_direction == 'S' { -lat } else { lat };

        let lon = dms_to_decimal(dms.lon_degrees, dms.lon_minutes, dms.lon_seconds);
        let lon = if dms.lon_direction == 'W' { -lon } else { lon };

        let mut coord = Self::new(lat, lon);
        coord.altitude = dms.altitude;
        coord
    }

    /// Format as a human-readable string.
    ///
    /// Example: `40°42'46.1"N 74°00'21.6"W`
    pub fn to_display_string(&self) -> String {
        let dms = self.to_dms();
        let deg = '\u{00B0}';
        let mut s = format!(
            "{lat_d}{deg}{lat_m}'{lat_s:.1}\"{lat_dir}  {lon_d}{deg}{lon_m}'{lon_s:.1}\"{lon_dir}",
            lat_d = dms.lat_degrees,
            lat_m = dms.lat_minutes,
            lat_s = dms.lat_seconds,
            lat_dir = dms.lat_direction,
            lon_d = dms.lon_degrees,
            lon_m = dms.lon_minutes,
            lon_s = dms.lon_seconds,
            lon_dir = dms.lon_direction,
        );
        if let Some(alt) = self.altitude {
            s.push_str(&format!("  {alt:.1}m"));
        }
        s
    }

    /// Format as ISO 6709 string (e.g., "+40.7128-074.0060/").
    pub fn to_iso6709(&self) -> String {
        let lat_sign = if self.latitude >= 0.0 { "+" } else { "" };
        let lon_sign = if self.longitude >= 0.0 { "+" } else { "" };
        if let Some(alt) = self.altitude {
            let alt_sign = if alt >= 0.0 { "+" } else { "" };
            format!(
                "{lat_sign}{:.4}{lon_sign}{:.4}{alt_sign}{:.1}/",
                self.latitude, self.longitude, alt
            )
        } else {
            format!(
                "{lat_sign}{:.4}{lon_sign}{:.4}/",
                self.latitude, self.longitude
            )
        }
    }

    /// Calculate the Haversine distance to another coordinate in kilometers.
    pub fn distance_km(&self, other: &GeoCoordinate) -> f64 {
        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let dlat = (other.latitude - self.latitude).to_radians();
        let dlon = (other.longitude - self.longitude).to_radians();

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        EARTH_RADIUS_KM * c
    }

    /// Calculate the initial bearing (azimuth) to another coordinate in degrees.
    pub fn bearing_to(&self, other: &GeoCoordinate) -> f64 {
        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let dlon = (other.longitude - self.longitude).to_radians();

        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();

        let bearing = y.atan2(x).to_degrees();
        (bearing + 360.0) % 360.0
    }

    /// Validate the coordinate values.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.latitude < -90.0 || self.latitude > 90.0 {
            issues.push(format!(
                "Latitude {:.4} out of range [-90, 90]",
                self.latitude
            ));
        }
        if self.longitude < -180.0 || self.longitude > 180.0 {
            issues.push(format!(
                "Longitude {:.4} out of range [-180, 180]",
                self.longitude
            ));
        }
        issues
    }

    /// Check if this coordinate is within a bounding box.
    pub fn is_within(&self, bbox: &GeoBoundingBox) -> bool {
        self.latitude >= bbox.south
            && self.latitude <= bbox.north
            && self.longitude >= bbox.west
            && self.longitude <= bbox.east
    }

    /// Parse from EXIF GPS metadata fields.
    ///
    /// Expects fields like `GPSLatitude`, `GPSLatitudeRef`, `GPSLongitude`, `GPSLongitudeRef`.
    pub fn from_metadata(metadata: &Metadata) -> Option<Self> {
        // Try decimal format first (XMP-style)
        if let (Some(lat_str), Some(lon_str)) = (
            metadata.get("GPSLatitude").and_then(|v| v.as_text()),
            metadata.get("GPSLongitude").and_then(|v| v.as_text()),
        ) {
            if let (Ok(lat), Ok(lon)) = (lat_str.parse::<f64>(), lon_str.parse::<f64>()) {
                // Check for direction references
                let lat_ref = metadata
                    .get("GPSLatitudeRef")
                    .and_then(|v| v.as_text())
                    .unwrap_or(if lat >= 0.0 { "N" } else { "S" });
                let lon_ref = metadata
                    .get("GPSLongitudeRef")
                    .and_then(|v| v.as_text())
                    .unwrap_or(if lon >= 0.0 { "E" } else { "W" });

                let lat = if lat_ref == "S" {
                    -lat.abs()
                } else {
                    lat.abs()
                };
                let lon = if lon_ref == "W" {
                    -lon.abs()
                } else {
                    lon.abs()
                };

                let mut coord = GeoCoordinate::new(lat, lon);

                // Try to get altitude
                if let Some(alt_str) = metadata.get("GPSAltitude").and_then(|v| v.as_text()) {
                    if let Ok(alt) = alt_str.parse::<f64>() {
                        let alt_ref = metadata
                            .get("GPSAltitudeRef")
                            .and_then(|v| v.as_text())
                            .unwrap_or("0");
                        let alt = if alt_ref == "1" { -alt } else { alt };
                        coord.altitude = Some(alt);
                    }
                }

                return Some(coord);
            }
        }

        // Try float values
        if let (Some(lat), Some(lon)) = (
            metadata.get("GPSLatitude").and_then(|v| v.as_float()),
            metadata.get("GPSLongitude").and_then(|v| v.as_float()),
        ) {
            let mut coord = GeoCoordinate::new(lat, lon);
            if let Some(alt) = metadata.get("GPSAltitude").and_then(|v| v.as_float()) {
                coord.altitude = Some(alt);
            }
            return Some(coord);
        }

        None
    }

    /// Write to a `Metadata` container as GPS fields.
    pub fn to_metadata(&self, metadata: &mut Metadata) {
        metadata.insert(
            "GPSLatitude".to_string(),
            MetadataValue::Text(format!("{:.6}", self.latitude.abs())),
        );
        metadata.insert(
            "GPSLatitudeRef".to_string(),
            MetadataValue::Text(if self.latitude >= 0.0 { "N" } else { "S" }.to_string()),
        );
        metadata.insert(
            "GPSLongitude".to_string(),
            MetadataValue::Text(format!("{:.6}", self.longitude.abs())),
        );
        metadata.insert(
            "GPSLongitudeRef".to_string(),
            MetadataValue::Text(if self.longitude >= 0.0 { "E" } else { "W" }.to_string()),
        );

        if let Some(alt) = self.altitude {
            metadata.insert(
                "GPSAltitude".to_string(),
                MetadataValue::Text(format!("{:.2}", alt.abs())),
            );
            metadata.insert(
                "GPSAltitudeRef".to_string(),
                MetadataValue::Text(if alt >= 0.0 { "0" } else { "1" }.to_string()),
            );
        }
    }
}

impl Default for GeoCoordinate {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
            altitude: None,
        }
    }
}

impl std::fmt::Display for GeoCoordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

/// Coordinate in Degrees-Minutes-Seconds (DMS) format.
#[derive(Debug, Clone, PartialEq)]
pub struct DmsCoordinate {
    /// Latitude degrees (0-90).
    pub lat_degrees: u32,
    /// Latitude minutes (0-59).
    pub lat_minutes: u32,
    /// Latitude seconds (0.0-59.999...).
    pub lat_seconds: f64,
    /// Latitude direction ('N' or 'S').
    pub lat_direction: char,
    /// Longitude degrees (0-180).
    pub lon_degrees: u32,
    /// Longitude minutes (0-59).
    pub lon_minutes: u32,
    /// Longitude seconds (0.0-59.999...).
    pub lon_seconds: f64,
    /// Longitude direction ('E' or 'W').
    pub lon_direction: char,
    /// Altitude in meters (optional).
    pub altitude: Option<f64>,
}

impl DmsCoordinate {
    /// Convert to decimal degrees.
    pub fn to_decimal(&self) -> GeoCoordinate {
        GeoCoordinate::from_dms(self)
    }
}

impl std::fmt::Display for DmsCoordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let deg = '\u{00B0}';
        write!(
            f,
            "{lat_d}{deg}{lat_m}'{lat_s:.1}\"{lat_dir} {lon_d}{deg}{lon_m}'{lon_s:.1}\"{lon_dir}",
            lat_d = self.lat_degrees,
            lat_m = self.lat_minutes,
            lat_s = self.lat_seconds,
            lat_dir = self.lat_direction,
            lon_d = self.lon_degrees,
            lon_m = self.lon_minutes,
            lon_s = self.lon_seconds,
            lon_dir = self.lon_direction,
        )
    }
}

/// A geographic bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoBoundingBox {
    /// Northern latitude bound.
    pub north: f64,
    /// Southern latitude bound.
    pub south: f64,
    /// Eastern longitude bound.
    pub east: f64,
    /// Western longitude bound.
    pub west: f64,
}

impl GeoBoundingBox {
    /// Create a bounding box from corners.
    pub fn new(north: f64, south: f64, east: f64, west: f64) -> Self {
        Self {
            north,
            south,
            east,
            west,
        }
    }

    /// Create a bounding box from a center point and radius in kilometers.
    pub fn from_center(center: &GeoCoordinate, radius_km: f64) -> Self {
        // Approximate: 1 degree latitude ~ 111 km
        let lat_delta = radius_km / 111.0;
        // 1 degree longitude ~ 111 * cos(latitude) km
        let cos_lat = center.latitude().to_radians().cos();
        let lon_delta = if cos_lat.abs() > 1e-10 {
            radius_km / (111.0 * cos_lat)
        } else {
            180.0 // near poles, cover all longitudes
        };

        Self {
            north: (center.latitude() + lat_delta).min(90.0),
            south: (center.latitude() - lat_delta).max(-90.0),
            east: center.longitude() + lon_delta,
            west: center.longitude() - lon_delta,
        }
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> GeoCoordinate {
        GeoCoordinate::new(
            (self.north + self.south) / 2.0,
            (self.east + self.west) / 2.0,
        )
    }

    /// Width in degrees (longitude span).
    pub fn width(&self) -> f64 {
        self.east - self.west
    }

    /// Height in degrees (latitude span).
    pub fn height(&self) -> f64 {
        self.north - self.south
    }

    /// Check if a coordinate is within this bounding box.
    pub fn contains(&self, coord: &GeoCoordinate) -> bool {
        coord.is_within(self)
    }

    /// Check if this bounding box overlaps with another.
    pub fn overlaps(&self, other: &GeoBoundingBox) -> bool {
        self.north >= other.south
            && self.south <= other.north
            && self.east >= other.west
            && self.west <= other.east
    }

    /// Validate the bounding box.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.north < self.south {
            issues.push(format!(
                "North ({:.4}) < South ({:.4})",
                self.north, self.south
            ));
        }
        if self.east < self.west {
            issues.push(format!(
                "East ({:.4}) < West ({:.4}) (may be intended for dateline crossing)",
                self.east, self.west
            ));
        }
        issues
    }
}

/// Parse a DMS string like "40 42 46.08" into (degrees, minutes, seconds).
///
/// # Errors
///
/// Returns an error if the string cannot be parsed.
pub fn parse_dms_string(s: &str) -> Result<(u32, u32, f64), Error> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(Error::ParseError(format!(
            "Invalid DMS string: '{s}' (need at least degrees and minutes)"
        )));
    }

    let degrees: u32 = parts[0]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid degrees in DMS: '{}'", parts[0])))?;
    let minutes: u32 = parts[1]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid minutes in DMS: '{}'", parts[1])))?;
    let seconds: f64 = if parts.len() >= 3 {
        parts[2]
            .parse()
            .map_err(|_| Error::ParseError(format!("Invalid seconds in DMS: '{}'", parts[2])))?
    } else {
        0.0
    };

    Ok((degrees, minutes, seconds))
}

// ---- Internal helpers ----

/// Convert decimal degrees to DMS.
fn decimal_to_dms(decimal: f64) -> (u32, u32, f64) {
    let abs = decimal.abs();
    let degrees = abs as u32;
    let min_float = (abs - f64::from(degrees)) * 60.0;
    let minutes = min_float as u32;
    let seconds = (min_float - f64::from(minutes)) * 60.0;
    (degrees, minutes, seconds)
}

/// Convert DMS to decimal degrees.
fn dms_to_decimal(degrees: u32, minutes: u32, seconds: f64) -> f64 {
    f64::from(degrees) + f64::from(minutes) / 60.0 + seconds / 3600.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetadataFormat;

    // ---- Coordinate tests ----

    #[test]
    fn test_coordinate_new() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        assert!((c.latitude() - 40.7128).abs() < 0.0001);
        assert!((c.longitude() - (-74.0060)).abs() < 0.0001);
        assert!(c.altitude().is_none());
        assert!(c.is_north());
        assert!(!c.is_east());
    }

    #[test]
    fn test_coordinate_with_altitude() {
        let c = GeoCoordinate::new(35.6762, 139.6503).with_altitude(40.0);
        assert_eq!(c.altitude(), Some(40.0));
    }

    #[test]
    fn test_coordinate_default() {
        let c = GeoCoordinate::default();
        assert!((c.latitude() - 0.0).abs() < f64::EPSILON);
        assert!((c.longitude() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coordinate_southern_hemisphere() {
        let c = GeoCoordinate::new(-33.8688, 151.2093); // Sydney
        assert!(!c.is_north());
        assert!(c.is_east());
    }

    #[test]
    fn test_coordinate_validate_valid() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        assert!(c.validate().is_empty());
    }

    #[test]
    fn test_coordinate_validate_invalid_lat() {
        let c = GeoCoordinate::new(95.0, 0.0);
        let issues = c.validate();
        assert!(!issues.is_empty());
        assert!(issues[0].contains("Latitude"));
    }

    #[test]
    fn test_coordinate_validate_invalid_lon() {
        let c = GeoCoordinate::new(0.0, 200.0);
        let issues = c.validate();
        assert!(!issues.is_empty());
        assert!(issues[0].contains("Longitude"));
    }

    // ---- DMS conversion tests ----

    #[test]
    fn test_decimal_to_dms() {
        let (d, m, s) = decimal_to_dms(40.7128);
        assert_eq!(d, 40);
        assert_eq!(m, 42);
        assert!((s - 46.08).abs() < 0.1);
    }

    #[test]
    fn test_dms_to_decimal() {
        let decimal = dms_to_decimal(40, 42, 46.08);
        assert!((decimal - 40.7128).abs() < 0.0001);
    }

    #[test]
    fn test_dms_round_trip() {
        let original = 40.7128;
        let (d, m, s) = decimal_to_dms(original);
        let restored = dms_to_decimal(d, m, s);
        assert!((original - restored).abs() < 0.0001);
    }

    #[test]
    fn test_coordinate_to_dms() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        let dms = c.to_dms();
        assert_eq!(dms.lat_degrees, 40);
        assert_eq!(dms.lat_direction, 'N');
        assert_eq!(dms.lon_degrees, 74);
        assert_eq!(dms.lon_direction, 'W');
    }

    #[test]
    fn test_coordinate_from_dms() {
        let dms = DmsCoordinate {
            lat_degrees: 40,
            lat_minutes: 42,
            lat_seconds: 46.08,
            lat_direction: 'N',
            lon_degrees: 74,
            lon_minutes: 0,
            lon_seconds: 21.6,
            lon_direction: 'W',
            altitude: None,
        };

        let coord = GeoCoordinate::from_dms(&dms);
        assert!((coord.latitude() - 40.7128).abs() < 0.001);
        assert!(coord.longitude() < 0.0); // West
    }

    #[test]
    fn test_coordinate_from_dms_south() {
        let dms = DmsCoordinate {
            lat_degrees: 33,
            lat_minutes: 52,
            lat_seconds: 7.68,
            lat_direction: 'S',
            lon_degrees: 151,
            lon_minutes: 12,
            lon_seconds: 33.48,
            lon_direction: 'E',
            altitude: Some(10.0),
        };

        let coord = GeoCoordinate::from_dms(&dms);
        assert!(coord.latitude() < 0.0); // South
        assert!(coord.longitude() > 0.0); // East
        assert_eq!(coord.altitude(), Some(10.0));
    }

    #[test]
    fn test_dms_coordinate_to_decimal() {
        let dms = DmsCoordinate {
            lat_degrees: 40,
            lat_minutes: 42,
            lat_seconds: 46.08,
            lat_direction: 'N',
            lon_degrees: 74,
            lon_minutes: 0,
            lon_seconds: 21.6,
            lon_direction: 'W',
            altitude: None,
        };
        let coord = dms.to_decimal();
        assert!((coord.latitude() - 40.7128).abs() < 0.001);
    }

    // ---- Display tests ----

    #[test]
    fn test_coordinate_display_string() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        let display = c.to_display_string();
        assert!(display.contains("40"));
        assert!(display.contains("N"));
        assert!(display.contains("74"));
        assert!(display.contains("W"));
    }

    #[test]
    fn test_coordinate_display_with_altitude() {
        let c = GeoCoordinate::new(35.6762, 139.6503).with_altitude(40.0);
        let display = c.to_display_string();
        assert!(display.contains("40.0m"));
    }

    #[test]
    fn test_coordinate_to_iso6709() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        let iso = c.to_iso6709();
        assert!(iso.starts_with('+'));
        assert!(iso.contains('-'));
        assert!(iso.ends_with('/'));
    }

    #[test]
    fn test_coordinate_to_iso6709_with_altitude() {
        let c = GeoCoordinate::new(40.7128, -74.0060).with_altitude(10.5);
        let iso = c.to_iso6709();
        assert!(iso.contains("10.5"));
    }

    #[test]
    fn test_coordinate_display_trait() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        let s = format!("{c}");
        assert!(!s.is_empty());
    }

    // ---- Distance and bearing tests ----

    #[test]
    fn test_distance_same_point() {
        let c = GeoCoordinate::new(40.7128, -74.0060);
        assert!((c.distance_km(&c) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_nyc_to_london() {
        let nyc = GeoCoordinate::new(40.7128, -74.0060);
        let london = GeoCoordinate::new(51.5074, -0.1278);
        let dist = nyc.distance_km(&london);
        // Approximately 5570 km
        assert!(dist > 5500.0 && dist < 5700.0);
    }

    #[test]
    fn test_distance_antipodes() {
        let a = GeoCoordinate::new(0.0, 0.0);
        let b = GeoCoordinate::new(0.0, 180.0);
        let dist = a.distance_km(&b);
        // Half the earth's circumference ~ 20015 km
        assert!(dist > 19000.0 && dist < 21000.0);
    }

    #[test]
    fn test_bearing_east() {
        let a = GeoCoordinate::new(0.0, 0.0);
        let b = GeoCoordinate::new(0.0, 1.0);
        let bearing = a.bearing_to(&b);
        assert!((bearing - 90.0).abs() < 1.0); // Should be approximately East
    }

    #[test]
    fn test_bearing_north() {
        let a = GeoCoordinate::new(0.0, 0.0);
        let b = GeoCoordinate::new(1.0, 0.0);
        let bearing = a.bearing_to(&b);
        assert!(bearing < 1.0 || bearing > 359.0); // Should be approximately North (0)
    }

    // ---- Bounding box tests ----

    #[test]
    fn test_bbox_new() {
        let bbox = GeoBoundingBox::new(41.0, 40.0, -73.0, -75.0);
        assert!((bbox.height() - 1.0).abs() < f64::EPSILON);
        assert!((bbox.width() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bbox_center() {
        let bbox = GeoBoundingBox::new(41.0, 40.0, -73.0, -75.0);
        let center = bbox.center();
        assert!((center.latitude() - 40.5).abs() < f64::EPSILON);
        assert!((center.longitude() - (-74.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bbox_contains() {
        let bbox = GeoBoundingBox::new(41.0, 40.0, -73.0, -75.0);
        let inside = GeoCoordinate::new(40.5, -74.0);
        let outside = GeoCoordinate::new(42.0, -74.0);

        assert!(bbox.contains(&inside));
        assert!(!bbox.contains(&outside));
    }

    #[test]
    fn test_bbox_from_center() {
        let center = GeoCoordinate::new(40.7128, -74.0060);
        let bbox = GeoBoundingBox::from_center(&center, 10.0);

        assert!(bbox.contains(&center));
        assert!(bbox.north > center.latitude());
        assert!(bbox.south < center.latitude());
        assert!(bbox.east > center.longitude());
        assert!(bbox.west < center.longitude());
    }

    #[test]
    fn test_bbox_overlaps() {
        let b1 = GeoBoundingBox::new(41.0, 40.0, -73.0, -75.0);
        let b2 = GeoBoundingBox::new(40.5, 39.5, -72.0, -74.0);
        let b3 = GeoBoundingBox::new(50.0, 49.0, 0.0, -1.0);

        assert!(b1.overlaps(&b2));
        assert!(!b1.overlaps(&b3));
    }

    #[test]
    fn test_bbox_validate_valid() {
        let bbox = GeoBoundingBox::new(41.0, 40.0, -73.0, -75.0);
        assert!(bbox.validate().is_empty());
    }

    #[test]
    fn test_bbox_validate_inverted() {
        let bbox = GeoBoundingBox::new(40.0, 41.0, -73.0, -75.0); // north < south
        let issues = bbox.validate();
        assert!(!issues.is_empty());
    }

    // ---- Parse DMS string tests ----

    #[test]
    fn test_parse_dms_string_full() {
        let (d, m, s) = parse_dms_string("40 42 46.08").expect("should parse");
        assert_eq!(d, 40);
        assert_eq!(m, 42);
        assert!((s - 46.08).abs() < 0.01);
    }

    #[test]
    fn test_parse_dms_string_no_seconds() {
        let (d, m, s) = parse_dms_string("40 42").expect("should parse");
        assert_eq!(d, 40);
        assert_eq!(m, 42);
        assert!((s - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_dms_string_invalid() {
        assert!(parse_dms_string("invalid").is_err());
        assert!(parse_dms_string("40").is_err());
    }

    // ---- Metadata integration tests ----

    #[test]
    fn test_coordinate_metadata_round_trip() {
        let original = GeoCoordinate::new(40.7128, -74.0060).with_altitude(10.5);

        let mut metadata = Metadata::new(MetadataFormat::Exif);
        original.to_metadata(&mut metadata);

        let restored = GeoCoordinate::from_metadata(&metadata).expect("should parse");
        assert!((restored.latitude() - original.latitude()).abs() < 0.001);
        assert!((restored.longitude() - original.longitude()).abs() < 0.001);
        assert!(restored.altitude().is_some());
    }

    #[test]
    fn test_coordinate_from_metadata_empty() {
        let metadata = Metadata::new(MetadataFormat::Exif);
        assert!(GeoCoordinate::from_metadata(&metadata).is_none());
    }

    #[test]
    fn test_coordinate_from_metadata_south_west() {
        let mut metadata = Metadata::new(MetadataFormat::Exif);
        metadata.insert("GPSLatitude".into(), MetadataValue::Text("33.8688".into()));
        metadata.insert("GPSLatitudeRef".into(), MetadataValue::Text("S".into()));
        metadata.insert(
            "GPSLongitude".into(),
            MetadataValue::Text("151.2093".into()),
        );
        metadata.insert("GPSLongitudeRef".into(), MetadataValue::Text("E".into()));

        let coord = GeoCoordinate::from_metadata(&metadata).expect("should parse");
        assert!(coord.latitude() < 0.0);
        assert!(coord.longitude() > 0.0);
    }

    #[test]
    fn test_coordinate_from_metadata_float_values() {
        let mut metadata = Metadata::new(MetadataFormat::Exif);
        metadata.insert("GPSLatitude".into(), MetadataValue::Float(40.7128));
        metadata.insert("GPSLongitude".into(), MetadataValue::Float(-74.0060));
        metadata.insert("GPSAltitude".into(), MetadataValue::Float(25.0));

        let coord = GeoCoordinate::from_metadata(&metadata).expect("should parse");
        assert!((coord.latitude() - 40.7128).abs() < 0.001);
        assert!((coord.longitude() - (-74.0060)).abs() < 0.001);
        assert_eq!(coord.altitude(), Some(25.0));
    }

    #[test]
    fn test_coordinate_to_metadata_direction_refs() {
        let coord = GeoCoordinate::new(-33.8688, 151.2093);
        let mut metadata = Metadata::new(MetadataFormat::Exif);
        coord.to_metadata(&mut metadata);

        assert_eq!(
            metadata.get("GPSLatitudeRef").and_then(|v| v.as_text()),
            Some("S")
        );
        assert_eq!(
            metadata.get("GPSLongitudeRef").and_then(|v| v.as_text()),
            Some("E")
        );
    }

    #[test]
    fn test_coordinate_is_within_bbox() {
        let nyc = GeoCoordinate::new(40.7128, -74.0060);
        let us_bbox = GeoBoundingBox::new(49.0, 24.0, -66.0, -125.0);
        let europe_bbox = GeoBoundingBox::new(71.0, 35.0, 40.0, -10.0);

        assert!(nyc.is_within(&us_bbox));
        assert!(!nyc.is_within(&europe_bbox));
    }

    #[test]
    fn test_coordinate_from_metadata_altitude_below_sea_level() {
        let mut metadata = Metadata::new(MetadataFormat::Exif);
        metadata.insert("GPSLatitude".into(), MetadataValue::Text("31.5".into()));
        metadata.insert("GPSLatitudeRef".into(), MetadataValue::Text("N".into()));
        metadata.insert("GPSLongitude".into(), MetadataValue::Text("35.5".into()));
        metadata.insert("GPSLongitudeRef".into(), MetadataValue::Text("E".into()));
        metadata.insert("GPSAltitude".into(), MetadataValue::Text("430.0".into()));
        metadata.insert("GPSAltitudeRef".into(), MetadataValue::Text("1".into())); // below sea level

        let coord = GeoCoordinate::from_metadata(&metadata).expect("should parse");
        assert!(coord.altitude().expect("should have altitude") < 0.0);
    }
}
