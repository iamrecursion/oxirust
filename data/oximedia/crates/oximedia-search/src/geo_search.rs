#![allow(dead_code)]
//! Location-based media search using GPS metadata.
//!
//! This module enables searching for media assets by geographic location.
//! Assets are indexed with latitude/longitude coordinates extracted from
//! EXIF/XMP GPS metadata, and can be retrieved by:
//!
//! - **Radius search**: all assets within N kilometres of a point
//! - **Bounding-box search**: all assets inside a lat/lon rectangle
//! - **Nearest-neighbour**: the K closest assets to a query point
//!
//! # Coordinate system
//!
//! All coordinates use WGS-84 decimal degrees (latitude in `[-90, 90]`,
//! longitude in `[-180, 180]`).  Distances are computed using the
//! **Haversine formula**, which gives great-circle distances accurate
//! to within ~0.5% for typical media-geolocation use-cases.
//!
//! # Patent-free
//!
//! Only standard trigonometric (Haversine) geometry is used.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Coordinate and location types
// ---------------------------------------------------------------------------

/// Earth's mean radius in kilometres.
const EARTH_RADIUS_KM: f64 = 6_371.0;

/// A WGS-84 geographic coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    /// Latitude in decimal degrees, `[-90, 90]`.
    pub lat: f64,
    /// Longitude in decimal degrees, `[-180, 180]`.
    pub lon: f64,
}

impl GeoPoint {
    /// Create a new GeoPoint.
    ///
    /// # Errors
    ///
    /// Returns an error if `lat` or `lon` are outside valid ranges.
    pub fn new(lat: f64, lon: f64) -> SearchResult<Self> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(SearchError::InvalidQuery(format!(
                "Latitude {lat} is outside [-90, 90]"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(SearchError::InvalidQuery(format!(
                "Longitude {lon} is outside [-180, 180]"
            )));
        }
        Ok(Self { lat, lon })
    }

    /// Create a GeoPoint without validation (for trusted internal data).
    #[must_use]
    pub fn unchecked(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Compute the Haversine great-circle distance to another point in km.
    #[must_use]
    pub fn distance_km(&self, other: &Self) -> f64 {
        haversine_km(self.lat, self.lon, other.lat, other.lon)
    }
}

/// A bounding box defined by two opposite corners.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoBoundingBox {
    /// South-west corner (minimum lat/lon).
    pub south_west: GeoPoint,
    /// North-east corner (maximum lat/lon).
    pub north_east: GeoPoint,
}

impl GeoBoundingBox {
    /// Create a bounding box from two opposite corners.
    ///
    /// # Errors
    ///
    /// Returns an error if either point has invalid coordinates.
    pub fn new(south_west: GeoPoint, north_east: GeoPoint) -> SearchResult<Self> {
        if south_west.lat > north_east.lat {
            return Err(SearchError::InvalidQuery(format!(
                "south_west lat {} > north_east lat {}",
                south_west.lat, north_east.lat
            )));
        }
        Ok(Self {
            south_west,
            north_east,
        })
    }

    /// Check whether a point is inside the bounding box.
    ///
    /// Handles anti-meridian crossing when `south_west.lon > north_east.lon`.
    #[must_use]
    pub fn contains(&self, point: &GeoPoint) -> bool {
        let lat_ok = (self.south_west.lat..=self.north_east.lat).contains(&point.lat);
        let lon_ok = if self.south_west.lon <= self.north_east.lon {
            (self.south_west.lon..=self.north_east.lon).contains(&point.lon)
        } else {
            // Anti-meridian crossing.
            point.lon >= self.south_west.lon || point.lon <= self.north_east.lon
        };
        lat_ok && lon_ok
    }
}

/// GPS location data attached to a media asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetLocation {
    /// The geographic position.
    pub point: GeoPoint,
    /// Altitude in metres above sea level (optional).
    pub altitude_m: Option<f64>,
    /// GPS accuracy estimate in metres (optional).
    pub accuracy_m: Option<f64>,
    /// Human-readable place name or address (optional).
    pub place_name: Option<String>,
    /// Country code (ISO 3166-1 alpha-2, optional).
    pub country_code: Option<String>,
}

impl AssetLocation {
    /// Create a minimal location from a coordinate.
    #[must_use]
    pub fn from_point(point: GeoPoint) -> Self {
        Self {
            point,
            altitude_m: None,
            accuracy_m: None,
            place_name: None,
            country_code: None,
        }
    }
}

/// A geo search result.
#[derive(Debug, Clone)]
pub struct GeoSearchResult {
    /// The matched asset.
    pub asset_id: Uuid,
    /// The asset's location.
    pub location: AssetLocation,
    /// Distance from the query point in km (for radius/NN searches).
    pub distance_km: f64,
    /// Relevance score (1 / (1 + distance_km) for proximity).
    pub score: f64,
}

// ---------------------------------------------------------------------------
// GeoIndex
// ---------------------------------------------------------------------------

/// Indexes media assets by geographic location and supports spatial queries.
///
/// # Implementation
///
/// Uses a flat list with linear scan for collections up to ~100K assets
/// (typical for media libraries). Haversine distance computation is fast
/// enough for this scale; larger collections can be sharded by coarse
/// lat/lon grid cells.
#[derive(Debug, Default)]
pub struct GeoIndex {
    /// Asset ID -> location mapping.
    locations: HashMap<Uuid, AssetLocation>,
}

impl GeoIndex {
    /// Create an empty geo index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Index or update the location for an asset.
    pub fn add_location(&mut self, asset_id: Uuid, location: AssetLocation) {
        self.locations.insert(asset_id, location);
    }

    /// Remove the location entry for an asset.
    pub fn remove_location(&mut self, asset_id: Uuid) {
        self.locations.remove(&asset_id);
    }

    /// Look up the location for a specific asset.
    #[must_use]
    pub fn get_location(&self, asset_id: Uuid) -> Option<&AssetLocation> {
        self.locations.get(&asset_id)
    }

    /// Number of indexed assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Find all assets within `radius_km` kilometres of `center`.
    ///
    /// Results are sorted by ascending distance.
    #[must_use]
    pub fn search_radius(&self, center: &GeoPoint, radius_km: f64) -> Vec<GeoSearchResult> {
        let mut results: Vec<GeoSearchResult> = self
            .locations
            .iter()
            .filter_map(|(&asset_id, loc)| {
                let dist = center.distance_km(&loc.point);
                if dist <= radius_km {
                    let score = 1.0 / (1.0 + dist);
                    Some(GeoSearchResult {
                        asset_id,
                        location: loc.clone(),
                        distance_km: dist,
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance_km
                .partial_cmp(&b.distance_km)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Find all assets inside a bounding box.
    ///
    /// Results are sorted by their distance from the bounding-box centre.
    #[must_use]
    pub fn search_bbox(&self, bbox: &GeoBoundingBox) -> Vec<GeoSearchResult> {
        // Compute bbox centre for distance-based sorting.
        let centre_lat = (bbox.south_west.lat + bbox.north_east.lat) / 2.0;
        let centre_lon = (bbox.south_west.lon + bbox.north_east.lon) / 2.0;
        // Longitude wrapping: if bbox crosses anti-meridian, adjust.
        let centre_lon = if bbox.south_west.lon > bbox.north_east.lon {
            let adjusted = (bbox.south_west.lon + bbox.north_east.lon + 360.0) / 2.0;
            if adjusted > 180.0 {
                adjusted - 360.0
            } else {
                adjusted
            }
        } else {
            centre_lon
        };
        let centre = GeoPoint::unchecked(centre_lat, centre_lon);

        let mut results: Vec<GeoSearchResult> = self
            .locations
            .iter()
            .filter_map(|(&asset_id, loc)| {
                if bbox.contains(&loc.point) {
                    let dist = centre.distance_km(&loc.point);
                    let score = 1.0 / (1.0 + dist);
                    Some(GeoSearchResult {
                        asset_id,
                        location: loc.clone(),
                        distance_km: dist,
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance_km
                .partial_cmp(&b.distance_km)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Return the `k` assets nearest to `query_point`.
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is zero.
    pub fn search_nearest(
        &self,
        query_point: &GeoPoint,
        k: usize,
    ) -> SearchResult<Vec<GeoSearchResult>> {
        if k == 0 {
            return Err(SearchError::InvalidQuery("k must be at least 1".into()));
        }

        let mut all: Vec<GeoSearchResult> = self
            .locations
            .iter()
            .map(|(&asset_id, loc)| {
                let dist = query_point.distance_km(&loc.point);
                let score = 1.0 / (1.0 + dist);
                GeoSearchResult {
                    asset_id,
                    location: loc.clone(),
                    distance_km: dist,
                    score,
                }
            })
            .collect();

        all.sort_by(|a, b| {
            a.distance_km
                .partial_cmp(&b.distance_km)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(k);
        Ok(all)
    }

    /// Filter assets by country code (ISO 3166-1 alpha-2).
    #[must_use]
    pub fn filter_by_country(&self, country_code: &str) -> Vec<(Uuid, &AssetLocation)> {
        let code = country_code.to_uppercase();
        self.locations
            .iter()
            .filter(|(_, loc)| {
                loc.country_code
                    .as_deref()
                    .map(str::to_uppercase)
                    .as_deref()
                    == Some(&code)
            })
            .map(|(&id, loc)| (id, loc))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Haversine distance
// ---------------------------------------------------------------------------

/// Compute great-circle distance in km using the Haversine formula.
#[must_use]
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let a = (d_lat / 2.0).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_KM * c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn london() -> GeoPoint {
        GeoPoint::unchecked(51.5074, -0.1278)
    }

    fn paris() -> GeoPoint {
        GeoPoint::unchecked(48.8566, 2.3522)
    }

    fn new_york() -> GeoPoint {
        GeoPoint::unchecked(40.7128, -74.0060)
    }

    fn sydney() -> GeoPoint {
        GeoPoint::unchecked(-33.8688, 151.2093)
    }

    #[test]
    fn test_geopoint_new_valid() {
        let p = GeoPoint::new(51.5, -0.12);
        assert!(p.is_ok());
    }

    #[test]
    fn test_geopoint_new_invalid_lat() {
        assert!(GeoPoint::new(91.0, 0.0).is_err());
        assert!(GeoPoint::new(-91.0, 0.0).is_err());
    }

    #[test]
    fn test_geopoint_new_invalid_lon() {
        assert!(GeoPoint::new(0.0, 181.0).is_err());
        assert!(GeoPoint::new(0.0, -181.0).is_err());
    }

    #[test]
    fn test_haversine_london_to_paris() {
        let dist = london().distance_km(&paris());
        // Real distance ~343 km; allow 2% tolerance.
        assert!((dist - 343.0).abs() < 10.0, "dist = {dist}");
    }

    #[test]
    fn test_haversine_same_point_is_zero() {
        let p = london();
        let dist = p.distance_km(&p);
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_haversine_london_to_new_york() {
        let dist = london().distance_km(&new_york());
        // ~5,570 km
        assert!((dist - 5570.0).abs() < 100.0, "dist = {dist}");
    }

    #[test]
    fn test_bounding_box_contains() {
        let sw = GeoPoint::unchecked(48.0, -1.0);
        let ne = GeoPoint::unchecked(52.0, 3.0);
        let bbox = GeoBoundingBox::new(sw, ne).expect("valid bbox");

        assert!(bbox.contains(&london())); // 51.5, -0.12
        assert!(bbox.contains(&paris())); // 48.85, 2.35
        assert!(!bbox.contains(&sydney())); // far away
    }

    #[test]
    fn test_bounding_box_invalid_lat_order() {
        let sw = GeoPoint::unchecked(52.0, 0.0);
        let ne = GeoPoint::unchecked(48.0, 3.0);
        assert!(GeoBoundingBox::new(sw, ne).is_err());
    }

    #[test]
    fn test_geo_index_add_and_get() {
        let mut idx = GeoIndex::new();
        let id = Uuid::new_v4();
        idx.add_location(id, AssetLocation::from_point(london()));
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
        let loc = idx.get_location(id);
        assert!(loc.is_some());
        assert!((loc.map(|l| l.point.lat).unwrap_or(0.0) - 51.5074).abs() < 1e-4);
    }

    #[test]
    fn test_geo_index_remove() {
        let mut idx = GeoIndex::new();
        let id = Uuid::new_v4();
        idx.add_location(id, AssetLocation::from_point(london()));
        idx.remove_location(id);
        assert!(idx.is_empty());
        assert!(idx.get_location(id).is_none());
    }

    #[test]
    fn test_search_radius_finds_nearby() {
        let mut idx = GeoIndex::new();
        let london_id = Uuid::new_v4();
        let paris_id = Uuid::new_v4();
        let sydney_id = Uuid::new_v4();

        idx.add_location(london_id, AssetLocation::from_point(london()));
        idx.add_location(paris_id, AssetLocation::from_point(paris()));
        idx.add_location(sydney_id, AssetLocation::from_point(sydney()));

        // 500 km from London should include Paris (~343 km) but not Sydney.
        let results = idx.search_radius(&london(), 500.0);
        let ids: Vec<Uuid> = results.iter().map(|r| r.asset_id).collect();
        assert!(ids.contains(&london_id)); // 0 km
        assert!(ids.contains(&paris_id)); // ~343 km
        assert!(!ids.contains(&sydney_id));
    }

    #[test]
    fn test_search_radius_sorted_by_distance() {
        let mut idx = GeoIndex::new();
        let london_id = Uuid::new_v4();
        let paris_id = Uuid::new_v4();
        idx.add_location(london_id, AssetLocation::from_point(london()));
        idx.add_location(paris_id, AssetLocation::from_point(paris()));

        let results = idx.search_radius(&london(), 500.0);
        assert_eq!(results[0].asset_id, london_id); // closest first
        assert!(results[0].distance_km < results[1].distance_km);
    }

    #[test]
    fn test_search_bbox_finds_europe() {
        let mut idx = GeoIndex::new();
        let london_id = Uuid::new_v4();
        let paris_id = Uuid::new_v4();
        let ny_id = Uuid::new_v4();

        idx.add_location(london_id, AssetLocation::from_point(london()));
        idx.add_location(paris_id, AssetLocation::from_point(paris()));
        idx.add_location(ny_id, AssetLocation::from_point(new_york()));

        // Bounding box covering western Europe.
        let sw = GeoPoint::unchecked(45.0, -5.0);
        let ne = GeoPoint::unchecked(55.0, 10.0);
        let bbox = GeoBoundingBox::new(sw, ne).expect("valid");

        let results = idx.search_bbox(&bbox);
        let ids: Vec<Uuid> = results.iter().map(|r| r.asset_id).collect();
        assert!(ids.contains(&london_id));
        assert!(ids.contains(&paris_id));
        assert!(!ids.contains(&ny_id));
    }

    #[test]
    fn test_search_nearest_k() {
        let mut idx = GeoIndex::new();
        let london_id = Uuid::new_v4();
        let paris_id = Uuid::new_v4();
        let ny_id = Uuid::new_v4();

        idx.add_location(london_id, AssetLocation::from_point(london()));
        idx.add_location(paris_id, AssetLocation::from_point(paris()));
        idx.add_location(ny_id, AssetLocation::from_point(new_york()));

        let results = idx.search_nearest(&london(), 2).expect("ok");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].asset_id, london_id);
        assert_eq!(results[1].asset_id, paris_id);
    }

    #[test]
    fn test_search_nearest_zero_returns_err() {
        let idx = GeoIndex::new();
        assert!(idx.search_nearest(&london(), 0).is_err());
    }

    #[test]
    fn test_search_radius_empty_index() {
        let idx = GeoIndex::new();
        let results = idx.search_radius(&london(), 1000.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter_by_country() {
        let mut idx = GeoIndex::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let mut loc_uk = AssetLocation::from_point(london());
        loc_uk.country_code = Some("GB".into());
        let mut loc_fr = AssetLocation::from_point(paris());
        loc_fr.country_code = Some("FR".into());
        let mut loc_uk2 = AssetLocation::from_point(GeoPoint::unchecked(53.48, -2.24));
        loc_uk2.country_code = Some("GB".into());

        idx.add_location(id1, loc_uk);
        idx.add_location(id2, loc_fr);
        idx.add_location(id3, loc_uk2);

        let gb_assets = idx.filter_by_country("GB");
        assert_eq!(gb_assets.len(), 2);
        let gb_ids: Vec<Uuid> = gb_assets.iter().map(|&(id, _)| id).collect();
        assert!(gb_ids.contains(&id1));
        assert!(gb_ids.contains(&id3));
    }

    #[test]
    fn test_asset_location_altitude() {
        let pt = GeoPoint::unchecked(47.6, 8.8);
        let mut loc = AssetLocation::from_point(pt);
        loc.altitude_m = Some(1200.0);
        loc.place_name = Some("Swiss Alps".into());
        assert_eq!(loc.altitude_m, Some(1200.0));
        assert_eq!(loc.place_name.as_deref(), Some("Swiss Alps"));
    }

    #[test]
    fn test_geopoint_serialization() {
        let p = GeoPoint::unchecked(48.85, 2.35);
        let json = serde_json::to_string(&p).expect("serialize");
        let back: GeoPoint = serde_json::from_str(&json).expect("deserialize");
        assert!((back.lat - 48.85).abs() < 1e-6);
        assert!((back.lon - 2.35).abs() < 1e-6);
    }

    #[test]
    fn test_geo_search_result_score_decreases_with_distance() {
        let mut idx = GeoIndex::new();
        let close_id = Uuid::new_v4();
        let far_id = Uuid::new_v4();
        idx.add_location(
            close_id,
            AssetLocation::from_point(GeoPoint::unchecked(51.5, -0.12)),
        );
        idx.add_location(
            far_id,
            AssetLocation::from_point(GeoPoint::unchecked(51.5 + 5.0, -0.12)),
        );

        let results = idx.search_radius(&GeoPoint::unchecked(51.5, -0.12), 1000.0);
        assert!(results[0].score > results[1].score);
    }
}
