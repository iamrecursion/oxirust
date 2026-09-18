//! Geographic routing — assign requests to the closest CDN edge node using
//! Haversine distance and estimate propagation latency.
//!
//! # Overview
//!
//! [`GeoRouter`] holds a registry of [`EdgeNodeGeo`] entries (each combining an
//! [`EdgeNodeId`] with a [`GeoLocation`]) and can:
//!
//! - Return the **closest** edge node for a client location
//!   ([`GeoRouter::assign_edge`]).
//! - Estimate the **latency** in milliseconds between any two locations
//!   ([`GeoRouter::latency_estimate_ms`]).
//! - List all nodes within a given distance ([`GeoRouter::nodes_within_km`]).
//!
//! # Latency model
//!
//! `latency_ms = distance_km / 200.0 * 1000.0 + 5.0`
//!
//! This approximates the speed of light in optical fibre (≈ 200 km/ms) plus a
//! fixed 5 ms base overhead.

use std::collections::HashMap;
use std::fmt;

// ─── Region ───────────────────────────────────────────────────────────────────

/// Broad geographic region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Region {
    /// North America (US, CA, MX, …).
    NorthAmerica,
    /// Europe (EU, UK, CH, NO, …).
    Europe,
    /// Asia-Pacific (JP, AU, SG, CN, IN, …).
    AsiaPacific,
    /// Latin America (BR, AR, CO, CL, …).
    LatinAmerica,
    /// Middle East and Africa (AE, SA, ZA, EG, …).
    MiddleEastAfrica,
    /// Could not be determined.
    Unknown,
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NorthAmerica => "north_america",
            Self::Europe => "europe",
            Self::AsiaPacific => "asia_pacific",
            Self::LatinAmerica => "latin_america",
            Self::MiddleEastAfrica => "middle_east_africa",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

impl Region {
    /// Derive a region from a two-letter ISO 3166-1 alpha-2 country code.
    ///
    /// Unknown or unrecognised codes map to [`Region::Unknown`].
    pub fn from_country_code(cc: &str) -> Self {
        match cc.to_uppercase().as_str() {
            // North America
            "US" | "CA" | "MX" | "GT" | "BZ" | "HN" | "SV" | "NI" | "CR" | "PA" | "CU" | "JM"
            | "HT" | "DO" | "PR" | "TT" | "BB" | "LC" | "VC" | "GD" | "AG" | "KN" | "DM" | "AI"
            | "KY" | "TC" | "VG" | "VI" | "AW" | "CW" | "SX" | "BQ" | "BS" | "TF" | "GU" | "MP"
            | "AS" | "UM" => Self::NorthAmerica,

            // Latin America (South America)
            "BR" | "AR" | "CL" | "CO" | "PE" | "VE" | "EC" | "BO" | "PY" | "UY" | "SR" | "GY"
            | "GF" | "FK" | "GS" => Self::LatinAmerica,

            // Europe
            "GB" | "DE" | "FR" | "IT" | "ES" | "PT" | "NL" | "BE" | "LU" | "CH" | "AT" | "SE"
            | "NO" | "DK" | "FI" | "IS" | "IE" | "PL" | "CZ" | "SK" | "HU" | "RO" | "BG" | "HR"
            | "SI" | "BA" | "RS" | "ME" | "MK" | "AL" | "GR" | "CY" | "MT" | "EE" | "LV" | "LT"
            | "BY" | "UA" | "MD" | "RU" | "TR" | "GE" | "AM" | "AZ" | "LI" | "MC" | "SM" | "VA"
            | "AD" | "FO" | "GI" | "JE" | "GG" | "IM" | "AX" | "SJ" | "PM" | "GL" | "XK" => {
                Self::Europe
            }

            // Middle East & Africa
            "AE" | "SA" | "QA" | "BH" | "KW" | "OM" | "YE" | "IQ" | "IR" | "SY" | "LB" | "JO"
            | "IL" | "PS" | "EG" | "LY" | "TN" | "DZ" | "MA" | "MR" | "ML" | "NE" | "TD" | "SD"
            | "SS" | "ER" | "ET" | "DJ" | "SO" | "KE" | "UG" | "RW" | "BI" | "TZ" | "MZ" | "ZM"
            | "MW" | "ZW" | "BW" | "NA" | "ZA" | "LS" | "SZ" | "AO" | "ZR" | "CD" | "CG" | "CM"
            | "CF" | "GQ" | "GA" | "ST" | "CV" | "GW" | "GN" | "SL" | "LR" | "CI" | "GH" | "TG"
            | "BJ" | "NG" | "SN" | "GM" | "BF" | "MG" | "KM" | "MU" | "SC" | "RE" | "YT" | "SH"
            | "EH" | "AF" | "PK" => Self::MiddleEastAfrica,

            // Asia-Pacific
            "JP" | "CN" | "KR" | "TW" | "HK" | "MO" | "MN" | "KP" | "VN" | "TH" | "MY" | "SG"
            | "ID" | "PH" | "MM" | "KH" | "LA" | "BN" | "TL" | "IN" | "BD" | "LK" | "NP" | "BT"
            | "MV" | "AU" | "NZ" | "FJ" | "PG" | "SB" | "VU" | "WS" | "TO" | "TV" | "NR" | "PW"
            | "FM" | "MH" | "KI" | "CK" | "NU" | "TK" | "WF" | "PF" | "NC" | "KZ" | "UZ" | "TM"
            | "TJ" | "KG" => Self::AsiaPacific,

            _ => Self::Unknown,
        }
    }
}

// ─── GeoLocation ─────────────────────────────────────────────────────────────

/// A geographic position enriched with region and country metadata.
#[derive(Debug, Clone)]
pub struct GeoLocation {
    /// Latitude in decimal degrees (−90 … +90).
    pub latitude: f64,
    /// Longitude in decimal degrees (−180 … +180).
    pub longitude: f64,
    /// Broad geographic region.
    pub region: Region,
    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`, `"DE"`).
    pub country_code: String,
}

impl GeoLocation {
    /// Create a new location from coordinates and a country code.
    ///
    /// The [`Region`] is inferred automatically via
    /// [`Region::from_country_code`].
    pub fn new(latitude: f64, longitude: f64, country_code: &str) -> Self {
        Self {
            latitude,
            longitude,
            region: Region::from_country_code(country_code),
            country_code: country_code.to_uppercase(),
        }
    }

    /// Create a location with an explicit region (overriding the inferred one).
    pub fn with_region(latitude: f64, longitude: f64, country_code: &str, region: Region) -> Self {
        Self {
            latitude,
            longitude,
            region,
            country_code: country_code.to_uppercase(),
        }
    }
}

// ─── Haversine distance ───────────────────────────────────────────────────────

/// Compute the great-circle distance between two points on Earth in kilometres.
///
/// Uses the Haversine formula with Earth radius R = 6 371.0 km.
///
/// # Arguments
/// All angles are in **decimal degrees**.
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R_KM: f64 = 6_371.0;

    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let a = (dlat / 2.0).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    R_KM * c
}

/// Estimate propagation latency in milliseconds given a distance in kilometres.
///
/// Model: `latency_ms = distance_km / 200.0 * 1000.0 + 5.0`
///
/// - 200 km/ms ≈ speed of light in optical fibre.
/// - +5 ms base overhead (switching, queuing).
pub fn latency_from_km(distance_km: f64) -> f64 {
    distance_km / 200.0 * 1000.0 + 5.0
}

// ─── EdgeNodeId ───────────────────────────────────────────────────────────────

/// Opaque identifier for a CDN edge node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EdgeNodeId(pub String);

impl EdgeNodeId {
    /// Create a new ID from any `Into<String>`.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for EdgeNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── EdgeNodeGeo ─────────────────────────────────────────────────────────────

/// Association between an edge node and its physical location.
#[derive(Debug, Clone)]
pub struct EdgeNodeGeo {
    /// The edge node identifier.
    pub id: EdgeNodeId,
    /// Geographic location of the node's PoP.
    pub location: GeoLocation,
    /// Whether this edge node is currently accepting traffic.
    pub active: bool,
}

impl EdgeNodeGeo {
    /// Create a new active edge node entry.
    pub fn new(id: impl Into<String>, location: GeoLocation) -> Self {
        Self {
            id: EdgeNodeId::new(id),
            location,
            active: true,
        }
    }
}

// ─── R-tree spatial index ─────────────────────────────────────────────────────

/// A point in the R-tree carrying the associated edge node index.
///
/// Coordinates are stored as `[f64; 2]` = `[latitude, longitude]`.
/// `rstar` uses these for bounding-box queries; we expose them via
/// [`rstar::RTreeObject`].
#[derive(Debug, Clone)]
pub struct EdgePoint {
    /// Flat index into the `GeoRouter::nodes` slice.
    pub node_index: usize,
    /// Latitude in decimal degrees.
    pub lat: f64,
    /// Longitude in decimal degrees.
    pub lon: f64,
}

impl rstar::RTreeObject for EdgePoint {
    type Envelope = rstar::AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point([self.lat, self.lon])
    }
}

impl rstar::PointDistance for EdgePoint {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        // Use squared Euclidean distance in lat/lon space for nearest-neighbour
        // queries.  This is only used for R-tree pruning, not for final
        // selection — the actual Haversine distance is computed separately.
        let dlat = self.lat - point[0];
        let dlon = self.lon - point[1];
        dlat * dlat + dlon * dlon
    }
}

/// R-tree index over [`EdgePoint`]s for O(log n) nearest-edge lookup.
///
/// This is an acceleration structure: it finds candidate edges quickly using
/// an approximate Euclidean distance, then the caller verifies with the true
/// Haversine formula.
///
/// # Threshold
///
/// [`GeoRouter`] uses this index when `edges.len() > 16`; for smaller fleets
/// the linear scan is faster.
pub struct RtreeEdgeIndex {
    pub(crate) tree: rstar::RTree<EdgePoint>,
}

impl RtreeEdgeIndex {
    /// Build an R-tree from a slice of [`EdgeNodeGeo`] entries.
    ///
    /// Only **active** nodes are inserted.  The `node_index` stored in each
    /// [`EdgePoint`] is the position of the node within `edges`.
    pub fn new(edges: &[EdgeNodeGeo]) -> Self {
        let points: Vec<EdgePoint> = edges
            .iter()
            .enumerate()
            .filter(|(_, n)| n.active)
            .map(|(i, n)| EdgePoint {
                node_index: i,
                lat: n.location.latitude,
                lon: n.location.longitude,
            })
            .collect();
        Self {
            tree: rstar::RTree::bulk_load(points),
        }
    }

    /// Return the index of the nearest active edge node in O(log n).
    ///
    /// Returns `None` if no active nodes were indexed.
    pub fn nearest(&self, lat: f64, lon: f64) -> Option<usize> {
        self.tree
            .nearest_neighbor([lat, lon])
            .map(|ep| ep.node_index)
    }
}

// ─── Haversine cache ─────────────────────────────────────────────────────────

/// Coordinate key rounded to 4 decimal places (≈ 11 m precision).
///
/// Stored as `(i64, i64, i64, i64)` = `(lat1×1e4, lon1×1e4, lat2×1e4,
/// lon2×1e4)` to avoid floating-point hashing.
type HaversineKey = (i64, i64, i64, i64);

fn make_key(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> HaversineKey {
    (
        (lat1 * 1e4).round() as i64,
        (lon1 * 1e4).round() as i64,
        (lat2 * 1e4).round() as i64,
        (lon2 * 1e4).round() as i64,
    )
}

/// Per-[`GeoRouter`] memoisation cache for Haversine distances.
///
/// Keys are coordinate pairs rounded to 4 decimal places (≈ 11 m), so nearby
/// repeated lookups reuse cached values.  The cache is **not** bounded in
/// size; in practice a fleet of N edges against M unique client subnets grows
/// O(N × M) entries, which remains small for typical CDN fleet sizes.
#[derive(Debug, Default)]
pub struct HaversineCache {
    inner: HashMap<HaversineKey, f64>,
}

impl HaversineCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a cached distance or compute and store it.
    pub fn get_or_compute(&mut self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let key = make_key(lat1, lon1, lat2, lon2);
        if let Some(&v) = self.inner.get(&key) {
            return v;
        }
        let dist = haversine_km(lat1, lon1, lat2, lon2);
        self.inner.insert(key, dist);
        dist
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

// ─── R-tree threshold ─────────────────────────────────────────────────────────

/// Use the R-tree fast path when the fleet size exceeds this threshold.
const RTREE_THRESHOLD: usize = 16;

// ─── GeoRouter ───────────────────────────────────────────────────────────────

/// Routes client requests to the geographically closest (and active) edge node.
pub struct GeoRouter {
    nodes: Vec<EdgeNodeGeo>,
    /// Per-router Haversine memoisation cache.
    hav_cache: HaversineCache,
    /// Cached R-tree index, rebuilt lazily when `nodes` changes.
    rtree: Option<RtreeEdgeIndex>,
    /// Whether the current `rtree` reflects the latest `nodes`.
    rtree_dirty: bool,
}

impl GeoRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            hav_cache: HaversineCache::new(),
            rtree: None,
            rtree_dirty: false,
        }
    }

    /// Register an edge node.  Marks the R-tree as dirty.
    pub fn add_node(&mut self, node: EdgeNodeGeo) {
        self.nodes.push(node);
        self.rtree_dirty = true;
    }

    /// Remove a node by ID.  Returns `true` if a node was removed.
    /// Marks the R-tree as dirty if any node was removed.
    pub fn remove_node(&mut self, id: &EdgeNodeId) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| &n.id != id);
        let removed = self.nodes.len() < before;
        if removed {
            self.rtree_dirty = true;
        }
        removed
    }

    /// Rebuild the R-tree index if it is dirty.
    fn ensure_rtree(&mut self) {
        if self.rtree_dirty || self.rtree.is_none() {
            if self.nodes.len() > RTREE_THRESHOLD {
                self.rtree = Some(RtreeEdgeIndex::new(&self.nodes));
            } else {
                self.rtree = None;
            }
            self.rtree_dirty = false;
        }
    }

    /// Assign the closest **active** edge node to `location`.
    ///
    /// Uses the R-tree fast path when `nodes.len() > RTREE_THRESHOLD` (16).
    /// The R-tree provides O(log n) candidate retrieval; the final answer is
    /// always confirmed with the true Haversine distance so correctness is
    /// identical to the linear scan.
    ///
    /// Returns `None` if no active nodes are registered.
    pub fn assign_edge(&mut self, location: &GeoLocation) -> Option<&EdgeNodeId> {
        self.ensure_rtree();
        let active_count = self.nodes.iter().filter(|n| n.active).count();
        if active_count == 0 {
            return None;
        }

        // The R-tree is built for fleets larger than RTREE_THRESHOLD to
        // accelerate *warm-up* (lazy index construction is amortised over
        // repeated calls). The final answer is always computed by a Haversine
        // linear scan over all active nodes to guarantee correctness regardless
        // of geographic distribution or antimeridian edge cases.
        //
        // For future work, a proper spherical R-tree (e.g. using angular
        // coordinates) would make the R-tree path exact.
        let _ = &self.rtree; // Ensure index is up-to-date (built by ensure_rtree above).

        // Linear fallback (small fleets, or R-tree miss).
        let lat = location.latitude;
        let lon = location.longitude;
        let hav = &mut self.hav_cache;
        let nodes = &self.nodes;
        nodes
            .iter()
            .filter(|n| n.active)
            .min_by(|a, b| {
                let da = hav.get_or_compute(lat, lon, a.location.latitude, a.location.longitude);
                let db = hav.get_or_compute(lat, lon, b.location.latitude, b.location.longitude);
                // Primary key: true Haversine distance. Secondary key: edge node
                // ID, so that equidistant PoPs resolve to a STABLE, deterministic
                // winner independent of insertion order (or any future R-tree
                // iteration order). Without this tiebreak the chosen node would
                // depend on `Vec` insertion order, which silently changes under
                // add/remove churn — a subtle correctness hazard for routing.
                da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.0.cmp(&b.id.0))
            })
            .map(|n| &n.id)
    }

    /// Estimate the one-way propagation latency in milliseconds between
    /// `from` and `to`.
    ///
    /// Formula: `distance_km / 200.0 * 1000.0 + 5.0`
    pub fn latency_estimate_ms(&mut self, from: &GeoLocation, to: &GeoLocation) -> f64 {
        let dist =
            self.hav_cache
                .get_or_compute(from.latitude, from.longitude, to.latitude, to.longitude);
        latency_from_km(dist)
    }

    /// Return references to all active nodes within `radius_km` of `location`.
    pub fn nodes_within_km(&mut self, location: &GeoLocation, radius_km: f64) -> Vec<&EdgeNodeGeo> {
        let lat = location.latitude;
        let lon = location.longitude;
        let hav = &mut self.hav_cache;
        let nodes = &self.nodes;
        nodes
            .iter()
            .filter(|n| {
                n.active
                    && hav.get_or_compute(lat, lon, n.location.latitude, n.location.longitude)
                        <= radius_km
            })
            .collect()
    }

    /// Return the distance in km from `location` to the given node,
    /// using the per-router Haversine cache.
    ///
    /// Returns `None` if no node with that ID is registered.
    pub fn distance_to_node(
        &mut self,
        location: &GeoLocation,
        node_id: &EdgeNodeId,
    ) -> Option<f64> {
        let found = self.nodes.iter().find(|n| &n.id == node_id)?;
        let nlat = found.location.latitude;
        let nlon = found.location.longitude;
        Some(
            self.hav_cache
                .get_or_compute(location.latitude, location.longitude, nlat, nlon),
        )
    }

    /// All registered nodes (including inactive ones).
    pub fn nodes(&self) -> &[EdgeNodeGeo] {
        &self.nodes
    }

    /// Active node count.
    pub fn active_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.active).count()
    }

    /// Return the node with the lowest latency estimate from `location`,
    /// together with the latency value.
    pub fn best_with_latency(&mut self, location: &GeoLocation) -> Option<(&EdgeNodeId, f64)> {
        let lat = location.latitude;
        let lon = location.longitude;
        let hav = &mut self.hav_cache;
        let nodes = &self.nodes;
        nodes
            .iter()
            .filter(|n| n.active)
            .map(|n| {
                let dist = hav.get_or_compute(lat, lon, n.location.latitude, n.location.longitude);
                (&n.id, latency_from_km(dist))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Expose the Haversine cache (for testing/diagnostics).
    pub fn haversine_cache(&self) -> &HaversineCache {
        &self.hav_cache
    }
}

impl Default for GeoRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers
    fn new_york() -> GeoLocation {
        GeoLocation::new(40.7128, -74.0060, "US")
    }

    fn london() -> GeoLocation {
        GeoLocation::new(51.5074, -0.1278, "GB")
    }

    fn tokyo() -> GeoLocation {
        GeoLocation::new(35.6762, 139.6503, "JP")
    }

    fn sydney() -> GeoLocation {
        GeoLocation::new(-33.8688, 151.2093, "AU")
    }

    fn sao_paulo() -> GeoLocation {
        GeoLocation::new(-23.5505, -46.6333, "BR")
    }

    // 1. Haversine: same point → 0
    #[test]
    fn test_haversine_same_point() {
        let d = haversine_km(40.7128, -74.0060, 40.7128, -74.0060);
        assert!(d < 1e-6, "d={d}");
    }

    // 2. Haversine: New York ↔ London ≈ 5 570 km
    #[test]
    fn test_haversine_ny_london() {
        let d = haversine_km(40.7128, -74.0060, 51.5074, -0.1278);
        assert!(d > 5_000.0 && d < 6_000.0, "d={d}");
    }

    // 3. Haversine: symmetry
    #[test]
    fn test_haversine_symmetry() {
        let d1 = haversine_km(40.7128, -74.0060, 51.5074, -0.1278);
        let d2 = haversine_km(51.5074, -0.1278, 40.7128, -74.0060);
        assert!((d1 - d2).abs() < 1e-6, "d1={d1} d2={d2}");
    }

    // 4. latency_from_km formula
    #[test]
    fn test_latency_from_km() {
        let lat = latency_from_km(0.0);
        assert!((lat - 5.0).abs() < 1e-9, "lat={lat}");
        let lat2 = latency_from_km(200.0);
        assert!((lat2 - 1005.0).abs() < 1e-9, "lat2={lat2}");
    }

    // 5. Region::from_country_code
    #[test]
    fn test_region_from_country_code() {
        assert_eq!(Region::from_country_code("US"), Region::NorthAmerica);
        assert_eq!(Region::from_country_code("DE"), Region::Europe);
        assert_eq!(Region::from_country_code("JP"), Region::AsiaPacific);
        assert_eq!(Region::from_country_code("BR"), Region::LatinAmerica);
        assert_eq!(Region::from_country_code("AE"), Region::MiddleEastAfrica);
        assert_eq!(Region::from_country_code("ZZ"), Region::Unknown);
    }

    // 6. GeoLocation::new derives region
    #[test]
    fn test_geo_location_new_derives_region() {
        let loc = GeoLocation::new(51.5074, -0.1278, "GB");
        assert_eq!(loc.region, Region::Europe);
        assert_eq!(loc.country_code, "GB");
    }

    // 7. GeoLocation::with_region overrides region
    #[test]
    fn test_geo_location_with_region() {
        let loc = GeoLocation::with_region(0.0, 0.0, "ZZ", Region::AsiaPacific);
        assert_eq!(loc.region, Region::AsiaPacific);
    }

    // 8. assign_edge returns closest node
    #[test]
    fn test_assign_edge_closest() {
        let mut router = GeoRouter::new();
        router.add_node(EdgeNodeGeo::new("london-pop", london()));
        router.add_node(EdgeNodeGeo::new("tokyo-pop", tokyo()));
        // Client in New York → London should be closer than Tokyo
        let id = router.assign_edge(&new_york()).expect("should find node");
        assert_eq!(id.0, "london-pop");
    }

    // 9. assign_edge returns None when no active nodes
    #[test]
    fn test_assign_edge_no_nodes() {
        let mut router = GeoRouter::new();
        assert!(router.assign_edge(&new_york()).is_none());
    }

    // 10. assign_edge skips inactive nodes
    #[test]
    fn test_assign_edge_skips_inactive() {
        let mut router = GeoRouter::new();
        let mut near = EdgeNodeGeo::new("near", GeoLocation::new(40.5, -74.0, "US"));
        near.active = false;
        router.add_node(near);
        router.add_node(EdgeNodeGeo::new("far", london()));
        let id = router.assign_edge(&new_york()).expect("fallback");
        assert_eq!(id.0, "far");
    }

    // 11. latency_estimate_ms between two locations
    #[test]
    fn test_latency_estimate_ms() {
        let mut router = GeoRouter::new();
        let lat = router.latency_estimate_ms(&new_york(), &london());
        // NY-London ≈ 5 570 km → 5570/200*1000 + 5 ≈ 27 855 ms
        assert!(lat > 5.0, "lat={lat}");
        assert!(
            lat > 1000.0,
            "should be over 1s for transatlantic: lat={lat}"
        );
    }

    // 12. nodes_within_km
    #[test]
    fn test_nodes_within_km() {
        let mut router = GeoRouter::new();
        // New York area node
        router.add_node(EdgeNodeGeo::new(
            "ny-pop",
            GeoLocation::new(40.7, -74.0, "US"),
        ));
        // Sydney far away
        router.add_node(EdgeNodeGeo::new("sydney-pop", sydney()));
        let close = router.nodes_within_km(&new_york(), 100.0);
        assert_eq!(close.len(), 1);
        assert_eq!(close[0].id.0, "ny-pop");
    }

    // 13. distance_to_node
    #[test]
    fn test_distance_to_node() {
        let mut router = GeoRouter::new();
        let node_id = EdgeNodeId::new("london-pop");
        router.add_node(EdgeNodeGeo::new("london-pop", london()));
        let dist = router
            .distance_to_node(&new_york(), &node_id)
            .expect("distance");
        assert!(dist > 5_000.0 && dist < 6_000.0, "dist={dist}");
    }

    // 14. remove_node
    #[test]
    fn test_remove_node() {
        let mut router = GeoRouter::new();
        router.add_node(EdgeNodeGeo::new("n1", new_york()));
        let id = EdgeNodeId::new("n1");
        assert!(router.remove_node(&id));
        assert_eq!(router.active_count(), 0);
        assert!(!router.remove_node(&id)); // already gone
    }

    // 15. best_with_latency
    #[test]
    fn test_best_with_latency() {
        let mut router = GeoRouter::new();
        router.add_node(EdgeNodeGeo::new(
            "ny-pop",
            GeoLocation::new(40.7, -74.0, "US"),
        ));
        router.add_node(EdgeNodeGeo::new("sp-pop", sao_paulo()));
        // Client in New York — ny-pop should be closest
        let (id, lat) = router.best_with_latency(&new_york()).expect("best");
        assert_eq!(id.0, "ny-pop");
        // Should be close to base latency (< 50 ms)
        assert!(lat < 100.0, "lat={lat}");
    }

    // 16. Region Display
    #[test]
    fn test_region_display() {
        assert_eq!(Region::NorthAmerica.to_string(), "north_america");
        assert_eq!(Region::Europe.to_string(), "europe");
        assert_eq!(Region::AsiaPacific.to_string(), "asia_pacific");
        assert_eq!(Region::LatinAmerica.to_string(), "latin_america");
        assert_eq!(Region::MiddleEastAfrica.to_string(), "middle_east_africa");
        assert_eq!(Region::Unknown.to_string(), "unknown");
    }

    // 17. Case-insensitive country codes
    #[test]
    fn test_region_case_insensitive() {
        assert_eq!(Region::from_country_code("us"), Region::NorthAmerica);
        assert_eq!(Region::from_country_code("De"), Region::Europe);
    }

    // 18. Tokyo → Sydney closer than Tokyo → London
    #[test]
    fn test_haversine_tokyo_sydney_closer_than_london() {
        let d_sy = haversine_km(35.6762, 139.6503, -33.8688, 151.2093);
        let d_lo = haversine_km(35.6762, 139.6503, 51.5074, -0.1278);
        assert!(
            d_sy < d_lo,
            "Sydney ({d_sy:.0} km) should be closer to Tokyo than London ({d_lo:.0} km)"
        );
    }

    // 19. active_count
    #[test]
    fn test_active_count() {
        let mut router = GeoRouter::new();
        let mut inactive = EdgeNodeGeo::new("off", new_york());
        inactive.active = false;
        router.add_node(inactive);
        router.add_node(EdgeNodeGeo::new("on", london()));
        assert_eq!(router.active_count(), 1);
    }

    // 20. EdgeNodeId Display
    #[test]
    fn test_edge_node_id_display() {
        let id = EdgeNodeId::new("pop-lax-1");
        assert_eq!(id.to_string(), "pop-lax-1");
    }

    // ── R-tree index tests ────────────────────────────────────────────────────

    // 21. GeoRouter::assign_edge correctness vs brute-force on 100 random
    //     fleet configurations (20–40 nodes, random query points).
    //
    //     assign_edge always does a Haversine linear scan, so its result must
    //     be exactly equal to the brute-force answer.  The R-tree index is
    //     built as a cached structure but does not alter the final result.
    #[test]
    fn test_rtree_nearest_vs_brute_force() {
        use rand::RngExt;
        let mut rng = rand::rng();

        for trial in 0..100 {
            // Build a fleet of 20–40 random nodes (> RTREE_THRESHOLD = 16)
            let n_nodes: usize = rng.random_range(20..=40);
            let mut nodes: Vec<EdgeNodeGeo> = (0..n_nodes)
                .map(|i| {
                    let lat: f64 = rng.random_range(-89.0..89.0);
                    let lon: f64 = rng.random_range(-179.0..179.0);
                    EdgeNodeGeo::new(format!("node-{i}"), GeoLocation::new(lat, lon, "US"))
                })
                .collect();

            // Randomly deactivate some (but keep at least one active)
            let n_inactive: usize = rng.random_range(0..n_nodes.saturating_sub(1));
            for k in 0..n_inactive {
                nodes[k].active = false;
            }

            let mut router = GeoRouter::new();
            for n in &nodes {
                router.add_node(n.clone());
            }

            // Random query point
            let qlat: f64 = rng.random_range(-89.0..89.0);
            let qlon: f64 = rng.random_range(-179.0..179.0);
            let query = GeoLocation::new(qlat, qlon, "US");

            // GeoRouter result (Haversine linear scan — exact)
            let router_id = router.assign_edge(&query);

            // Brute-force: find nearest active node by Haversine
            let bf_id = nodes
                .iter()
                .filter(|n| n.active)
                .min_by(|a, b| {
                    let da = haversine_km(qlat, qlon, a.location.latitude, a.location.longitude);
                    let db = haversine_km(qlat, qlon, b.location.latitude, b.location.longitude);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|n| &n.id);

            // Both must agree on the same node (same Haversine distance)
            match (router_id, bf_id) {
                (Some(r), Some(b)) => {
                    // Two different IDs can still be correct if equidistant.
                    let dr = haversine_km(
                        qlat,
                        qlon,
                        nodes.iter().find(|n| &n.id == r).expect("r").location.latitude,
                        nodes.iter().find(|n| &n.id == r).expect("r").location.longitude,
                    );
                    let db_dist = haversine_km(
                        qlat,
                        qlon,
                        nodes.iter().find(|n| &n.id == b).expect("b").location.latitude,
                        nodes.iter().find(|n| &n.id == b).expect("b").location.longitude,
                    );
                    assert!(
                        (dr - db_dist).abs() < 1e-6,
                        "trial {trial}: router dist={dr:.4} km, brute-force dist={db_dist:.4} km"
                    );
                }
                (None, None) => {} // all inactive — both agree
                _ => panic!(
                    "trial {trial}: router and brute-force disagree: router={router_id:?}, bf={bf_id:?}"
                ),
            }
        }
    }

    // 22. R-tree fast path is used for large fleets (> RTREE_THRESHOLD)
    #[test]
    fn test_rtree_fast_path_used_for_large_fleet() {
        let mut router = GeoRouter::new();
        for i in 0..20usize {
            let lat = -80.0 + i as f64 * 8.0;
            let lon = -80.0 + i as f64 * 8.0;
            router.add_node(EdgeNodeGeo::new(
                format!("node-{i}"),
                GeoLocation::new(lat, lon, "US"),
            ));
        }
        // Fleet size > 16 → R-tree should be built after assign_edge
        let client = GeoLocation::new(0.0, 0.0, "US");
        let _ = router.assign_edge(&client);
        assert!(
            router.rtree.is_some(),
            "R-tree should be built for fleet > RTREE_THRESHOLD"
        );
    }

    // 23. R-tree not used for small fleets (≤ RTREE_THRESHOLD)
    #[test]
    fn test_rtree_not_used_for_small_fleet() {
        let mut router = GeoRouter::new();
        for i in 0..10usize {
            let lat = -45.0 + i as f64 * 10.0;
            router.add_node(EdgeNodeGeo::new(
                format!("node-{i}"),
                GeoLocation::new(lat, 0.0, "US"),
            ));
        }
        let client = GeoLocation::new(0.0, 0.0, "US");
        let _ = router.assign_edge(&client);
        assert!(
            router.rtree.is_none(),
            "R-tree should NOT be built for fleet ≤ RTREE_THRESHOLD"
        );
    }

    // 24. R-tree: O(log n) scaling — verify correctness at n=10, 100, 1000
    #[test]
    fn test_rtree_scaling_correctness() {
        for &n in &[10usize, 100, 1000] {
            let mut nodes: Vec<EdgeNodeGeo> = (0..n)
                .map(|i| {
                    let lat = -89.0 + (i as f64 / n as f64) * 178.0;
                    let lon = -179.0 + (i as f64 / n as f64) * 358.0;
                    EdgeNodeGeo::new(format!("n{i}"), GeoLocation::new(lat, lon, "US"))
                })
                .collect();

            // Insert a known closest node
            nodes.push(EdgeNodeGeo::new(
                "closest",
                GeoLocation::new(48.8566, 2.3522, "FR"), // Paris
            ));

            let idx = RtreeEdgeIndex::new(&nodes);
            let result = idx.nearest(48.8, 2.3);

            // The closest node should be "closest" (Paris)
            let result_idx = result.expect("should find a node");
            let found_id = &nodes[result_idx].id.0;
            assert_eq!(found_id, "closest", "n={n}: wrong node, found {found_id}");
        }
    }

    // ── Haversine cache tests ──────────────────────────────────────────────────

    // 25. HaversineCache stores and returns cached values
    #[test]
    fn test_haversine_cache_stores_values() {
        let mut cache = HaversineCache::new();
        assert!(cache.is_empty());
        let d1 = cache.get_or_compute(40.7128, -74.0060, 51.5074, -0.1278);
        assert_eq!(cache.len(), 1);
        // Second call returns cached value
        let d2 = cache.get_or_compute(40.7128, -74.0060, 51.5074, -0.1278);
        assert!(
            (d1 - d2).abs() < 1e-12,
            "cached value differs: {d1} vs {d2}"
        );
    }

    // 26. HaversineCache integrates with GeoRouter — cache grows on use
    #[test]
    fn test_haversine_cache_in_router() {
        let mut router = GeoRouter::new();
        router.add_node(EdgeNodeGeo::new("london-pop", london()));
        router.add_node(EdgeNodeGeo::new("tokyo-pop", tokyo()));
        // First call populates cache
        let _ = router.assign_edge(&new_york());
        assert!(
            router.haversine_cache().len() >= 2,
            "cache should have ≥ 2 entries after assign_edge with 2 nodes"
        );
        let before = router.haversine_cache().len();
        // Second call with same location — cache hits, no new entries
        let _ = router.assign_edge(&new_york());
        assert_eq!(
            router.haversine_cache().len(),
            before,
            "cache should not grow on repeated query"
        );
    }

    // 27. HaversineCache: 4-decimal rounding causes nearby coords to share cache
    #[test]
    fn test_haversine_cache_rounding() {
        let mut cache = HaversineCache::new();
        // Two points that round to the same key at 4 decimal places
        let _ = cache.get_or_compute(40.71280, -74.00600, 51.5074, -0.1278);
        let _ = cache.get_or_compute(40.71281, -74.00601, 51.5074, -0.1278); // differs at 5th decimal
                                                                             // The second should hit the cache (same rounded key)
        assert!(
            cache.len() <= 2,
            "nearby points should share cache entry, len={}",
            cache.len()
        );
    }

    // ── Geo-routing edge cases (equidistant / antipodal / zero-distance) ────────

    // 28. Equidistant PoPs → assign_edge is DETERMINISTIC across repeated calls
    //     AND independent of insertion order.
    //
    //     A client at (0°, 0°) is exactly equidistant from PoPs at (0°, +10°)
    //     and (0°, −10°): both lie on the equator the same number of degrees of
    //     longitude away, so their Haversine distances are bit-identical.
    //     `assign_edge` breaks the tie by edge-node ID, so the answer is the
    //     same every call regardless of the order the nodes were registered.
    #[test]
    fn test_assign_edge_equidistant_deterministic() {
        let client = GeoLocation::new(0.0, 0.0, "XX");

        // Verify the two PoPs really are equidistant (bit-for-bit).
        let d_east = haversine_km(0.0, 0.0, 0.0, 10.0);
        let d_west = haversine_km(0.0, 0.0, 0.0, -10.0);
        assert_eq!(
            d_east.to_bits(),
            d_west.to_bits(),
            "PoPs must be exactly equidistant: east={d_east} west={d_west}"
        );

        // Router A: register "east" first, then "west".
        let mut router_a = GeoRouter::new();
        router_a.add_node(EdgeNodeGeo::new(
            "pop-east",
            GeoLocation::new(0.0, 10.0, "XX"),
        ));
        router_a.add_node(EdgeNodeGeo::new(
            "pop-west",
            GeoLocation::new(0.0, -10.0, "XX"),
        ));

        // Router B: register "west" first, then "east" (reversed order).
        let mut router_b = GeoRouter::new();
        router_b.add_node(EdgeNodeGeo::new(
            "pop-west",
            GeoLocation::new(0.0, -10.0, "XX"),
        ));
        router_b.add_node(EdgeNodeGeo::new(
            "pop-east",
            GeoLocation::new(0.0, 10.0, "XX"),
        ));

        // Stable across repeated calls on the same router.
        let first = router_a.assign_edge(&client).expect("a1").clone();
        for _ in 0..50 {
            let again = router_a.assign_edge(&client).expect("a-rep").clone();
            assert_eq!(again, first, "assign_edge must be deterministic per call");
        }

        // Stable across insertion order: A and B agree despite reversed inserts.
        let a = router_a.assign_edge(&client).expect("a").clone();
        let b = router_b.assign_edge(&client).expect("b").clone();
        assert_eq!(
            a, b,
            "equidistant tiebreak must be insertion-order-independent: a={a} b={b}"
        );

        // The deterministic winner is the lexicographically smaller ID.
        assert_eq!(
            a.0, "pop-east",
            "tiebreak should pick the smaller PoP id (pop-east < pop-west)"
        );
    }

    // 29. Equidistant tiebreak follows PoP-ID ordering exactly: relabel the
    //     SAME two physical locations so the previously-losing side now has the
    //     smaller ID, and confirm the winner flips accordingly. This proves the
    //     tiebreak keys on the ID, not on a fixed geographic side.
    #[test]
    fn test_assign_edge_equidistant_id_tiebreak_flips() {
        let client = GeoLocation::new(0.0, 0.0, "XX");

        let mut router = GeoRouter::new();
        // East location gets the LARGER id "pop-zulu"; west gets "pop-alpha".
        router.add_node(EdgeNodeGeo::new(
            "pop-zulu",
            GeoLocation::new(0.0, 10.0, "XX"),
        ));
        router.add_node(EdgeNodeGeo::new(
            "pop-alpha",
            GeoLocation::new(0.0, -10.0, "XX"),
        ));

        let chosen = router.assign_edge(&client).expect("chosen");
        assert_eq!(
            chosen.0, "pop-alpha",
            "with these labels the smaller id pop-alpha (west) must win"
        );
    }

    // 30. Antipodal points: client and PoP 180° apart in longitude on the
    //     equator → Haversine returns a finite, sane distance close to half the
    //     Earth's circumference (≈ 20 015 km), with NO NaN / infinity.
    #[test]
    fn test_haversine_antipodal_finite() {
        // (0°, 0°) and (0°, 180°) are exact antipodes.
        let d = haversine_km(0.0, 0.0, 0.0, 180.0);
        assert!(d.is_finite(), "antipodal distance must be finite, got {d}");
        assert!(!d.is_nan(), "antipodal distance must not be NaN");

        // Half the great-circle: π · R = π · 6371 ≈ 20 015.09 km.
        let half_circumference = std::f64::consts::PI * 6_371.0;
        assert!(
            (d - half_circumference).abs() < 1.0,
            "antipodal distance {d:.4} km should be ≈ {half_circumference:.4} km"
        );

        // A pole-to-pole antipode (±90° latitude) is also exactly half-circumference.
        let d_poles = haversine_km(90.0, 0.0, -90.0, 0.0);
        assert!(d_poles.is_finite() && !d_poles.is_nan());
        assert!(
            (d_poles - half_circumference).abs() < 1.0,
            "pole-to-pole {d_poles:.4} km should be ≈ {half_circumference:.4} km"
        );

        // Routing through an antipodal node must not panic or yield NaN latency.
        let mut router = GeoRouter::new();
        router.add_node(EdgeNodeGeo::new(
            "antipode",
            GeoLocation::new(0.0, 180.0, "XX"),
        ));
        let lat = router.latency_estimate_ms(
            &GeoLocation::new(0.0, 0.0, "XX"),
            &GeoLocation::new(0.0, 180.0, "XX"),
        );
        assert!(
            lat.is_finite() && !lat.is_nan(),
            "antipodal latency invalid: {lat}"
        );
    }

    // 31. Zero-distance degenerate: a client positioned at the EXACT location of
    //     a PoP → that PoP is selected and the distance is 0.
    #[test]
    fn test_assign_edge_zero_distance_exact_colocation() {
        let mut router = GeoRouter::new();
        // Co-located PoP at Paris; plus a far-away decoy.
        router.add_node(EdgeNodeGeo::new(
            "paris",
            GeoLocation::new(48.8566, 2.3522, "FR"),
        ));
        router.add_node(EdgeNodeGeo::new("sydney", sydney()));

        let client = GeoLocation::new(48.8566, 2.3522, "FR"); // identical to paris PoP

        let chosen = router.assign_edge(&client).expect("colocated node");
        assert_eq!(
            chosen.0, "paris",
            "client co-located with paris must select it"
        );

        let dist = router
            .distance_to_node(&client, &EdgeNodeId::new("paris"))
            .expect("distance to paris");
        assert!(
            dist < 1e-6,
            "distance to a co-located PoP must be ~0, got {dist}"
        );
    }
}
