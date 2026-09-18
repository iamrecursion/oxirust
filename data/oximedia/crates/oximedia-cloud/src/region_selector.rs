#![allow(dead_code)]
//! Cloud region selection and latency modelling.
//!
//! [`RegionSelector`] picks the best [`AwsRegion`] for a client at a given
//! origin latitude/longitude based on estimated network latency.  The latency
//! model is a simple great-circle approximation — suitable for pre-flight
//! planning and CDN origin selection rather than precise RTT prediction.

// ─────────────────────────────────────────────────────────────────────────────
// AwsRegion
// ─────────────────────────────────────────────────────────────────────────────

/// AWS geographic region identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AwsRegion {
    /// US East (N. Virginia) — `us-east-1`.
    UsEast1,
    /// US East (Ohio) — `us-east-2`.
    UsEast2,
    /// US West (N. California) — `us-west-1`.
    UsWest1,
    /// US West (Oregon) — `us-west-2`.
    UsWest2,
    /// Europe (Ireland) — `eu-west-1`.
    EuWest1,
    /// Europe (London) — `eu-west-2`.
    EuWest2,
    /// Europe (Frankfurt) — `eu-central-1`.
    EuCentral1,
    /// Asia Pacific (Tokyo) — `ap-northeast-1`.
    ApNortheast1,
    /// Asia Pacific (Seoul) — `ap-northeast-2`.
    ApNortheast2,
    /// Asia Pacific (Singapore) — `ap-southeast-1`.
    ApSoutheast1,
    /// Asia Pacific (Sydney) — `ap-southeast-2`.
    ApSoutheast2,
    /// South America (São Paulo) — `sa-east-1`.
    SaEast1,
}

impl AwsRegion {
    /// The canonical region identifier string (e.g. `"us-east-1"`).
    pub fn id(self) -> &'static str {
        match self {
            Self::UsEast1 => "us-east-1",
            Self::UsEast2 => "us-east-2",
            Self::UsWest1 => "us-west-1",
            Self::UsWest2 => "us-west-2",
            Self::EuWest1 => "eu-west-1",
            Self::EuWest2 => "eu-west-2",
            Self::EuCentral1 => "eu-central-1",
            Self::ApNortheast1 => "ap-northeast-1",
            Self::ApNortheast2 => "ap-northeast-2",
            Self::ApSoutheast1 => "ap-southeast-1",
            Self::ApSoutheast2 => "ap-southeast-2",
            Self::SaEast1 => "sa-east-1",
        }
    }

    /// Approximate geographic centre of the region as `(latitude, longitude)`
    /// in decimal degrees.
    pub fn coordinates(self) -> (f64, f64) {
        match self {
            Self::UsEast1 => (39.0, -77.5),
            Self::UsEast2 => (40.0, -82.9),
            Self::UsWest1 => (37.3, -121.9),
            Self::UsWest2 => (45.8, -119.7),
            Self::EuWest1 => (53.3, -6.2),
            Self::EuWest2 => (51.5, -0.1),
            Self::EuCentral1 => (50.1, 8.7),
            Self::ApNortheast1 => (35.7, 139.7),
            Self::ApNortheast2 => (37.6, 127.0),
            Self::ApSoutheast1 => (1.3, 103.8),
            Self::ApSoutheast2 => (-33.9, 151.2),
            Self::SaEast1 => (-23.5, -46.6),
        }
    }

    /// Baseline (best-case) RTT in milliseconds from infrastructure within the
    /// same region.  Used as a floor when estimating latency.
    pub fn latency_ms(self) -> f64 {
        match self {
            Self::UsEast1 | Self::UsEast2 => 1.0,
            Self::UsWest1 | Self::UsWest2 => 1.5,
            Self::EuWest1 | Self::EuWest2 | Self::EuCentral1 => 2.0,
            Self::ApNortheast1 | Self::ApNortheast2 => 3.0,
            Self::ApSoutheast1 | Self::ApSoutheast2 => 4.0,
            Self::SaEast1 => 5.0,
        }
    }

    /// Human-readable display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::UsEast1 => "US East (N. Virginia)",
            Self::UsEast2 => "US East (Ohio)",
            Self::UsWest1 => "US West (N. California)",
            Self::UsWest2 => "US West (Oregon)",
            Self::EuWest1 => "Europe (Ireland)",
            Self::EuWest2 => "Europe (London)",
            Self::EuCentral1 => "Europe (Frankfurt)",
            Self::ApNortheast1 => "Asia Pacific (Tokyo)",
            Self::ApNortheast2 => "Asia Pacific (Seoul)",
            Self::ApSoutheast1 => "Asia Pacific (Singapore)",
            Self::ApSoutheast2 => "Asia Pacific (Sydney)",
            Self::SaEast1 => "South America (São Paulo)",
        }
    }

    /// All defined regions.
    pub fn all() -> &'static [Self] {
        &[
            Self::UsEast1,
            Self::UsEast2,
            Self::UsWest1,
            Self::UsWest2,
            Self::EuWest1,
            Self::EuWest2,
            Self::EuCentral1,
            Self::ApNortheast1,
            Self::ApNortheast2,
            Self::ApSoutheast1,
            Self::ApSoutheast2,
            Self::SaEast1,
        ]
    }
}

impl std::fmt::Display for AwsRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Latency estimation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Haversine great-circle distance in kilometres between two lat/lon points.
#[allow(clippy::cast_precision_loss)]
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371.0; // Earth radius in km
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    R * c
}

/// Estimate one-way propagation latency in milliseconds for a given distance.
///
/// Assumes a speed-of-light fraction of ~0.67 in fibre and adds a constant
/// overhead for routing and processing.
fn propagation_latency_ms(distance_km: f64) -> f64 {
    const SPEED_OF_LIGHT_KM_PER_MS: f64 = 299.792; // km/ms
    const FIBRE_FRACTION: f64 = 0.67;
    const OVERHEAD_MS: f64 = 5.0;
    let propagation = distance_km / (SPEED_OF_LIGHT_KM_PER_MS * FIBRE_FRACTION);
    propagation + OVERHEAD_MS
}

// ─────────────────────────────────────────────────────────────────────────────
// RegionSelector
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the lowest-latency AWS region for a given client location.
///
/// ```rust
/// use oximedia_cloud::region_selector::{AwsRegion, RegionSelector};
///
/// let selector = RegionSelector::default();
/// // Client near London
/// let (region, latency) = selector.select_nearest(51.5, -0.1);
/// assert_eq!(region, AwsRegion::EuWest2);
/// let _ = latency; // estimated RTT in ms
/// ```
#[derive(Debug, Clone, Default)]
pub struct RegionSelector {
    /// Candidate regions to consider (defaults to all regions when empty).
    candidates: Vec<AwsRegion>,
}

impl RegionSelector {
    /// Create a selector that considers all known regions.
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Restrict the selector to a specific subset of regions.
    pub fn with_candidates(candidates: Vec<AwsRegion>) -> Self {
        Self { candidates }
    }

    /// The set of regions currently considered.
    fn effective_candidates(&self) -> &[AwsRegion] {
        if self.candidates.is_empty() {
            AwsRegion::all()
        } else {
            &self.candidates
        }
    }

    /// Estimate round-trip latency from a client at `(lat, lon)` to `region`
    /// in milliseconds.
    pub fn estimate_latency_ms(&self, lat: f64, lon: f64, region: AwsRegion) -> f64 {
        let (rlat, rlon) = region.coordinates();
        let dist = haversine_km(lat, lon, rlat, rlon);
        // Round-trip = 2 × one-way propagation + region baseline.
        propagation_latency_ms(dist) * 2.0 + region.latency_ms()
    }

    /// Return the region with the lowest estimated RTT for a client at
    /// `(lat, lon)`, together with that estimated RTT in milliseconds.
    ///
    /// Falls back to [`AwsRegion::UsEast1`] with a large RTT estimate if the
    /// candidate list is somehow empty.
    pub fn select_nearest(&self, lat: f64, lon: f64) -> (AwsRegion, f64) {
        let mut best_region = AwsRegion::UsEast1;
        let mut best_latency = f64::MAX;

        for &region in self.effective_candidates() {
            let latency = self.estimate_latency_ms(lat, lon, region);
            if latency < best_latency {
                best_latency = latency;
                best_region = region;
            }
        }

        (best_region, best_latency)
    }

    /// Return all regions sorted by estimated latency for a client at
    /// `(lat, lon)`, lowest first.
    pub fn rank_regions(&self, lat: f64, lon: f64) -> Vec<(AwsRegion, f64)> {
        let mut ranked: Vec<(AwsRegion, f64)> = self
            .effective_candidates()
            .iter()
            .map(|&r| (r, self.estimate_latency_ms(lat, lon, r)))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_id() {
        assert_eq!(AwsRegion::UsEast1.id(), "us-east-1");
        assert_eq!(AwsRegion::ApNortheast1.id(), "ap-northeast-1");
    }

    #[test]
    fn test_region_display() {
        assert_eq!(AwsRegion::EuWest1.to_string(), "eu-west-1");
    }

    #[test]
    fn test_region_display_name() {
        assert!(AwsRegion::UsEast1.display_name().contains("Virginia"));
    }

    #[test]
    fn test_region_latency_ms_positive() {
        for &r in AwsRegion::all() {
            assert!(r.latency_ms() > 0.0, "latency should be positive for {r}");
        }
    }

    #[test]
    fn test_region_coordinates_bounds() {
        for &r in AwsRegion::all() {
            let (lat, lon) = r.coordinates();
            assert!((-90.0..=90.0).contains(&lat), "lat out of range for {r}");
            assert!((-180.0..=180.0).contains(&lon), "lon out of range for {r}");
        }
    }

    #[test]
    fn test_all_regions_count() {
        assert_eq!(AwsRegion::all().len(), 12);
    }

    #[test]
    fn test_haversine_same_point_is_zero() {
        let d = haversine_km(51.5, -0.1, 51.5, -0.1);
        assert!(d < 1e-6);
    }

    #[test]
    fn test_haversine_london_to_nyc_approx() {
        // London ↔ New York ≈ 5,570 km
        let d = haversine_km(51.5, -0.1, 40.7, -74.0);
        assert!((5_000.0..6_200.0).contains(&d), "distance was {d:.0} km");
    }

    #[test]
    fn test_propagation_latency_positive() {
        assert!(propagation_latency_ms(0.0) > 0.0);
        assert!(propagation_latency_ms(10_000.0) > propagation_latency_ms(1_000.0));
    }

    #[test]
    fn test_select_nearest_london_is_eu() {
        let selector = RegionSelector::new();
        let (region, _latency) = selector.select_nearest(51.5, -0.1);
        // London should resolve to one of the European regions.
        let eu_regions = [
            AwsRegion::EuWest1,
            AwsRegion::EuWest2,
            AwsRegion::EuCentral1,
        ];
        assert!(
            eu_regions.contains(&region),
            "expected EU region, got {region}"
        );
    }

    #[test]
    fn test_select_nearest_tokyo_is_ap() {
        let selector = RegionSelector::new();
        let (region, _) = selector.select_nearest(35.7, 139.7);
        let ap_regions = [
            AwsRegion::ApNortheast1,
            AwsRegion::ApNortheast2,
            AwsRegion::ApSoutheast1,
        ];
        assert!(
            ap_regions.contains(&region),
            "expected AP region, got {region}"
        );
    }

    #[test]
    fn test_rank_regions_sorted_ascending() {
        let selector = RegionSelector::new();
        let ranked = selector.rank_regions(39.0, -77.5); // near us-east-1
        for window in ranked.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }

    #[test]
    fn test_with_candidates_restricts_choice() {
        let selector =
            RegionSelector::with_candidates(vec![AwsRegion::SaEast1, AwsRegion::ApSoutheast2]);
        let (region, _) = selector.select_nearest(0.0, 0.0); // equator / prime meridian
        assert!(
            region == AwsRegion::SaEast1 || region == AwsRegion::ApSoutheast2,
            "got {region}"
        );
    }

    #[test]
    fn test_estimate_latency_same_datacenter() {
        let selector = RegionSelector::new();
        let (rlat, rlon) = AwsRegion::UsEast1.coordinates();
        let latency = selector.estimate_latency_ms(rlat, rlon, AwsRegion::UsEast1);
        // Should be close to 2 × overhead + baseline.
        assert!(
            latency < 30.0,
            "latency {latency:.1} ms higher than expected"
        );
    }

    #[test]
    fn test_rank_regions_length_matches_all() {
        let selector = RegionSelector::new();
        let ranked = selector.rank_regions(0.0, 0.0);
        assert_eq!(ranked.len(), AwsRegion::all().len());
    }
}
