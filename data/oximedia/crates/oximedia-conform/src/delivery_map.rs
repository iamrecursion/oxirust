//! Delivery mapping for conformed media.
//!
//! Maps a conformed media asset to one or more delivery destinations
//! (broadcast, streaming, cinema, social) and validates that the asset
//! meets each destination's requirements.

#![allow(dead_code)]

/// A delivery destination category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeliveryDestination {
    /// Terrestrial / satellite broadcast (e.g. ATSC, DVB).
    Broadcast,
    /// OTT / online streaming (e.g. Netflix, Amazon, Disney+).
    Streaming,
    /// Digital Cinema Package (DCP / IMF).
    Cinema,
    /// Social media platform (`YouTube`, Instagram, `TikTok`, etc.).
    Social,
}

impl DeliveryDestination {
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Broadcast => "Broadcast",
            Self::Streaming => "Streaming",
            Self::Cinema => "Cinema",
            Self::Social => "Social Media",
        }
    }

    /// Maximum allowed video bitrate in Mbit/s for this destination.
    #[must_use]
    pub fn max_bitrate_mbps(&self) -> f64 {
        match self {
            Self::Broadcast => 50.0,
            Self::Streaming => 25.0,
            Self::Cinema => 250.0,
            Self::Social => 8.0,
        }
    }

    /// Minimum required audio loudness (LUFS) target for the destination.
    #[must_use]
    pub fn loudness_target_lufs(&self) -> f64 {
        match self {
            Self::Broadcast => -24.0,
            Self::Streaming => -23.0,
            Self::Cinema => -26.0,
            Self::Social => -14.0,
        }
    }
}

/// A route entry mapping an asset identifier to a destination with metadata.
#[derive(Clone, Debug)]
pub struct DeliveryRoute {
    /// Unique asset identifier (e.g. clip UUID or reel name).
    pub asset_id: String,
    /// Target delivery destination.
    pub destination: DeliveryDestination,
    /// Optional version / variant label (e.g. "SDR", "HDR10").
    pub variant: Option<String>,
    /// Bitrate assigned to this route in Mbit/s.
    pub assigned_bitrate_mbps: f64,
}

impl DeliveryRoute {
    /// Create a new delivery route.
    pub fn new(
        asset_id: impl Into<String>,
        destination: DeliveryDestination,
        assigned_bitrate_mbps: f64,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            destination,
            variant: None,
            assigned_bitrate_mbps,
        }
    }

    /// Attach a variant label to this route.
    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = Some(variant.into());
        self
    }

    /// Returns `true` when the assigned bitrate does not exceed the destination limit.
    #[must_use]
    pub fn bitrate_is_compliant(&self) -> bool {
        self.assigned_bitrate_mbps <= self.destination.max_bitrate_mbps()
    }
}

/// A map of delivery routes keyed by asset ID and destination.
#[derive(Clone, Debug, Default)]
pub struct DeliveryMap {
    routes: Vec<DeliveryRoute>,
}

impl DeliveryMap {
    /// Create an empty delivery map.
    #[must_use]
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Add a route to the map.
    pub fn add_route(&mut self, route: DeliveryRoute) {
        self.routes.push(route);
    }

    /// Find all routes for a given destination.
    #[must_use]
    pub fn find_for_destination(&self, dest: DeliveryDestination) -> Vec<&DeliveryRoute> {
        self.routes
            .iter()
            .filter(|r| r.destination == dest)
            .collect()
    }

    /// Find routes for a specific asset.
    #[must_use]
    pub fn find_for_asset(&self, asset_id: &str) -> Vec<&DeliveryRoute> {
        self.routes
            .iter()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// Total number of routes registered.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// All routes.
    #[must_use]
    pub fn routes(&self) -> &[DeliveryRoute] {
        &self.routes
    }
}

/// Validates a [`DeliveryMap`] for bitrate compliance across all routes.
#[derive(Clone, Debug, Default)]
pub struct DeliveryMapValidator;

impl DeliveryMapValidator {
    /// Create a new validator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate every route in the map.
    ///
    /// Returns a list of human-readable error strings for non-compliant routes.
    #[must_use]
    pub fn validate(&self, map: &DeliveryMap) -> Vec<String> {
        let mut errors = Vec::new();
        for route in map.routes() {
            if !route.bitrate_is_compliant() {
                errors.push(format!(
                    "Asset '{}' -> {}: bitrate {:.1} Mbps exceeds {} limit {:.1} Mbps",
                    route.asset_id,
                    route.destination.name(),
                    route.assigned_bitrate_mbps,
                    route.destination.name(),
                    route.destination.max_bitrate_mbps(),
                ));
            }
        }
        errors
    }

    /// Returns `true` when all routes in the map are bitrate-compliant.
    #[must_use]
    pub fn is_valid(&self, map: &DeliveryMap) -> bool {
        self.validate(map).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_destination_name_broadcast() {
        assert_eq!(DeliveryDestination::Broadcast.name(), "Broadcast");
    }

    #[test]
    fn test_destination_name_streaming() {
        assert_eq!(DeliveryDestination::Streaming.name(), "Streaming");
    }

    #[test]
    fn test_destination_name_cinema() {
        assert_eq!(DeliveryDestination::Cinema.name(), "Cinema");
    }

    #[test]
    fn test_destination_name_social() {
        assert_eq!(DeliveryDestination::Social.name(), "Social Media");
    }

    #[test]
    fn test_max_bitrate_broadcast() {
        assert_eq!(DeliveryDestination::Broadcast.max_bitrate_mbps(), 50.0);
    }

    #[test]
    fn test_max_bitrate_cinema_highest() {
        assert!(DeliveryDestination::Cinema.max_bitrate_mbps() > 100.0);
    }

    #[test]
    fn test_max_bitrate_social_lowest() {
        assert!(
            DeliveryDestination::Social.max_bitrate_mbps()
                < DeliveryDestination::Streaming.max_bitrate_mbps()
        );
    }

    #[test]
    fn test_loudness_target_social_highest() {
        // Social platforms have the highest (least negative) target
        assert!(
            DeliveryDestination::Social.loudness_target_lufs()
                > DeliveryDestination::Cinema.loudness_target_lufs()
        );
    }

    #[test]
    fn test_route_bitrate_compliant() {
        let route = DeliveryRoute::new("asset-001", DeliveryDestination::Broadcast, 40.0);
        assert!(route.bitrate_is_compliant());
    }

    #[test]
    fn test_route_bitrate_non_compliant() {
        let route = DeliveryRoute::new("asset-001", DeliveryDestination::Social, 15.0);
        assert!(!route.bitrate_is_compliant());
    }

    #[test]
    fn test_route_with_variant() {
        let route =
            DeliveryRoute::new("a", DeliveryDestination::Streaming, 10.0).with_variant("HDR10");
        assert_eq!(route.variant.as_deref(), Some("HDR10"));
    }

    #[test]
    fn test_delivery_map_add_and_count() {
        let mut map = DeliveryMap::new();
        map.add_route(DeliveryRoute::new(
            "x",
            DeliveryDestination::Broadcast,
            20.0,
        ));
        map.add_route(DeliveryRoute::new("y", DeliveryDestination::Streaming, 5.0));
        assert_eq!(map.route_count(), 2);
    }

    #[test]
    fn test_find_for_destination() {
        let mut map = DeliveryMap::new();
        map.add_route(DeliveryRoute::new(
            "a",
            DeliveryDestination::Broadcast,
            20.0,
        ));
        map.add_route(DeliveryRoute::new("b", DeliveryDestination::Social, 5.0));
        map.add_route(DeliveryRoute::new(
            "c",
            DeliveryDestination::Broadcast,
            15.0,
        ));
        let bc = map.find_for_destination(DeliveryDestination::Broadcast);
        assert_eq!(bc.len(), 2);
    }

    #[test]
    fn test_find_for_asset() {
        let mut map = DeliveryMap::new();
        map.add_route(DeliveryRoute::new(
            "clip-42",
            DeliveryDestination::Broadcast,
            20.0,
        ));
        map.add_route(DeliveryRoute::new(
            "clip-42",
            DeliveryDestination::Social,
            5.0,
        ));
        map.add_route(DeliveryRoute::new(
            "clip-99",
            DeliveryDestination::Streaming,
            10.0,
        ));
        let found = map.find_for_asset("clip-42");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_validator_all_compliant() {
        let mut map = DeliveryMap::new();
        map.add_route(DeliveryRoute::new(
            "a",
            DeliveryDestination::Broadcast,
            30.0,
        ));
        map.add_route(DeliveryRoute::new(
            "b",
            DeliveryDestination::Streaming,
            10.0,
        ));
        let v = DeliveryMapValidator::new();
        assert!(v.is_valid(&map));
        assert!(v.validate(&map).is_empty());
    }

    #[test]
    fn test_validator_non_compliant_route() {
        let mut map = DeliveryMap::new();
        map.add_route(DeliveryRoute::new("a", DeliveryDestination::Social, 20.0)); // 20 > 8
        let v = DeliveryMapValidator::new();
        assert!(!v.is_valid(&map));
        let errors = v.validate(&map);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("asset 'a'") || errors[0].contains("Asset 'a'"));
    }
}
