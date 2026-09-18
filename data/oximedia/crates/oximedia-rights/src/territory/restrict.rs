//! Territory restrictions

use super::{TerritoryZone, WorldRegion};
use serde::{Deserialize, Serialize};

/// Territory restriction type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TerritoryRestriction {
    /// Allowed in specific territories
    AllowedIn(TerritoryZone),
    /// Prohibited in specific territories
    ProhibitedIn(TerritoryZone),
    /// Worldwide (no restrictions)
    Worldwide,
}

impl TerritoryRestriction {
    /// Create a worldwide restriction (no restrictions)
    pub fn worldwide() -> Self {
        TerritoryRestriction::Worldwide
    }

    /// Create restriction allowed only in specified countries
    pub fn allowed_countries(countries: Vec<String>) -> Self {
        let zone = TerritoryZone::new("Allowed").add_countries(countries);
        TerritoryRestriction::AllowedIn(zone)
    }

    /// Create restriction prohibited in specified countries
    pub fn prohibited_countries(countries: Vec<String>) -> Self {
        let zone = TerritoryZone::new("Prohibited").add_countries(countries);
        TerritoryRestriction::ProhibitedIn(zone)
    }

    /// Create restriction allowed in a specific region
    pub fn allowed_region(region: WorldRegion) -> Self {
        let zone = TerritoryZone::new(format!("Allowed {region:?}")).add_region(region);
        TerritoryRestriction::AllowedIn(zone)
    }

    /// Create restriction prohibited in a specific region
    pub fn prohibited_region(region: WorldRegion) -> Self {
        let zone = TerritoryZone::new(format!("Prohibited {region:?}")).add_region(region);
        TerritoryRestriction::ProhibitedIn(zone)
    }

    /// Check if a country code is allowed
    pub fn is_allowed(&self, country_code: &str) -> bool {
        match self {
            TerritoryRestriction::Worldwide => true,
            TerritoryRestriction::AllowedIn(zone) => zone.contains(country_code),
            TerritoryRestriction::ProhibitedIn(zone) => !zone.contains(country_code),
        }
    }

    /// Get all allowed countries (None if worldwide)
    pub fn get_allowed_countries(&self) -> Option<Vec<String>> {
        match self {
            TerritoryRestriction::Worldwide => None,
            TerritoryRestriction::AllowedIn(zone) => Some(zone.all_countries()),
            TerritoryRestriction::ProhibitedIn(_) => {
                // For prohibited, we can't easily list all allowed countries
                None
            }
        }
    }
}

impl Default for TerritoryRestriction {
    fn default() -> Self {
        Self::worldwide()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worldwide_restriction() {
        let restriction = TerritoryRestriction::worldwide();
        assert!(restriction.is_allowed("US"));
        assert!(restriction.is_allowed("JP"));
        assert!(restriction.is_allowed("ANY"));
    }

    #[test]
    fn test_allowed_countries() {
        let restriction =
            TerritoryRestriction::allowed_countries(vec!["US".to_string(), "GB".to_string()]);

        assert!(restriction.is_allowed("US"));
        assert!(restriction.is_allowed("GB"));
        assert!(!restriction.is_allowed("JP"));
    }

    #[test]
    fn test_prohibited_countries() {
        let restriction = TerritoryRestriction::prohibited_countries(vec!["US".to_string()]);

        assert!(!restriction.is_allowed("US"));
        assert!(restriction.is_allowed("GB"));
        assert!(restriction.is_allowed("JP"));
    }

    #[test]
    fn test_allowed_region() {
        let restriction = TerritoryRestriction::allowed_region(WorldRegion::Europe);

        assert!(restriction.is_allowed("GB"));
        assert!(restriction.is_allowed("DE"));
        assert!(!restriction.is_allowed("US"));
    }

    #[test]
    fn test_prohibited_region() {
        let restriction = TerritoryRestriction::prohibited_region(WorldRegion::Asia);

        assert!(restriction.is_allowed("US"));
        assert!(!restriction.is_allowed("JP"));
        assert!(!restriction.is_allowed("CN"));
    }
}
