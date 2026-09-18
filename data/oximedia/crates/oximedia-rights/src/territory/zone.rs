//! Territory zones and regions

use serde::{Deserialize, Serialize};

/// World regions for territory management
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorldRegion {
    /// North America
    NorthAmerica,
    /// South America
    SouthAmerica,
    /// Europe
    Europe,
    /// Asia
    Asia,
    /// Africa
    Africa,
    /// Oceania
    Oceania,
    /// Middle East
    MiddleEast,
    /// Worldwide
    Worldwide,
}

impl WorldRegion {
    /// Get the list of country codes in this region
    pub fn country_codes(&self) -> Vec<&'static str> {
        match self {
            WorldRegion::NorthAmerica => vec!["US", "CA", "MX"],
            WorldRegion::SouthAmerica => vec!["BR", "AR", "CL", "CO", "PE", "VE"],
            WorldRegion::Europe => vec![
                "GB", "FR", "DE", "IT", "ES", "NL", "BE", "CH", "AT", "SE", "NO", "DK", "FI", "PL",
                "PT", "GR", "CZ", "IE", "HU", "RO",
            ],
            WorldRegion::Asia => vec![
                "CN", "JP", "IN", "KR", "ID", "TH", "MY", "SG", "PH", "VN", "TW", "HK",
            ],
            WorldRegion::Africa => vec!["ZA", "NG", "EG", "KE", "MA", "ET"],
            WorldRegion::Oceania => vec!["AU", "NZ"],
            WorldRegion::MiddleEast => vec!["AE", "SA", "IL", "TR", "IR", "IQ"],
            WorldRegion::Worldwide => vec![], // Special case
        }
    }

    /// Check if a country code belongs to this region
    pub fn contains(&self, country_code: &str) -> bool {
        if *self == WorldRegion::Worldwide {
            return true;
        }
        self.country_codes().contains(&country_code)
    }
}

/// Territory zone definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerritoryZone {
    /// Zone name
    pub name: String,
    /// List of country codes (ISO 3166-1 alpha-2)
    pub countries: Vec<String>,
    /// Regions included
    pub regions: Vec<WorldRegion>,
}

impl TerritoryZone {
    /// Create a new territory zone
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            countries: vec![],
            regions: vec![],
        }
    }

    /// Add a country code
    pub fn add_country(mut self, code: impl Into<String>) -> Self {
        self.countries.push(code.into());
        self
    }

    /// Add multiple country codes
    pub fn add_countries(mut self, codes: Vec<String>) -> Self {
        self.countries.extend(codes);
        self
    }

    /// Add a region
    pub fn add_region(mut self, region: WorldRegion) -> Self {
        self.regions.push(region);
        self
    }

    /// Check if a country code is in this zone
    pub fn contains(&self, country_code: &str) -> bool {
        // Check direct country list
        if self.countries.contains(&country_code.to_string()) {
            return true;
        }

        // Check regions
        for region in &self.regions {
            if region.contains(country_code) {
                return true;
            }
        }

        false
    }

    /// Get all country codes in this zone
    pub fn all_countries(&self) -> Vec<String> {
        let mut countries = self.countries.clone();

        for region in &self.regions {
            countries.extend(
                region
                    .country_codes()
                    .iter()
                    .map(std::string::ToString::to_string),
            );
        }

        countries.sort();
        countries.dedup();
        countries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_region_contains() {
        assert!(WorldRegion::NorthAmerica.contains("US"));
        assert!(WorldRegion::Europe.contains("GB"));
        assert!(!WorldRegion::Asia.contains("US"));
    }

    #[test]
    fn test_worldwide_region() {
        assert!(WorldRegion::Worldwide.contains("US"));
        assert!(WorldRegion::Worldwide.contains("JP"));
        assert!(WorldRegion::Worldwide.contains("ANY"));
    }

    #[test]
    fn test_territory_zone() {
        let zone = TerritoryZone::new("EU + US")
            .add_country("US")
            .add_region(WorldRegion::Europe);

        assert!(zone.contains("US"));
        assert!(zone.contains("GB"));
        assert!(zone.contains("DE"));
        assert!(!zone.contains("JP"));
    }

    #[test]
    fn test_all_countries() {
        let zone = TerritoryZone::new("Test")
            .add_country("US")
            .add_country("CA");

        let countries = zone.all_countries();
        assert_eq!(countries.len(), 2);
        assert!(countries.contains(&"US".to_string()));
        assert!(countries.contains(&"CA".to_string()));
    }
}
