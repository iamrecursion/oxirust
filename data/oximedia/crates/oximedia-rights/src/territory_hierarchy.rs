//! Hierarchical territory support: continent -> country -> state/province.
//!
//! This module extends the flat ISO 3166-1 territory model with a three-level
//! geographic hierarchy.  A right granted at the continent level automatically
//! covers every country and subdivision within that continent, while a right
//! granted at the country level covers all subdivisions of that country.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── RegionLevel ─────────────────────────────────────────────────────────────

/// The granularity level of a geographic region in the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionLevel {
    /// Continent (e.g. "EUROPE", "NORTH_AMERICA").
    Continent,
    /// Sovereign country (ISO 3166-1 alpha-2, e.g. "US", "DE").
    Country,
    /// State, province, or administrative subdivision (e.g. "US-CA", "DE-BY").
    Subdivision,
}

impl RegionLevel {
    /// Numeric depth in the hierarchy (0 = broadest).
    pub fn depth(&self) -> u8 {
        match self {
            Self::Continent => 0,
            Self::Country => 1,
            Self::Subdivision => 2,
        }
    }
}

// ── HierarchicalRegion ──────────────────────────────────────────────────────

/// A node in the three-level geographic hierarchy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierarchicalRegion {
    /// Canonical code for this region (uppercased).
    pub code: String,
    /// Level of this region in the hierarchy.
    pub level: RegionLevel,
    /// Code of the parent region (`None` for continents).
    pub parent_code: Option<String>,
}

impl HierarchicalRegion {
    /// Create a new region.
    pub fn new(code: &str, level: RegionLevel, parent_code: Option<&str>) -> Self {
        Self {
            code: code.to_uppercase(),
            level,
            parent_code: parent_code.map(|p| p.to_uppercase()),
        }
    }

    /// Convenience: create a continent node.
    pub fn continent(code: &str) -> Self {
        Self::new(code, RegionLevel::Continent, None)
    }

    /// Convenience: create a country node under a continent.
    pub fn country(code: &str, continent: &str) -> Self {
        Self::new(code, RegionLevel::Country, Some(continent))
    }

    /// Convenience: create a subdivision node under a country.
    pub fn subdivision(code: &str, country: &str) -> Self {
        Self::new(code, RegionLevel::Subdivision, Some(country))
    }

    /// Returns `true` if this region is the root of its branch (a continent).
    pub fn is_root(&self) -> bool {
        self.parent_code.is_none()
    }
}

// ── TerritoryHierarchy ──────────────────────────────────────────────────────

/// Registry that maps region codes to their hierarchical position and
/// provides ancestor / descendant lookups.
#[derive(Debug, Default)]
pub struct TerritoryHierarchy {
    regions: HashMap<String, HierarchicalRegion>,
}

impl TerritoryHierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a hierarchy pre-populated with standard continents and a broad
    /// set of well-known countries.
    pub fn with_defaults() -> Self {
        let mut h = Self::new();

        // Continents
        for code in &[
            "EUROPE",
            "NORTH_AMERICA",
            "SOUTH_AMERICA",
            "ASIA",
            "AFRICA",
            "OCEANIA",
            "ANTARCTICA",
        ] {
            h.register(HierarchicalRegion::continent(code));
        }

        // A representative set of countries per continent
        let country_map: &[(&str, &[&str])] = &[
            (
                "EUROPE",
                &[
                    "GB", "DE", "FR", "ES", "IT", "NL", "BE", "SE", "NO", "FI", "DK", "PL", "AT",
                    "CH", "IE", "PT", "CZ", "GR", "RO", "HU",
                ],
            ),
            ("NORTH_AMERICA", &["US", "CA", "MX"]),
            (
                "SOUTH_AMERICA",
                &["BR", "AR", "CL", "CO", "PE", "VE", "UY", "EC"],
            ),
            (
                "ASIA",
                &[
                    "JP", "CN", "KR", "IN", "TW", "SG", "TH", "MY", "PH", "ID", "VN", "IL", "AE",
                    "SA",
                ],
            ),
            ("AFRICA", &["ZA", "NG", "KE", "EG", "GH", "TZ", "ET", "MA"]),
            ("OCEANIA", &["AU", "NZ", "FJ", "PG"]),
        ];

        for &(continent, countries) in country_map {
            for &cc in countries {
                h.register(HierarchicalRegion::country(cc, continent));
            }
        }

        // US states (sample)
        for sub in &["US-CA", "US-NY", "US-TX", "US-FL", "US-IL", "US-WA"] {
            h.register(HierarchicalRegion::subdivision(sub, "US"));
        }

        // German Laender (sample)
        for sub in &["DE-BY", "DE-BE", "DE-HH", "DE-NW"] {
            h.register(HierarchicalRegion::subdivision(sub, "DE"));
        }

        h
    }

    /// Register a region.  Replaces any existing entry with the same code.
    pub fn register(&mut self, region: HierarchicalRegion) {
        self.regions.insert(region.code.clone(), region);
    }

    /// Look up a region by code.
    pub fn get(&self, code: &str) -> Option<&HierarchicalRegion> {
        self.regions.get(&code.to_uppercase())
    }

    /// Total number of registered regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Return the chain of ancestor codes from a region up to (and including)
    /// its continent root.
    ///
    /// The returned vec starts with the region itself and ends at the root.
    pub fn ancestors(&self, code: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = code.to_uppercase();
        loop {
            result.push(current.clone());
            match self.regions.get(&current) {
                Some(r) => match &r.parent_code {
                    Some(parent) => current = parent.clone(),
                    None => break,
                },
                None => break,
            }
        }
        result
    }

    /// Return all direct children of `parent_code`.
    pub fn children(&self, parent_code: &str) -> Vec<&HierarchicalRegion> {
        let upper = parent_code.to_uppercase();
        self.regions
            .values()
            .filter(|r| r.parent_code.as_deref() == Some(upper.as_str()))
            .collect()
    }

    /// Return every descendant (children, grandchildren, ...) of `code`.
    pub fn descendants(&self, code: &str) -> Vec<&HierarchicalRegion> {
        let mut result = Vec::new();
        let mut queue = vec![code.to_uppercase()];
        while let Some(current) = queue.pop() {
            for child in self.children(&current) {
                result.push(child);
                queue.push(child.code.clone());
            }
        }
        result
    }

    /// Returns `true` when `ancestor_code` is equal to or an ancestor of
    /// `descendant_code` in the hierarchy.
    ///
    /// This is the key method for rights inclusion checks: a right granted at
    /// the continent level `includes` any country within that continent.
    pub fn includes(&self, ancestor_code: &str, descendant_code: &str) -> bool {
        let ancestor_upper = ancestor_code.to_uppercase();
        let ancestors = self.ancestors(descendant_code);
        ancestors.iter().any(|a| *a == ancestor_upper)
    }

    /// All countries that belong to the given continent.
    pub fn countries_in_continent(&self, continent_code: &str) -> Vec<&HierarchicalRegion> {
        self.children(continent_code)
            .into_iter()
            .filter(|r| r.level == RegionLevel::Country)
            .collect()
    }

    /// All subdivisions of a country.
    pub fn subdivisions_of_country(&self, country_code: &str) -> Vec<&HierarchicalRegion> {
        self.children(country_code)
            .into_iter()
            .filter(|r| r.level == RegionLevel::Subdivision)
            .collect()
    }
}

// ── HierarchicalTerritoryRight ──────────────────────────────────────────────

/// A time-bounded rights record that uses hierarchical territory matching.
#[derive(Debug, Clone)]
pub struct HierarchicalTerritoryRight {
    /// Identifier for the asset this right covers.
    pub asset_id: String,
    /// Code of the territory at any level in the hierarchy.
    pub territory_code: String,
    /// Type of right (e.g. "broadcast", "streaming").
    pub right_type: String,
    /// Unix-epoch ms when this right becomes valid.
    pub valid_from_ms: u64,
    /// Unix-epoch ms when this right expires (`None` = no expiry).
    pub valid_until_ms: Option<u64>,
}

impl HierarchicalTerritoryRight {
    /// Create a new hierarchical territory right.
    pub fn new(
        asset_id: &str,
        territory_code: &str,
        right_type: &str,
        valid_from_ms: u64,
        valid_until_ms: Option<u64>,
    ) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            territory_code: territory_code.to_uppercase(),
            right_type: right_type.to_string(),
            valid_from_ms,
            valid_until_ms,
        }
    }

    /// Returns `true` when `now_ms` is within `[valid_from_ms, valid_until_ms)`.
    pub fn is_valid(&self, now_ms: u64) -> bool {
        if now_ms < self.valid_from_ms {
            return false;
        }
        match self.valid_until_ms {
            Some(until) => now_ms < until,
            None => true,
        }
    }
}

// ── HierarchicalRightsManager ───────────────────────────────────────────────

/// Registry of hierarchical territory rights.
///
/// Uses `TerritoryHierarchy` to determine whether a right granted at a
/// higher-level region covers a query at a lower-level region.
#[derive(Debug)]
pub struct HierarchicalRightsManager {
    hierarchy: TerritoryHierarchy,
    rights: Vec<HierarchicalTerritoryRight>,
}

impl HierarchicalRightsManager {
    /// Create a new manager with the given hierarchy.
    pub fn new(hierarchy: TerritoryHierarchy) -> Self {
        Self {
            hierarchy,
            rights: Vec::new(),
        }
    }

    /// Create a manager with the default hierarchy.
    pub fn with_defaults() -> Self {
        Self::new(TerritoryHierarchy::with_defaults())
    }

    /// Register a territory right.
    pub fn add_right(&mut self, right: HierarchicalTerritoryRight) {
        self.rights.push(right);
    }

    /// Returns `true` if there is at least one valid, matching right for the
    /// given asset, territory, and right-type at `now_ms`.
    ///
    /// A right at a higher hierarchy level (e.g. continent) covers queries at
    /// lower levels (e.g. country, subdivision).
    pub fn has_right(
        &self,
        asset_id: &str,
        territory_code: &str,
        right_type: &str,
        now_ms: u64,
    ) -> bool {
        self.rights.iter().any(|r| {
            r.asset_id == asset_id
                && r.right_type == right_type
                && r.is_valid(now_ms)
                && self.hierarchy.includes(&r.territory_code, territory_code)
        })
    }

    /// Return a reference to the underlying hierarchy.
    pub fn hierarchy(&self) -> &TerritoryHierarchy {
        &self.hierarchy
    }

    /// Return all rights records.
    pub fn rights(&self) -> &[HierarchicalTerritoryRight] {
        &self.rights
    }

    /// All rights that are currently valid at `now_ms`.
    pub fn active_rights(&self, now_ms: u64) -> Vec<&HierarchicalTerritoryRight> {
        self.rights.iter().filter(|r| r.is_valid(now_ms)).collect()
    }

    /// Determine all territory codes (at any level) that have an active right
    /// of `right_type` for `asset_id` at `now_ms`, expanding from higher
    /// levels to enumerate covered lower levels.
    pub fn covered_territories(
        &self,
        asset_id: &str,
        right_type: &str,
        now_ms: u64,
    ) -> Vec<String> {
        let mut result = Vec::new();
        for r in &self.rights {
            if r.asset_id != asset_id || r.right_type != right_type || !r.is_valid(now_ms) {
                continue;
            }
            result.push(r.territory_code.clone());
            for desc in self.hierarchy.descendants(&r.territory_code) {
                if !result.contains(&desc.code) {
                    result.push(desc.code.clone());
                }
            }
        }
        result
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RegionLevel ─────────────────────────────────────────────────────────

    #[test]
    fn test_region_level_depth_continent() {
        assert_eq!(RegionLevel::Continent.depth(), 0);
    }

    #[test]
    fn test_region_level_depth_country() {
        assert_eq!(RegionLevel::Country.depth(), 1);
    }

    #[test]
    fn test_region_level_depth_subdivision() {
        assert_eq!(RegionLevel::Subdivision.depth(), 2);
    }

    // ── HierarchicalRegion ──────────────────────────────────────────────────

    #[test]
    fn test_region_continent_is_root() {
        let r = HierarchicalRegion::continent("EUROPE");
        assert!(r.is_root());
        assert_eq!(r.level, RegionLevel::Continent);
    }

    #[test]
    fn test_region_country_has_parent() {
        let r = HierarchicalRegion::country("DE", "EUROPE");
        assert!(!r.is_root());
        assert_eq!(r.parent_code.as_deref(), Some("EUROPE"));
    }

    #[test]
    fn test_region_subdivision_has_parent() {
        let r = HierarchicalRegion::subdivision("US-CA", "US");
        assert_eq!(r.parent_code.as_deref(), Some("US"));
        assert_eq!(r.level, RegionLevel::Subdivision);
    }

    #[test]
    fn test_region_code_uppercased() {
        let r = HierarchicalRegion::country("de", "europe");
        assert_eq!(r.code, "DE");
        assert_eq!(r.parent_code.as_deref(), Some("EUROPE"));
    }

    // ── TerritoryHierarchy ──────────────────────────────────────────────────

    #[test]
    fn test_hierarchy_register_and_get() {
        let mut h = TerritoryHierarchy::new();
        h.register(HierarchicalRegion::continent("EUROPE"));
        assert!(h.get("EUROPE").is_some());
        assert!(h.get("ASIA").is_none());
    }

    #[test]
    fn test_hierarchy_len() {
        let mut h = TerritoryHierarchy::new();
        h.register(HierarchicalRegion::continent("EUROPE"));
        h.register(HierarchicalRegion::country("DE", "EUROPE"));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_hierarchy_with_defaults_has_continents() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.get("EUROPE").is_some());
        assert!(h.get("ASIA").is_some());
        assert!(h.get("NORTH_AMERICA").is_some());
    }

    #[test]
    fn test_hierarchy_with_defaults_has_countries() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.get("US").is_some());
        assert!(h.get("JP").is_some());
        assert!(h.get("DE").is_some());
    }

    #[test]
    fn test_hierarchy_with_defaults_has_subdivisions() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.get("US-CA").is_some());
        assert!(h.get("DE-BY").is_some());
    }

    #[test]
    fn test_hierarchy_ancestors_subdivision() {
        let h = TerritoryHierarchy::with_defaults();
        let anc = h.ancestors("US-CA");
        assert_eq!(anc, vec!["US-CA", "US", "NORTH_AMERICA"]);
    }

    #[test]
    fn test_hierarchy_ancestors_country() {
        let h = TerritoryHierarchy::with_defaults();
        let anc = h.ancestors("DE");
        assert_eq!(anc, vec!["DE", "EUROPE"]);
    }

    #[test]
    fn test_hierarchy_ancestors_continent() {
        let h = TerritoryHierarchy::with_defaults();
        let anc = h.ancestors("EUROPE");
        assert_eq!(anc, vec!["EUROPE"]);
    }

    #[test]
    fn test_hierarchy_ancestors_unknown_code() {
        let h = TerritoryHierarchy::with_defaults();
        let anc = h.ancestors("ZZ");
        assert_eq!(anc, vec!["ZZ"]);
    }

    #[test]
    fn test_hierarchy_children_continent() {
        let h = TerritoryHierarchy::with_defaults();
        let children = h.children("NORTH_AMERICA");
        let codes: Vec<&str> = children.iter().map(|c| c.code.as_str()).collect();
        assert!(codes.contains(&"US"));
        assert!(codes.contains(&"CA"));
        assert!(codes.contains(&"MX"));
    }

    #[test]
    fn test_hierarchy_children_country_has_subdivisions() {
        let h = TerritoryHierarchy::with_defaults();
        let children = h.children("US");
        assert!(!children.is_empty());
        assert!(children.iter().all(|c| c.level == RegionLevel::Subdivision));
    }

    #[test]
    fn test_hierarchy_descendants() {
        let h = TerritoryHierarchy::with_defaults();
        let desc = h.descendants("NORTH_AMERICA");
        // Should include US, CA, MX and US subdivisions
        let codes: Vec<&str> = desc.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"US"));
        assert!(codes.contains(&"CA"));
        assert!(codes.contains(&"US-CA"));
    }

    #[test]
    fn test_hierarchy_includes_same_code() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.includes("US", "US"));
    }

    #[test]
    fn test_hierarchy_includes_continent_to_country() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.includes("EUROPE", "DE"));
        assert!(h.includes("NORTH_AMERICA", "US"));
    }

    #[test]
    fn test_hierarchy_includes_continent_to_subdivision() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.includes("NORTH_AMERICA", "US-CA"));
    }

    #[test]
    fn test_hierarchy_includes_country_to_subdivision() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.includes("US", "US-CA"));
        assert!(h.includes("DE", "DE-BY"));
    }

    #[test]
    fn test_hierarchy_does_not_include_sibling() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(!h.includes("US", "CA"));
        assert!(!h.includes("DE", "FR"));
    }

    #[test]
    fn test_hierarchy_does_not_include_child_to_parent() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(!h.includes("US-CA", "US"));
        assert!(!h.includes("US", "NORTH_AMERICA"));
    }

    #[test]
    fn test_hierarchy_countries_in_continent() {
        let h = TerritoryHierarchy::with_defaults();
        let countries = h.countries_in_continent("OCEANIA");
        let codes: Vec<&str> = countries.iter().map(|c| c.code.as_str()).collect();
        assert!(codes.contains(&"AU"));
        assert!(codes.contains(&"NZ"));
    }

    #[test]
    fn test_hierarchy_subdivisions_of_country() {
        let h = TerritoryHierarchy::with_defaults();
        let subs = h.subdivisions_of_country("DE");
        let codes: Vec<&str> = subs.iter().map(|s| s.code.as_str()).collect();
        assert!(codes.contains(&"DE-BY"));
        assert!(codes.contains(&"DE-BE"));
    }

    // ── HierarchicalTerritoryRight ──────────────────────────────────────────

    #[test]
    fn test_right_is_valid_within_window() {
        let r = HierarchicalTerritoryRight::new("a1", "EUROPE", "broadcast", 1000, Some(5000));
        assert!(r.is_valid(3000));
        assert!(!r.is_valid(5000)); // exclusive
    }

    #[test]
    fn test_right_no_expiry() {
        let r = HierarchicalTerritoryRight::new("a1", "US", "streaming", 0, None);
        assert!(r.is_valid(999_999));
    }

    #[test]
    fn test_right_before_start() {
        let r = HierarchicalTerritoryRight::new("a1", "US", "streaming", 1000, None);
        assert!(!r.is_valid(500));
    }

    // ── HierarchicalRightsManager ───────────────────────────────────────────

    #[test]
    fn test_manager_continent_right_covers_country() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "EUROPE",
            "broadcast",
            0,
            None,
        ));
        assert!(mgr.has_right("asset-1", "DE", "broadcast", 1000));
        assert!(mgr.has_right("asset-1", "FR", "broadcast", 1000));
        assert!(!mgr.has_right("asset-1", "US", "broadcast", 1000));
    }

    #[test]
    fn test_manager_continent_right_covers_subdivision() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "NORTH_AMERICA",
            "streaming",
            0,
            None,
        ));
        assert!(mgr.has_right("asset-1", "US-CA", "streaming", 1000));
        assert!(mgr.has_right("asset-1", "US-NY", "streaming", 1000));
    }

    #[test]
    fn test_manager_country_right_covers_subdivision() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "US",
            "broadcast",
            0,
            None,
        ));
        assert!(mgr.has_right("asset-1", "US-TX", "broadcast", 1000));
        assert!(!mgr.has_right("asset-1", "DE-BY", "broadcast", 1000));
    }

    #[test]
    fn test_manager_subdivision_right_does_not_cover_sibling() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "US-CA",
            "broadcast",
            0,
            None,
        ));
        assert!(mgr.has_right("asset-1", "US-CA", "broadcast", 1000));
        assert!(!mgr.has_right("asset-1", "US-NY", "broadcast", 1000));
    }

    #[test]
    fn test_manager_wrong_right_type() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "EUROPE",
            "broadcast",
            0,
            None,
        ));
        assert!(!mgr.has_right("asset-1", "DE", "streaming", 1000));
    }

    #[test]
    fn test_manager_expired_right_not_returned() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "EUROPE",
            "broadcast",
            0,
            Some(500),
        ));
        assert!(!mgr.has_right("asset-1", "DE", "broadcast", 1000));
    }

    #[test]
    fn test_manager_active_rights() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "a1",
            "US",
            "broadcast",
            0,
            Some(500),
        ));
        mgr.add_right(HierarchicalTerritoryRight::new(
            "a2",
            "DE",
            "broadcast",
            0,
            None,
        ));
        assert_eq!(mgr.active_rights(1000).len(), 1);
    }

    #[test]
    fn test_manager_covered_territories() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "US",
            "streaming",
            0,
            None,
        ));
        let covered = mgr.covered_territories("asset-1", "streaming", 1000);
        assert!(covered.contains(&"US".to_string()));
        assert!(covered.contains(&"US-CA".to_string()));
        assert!(covered.contains(&"US-NY".to_string()));
        assert!(!covered.contains(&"DE".to_string()));
    }

    #[test]
    fn test_manager_covered_territories_continent() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "NORTH_AMERICA",
            "broadcast",
            0,
            None,
        ));
        let covered = mgr.covered_territories("asset-1", "broadcast", 1000);
        assert!(covered.contains(&"NORTH_AMERICA".to_string()));
        assert!(covered.contains(&"US".to_string()));
        assert!(covered.contains(&"CA".to_string()));
        assert!(covered.contains(&"MX".to_string()));
        assert!(covered.contains(&"US-CA".to_string()));
    }

    #[test]
    fn test_manager_multiple_rights_merge_coverage() {
        let mut mgr = HierarchicalRightsManager::with_defaults();
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "US",
            "streaming",
            0,
            None,
        ));
        mgr.add_right(HierarchicalTerritoryRight::new(
            "asset-1",
            "DE",
            "streaming",
            0,
            None,
        ));
        assert!(mgr.has_right("asset-1", "US-CA", "streaming", 1000));
        assert!(mgr.has_right("asset-1", "DE-BY", "streaming", 1000));
        assert!(!mgr.has_right("asset-1", "FR", "streaming", 1000));
    }

    #[test]
    fn test_hierarchy_case_insensitive_includes() {
        let h = TerritoryHierarchy::with_defaults();
        assert!(h.includes("europe", "de"));
        assert!(h.includes("EUROPE", "de"));
        assert!(h.includes("us", "us-ca"));
    }

    #[test]
    fn test_hierarchy_empty() {
        let h = TerritoryHierarchy::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }
}
