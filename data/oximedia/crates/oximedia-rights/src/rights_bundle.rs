#![allow(dead_code)]
//! Bundle multiple rights into groups for batch operations.
//!
//! Provides tools for grouping related rights, applying bulk operations
//! (grant, revoke, transfer), and querying bundle membership.

use std::collections::{HashMap, HashSet};

/// Unique identifier for a rights bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BundleId(pub String);

impl BundleId {
    /// Create a new bundle identifier.
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Get the inner string value.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BundleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of a rights bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleStatus {
    /// Bundle is active and all rights are valid.
    Active,
    /// Bundle is pending approval.
    Pending,
    /// Bundle is suspended.
    Suspended,
    /// Bundle has expired.
    Expired,
    /// Bundle has been revoked.
    Revoked,
}

impl BundleStatus {
    /// Check if the bundle is in a usable state.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, BundleStatus::Active)
    }

    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            BundleStatus::Active => "active",
            BundleStatus::Pending => "pending",
            BundleStatus::Suspended => "suspended",
            BundleStatus::Expired => "expired",
            BundleStatus::Revoked => "revoked",
        }
    }
}

/// A single right entry within a bundle.
#[derive(Debug, Clone)]
pub struct RightEntry {
    /// Unique right identifier.
    pub right_id: String,
    /// Content or asset this right applies to.
    pub asset_id: String,
    /// Type of right (e.g., "broadcast", "streaming", "download").
    pub right_type: String,
    /// Territory code (ISO 3166-1 alpha-2) or "WORLDWIDE".
    pub territory: String,
    /// Start date as ISO 8601 string, empty means immediate.
    pub start_date: String,
    /// End date as ISO 8601 string, empty means perpetual.
    pub end_date: String,
}

impl RightEntry {
    /// Create a new right entry.
    #[must_use]
    pub fn new(right_id: &str, asset_id: &str, right_type: &str) -> Self {
        Self {
            right_id: right_id.to_string(),
            asset_id: asset_id.to_string(),
            right_type: right_type.to_string(),
            territory: "WORLDWIDE".to_string(),
            start_date: String::new(),
            end_date: String::new(),
        }
    }

    /// Set the territory.
    #[must_use]
    pub fn with_territory(mut self, territory: &str) -> Self {
        self.territory = territory.to_string();
        self
    }

    /// Set the date range.
    #[must_use]
    pub fn with_dates(mut self, start: &str, end: &str) -> Self {
        self.start_date = start.to_string();
        self.end_date = end.to_string();
        self
    }

    /// Check if this right is perpetual (no end date).
    #[must_use]
    pub fn is_perpetual(&self) -> bool {
        self.end_date.is_empty()
    }

    /// Check if this right is worldwide.
    #[must_use]
    pub fn is_worldwide(&self) -> bool {
        self.territory == "WORLDWIDE"
    }
}

/// A bundle of related rights grouped for batch operations.
#[derive(Debug, Clone)]
pub struct RightsBundle {
    /// Bundle identifier.
    pub id: BundleId,
    /// Human-readable name.
    pub name: String,
    /// Description of the bundle.
    pub description: String,
    /// Current status.
    pub status: BundleStatus,
    /// Owner or licensee.
    pub owner: String,
    /// Rights in this bundle.
    rights: Vec<RightEntry>,
    /// Tags for categorization.
    tags: HashSet<String>,
}

impl RightsBundle {
    /// Create a new rights bundle.
    #[must_use]
    pub fn new(id: &str, name: &str, owner: &str) -> Self {
        Self {
            id: BundleId::new(id),
            name: name.to_string(),
            description: String::new(),
            status: BundleStatus::Pending,
            owner: owner.to_string(),
            rights: Vec::new(),
            tags: HashSet::new(),
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add a right entry to the bundle.
    pub fn add_right(&mut self, entry: RightEntry) {
        self.rights.push(entry);
    }

    /// Remove a right by its ID. Returns true if found and removed.
    pub fn remove_right(&mut self, right_id: &str) -> bool {
        let initial_len = self.rights.len();
        self.rights.retain(|r| r.right_id != right_id);
        self.rights.len() < initial_len
    }

    /// Get all rights in the bundle.
    #[must_use]
    pub fn rights(&self) -> &[RightEntry] {
        &self.rights
    }

    /// Get the number of rights in the bundle.
    #[must_use]
    pub fn right_count(&self) -> usize {
        self.rights.len()
    }

    /// Find rights by asset ID.
    #[must_use]
    pub fn rights_for_asset(&self, asset_id: &str) -> Vec<&RightEntry> {
        self.rights
            .iter()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// Find rights by territory.
    #[must_use]
    pub fn rights_for_territory(&self, territory: &str) -> Vec<&RightEntry> {
        self.rights
            .iter()
            .filter(|r| r.territory == territory || r.territory == "WORLDWIDE")
            .collect()
    }

    /// Activate the bundle.
    pub fn activate(&mut self) {
        self.status = BundleStatus::Active;
    }

    /// Suspend the bundle.
    pub fn suspend(&mut self) {
        self.status = BundleStatus::Suspended;
    }

    /// Revoke the bundle.
    pub fn revoke(&mut self) {
        self.status = BundleStatus::Revoked;
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: &str) {
        self.tags.insert(tag.to_string());
    }

    /// Check if the bundle has a tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Get all unique asset IDs referenced in this bundle.
    #[must_use]
    pub fn asset_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.rights.iter().map(|r| r.asset_id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Check if the bundle is empty (no rights).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rights.is_empty()
    }
}

/// Registry for managing multiple bundles.
pub struct BundleRegistry {
    /// Bundles indexed by their ID.
    bundles: HashMap<String, RightsBundle>,
}

impl BundleRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bundles: HashMap::new(),
        }
    }

    /// Register a bundle.
    pub fn register(&mut self, bundle: RightsBundle) {
        self.bundles.insert(bundle.id.0.clone(), bundle);
    }

    /// Get a bundle by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&RightsBundle> {
        self.bundles.get(id)
    }

    /// Get a mutable bundle by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut RightsBundle> {
        self.bundles.get_mut(id)
    }

    /// Remove a bundle by ID.
    pub fn remove(&mut self, id: &str) -> Option<RightsBundle> {
        self.bundles.remove(id)
    }

    /// Find all bundles owned by a specific owner.
    #[must_use]
    pub fn find_by_owner(&self, owner: &str) -> Vec<&RightsBundle> {
        self.bundles.values().filter(|b| b.owner == owner).collect()
    }

    /// Find all active bundles.
    #[must_use]
    pub fn find_active(&self) -> Vec<&RightsBundle> {
        self.bundles
            .values()
            .filter(|b| b.status == BundleStatus::Active)
            .collect()
    }

    /// Get the total number of bundles.
    #[must_use]
    pub fn count(&self) -> usize {
        self.bundles.len()
    }

    /// List all bundle IDs.
    #[must_use]
    pub fn list_ids(&self) -> Vec<&str> {
        self.bundles.keys().map(String::as_str).collect()
    }
}

impl Default for BundleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_id() {
        let id = BundleId::new("bundle-001");
        assert_eq!(id.as_str(), "bundle-001");
        assert_eq!(id.to_string(), "bundle-001");
    }

    #[test]
    fn test_bundle_status() {
        assert!(BundleStatus::Active.is_usable());
        assert!(!BundleStatus::Pending.is_usable());
        assert!(!BundleStatus::Suspended.is_usable());
        assert!(!BundleStatus::Expired.is_usable());
        assert!(!BundleStatus::Revoked.is_usable());
    }

    #[test]
    fn test_bundle_status_labels() {
        assert_eq!(BundleStatus::Active.label(), "active");
        assert_eq!(BundleStatus::Revoked.label(), "revoked");
    }

    #[test]
    fn test_right_entry_creation() {
        let entry = RightEntry::new("r1", "asset-100", "broadcast")
            .with_territory("US")
            .with_dates("2024-01-01", "2025-12-31");

        assert_eq!(entry.right_id, "r1");
        assert_eq!(entry.territory, "US");
        assert!(!entry.is_perpetual());
        assert!(!entry.is_worldwide());
    }

    #[test]
    fn test_right_entry_perpetual_worldwide() {
        let entry = RightEntry::new("r2", "asset-200", "streaming");
        assert!(entry.is_perpetual());
        assert!(entry.is_worldwide());
    }

    #[test]
    fn test_bundle_creation() {
        let bundle = RightsBundle::new("b1", "Test Bundle", "Acme Corp")
            .with_description("Test description");
        assert_eq!(bundle.id.as_str(), "b1");
        assert_eq!(bundle.name, "Test Bundle");
        assert_eq!(bundle.status, BundleStatus::Pending);
        assert!(bundle.is_empty());
    }

    #[test]
    fn test_bundle_add_remove_rights() {
        let mut bundle = RightsBundle::new("b2", "Bundle", "Owner");
        bundle.add_right(RightEntry::new("r1", "a1", "broadcast"));
        bundle.add_right(RightEntry::new("r2", "a2", "streaming"));
        assert_eq!(bundle.right_count(), 2);

        assert!(bundle.remove_right("r1"));
        assert_eq!(bundle.right_count(), 1);
        assert!(!bundle.remove_right("nonexistent"));
    }

    #[test]
    fn test_bundle_status_transitions() {
        let mut bundle = RightsBundle::new("b3", "Bundle", "Owner");
        assert_eq!(bundle.status, BundleStatus::Pending);
        bundle.activate();
        assert_eq!(bundle.status, BundleStatus::Active);
        bundle.suspend();
        assert_eq!(bundle.status, BundleStatus::Suspended);
        bundle.revoke();
        assert_eq!(bundle.status, BundleStatus::Revoked);
    }

    #[test]
    fn test_bundle_find_by_asset() {
        let mut bundle = RightsBundle::new("b4", "Bundle", "Owner");
        bundle.add_right(RightEntry::new("r1", "asset-A", "broadcast"));
        bundle.add_right(RightEntry::new("r2", "asset-B", "streaming"));
        bundle.add_right(RightEntry::new("r3", "asset-A", "download"));

        let results = bundle.rights_for_asset("asset-A");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bundle_find_by_territory() {
        let mut bundle = RightsBundle::new("b5", "Bundle", "Owner");
        bundle.add_right(RightEntry::new("r1", "a1", "broadcast").with_territory("US"));
        bundle.add_right(RightEntry::new("r2", "a2", "broadcast").with_territory("UK"));
        bundle.add_right(RightEntry::new("r3", "a3", "broadcast")); // WORLDWIDE

        let us_rights = bundle.rights_for_territory("US");
        assert_eq!(us_rights.len(), 2); // US + WORLDWIDE
    }

    #[test]
    fn test_bundle_tags() {
        let mut bundle = RightsBundle::new("b6", "Bundle", "Owner");
        bundle.add_tag("premium");
        bundle.add_tag("sports");
        assert!(bundle.has_tag("premium"));
        assert!(!bundle.has_tag("news"));
    }

    #[test]
    fn test_bundle_asset_ids() {
        let mut bundle = RightsBundle::new("b7", "Bundle", "Owner");
        bundle.add_right(RightEntry::new("r1", "asset-A", "broadcast"));
        bundle.add_right(RightEntry::new("r2", "asset-B", "streaming"));
        bundle.add_right(RightEntry::new("r3", "asset-A", "download"));

        let ids = bundle.asset_ids();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_registry_basic_operations() {
        let mut reg = BundleRegistry::new();
        let mut bundle = RightsBundle::new("b1", "Bundle 1", "Owner A");
        bundle.activate();
        reg.register(bundle);
        reg.register(RightsBundle::new("b2", "Bundle 2", "Owner B"));

        assert_eq!(reg.count(), 2);
        assert!(reg.get("b1").is_some());
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn test_registry_find_by_owner() {
        let mut reg = BundleRegistry::new();
        reg.register(RightsBundle::new("b1", "B1", "Alice"));
        reg.register(RightsBundle::new("b2", "B2", "Alice"));
        reg.register(RightsBundle::new("b3", "B3", "Bob"));

        let alice_bundles = reg.find_by_owner("Alice");
        assert_eq!(alice_bundles.len(), 2);
    }

    #[test]
    fn test_registry_find_active() {
        let mut reg = BundleRegistry::new();
        let mut active = RightsBundle::new("b1", "Active", "Owner");
        active.activate();
        reg.register(active);
        reg.register(RightsBundle::new("b2", "Pending", "Owner"));

        let active_bundles = reg.find_active();
        assert_eq!(active_bundles.len(), 1);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = BundleRegistry::new();
        reg.register(RightsBundle::new("b1", "B1", "Owner"));
        assert_eq!(reg.count(), 1);
        let removed = reg.remove("b1");
        assert!(removed.is_some());
        assert_eq!(reg.count(), 0);
    }
}
