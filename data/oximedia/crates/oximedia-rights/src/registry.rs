//! Rights registry: in-memory store for rights entries with lookup, conflict
//! detection, usage checking, and expiry tracking.

use crate::license::LicenseType;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// LicenseType aliases used by the task spec
// ---------------------------------------------------------------------------
/// CC_BY alias for `LicenseType::CreativeCommonsBy`.
pub const LICENSE_CC_BY: &str = "cc_by";
/// CC_BY_SA alias for `LicenseType::CreativeCommonsBySa`.
pub const LICENSE_CC_BY_SA: &str = "cc_by_sa";
/// CC_BY_NC alias for `LicenseType::CreativeCommonsByNc`.
pub const LICENSE_CC_BY_NC: &str = "cc_by_nc";
/// All Rights Reserved alias for `LicenseType::RightsManaged`.
pub const LICENSE_ALL_RIGHTS_RESERVED: &str = "rights_managed";
/// Royalty-Free alias for `LicenseType::RoyaltyFree`.
pub const LICENSE_ROYALTY_FREE: &str = "royalty_free";

// ---------------------------------------------------------------------------
// Use-case type
// ---------------------------------------------------------------------------

/// The intended use-case for an asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UseCase {
    /// Commercial broadcast / streaming.
    Commercial,
    /// Non-commercial / educational use.
    NonCommercial,
    /// Editorial or news use.
    Editorial,
    /// Personal / private use.
    Personal,
    /// Other / custom use-case.
    Other(String),
}

impl UseCase {
    /// Returns true when the use-case is considered commercial.
    #[must_use]
    pub fn is_commercial(&self) -> bool {
        matches!(self, Self::Commercial)
    }
}

// ---------------------------------------------------------------------------
// Rights restrictions
// ---------------------------------------------------------------------------

/// Named usage restriction attached to a rights entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Restriction {
    /// No commercial use allowed.
    NoCommercial,
    /// Modifications (derivatives) not permitted.
    NoDerivatives,
    /// Must share under same license.
    ShareAlike,
    /// Attribution to the rights holder is required.
    AttributionRequired,
    /// Custom restriction text.
    Custom(String),
}

// ---------------------------------------------------------------------------
// RightsEntry
// ---------------------------------------------------------------------------

/// A single rights record for an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsEntry {
    /// Unique entry identifier.
    pub id: String,
    /// ID of the asset these rights apply to.
    pub asset_id: String,
    /// ISO 3166-1 alpha-2 territory code, e.g. "US", "GB", or "WW" for worldwide.
    pub territory: String,
    /// License type.
    pub license_type: LicenseType,
    /// Optional expiry date-time.  `None` means the rights never expire.
    pub expiry: Option<DateTime<Utc>>,
    /// Usage restrictions.
    pub restrictions: Vec<Restriction>,
    /// Rights holder / licensor name.
    pub rights_holder: String,
    /// When this entry was created.
    pub created_at: DateTime<Utc>,
}

impl RightsEntry {
    /// Create a new rights entry.
    #[must_use]
    pub fn new(
        asset_id: impl Into<String>,
        territory: impl Into<String>,
        license_type: LicenseType,
        rights_holder: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            asset_id: asset_id.into(),
            territory: territory.into(),
            license_type,
            expiry: None,
            restrictions: Vec::new(),
            rights_holder: rights_holder.into(),
            created_at: Utc::now(),
        }
    }

    /// Attach an expiry date-time.
    #[must_use]
    pub fn with_expiry(mut self, expiry: DateTime<Utc>) -> Self {
        self.expiry = Some(expiry);
        self
    }

    /// Add a usage restriction.
    #[must_use]
    pub fn with_restriction(mut self, restriction: Restriction) -> Self {
        self.restrictions.push(restriction);
        self
    }

    /// Returns `true` if this entry is currently valid (not expired).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self.expiry {
            None => true,
            Some(exp) => Utc::now() < exp,
        }
    }

    /// Returns `true` when this entry will expire within `days` days.
    #[must_use]
    pub fn expires_within(&self, days: i64) -> bool {
        match self.expiry {
            None => false,
            Some(exp) => {
                let threshold = Utc::now() + Duration::days(days);
                exp <= threshold && exp > Utc::now()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RightsRegistry
// ---------------------------------------------------------------------------

/// In-memory registry that stores and looks up rights entries.
#[derive(Debug, Default)]
pub struct RightsRegistry {
    /// Storage keyed by `(asset_id, territory)`.
    entries: HashMap<String, Vec<RightsEntry>>,
}

impl RightsRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rights entry.
    pub fn register(&mut self, entry: RightsEntry) {
        let key = format!("{}:{}", entry.asset_id, entry.territory);
        self.entries.entry(key).or_default().push(entry);
    }

    /// Look up all valid rights entries for an asset in a given territory,
    /// also checking the worldwide ("WW") entries.
    #[must_use]
    pub fn lookup(&self, asset_id: &str, territory: &str) -> Vec<&RightsEntry> {
        let exact_key = format!("{asset_id}:{territory}");
        let worldwide_key = format!("{asset_id}:WW");

        let mut result: Vec<&RightsEntry> = self
            .entries
            .get(&exact_key)
            .map(|v| v.iter().filter(|e| e.is_valid()).collect())
            .unwrap_or_default();

        if territory != "WW" {
            if let Some(ww) = self.entries.get(&worldwide_key) {
                result.extend(ww.iter().filter(|e| e.is_valid()));
            }
        }

        result
    }

    /// Return every entry in the registry.
    #[must_use]
    pub fn all_entries(&self) -> Vec<&RightsEntry> {
        self.entries.values().flatten().collect()
    }

    /// Remove an entry by its ID.  Returns `true` if the entry was found.
    pub fn remove(&mut self, entry_id: &str) -> bool {
        let mut removed = false;
        for entries in self.entries.values_mut() {
            if let Some(pos) = entries.iter().position(|e| e.id == entry_id) {
                entries.remove(pos);
                removed = true;
                break;
            }
        }
        removed
    }
}

// ---------------------------------------------------------------------------
// UsageChecker
// ---------------------------------------------------------------------------

/// Checks whether a specific use of an asset is permitted.
pub struct UsageChecker<'a> {
    registry: &'a RightsRegistry,
}

impl<'a> UsageChecker<'a> {
    /// Create a new checker backed by a registry.
    #[must_use]
    pub fn new(registry: &'a RightsRegistry) -> Self {
        Self { registry }
    }

    /// Returns `true` when the asset may be used for the given purpose in the
    /// specified territory, based on the current rights entries.
    #[must_use]
    pub fn is_allowed(&self, asset_id: &str, territory: &str, use_case: &UseCase) -> bool {
        let entries = self.registry.lookup(asset_id, territory);
        if entries.is_empty() {
            return false;
        }

        // Any single valid, unrestricted (or permissive) entry is enough.
        entries.iter().any(|e| Self::entry_permits(e, use_case))
    }

    fn entry_permits(entry: &RightsEntry, use_case: &UseCase) -> bool {
        if !entry.is_valid() {
            return false;
        }
        // Check license-level commercial restriction
        if use_case.is_commercial() && !entry.license_type.allows_commercial_use() {
            return false;
        }
        // Check explicit NoCommercial restriction
        if use_case.is_commercial() && entry.restrictions.contains(&Restriction::NoCommercial) {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// RightsConflict
// ---------------------------------------------------------------------------

/// Describes a conflict between two rights entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsConflict {
    /// ID of the first conflicting entry.
    pub entry_a_id: String,
    /// ID of the second conflicting entry.
    pub entry_b_id: String,
    /// Human-readable reason.
    pub reason: String,
}

/// Detects conflicting rights entries within a registry.
pub struct ConflictDetector<'a> {
    registry: &'a RightsRegistry,
}

impl<'a> ConflictDetector<'a> {
    /// Create a new detector.
    #[must_use]
    pub fn new(registry: &'a RightsRegistry) -> Self {
        Self { registry }
    }

    /// Scan all entries and return any detected conflicts.
    ///
    /// Two entries conflict when they apply to the same asset + territory but
    /// carry incompatible license types (e.g., Exclusive paired with
    /// NonExclusive, or contradictory CC licences).
    #[must_use]
    pub fn detect(&self) -> Vec<RightsConflict> {
        let mut conflicts = Vec::new();

        for entries in self.registry.entries.values() {
            let valid: Vec<&RightsEntry> = entries.iter().filter(|e| e.is_valid()).collect();

            // O(n²) within each key bucket — typically very small.
            for (i, a) in valid.iter().enumerate() {
                for b in valid.iter().skip(i + 1) {
                    if let Some(reason) = Self::detect_conflict(a, b) {
                        conflicts.push(RightsConflict {
                            entry_a_id: a.id.clone(),
                            entry_b_id: b.id.clone(),
                            reason,
                        });
                    }
                }
            }
        }

        conflicts
    }

    fn detect_conflict(a: &RightsEntry, b: &RightsEntry) -> Option<String> {
        // Exclusive + any other entry is a conflict
        let a_exclusive = matches!(a.license_type, LicenseType::Exclusive);
        let b_exclusive = matches!(b.license_type, LicenseType::Exclusive);

        if a_exclusive || b_exclusive {
            return Some(format!(
                "Exclusive license '{}' conflicts with entry '{}'",
                a.id, b.id
            ));
        }

        // CC ShareAlike conflicting with NoDerivatives
        let a_share_alike = a.restrictions.contains(&Restriction::ShareAlike);
        let b_no_deriv = b.restrictions.contains(&Restriction::NoDerivatives);
        let b_share_alike = b.restrictions.contains(&Restriction::ShareAlike);
        let a_no_deriv = a.restrictions.contains(&Restriction::NoDerivatives);

        if (a_share_alike && b_no_deriv) || (b_share_alike && a_no_deriv) {
            return Some(format!(
                "ShareAlike restriction conflicts with NoDerivatives between '{}' and '{}'",
                a.id, b.id
            ));
        }

        None
    }
}

// ---------------------------------------------------------------------------
// ExpiryChecker
// ---------------------------------------------------------------------------

/// Checks and reports on expiring rights entries.
pub struct ExpiryChecker<'a> {
    registry: &'a RightsRegistry,
}

impl<'a> ExpiryChecker<'a> {
    /// Create a new expiry checker.
    #[must_use]
    pub fn new(registry: &'a RightsRegistry) -> Self {
        Self { registry }
    }

    /// Return entries that expire within the next `days` days.
    #[must_use]
    pub fn expiring_soon(&self, days: i64) -> Vec<&RightsEntry> {
        self.registry
            .all_entries()
            .into_iter()
            .filter(|e| e.expires_within(days))
            .collect()
    }

    /// Return all entries that have already expired.
    #[must_use]
    pub fn already_expired(&self) -> Vec<&RightsEntry> {
        self.registry
            .all_entries()
            .into_iter()
            .filter(|e| e.expiry.is_some_and(|exp| Utc::now() >= exp))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::LicenseType;

    fn entry(asset: &str, territory: &str, license: LicenseType) -> RightsEntry {
        RightsEntry::new(asset, territory, license, "Holder Corp")
    }

    // --- RightsEntry ---

    #[test]
    fn test_rights_entry_new() {
        let e = entry("asset-1", "US", LicenseType::RoyaltyFree);
        assert_eq!(e.asset_id, "asset-1");
        assert_eq!(e.territory, "US");
        assert!(e.is_valid());
        assert!(e.expiry.is_none());
    }

    #[test]
    fn test_rights_entry_expiry() {
        let past = Utc::now() - Duration::days(1);
        let e = entry("asset-1", "US", LicenseType::RoyaltyFree).with_expiry(past);
        assert!(!e.is_valid());
    }

    #[test]
    fn test_rights_entry_expires_within() {
        let soon = Utc::now() + Duration::days(5);
        let e = entry("asset-1", "US", LicenseType::RoyaltyFree).with_expiry(soon);
        assert!(e.expires_within(10));
        assert!(!e.expires_within(3));
    }

    #[test]
    fn test_rights_entry_with_restriction() {
        let e = entry("asset-1", "US", LicenseType::CreativeCommonsByNc)
            .with_restriction(Restriction::NoCommercial);
        assert!(e.restrictions.contains(&Restriction::NoCommercial));
    }

    // --- RightsRegistry ---

    #[test]
    fn test_registry_register_and_lookup() {
        let mut registry = RightsRegistry::new();
        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree));
        let found = registry.lookup("vid-1", "US");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].asset_id, "vid-1");
    }

    #[test]
    fn test_registry_worldwide_fallback() {
        let mut registry = RightsRegistry::new();
        registry.register(entry("vid-1", "WW", LicenseType::RoyaltyFree));
        // Lookup for "JP" should find the WW entry
        let found = registry.lookup("vid-1", "JP");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_registry_expired_not_returned() {
        let mut registry = RightsRegistry::new();
        let past = Utc::now() - Duration::days(1);
        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree).with_expiry(past));
        assert!(registry.lookup("vid-1", "US").is_empty());
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = RightsRegistry::new();
        let e = entry("vid-1", "US", LicenseType::RoyaltyFree);
        let id = e.id.clone();
        registry.register(e);
        assert!(registry.remove(&id));
        assert!(registry.lookup("vid-1", "US").is_empty());
        // Removing again should return false
        assert!(!registry.remove(&id));
    }

    // --- UsageChecker ---

    #[test]
    fn test_usage_checker_allowed_commercial() {
        let mut registry = RightsRegistry::new();
        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree));
        let checker = UsageChecker::new(&registry);
        assert!(checker.is_allowed("vid-1", "US", &UseCase::Commercial));
    }

    #[test]
    fn test_usage_checker_blocked_commercial_cc_nc() {
        let mut registry = RightsRegistry::new();
        registry.register(entry("vid-1", "US", LicenseType::CreativeCommonsByNc));
        let checker = UsageChecker::new(&registry);
        assert!(!checker.is_allowed("vid-1", "US", &UseCase::Commercial));
        assert!(checker.is_allowed("vid-1", "US", &UseCase::NonCommercial));
    }

    #[test]
    fn test_usage_checker_no_rights() {
        let registry = RightsRegistry::new();
        let checker = UsageChecker::new(&registry);
        assert!(!checker.is_allowed("unknown", "US", &UseCase::Commercial));
    }

    #[test]
    fn test_usage_checker_explicit_no_commercial_restriction() {
        let mut registry = RightsRegistry::new();
        // RoyaltyFree allows commercial by license but we add NoCommercial restriction
        registry.register(
            entry("vid-1", "US", LicenseType::RoyaltyFree)
                .with_restriction(Restriction::NoCommercial),
        );
        let checker = UsageChecker::new(&registry);
        assert!(!checker.is_allowed("vid-1", "US", &UseCase::Commercial));
    }

    // --- ConflictDetector ---

    #[test]
    fn test_conflict_exclusive_conflicts() {
        let mut registry = RightsRegistry::new();
        registry.register(entry("vid-1", "US", LicenseType::Exclusive));
        registry.register(entry("vid-1", "US", LicenseType::NonExclusive));
        let detector = ConflictDetector::new(&registry);
        let conflicts = detector.detect();
        assert!(!conflicts.is_empty());
    }

    #[test]
    fn test_conflict_none_when_compatible() {
        let mut registry = RightsRegistry::new();
        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree));
        registry.register(entry("vid-1", "US", LicenseType::NonExclusive));
        let detector = ConflictDetector::new(&registry);
        let conflicts = detector.detect();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_conflict_share_alike_vs_no_derivatives() {
        let mut registry = RightsRegistry::new();
        registry.register(
            entry("vid-1", "US", LicenseType::CreativeCommonsBy)
                .with_restriction(Restriction::ShareAlike),
        );
        registry.register(
            entry("vid-1", "US", LicenseType::CreativeCommonsBy)
                .with_restriction(Restriction::NoDerivatives),
        );
        let detector = ConflictDetector::new(&registry);
        let conflicts = detector.detect();
        assert!(!conflicts.is_empty());
    }

    // --- ExpiryChecker ---

    #[test]
    fn test_expiry_checker_expiring_soon() {
        let mut registry = RightsRegistry::new();
        let soon = Utc::now() + Duration::days(3);
        let far = Utc::now() + Duration::days(60);

        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree).with_expiry(soon));
        registry.register(entry("vid-2", "US", LicenseType::RoyaltyFree).with_expiry(far));

        let checker = ExpiryChecker::new(&registry);
        let expiring = checker.expiring_soon(7);
        assert_eq!(expiring.len(), 1);
        assert_eq!(expiring[0].asset_id, "vid-1");
    }

    #[test]
    fn test_expiry_checker_already_expired() {
        let mut registry = RightsRegistry::new();
        let past = Utc::now() - Duration::days(1);
        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree).with_expiry(past));
        registry.register(entry("vid-2", "US", LicenseType::RoyaltyFree)); // no expiry

        let checker = ExpiryChecker::new(&registry);
        let expired = checker.already_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].asset_id, "vid-1");
    }

    #[test]
    fn test_expiry_checker_none_expiring() {
        let mut registry = RightsRegistry::new();
        let far = Utc::now() + Duration::days(90);
        registry.register(entry("vid-1", "US", LicenseType::RoyaltyFree).with_expiry(far));

        let checker = ExpiryChecker::new(&registry);
        assert!(checker.expiring_soon(7).is_empty());
    }

    #[test]
    fn test_use_case_is_commercial() {
        assert!(UseCase::Commercial.is_commercial());
        assert!(!UseCase::NonCommercial.is_commercial());
        assert!(!UseCase::Personal.is_commercial());
    }
}
