#![allow(dead_code)]
//! Entitlement management for DRM-protected content.
//!
//! Models subscription-based and purchase-based entitlements, providing
//! grant/revoke/check semantics for a content entitlement store.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Type of entitlement granted to a user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntitlementType {
    /// Subscription-based access (e.g., monthly plan).
    Subscription,
    /// One-time purchase of specific content.
    Purchase,
    /// Rental with a fixed expiry window.
    Rental,
    /// Promotional or trial access.
    Trial,
    /// Granted by a third-party bundle.
    Bundle,
}

impl EntitlementType {
    /// Returns `true` if this entitlement type requires an active subscription.
    pub fn requires_subscription(&self) -> bool {
        matches!(self, EntitlementType::Subscription | EntitlementType::Trial)
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            EntitlementType::Subscription => "Subscription",
            EntitlementType::Purchase => "Purchase",
            EntitlementType::Rental => "Rental",
            EntitlementType::Trial => "Trial",
            EntitlementType::Bundle => "Bundle",
        }
    }
}

/// A single entitlement record for a (user, content) pair.
#[derive(Debug, Clone)]
pub struct Entitlement {
    /// Unique entitlement identifier.
    pub id: String,
    /// User this entitlement belongs to.
    pub user_id: String,
    /// Content asset identifier.
    pub asset_id: String,
    /// Kind of entitlement.
    pub kind: EntitlementType,
    /// When this entitlement was granted (UNIX epoch seconds).
    pub granted_at: SystemTime,
    /// Optional expiry – `None` means it never expires.
    pub expires_at: Option<SystemTime>,
    /// Whether the entitlement has been explicitly revoked.
    pub revoked: bool,
}

impl Entitlement {
    /// Create a new non-expiring entitlement.
    pub fn new(
        id: impl Into<String>,
        user_id: impl Into<String>,
        asset_id: impl Into<String>,
        kind: EntitlementType,
    ) -> Self {
        Self {
            id: id.into(),
            user_id: user_id.into(),
            asset_id: asset_id.into(),
            kind,
            granted_at: SystemTime::now(),
            expires_at: None,
            revoked: false,
        }
    }

    /// Create a new entitlement with an explicit expiry duration from now.
    pub fn with_ttl(
        id: impl Into<String>,
        user_id: impl Into<String>,
        asset_id: impl Into<String>,
        kind: EntitlementType,
        ttl: Duration,
    ) -> Self {
        let mut e = Self::new(id, user_id, asset_id, kind);
        e.expires_at = Some(SystemTime::now() + ttl);
        e
    }

    /// Returns `true` when the entitlement is valid at the given instant:
    /// not revoked and not yet expired.
    pub fn is_valid_at(&self, when: SystemTime) -> bool {
        if self.revoked {
            return false;
        }
        match self.expires_at {
            Some(exp) => when < exp,
            None => true,
        }
    }

    /// Convenience: validity check against the current wall-clock time.
    pub fn is_currently_valid(&self) -> bool {
        self.is_valid_at(SystemTime::now())
    }
}

/// In-memory store for managing entitlements.
///
/// Keyed by entitlement ID for O(1) access.
#[derive(Debug, Default)]
pub struct EntitlementStore {
    records: HashMap<String, Entitlement>,
}

impl EntitlementStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Grant (insert) an entitlement. Overwrites any previous record with the same ID.
    pub fn grant(&mut self, entitlement: Entitlement) {
        self.records.insert(entitlement.id.clone(), entitlement);
    }

    /// Revoke an entitlement by ID. Returns `true` if the record existed.
    pub fn revoke(&mut self, entitlement_id: &str) -> bool {
        if let Some(record) = self.records.get_mut(entitlement_id) {
            record.revoked = true;
            true
        } else {
            false
        }
    }

    /// Check whether `user_id` holds a currently-valid entitlement for `asset_id`.
    pub fn check(&self, user_id: &str, asset_id: &str) -> bool {
        let now = SystemTime::now();
        self.records
            .values()
            .any(|e| e.user_id == user_id && e.asset_id == asset_id && e.is_valid_at(now))
    }

    /// Look up a single entitlement by ID.
    pub fn get(&self, entitlement_id: &str) -> Option<&Entitlement> {
        self.records.get(entitlement_id)
    }

    /// Return all valid entitlements for a user at the current time.
    pub fn valid_for_user(&self, user_id: &str) -> Vec<&Entitlement> {
        let now = SystemTime::now();
        self.records
            .values()
            .filter(|e| e.user_id == user_id && e.is_valid_at(now))
            .collect()
    }

    /// Total number of entitlement records (including revoked/expired ones).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if the store contains no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // EntitlementType tests

    #[test]
    fn test_subscription_requires_subscription() {
        assert!(EntitlementType::Subscription.requires_subscription());
    }

    #[test]
    fn test_trial_requires_subscription() {
        assert!(EntitlementType::Trial.requires_subscription());
    }

    #[test]
    fn test_purchase_does_not_require_subscription() {
        assert!(!EntitlementType::Purchase.requires_subscription());
    }

    #[test]
    fn test_rental_does_not_require_subscription() {
        assert!(!EntitlementType::Rental.requires_subscription());
    }

    #[test]
    fn test_bundle_does_not_require_subscription() {
        assert!(!EntitlementType::Bundle.requires_subscription());
    }

    #[test]
    fn test_label_is_non_empty() {
        for kind in &[
            EntitlementType::Subscription,
            EntitlementType::Purchase,
            EntitlementType::Rental,
            EntitlementType::Trial,
            EntitlementType::Bundle,
        ] {
            assert!(!kind.label().is_empty());
        }
    }

    // Entitlement validity tests

    #[test]
    fn test_is_valid_at_no_expiry() {
        let e = Entitlement::new("e1", "u1", "a1", EntitlementType::Purchase);
        let future = SystemTime::now() + Duration::from_secs(100);
        assert!(e.is_valid_at(future));
    }

    #[test]
    fn test_is_valid_before_expiry() {
        let e = Entitlement::with_ttl(
            "e2",
            "u1",
            "a1",
            EntitlementType::Rental,
            Duration::from_hours(1),
        );
        let soon = SystemTime::now() + Duration::from_mins(1);
        assert!(e.is_valid_at(soon));
    }

    #[test]
    fn test_is_not_valid_after_expiry() {
        let mut e = Entitlement::new("e3", "u1", "a1", EntitlementType::Rental);
        // Set expiry in the past
        e.expires_at = Some(SystemTime::now() - Duration::from_secs(1));
        assert!(!e.is_valid_at(SystemTime::now()));
    }

    #[test]
    fn test_is_not_valid_when_revoked() {
        let mut e = Entitlement::new("e4", "u1", "a1", EntitlementType::Purchase);
        e.revoked = true;
        let future = SystemTime::now() + Duration::from_secs(100);
        assert!(!e.is_valid_at(future));
    }

    // EntitlementStore tests

    #[test]
    fn test_store_grant_and_check() {
        let mut store = EntitlementStore::new();
        let e = Entitlement::new("e5", "user_a", "asset_x", EntitlementType::Purchase);
        store.grant(e);
        assert!(store.check("user_a", "asset_x"));
    }

    #[test]
    fn test_store_check_wrong_user_fails() {
        let mut store = EntitlementStore::new();
        let e = Entitlement::new("e6", "user_a", "asset_x", EntitlementType::Purchase);
        store.grant(e);
        assert!(!store.check("user_b", "asset_x"));
    }

    #[test]
    fn test_store_revoke() {
        let mut store = EntitlementStore::new();
        let e = Entitlement::new("e7", "user_a", "asset_y", EntitlementType::Subscription);
        store.grant(e);
        assert!(store.check("user_a", "asset_y"));
        assert!(store.revoke("e7"));
        assert!(!store.check("user_a", "asset_y"));
    }

    #[test]
    fn test_store_revoke_nonexistent_returns_false() {
        let mut store = EntitlementStore::new();
        assert!(!store.revoke("no_such_id"));
    }

    #[test]
    fn test_store_len_and_is_empty() {
        let mut store = EntitlementStore::new();
        assert!(store.is_empty());
        store.grant(Entitlement::new("e8", "u1", "a1", EntitlementType::Trial));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_store_valid_for_user() {
        let mut store = EntitlementStore::new();
        store.grant(Entitlement::new(
            "e9",
            "u2",
            "a1",
            EntitlementType::Purchase,
        ));
        // Insert an already-expired entitlement
        let mut expired = Entitlement::new("e10", "u2", "a2", EntitlementType::Rental);
        expired.expires_at = Some(SystemTime::now() - Duration::from_secs(1));
        store.grant(expired);
        let valid = store.valid_for_user("u2");
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0].asset_id, "a1");
    }
}
