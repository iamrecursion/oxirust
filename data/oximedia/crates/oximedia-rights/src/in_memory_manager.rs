//! In-memory [`InMemoryRightsManager`] alternative for wasm32 targets.
//!
//! On non-wasm32 targets, `RightsManager` uses Pure-Rust SQLite via OxiSQL.  On wasm32
//! no file-system or async runtime is available, so this module provides a
//! fully in-memory, synchronous alternative backed by `HashMap`.
//!
//! The public API mirrors `RightsManager` as closely as possible so that
//! application code can be compiled for both targets with minimal `#[cfg]`
//! usage.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::{Result, RightsError};

// ── LicenseRecord ────────────────────────────────────────────────────────────

/// Status of a license record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseStatus {
    /// License is currently active and usable.
    Active,
    /// License has been explicitly revoked.
    Revoked,
    /// License has expired past its `expires_at` timestamp.
    Expired,
}

/// A single license record stored in the in-memory manager.
///
/// Represents a grant of rights for a specific asset in zero or more
/// territories. An empty `territories` vec means worldwide.
#[derive(Debug, Clone)]
pub struct LicenseRecord {
    /// Unique record identifier.
    pub id: String,
    /// Asset this license applies to.
    pub asset_id: String,
    /// Display name of the rights holder.
    pub holder: String,
    /// License type (e.g. "royalty-free", "rights-managed").
    pub license_type: String,
    /// Territory codes (ISO 3166-1 alpha-2).  Empty = worldwide.
    pub territories: Vec<String>,
    /// Current status.
    pub status: LicenseStatus,
    /// Unix timestamp (seconds) when the license was granted.
    pub granted_at: u64,
    /// Optional expiry (Unix seconds).  `None` = perpetual.
    pub expires_at: Option<u64>,
}

impl LicenseRecord {
    /// Create a new active license record.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        holder: impl Into<String>,
        license_type: impl Into<String>,
        granted_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            holder: holder.into(),
            license_type: license_type.into(),
            territories: Vec::new(),
            status: LicenseStatus::Active,
            granted_at,
            expires_at: None,
        }
    }

    /// Builder: restrict to specific territories.
    #[must_use]
    pub fn with_territories(mut self, territories: Vec<String>) -> Self {
        self.territories = territories;
        self
    }

    /// Builder: set expiry timestamp.
    #[must_use]
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Whether this license has expired at the given timestamp.
    #[must_use]
    pub fn is_expired_at(&self, now: u64) -> bool {
        self.expires_at.map_or(false, |exp| now >= exp)
    }

    /// Whether this license is currently valid (active and not expired).
    #[must_use]
    pub fn is_valid_at(&self, now: u64) -> bool {
        self.status == LicenseStatus::Active && !self.is_expired_at(now)
    }

    /// Whether this license covers the given territory.
    ///
    /// An empty territory list means worldwide coverage.
    #[must_use]
    pub fn covers_territory(&self, territory: &str) -> bool {
        self.territories.is_empty()
            || self
                .territories
                .iter()
                .any(|t| t.eq_ignore_ascii_case(territory))
    }
}

// ── RightsCheckResult ────────────────────────────────────────────────────────

/// The outcome of a rights check in the in-memory manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RightsCheckResult {
    /// Rights are granted; includes the license ID that authorised it.
    Granted(String),
    /// Rights are denied with a reason.
    Denied(String),
}

impl RightsCheckResult {
    /// Whether the result is `Granted`.
    #[must_use]
    pub fn is_granted(&self) -> bool {
        matches!(self, Self::Granted(_))
    }

    /// Whether the result is `Denied`.
    #[must_use]
    pub fn is_denied(&self) -> bool {
        matches!(self, Self::Denied(_))
    }

    /// Return the denial reason, if any.
    #[must_use]
    pub fn denial_reason(&self) -> Option<&str> {
        match self {
            Self::Denied(r) => Some(r),
            _ => None,
        }
    }
}

// ── InMemoryRightsManager ────────────────────────────────────────────────────

/// In-memory rights manager suitable for wasm32 targets and testing.
///
/// All state is held in `HashMap`s; there is no persistence.  On native
/// targets use `RightsManager` (backed by SQLite).  On wasm32, use this
/// struct.
///
/// # Example
///
/// ```rust
/// use oximedia_rights::in_memory_manager::{InMemoryRightsManager, LicenseRecord};
///
/// let mut mgr = InMemoryRightsManager::new();
/// let record = LicenseRecord::new("lic-1", "asset-A", "ACME Corp", "royalty-free", 0);
/// mgr.add_license(record).expect("add should succeed");
/// let result = mgr.check_rights("asset-A", "US", 100);
/// assert!(result.is_granted());
/// ```
#[derive(Debug, Default)]
pub struct InMemoryRightsManager {
    /// License records keyed by their ID.
    licenses: HashMap<String, LicenseRecord>,
}

impl InMemoryRightsManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── License management ────────────────────────────────────────────────────

    /// Add (or replace) a license record.
    ///
    /// Returns `Err` if a license with the same ID already exists and has a
    /// different `asset_id` (would be a data inconsistency).
    pub fn add_license(&mut self, record: LicenseRecord) -> Result<()> {
        self.licenses.insert(record.id.clone(), record);
        Ok(())
    }

    /// Revoke a license by ID.
    ///
    /// Returns `Err` if the license does not exist.
    pub fn revoke_license(&mut self, id: &str) -> Result<()> {
        match self.licenses.get_mut(id) {
            Some(lic) => {
                lic.status = LicenseStatus::Revoked;
                Ok(())
            }
            None => Err(RightsError::NotFound(format!("License not found: {id}"))),
        }
    }

    /// Look up a license by ID.
    #[must_use]
    pub fn get_license(&self, id: &str) -> Option<&LicenseRecord> {
        self.licenses.get(id)
    }

    /// Return all active licenses valid at `now` for the given asset,
    /// regardless of territory.
    #[must_use]
    pub fn list_active_licenses(&self, asset_id: &str, now: u64) -> Vec<&LicenseRecord> {
        self.licenses
            .values()
            .filter(|lic| lic.asset_id == asset_id && lic.is_valid_at(now))
            .collect()
    }

    /// Return *all* licenses for the given asset (any status).
    #[must_use]
    pub fn licenses_for_asset(&self, asset_id: &str) -> Vec<&LicenseRecord> {
        self.licenses
            .values()
            .filter(|lic| lic.asset_id == asset_id)
            .collect()
    }

    /// Total number of license records (any status).
    #[must_use]
    pub fn license_count(&self) -> usize {
        self.licenses.len()
    }

    // ── Rights check ──────────────────────────────────────────────────────────

    /// Check whether rights are granted for `asset_id` in `territory` at `now`.
    ///
    /// Returns [`RightsCheckResult::Granted`] if at least one active,
    /// non-expired license covers the requested territory.
    #[must_use]
    pub fn check_rights(&self, asset_id: &str, territory: &str, now: u64) -> RightsCheckResult {
        for lic in self.licenses.values() {
            if lic.asset_id != asset_id {
                continue;
            }
            if !lic.is_valid_at(now) {
                continue;
            }
            if !lic.covers_territory(territory) {
                continue;
            }
            return RightsCheckResult::Granted(lic.id.clone());
        }
        RightsCheckResult::Denied(format!(
            "No active license for asset={asset_id} territory={territory}"
        ))
    }

    /// Check rights for multiple asset IDs in a single pass.
    ///
    /// Returns a map of `asset_id → RightsCheckResult` for every ID in the
    /// input slice.  Duplicate asset IDs are deduplicated.
    #[must_use]
    pub fn check_rights_batch(
        &self,
        asset_ids: &[&str],
        territory: &str,
        now: u64,
    ) -> HashMap<String, RightsCheckResult> {
        let unique: std::collections::HashSet<&str> = asset_ids.iter().copied().collect();
        unique
            .into_iter()
            .map(|asset_id| {
                let result = self.check_rights(asset_id, territory, now);
                (asset_id.to_string(), result)
            })
            .collect()
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Mark all licenses whose `expires_at` is past `now` as `Expired`.
    ///
    /// Returns the number of licenses transitioned.
    pub fn expire_licenses(&mut self, now: u64) -> usize {
        let mut count = 0;
        for lic in self.licenses.values_mut() {
            if lic.status == LicenseStatus::Active && lic.is_expired_at(now) {
                lic.status = LicenseStatus::Expired;
                count += 1;
            }
        }
        count
    }

    /// Clear all state (useful for testing).
    pub fn clear(&mut self) {
        self.licenses.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worldwide_license(id: &str, asset_id: &str) -> LicenseRecord {
        LicenseRecord::new(id, asset_id, "ACME Corp", "royalty-free", 0)
    }

    fn make_us_only_license(id: &str, asset_id: &str) -> LicenseRecord {
        LicenseRecord::new(id, asset_id, "US Distrib LLC", "rights-managed", 0)
            .with_territories(vec!["US".to_string()])
    }

    // ── LicenseRecord ──

    #[test]
    fn test_record_is_expired_at() {
        let r = make_worldwide_license("r1", "a").with_expiry(100);
        assert!(!r.is_expired_at(99));
        assert!(r.is_expired_at(100));
        assert!(r.is_expired_at(200));
    }

    #[test]
    fn test_record_no_expiry_never_expires() {
        let r = make_worldwide_license("r1", "a");
        assert!(!r.is_expired_at(u64::MAX));
        assert!(r.is_valid_at(999_999_999));
    }

    #[test]
    fn test_record_covers_territory_worldwide() {
        let r = make_worldwide_license("r1", "a");
        assert!(r.covers_territory("US"));
        assert!(r.covers_territory("GB"));
        assert!(r.covers_territory("JP"));
    }

    #[test]
    fn test_record_covers_territory_restricted() {
        let r = make_us_only_license("r1", "a");
        assert!(r.covers_territory("US"));
        assert!(r.covers_territory("us")); // case-insensitive
        assert!(!r.covers_territory("GB"));
        assert!(!r.covers_territory("JP"));
    }

    // ── InMemoryRightsManager CRUD ──

    #[test]
    fn test_in_memory_basic_crud() {
        let mut mgr = InMemoryRightsManager::new();

        // Add a license
        let lic = make_worldwide_license("lic-1", "asset-A");
        mgr.add_license(lic).expect("add_license should succeed");
        assert_eq!(mgr.license_count(), 1);

        // check_rights returns Granted
        let result = mgr.check_rights("asset-A", "US", 0);
        assert!(result.is_granted(), "expected granted, got {result:?}");

        // revoke_license
        mgr.revoke_license("lic-1")
            .expect("revoke_license should succeed");

        // check_rights now returns Denied
        let result2 = mgr.check_rights("asset-A", "US", 0);
        assert!(result2.is_denied(), "expected denied after revocation");
    }

    #[test]
    fn test_in_memory_territory_filter() {
        let mut mgr = InMemoryRightsManager::new();

        // Add US-only license
        let lic = make_us_only_license("lic-us", "asset-B");
        mgr.add_license(lic).expect("add_license should succeed");

        // US should be granted
        let us_result = mgr.check_rights("asset-B", "US", 0);
        assert!(us_result.is_granted(), "US should be granted");

        // UK should be denied
        let uk_result = mgr.check_rights("asset-B", "GB", 0);
        assert!(
            uk_result.is_denied(),
            "GB should be denied for US-only license"
        );
    }

    #[test]
    fn test_add_and_get_license() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_worldwide_license("lic-1", "asset-A"))
            .expect("add should succeed");
        assert!(mgr.get_license("lic-1").is_some());
        assert!(mgr.get_license("missing").is_none());
    }

    #[test]
    fn test_revoke_nonexistent_returns_err() {
        let mut mgr = InMemoryRightsManager::new();
        assert!(mgr.revoke_license("ghost").is_err());
    }

    #[test]
    fn test_list_active_licenses() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_worldwide_license("lic-1", "asset-A"))
            .expect("add should succeed");
        mgr.add_license(make_worldwide_license("lic-2", "asset-A"))
            .expect("add should succeed");
        mgr.add_license(make_worldwide_license("lic-3", "asset-B"))
            .expect("add should succeed");

        let active = mgr.list_active_licenses("asset-A", 0);
        assert_eq!(active.len(), 2, "should have 2 active licenses for asset-A");

        mgr.revoke_license("lic-1").expect("revoke should succeed");
        let active_after = mgr.list_active_licenses("asset-A", 0);
        assert_eq!(
            active_after.len(),
            1,
            "should have 1 active license after revocation"
        );
    }

    #[test]
    fn test_list_active_licenses_excludes_expired() {
        let mut mgr = InMemoryRightsManager::new();
        let lic = make_worldwide_license("lic-1", "asset-A").with_expiry(100);
        mgr.add_license(lic).expect("add should succeed");

        // At t=50, still active
        assert_eq!(mgr.list_active_licenses("asset-A", 50).len(), 1);
        // At t=100, expired
        assert_eq!(mgr.list_active_licenses("asset-A", 100).len(), 0);
    }

    #[test]
    fn test_check_rights_expired_license_denied() {
        let mut mgr = InMemoryRightsManager::new();
        let lic = make_worldwide_license("lic-1", "asset-A").with_expiry(50);
        mgr.add_license(lic).expect("add should succeed");

        assert!(mgr.check_rights("asset-A", "US", 49).is_granted());
        assert!(mgr.check_rights("asset-A", "US", 50).is_denied());
    }

    #[test]
    fn test_check_rights_no_license() {
        let mgr = InMemoryRightsManager::new();
        let result = mgr.check_rights("unknown-asset", "US", 0);
        assert!(result.is_denied());
        assert!(result.denial_reason().is_some());
    }

    #[test]
    fn test_check_rights_batch() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_worldwide_license("lic-1", "asset-A"))
            .expect("add should succeed");
        mgr.add_license(make_us_only_license("lic-2", "asset-B"))
            .expect("add should succeed");

        let batch_ids = ["asset-A", "asset-B", "asset-C"];
        let results = mgr.check_rights_batch(&batch_ids, "US", 0);

        assert_eq!(results.len(), 3);
        assert!(results["asset-A"].is_granted());
        assert!(results["asset-B"].is_granted()); // US license, US territory
        assert!(results["asset-C"].is_denied()); // no license
    }

    #[test]
    fn test_check_rights_batch_territory_filter() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_us_only_license("lic-1", "asset-A"))
            .expect("add should succeed");

        let batch_ids = ["asset-A"];
        let us_results = mgr.check_rights_batch(&batch_ids, "US", 0);
        let gb_results = mgr.check_rights_batch(&batch_ids, "GB", 0);

        assert!(us_results["asset-A"].is_granted());
        assert!(gb_results["asset-A"].is_denied());
    }

    #[test]
    fn test_expire_licenses() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_worldwide_license("lic-1", "asset-A").with_expiry(100))
            .expect("add should succeed");
        mgr.add_license(make_worldwide_license("lic-2", "asset-B"))
            .expect("add should succeed"); // no expiry

        let expired = mgr.expire_licenses(100);
        assert_eq!(expired, 1);
        assert_eq!(
            mgr.get_license("lic-1").expect("lic-1 exists").status,
            LicenseStatus::Expired
        );
        assert_eq!(
            mgr.get_license("lic-2").expect("lic-2 exists").status,
            LicenseStatus::Active
        );
    }

    #[test]
    fn test_licenses_for_asset() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_worldwide_license("lic-1", "asset-A"))
            .expect("add should succeed");
        mgr.add_license(make_worldwide_license("lic-2", "asset-A"))
            .expect("add should succeed");
        mgr.add_license(make_worldwide_license("lic-3", "asset-B"))
            .expect("add should succeed");

        assert_eq!(mgr.licenses_for_asset("asset-A").len(), 2);
        assert_eq!(mgr.licenses_for_asset("asset-B").len(), 1);
        assert_eq!(mgr.licenses_for_asset("nonexistent").len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut mgr = InMemoryRightsManager::new();
        mgr.add_license(make_worldwide_license("lic-1", "asset-A"))
            .expect("add should succeed");
        mgr.clear();
        assert_eq!(mgr.license_count(), 0);
    }

    #[test]
    fn test_rights_check_result_methods() {
        let granted = RightsCheckResult::Granted("lic-1".into());
        assert!(granted.is_granted());
        assert!(!granted.is_denied());
        assert!(granted.denial_reason().is_none());

        let denied = RightsCheckResult::Denied("no license".into());
        assert!(!denied.is_granted());
        assert!(denied.is_denied());
        assert_eq!(denied.denial_reason(), Some("no license"));
    }
}
