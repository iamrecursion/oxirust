//! In-memory `RightsManager` alternative for `wasm32` targets.
//!
//! On non-wasm32 targets, `RightsManager` uses Pure-Rust SQLite via OxiSQL. On wasm32
//! no file-system or async runtime is available, so this module provides a
//! fully in-memory, synchronous alternative backed by `HashMap`.
//!
//! The API mirrors the persistent manager as closely as possible so that
//! application code can be compiled for both targets with minimal `#[cfg]` usage.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::rights_check::{CheckRequest, CheckResult, RightsGrant};
use crate::{Result, RightsError};

// ── WasmRightsRecord ────────────────────────────────────────────────────────

/// A simplified rights record stored in the wasm-compatible manager.
#[derive(Debug, Clone)]
pub struct WasmRightsRecord {
    /// Unique record identifier.
    pub id: String,
    /// Asset this record applies to.
    pub asset_id: String,
    /// Display name of the rights holder.
    pub holder: String,
    /// Whether this record is currently active.
    pub active: bool,
    /// Unix timestamp (seconds) when the right was granted.
    pub granted_at: u64,
    /// Optional expiry (Unix seconds).
    pub expires_at: Option<u64>,
    /// Free-text notes.
    pub notes: String,
}

impl WasmRightsRecord {
    /// Create a new active record.
    #[must_use]
    pub fn new(id: &str, asset_id: &str, holder: &str, granted_at: u64) -> Self {
        Self {
            id: id.to_string(),
            asset_id: asset_id.to_string(),
            holder: holder.to_string(),
            active: true,
            granted_at,
            expires_at: None,
            notes: String::new(),
        }
    }

    /// Builder: set expiry timestamp.
    #[must_use]
    pub fn with_expiry(mut self, ts: u64) -> Self {
        self.expires_at = Some(ts);
        self
    }

    /// Builder: set notes.
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }

    /// Whether this record has expired at the given timestamp.
    #[must_use]
    pub fn is_expired_at(&self, now: u64) -> bool {
        self.expires_at.map_or(false, |exp| now >= exp)
    }

    /// Whether the record is currently valid (active and not expired).
    #[must_use]
    pub fn is_valid_at(&self, now: u64) -> bool {
        self.active && !self.is_expired_at(now)
    }
}

// ── WasmRightsManager ───────────────────────────────────────────────────────

/// In-memory rights manager suitable for wasm32 targets.
///
/// All state is held in `HashMap`s; there is no persistence across page
/// reloads. Use this struct on wasm32, and `RightsManager` on native targets.
///
/// # Example
/// ```
/// use oximedia_rights::rights_manager_wasm::WasmRightsManager;
/// use oximedia_rights::rights_check::{ActionKind, CheckRequest};
///
/// let mut mgr = WasmRightsManager::new();
/// mgr.add_record(
///     oximedia_rights::rights_manager_wasm::WasmRightsRecord::new("r1", "asset-1", "Alice", 0)
/// );
/// assert_eq!(mgr.record_count(), 1);
/// ```
#[derive(Debug, Default)]
pub struct WasmRightsManager {
    /// Records keyed by their ID.
    records: HashMap<String, WasmRightsRecord>,
    /// Rights grants keyed by grant ID.
    grants: HashMap<String, RightsGrant>,
    /// Asset-to-grant index: asset_id → list of grant IDs.
    asset_grants: HashMap<String, Vec<String>>,
}

impl WasmRightsManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── Record management ──────────────────────────────────────────────────

    /// Insert or replace a rights record.
    pub fn add_record(&mut self, record: WasmRightsRecord) {
        self.records.insert(record.id.clone(), record);
    }

    /// Remove a record by ID, returning it if found.
    pub fn remove_record(&mut self, id: &str) -> Option<WasmRightsRecord> {
        self.records.remove(id)
    }

    /// Look up a record by ID.
    #[must_use]
    pub fn get_record(&self, id: &str) -> Option<&WasmRightsRecord> {
        self.records.get(id)
    }

    /// Return all records for a specific asset.
    #[must_use]
    pub fn records_for_asset(&self, asset_id: &str) -> Vec<&WasmRightsRecord> {
        self.records
            .values()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// Return all active records at a given timestamp.
    #[must_use]
    pub fn active_records_at(&self, now: u64) -> Vec<&WasmRightsRecord> {
        self.records
            .values()
            .filter(|r| r.is_valid_at(now))
            .collect()
    }

    /// Total number of records.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Deactivate a record by ID.
    ///
    /// Returns `Err` if the record does not exist.
    pub fn deactivate_record(&mut self, id: &str) -> Result<()> {
        match self.records.get_mut(id) {
            Some(r) => {
                r.active = false;
                Ok(())
            }
            None => Err(RightsError::NotFound(format!("Record not found: {id}"))),
        }
    }

    // ── Grant management ───────────────────────────────────────────────────

    /// Register a rights grant.
    pub fn add_grant(&mut self, grant: RightsGrant) {
        self.asset_grants
            .entry(grant.asset_id.clone())
            .or_default()
            .push(grant.id.clone());
        self.grants.insert(grant.id.clone(), grant);
    }

    /// Remove a grant by ID.
    pub fn remove_grant(&mut self, id: &str) -> Option<RightsGrant> {
        if let Some(grant) = self.grants.remove(id) {
            if let Some(ids) = self.asset_grants.get_mut(&grant.asset_id) {
                ids.retain(|g| g != id);
            }
            Some(grant)
        } else {
            None
        }
    }

    /// Revoke a grant (mark it revoked without removing it).
    ///
    /// Returns `Err` if the grant does not exist.
    pub fn revoke_grant(&mut self, id: &str) -> Result<()> {
        match self.grants.get_mut(id) {
            Some(g) => {
                g.revoked = true;
                Ok(())
            }
            None => Err(RightsError::NotFound(format!("Grant not found: {id}"))),
        }
    }

    /// List all grants for a given asset.
    #[must_use]
    pub fn grants_for_asset(&self, asset_id: &str) -> Vec<&RightsGrant> {
        self.asset_grants
            .get(asset_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.grants.get(id.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total number of grants.
    #[must_use]
    pub fn grant_count(&self) -> usize {
        self.grants.len()
    }

    // ── Rights check ───────────────────────────────────────────────────────

    /// Check whether an action is permitted for an asset.
    ///
    /// Evaluates all registered grants for the asset against the request
    /// parameters.
    #[must_use]
    pub fn check(&self, req: &CheckRequest) -> CheckResult {
        let grants = self.grants_for_asset(&req.asset_id);
        for grant in grants {
            if grant.revoked {
                continue;
            }
            if !grant.permits_action(req.action) {
                continue;
            }
            if !grant.is_valid_at(req.now) {
                continue;
            }
            if !grant.covers_territory(&req.territory) {
                continue;
            }
            if !grant.covers_platform(&req.platform) {
                continue;
            }
            return CheckResult::Allowed(grant.id.clone());
        }
        CheckResult::Denied(format!(
            "No grant for asset={} action={:?} territory={} platform={}",
            req.asset_id, req.action, req.territory, req.platform,
        ))
    }

    /// Convenience: return `true` if the action is allowed.
    #[must_use]
    pub fn is_allowed(&self, req: &CheckRequest) -> bool {
        self.check(req).is_allowed()
    }

    // ── Bulk query helpers ─────────────────────────────────────────────────

    /// Return all asset IDs that have at least one registered record.
    #[must_use]
    pub fn all_asset_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self
            .records
            .values()
            .map(|r| r.asset_id.as_str())
            .chain(self.grants.values().map(|g| g.asset_id.as_str()))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Expire all records whose `expires_at` is past `now`.
    ///
    /// Returns the number of records deactivated.
    pub fn expire_records(&mut self, now: u64) -> usize {
        let mut count = 0;
        for record in self.records.values_mut() {
            if record.active && record.is_expired_at(now) {
                record.active = false;
                count += 1;
            }
        }
        count
    }

    /// Clear all state (useful for testing).
    pub fn clear(&mut self) {
        self.records.clear();
        self.grants.clear();
        self.asset_grants.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights_check::ActionKind;

    fn sample_manager() -> WasmRightsManager {
        let mut mgr = WasmRightsManager::new();
        mgr.add_record(WasmRightsRecord::new("r1", "asset-A", "Alice", 0).with_expiry(10_000));
        mgr.add_record(WasmRightsRecord::new("r2", "asset-B", "Bob", 500));
        mgr
    }

    #[test]
    fn test_record_count() {
        assert_eq!(sample_manager().record_count(), 2);
    }

    #[test]
    fn test_get_record_found() {
        let mgr = sample_manager();
        let r = mgr.get_record("r1");
        assert!(r.is_some());
        assert_eq!(r.expect("test expects record").asset_id, "asset-A");
    }

    #[test]
    fn test_get_record_not_found() {
        assert!(sample_manager().get_record("ghost").is_none());
    }

    #[test]
    fn test_records_for_asset() {
        let mgr = sample_manager();
        assert_eq!(mgr.records_for_asset("asset-A").len(), 1);
        assert_eq!(mgr.records_for_asset("nonexistent").len(), 0);
    }

    #[test]
    fn test_active_records_at_within_expiry() {
        let mgr = sample_manager();
        // r1 expires at 10_000; r2 has no expiry
        let active = mgr.active_records_at(5_000);
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_active_records_at_past_expiry() {
        let mgr = sample_manager();
        // r1 has expired at 10_000
        let active = mgr.active_records_at(10_001);
        // Only r2 (no expiry) remains
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "r2");
    }

    #[test]
    fn test_remove_record() {
        let mut mgr = sample_manager();
        let removed = mgr.remove_record("r1");
        assert!(removed.is_some());
        assert_eq!(mgr.record_count(), 1);
    }

    #[test]
    fn test_deactivate_record() {
        let mut mgr = sample_manager();
        let result = mgr.deactivate_record("r2");
        assert!(result.is_ok());
        assert!(!mgr.get_record("r2").expect("r2 should exist").active);
    }

    #[test]
    fn test_deactivate_record_not_found() {
        let mut mgr = sample_manager();
        assert!(mgr.deactivate_record("ghost").is_err());
    }

    #[test]
    fn test_expire_records() {
        let mut mgr = sample_manager();
        let count = mgr.expire_records(10_001);
        assert_eq!(count, 1);
        assert!(!mgr.get_record("r1").expect("r1 exists").active);
    }

    #[test]
    fn test_grant_add_and_check_allowed() {
        let mut mgr = WasmRightsManager::new();
        let grant = RightsGrant::new("g1", "asset-X")
            .with_action(ActionKind::Stream)
            .with_window(0, u64::MAX);
        mgr.add_grant(grant);

        let req = CheckRequest::new("asset-X", ActionKind::Stream, "US", "web", 100);
        assert!(mgr.is_allowed(&req));
    }

    #[test]
    fn test_grant_check_denied_wrong_action() {
        let mut mgr = WasmRightsManager::new();
        let grant = RightsGrant::new("g2", "asset-Y")
            .with_action(ActionKind::Download)
            .with_window(0, u64::MAX);
        mgr.add_grant(grant);

        let req = CheckRequest::new("asset-Y", ActionKind::Stream, "US", "web", 100);
        assert!(!mgr.is_allowed(&req));
    }

    #[test]
    fn test_revoke_grant() {
        let mut mgr = WasmRightsManager::new();
        let grant = RightsGrant::new("g3", "asset-Z")
            .with_action(ActionKind::Stream)
            .with_window(0, u64::MAX);
        mgr.add_grant(grant);

        mgr.revoke_grant("g3").expect("revoke should succeed");
        let req = CheckRequest::new("asset-Z", ActionKind::Stream, "US", "web", 100);
        assert!(!mgr.is_allowed(&req));
    }

    #[test]
    fn test_revoke_grant_not_found() {
        let mut mgr = WasmRightsManager::new();
        assert!(mgr.revoke_grant("missing").is_err());
    }

    #[test]
    fn test_remove_grant() {
        let mut mgr = WasmRightsManager::new();
        mgr.add_grant(
            RightsGrant::new("g4", "asset-W")
                .with_action(ActionKind::Stream)
                .with_window(0, u64::MAX),
        );
        assert_eq!(mgr.grant_count(), 1);
        let removed = mgr.remove_grant("g4");
        assert!(removed.is_some());
        assert_eq!(mgr.grant_count(), 0);
    }

    #[test]
    fn test_grants_for_asset() {
        let mut mgr = WasmRightsManager::new();
        mgr.add_grant(
            RightsGrant::new("g5", "asset-V")
                .with_action(ActionKind::Broadcast)
                .with_window(0, u64::MAX),
        );
        mgr.add_grant(
            RightsGrant::new("g6", "asset-V")
                .with_action(ActionKind::Archive)
                .with_window(0, u64::MAX),
        );
        assert_eq!(mgr.grants_for_asset("asset-V").len(), 2);
        assert_eq!(mgr.grants_for_asset("other").len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut mgr = sample_manager();
        mgr.clear();
        assert_eq!(mgr.record_count(), 0);
    }

    #[test]
    fn test_all_asset_ids() {
        let mgr = sample_manager();
        let ids = mgr.all_asset_ids();
        assert!(ids.contains(&"asset-A"));
        assert!(ids.contains(&"asset-B"));
    }

    #[test]
    fn test_wasm_record_is_expired() {
        let r = WasmRightsRecord::new("r", "a", "h", 0).with_expiry(100);
        assert!(!r.is_expired_at(99));
        assert!(r.is_expired_at(100));
        assert!(r.is_expired_at(200));
    }

    #[test]
    fn test_wasm_record_no_expiry() {
        let r = WasmRightsRecord::new("r", "a", "h", 0);
        assert!(!r.is_expired_at(u64::MAX));
        assert!(r.is_valid_at(999_999_999));
    }
}
