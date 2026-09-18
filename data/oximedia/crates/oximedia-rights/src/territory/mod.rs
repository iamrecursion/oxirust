//! Territory-based rights management module.

#![allow(dead_code)]

pub mod restrict;
pub mod rights;
pub mod validate;
pub mod zone;

pub use restrict::TerritoryRestriction;
pub use rights::{Territory, TerritoryRights};
pub use validate::TerritoryValidator;
pub use zone::{TerritoryZone, WorldRegion};

// ── TerritoryCode ─────────────────────────────────────────────────────────────

/// An ISO 3166-1 territory code, or the special "WW" sentinel for worldwide rights
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerritoryCode {
    /// The code string (e.g. "US", "GB", "WW")
    pub code: String,
}

impl TerritoryCode {
    /// Create a territory code from a string
    pub fn new(code: &str) -> Self {
        Self {
            code: code.to_uppercase(),
        }
    }

    /// The special worldwide sentinel – represents all territories
    pub fn worldwide() -> Self {
        Self::new("WW")
    }

    /// Returns `true` if this code represents worldwide coverage
    pub fn is_worldwide(&self) -> bool {
        self.code == "WW"
    }

    /// Returns `true` if this territory includes `other`.
    ///
    /// Worldwide includes everything; any other code only includes itself.
    pub fn includes(&self, other: &TerritoryCode) -> bool {
        self.is_worldwide() || self.code == other.code
    }
}

// ── TerritoryRight ────────────────────────────────────────────────────────────

/// A time-bounded, territory-scoped rights record for one asset
#[derive(Debug, Clone)]
pub struct TerritoryRight {
    /// Identifier for the asset this right covers
    pub asset_id: String,
    /// Geographic scope of this right
    pub territory: TerritoryCode,
    /// Type of right (e.g. "broadcast", "streaming", "theatrical")
    pub right_type: String,
    /// Unix-epoch ms when this right becomes valid
    pub valid_from_ms: u64,
    /// Unix-epoch ms when this right expires (`None` = no expiry)
    pub valid_until_ms: Option<u64>,
}

impl TerritoryRight {
    /// Create a new territory right
    pub fn new(
        asset_id: &str,
        territory: TerritoryCode,
        right_type: &str,
        valid_from_ms: u64,
        valid_until_ms: Option<u64>,
    ) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            territory,
            right_type: right_type.to_string(),
            valid_from_ms,
            valid_until_ms,
        }
    }

    /// Returns `true` when `now_ms` is within `[valid_from_ms, valid_until_ms)`
    pub fn is_valid(&self, now_ms: u64) -> bool {
        if now_ms < self.valid_from_ms {
            return false;
        }
        match self.valid_until_ms {
            Some(until) => now_ms < until,
            None => true,
        }
    }

    /// Returns `true` when the right has passed its `valid_until_ms`
    pub fn is_expired(&self, now_ms: u64) -> bool {
        match self.valid_until_ms {
            Some(until) => now_ms >= until,
            None => false,
        }
    }
}

// ── TerritoryRightsManager ────────────────────────────────────────────────────

/// Registry of territory rights records
#[derive(Debug, Default)]
pub struct TerritoryRightsManager {
    /// All registered rights
    pub rights: Vec<TerritoryRight>,
}

impl TerritoryRightsManager {
    /// Create an empty manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new territory right
    pub fn add_right(&mut self, right: TerritoryRight) {
        self.rights.push(right);
    }

    /// Returns `true` if there is at least one valid, matching right for the
    /// given asset, territory, and right-type at `now_ms`
    pub fn has_right(
        &self,
        asset_id: &str,
        territory: &TerritoryCode,
        right_type: &str,
        now_ms: u64,
    ) -> bool {
        self.rights.iter().any(|r| {
            r.asset_id == asset_id
                && r.right_type == right_type
                && r.territory.includes(territory)
                && r.is_valid(now_ms)
        })
    }

    /// All rights recorded for `asset_id`, regardless of validity
    pub fn rights_for(&self, asset_id: &str) -> Vec<&TerritoryRight> {
        self.rights
            .iter()
            .filter(|r| r.asset_id == asset_id)
            .collect()
    }

    /// All rights that are currently valid at `now_ms`
    pub fn active_rights(&self, now_ms: u64) -> Vec<&TerritoryRight> {
        self.rights.iter().filter(|r| r.is_valid(now_ms)).collect()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn us() -> TerritoryCode {
        TerritoryCode::new("US")
    }

    fn gb() -> TerritoryCode {
        TerritoryCode::new("GB")
    }

    fn ww() -> TerritoryCode {
        TerritoryCode::worldwide()
    }

    fn right_us_broadcast(from: u64, until: Option<u64>) -> TerritoryRight {
        TerritoryRight::new("asset-1", us(), "broadcast", from, until)
    }

    // ── TerritoryCode ────────────────────────────────────────────────────────

    #[test]
    fn test_code_uppercased() {
        let code = TerritoryCode::new("us");
        assert_eq!(code.code, "US");
    }

    #[test]
    fn test_worldwide_sentinel() {
        assert!(ww().is_worldwide());
    }

    #[test]
    fn test_non_ww_is_not_worldwide() {
        assert!(!us().is_worldwide());
    }

    #[test]
    fn test_worldwide_includes_any_code() {
        assert!(ww().includes(&us()));
        assert!(ww().includes(&gb()));
    }

    #[test]
    fn test_code_includes_itself() {
        assert!(us().includes(&us()));
    }

    #[test]
    fn test_code_does_not_include_different_code() {
        assert!(!us().includes(&gb()));
    }

    // ── TerritoryRight ───────────────────────────────────────────────────────

    #[test]
    fn test_right_is_valid_within_window() {
        let r = right_us_broadcast(1_000, Some(5_000));
        assert!(r.is_valid(1_000));
        assert!(r.is_valid(3_000));
        assert!(!r.is_valid(5_000)); // exclusive upper bound
    }

    #[test]
    fn test_right_is_valid_before_start_false() {
        let r = right_us_broadcast(2_000, None);
        assert!(!r.is_valid(1_999));
    }

    #[test]
    fn test_right_no_expiry_is_always_valid_after_start() {
        let r = right_us_broadcast(0, None);
        assert!(r.is_valid(999_999_999));
    }

    #[test]
    fn test_right_is_expired_after_until() {
        let r = right_us_broadcast(0, Some(5_000));
        assert!(r.is_expired(5_000));
        assert!(r.is_expired(9_999));
        assert!(!r.is_expired(4_999));
    }

    #[test]
    fn test_right_no_expiry_is_never_expired() {
        let r = right_us_broadcast(0, None);
        assert!(!r.is_expired(999_999_999));
    }

    // ── TerritoryRightsManager ───────────────────────────────────────────────

    #[test]
    fn test_manager_has_right_matching() {
        let mut mgr = TerritoryRightsManager::new();
        mgr.add_right(right_us_broadcast(0, None));
        assert!(mgr.has_right("asset-1", &us(), "broadcast", 1_000));
    }

    #[test]
    fn test_manager_has_right_worldwide_covers_specific() {
        let mut mgr = TerritoryRightsManager::new();
        mgr.add_right(TerritoryRight::new("asset-1", ww(), "streaming", 0, None));
        assert!(mgr.has_right("asset-1", &gb(), "streaming", 1_000));
    }

    #[test]
    fn test_manager_has_right_wrong_type_false() {
        let mut mgr = TerritoryRightsManager::new();
        mgr.add_right(right_us_broadcast(0, None));
        assert!(!mgr.has_right("asset-1", &us(), "streaming", 1_000));
    }

    #[test]
    fn test_manager_rights_for_asset() {
        let mut mgr = TerritoryRightsManager::new();
        mgr.add_right(right_us_broadcast(0, None));
        mgr.add_right(TerritoryRight::new("asset-2", us(), "broadcast", 0, None));
        let rights = mgr.rights_for("asset-1");
        assert_eq!(rights.len(), 1);
    }

    #[test]
    fn test_manager_active_rights_at_time() {
        let mut mgr = TerritoryRightsManager::new();
        mgr.add_right(TerritoryRight::new("a1", us(), "broadcast", 0, Some(1_000)));
        mgr.add_right(TerritoryRight::new("a2", us(), "broadcast", 0, None));
        // At t=2000, the first right has expired
        let active = mgr.active_rights(2_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].asset_id, "a2");
    }
}
