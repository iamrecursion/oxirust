//! Content embargo management.
//!
//! Embargoes restrict access to an asset in specific territories until a
//! defined lift date.

use std::collections::HashMap;

/// A territorial embargo zone applied to an asset.
#[derive(Debug, Clone)]
pub struct EmbargoZone {
    /// ISO 3166-1 alpha-2 territory code (e.g. "DE", "CN").
    pub territory_code: String,
    /// Unix timestamp (ms) when the embargo is lifted.
    pub lift_at_ms: u64,
    /// Human-readable reason for the embargo.
    pub reason: String,
}

impl EmbargoZone {
    /// Create a new embargo zone.
    pub fn new(
        territory_code: impl Into<String>,
        lift_at_ms: u64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            territory_code: territory_code.into(),
            lift_at_ms,
            reason: reason.into(),
        }
    }
}

/// An embargo record grouping all territorial restrictions for a single asset.
#[derive(Debug, Clone)]
pub struct EmbargoRecord {
    /// Unique identifier of the embargoed asset.
    pub asset_id: String,
    /// Per-territory embargo zones.
    pub zones: Vec<EmbargoZone>,
    /// Unix timestamp (ms) when this record was created.
    pub created_at_ms: u64,
}

impl EmbargoRecord {
    /// Create a new embargo record.
    pub fn new(asset_id: impl Into<String>, created_at_ms: u64) -> Self {
        Self {
            asset_id: asset_id.into(),
            zones: Vec::new(),
            created_at_ms,
        }
    }

    /// Add an embargo zone.
    pub fn add_zone(&mut self, zone: EmbargoZone) {
        self.zones.push(zone);
    }

    /// Returns `true` if the asset is currently embargoed in `territory` at
    /// `current_ms`.
    #[must_use]
    pub fn is_embargoed_in(&self, territory: &str, current_ms: u64) -> bool {
        self.zones
            .iter()
            .any(|z| z.territory_code == territory && current_ms < z.lift_at_ms)
    }

    /// Returns the territory codes where the embargo has been lifted as of
    /// `current_ms` (i.e., was embargoed but is now available).
    #[must_use]
    pub fn lift_all_before(&self, current_ms: u64) -> Vec<String> {
        self.zones
            .iter()
            .filter(|z| current_ms >= z.lift_at_ms)
            .map(|z| z.territory_code.clone())
            .collect()
    }
}

/// Alert describing an embargo that is about to be lifted.
#[derive(Debug, Clone)]
pub struct EmbargoAlert {
    /// Asset identifier.
    pub asset_id: String,
    /// Territory code.
    pub territory: String,
    /// Unix timestamp (ms) when the embargo lifts.
    pub lift_at_ms: u64,
    /// Number of days until the embargo is lifted (may be negative if past).
    pub days_until_lift: i64,
}

/// Central manager for all asset embargoes.
#[derive(Debug, Default)]
pub struct EmbargoManager {
    /// Keyed by asset_id.
    records: HashMap<String, EmbargoRecord>,
}

impl EmbargoManager {
    /// Create a new empty embargo manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or merge an embargo record.  If a record already exists for the
    /// asset, the new zones are appended.
    pub fn add_embargo(&mut self, record: EmbargoRecord) {
        let entry = self
            .records
            .entry(record.asset_id.clone())
            .or_insert_with(|| EmbargoRecord::new(record.asset_id.clone(), record.created_at_ms));
        for zone in record.zones {
            entry.zones.push(zone);
        }
    }

    /// Returns `true` if `asset_id` is currently accessible in `territory` at
    /// `current_ms` (i.e., not embargoed).
    #[must_use]
    pub fn check_access(&self, asset_id: &str, territory: &str, current_ms: u64) -> bool {
        match self.records.get(asset_id) {
            None => true, // no embargo record → access granted
            Some(rec) => !rec.is_embargoed_in(territory, current_ms),
        }
    }

    /// Return `(asset_id, territory)` pairs whose embargo will be lifted
    /// within the next `within_ms` milliseconds from `current_ms`.
    ///
    /// Note: this method needs a reference time; we accept it implicitly via
    /// the `within_ms` window compared against stored lift times.
    #[must_use]
    pub fn embargoes_lifting_soon(&self, current_ms: u64, within_ms: u64) -> Vec<(String, String)> {
        let window_end = current_ms + within_ms;
        let mut result = Vec::new();
        for rec in self.records.values() {
            for zone in &rec.zones {
                if zone.lift_at_ms > current_ms && zone.lift_at_ms <= window_end {
                    result.push((rec.asset_id.clone(), zone.territory_code.clone()));
                }
            }
        }
        result
    }

    /// Build alerts for embargoes lifting within `within_ms` of `current_ms`.
    #[must_use]
    pub fn build_alerts(&self, current_ms: u64, within_ms: u64) -> Vec<EmbargoAlert> {
        let ms_per_day = 86_400_000_u64;
        let window_end = current_ms + within_ms;
        let mut alerts = Vec::new();

        for rec in self.records.values() {
            for zone in &rec.zones {
                if zone.lift_at_ms > current_ms && zone.lift_at_ms <= window_end {
                    let diff_ms = zone.lift_at_ms.saturating_sub(current_ms) as i64;
                    let days_until_lift = diff_ms / ms_per_day as i64;
                    alerts.push(EmbargoAlert {
                        asset_id: rec.asset_id.clone(),
                        territory: zone.territory_code.clone(),
                        lift_at_ms: zone.lift_at_ms,
                        days_until_lift,
                    });
                }
            }
        }
        alerts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000_000_u64;
    const PAST: u64 = 1_600_000_000_000_u64;
    const FUTURE: u64 = 1_800_000_000_000_u64;

    fn record_with_zone(asset: &str, territory: &str, lift_at: u64) -> EmbargoRecord {
        let mut r = EmbargoRecord::new(asset, NOW);
        r.add_zone(EmbargoZone::new(territory, lift_at, "test reason"));
        r
    }

    // --- EmbargoZone ---

    #[test]
    fn test_embargo_zone_new() {
        let z = EmbargoZone::new("DE", FUTURE, "regulatory");
        assert_eq!(z.territory_code, "DE");
        assert_eq!(z.lift_at_ms, FUTURE);
        assert_eq!(z.reason, "regulatory");
    }

    // --- EmbargoRecord ---

    #[test]
    fn test_embargo_record_is_embargoed_in_active() {
        let r = record_with_zone("asset-1", "DE", FUTURE);
        assert!(r.is_embargoed_in("DE", NOW));
    }

    #[test]
    fn test_embargo_record_is_embargoed_in_lifted() {
        let r = record_with_zone("asset-1", "DE", PAST);
        assert!(!r.is_embargoed_in("DE", NOW));
    }

    #[test]
    fn test_embargo_record_is_embargoed_in_different_territory() {
        let r = record_with_zone("asset-1", "DE", FUTURE);
        assert!(!r.is_embargoed_in("US", NOW));
    }

    #[test]
    fn test_embargo_record_lift_all_before() {
        let mut r = EmbargoRecord::new("asset-2", NOW);
        r.add_zone(EmbargoZone::new("DE", PAST, "old embargo"));
        r.add_zone(EmbargoZone::new("CN", FUTURE, "active embargo"));
        let lifted = r.lift_all_before(NOW);
        assert_eq!(lifted, vec!["DE"]);
    }

    #[test]
    fn test_embargo_record_lift_all_none() {
        let r = record_with_zone("asset-3", "FR", FUTURE);
        let lifted = r.lift_all_before(NOW);
        assert!(lifted.is_empty());
    }

    // --- EmbargoManager ---

    #[test]
    fn test_embargo_manager_check_access_no_record() {
        let mgr = EmbargoManager::new();
        assert!(mgr.check_access("unknown", "US", NOW));
    }

    #[test]
    fn test_embargo_manager_check_access_blocked() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(record_with_zone("vid-1", "DE", FUTURE));
        assert!(!mgr.check_access("vid-1", "DE", NOW));
    }

    #[test]
    fn test_embargo_manager_check_access_allowed_after_lift() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(record_with_zone("vid-2", "DE", PAST));
        assert!(mgr.check_access("vid-2", "DE", NOW));
    }

    #[test]
    fn test_embargo_manager_embargoes_lifting_soon() {
        let soon = NOW + 3 * 86_400_000; // 3 days from now
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(record_with_zone("vid-3", "FR", soon));
        mgr.add_embargo(record_with_zone("vid-4", "IT", FUTURE)); // too far

        let within_week = 7 * 86_400_000_u64;
        let lifting = mgr.embargoes_lifting_soon(NOW, within_week);
        assert_eq!(lifting.len(), 1);
        assert_eq!(lifting[0].0, "vid-3");
        assert_eq!(lifting[0].1, "FR");
    }

    #[test]
    fn test_embargo_manager_add_embargo_merges() {
        let mut mgr = EmbargoManager::new();
        mgr.add_embargo(record_with_zone("vid-5", "DE", FUTURE));
        mgr.add_embargo(record_with_zone("vid-5", "FR", FUTURE));
        // Both zones should exist
        assert!(!mgr.check_access("vid-5", "DE", NOW));
        assert!(!mgr.check_access("vid-5", "FR", NOW));
    }
}
