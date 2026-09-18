//! Synchronization rights management.
//!
//! "Sync rights" (or synchronisation rights) are needed when music is paired
//! with moving images.  This module models sync licenses, master + sync
//! pairings, and blanket licenses.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── SyncLicenseType ───────────────────────────────────────────────────────────

/// Classification of a synchronisation license.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncLicenseType {
    /// One-off license for a specific production.
    OneOff,
    /// Blanket license covering multiple productions or broadcasts.
    Blanket,
    /// Free / open license (Creative Commons or similar).
    OpenLicense,
    /// Library music package (pre-cleared).
    Library,
}

// ── SyncLicense ───────────────────────────────────────────────────────────────

/// A synchronisation license granting rights to pair a musical work with video.
#[derive(Debug, Clone)]
pub struct SyncLicense {
    /// Unique license identifier.
    pub id: u32,
    /// Title of the musical composition.
    pub composition_title: String,
    /// Name of the publisher / composition rights holder.
    pub publisher: String,
    /// License classification.
    pub license_type: SyncLicenseType,
    /// Territory covered (ISO 3166 code, or `"WORLD"` for worldwide).
    pub territory: String,
    /// Unix timestamp when the license expires.  `None` means perpetual.
    pub expires_at: Option<i64>,
    /// Licensed fee paid (in currency units).
    pub fee: f64,
    /// ISO 4217 currency code.
    pub currency: String,
}

impl SyncLicense {
    /// Create a new `SyncLicense`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        composition_title: impl Into<String>,
        publisher: impl Into<String>,
        license_type: SyncLicenseType,
        territory: impl Into<String>,
        expires_at: Option<i64>,
        fee: f64,
        currency: impl Into<String>,
    ) -> Self {
        Self {
            id,
            composition_title: composition_title.into(),
            publisher: publisher.into(),
            license_type,
            territory: territory.into(),
            expires_at,
            fee: fee.max(0.0),
            currency: currency.into(),
        }
    }

    /// Return `true` if the license is still valid at `now`.
    pub fn is_valid_at(&self, now: i64) -> bool {
        self.expires_at.is_none_or(|exp| now <= exp)
    }
}

// ── MasterLicense ─────────────────────────────────────────────────────────────

/// A "master use" license granting rights to use a specific recording.
#[derive(Debug, Clone)]
pub struct MasterLicense {
    /// Unique license identifier.
    pub id: u32,
    /// Name of the recording (track title + artist).
    pub recording_label: String,
    /// Name of the record label or master rights holder.
    pub rights_holder: String,
    /// Territory covered.
    pub territory: String,
    /// Unix timestamp when the license expires.
    pub expires_at: Option<i64>,
    /// Licensed fee.
    pub fee: f64,
    /// Currency.
    pub currency: String,
}

impl MasterLicense {
    /// Create a new `MasterLicense`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        recording_label: impl Into<String>,
        rights_holder: impl Into<String>,
        territory: impl Into<String>,
        expires_at: Option<i64>,
        fee: f64,
        currency: impl Into<String>,
    ) -> Self {
        Self {
            id,
            recording_label: recording_label.into(),
            rights_holder: rights_holder.into(),
            territory: territory.into(),
            expires_at,
            fee: fee.max(0.0),
            currency: currency.into(),
        }
    }

    /// Return `true` if the master license is still valid at `now`.
    pub fn is_valid_at(&self, now: i64) -> bool {
        self.expires_at.is_none_or(|exp| now <= exp)
    }
}

// ── SyncMasterPair ────────────────────────────────────────────────────────────

/// A linked pair of a sync (composition) license and a master (recording) license.
///
/// Using a recording in a video requires *both* licenses.
#[derive(Debug, Clone)]
pub struct SyncMasterPair {
    /// The sync (composition) license.
    pub sync_license: SyncLicense,
    /// The master (recording) license.
    pub master_license: MasterLicense,
}

impl SyncMasterPair {
    /// Create a new `SyncMasterPair`.
    pub fn new(sync_license: SyncLicense, master_license: MasterLicense) -> Self {
        Self {
            sync_license,
            master_license,
        }
    }

    /// Return `true` if *both* licenses are valid at `now`.
    pub fn is_fully_cleared(&self, now: i64) -> bool {
        self.sync_license.is_valid_at(now) && self.master_license.is_valid_at(now)
    }

    /// Combined total fee for both licenses.
    pub fn total_fee(&self) -> f64 {
        self.sync_license.fee + self.master_license.fee
    }
}

// ── BlanketLicense ────────────────────────────────────────────────────────────

/// A blanket license covering multiple tracks from a catalogue.
#[derive(Debug, Clone)]
pub struct BlanketLicense {
    /// Unique identifier.
    pub id: u32,
    /// Name of the licensing organisation (e.g. "ASCAP", "BMI").
    pub organisation: String,
    /// Territory covered.
    pub territory: String,
    /// Annual fee.
    pub annual_fee: f64,
    /// Currency.
    pub currency: String,
    /// Unix timestamp of the license start.
    pub starts_at: i64,
    /// Unix timestamp of the license end.
    pub ends_at: i64,
    /// Maximum number of productions covered.  `None` means unlimited.
    pub max_productions: Option<u32>,
}

impl BlanketLicense {
    /// Create a new `BlanketLicense`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        organisation: impl Into<String>,
        territory: impl Into<String>,
        annual_fee: f64,
        currency: impl Into<String>,
        starts_at: i64,
        ends_at: i64,
        max_productions: Option<u32>,
    ) -> Self {
        Self {
            id,
            organisation: organisation.into(),
            territory: territory.into(),
            annual_fee: annual_fee.max(0.0),
            currency: currency.into(),
            starts_at,
            ends_at,
            max_productions,
        }
    }

    /// Return `true` if the blanket license is active at `now`.
    pub fn is_active_at(&self, now: i64) -> bool {
        now >= self.starts_at && now <= self.ends_at
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> i64 {
        (self.ends_at - self.starts_at).max(0)
    }
}

// ── SyncRightsRegistry ────────────────────────────────────────────────────────

/// Central registry for all sync-rights licenses associated with a production.
#[derive(Debug, Default)]
pub struct SyncRightsRegistry {
    sync_licenses: HashMap<u32, SyncLicense>,
    master_licenses: HashMap<u32, MasterLicense>,
    pairs: Vec<SyncMasterPair>,
    blankets: HashMap<u32, BlanketLicense>,
}

impl SyncRightsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a sync license.
    pub fn add_sync_license(&mut self, license: SyncLicense) {
        self.sync_licenses.insert(license.id, license);
    }

    /// Register a master license.
    pub fn add_master_license(&mut self, license: MasterLicense) {
        self.master_licenses.insert(license.id, license);
    }

    /// Register a sync + master pair.
    pub fn add_pair(&mut self, pair: SyncMasterPair) {
        self.pairs.push(pair);
    }

    /// Register a blanket license.
    pub fn add_blanket(&mut self, blanket: BlanketLicense) {
        self.blankets.insert(blanket.id, blanket);
    }

    /// Return all pairs that are fully cleared at `now`.
    pub fn cleared_pairs(&self, now: i64) -> Vec<&SyncMasterPair> {
        self.pairs
            .iter()
            .filter(|p| p.is_fully_cleared(now))
            .collect()
    }

    /// Return `true` if any active blanket license covers `territory` at `now`.
    pub fn has_blanket_coverage(&self, territory: &str, now: i64) -> bool {
        self.blankets
            .values()
            .any(|b| b.is_active_at(now) && (b.territory == territory || b.territory == "WORLD"))
    }

    /// Total licensing cost across all sync and master licenses.
    pub fn total_licensing_cost(&self) -> f64 {
        let sync: f64 = self.sync_licenses.values().map(|l| l.fee).sum();
        let master: f64 = self.master_licenses.values().map(|l| l.fee).sum();
        sync + master
    }

    /// Number of sync licenses.
    pub fn sync_license_count(&self) -> usize {
        self.sync_licenses.len()
    }

    /// Number of master licenses.
    pub fn master_license_count(&self) -> usize {
        self.master_licenses.len()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sync(id: u32, expires: Option<i64>) -> SyncLicense {
        SyncLicense::new(
            id,
            "My Song",
            "Music Publisher",
            SyncLicenseType::OneOff,
            "US",
            expires,
            500.0,
            "USD",
        )
    }

    fn make_master(id: u32, expires: Option<i64>) -> MasterLicense {
        MasterLicense::new(
            id,
            "My Song - Album Version",
            "Record Label",
            "US",
            expires,
            300.0,
            "USD",
        )
    }

    #[test]
    fn test_sync_license_valid_perpetual() {
        let lic = make_sync(1, None);
        assert!(lic.is_valid_at(99_999_999));
    }

    #[test]
    fn test_sync_license_expired() {
        let lic = make_sync(1, Some(1000));
        assert!(!lic.is_valid_at(1001));
    }

    #[test]
    fn test_sync_license_at_expiry() {
        let lic = make_sync(1, Some(1000));
        assert!(lic.is_valid_at(1000));
    }

    #[test]
    fn test_master_license_valid_perpetual() {
        let lic = make_master(1, None);
        assert!(lic.is_valid_at(99_999_999));
    }

    #[test]
    fn test_sync_master_pair_fully_cleared() {
        let pair = SyncMasterPair::new(make_sync(1, None), make_master(2, None));
        assert!(pair.is_fully_cleared(99999));
    }

    #[test]
    fn test_sync_master_pair_not_cleared_if_sync_expired() {
        let pair = SyncMasterPair::new(make_sync(1, Some(100)), make_master(2, None));
        assert!(!pair.is_fully_cleared(200));
    }

    #[test]
    fn test_sync_master_pair_total_fee() {
        let pair = SyncMasterPair::new(make_sync(1, None), make_master(2, None));
        assert!((pair.total_fee() - 800.0).abs() < 1e-9);
    }

    #[test]
    fn test_blanket_license_active() {
        let b = BlanketLicense::new(1, "ASCAP", "US", 2000.0, "USD", 0, 10000, None);
        assert!(b.is_active_at(5000));
        assert!(!b.is_active_at(10001));
    }

    #[test]
    fn test_blanket_license_duration() {
        let b = BlanketLicense::new(1, "BMI", "WORLD", 1500.0, "USD", 0, 3600, None);
        assert_eq!(b.duration_seconds(), 3600);
    }

    #[test]
    fn test_sync_rights_registry_cleared_pairs() {
        let mut reg = SyncRightsRegistry::new();
        reg.add_pair(SyncMasterPair::new(
            make_sync(1, None),
            make_master(2, None),
        ));
        reg.add_pair(SyncMasterPair::new(
            make_sync(3, Some(50)),
            make_master(4, None),
        ));
        let cleared = reg.cleared_pairs(100);
        assert_eq!(cleared.len(), 1);
    }

    #[test]
    fn test_sync_rights_registry_blanket_coverage() {
        let mut reg = SyncRightsRegistry::new();
        reg.add_blanket(BlanketLicense::new(
            1, "PRS", "GB", 1000.0, "GBP", 0, 99999, None,
        ));
        assert!(reg.has_blanket_coverage("GB", 5000));
        assert!(!reg.has_blanket_coverage("DE", 5000));
    }

    #[test]
    fn test_sync_rights_registry_world_blanket_covers_any_territory() {
        let mut reg = SyncRightsRegistry::new();
        reg.add_blanket(BlanketLicense::new(
            1, "Global", "WORLD", 5000.0, "USD", 0, 99999, None,
        ));
        assert!(reg.has_blanket_coverage("JP", 1000));
        assert!(reg.has_blanket_coverage("BR", 1000));
    }

    #[test]
    fn test_sync_rights_registry_total_licensing_cost() {
        let mut reg = SyncRightsRegistry::new();
        reg.add_sync_license(make_sync(1, None));
        reg.add_sync_license(make_sync(2, None));
        reg.add_master_license(make_master(3, None));
        // 500 + 500 + 300 = 1300
        assert!((reg.total_licensing_cost() - 1300.0).abs() < 1e-9);
    }

    #[test]
    fn test_sync_rights_registry_counts() {
        let mut reg = SyncRightsRegistry::new();
        reg.add_sync_license(make_sync(1, None));
        reg.add_master_license(make_master(2, None));
        assert_eq!(reg.sync_license_count(), 1);
        assert_eq!(reg.master_license_count(), 1);
    }
}
