//! Equipment inventory tracking for broadcast automation.
//!
//! Tracks device serial numbers, firmware versions, maintenance schedules,
//! and overall asset lifecycle for broadcast equipment.
//!
//! # Design
//!
//! [`EquipmentInventory`] is the central registry.  Each piece of equipment
//! is described by an [`EquipmentRecord`] that includes:
//!
//! - Identity: serial number, model, manufacturer, asset tag.
//! - Firmware/software version.
//! - Installation and last-seen timestamps.
//! - Maintenance schedule: last service date and recommended service interval.
//! - Current operational status.
//!
//! The registry exposes queries for:
//! - Equipment due for service.
//! - Equipment running outdated firmware.
//! - Equipment in a specific status (operational / fault / decommissioned).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Equipment status
// ─────────────────────────────────────────────────────────────────────────────

/// Operational status of a piece of equipment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquipmentStatus {
    /// Fully operational.
    Operational,
    /// Operational with minor issues; maintenance scheduled.
    Degraded,
    /// Equipment has faulted; currently off-air or in standby.
    Faulted,
    /// Equipment is undergoing planned maintenance.
    Maintenance,
    /// Equipment has been decommissioned and should not be used.
    Decommissioned,
}

impl EquipmentStatus {
    /// Returns `true` if the equipment is available for use.
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Operational | Self::Degraded)
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Operational => "operational",
            Self::Degraded => "degraded",
            Self::Faulted => "faulted",
            Self::Maintenance => "maintenance",
            Self::Decommissioned => "decommissioned",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Firmware version
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed semantic-version-like firmware identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirmwareVersion {
    /// Raw version string as reported by the device (e.g. `"3.2.1-build42"`).
    pub version_string: String,
    /// Major version component (parsed from the string if possible).
    pub major: u32,
    /// Minor version component.
    pub minor: u32,
    /// Patch/build component.
    pub patch: u32,
}

impl FirmwareVersion {
    /// Create a firmware version from a `"MAJOR.MINOR.PATCH"` string.
    /// Non-conforming strings are stored verbatim with all numeric parts as 0.
    pub fn parse(version_string: impl Into<String>) -> Self {
        let s = version_string.into();
        let mut parts = s.splitn(4, '.').map(|p| p.parse::<u32>().unwrap_or(0));
        let major = parts.next().unwrap_or(0);
        let minor = parts.next().unwrap_or(0);
        let patch = parts.next().unwrap_or(0);
        Self {
            version_string: s,
            major,
            minor,
            patch,
        }
    }

    /// Returns `true` if `self` is older than `other` (numeric comparison).
    pub fn is_older_than(&self, other: &Self) -> bool {
        (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    }
}

impl std::fmt::Display for FirmwareVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.version_string)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Equipment record
// ─────────────────────────────────────────────────────────────────────────────

/// Complete record for a single piece of broadcast equipment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquipmentRecord {
    // ── Identity ──────────────────────────────────────────────────────────────
    /// Unique asset identifier (inventory tag / barcode).
    pub asset_id: String,
    /// Device serial number.
    pub serial_number: String,
    /// Manufacturer name.
    pub manufacturer: String,
    /// Model/product name.
    pub model: String,
    /// Human-readable description.
    pub description: Option<String>,

    // ── Location ──────────────────────────────────────────────────────────────
    /// Physical location (rack, room, site).
    pub location: String,
    /// Channel or system the equipment is assigned to.
    pub channel_assignment: Option<String>,

    // ── Firmware ──────────────────────────────────────────────────────────────
    /// Currently installed firmware version.
    pub firmware_version: FirmwareVersion,
    /// Recommended minimum firmware version (if known).
    pub required_firmware: Option<FirmwareVersion>,

    // ── Maintenance ───────────────────────────────────────────────────────────
    /// Unix epoch day of initial installation (days since 1970-01-01).
    pub installation_day: u32,
    /// Unix epoch day of last preventive maintenance service.
    pub last_service_day: Option<u32>,
    /// How many days between recommended services (e.g. 365 = annual).
    pub service_interval_days: u32,

    // ── Status ────────────────────────────────────────────────────────────────
    /// Current operational status.
    pub status: EquipmentStatus,
    /// Unix epoch day of last status check.
    pub last_seen_day: Option<u32>,
    /// Free-form notes (latest fault description, etc.).
    pub notes: Option<String>,
}

impl EquipmentRecord {
    /// Create a new equipment record.
    pub fn new(
        asset_id: impl Into<String>,
        serial_number: impl Into<String>,
        manufacturer: impl Into<String>,
        model: impl Into<String>,
        location: impl Into<String>,
        firmware_version: FirmwareVersion,
        installation_day: u32,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            serial_number: serial_number.into(),
            manufacturer: manufacturer.into(),
            model: model.into(),
            description: None,
            location: location.into(),
            channel_assignment: None,
            firmware_version,
            required_firmware: None,
            installation_day,
            last_service_day: None,
            service_interval_days: 365,
            status: EquipmentStatus::Operational,
            last_seen_day: None,
            notes: None,
        }
    }

    /// Returns `true` if the equipment is due for service based on
    /// `current_day` and the configured `service_interval_days`.
    pub fn is_due_for_service(&self, current_day: u32) -> bool {
        let base_day = self.last_service_day.unwrap_or(self.installation_day);
        current_day.saturating_sub(base_day) >= self.service_interval_days
    }

    /// Returns `true` if the installed firmware is older than the required
    /// minimum version.
    pub fn has_outdated_firmware(&self) -> bool {
        match &self.required_firmware {
            Some(required) => self.firmware_version.is_older_than(required),
            None => false,
        }
    }

    /// Record a maintenance service, resetting the service counter.
    pub fn record_service(&mut self, day: u32, notes: Option<String>) {
        self.last_service_day = Some(day);
        if let Some(n) = notes {
            self.notes = Some(n);
        }
        if self.status == EquipmentStatus::Maintenance {
            self.status = EquipmentStatus::Operational;
        }
    }

    /// Update the firmware version.
    pub fn update_firmware(&mut self, version: FirmwareVersion) {
        info!(
            "Firmware updated for asset '{}': {} → {}",
            self.asset_id, self.firmware_version, version
        );
        self.firmware_version = version;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inventory
// ─────────────────────────────────────────────────────────────────────────────

/// Central equipment inventory registry.
#[derive(Debug, Default)]
pub struct EquipmentInventory {
    records: HashMap<String, EquipmentRecord>,
}

impl EquipmentInventory {
    /// Create an empty inventory.
    pub fn new() -> Self {
        Self::default()
    }

    // ── CRUD ──────────────────────────────────────────────────────────────────

    /// Add or replace an equipment record.
    pub fn upsert(&mut self, record: EquipmentRecord) {
        info!(
            "Equipment upserted: '{}' ({} {} s/n {})",
            record.asset_id, record.manufacturer, record.model, record.serial_number
        );
        self.records.insert(record.asset_id.clone(), record);
    }

    /// Remove a record by asset ID.  Returns the removed record, if any.
    pub fn remove(&mut self, asset_id: &str) -> Option<EquipmentRecord> {
        self.records.remove(asset_id)
    }

    /// Get an immutable reference to a record.
    pub fn get(&self, asset_id: &str) -> Option<&EquipmentRecord> {
        self.records.get(asset_id)
    }

    /// Get a mutable reference to a record.
    pub fn get_mut(&mut self, asset_id: &str) -> Option<&mut EquipmentRecord> {
        self.records.get_mut(asset_id)
    }

    /// Total number of records.
    pub fn count(&self) -> usize {
        self.records.len()
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return all equipment due for preventive maintenance as of `current_day`.
    pub fn due_for_service(&self, current_day: u32) -> Vec<&EquipmentRecord> {
        self.records
            .values()
            .filter(|r| r.is_due_for_service(current_day))
            .collect()
    }

    /// Return all equipment with outdated firmware.
    pub fn with_outdated_firmware(&self) -> Vec<&EquipmentRecord> {
        self.records
            .values()
            .filter(|r| r.has_outdated_firmware())
            .collect()
    }

    /// Return all equipment in a given status.
    pub fn by_status(&self, status: EquipmentStatus) -> Vec<&EquipmentRecord> {
        self.records
            .values()
            .filter(|r| r.status == status)
            .collect()
    }

    /// Return all equipment assigned to a specific channel.
    pub fn by_channel(&self, channel: &str) -> Vec<&EquipmentRecord> {
        self.records
            .values()
            .filter(|r| r.channel_assignment.as_deref() == Some(channel))
            .collect()
    }

    /// Return equipment at a specific location.
    pub fn by_location(&self, location: &str) -> Vec<&EquipmentRecord> {
        self.records
            .values()
            .filter(|r| r.location == location)
            .collect()
    }

    /// Update status for a specific asset.  Returns `false` if not found.
    pub fn set_status(&mut self, asset_id: &str, status: EquipmentStatus) -> bool {
        if let Some(record) = self.records.get_mut(asset_id) {
            if record.status != status {
                warn!(
                    "Equipment '{}' status: {} → {}",
                    asset_id,
                    record.status.label(),
                    status.label()
                );
                record.status = status;
            }
            true
        } else {
            false
        }
    }

    /// Generate a summary of the inventory by status.
    pub fn status_summary(&self) -> HashMap<String, usize> {
        let mut summary: HashMap<String, usize> = HashMap::new();
        for record in self.records.values() {
            *summary
                .entry(record.status.label().to_string())
                .or_insert(0) += 1;
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, fw: &str) -> EquipmentRecord {
        EquipmentRecord::new(
            id,
            format!("SN-{id}"),
            "ACME",
            "BroadcastDevice",
            "MCR-Rack1",
            FirmwareVersion::parse(fw),
            1000,
        )
    }

    #[test]
    fn test_upsert_and_count() {
        let mut inv = EquipmentInventory::new();
        inv.upsert(make_record("dev1", "2.1.0"));
        inv.upsert(make_record("dev2", "1.0.0"));
        assert_eq!(inv.count(), 2);
    }

    #[test]
    fn test_get_returns_record() {
        let mut inv = EquipmentInventory::new();
        inv.upsert(make_record("x", "1.0.0"));
        assert!(inv.get("x").is_some());
        assert!(inv.get("missing").is_none());
    }

    #[test]
    fn test_due_for_service_detects_overdue() {
        let mut inv = EquipmentInventory::new();
        let mut r = make_record("svc", "1.0.0");
        r.service_interval_days = 30;
        r.installation_day = 0;
        inv.upsert(r);
        // 35 days later → due
        assert_eq!(inv.due_for_service(35).len(), 1);
        // 20 days later → not due
        assert_eq!(inv.due_for_service(20).len(), 0);
    }

    #[test]
    fn test_outdated_firmware() {
        let mut inv = EquipmentInventory::new();
        let mut r = make_record("fw_old", "1.0.0");
        r.required_firmware = Some(FirmwareVersion::parse("2.0.0"));
        inv.upsert(r);
        assert_eq!(inv.with_outdated_firmware().len(), 1);
    }

    #[test]
    fn test_firmware_up_to_date() {
        let mut inv = EquipmentInventory::new();
        let mut r = make_record("fw_ok", "3.1.0");
        r.required_firmware = Some(FirmwareVersion::parse("2.0.0"));
        inv.upsert(r);
        assert_eq!(inv.with_outdated_firmware().len(), 0);
    }

    #[test]
    fn test_by_status() {
        let mut inv = EquipmentInventory::new();
        inv.upsert(make_record("op", "1.0.0"));
        let mut r2 = make_record("fault", "1.0.0");
        r2.status = EquipmentStatus::Faulted;
        inv.upsert(r2);

        assert_eq!(inv.by_status(EquipmentStatus::Operational).len(), 1);
        assert_eq!(inv.by_status(EquipmentStatus::Faulted).len(), 1);
    }

    #[test]
    fn test_set_status() {
        let mut inv = EquipmentInventory::new();
        inv.upsert(make_record("ch", "1.0.0"));
        assert!(inv.set_status("ch", EquipmentStatus::Faulted));
        assert_eq!(
            inv.get("ch").expect("should exist").status,
            EquipmentStatus::Faulted
        );
        assert!(!inv.set_status("nonexistent", EquipmentStatus::Faulted));
    }

    #[test]
    fn test_by_channel() {
        let mut inv = EquipmentInventory::new();
        let mut r = make_record("enc1", "1.0.0");
        r.channel_assignment = Some("CH1".to_string());
        inv.upsert(r);
        inv.upsert(make_record("enc2", "1.0.0")); // no channel

        assert_eq!(inv.by_channel("CH1").len(), 1);
        assert_eq!(inv.by_channel("CH2").len(), 0);
    }

    #[test]
    fn test_record_service_resets_counter() {
        let mut r = make_record("srv", "1.0.0");
        r.service_interval_days = 30;
        r.installation_day = 0;
        // Due after 35 days
        assert!(r.is_due_for_service(35));
        // Service performed at day 35
        r.record_service(35, None);
        // Now not due until day 65
        assert!(!r.is_due_for_service(60));
        assert!(r.is_due_for_service(65));
    }

    #[test]
    fn test_firmware_version_ordering() {
        let old = FirmwareVersion::parse("1.2.3");
        let new = FirmwareVersion::parse("2.0.0");
        assert!(old.is_older_than(&new));
        assert!(!new.is_older_than(&old));
        assert!(!old.is_older_than(&old));
    }

    #[test]
    fn test_status_summary() {
        let mut inv = EquipmentInventory::new();
        inv.upsert(make_record("a", "1.0.0"));
        inv.upsert(make_record("b", "1.0.0"));
        let mut r = make_record("c", "1.0.0");
        r.status = EquipmentStatus::Faulted;
        inv.upsert(r);

        let summary = inv.status_summary();
        assert_eq!(summary.get("operational").copied().unwrap_or(0), 2);
        assert_eq!(summary.get("faulted").copied().unwrap_or(0), 1);
    }
}
