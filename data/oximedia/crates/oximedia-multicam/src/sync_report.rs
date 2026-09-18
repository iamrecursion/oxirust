//! Synchronisation status reporting across multiple camera streams.

#![allow(dead_code)]

/// Describes the current synchronisation state of a camera stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// The camera is fully synchronised within tolerance.
    Synced,
    /// A sync operation is currently in progress.
    Syncing,
    /// The camera has drifted outside the acceptable window.
    Drifted,
    /// Synchronisation data has not been received / camera is offline.
    Unknown,
}

impl SyncStatus {
    /// Returns `true` only when the status is `Synced`.
    #[must_use]
    pub fn is_synced(self) -> bool {
        self == Self::Synced
    }

    /// Returns `true` when the camera is actively communicating
    /// (i.e. not `Unknown`).
    #[must_use]
    pub fn is_active(self) -> bool {
        self != Self::Unknown
    }

    /// Returns `true` when the camera has drifted.
    #[must_use]
    pub fn is_drifted(self) -> bool {
        self == Self::Drifted
    }
}

/// Sync information for a single camera in a multi-camera session.
#[derive(Debug, Clone)]
pub struct CameraSyncEntry {
    /// Unique camera identifier.
    pub camera_id: u32,
    /// Human-readable camera label.
    pub label: String,
    /// Current sync status.
    pub status: SyncStatus,
    /// Signed timing offset relative to the master clock, in milliseconds.
    pub offset_ms_raw: f64,
}

impl CameraSyncEntry {
    /// Creates a new sync entry.
    pub fn new(
        camera_id: u32,
        label: impl Into<String>,
        status: SyncStatus,
        offset_ms: f64,
    ) -> Self {
        Self {
            camera_id,
            label: label.into(),
            status,
            offset_ms_raw: offset_ms,
        }
    }

    /// Returns the signed timing offset in milliseconds.
    #[must_use]
    pub fn offset_ms(&self) -> f64 {
        self.offset_ms_raw
    }

    /// Returns the absolute value of the offset in milliseconds.
    #[must_use]
    pub fn abs_offset_ms(&self) -> f64 {
        self.offset_ms_raw.abs()
    }

    /// Returns `true` when this entry is fully synchronised.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.status.is_synced()
    }
}

/// Aggregated synchronisation report for an entire multi-camera session.
#[derive(Debug, Default)]
pub struct SyncReport {
    entries: Vec<CameraSyncEntry>,
}

impl SyncReport {
    /// Creates an empty `SyncReport`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a camera sync entry to the report.
    pub fn add_entry(&mut self, entry: CameraSyncEntry) {
        self.entries.push(entry);
    }

    /// Returns `true` if every active (non-`Unknown`) camera is `Synced`.
    #[must_use]
    pub fn all_synced(&self) -> bool {
        self.entries
            .iter()
            .filter(|e| e.status.is_active())
            .all(CameraSyncEntry::is_synced)
    }

    /// Returns the entry with the largest absolute offset, or `None` if empty.
    #[must_use]
    pub fn worst_offset_entry(&self) -> Option<&CameraSyncEntry> {
        self.entries.iter().max_by(|a, b| {
            a.abs_offset_ms()
                .partial_cmp(&b.abs_offset_ms())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the largest absolute timing offset across all cameras in ms.
    #[must_use]
    pub fn worst_offset_ms(&self) -> f64 {
        self.worst_offset_entry()
            .map_or(0.0, CameraSyncEntry::abs_offset_ms)
    }

    /// Returns the number of cameras in the report.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of cameras with `Synced` status.
    #[must_use]
    pub fn synced_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_synced()).count()
    }

    /// Returns an iterator over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &CameraSyncEntry> {
        self.entries.iter()
    }

    /// Returns all entries with `Drifted` status.
    #[must_use]
    pub fn drifted_cameras(&self) -> Vec<&CameraSyncEntry> {
        self.entries
            .iter()
            .filter(|e| e.status.is_drifted())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synced(id: u32, offset: f64) -> CameraSyncEntry {
        CameraSyncEntry::new(id, format!("Cam{id}"), SyncStatus::Synced, offset)
    }

    fn drifted(id: u32, offset: f64) -> CameraSyncEntry {
        CameraSyncEntry::new(id, format!("Cam{id}"), SyncStatus::Drifted, offset)
    }

    fn unknown(id: u32) -> CameraSyncEntry {
        CameraSyncEntry::new(id, format!("Cam{id}"), SyncStatus::Unknown, 0.0)
    }

    #[test]
    fn test_sync_status_is_synced() {
        assert!(SyncStatus::Synced.is_synced());
        assert!(!SyncStatus::Drifted.is_synced());
    }

    #[test]
    fn test_sync_status_is_active() {
        assert!(SyncStatus::Synced.is_active());
        assert!(SyncStatus::Drifted.is_active());
        assert!(!SyncStatus::Unknown.is_active());
    }

    #[test]
    fn test_sync_status_is_drifted() {
        assert!(SyncStatus::Drifted.is_drifted());
        assert!(!SyncStatus::Synced.is_drifted());
    }

    #[test]
    fn test_entry_offset_ms() {
        let e = synced(1, -2.5);
        assert_eq!(e.offset_ms(), -2.5);
        assert_eq!(e.abs_offset_ms(), 2.5);
    }

    #[test]
    fn test_entry_is_synced() {
        assert!(synced(1, 0.0).is_synced());
        assert!(!drifted(1, 5.0).is_synced());
    }

    #[test]
    fn test_report_all_synced_true() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 0.1));
        r.add_entry(synced(2, -0.2));
        assert!(r.all_synced());
    }

    #[test]
    fn test_report_all_synced_false_with_drift() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 0.0));
        r.add_entry(drifted(2, 15.0));
        assert!(!r.all_synced());
    }

    #[test]
    fn test_report_all_synced_ignores_unknown() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 0.0));
        r.add_entry(unknown(2));
        assert!(r.all_synced());
    }

    #[test]
    fn test_report_worst_offset_ms() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 1.0));
        r.add_entry(drifted(2, -8.0));
        r.add_entry(synced(3, 3.0));
        assert_eq!(r.worst_offset_ms(), 8.0);
    }

    #[test]
    fn test_report_worst_offset_empty() {
        let r = SyncReport::new();
        assert_eq!(r.worst_offset_ms(), 0.0);
    }

    #[test]
    fn test_report_synced_count() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 0.0));
        r.add_entry(drifted(2, 5.0));
        r.add_entry(synced(3, -0.5));
        assert_eq!(r.synced_count(), 2);
    }

    #[test]
    fn test_report_drifted_cameras() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 0.0));
        r.add_entry(drifted(2, 12.0));
        let drifted_cams = r.drifted_cameras();
        assert_eq!(drifted_cams.len(), 1);
        assert_eq!(drifted_cams[0].camera_id, 2);
    }

    #[test]
    fn test_report_camera_count() {
        let mut r = SyncReport::new();
        r.add_entry(synced(1, 0.0));
        r.add_entry(synced(2, 0.0));
        r.add_entry(unknown(3));
        assert_eq!(r.camera_count(), 3);
    }
}
