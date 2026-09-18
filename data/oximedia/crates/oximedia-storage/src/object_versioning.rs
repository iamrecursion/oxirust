#![allow(dead_code)]
//! Object versioning — version listing, restore, and delete-marker management.
//!
//! This module provides `ObjectVersionManager` which tracks multiple versions of
//! storage objects, supporting list, restore-to-version, soft-delete markers,
//! and hard-delete of specific versions.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// A single recorded version of a stored object.
#[derive(Debug, Clone)]
pub struct ObjectVersion {
    /// Globally unique version identifier (UUID-style string).
    pub version_id: String,
    /// Content hash (ETag) of this version.
    pub etag: String,
    /// When this version was created.
    pub last_modified: DateTime<Utc>,
    /// Object size in bytes.
    pub size: u64,
    /// Whether this is the currently active / latest version.
    pub is_latest: bool,
    /// Whether this version is a delete marker (no content).
    pub is_delete_marker: bool,
}

impl ObjectVersion {
    /// Create a regular (non-delete-marker) version.
    pub fn new(
        version_id: impl Into<String>,
        etag: impl Into<String>,
        size: u64,
        is_latest: bool,
    ) -> Self {
        Self {
            version_id: version_id.into(),
            etag: etag.into(),
            last_modified: Utc::now(),
            size,
            is_latest,
            is_delete_marker: false,
        }
    }

    /// Create a delete-marker version.
    pub fn delete_marker(version_id: impl Into<String>) -> Self {
        Self {
            version_id: version_id.into(),
            etag: String::new(),
            last_modified: Utc::now(),
            size: 0,
            is_latest: true,
            is_delete_marker: true,
        }
    }
}

/// Error type for versioning operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VersioningError {
    /// The object key was not found.
    KeyNotFound(String),
    /// The specific version_id was not found for the given key.
    VersionNotFound {
        /// Object key that was queried.
        key: String,
        /// Version identifier that could not be found.
        version_id: String,
    },
    /// Attempt to restore a delete-marker as if it were real content.
    CannotRestoreDeleteMarker(String),
}

impl std::fmt::Display for VersioningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::VersionNotFound { key, version_id } => {
                write!(f, "version '{version_id}' not found for key '{key}'")
            }
            Self::CannotRestoreDeleteMarker(vid) => {
                write!(f, "cannot restore: version '{vid}' is a delete marker")
            }
        }
    }
}

impl std::error::Error for VersioningError {}

/// Manages versions for multiple object keys in a single namespace.
///
/// Each key can have an ordered list of versions; the most recently added
/// non-marker version (or the most recent delete-marker) is `is_latest`.
#[derive(Debug, Default)]
pub struct ObjectVersionManager {
    /// key → ordered list of versions (newest last, i.e. `versions.last()` = latest).
    versions: HashMap<String, Vec<ObjectVersion>>,
    /// Monotonic counter for generating deterministic version IDs in tests.
    next_seq: u64,
}

impl ObjectVersionManager {
    /// Create a new, empty version manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate the next version ID string using an internal counter.
    fn next_version_id(&mut self) -> String {
        self.next_seq += 1;
        format!("v{:016x}", self.next_seq)
    }

    /// Add a new version for `key` with the given `etag` and `size`.
    ///
    /// The new version becomes `is_latest`; all prior versions lose that flag.
    /// Returns the generated `version_id`.
    pub fn add_version(
        &mut self,
        key: impl Into<String>,
        etag: impl Into<String>,
        size: u64,
    ) -> String {
        let key = key.into();
        let version_id = self.next_version_id();
        let entry = self.versions.entry(key).or_default();

        // Demote previous latest
        for v in entry.iter_mut() {
            v.is_latest = false;
        }

        entry.push(ObjectVersion::new(version_id.clone(), etag, size, true));
        version_id
    }

    /// Insert a delete-marker for `key`, making it the latest version.
    ///
    /// The marker represents a soft-delete: the key appears "not found" to
    /// normal GET operations but its version history is preserved.
    /// Returns the generated `version_id`.
    pub fn add_delete_marker(&mut self, key: impl Into<String>) -> String {
        let key = key.into();
        let version_id = self.next_version_id();
        let entry = self.versions.entry(key).or_default();

        for v in entry.iter_mut() {
            v.is_latest = false;
        }

        entry.push(ObjectVersion::delete_marker(version_id.clone()));
        version_id
    }

    /// List all recorded versions for `key` (newest last).
    ///
    /// Returns an empty slice if the key has never been written.
    pub fn list_versions(&self, key: &str) -> Vec<ObjectVersion> {
        self.versions.get(key).cloned().unwrap_or_default()
    }

    /// Restore `key` to a specific `version_id`.
    ///
    /// The target version is promoted to `is_latest = true`; all other versions
    /// including any subsequent delete markers become non-latest.
    ///
    /// # Errors
    ///
    /// - `KeyNotFound` if `key` has no versions.
    /// - `VersionNotFound` if `version_id` is unknown for this key.
    /// - `CannotRestoreDeleteMarker` if the target version is a delete marker.
    pub fn restore_version(&mut self, key: &str, version_id: &str) -> Result<(), VersioningError> {
        let entry = self
            .versions
            .get_mut(key)
            .ok_or_else(|| VersioningError::KeyNotFound(key.to_string()))?;

        // Find the target version
        let target_idx = entry
            .iter()
            .position(|v| v.version_id == version_id)
            .ok_or_else(|| VersioningError::VersionNotFound {
                key: key.to_string(),
                version_id: version_id.to_string(),
            })?;

        if entry[target_idx].is_delete_marker {
            return Err(VersioningError::CannotRestoreDeleteMarker(
                version_id.to_string(),
            ));
        }

        // Demote all, then promote target
        for v in entry.iter_mut() {
            v.is_latest = false;
        }
        entry[target_idx].is_latest = true;
        Ok(())
    }

    /// Permanently delete a specific version.
    ///
    /// Once deleted, the version record is removed from the history entirely.
    /// If the deleted version was `is_latest`, the immediately preceding version
    /// (if any) is promoted to `is_latest`.
    ///
    /// # Errors
    ///
    /// - `KeyNotFound` if `key` has no versions.
    /// - `VersionNotFound` if `version_id` is not present.
    pub fn delete_version(&mut self, key: &str, version_id: &str) -> Result<(), VersioningError> {
        let entry = self
            .versions
            .get_mut(key)
            .ok_or_else(|| VersioningError::KeyNotFound(key.to_string()))?;

        let target_idx = entry
            .iter()
            .position(|v| v.version_id == version_id)
            .ok_or_else(|| VersioningError::VersionNotFound {
                key: key.to_string(),
                version_id: version_id.to_string(),
            })?;

        let was_latest = entry[target_idx].is_latest;
        entry.remove(target_idx);

        // If the deleted version was the latest, promote the new last entry
        if was_latest {
            if let Some(last) = entry.last_mut() {
                last.is_latest = true;
            }
        }

        // Remove the key entirely if no versions remain
        if entry.is_empty() {
            self.versions.remove(key);
        }
        Ok(())
    }

    /// Return the currently-latest version for `key`, or `None` if unknown / deleted.
    pub fn latest_version(&self, key: &str) -> Option<&ObjectVersion> {
        self.versions
            .get(key)
            .and_then(|vs| vs.iter().find(|v| v.is_latest))
    }

    /// Whether `key` exists and its latest version is NOT a delete marker.
    pub fn is_accessible(&self, key: &str) -> bool {
        self.latest_version(key)
            .map(|v| !v.is_delete_marker)
            .unwrap_or(false)
    }

    /// Total number of version records across all keys.
    pub fn total_version_count(&self) -> usize {
        self.versions.values().map(Vec::len).sum()
    }

    /// Number of distinct keys tracked.
    pub fn key_count(&self) -> usize {
        self.versions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mgr() -> ObjectVersionManager {
        ObjectVersionManager::new()
    }

    // ── add_version ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_single_version_is_latest() {
        let mut m = make_mgr();
        let vid = m.add_version("a.mp4", "etag1", 1000);
        let vers = m.list_versions("a.mp4");
        assert_eq!(vers.len(), 1);
        assert_eq!(vers[0].version_id, vid);
        assert!(vers[0].is_latest);
        assert!(!vers[0].is_delete_marker);
    }

    #[test]
    fn test_add_multiple_versions_only_last_is_latest() {
        let mut m = make_mgr();
        let v1 = m.add_version("b.mkv", "e1", 100);
        let v2 = m.add_version("b.mkv", "e2", 200);
        let v3 = m.add_version("b.mkv", "e3", 300);
        let vers = m.list_versions("b.mkv");
        assert_eq!(vers.len(), 3);
        // v1 and v2 not latest, v3 is latest
        let ids_latest: Vec<_> = vers
            .iter()
            .filter(|v| v.is_latest)
            .map(|v| v.version_id.clone())
            .collect();
        assert_eq!(ids_latest, vec![v3.clone()]);
        let _ = (v1, v2);
    }

    #[test]
    fn test_list_versions_empty_key() {
        let m = make_mgr();
        assert!(m.list_versions("no_such_key").is_empty());
    }

    // ── add_delete_marker ─────────────────────────────────────────────────────

    #[test]
    fn test_delete_marker_makes_key_inaccessible() {
        let mut m = make_mgr();
        m.add_version("c.flac", "e1", 500);
        assert!(m.is_accessible("c.flac"));
        m.add_delete_marker("c.flac");
        assert!(!m.is_accessible("c.flac"));
    }

    #[test]
    fn test_delete_marker_is_in_version_list() {
        let mut m = make_mgr();
        m.add_version("d.wav", "e1", 1024);
        let marker_vid = m.add_delete_marker("d.wav");
        let vers = m.list_versions("d.wav");
        assert_eq!(vers.len(), 2);
        let marker = vers
            .iter()
            .find(|v| v.version_id == marker_vid)
            .expect("marker should exist");
        assert!(marker.is_delete_marker);
        assert!(marker.is_latest);
    }

    // ── restore_version ───────────────────────────────────────────────────────

    #[test]
    fn test_restore_previous_version() {
        let mut m = make_mgr();
        let v1 = m.add_version("e.mp4", "e1", 100);
        m.add_version("e.mp4", "e2", 200);
        m.restore_version("e.mp4", &v1)
            .expect("restore should succeed");
        let latest = m.latest_version("e.mp4").expect("should have latest");
        assert_eq!(latest.version_id, v1);
    }

    #[test]
    fn test_restore_after_delete_marker() {
        let mut m = make_mgr();
        let v1 = m.add_version("f.ogg", "e1", 512);
        m.add_delete_marker("f.ogg");
        assert!(!m.is_accessible("f.ogg"));
        m.restore_version("f.ogg", &v1)
            .expect("restore should succeed");
        assert!(m.is_accessible("f.ogg"));
    }

    #[test]
    fn test_restore_nonexistent_key_errors() {
        let mut m = make_mgr();
        let err = m.restore_version("no_key", "v1").unwrap_err();
        assert!(matches!(err, VersioningError::KeyNotFound(_)));
    }

    #[test]
    fn test_restore_nonexistent_version_errors() {
        let mut m = make_mgr();
        m.add_version("g.mp3", "e1", 100);
        let err = m.restore_version("g.mp3", "nonexistent").unwrap_err();
        assert!(matches!(err, VersioningError::VersionNotFound { .. }));
    }

    #[test]
    fn test_restore_delete_marker_errors() {
        let mut m = make_mgr();
        m.add_version("h.png", "e1", 1024);
        let marker_vid = m.add_delete_marker("h.png");
        let err = m.restore_version("h.png", &marker_vid).unwrap_err();
        assert!(matches!(err, VersioningError::CannotRestoreDeleteMarker(_)));
    }

    // ── delete_version ────────────────────────────────────────────────────────

    #[test]
    fn test_delete_specific_version() {
        let mut m = make_mgr();
        let v1 = m.add_version("i.webm", "e1", 100);
        let v2 = m.add_version("i.webm", "e2", 200);
        m.delete_version("i.webm", &v1)
            .expect("delete version should succeed");
        let vers = m.list_versions("i.webm");
        assert_eq!(vers.len(), 1);
        assert_eq!(vers[0].version_id, v2);
    }

    #[test]
    fn test_delete_latest_promotes_prev() {
        let mut m = make_mgr();
        let v1 = m.add_version("j.avi", "e1", 100);
        let v2 = m.add_version("j.avi", "e2", 200);
        m.delete_version("j.avi", &v2)
            .expect("delete version should succeed");
        let latest = m.latest_version("j.avi").expect("should have latest");
        assert_eq!(latest.version_id, v1);
        let _ = v2;
    }

    #[test]
    fn test_delete_only_version_removes_key() {
        let mut m = make_mgr();
        let v1 = m.add_version("k.mp4", "e1", 100);
        m.delete_version("k.mp4", &v1)
            .expect("delete version should succeed");
        assert_eq!(m.key_count(), 0);
        assert!(m.list_versions("k.mp4").is_empty());
    }

    #[test]
    fn test_delete_version_nonexistent_key_errors() {
        let mut m = make_mgr();
        let err = m.delete_version("no_key", "v1").unwrap_err();
        assert!(matches!(err, VersioningError::KeyNotFound(_)));
    }

    // ── aggregate helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_total_version_count() {
        let mut m = make_mgr();
        m.add_version("l.mp4", "e1", 100);
        m.add_version("l.mp4", "e2", 200);
        m.add_version("m.mp3", "e3", 300);
        assert_eq!(m.total_version_count(), 3);
        assert_eq!(m.key_count(), 2);
    }

    #[test]
    fn test_versioning_error_display() {
        let e1 = VersioningError::KeyNotFound("key1".to_string());
        assert!(e1.to_string().contains("key1"));
        let e2 = VersioningError::VersionNotFound {
            key: "k".to_string(),
            version_id: "v99".to_string(),
        };
        assert!(e2.to_string().contains("v99"));
        let e3 = VersioningError::CannotRestoreDeleteMarker("vdm".to_string());
        assert!(e3.to_string().contains("vdm"));
    }
}
