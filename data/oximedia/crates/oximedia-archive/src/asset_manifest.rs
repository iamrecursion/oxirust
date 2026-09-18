#![allow(dead_code)]
//! Asset manifest: per-asset entries, manifest collection, and validation.

use std::collections::HashMap;

/// A single entry in an asset manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    /// Unique asset identifier (e.g. UUID or content hash).
    pub id: String,
    /// Relative or absolute path to the asset.
    pub path: String,
    /// Expected size in bytes.
    pub size_bytes: u64,
    /// Expected SHA-256 checksum (hex-encoded), if available.
    pub checksum_sha256: Option<String>,
    /// MIME type string.
    pub mime_type: String,
}

impl ManifestEntry {
    /// Creates a new `ManifestEntry`.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        path: impl Into<String>,
        size_bytes: u64,
        checksum_sha256: Option<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            size_bytes,
            checksum_sha256,
            mime_type: mime_type.into(),
        }
    }

    /// Returns `true` if the entry has a non-empty ID, a non-empty path,
    /// a non-zero size, and a non-empty MIME type.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.path.is_empty()
            && self.size_bytes > 0
            && !self.mime_type.is_empty()
    }

    /// Returns `true` if a SHA-256 checksum is present.
    #[must_use]
    pub fn has_checksum(&self) -> bool {
        self.checksum_sha256.is_some()
    }
}

/// A manifest containing multiple `ManifestEntry` records.
#[derive(Debug, Default, Clone)]
pub struct AssetManifest {
    entries: HashMap<String, ManifestEntry>,
}

impl AssetManifest {
    /// Creates an empty `AssetManifest`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Adds or replaces a `ManifestEntry`, keyed by its `id`.
    pub fn add(&mut self, entry: ManifestEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Removes the entry with the given ID, if present.
    pub fn remove(&mut self, id: &str) -> Option<ManifestEntry> {
        self.entries.remove(id)
    }

    /// Finds a `ManifestEntry` by its `id`.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&ManifestEntry> {
        self.entries.get(id)
    }

    /// Returns the total size in bytes of all entries.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.size_bytes).sum()
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the manifest contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &ManifestEntry> {
        self.entries.values()
    }

    /// Returns all entries missing a SHA-256 checksum.
    #[must_use]
    pub fn entries_without_checksum(&self) -> Vec<&ManifestEntry> {
        self.entries
            .values()
            .filter(|e| !e.has_checksum())
            .collect()
    }
}

/// Validates an `AssetManifest` and collects any problems.
#[derive(Debug, Clone, Default)]
pub struct ManifestValidator {
    /// If `true`, entries without a checksum are flagged as errors.
    pub require_checksum: bool,
}

impl ManifestValidator {
    /// Creates a new `ManifestValidator`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            require_checksum: false,
        }
    }

    /// Enables checksum requirement validation.
    #[must_use]
    pub fn requiring_checksum(mut self) -> Self {
        self.require_checksum = true;
        self
    }

    /// Validates every entry in the manifest.
    ///
    /// Returns `Ok(())` when the manifest is clean, or `Err(Vec<String>)` with
    /// one human-readable error message per invalid entry.
    pub fn validate(&self, manifest: &AssetManifest) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        for entry in manifest.iter() {
            if !entry.is_valid() {
                errors.push(format!(
                    "Entry '{}' is invalid (empty id/path/mime or zero size)",
                    entry.id
                ));
            }
            if self.require_checksum && entry.checksum_sha256.is_none() {
                errors.push(format!(
                    "Entry '{}' is missing a SHA-256 checksum",
                    entry.id
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_entry(id: &str) -> ManifestEntry {
        ManifestEntry::new(
            id,
            "/archive/clip.mxf",
            1024,
            Some("abc123".into()),
            "video/mxf",
        )
    }

    fn entry_no_checksum(id: &str) -> ManifestEntry {
        ManifestEntry::new(id, "/archive/clip.mxf", 1024, None, "video/mxf")
    }

    // --- ManifestEntry ---

    #[test]
    fn valid_entry_passes() {
        assert!(good_entry("a1").is_valid());
    }

    #[test]
    fn empty_id_invalid() {
        let e = ManifestEntry::new("", "/path", 1, None, "video/mp4");
        assert!(!e.is_valid());
    }

    #[test]
    fn zero_size_invalid() {
        let e = ManifestEntry::new("id1", "/path", 0, None, "video/mp4");
        assert!(!e.is_valid());
    }

    #[test]
    fn empty_mime_invalid() {
        let e = ManifestEntry::new("id1", "/path", 100, None, "");
        assert!(!e.is_valid());
    }

    #[test]
    fn has_checksum_true_when_present() {
        assert!(good_entry("x").has_checksum());
    }

    #[test]
    fn has_checksum_false_when_absent() {
        assert!(!entry_no_checksum("x").has_checksum());
    }

    // --- AssetManifest ---

    #[test]
    fn empty_manifest_zero_size() {
        let m = AssetManifest::new();
        assert_eq!(m.total_size_bytes(), 0);
    }

    #[test]
    fn total_size_bytes_sums_entries() {
        let mut m = AssetManifest::new();
        m.add(ManifestEntry::new("a", "/a", 500, None, "video/mp4"));
        m.add(ManifestEntry::new("b", "/b", 300, None, "video/mp4"));
        assert_eq!(m.total_size_bytes(), 800);
    }

    #[test]
    fn find_by_id_returns_entry() {
        let mut m = AssetManifest::new();
        m.add(good_entry("abc"));
        assert!(m.find_by_id("abc").is_some());
    }

    #[test]
    fn find_by_id_missing_returns_none() {
        let m = AssetManifest::new();
        assert!(m.find_by_id("nope").is_none());
    }

    #[test]
    fn remove_entry_decrements_len() {
        let mut m = AssetManifest::new();
        m.add(good_entry("r1"));
        m.remove("r1");
        assert!(m.is_empty());
    }

    #[test]
    fn entries_without_checksum_found() {
        let mut m = AssetManifest::new();
        m.add(good_entry("with"));
        m.add(entry_no_checksum("without"));
        assert_eq!(m.entries_without_checksum().len(), 1);
    }

    // --- ManifestValidator ---

    #[test]
    fn valid_manifest_passes_validation() {
        let mut m = AssetManifest::new();
        m.add(good_entry("v1"));
        let validator = ManifestValidator::new();
        assert!(validator.validate(&m).is_ok());
    }

    #[test]
    fn invalid_entry_causes_error() {
        let mut m = AssetManifest::new();
        m.add(ManifestEntry::new("", "/p", 0, None, ""));
        let validator = ManifestValidator::new();
        assert!(validator.validate(&m).is_err());
    }

    #[test]
    fn require_checksum_flags_missing() {
        let mut m = AssetManifest::new();
        m.add(entry_no_checksum("nc1"));
        let validator = ManifestValidator::new().requiring_checksum();
        let result = validator.validate(&m);
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("SHA-256"));
    }

    #[test]
    fn empty_manifest_validates_ok() {
        let m = AssetManifest::new();
        let validator = ManifestValidator::new().requiring_checksum();
        assert!(validator.validate(&m).is_ok());
    }
}
