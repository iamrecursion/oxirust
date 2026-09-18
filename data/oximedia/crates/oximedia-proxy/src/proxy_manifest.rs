//! Proxy manifest — a serialisable record of all proxy variants for a project.
//!
//! Provides `ProxyManifest`, `ManifestEntry`, and `ManifestValidator` for
//! describing, querying, and validating a project's full proxy asset set.

#![allow(dead_code)]

use crate::proxy_quality::ProxyQualityTier;

/// A single entry in the proxy manifest describing one proxy variant.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    /// Absolute path to the original media.
    pub original_path: String,
    /// Absolute path to this proxy file.
    pub proxy_path: String,
    /// Quality tier of this proxy.
    pub tier: ProxyQualityTier,
    /// Bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Duration in seconds.
    pub duration_s: f64,
    /// Optional checksum (e.g. SHA-256 hex string).
    pub checksum: Option<String>,
}

impl ManifestEntry {
    /// Create a new manifest entry.
    pub fn new(
        original_path: impl Into<String>,
        proxy_path: impl Into<String>,
        tier: ProxyQualityTier,
        bitrate_kbps: u32,
        duration_s: f64,
    ) -> Self {
        Self {
            original_path: original_path.into(),
            proxy_path: proxy_path.into(),
            tier,
            bitrate_kbps,
            duration_s,
            checksum: None,
        }
    }

    /// Return `true` when all required fields have valid values.
    pub fn is_valid(&self) -> bool {
        !self.original_path.is_empty()
            && !self.proxy_path.is_empty()
            && self.bitrate_kbps > 0
            && self.duration_s > 0.0
    }

    /// Estimated file size in bytes based on bitrate and duration.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn estimated_size_bytes(&self) -> u64 {
        // bits/s * seconds / 8 = bytes
        ((self.bitrate_kbps as f64) * 1000.0 * self.duration_s / 8.0) as u64
    }
}

/// A manifest collecting all proxy entries for a project.
#[derive(Debug, Default)]
pub struct ProxyManifest {
    entries: Vec<ManifestEntry>,
}

impl ProxyManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry.  Duplicates (same original + tier) are allowed.
    pub fn add_entry(&mut self, entry: ManifestEntry) {
        self.entries.push(entry);
    }

    /// Return the total entry count.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no entries have been added.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find all entries whose `original_path` matches.
    pub fn entries_for_original(&self, original_path: &str) -> Vec<&ManifestEntry> {
        self.entries
            .iter()
            .filter(|e| e.original_path == original_path)
            .collect()
    }

    /// Find the entry with the highest `bitrate_kbps` for a given original path.
    pub fn find_best_quality(&self, original_path: &str) -> Option<&ManifestEntry> {
        self.entries_for_original(original_path)
            .into_iter()
            .max_by_key(|e| e.bitrate_kbps)
    }

    /// Find the entry matching a specific tier for a given original path.
    pub fn find_by_tier(
        &self,
        original_path: &str,
        tier: ProxyQualityTier,
    ) -> Option<&ManifestEntry> {
        self.entries
            .iter()
            .find(|e| e.original_path == original_path && e.tier == tier)
    }

    /// Return all entries for a given tier across all originals.
    pub fn entries_by_tier(&self, tier: ProxyQualityTier) -> Vec<&ManifestEntry> {
        self.entries.iter().filter(|e| e.tier == tier).collect()
    }

    /// Return a list of unique original paths in the manifest.
    pub fn unique_originals(&self) -> Vec<&str> {
        let mut seen = std::collections::HashSet::new();
        self.entries
            .iter()
            .filter_map(|e| {
                if seen.insert(e.original_path.as_str()) {
                    Some(e.original_path.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Total estimated storage in bytes across all proxy entries.
    pub fn total_estimated_size_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.estimated_size_bytes()).sum()
    }

    /// Remove all entries for a given original path. Returns count removed.
    pub fn remove_original(&mut self, original_path: &str) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.original_path != original_path);
        before - self.entries.len()
    }
}

/// Validates a `ProxyManifest` for consistency.
pub struct ManifestValidator<'a> {
    manifest: &'a ProxyManifest,
}

impl<'a> ManifestValidator<'a> {
    /// Create a validator for the given manifest.
    pub fn new(manifest: &'a ProxyManifest) -> Self {
        Self { manifest }
    }

    /// Run all validation checks.  Returns a list of error description strings.
    pub fn validate(&self) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();

        for (i, entry) in self.manifest.entries.iter().enumerate() {
            if entry.original_path.is_empty() {
                errors.push(format!("Entry {i}: original_path is empty"));
            }
            if entry.proxy_path.is_empty() {
                errors.push(format!("Entry {i}: proxy_path is empty"));
            }
            if entry.bitrate_kbps == 0 {
                errors.push(format!("Entry {i}: bitrate_kbps is zero"));
            }
            if entry.duration_s <= 0.0 {
                errors.push(format!("Entry {i}: duration_s <= 0"));
            }
        }

        // Check for duplicate (original_path, tier) combinations
        let mut seen = std::collections::HashSet::new();
        for (i, entry) in self.manifest.entries.iter().enumerate() {
            let key = (entry.original_path.clone(), entry.tier);
            if !seen.insert(key) {
                errors.push(format!(
                    "Entry {i}: duplicate (original_path, tier) combination for '{}'",
                    entry.original_path
                ));
            }
        }

        errors
    }

    /// Return `true` when the manifest has no validation errors.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft_entry(orig: &str, proxy: &str) -> ManifestEntry {
        ManifestEntry::new(orig, proxy, ProxyQualityTier::Draft, 500, 120.0)
    }

    fn review_entry(orig: &str, proxy: &str) -> ManifestEntry {
        ManifestEntry::new(orig, proxy, ProxyQualityTier::Review, 2000, 120.0)
    }

    #[test]
    fn test_entry_is_valid() {
        let e = draft_entry("/orig.mov", "/p.mp4");
        assert!(e.is_valid());
    }

    #[test]
    fn test_entry_invalid_empty_original() {
        let e = draft_entry("", "/p.mp4");
        assert!(!e.is_valid());
    }

    #[test]
    fn test_entry_invalid_zero_bitrate() {
        let mut e = draft_entry("/orig.mov", "/p.mp4");
        e.bitrate_kbps = 0;
        assert!(!e.is_valid());
    }

    #[test]
    fn test_entry_estimated_size_bytes() {
        // 500 kbps * 120 s / 8 = 7_500_000 bytes
        let e = draft_entry("/orig.mov", "/p.mp4");
        assert_eq!(e.estimated_size_bytes(), 7_500_000);
    }

    #[test]
    fn test_manifest_add_and_count() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/orig.mov", "/p.mp4"));
        assert_eq!(m.entry_count(), 1);
    }

    #[test]
    fn test_manifest_find_best_quality() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/orig.mov", "/draft.mp4"));
        m.add_entry(review_entry("/orig.mov", "/review.mp4"));
        let best = m
            .find_best_quality("/orig.mov")
            .expect("should succeed in test");
        assert_eq!(best.proxy_path, "/review.mp4");
    }

    #[test]
    fn test_manifest_find_best_quality_missing() {
        let m = ProxyManifest::new();
        assert!(m.find_best_quality("/orig.mov").is_none());
    }

    #[test]
    fn test_manifest_find_by_tier() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/orig.mov", "/draft.mp4"));
        m.add_entry(review_entry("/orig.mov", "/review.mp4"));
        let e = m
            .find_by_tier("/orig.mov", ProxyQualityTier::Draft)
            .expect("should succeed in test");
        assert_eq!(e.proxy_path, "/draft.mp4");
    }

    #[test]
    fn test_manifest_entries_by_tier() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/a.mov", "/a_d.mp4"));
        m.add_entry(draft_entry("/b.mov", "/b_d.mp4"));
        m.add_entry(review_entry("/a.mov", "/a_r.mp4"));
        let drafts = m.entries_by_tier(ProxyQualityTier::Draft);
        assert_eq!(drafts.len(), 2);
    }

    #[test]
    fn test_manifest_unique_originals() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/a.mov", "/a_d.mp4"));
        m.add_entry(review_entry("/a.mov", "/a_r.mp4"));
        m.add_entry(draft_entry("/b.mov", "/b_d.mp4"));
        assert_eq!(m.unique_originals().len(), 2);
    }

    #[test]
    fn test_manifest_remove_original() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/a.mov", "/a_d.mp4"));
        m.add_entry(review_entry("/a.mov", "/a_r.mp4"));
        let removed = m.remove_original("/a.mov");
        assert_eq!(removed, 2);
        assert!(m.is_empty());
    }

    #[test]
    fn test_manifest_total_size() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/a.mov", "/a_d.mp4")); // 7_500_000
        m.add_entry(draft_entry("/b.mov", "/b_d.mp4")); // 7_500_000
        assert_eq!(m.total_estimated_size_bytes(), 15_000_000);
    }

    #[test]
    fn test_validator_valid_manifest() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/orig.mov", "/d.mp4"));
        m.add_entry(review_entry("/orig.mov", "/r.mp4"));
        let v = ManifestValidator::new(&m);
        assert!(v.is_valid());
    }

    #[test]
    fn test_validator_detects_duplicate_tier() {
        let mut m = ProxyManifest::new();
        m.add_entry(draft_entry("/orig.mov", "/d1.mp4"));
        m.add_entry(draft_entry("/orig.mov", "/d2.mp4")); // duplicate Draft for same original
        let v = ManifestValidator::new(&m);
        assert!(!v.is_valid());
        assert!(!v.validate().is_empty());
    }

    #[test]
    fn test_validator_detects_empty_proxy_path() {
        let mut m = ProxyManifest::new();
        m.add_entry(ManifestEntry::new(
            "/orig.mov",
            "",
            ProxyQualityTier::Draft,
            500,
            60.0,
        ));
        let v = ManifestValidator::new(&m);
        assert!(!v.is_valid());
    }
}
