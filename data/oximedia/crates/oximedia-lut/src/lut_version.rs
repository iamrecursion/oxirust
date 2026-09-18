#![allow(dead_code)]
//! LUT version management — history, compatibility checks, and rollback.

/// A semantic-style version for a LUT file or preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LutVersion {
    /// Major version — incompatible API/format changes.
    pub major: u32,
    /// Minor version — backwards-compatible additions.
    pub minor: u32,
    /// Patch version — bug fixes and metadata tweaks.
    pub patch: u32,
}

impl LutVersion {
    /// Create a new `LutVersion`.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `true` when this version can be loaded by a system expecting `required`.
    ///
    /// Compatibility rule: same major, and this minor/patch >= required minor/patch.
    #[must_use]
    pub fn is_compatible_with(self, required: LutVersion) -> bool {
        self.major == required.major && self >= required
    }

    /// Format as `"major.minor.patch"` string.
    #[must_use]
    pub fn as_string(self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for LutVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A named tag attached to a specific LUT version.
#[derive(Debug, Clone)]
pub struct LutVersionTag {
    /// Version this tag is attached to.
    pub version: LutVersion,
    /// Human-readable label (e.g. "release", "golden-master").
    pub tag: String,
}

impl LutVersionTag {
    /// Create a new tag for a version.
    #[must_use]
    pub fn new(version: LutVersion, tag: impl Into<String>) -> Self {
        Self {
            version,
            tag: tag.into(),
        }
    }

    /// Returns the label string for this tag.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.tag
    }
}

/// A history entry recording a specific LUT version along with a change description.
#[derive(Debug, Clone)]
pub struct LutVersionEntry {
    /// Version of this entry.
    pub version: LutVersion,
    /// Short description of changes in this version.
    pub description: String,
    /// Optional tag for this entry.
    pub tag: Option<LutVersionTag>,
}

/// Ordered history of LUT versions, newest last.
#[derive(Debug, Clone, Default)]
pub struct LutVersionHistory {
    entries: Vec<LutVersionEntry>,
}

impl LutVersionHistory {
    /// Create an empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new version entry to the history.
    ///
    /// The new version must be strictly greater than the current latest, or the history
    /// must be empty.
    pub fn add(&mut self, version: LutVersion, description: impl Into<String>) -> bool {
        if let Some(latest) = self.entries.last() {
            if version <= latest.version {
                return false;
            }
        }
        self.entries.push(LutVersionEntry {
            version,
            description: description.into(),
            tag: None,
        });
        true
    }

    /// Attach a tag to the most recent entry.
    pub fn tag_latest(&mut self, tag: impl Into<String>) -> bool {
        if let Some(entry) = self.entries.last_mut() {
            let t = LutVersionTag::new(entry.version, tag);
            entry.tag = Some(t);
            true
        } else {
            false
        }
    }

    /// Returns the latest version, or `None` if history is empty.
    #[must_use]
    pub fn latest(&self) -> Option<LutVersion> {
        self.entries.last().map(|e| e.version)
    }

    /// Return the version just before the latest (one rollback step), or `None`.
    #[must_use]
    pub fn rollback(&self) -> Option<LutVersion> {
        let n = self.entries.len();
        if n < 2 {
            return None;
        }
        Some(self.entries[n - 2].version)
    }

    /// Number of entries in the history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &LutVersionEntry> {
        self.entries.iter()
    }

    /// Find the entry for a specific version, if it exists.
    #[must_use]
    pub fn find(&self, version: LutVersion) -> Option<&LutVersionEntry> {
        self.entries.iter().find(|e| e.version == version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> LutVersion {
        LutVersion::new(major, minor, patch)
    }

    #[test]
    fn test_version_display() {
        assert_eq!(v(1, 2, 3).as_string(), "1.2.3");
    }

    #[test]
    fn test_version_ordering() {
        assert!(v(1, 0, 1) > v(1, 0, 0));
        assert!(v(2, 0, 0) > v(1, 9, 9));
    }

    #[test]
    fn test_compatible_same_version() {
        assert!(v(1, 2, 3).is_compatible_with(v(1, 2, 3)));
    }

    #[test]
    fn test_compatible_newer_minor() {
        assert!(v(1, 3, 0).is_compatible_with(v(1, 2, 0)));
    }

    #[test]
    fn test_incompatible_older_minor() {
        assert!(!v(1, 1, 0).is_compatible_with(v(1, 2, 0)));
    }

    #[test]
    fn test_incompatible_different_major() {
        assert!(!v(2, 0, 0).is_compatible_with(v(1, 0, 0)));
    }

    #[test]
    fn test_tag_label() {
        let tag = LutVersionTag::new(v(1, 0, 0), "golden-master");
        assert_eq!(tag.label(), "golden-master");
    }

    #[test]
    fn test_history_empty() {
        let h = LutVersionHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.latest(), None);
        assert_eq!(h.rollback(), None);
    }

    #[test]
    fn test_history_add_and_latest() {
        let mut h = LutVersionHistory::new();
        assert!(h.add(v(1, 0, 0), "initial"));
        assert_eq!(h.latest(), Some(v(1, 0, 0)));
    }

    #[test]
    fn test_history_rejects_older_version() {
        let mut h = LutVersionHistory::new();
        h.add(v(1, 1, 0), "v1.1");
        assert!(!h.add(v(1, 0, 0), "old"));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn test_history_rollback() {
        let mut h = LutVersionHistory::new();
        h.add(v(1, 0, 0), "first");
        h.add(v(1, 1, 0), "second");
        assert_eq!(h.rollback(), Some(v(1, 0, 0)));
    }

    #[test]
    fn test_history_find_entry() {
        let mut h = LutVersionHistory::new();
        h.add(v(1, 0, 0), "first");
        h.add(v(1, 1, 0), "second");
        let entry = h.find(v(1, 0, 0));
        assert!(entry.is_some());
        assert_eq!(entry.expect("should succeed in test").description, "first");
    }

    #[test]
    fn test_tag_latest() {
        let mut h = LutVersionHistory::new();
        h.add(v(1, 0, 0), "initial");
        assert!(h.tag_latest("release"));
        let entry = h.find(v(1, 0, 0)).expect("should succeed in test");
        assert_eq!(
            entry.tag.as_ref().expect("should succeed in test").label(),
            "release"
        );
    }
}
