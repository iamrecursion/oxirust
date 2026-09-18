//! Semantic versioning helpers for media codec and format metadata.
//!
//! Provides [`Version`], [`VersionRange`], and [`VersionRegistry`] for
//! tracking and negotiating codec / container versions across the `OxiMedia`
//! pipeline.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// A semantic version number (`major.minor.patch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// Major version – incompatible API changes.
    pub major: u16,
    /// Minor version – backwards-compatible feature additions.
    pub minor: u16,
    /// Patch version – backwards-compatible bug fixes.
    pub patch: u16,
}

impl Version {
    /// Construct a [`Version`] from its three components.
    #[must_use]
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `true` if `self` is compatible with `required`.
    ///
    /// Compatibility rule: same major version and `self` is at least as recent
    /// (i.e. `self >= required`).
    #[must_use]
    pub fn is_compatible_with(self, required: Self) -> bool {
        self.major == required.major && self >= required
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// An inclusive range of [`Version`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionRange {
    /// The lowest acceptable version (inclusive).
    pub min: Version,
    /// The highest acceptable version (inclusive).
    pub max: Version,
}

impl VersionRange {
    /// Create a new [`VersionRange`].
    #[must_use]
    pub fn new(min: Version, max: Version) -> Self {
        Self { min, max }
    }

    /// Returns `true` if `v` falls within this range (inclusive on both ends).
    #[must_use]
    pub fn contains(self, v: Version) -> bool {
        v >= self.min && v <= self.max
    }
}

/// Registration record stored inside [`VersionRegistry`].
#[derive(Debug, Clone)]
struct VersionRecord {
    version: Version,
    is_stable: bool,
    is_deprecated: bool,
}

/// A registry that tracks known versions of a codec, format, or API.
#[derive(Debug, Default)]
pub struct VersionRegistry {
    records: HashMap<String, Vec<VersionRecord>>,
}

impl VersionRegistry {
    /// Create an empty [`VersionRegistry`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Register a version for `name`.
    ///
    /// `stable` indicates whether this is a stable release.
    pub fn register(&mut self, name: impl Into<String>, version: Version, stable: bool) {
        let entry = self.records.entry(name.into()).or_default();
        entry.push(VersionRecord {
            version,
            is_stable: stable,
            is_deprecated: false,
        });
    }

    /// Mark a specific version of `name` as deprecated.
    ///
    /// Returns `true` if the version was found and marked.
    pub fn deprecate(&mut self, name: &str, version: Version) -> bool {
        if let Some(records) = self.records.get_mut(name) {
            for rec in records.iter_mut() {
                if rec.version == version {
                    rec.is_deprecated = true;
                    return true;
                }
            }
        }
        false
    }

    /// Returns the latest stable [`Version`] registered for `name`, or `None`.
    #[must_use]
    pub fn latest_stable(&self, name: &str) -> Option<Version> {
        self.records.get(name).and_then(|records| {
            records
                .iter()
                .filter(|r| r.is_stable && !r.is_deprecated)
                .map(|r| r.version)
                .max()
        })
    }

    /// Returns all deprecated versions for `name`.
    #[must_use]
    pub fn deprecated_versions(&self, name: &str) -> Vec<Version> {
        self.records
            .get(name)
            .map(|records| {
                records
                    .iter()
                    .filter(|r| r.is_deprecated)
                    .map(|r| r.version)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns all registered versions for `name` sorted ascending.
    #[must_use]
    pub fn all_versions(&self, name: &str) -> Vec<Version> {
        let mut v: Vec<Version> = self
            .records
            .get(name)
            .map(|records| records.iter().map(|r| r.version).collect())
            .unwrap_or_default();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u16, minor: u16, patch: u16) -> Version {
        Version::new(major, minor, patch)
    }

    #[test]
    fn test_version_display() {
        assert_eq!(v(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn test_version_ordering() {
        assert!(v(1, 2, 3) > v(1, 2, 2));
        assert!(v(2, 0, 0) > v(1, 9, 9));
        assert_eq!(v(1, 0, 0), v(1, 0, 0));
    }

    #[test]
    fn test_is_compatible_with_same() {
        assert!(v(1, 2, 3).is_compatible_with(v(1, 2, 3)));
    }

    #[test]
    fn test_is_compatible_with_newer_minor() {
        assert!(v(1, 3, 0).is_compatible_with(v(1, 2, 0)));
    }

    #[test]
    fn test_is_compatible_with_different_major() {
        assert!(!v(2, 0, 0).is_compatible_with(v(1, 0, 0)));
    }

    #[test]
    fn test_is_compatible_with_older_fails() {
        assert!(!v(1, 1, 0).is_compatible_with(v(1, 2, 0)));
    }

    #[test]
    fn test_version_range_contains() {
        let range = VersionRange::new(v(1, 0, 0), v(1, 9, 9));
        assert!(range.contains(v(1, 5, 0)));
        assert!(range.contains(v(1, 0, 0)));
        assert!(range.contains(v(1, 9, 9)));
        assert!(!range.contains(v(2, 0, 0)));
        assert!(!range.contains(v(0, 9, 9)));
    }

    #[test]
    fn test_registry_latest_stable_none() {
        let reg = VersionRegistry::new();
        assert!(reg.latest_stable("av1").is_none());
    }

    #[test]
    fn test_registry_register_and_latest() {
        let mut reg = VersionRegistry::new();
        reg.register("av1", v(1, 0, 0), true);
        reg.register("av1", v(1, 1, 0), true);
        assert_eq!(reg.latest_stable("av1"), Some(v(1, 1, 0)));
    }

    #[test]
    fn test_registry_unstable_not_latest_stable() {
        let mut reg = VersionRegistry::new();
        reg.register("vp9", v(1, 0, 0), true);
        reg.register("vp9", v(2, 0, 0), false); // unstable
        assert_eq!(reg.latest_stable("vp9"), Some(v(1, 0, 0)));
    }

    #[test]
    fn test_registry_deprecated_versions() {
        let mut reg = VersionRegistry::new();
        reg.register("opus", v(1, 0, 0), true);
        reg.register("opus", v(1, 1, 0), true);
        reg.deprecate("opus", v(1, 0, 0));
        let dep = reg.deprecated_versions("opus");
        assert_eq!(dep, vec![v(1, 0, 0)]);
    }

    #[test]
    fn test_registry_deprecated_excluded_from_latest_stable() {
        let mut reg = VersionRegistry::new();
        reg.register("flac", v(1, 0, 0), true);
        reg.deprecate("flac", v(1, 0, 0));
        assert!(reg.latest_stable("flac").is_none());
    }

    #[test]
    fn test_registry_all_versions_sorted() {
        let mut reg = VersionRegistry::new();
        reg.register("theora", v(1, 2, 0), true);
        reg.register("theora", v(1, 0, 0), true);
        reg.register("theora", v(1, 1, 0), false);
        let all = reg.all_versions("theora");
        assert_eq!(all, vec![v(1, 0, 0), v(1, 1, 0), v(1, 2, 0)]);
    }

    #[test]
    fn test_registry_deprecate_unknown_returns_false() {
        let mut reg = VersionRegistry::new();
        assert!(!reg.deprecate("nonexistent", v(1, 0, 0)));
    }
}
