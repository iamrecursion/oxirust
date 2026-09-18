//! API versioning registry: semantic version tracking, compatibility checks, and deprecation.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// A semantic API version with major, minor, and patch components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ApiVersion {
    /// Creates a new `ApiVersion`.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `true` if this version is backward-compatible with `other`.
    ///
    /// Compatibility rule: same major version **and** `self >= other`.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major && *self >= *other
    }

    /// Returns `true` if this is a pre-release (major == 0).
    pub fn is_pre_release(&self) -> bool {
        self.major == 0
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// An inclusive range of `ApiVersion` values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionRange {
    pub min: ApiVersion,
    pub max: ApiVersion,
}

impl VersionRange {
    /// Creates a new `VersionRange`.
    pub fn new(min: ApiVersion, max: ApiVersion) -> Self {
        Self { min, max }
    }

    /// Returns `true` if `version` falls within `[min, max]` (inclusive).
    pub fn contains(&self, version: &ApiVersion) -> bool {
        *version >= self.min && *version <= self.max
    }

    /// Returns the number of minor versions between `min` and `max` within
    /// the same major version.  Returns 0 if majors differ or min > max.
    pub fn minor_span(&self) -> u32 {
        if self.min.major != self.max.major || self.max < self.min {
            return 0;
        }
        self.max.minor.saturating_sub(self.min.minor)
    }
}

/// Registration details for a single API version.
#[derive(Debug, Clone)]
pub struct VersionEntry {
    /// The registered version.
    pub version: ApiVersion,
    /// Human-readable change summary.
    pub description: String,
    /// Whether this version is deprecated.
    pub deprecated: bool,
}

impl VersionEntry {
    fn new(version: ApiVersion, description: impl Into<String>) -> Self {
        Self {
            version,
            description: description.into(),
            deprecated: false,
        }
    }
}

/// Registry of all known API versions for a service.
#[derive(Debug, Default)]
pub struct ApiVersionRegistry {
    versions: HashMap<ApiVersion, VersionEntry>,
}

impl ApiVersionRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new API version with a description.
    ///
    /// If the version already exists the description is updated.
    pub fn register(&mut self, version: ApiVersion, description: impl Into<String>) {
        let desc: String = description.into();
        match self.versions.entry(version) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                e.get_mut().description = desc;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(VersionEntry::new(version, desc));
            }
        }
    }

    /// Marks a version as deprecated.  Returns `false` if the version is
    /// not registered.
    pub fn deprecate(&mut self, version: &ApiVersion) -> bool {
        if let Some(entry) = self.versions.get_mut(version) {
            entry.deprecated = true;
            true
        } else {
            false
        }
    }

    /// Resolves the highest registered version that is compatible with
    /// `requested` (same major, >= requested).  Returns `None` if no
    /// compatible version exists.
    pub fn resolve(&self, requested: &ApiVersion) -> Option<&VersionEntry> {
        self.versions
            .values()
            .filter(|e| e.version.is_compatible_with(requested))
            .max_by_key(|e| e.version)
    }

    /// Returns all registered versions that have been marked as deprecated.
    pub fn deprecated_versions(&self) -> Vec<&VersionEntry> {
        self.versions.values().filter(|e| e.deprecated).collect()
    }

    /// Returns all registered versions, sorted ascending.
    pub fn all_versions(&self) -> Vec<&VersionEntry> {
        let mut v: Vec<&VersionEntry> = self.versions.values().collect();
        v.sort_by_key(|e| e.version);
        v
    }

    /// Returns `true` if the given version is registered.
    pub fn is_registered(&self, version: &ApiVersion) -> bool {
        self.versions.contains_key(version)
    }

    /// Returns the number of registered versions.
    pub fn count(&self) -> usize {
        self.versions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V1_0: ApiVersion = ApiVersion::new(1, 0, 0);
    const V1_1: ApiVersion = ApiVersion::new(1, 1, 0);
    const V1_2: ApiVersion = ApiVersion::new(1, 2, 0);
    const V2_0: ApiVersion = ApiVersion::new(2, 0, 0);

    #[test]
    fn test_version_ordering() {
        assert!(V1_0 < V1_1);
        assert!(V1_1 < V1_2);
        assert!(V1_2 < V2_0);
    }

    #[test]
    fn test_version_display() {
        assert_eq!(V1_2.to_string(), "1.2.0");
    }

    #[test]
    fn test_is_compatible_same_major() {
        assert!(V1_2.is_compatible_with(&V1_0));
        assert!(V1_2.is_compatible_with(&V1_2));
    }

    #[test]
    fn test_is_compatible_different_major() {
        assert!(!V2_0.is_compatible_with(&V1_0));
    }

    #[test]
    fn test_is_compatible_older_requesting_newer() {
        assert!(!V1_0.is_compatible_with(&V1_2));
    }

    #[test]
    fn test_is_pre_release() {
        let v = ApiVersion::new(0, 9, 1);
        assert!(v.is_pre_release());
        assert!(!V1_0.is_pre_release());
    }

    #[test]
    fn test_version_range_contains() {
        let range = VersionRange::new(V1_0, V1_2);
        assert!(range.contains(&V1_0));
        assert!(range.contains(&V1_1));
        assert!(range.contains(&V1_2));
        assert!(!range.contains(&V2_0));
    }

    #[test]
    fn test_version_range_minor_span() {
        let range = VersionRange::new(V1_0, V1_2);
        assert_eq!(range.minor_span(), 2);
    }

    #[test]
    fn test_version_range_minor_span_different_major() {
        let range = VersionRange::new(V1_0, V2_0);
        assert_eq!(range.minor_span(), 0);
    }

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_0, "initial release");
        reg.register(V1_1, "added streaming");
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_registry_is_registered() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_0, "v1");
        assert!(reg.is_registered(&V1_0));
        assert!(!reg.is_registered(&V2_0));
    }

    #[test]
    fn test_registry_resolve_best_match() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_0, "v1.0");
        reg.register(V1_1, "v1.1");
        reg.register(V1_2, "v1.2");
        let resolved = reg.resolve(&V1_0).expect("should succeed in test");
        assert_eq!(resolved.version, V1_2);
    }

    #[test]
    fn test_registry_resolve_no_match() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_0, "v1.0");
        assert!(reg.resolve(&V2_0).is_none());
    }

    #[test]
    fn test_registry_deprecate() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_0, "v1.0");
        assert!(reg.deprecate(&V1_0));
        assert!(!reg.deprecated_versions().is_empty());
    }

    #[test]
    fn test_registry_deprecate_unknown() {
        let mut reg = ApiVersionRegistry::new();
        assert!(!reg.deprecate(&V1_0));
    }

    #[test]
    fn test_registry_deprecated_versions_list() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_0, "v1.0");
        reg.register(V1_1, "v1.1");
        reg.register(V1_2, "v1.2");
        reg.deprecate(&V1_0);
        reg.deprecate(&V1_1);
        let deprecated = reg.deprecated_versions();
        assert_eq!(deprecated.len(), 2);
    }

    #[test]
    fn test_registry_all_versions_sorted() {
        let mut reg = ApiVersionRegistry::new();
        reg.register(V1_2, "v1.2");
        reg.register(V1_0, "v1.0");
        reg.register(V1_1, "v1.1");
        let all = reg.all_versions();
        assert_eq!(all[0].version, V1_0);
        assert_eq!(all[1].version, V1_1);
        assert_eq!(all[2].version, V1_2);
    }
}
