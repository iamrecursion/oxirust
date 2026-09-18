//! Workflow versioning and migration support.
//!
//! Provides `WorkflowVersion`, `VersionMigration`, and
//! `WorkflowVersionRegistry` for managing schema evolution of stored
//! workflow definitions.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// WorkflowVersion
// ---------------------------------------------------------------------------

/// A semantic version for a workflow definition schema.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkflowVersion {
    /// Major component — incompatible API changes.
    pub major: u32,
    /// Minor component — backwards-compatible feature additions.
    pub minor: u32,
    /// Patch component — backwards-compatible bug fixes.
    pub patch: u32,
}

impl WorkflowVersion {
    /// Construct a new version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a `"major.minor.patch"` string.
    ///
    /// Returns `None` if the string is malformed.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Returns `true` if `other` is compatible with `self`.
    ///
    /// Compatibility requires the same major version and a minor ≥ own minor.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major && other.minor >= self.minor
    }

    /// Formatted string representation.
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for WorkflowVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ---------------------------------------------------------------------------
// VersionMigration
// ---------------------------------------------------------------------------

/// A migration that transforms a workflow payload from one version to another.
#[derive(Debug, Clone)]
pub struct VersionMigration {
    /// The schema version this migration originates from.
    pub from: WorkflowVersion,
    /// The schema version this migration produces.
    pub to: WorkflowVersion,
    /// Short description of what the migration does.
    pub description: String,
}

impl VersionMigration {
    /// Create a new migration descriptor.
    #[must_use]
    pub fn new(from: WorkflowVersion, to: WorkflowVersion, description: impl Into<String>) -> Self {
        Self {
            from,
            to,
            description: description.into(),
        }
    }

    /// Source version of this migration.
    #[must_use]
    pub fn from_version(&self) -> &WorkflowVersion {
        &self.from
    }

    /// Target version of this migration.
    #[must_use]
    pub fn to_version(&self) -> &WorkflowVersion {
        &self.to
    }

    /// Returns `true` if this migration is a major-version bump.
    #[must_use]
    pub fn is_major_migration(&self) -> bool {
        self.from.major != self.to.major
    }
}

// ---------------------------------------------------------------------------
// WorkflowVersionRegistry
// ---------------------------------------------------------------------------

/// Registry that tracks all known workflow versions and the migrations
/// between them.
#[derive(Debug, Default)]
pub struct WorkflowVersionRegistry {
    versions: Vec<WorkflowVersion>,
    migrations: HashMap<(WorkflowVersion, WorkflowVersion), VersionMigration>,
}

impl WorkflowVersionRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a known version.
    pub fn register(&mut self, version: WorkflowVersion) {
        if !self.versions.contains(&version) {
            self.versions.push(version.clone());
            self.versions.sort();
        }
    }

    /// Add a migration between two registered versions.
    pub fn add_migration(&mut self, migration: VersionMigration) {
        self.register(migration.from.clone());
        self.register(migration.to.clone());
        let key = (migration.from.clone(), migration.to.clone());
        self.migrations.insert(key, migration);
    }

    /// Return the highest registered version, or `None` if empty.
    #[must_use]
    pub fn latest(&self) -> Option<&WorkflowVersion> {
        self.versions.last()
    }

    /// Retrieve the migration from `from` to `to`, if registered.
    #[must_use]
    pub fn migrate(
        &self,
        from: &WorkflowVersion,
        to: &WorkflowVersion,
    ) -> Option<&VersionMigration> {
        self.migrations.get(&(from.clone(), to.clone()))
    }

    /// All registered versions, sorted ascending.
    #[must_use]
    pub fn all_versions(&self) -> &[WorkflowVersion] {
        &self.versions
    }

    /// All registered migrations.
    #[must_use]
    pub fn all_migrations(&self) -> Vec<&VersionMigration> {
        self.migrations.values().collect()
    }

    /// Number of registered versions.
    #[must_use]
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_new_and_display() {
        let v = WorkflowVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_parse_valid() {
        let v = WorkflowVersion::parse("2.0.1").expect("should succeed in test");
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 1);
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(WorkflowVersion::parse("1.2").is_none());
        assert!(WorkflowVersion::parse("abc").is_none());
        assert!(WorkflowVersion::parse("1.x.3").is_none());
    }

    #[test]
    fn test_version_ordering() {
        let v1 = WorkflowVersion::new(1, 0, 0);
        let v2 = WorkflowVersion::new(1, 1, 0);
        let v3 = WorkflowVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_is_compatible_with_same_major() {
        let v1 = WorkflowVersion::new(1, 2, 0);
        let v2 = WorkflowVersion::new(1, 3, 0);
        assert!(v1.is_compatible_with(&v2)); // v2 minor >= v1 minor
    }

    #[test]
    fn test_is_compatible_with_older_minor() {
        let v1 = WorkflowVersion::new(1, 3, 0);
        let v2 = WorkflowVersion::new(1, 2, 0);
        assert!(!v1.is_compatible_with(&v2)); // v2 minor < v1 minor
    }

    #[test]
    fn test_is_compatible_different_major() {
        let v1 = WorkflowVersion::new(1, 0, 0);
        let v2 = WorkflowVersion::new(2, 0, 0);
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_migration_from_version() {
        let from = WorkflowVersion::new(1, 0, 0);
        let to = WorkflowVersion::new(1, 1, 0);
        let m = VersionMigration::new(from.clone(), to, "add field");
        assert_eq!(m.from_version(), &from);
    }

    #[test]
    fn test_migration_is_major() {
        let m = VersionMigration::new(
            WorkflowVersion::new(1, 0, 0),
            WorkflowVersion::new(2, 0, 0),
            "major bump",
        );
        assert!(m.is_major_migration());
    }

    #[test]
    fn test_migration_not_major() {
        let m = VersionMigration::new(
            WorkflowVersion::new(1, 0, 0),
            WorkflowVersion::new(1, 1, 0),
            "minor bump",
        );
        assert!(!m.is_major_migration());
    }

    #[test]
    fn test_registry_register_and_latest() {
        let mut reg = WorkflowVersionRegistry::new();
        reg.register(WorkflowVersion::new(1, 0, 0));
        reg.register(WorkflowVersion::new(1, 1, 0));
        reg.register(WorkflowVersion::new(2, 0, 0));
        assert_eq!(
            reg.latest().expect("should succeed in test"),
            &WorkflowVersion::new(2, 0, 0)
        );
    }

    #[test]
    fn test_registry_add_migration_and_lookup() {
        let mut reg = WorkflowVersionRegistry::new();
        let from = WorkflowVersion::new(1, 0, 0);
        let to = WorkflowVersion::new(1, 1, 0);
        reg.add_migration(VersionMigration::new(from.clone(), to.clone(), "test"));
        let m = reg.migrate(&from, &to).expect("should succeed in test");
        assert_eq!(m.description, "test");
    }

    #[test]
    fn test_registry_missing_migration() {
        let reg = WorkflowVersionRegistry::new();
        let v = WorkflowVersion::new(1, 0, 0);
        assert!(reg.migrate(&v, &v).is_none());
    }

    #[test]
    fn test_registry_version_count() {
        let mut reg = WorkflowVersionRegistry::new();
        reg.register(WorkflowVersion::new(1, 0, 0));
        reg.register(WorkflowVersion::new(1, 0, 0)); // duplicate — should not increase count
        assert_eq!(reg.version_count(), 1);
    }

    #[test]
    fn test_registry_empty_latest() {
        let reg = WorkflowVersionRegistry::new();
        assert!(reg.latest().is_none());
    }
}
