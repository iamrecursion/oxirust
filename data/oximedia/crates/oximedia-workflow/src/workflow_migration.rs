#![allow(dead_code)]
//! Workflow schema migration and version upgrade utilities.
//!
//! Provides a framework for migrating workflow definitions between schema
//! versions, enabling backwards compatibility and safe rollouts of workflow
//! changes across the OxiMedia platform.

use std::collections::HashMap;
use std::fmt;

/// A semantic version triplet for workflow schemas.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (backwards-compatible additions).
    pub minor: u32,
    /// Patch version (bug fixes).
    pub patch: u32,
}

impl SchemaVersion {
    /// Create a new schema version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with another (same major version).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }

    /// Check if this version is newer than another.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        self > other
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Errors that can occur during migration.
#[derive(Debug, Clone, PartialEq)]
pub enum MigrationError {
    /// No migration path exists between the two versions.
    NoPath {
        /// The version migrating from.
        from: SchemaVersion,
        /// The version migrating to.
        to: SchemaVersion,
    },
    /// A migration step failed.
    StepFailed {
        /// Description of the step.
        step: String,
        /// Reason for failure.
        reason: String,
    },
    /// The definition is already at the target version.
    AlreadyCurrent,
    /// The target version is older than the current version.
    DowngradeNotSupported,
    /// Validation failed after migration.
    ValidationFailed(String),
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPath { from, to } => {
                write!(f, "no migration path from {from} to {to}")
            }
            Self::StepFailed { step, reason } => {
                write!(f, "migration step '{step}' failed: {reason}")
            }
            Self::AlreadyCurrent => write!(f, "already at target version"),
            Self::DowngradeNotSupported => write!(f, "downgrade not supported"),
            Self::ValidationFailed(msg) => write!(f, "validation failed: {msg}"),
        }
    }
}

/// A single field change within a migration step.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldChange {
    /// Add a new field with a default value.
    AddField {
        /// Name of the field.
        name: String,
        /// Default value as a string.
        default: String,
    },
    /// Remove a field.
    RemoveField {
        /// Name of the field to remove.
        name: String,
    },
    /// Rename a field.
    RenameField {
        /// Original name.
        from: String,
        /// New name.
        to: String,
    },
    /// Change the type of a field (with a conversion description).
    ChangeType {
        /// Name of the field.
        name: String,
        /// Description of the type change.
        description: String,
    },
}

/// A single migration step between adjacent versions.
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// Version this step migrates from.
    pub from: SchemaVersion,
    /// Version this step migrates to.
    pub to: SchemaVersion,
    /// Human-readable description of the migration.
    pub description: String,
    /// Field changes applied by this step.
    pub changes: Vec<FieldChange>,
}

impl MigrationStep {
    /// Create a new migration step.
    pub fn new(from: SchemaVersion, to: SchemaVersion, description: impl Into<String>) -> Self {
        Self {
            from,
            to,
            description: description.into(),
            changes: Vec::new(),
        }
    }

    /// Add a field change to this step.
    pub fn add_change(&mut self, change: FieldChange) {
        self.changes.push(change);
    }

    /// Apply this migration step to a workflow definition (key-value map).
    pub fn apply(&self, definition: &mut HashMap<String, String>) -> Result<(), MigrationError> {
        for change in &self.changes {
            match change {
                FieldChange::AddField { name, default } => {
                    definition
                        .entry(name.clone())
                        .or_insert_with(|| default.clone());
                }
                FieldChange::RemoveField { name } => {
                    definition.remove(name);
                }
                FieldChange::RenameField { from, to } => {
                    if let Some(value) = definition.remove(from) {
                        definition.insert(to.clone(), value);
                    }
                }
                FieldChange::ChangeType {
                    name,
                    description: _,
                } => {
                    if !definition.contains_key(name) {
                        return Err(MigrationError::StepFailed {
                            step: self.description.clone(),
                            reason: format!("field '{name}' not found for type change"),
                        });
                    }
                }
            }
        }
        // Update the version stamp in the definition.
        definition.insert("_schema_version".to_string(), self.to.to_string());
        Ok(())
    }
}

/// Registry of migration steps that can find paths between versions.
#[derive(Debug, Clone)]
pub struct MigrationRegistry {
    /// All registered migration steps.
    steps: Vec<MigrationStep>,
}

impl MigrationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Register a migration step.
    pub fn register(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Find a migration path from one version to another.
    pub fn find_path(
        &self,
        from: &SchemaVersion,
        to: &SchemaVersion,
    ) -> Result<Vec<&MigrationStep>, MigrationError> {
        if from == to {
            return Err(MigrationError::AlreadyCurrent);
        }
        if from > to {
            return Err(MigrationError::DowngradeNotSupported);
        }

        let mut path = Vec::new();
        let mut current = from.clone();

        while current != *to {
            let step = self
                .steps
                .iter()
                .find(|s| s.from == current)
                .ok_or_else(|| MigrationError::NoPath {
                    from: current.clone(),
                    to: to.clone(),
                })?;
            if step.to > *to {
                return Err(MigrationError::NoPath {
                    from: from.clone(),
                    to: to.clone(),
                });
            }
            current = step.to.clone();
            path.push(step);
        }

        Ok(path)
    }

    /// Execute a full migration on a definition.
    pub fn migrate(
        &self,
        definition: &mut HashMap<String, String>,
        from: &SchemaVersion,
        to: &SchemaVersion,
    ) -> Result<Vec<String>, MigrationError> {
        let path = self.find_path(from, to)?;
        let mut descriptions = Vec::new();
        for step in path {
            step.apply(definition)?;
            descriptions.push(step.description.clone());
        }
        Ok(descriptions)
    }

    /// Return how many steps are registered.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// List all registered versions (source and destination).
    pub fn known_versions(&self) -> Vec<SchemaVersion> {
        let mut versions: Vec<SchemaVersion> = self
            .steps
            .iter()
            .flat_map(|s| vec![s.from.clone(), s.to.clone()])
            .collect();
        versions.sort();
        versions.dedup();
        versions
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> SchemaVersion {
        SchemaVersion::new(major, minor, patch)
    }

    #[test]
    fn test_version_display() {
        assert_eq!(v(1, 2, 3).to_string(), "1.2.3");
        assert_eq!(v(0, 0, 1).to_string(), "0.0.1");
    }

    #[test]
    fn test_version_compatibility() {
        assert!(v(1, 0, 0).is_compatible_with(&v(1, 5, 3)));
        assert!(!v(1, 0, 0).is_compatible_with(&v(2, 0, 0)));
    }

    #[test]
    fn test_version_ordering() {
        assert!(v(1, 0, 0).is_newer_than(&v(0, 9, 9)));
        assert!(v(1, 1, 0).is_newer_than(&v(1, 0, 9)));
        assert!(!v(1, 0, 0).is_newer_than(&v(1, 0, 0)));
    }

    #[test]
    fn test_migration_step_add_field() {
        let mut step = MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "add priority");
        step.add_change(FieldChange::AddField {
            name: "priority".into(),
            default: "normal".into(),
        });

        let mut def = HashMap::new();
        def.insert("name".into(), "test".into());
        step.apply(&mut def).expect("should succeed in test");
        assert_eq!(
            def.get("priority").expect("should succeed in test"),
            "normal"
        );
        assert_eq!(
            def.get("_schema_version").expect("should succeed in test"),
            "1.1.0"
        );
    }

    #[test]
    fn test_migration_step_remove_field() {
        let mut step = MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "remove legacy");
        step.add_change(FieldChange::RemoveField {
            name: "legacy_flag".into(),
        });

        let mut def = HashMap::new();
        def.insert("legacy_flag".into(), "true".into());
        step.apply(&mut def).expect("should succeed in test");
        assert!(!def.contains_key("legacy_flag"));
    }

    #[test]
    fn test_migration_step_rename_field() {
        let mut step = MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "rename field");
        step.add_change(FieldChange::RenameField {
            from: "old_name".into(),
            to: "new_name".into(),
        });

        let mut def = HashMap::new();
        def.insert("old_name".into(), "value".into());
        step.apply(&mut def).expect("should succeed in test");
        assert!(!def.contains_key("old_name"));
        assert_eq!(
            def.get("new_name").expect("should succeed in test"),
            "value"
        );
    }

    #[test]
    fn test_change_type_missing_field() {
        let mut step = MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "change type");
        step.add_change(FieldChange::ChangeType {
            name: "missing".into(),
            description: "int -> str".into(),
        });

        let mut def = HashMap::new();
        let result = step.apply(&mut def);
        assert!(matches!(result, Err(MigrationError::StepFailed { .. })));
    }

    #[test]
    fn test_registry_find_path() {
        let mut reg = MigrationRegistry::new();
        reg.register(MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "step1"));
        reg.register(MigrationStep::new(v(1, 1, 0), v(1, 2, 0), "step2"));

        let path = reg
            .find_path(&v(1, 0, 0), &v(1, 2, 0))
            .expect("should succeed in test");
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].description, "step1");
        assert_eq!(path[1].description, "step2");
    }

    #[test]
    fn test_registry_already_current() {
        let reg = MigrationRegistry::new();
        let result = reg.find_path(&v(1, 0, 0), &v(1, 0, 0));
        assert!(matches!(result, Err(MigrationError::AlreadyCurrent)));
    }

    #[test]
    fn test_registry_downgrade_not_supported() {
        let reg = MigrationRegistry::new();
        let result = reg.find_path(&v(2, 0, 0), &v(1, 0, 0));
        assert!(matches!(result, Err(MigrationError::DowngradeNotSupported)));
    }

    #[test]
    fn test_registry_no_path() {
        let mut reg = MigrationRegistry::new();
        reg.register(MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "step1"));
        let result = reg.find_path(&v(1, 0, 0), &v(1, 3, 0));
        assert!(matches!(result, Err(MigrationError::NoPath { .. })));
    }

    #[test]
    fn test_full_migration() {
        let mut reg = MigrationRegistry::new();
        let mut s1 = MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "add timeout");
        s1.add_change(FieldChange::AddField {
            name: "timeout".into(),
            default: "30".into(),
        });
        let mut s2 = MigrationStep::new(v(1, 1, 0), v(1, 2, 0), "rename output");
        s2.add_change(FieldChange::RenameField {
            from: "dest".into(),
            to: "output_path".into(),
        });
        reg.register(s1);
        reg.register(s2);

        let mut def = HashMap::new();
        def.insert("name".into(), "wf1".into());
        def.insert("dest".into(), "/out".into());

        let descs = reg
            .migrate(&mut def, &v(1, 0, 0), &v(1, 2, 0))
            .expect("should succeed in test");
        assert_eq!(descs.len(), 2);
        assert_eq!(def.get("timeout").expect("should succeed in test"), "30");
        assert_eq!(
            def.get("output_path").expect("should succeed in test"),
            "/out"
        );
        assert!(!def.contains_key("dest"));
    }

    #[test]
    fn test_known_versions() {
        let mut reg = MigrationRegistry::new();
        reg.register(MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "s1"));
        reg.register(MigrationStep::new(v(1, 1, 0), v(2, 0, 0), "s2"));

        let versions = reg.known_versions();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0], v(1, 0, 0));
        assert_eq!(versions[1], v(1, 1, 0));
        assert_eq!(versions[2], v(2, 0, 0));
    }

    #[test]
    fn test_step_count() {
        let mut reg = MigrationRegistry::new();
        assert_eq!(reg.step_count(), 0);
        reg.register(MigrationStep::new(v(1, 0, 0), v(1, 1, 0), "s1"));
        assert_eq!(reg.step_count(), 1);
    }
}
