//! Conform version management.
//!
//! Tracks a directed acyclic graph of conform versions, allowing diffing,
//! ancestor traversal, and structured naming.

#![allow(dead_code)]

/// A single change recorded in a conform version.
#[derive(Debug, Clone)]
pub struct ConformChange {
    /// What kind of change this is.
    pub change_type: ChangeType,
    /// Human-readable description.
    pub description: String,
    /// ID of the affected element (clip, track, etc.).
    pub element_id: String,
}

/// Classification of a conform change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// New element added.
    Add,
    /// Element removed.
    Remove,
    /// Element modified.
    Modify,
    /// Elements reordered.
    Reorder,
    /// Color grading applied or changed.
    ColorGrade,
    /// Audio mix changed.
    AudioMix,
}

impl ChangeType {
    /// Is this considered a *major* change (bumps the major version number)?
    #[must_use]
    pub fn is_major(&self) -> bool {
        matches!(self, Self::Add | Self::Remove | Self::Reorder)
    }
}

/// A single conform version node.
#[derive(Debug, Clone)]
pub struct ConformVersion {
    /// Unique version identifier (UUID or user-assigned string).
    pub id: String,
    /// Human-readable name (e.g. `"v1.2"`).
    pub name: String,
    /// ID of the parent version, or `None` for the root.
    pub parent_id: Option<String>,
    /// List of changes from the parent to this version.
    pub changes: Vec<ConformChange>,
    /// Wall-clock creation time in milliseconds since the Unix epoch.
    pub created_at_ms: u64,
}

impl ConformVersion {
    /// Create a new version.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        parent_id: Option<String>,
        changes: Vec<ConformChange>,
        created_at_ms: u64,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            parent_id,
            changes,
            created_at_ms,
        }
    }
}

/// A directed acyclic graph of conform versions.
#[derive(Debug, Default)]
pub struct ConformVersionGraph {
    /// All versions, in insertion order.
    pub versions: Vec<ConformVersion>,
}

impl ConformVersionGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a version to the graph.
    pub fn add_version(&mut self, v: ConformVersion) {
        self.versions.push(v);
    }

    /// Return all ancestor versions of `id` (including the version itself),
    /// ordered from root to `id`.
    #[must_use]
    pub fn ancestors_of(&self, id: &str) -> Vec<&ConformVersion> {
        let mut ancestors = Vec::new();
        let mut current_id = Some(id.to_string());

        // Walk up the parent chain
        while let Some(cid) = current_id {
            if let Some(v) = self.versions.iter().find(|v| v.id == cid) {
                ancestors.push(v);
                current_id = v.parent_id.clone();
            } else {
                break;
            }
        }

        ancestors.reverse();
        ancestors
    }

    /// Return all changes that differ between versions `a_id` and `b_id`.
    ///
    /// Specifically, returns all changes that are in `b_id`'s lineage but
    /// **not** in `a_id`'s lineage (i.e. changes introduced between `a` and `b`).
    #[must_use]
    pub fn diff<'a>(&'a self, a_id: &str, b_id: &str) -> Vec<&'a ConformChange> {
        let a_ancestors: std::collections::HashSet<&str> = self
            .ancestors_of(a_id)
            .into_iter()
            .map(|v| v.id.as_str())
            .collect();

        let b_ancestors = self.ancestors_of(b_id);

        let mut result = Vec::new();
        for version in &b_ancestors {
            if !a_ancestors.contains(version.id.as_str()) {
                result.extend(version.changes.iter());
            }
        }
        result
    }

    /// Find a version by ID.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&ConformVersion> {
        self.versions.iter().find(|v| v.id == id)
    }
}

/// Version naming utilities.
pub struct VersionNaming;

impl VersionNaming {
    /// Generate a new version name based on the parent's name and the change type.
    ///
    /// - Root version: `"v1.0"`.
    /// - Major change (Add/Remove/Reorder): bump major component.
    /// - Minor change: bump minor component.
    #[must_use]
    pub fn generate(parent: Option<&str>, change_type: &ChangeType) -> String {
        let Some(parent_name) = parent else {
            return "v1.0".to_string();
        };

        // Parse `vMAJOR.MINOR` from parent name
        let stripped = parent_name.trim_start_matches('v');
        let (major, minor) = if let Some((m, n)) = stripped.split_once('.') {
            let major: u32 = m.parse().unwrap_or(1);
            let minor: u32 = n.parse().unwrap_or(0);
            (major, minor)
        } else {
            (1, 0)
        };

        if change_type.is_major() {
            format!("v{}.0", major + 1)
        } else {
            format!("v{}.{}", major, minor + 1)
        }
    }
}

/// A locked/approved checkpoint on a specific version.
#[derive(Debug, Clone)]
pub struct ConformCheckpoint {
    /// The version this checkpoint refers to.
    pub version_id: String,
    /// Whether this checkpoint is locked (immutable).
    pub locked: bool,
    /// Name/email of the approver, if approved.
    pub approved_by: Option<String>,
}

impl ConformCheckpoint {
    /// Create a new unlocked checkpoint.
    #[must_use]
    pub fn new(version_id: impl Into<String>) -> Self {
        Self {
            version_id: version_id.into(),
            locked: false,
            approved_by: None,
        }
    }

    /// Lock and approve this checkpoint.
    pub fn approve(&mut self, approver: impl Into<String>) {
        self.locked = true;
        self.approved_by = Some(approver.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_change(ct: ChangeType) -> ConformChange {
        ConformChange {
            change_type: ct,
            description: "test change".to_string(),
            element_id: "elem-1".to_string(),
        }
    }

    fn make_version(
        id: &str,
        name: &str,
        parent: Option<&str>,
        changes: Vec<ConformChange>,
    ) -> ConformVersion {
        ConformVersion::new(id, name, parent.map(str::to_string), changes, 0)
    }

    #[test]
    fn test_version_naming_root() {
        let name = VersionNaming::generate(None, &ChangeType::Add);
        assert_eq!(name, "v1.0");
    }

    #[test]
    fn test_version_naming_minor() {
        let name = VersionNaming::generate(Some("v1.0"), &ChangeType::Modify);
        assert_eq!(name, "v1.1");
    }

    #[test]
    fn test_version_naming_major_add() {
        let name = VersionNaming::generate(Some("v1.2"), &ChangeType::Add);
        assert_eq!(name, "v2.0");
    }

    #[test]
    fn test_version_naming_major_remove() {
        let name = VersionNaming::generate(Some("v2.0"), &ChangeType::Remove);
        assert_eq!(name, "v3.0");
    }

    #[test]
    fn test_version_naming_major_reorder() {
        let name = VersionNaming::generate(Some("v1.0"), &ChangeType::Reorder);
        assert_eq!(name, "v2.0");
    }

    #[test]
    fn test_version_naming_minor_color_grade() {
        let name = VersionNaming::generate(Some("v1.3"), &ChangeType::ColorGrade);
        assert_eq!(name, "v1.4");
    }

    #[test]
    fn test_graph_add_and_find() {
        let mut graph = ConformVersionGraph::new();
        let v = make_version("v1", "v1.0", None, vec![]);
        graph.add_version(v);
        assert!(graph.find("v1").is_some());
        assert!(graph.find("no-such").is_none());
    }

    #[test]
    fn test_ancestors_of_root() {
        let mut graph = ConformVersionGraph::new();
        graph.add_version(make_version("root", "v1.0", None, vec![]));
        let anc = graph.ancestors_of("root");
        assert_eq!(anc.len(), 1);
        assert_eq!(anc[0].id, "root");
    }

    #[test]
    fn test_ancestors_of_chain() {
        let mut graph = ConformVersionGraph::new();
        graph.add_version(make_version("v1", "v1.0", None, vec![]));
        graph.add_version(make_version("v2", "v1.1", Some("v1"), vec![]));
        graph.add_version(make_version("v3", "v1.2", Some("v2"), vec![]));

        let anc = graph.ancestors_of("v3");
        assert_eq!(anc.len(), 3);
        assert_eq!(anc[0].id, "v1");
        assert_eq!(anc[2].id, "v3");
    }

    #[test]
    fn test_diff_returns_new_changes() {
        let mut graph = ConformVersionGraph::new();
        graph.add_version(make_version("v1", "v1.0", None, vec![]));
        graph.add_version(make_version(
            "v2",
            "v1.1",
            Some("v1"),
            vec![make_change(ChangeType::Modify)],
        ));

        let changes = graph.diff("v1", "v2");
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].change_type, ChangeType::Modify);
    }

    #[test]
    fn test_checkpoint_approve() {
        let mut cp = ConformCheckpoint::new("v1");
        assert!(!cp.locked);
        cp.approve("editor@studio.com");
        assert!(cp.locked);
        assert_eq!(cp.approved_by.as_deref(), Some("editor@studio.com"));
    }

    #[test]
    fn test_change_type_is_major() {
        assert!(ChangeType::Add.is_major());
        assert!(ChangeType::Remove.is_major());
        assert!(ChangeType::Reorder.is_major());
        assert!(!ChangeType::Modify.is_major());
        assert!(!ChangeType::ColorGrade.is_major());
        assert!(!ChangeType::AudioMix.is_major());
    }
}
