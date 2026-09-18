//! Media project management for the MAM system.
//!
//! Provides project-level grouping of assets with lifecycle tracking.

#![allow(dead_code)]

use std::collections::HashMap;

/// Status of a media project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectStatus {
    /// Project is being set up.
    Planning,
    /// Project is actively being worked on.
    Active,
    /// Project is paused temporarily.
    OnHold,
    /// Project is complete and archived.
    Archived,
    /// Project has been cancelled.
    Cancelled,
}

impl ProjectStatus {
    /// Returns `true` if the project is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns a human-readable label for the status.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Planning => "Planning",
            Self::Active => "Active",
            Self::OnHold => "On Hold",
            Self::Archived => "Archived",
            Self::Cancelled => "Cancelled",
        }
    }
}

/// A media project that groups related assets.
#[derive(Debug, Clone)]
pub struct MediaProject {
    /// Unique identifier.
    pub id: u64,
    /// Display name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Current status.
    pub status: ProjectStatus,
    /// IDs of assets belonging to this project.
    asset_ids: Vec<u64>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl MediaProject {
    /// Create a new project in Planning status.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: None,
            status: ProjectStatus::Planning,
            asset_ids: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Number of assets linked to this project.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.asset_ids.len()
    }

    /// Add an asset id to the project.
    pub fn add_asset(&mut self, asset_id: u64) {
        if !self.asset_ids.contains(&asset_id) {
            self.asset_ids.push(asset_id);
        }
    }

    /// Remove an asset id from the project.
    pub fn remove_asset(&mut self, asset_id: u64) {
        self.asset_ids.retain(|&id| id != asset_id);
    }

    /// Returns `true` if the project is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    /// Activate the project.
    pub fn activate(&mut self) {
        self.status = ProjectStatus::Active;
    }

    /// Archive the project.
    pub fn archive(&mut self) {
        self.status = ProjectStatus::Archived;
    }

    /// All asset ids in this project.
    #[must_use]
    pub fn asset_ids(&self) -> &[u64] {
        &self.asset_ids
    }
}

/// Manages a collection of media projects.
#[derive(Debug, Default)]
pub struct ProjectManager {
    projects: HashMap<u64, MediaProject>,
    next_id: u64,
}

impl ProjectManager {
    /// Create a new, empty project manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new project and return its id.
    pub fn create(&mut self, name: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let project = MediaProject::new(id, name);
        self.projects.insert(id, project);
        id
    }

    /// Archive a project by id. Returns `false` if not found.
    pub fn archive(&mut self, id: u64) -> bool {
        if let Some(p) = self.projects.get_mut(&id) {
            p.archive();
            true
        } else {
            false
        }
    }

    /// Find all active projects.
    #[must_use]
    pub fn find_active(&self) -> Vec<&MediaProject> {
        self.projects.values().filter(|p| p.is_active()).collect()
    }

    /// Get a project by id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&MediaProject> {
        self.projects.get(&id)
    }

    /// Get a mutable reference to a project by id.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut MediaProject> {
        self.projects.get_mut(&id)
    }

    /// Total number of managed projects.
    #[must_use]
    pub fn project_count(&self) -> usize {
        self.projects.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_status_is_active() {
        assert!(ProjectStatus::Active.is_active());
        assert!(!ProjectStatus::Planning.is_active());
        assert!(!ProjectStatus::OnHold.is_active());
        assert!(!ProjectStatus::Archived.is_active());
        assert!(!ProjectStatus::Cancelled.is_active());
    }

    #[test]
    fn test_project_status_label() {
        assert_eq!(ProjectStatus::Active.label(), "Active");
        assert_eq!(ProjectStatus::Archived.label(), "Archived");
        assert_eq!(ProjectStatus::OnHold.label(), "On Hold");
        assert_eq!(ProjectStatus::Cancelled.label(), "Cancelled");
        assert_eq!(ProjectStatus::Planning.label(), "Planning");
    }

    #[test]
    fn test_media_project_new() {
        let p = MediaProject::new(1, "Test Project");
        assert_eq!(p.id, 1);
        assert_eq!(p.name, "Test Project");
        assert_eq!(p.asset_count(), 0);
        assert_eq!(p.status, ProjectStatus::Planning);
        assert!(!p.is_active());
    }

    #[test]
    fn test_media_project_add_remove_asset() {
        let mut p = MediaProject::new(0, "P");
        p.add_asset(10);
        p.add_asset(20);
        p.add_asset(10); // duplicate — should not double-add
        assert_eq!(p.asset_count(), 2);
        p.remove_asset(10);
        assert_eq!(p.asset_count(), 1);
        assert_eq!(p.asset_ids(), &[20]);
    }

    #[test]
    fn test_media_project_activate_archive() {
        let mut p = MediaProject::new(0, "P");
        p.activate();
        assert!(p.is_active());
        p.archive();
        assert!(!p.is_active());
        assert_eq!(p.status, ProjectStatus::Archived);
    }

    #[test]
    fn test_project_manager_create() {
        let mut mgr = ProjectManager::new();
        let id = mgr.create("Alpha");
        assert_eq!(id, 0);
        let id2 = mgr.create("Beta");
        assert_eq!(id2, 1);
        assert_eq!(mgr.project_count(), 2);
    }

    #[test]
    fn test_project_manager_archive() {
        let mut mgr = ProjectManager::new();
        let id = mgr.create("Project");
        if let Some(p) = mgr.get_mut(id) {
            p.activate();
        }
        assert!(mgr.archive(id));
        let p = mgr.get(id).expect("should succeed in test");
        assert_eq!(p.status, ProjectStatus::Archived);
    }

    #[test]
    fn test_project_manager_archive_missing() {
        let mut mgr = ProjectManager::new();
        assert!(!mgr.archive(999));
    }

    #[test]
    fn test_project_manager_find_active() {
        let mut mgr = ProjectManager::new();
        let id1 = mgr.create("Active Project");
        let _id2 = mgr.create("Planning Project");
        if let Some(p) = mgr.get_mut(id1) {
            p.activate();
        }
        let active = mgr.find_active();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, id1);
    }

    #[test]
    fn test_project_manager_find_active_empty() {
        let mgr = ProjectManager::new();
        assert!(mgr.find_active().is_empty());
    }

    #[test]
    fn test_project_manager_get_none() {
        let mgr = ProjectManager::new();
        assert!(mgr.get(42).is_none());
    }

    #[test]
    fn test_media_project_metadata() {
        let mut p = MediaProject::new(5, "Meta Project");
        p.metadata.insert("client".to_string(), "Acme".to_string());
        assert_eq!(p.metadata.get("client").map(|s| s.as_str()), Some("Acme"));
    }

    #[test]
    fn test_project_manager_multiple_active() {
        let mut mgr = ProjectManager::new();
        for i in 0..5u64 {
            let id = mgr.create(format!("Project {i}"));
            if i % 2 == 0 {
                if let Some(p) = mgr.get_mut(id) {
                    p.activate();
                }
            }
        }
        // Projects 0, 2, 4 are active
        assert_eq!(mgr.find_active().len(), 3);
    }
}
