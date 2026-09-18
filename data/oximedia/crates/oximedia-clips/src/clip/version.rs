//! Clip versioning support.

use super::ClipId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a clip version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VersionId(Uuid);

impl VersionId {
    /// Creates a new random version ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a version ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for VersionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for VersionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A version of a clip (e.g., different color grades, edits).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipVersion {
    /// Unique identifier.
    pub id: VersionId,

    /// Parent clip ID.
    pub clip_id: ClipId,

    /// Version number.
    pub version_number: u32,

    /// Version name.
    pub name: String,

    /// Description of changes.
    pub description: Option<String>,

    /// Created timestamp.
    pub created_at: DateTime<Utc>,

    /// Created by user.
    pub created_by: Option<String>,
}

impl ClipVersion {
    /// Creates a new clip version.
    #[must_use]
    pub fn new(clip_id: ClipId, version_number: u32, name: impl Into<String>) -> Self {
        Self {
            id: VersionId::new(),
            clip_id,
            version_number,
            name: name.into(),
            description: None,
            created_at: Utc::now(),
            created_by: None,
        }
    }

    /// Sets the description.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
    }

    /// Sets the creator.
    pub fn set_created_by(&mut self, user: impl Into<String>) {
        self.created_by = Some(user.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let clip_id = ClipId::new();
        let version = ClipVersion::new(clip_id, 1, "Color Grade v1");
        assert_eq!(version.version_number, 1);
        assert_eq!(version.name, "Color Grade v1");
    }
}
