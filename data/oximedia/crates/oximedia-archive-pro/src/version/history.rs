//! Version history tracking

use super::VersionInfo;
use serde::{Deserialize, Serialize};

/// History entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Version information
    pub version: VersionInfo,
    /// Action performed
    pub action: String,
    /// User who performed the action
    pub user: Option<String>,
}

/// Version history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// History entries
    pub entries: Vec<HistoryEntry>,
}

impl Default for VersionHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl VersionHistory {
    /// Create a new version history
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a history entry
    pub fn add_entry(&mut self, version: VersionInfo, action: &str, user: Option<String>) {
        self.entries.push(HistoryEntry {
            version,
            action: action.to_string(),
            user,
        });
    }

    /// Get entries for a specific version
    #[must_use]
    pub fn get_version_entries(&self, version: u32) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.version.version == version)
            .collect()
    }

    /// Get the latest entry
    #[must_use]
    pub fn latest(&self) -> Option<&HistoryEntry> {
        self.entries.last()
    }

    /// Generate a changelog
    #[must_use]
    pub fn generate_changelog(&self) -> String {
        let mut changelog = String::from("Version History\n===============\n\n");

        for entry in self.entries.iter().rev() {
            changelog.push_str(&format!(
                "Version {} - {} - {}\n",
                entry.version.version,
                entry.version.timestamp.format("%Y-%m-%d %H:%M:%S"),
                entry.action
            ));

            if let Some(ref comment) = entry.version.comment {
                changelog.push_str(&format!("  Comment: {comment}\n"));
            }
            if let Some(ref user) = entry.user {
                changelog.push_str(&format!("  User: {user}\n"));
            }
            changelog.push_str(&format!("  Size: {} bytes\n", entry.version.size));
            changelog.push('\n');
        }

        changelog
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_version_history() {
        let mut history = VersionHistory::new();

        let version = VersionInfo {
            version: 1,
            path: PathBuf::from("test.mkv"),
            checksum: "abc123".to_string(),
            timestamp: chrono::Utc::now(),
            comment: Some("Initial version".to_string()),
            size: 1024,
        };

        history.add_entry(version, "created", Some("user1".to_string()));
        assert_eq!(history.entries.len(), 1);

        let changelog = history.generate_changelog();
        assert!(changelog.contains("Version 1"));
        assert!(changelog.contains("Initial version"));
    }
}
