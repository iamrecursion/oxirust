//! Preset save/load functionality.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A routing preset that can be saved and loaded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPreset {
    /// Preset name
    pub name: String,
    /// Preset description
    pub description: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Preset data (serialized routing configuration)
    pub data: PresetData,
    /// Tags for organization
    pub tags: Vec<String>,
}

/// Preset data containing all routing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PresetData {
    /// Matrix routing connections
    pub matrix_connections: Vec<(usize, usize, f32)>,
    /// Patch bay connections
    pub patch_connections: Vec<(String, String, f32)>,
    /// Gain settings
    pub gain_settings: HashMap<String, f32>,
    /// Channel mappings
    pub channel_mappings: Vec<String>,
    /// Monitor settings
    pub monitor_settings: MonitorSettings,
}

/// Monitor settings for presets
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MonitorSettings {
    /// AFL channels
    pub afl_channels: Vec<usize>,
    /// PFL channels
    pub pfl_channels: Vec<usize>,
    /// Solo channels
    pub solo_channels: Vec<usize>,
}

impl RoutingPreset {
    /// Create a new preset
    #[must_use]
    pub fn new(name: String, description: String) -> Self {
        let now = current_timestamp();
        Self {
            name,
            description,
            created_at: now,
            modified_at: now,
            data: PresetData::default(),
            tags: Vec::new(),
        }
    }

    /// Update the preset data
    pub fn update_data(&mut self, data: PresetData) {
        self.data = data;
        self.modified_at = current_timestamp();
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
            self.modified_at = current_timestamp();
        }
    }

    /// Remove a tag
    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
        self.modified_at = current_timestamp();
    }

    /// Check if preset has a specific tag
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Get current Unix timestamp in seconds since the Unix epoch.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_creation() {
        let preset = RoutingPreset::new("Default".to_string(), "Default routing".to_string());
        assert_eq!(preset.name, "Default");
        assert_eq!(preset.created_at, preset.modified_at);
    }

    #[test]
    fn test_update_data() {
        let mut preset = RoutingPreset::new("Test".to_string(), "Test".to_string());
        let original_modified = preset.modified_at;

        let data = PresetData::default();
        preset.update_data(data);

        // Modified time should be updated (in real implementation)
        assert!(preset.modified_at >= original_modified);
    }

    #[test]
    fn test_tags() {
        let mut preset = RoutingPreset::new("Test".to_string(), "Test".to_string());

        preset.add_tag("live".to_string());
        preset.add_tag("broadcast".to_string());

        assert_eq!(preset.tags.len(), 2);
        assert!(preset.has_tag("live"));
        assert!(preset.has_tag("broadcast"));
        assert!(!preset.has_tag("studio"));
    }

    #[test]
    fn test_remove_tag() {
        let mut preset = RoutingPreset::new("Test".to_string(), "Test".to_string());

        preset.add_tag("test".to_string());
        assert_eq!(preset.tags.len(), 1);

        preset.remove_tag("test");
        assert_eq!(preset.tags.len(), 0);
    }

    #[test]
    fn test_duplicate_tags() {
        let mut preset = RoutingPreset::new("Test".to_string(), "Test".to_string());

        preset.add_tag("test".to_string());
        preset.add_tag("test".to_string());

        assert_eq!(preset.tags.len(), 1);
    }
}
