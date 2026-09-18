//! Preset management system.

use super::save::RoutingPreset;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Preset manager for storing and organizing routing presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetManager {
    /// All presets indexed by ID
    presets: HashMap<PresetId, RoutingPreset>,
    /// Next preset ID to assign
    next_id: u64,
    /// Current active preset
    active_preset: Option<PresetId>,
}

/// Unique identifier for a preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PresetId(u64);

impl PresetId {
    /// Create a new preset ID
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the inner ID value
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PresetManager {
    /// Create a new preset manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            presets: HashMap::new(),
            next_id: 1,
            active_preset: None,
        }
    }

    /// Add a new preset
    pub fn add_preset(&mut self, preset: RoutingPreset) -> PresetId {
        let id = PresetId::new(self.next_id);
        self.next_id += 1;
        self.presets.insert(id, preset);
        id
    }

    /// Remove a preset
    pub fn remove_preset(&mut self, id: PresetId) -> Option<RoutingPreset> {
        if self.active_preset == Some(id) {
            self.active_preset = None;
        }
        self.presets.remove(&id)
    }

    /// Get a preset by ID
    #[must_use]
    pub fn get_preset(&self, id: PresetId) -> Option<&RoutingPreset> {
        self.presets.get(&id)
    }

    /// Get a mutable reference to a preset
    pub fn get_preset_mut(&mut self, id: PresetId) -> Option<&mut RoutingPreset> {
        self.presets.get_mut(&id)
    }

    /// Find presets by name
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Vec<(PresetId, &RoutingPreset)> {
        self.presets
            .iter()
            .filter(|(_, preset)| preset.name == name)
            .map(|(&id, preset)| (id, preset))
            .collect()
    }

    /// Find presets by tag
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<(PresetId, &RoutingPreset)> {
        self.presets
            .iter()
            .filter(|(_, preset)| preset.has_tag(tag))
            .map(|(&id, preset)| (id, preset))
            .collect()
    }

    /// Get all presets
    #[must_use]
    pub fn all_presets(&self) -> Vec<(PresetId, &RoutingPreset)> {
        self.presets
            .iter()
            .map(|(&id, preset)| (id, preset))
            .collect()
    }

    /// Get preset count
    #[must_use]
    pub fn preset_count(&self) -> usize {
        self.presets.len()
    }

    /// Set active preset
    pub fn set_active(&mut self, id: PresetId) -> Result<(), PresetError> {
        if self.presets.contains_key(&id) {
            self.active_preset = Some(id);
            Ok(())
        } else {
            Err(PresetError::PresetNotFound(id))
        }
    }

    /// Get active preset ID
    #[must_use]
    pub const fn active_preset_id(&self) -> Option<PresetId> {
        self.active_preset
    }

    /// Get active preset
    #[must_use]
    pub fn active_preset(&self) -> Option<&RoutingPreset> {
        self.active_preset.and_then(|id| self.presets.get(&id))
    }

    /// Clear active preset
    pub fn clear_active(&mut self) {
        self.active_preset = None;
    }

    /// Duplicate a preset
    pub fn duplicate_preset(&mut self, id: PresetId) -> Result<PresetId, PresetError> {
        if let Some(preset) = self.presets.get(&id) {
            let mut new_preset = preset.clone();
            new_preset.name = format!("{} (Copy)", preset.name);
            Ok(self.add_preset(new_preset))
        } else {
            Err(PresetError::PresetNotFound(id))
        }
    }
}

/// Errors that can occur in preset operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum PresetError {
    /// Preset not found
    #[error("Preset not found: {0:?}")]
    PresetNotFound(PresetId),
    /// Preset name already exists
    #[error("Preset name already exists: {0}")]
    NameExists(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = PresetManager::new();
        assert_eq!(manager.preset_count(), 0);
        assert!(manager.active_preset_id().is_none());
    }

    #[test]
    fn test_add_preset() {
        let mut manager = PresetManager::new();
        let preset = RoutingPreset::new("Test".to_string(), "Test preset".to_string());

        let id = manager.add_preset(preset);
        assert_eq!(manager.preset_count(), 1);
        assert!(manager.get_preset(id).is_some());
    }

    #[test]
    fn test_remove_preset() {
        let mut manager = PresetManager::new();
        let preset = RoutingPreset::new("Test".to_string(), "Test".to_string());
        let id = manager.add_preset(preset);

        let removed = manager.remove_preset(id);
        assert!(removed.is_some());
        assert_eq!(manager.preset_count(), 0);
    }

    #[test]
    fn test_find_by_name() {
        let mut manager = PresetManager::new();

        let preset1 = RoutingPreset::new("Live Show".to_string(), "Description".to_string());
        let preset2 = RoutingPreset::new("Studio".to_string(), "Description".to_string());

        manager.add_preset(preset1);
        manager.add_preset(preset2);

        let found = manager.find_by_name("Live Show");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_find_by_tag() {
        let mut manager = PresetManager::new();

        let mut preset = RoutingPreset::new("Test".to_string(), "Test".to_string());
        preset.add_tag("broadcast".to_string());

        manager.add_preset(preset);

        let found = manager.find_by_tag("broadcast");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_active_preset() {
        let mut manager = PresetManager::new();
        let preset = RoutingPreset::new("Test".to_string(), "Test".to_string());
        let id = manager.add_preset(preset);

        manager.set_active(id).expect("should succeed in test");
        assert_eq!(manager.active_preset_id(), Some(id));
        assert!(manager.active_preset().is_some());
    }

    #[test]
    fn test_clear_active() {
        let mut manager = PresetManager::new();
        let preset = RoutingPreset::new("Test".to_string(), "Test".to_string());
        let id = manager.add_preset(preset);

        manager.set_active(id).expect("should succeed in test");
        manager.clear_active();
        assert!(manager.active_preset_id().is_none());
    }

    #[test]
    fn test_duplicate_preset() {
        let mut manager = PresetManager::new();
        let preset = RoutingPreset::new("Original".to_string(), "Test".to_string());
        let id = manager.add_preset(preset);

        let new_id = manager
            .duplicate_preset(id)
            .expect("should succeed in test");
        assert_eq!(manager.preset_count(), 2);

        let new_preset = manager.get_preset(new_id).expect("should succeed in test");
        assert!(new_preset.name.contains("Copy"));
    }

    #[test]
    fn test_set_invalid_active() {
        let mut manager = PresetManager::new();
        let invalid_id = PresetId::new(999);

        assert!(matches!(
            manager.set_active(invalid_id),
            Err(PresetError::PresetNotFound(_))
        ));
    }
}
