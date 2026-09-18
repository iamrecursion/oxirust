//! Custom preset management.

use crate::{Preset, PresetError, Result};
use std::collections::HashMap;

/// Manager for custom user presets.
#[derive(Debug, Default)]
pub struct CustomPresetManager {
    presets: HashMap<String, Preset>,
}

impl CustomPresetManager {
    /// Create a new custom preset manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom preset.
    pub fn add(&mut self, preset: Preset) -> Result<()> {
        if self.presets.contains_key(&preset.metadata.id) {
            return Err(PresetError::Invalid(format!(
                "Preset with ID '{}' already exists",
                preset.metadata.id
            )));
        }
        self.presets.insert(preset.metadata.id.clone(), preset);
        Ok(())
    }

    /// Get a custom preset by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Preset> {
        self.presets.get(id)
    }

    /// Remove a custom preset.
    pub fn remove(&mut self, id: &str) -> Result<Preset> {
        self.presets
            .remove(id)
            .ok_or_else(|| PresetError::NotFound(id.to_string()))
    }

    /// List all custom preset IDs.
    #[must_use]
    pub fn list_ids(&self) -> Vec<String> {
        self.presets.keys().cloned().collect()
    }

    /// Count of custom presets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// Clear all custom presets.
    pub fn clear(&mut self) {
        self.presets.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PresetCategory, PresetMetadata};
    use oximedia_transcode::PresetConfig;

    #[test]
    fn test_manager() {
        let mut manager = CustomPresetManager::new();
        let metadata = PresetMetadata::new("test", "Test", PresetCategory::Custom);
        let config = PresetConfig::default();
        let preset = Preset::new(metadata, config);

        assert!(manager.add(preset).is_ok());
        assert_eq!(manager.count(), 1);
        assert!(manager.get("test").is_some());
    }
}
