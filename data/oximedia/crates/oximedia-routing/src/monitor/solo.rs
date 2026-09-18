//! Solo management for monitoring systems.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Solo mode behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SoloMode {
    /// Solo-In-Place (mutes non-soloed channels)
    #[default]
    InPlace,
    /// After-Fader Listen (routes to monitor bus)
    AfterFader,
    /// Pre-Fader Listen (routes to monitor bus pre-fader)
    PreFader,
}

/// Solo management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoloManager {
    /// Channels currently soloed
    pub soloed_channels: HashSet<usize>,
    /// Solo mode
    pub mode: SoloMode,
    /// Solo level adjustment in dB
    pub solo_level_db: f32,
    /// Dim level for non-soloed channels (in SIP mode)
    pub dim_level_db: f32,
}

impl Default for SoloManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SoloManager {
    /// Create a new solo manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            soloed_channels: HashSet::new(),
            mode: SoloMode::InPlace,
            solo_level_db: 0.0,
            dim_level_db: f32::NEG_INFINITY,
        }
    }

    /// Solo a channel
    pub fn solo(&mut self, channel: usize) {
        self.soloed_channels.insert(channel);
    }

    /// Unsolo a channel
    pub fn unsolo(&mut self, channel: usize) {
        self.soloed_channels.remove(&channel);
    }

    /// Toggle solo for a channel
    pub fn toggle_solo(&mut self, channel: usize) {
        if self.soloed_channels.contains(&channel) {
            self.soloed_channels.remove(&channel);
        } else {
            self.soloed_channels.insert(channel);
        }
    }

    /// Clear all solos
    pub fn clear_all(&mut self) {
        self.soloed_channels.clear();
    }

    /// Check if a channel is soloed
    #[must_use]
    pub fn is_soloed(&self, channel: usize) -> bool {
        self.soloed_channels.contains(&channel)
    }

    /// Get number of soloed channels
    #[must_use]
    pub fn solo_count(&self) -> usize {
        self.soloed_channels.len()
    }

    /// Check if any channel is soloed
    #[must_use]
    pub fn has_solo(&self) -> bool {
        !self.soloed_channels.is_empty()
    }

    /// Check if a channel should be audible (based on solo state)
    #[must_use]
    pub fn is_audible(&self, channel: usize) -> bool {
        if self.has_solo() {
            self.is_soloed(channel)
        } else {
            true // No solos active, all audible
        }
    }

    /// Get effective level for a channel (considering solo state)
    #[must_use]
    pub fn effective_level_db(&self, channel: usize) -> f32 {
        if !self.has_solo() {
            0.0 // No solos active
        } else if self.is_soloed(channel) {
            self.solo_level_db
        } else {
            match self.mode {
                SoloMode::InPlace => self.dim_level_db,
                _ => 0.0, // AFL/PFL modes don't affect main mix
            }
        }
    }

    /// Set solo mode
    pub fn set_mode(&mut self, mode: SoloMode) {
        self.mode = mode;
    }

    /// Set solo level
    pub fn set_solo_level(&mut self, level_db: f32) {
        self.solo_level_db = level_db.clamp(-60.0, 12.0);
    }

    /// Set dim level for non-soloed channels
    pub fn set_dim_level(&mut self, dim_db: f32) {
        self.dim_level_db = dim_db;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solo_manager_creation() {
        let manager = SoloManager::new();
        assert_eq!(manager.solo_count(), 0);
        assert!(!manager.has_solo());
    }

    #[test]
    fn test_solo_channel() {
        let mut manager = SoloManager::new();
        manager.solo(0);
        assert!(manager.is_soloed(0));
        assert_eq!(manager.solo_count(), 1);
    }

    #[test]
    fn test_unsolo_channel() {
        let mut manager = SoloManager::new();
        manager.solo(0);
        manager.unsolo(0);
        assert!(!manager.is_soloed(0));
        assert_eq!(manager.solo_count(), 0);
    }

    #[test]
    fn test_toggle_solo() {
        let mut manager = SoloManager::new();
        manager.toggle_solo(0);
        assert!(manager.is_soloed(0));

        manager.toggle_solo(0);
        assert!(!manager.is_soloed(0));
    }

    #[test]
    fn test_is_audible() {
        let mut manager = SoloManager::new();

        // No solos - all audible
        assert!(manager.is_audible(0));
        assert!(manager.is_audible(1));

        // Solo channel 0
        manager.solo(0);
        assert!(manager.is_audible(0));
        assert!(!manager.is_audible(1));
    }

    #[test]
    fn test_effective_level() {
        let mut manager = SoloManager::new();
        manager.set_solo_level(3.0);

        // No solos
        assert!(manager.effective_level_db(0).abs() < f32::EPSILON);

        // Solo channel 0
        manager.solo(0);
        assert!((manager.effective_level_db(0) - 3.0).abs() < f32::EPSILON);
        assert!(
            manager.effective_level_db(1).is_infinite()
                && manager.effective_level_db(1).is_sign_negative()
        );
    }

    #[test]
    fn test_clear_all() {
        let mut manager = SoloManager::new();
        manager.solo(0);
        manager.solo(1);
        manager.solo(2);

        manager.clear_all();
        assert_eq!(manager.solo_count(), 0);
    }

    #[test]
    fn test_solo_modes() {
        let mut manager = SoloManager::new();
        assert_eq!(manager.mode, SoloMode::InPlace);

        manager.set_mode(SoloMode::AfterFader);
        assert_eq!(manager.mode, SoloMode::AfterFader);
    }

    #[test]
    fn test_multiple_solos() {
        let mut manager = SoloManager::new();
        manager.solo(0);
        manager.solo(2);
        manager.solo(4);

        assert_eq!(manager.solo_count(), 3);
        assert!(manager.is_soloed(0));
        assert!(!manager.is_soloed(1));
        assert!(manager.is_soloed(2));
    }
}
