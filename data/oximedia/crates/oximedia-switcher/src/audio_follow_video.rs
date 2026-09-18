//! Audio Follow Video (AFV) logic for production switcher.
//!
//! AFV automatically switches the audio mix to follow the current video source
//! selection, so operators do not need to manually adjust audio when cutting
//! between sources.

#![allow(dead_code)]

/// Operating mode for Audio Follow Video.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AfvMode {
    /// AFV is disabled; audio is independent of video routing
    Disabled,
    /// Normal AFV: audio follows program bus selection
    Normal,
    /// Split AFV: audio from preview follows separately
    Split,
}

impl AfvMode {
    /// Returns true if AFV is in an active (non-disabled) mode.
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Global configuration for Audio Follow Video behaviour.
#[derive(Debug, Clone)]
pub struct AfvConfig {
    /// Operating mode
    pub mode: AfvMode,
    /// Whether AFV applies to M/E rows
    pub follow_mix_effects: bool,
    /// Whether AFV applies to SuperSource inputs
    pub follow_supersources: bool,
}

impl AfvConfig {
    /// Return a default AFV configuration (Normal mode, follow M/E only).
    pub fn default() -> Self {
        Self {
            mode: AfvMode::Normal,
            follow_mix_effects: true,
            follow_supersources: false,
        }
    }
}

impl Default for AfvConfig {
    fn default() -> Self {
        AfvConfig::default()
    }
}

/// A single AFV mapping between a video input and an audio input.
#[derive(Debug, Clone, PartialEq)]
pub struct AfvMapping {
    /// Video input number
    pub video_input: u8,
    /// Audio input number that should follow this video input
    pub audio_input: u8,
    /// Whether this mapping is currently active
    pub enabled: bool,
}

impl AfvMapping {
    /// Create a new AFV mapping.
    pub fn new(video_input: u8, audio_input: u8, enabled: bool) -> Self {
        Self {
            video_input,
            audio_input,
            enabled,
        }
    }

    /// Returns true if this mapping is enabled.
    pub fn is_active(&self) -> bool {
        self.enabled
    }
}

/// Controller that manages AFV configuration and mappings.
#[derive(Debug, Clone)]
pub struct AfvController {
    /// Global AFV configuration
    pub config: AfvConfig,
    /// All registered video-to-audio mappings
    pub mappings: Vec<AfvMapping>,
}

impl AfvController {
    /// Create a new AFV controller with the given configuration.
    pub fn new(config: AfvConfig) -> Self {
        Self {
            config,
            mappings: Vec::new(),
        }
    }

    /// Register a new AFV mapping.
    pub fn add_mapping(&mut self, mapping: AfvMapping) {
        self.mappings.push(mapping);
    }

    /// Return the audio input that should follow the given video input.
    ///
    /// Returns `None` if no active mapping exists for this video input, or if
    /// AFV is disabled.
    pub fn audio_for_video(&self, video_input: u8) -> Option<u8> {
        if !self.config.mode.is_active() {
            return None;
        }
        self.mappings
            .iter()
            .find(|m| m.video_input == video_input && m.enabled)
            .map(|m| m.audio_input)
    }

    /// Return the number of currently active (enabled) mappings.
    pub fn active_mappings(&self) -> usize {
        self.mappings.iter().filter(|m| m.enabled).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AfvMode tests ---

    #[test]
    fn test_mode_normal_is_active() {
        assert!(AfvMode::Normal.is_active());
    }

    #[test]
    fn test_mode_split_is_active() {
        assert!(AfvMode::Split.is_active());
    }

    #[test]
    fn test_mode_disabled_is_not_active() {
        assert!(!AfvMode::Disabled.is_active());
    }

    // --- AfvConfig tests ---

    #[test]
    fn test_config_default_mode() {
        let cfg = AfvConfig::default();
        assert_eq!(cfg.mode, AfvMode::Normal);
    }

    #[test]
    fn test_config_default_follow_me() {
        let cfg = AfvConfig::default();
        assert!(cfg.follow_mix_effects);
    }

    #[test]
    fn test_config_default_no_supersource() {
        let cfg = AfvConfig::default();
        assert!(!cfg.follow_supersources);
    }

    // --- AfvMapping tests ---

    #[test]
    fn test_mapping_is_active_enabled() {
        let m = AfvMapping::new(1, 2, true);
        assert!(m.is_active());
    }

    #[test]
    fn test_mapping_is_active_disabled() {
        let m = AfvMapping::new(1, 2, false);
        assert!(!m.is_active());
    }

    #[test]
    fn test_mapping_fields() {
        let m = AfvMapping::new(3, 7, true);
        assert_eq!(m.video_input, 3);
        assert_eq!(m.audio_input, 7);
    }

    // --- AfvController tests ---

    #[test]
    fn test_controller_no_mappings_initially() {
        let ctrl = AfvController::new(AfvConfig::default());
        assert_eq!(ctrl.active_mappings(), 0);
    }

    #[test]
    fn test_controller_add_mapping_counted() {
        let mut ctrl = AfvController::new(AfvConfig::default());
        ctrl.add_mapping(AfvMapping::new(1, 1, true));
        ctrl.add_mapping(AfvMapping::new(2, 2, true));
        assert_eq!(ctrl.active_mappings(), 2);
    }

    #[test]
    fn test_controller_disabled_mapping_not_counted() {
        let mut ctrl = AfvController::new(AfvConfig::default());
        ctrl.add_mapping(AfvMapping::new(1, 1, false));
        assert_eq!(ctrl.active_mappings(), 0);
    }

    #[test]
    fn test_audio_for_video_found() {
        let mut ctrl = AfvController::new(AfvConfig::default());
        ctrl.add_mapping(AfvMapping::new(3, 5, true));
        assert_eq!(ctrl.audio_for_video(3), Some(5));
    }

    #[test]
    fn test_audio_for_video_not_found() {
        let ctrl = AfvController::new(AfvConfig::default());
        assert!(ctrl.audio_for_video(10).is_none());
    }

    #[test]
    fn test_audio_for_video_disabled_mode_returns_none() {
        let mut cfg = AfvConfig::default();
        cfg.mode = AfvMode::Disabled;
        let mut ctrl = AfvController::new(cfg);
        ctrl.add_mapping(AfvMapping::new(1, 1, true));
        // Even though mapping exists, AFV is disabled
        assert!(ctrl.audio_for_video(1).is_none());
    }

    #[test]
    fn test_audio_for_video_disabled_mapping_returns_none() {
        let mut ctrl = AfvController::new(AfvConfig::default());
        ctrl.add_mapping(AfvMapping::new(2, 4, false));
        assert!(ctrl.audio_for_video(2).is_none());
    }
}
