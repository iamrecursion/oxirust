#![allow(dead_code)]
//! Extended stabilization configuration and presets.
//!
//! This module provides a high-level configuration builder with named presets
//! for common stabilization scenarios. It complements the base `StabilizeConfig`
//! in the crate root by offering scenario-specific defaults.

/// Stabilization intent / mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizeMode {
    /// Lock-on: keep a fixed point in frame (tripod emulation).
    LockOn,
    /// Smooth: reduce jitter while preserving intentional motion.
    Smooth,
    /// Follow: smooth follow-cam that preserves panning.
    Follow,
    /// Cinematic: aggressive smoothing with controlled crop.
    Cinematic,
}

impl Default for StabilizeMode {
    fn default() -> Self {
        Self::Smooth
    }
}

impl StabilizeMode {
    /// Suggested smoothing radius for this mode (in frames).
    #[must_use]
    pub const fn suggested_radius(self) -> usize {
        match self {
            Self::LockOn => 60,
            Self::Smooth => 30,
            Self::Follow => 15,
            Self::Cinematic => 45,
        }
    }

    /// Suggested crop limit `[0, 1]` for this mode.
    #[must_use]
    pub fn suggested_crop(self) -> f64 {
        match self {
            Self::LockOn => 0.80,
            Self::Smooth => 0.90,
            Self::Follow => 0.95,
            Self::Cinematic => 0.85,
        }
    }
}

/// Full stabilization configuration including transform model, smoothing,
/// crop, and post-processing toggles.
#[derive(Debug, Clone)]
pub struct StabilizeConfigExt {
    /// Stabilization mode / intent.
    pub mode: StabilizeMode,
    /// Smoothing radius in frames.
    pub smoothing_radius: usize,
    /// Smoothing strength `[0, 1]`.
    pub smoothing_strength: f64,
    /// Maximum allowed crop ratio `[0, 1]`. Values closer to 0 allow more crop.
    pub max_crop: f64,
    /// Enable rolling-shutter correction.
    pub rolling_shutter: bool,
    /// Enable horizon leveling.
    pub horizon_level: bool,
    /// Target output width (0 = same as input).
    pub output_width: usize,
    /// Target output height (0 = same as input).
    pub output_height: usize,
    /// Enable adaptive smoothing based on scene activity.
    pub adaptive: bool,
}

impl Default for StabilizeConfigExt {
    fn default() -> Self {
        let mode = StabilizeMode::default();
        Self {
            mode,
            smoothing_radius: mode.suggested_radius(),
            smoothing_strength: 0.8,
            max_crop: mode.suggested_crop(),
            rolling_shutter: false,
            horizon_level: false,
            output_width: 0,
            output_height: 0,
            adaptive: false,
        }
    }
}

impl StabilizeConfigExt {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the stabilization mode and apply its suggested defaults.
    #[must_use]
    pub fn with_mode(mut self, mode: StabilizeMode) -> Self {
        self.mode = mode;
        self.smoothing_radius = mode.suggested_radius();
        self.max_crop = mode.suggested_crop();
        self
    }

    /// Override smoothing radius.
    #[must_use]
    pub const fn with_smoothing_radius(mut self, radius: usize) -> Self {
        self.smoothing_radius = radius;
        self
    }

    /// Override smoothing strength.
    #[must_use]
    pub fn with_smoothing_strength(mut self, strength: f64) -> Self {
        self.smoothing_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Override maximum crop.
    #[must_use]
    pub fn with_max_crop(mut self, crop: f64) -> Self {
        self.max_crop = crop.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable rolling-shutter correction.
    #[must_use]
    pub const fn with_rolling_shutter(mut self, enable: bool) -> Self {
        self.rolling_shutter = enable;
        self
    }

    /// Enable or disable horizon leveling.
    #[must_use]
    pub const fn with_horizon_level(mut self, enable: bool) -> Self {
        self.horizon_level = enable;
        self
    }

    /// Set target output dimensions.
    #[must_use]
    pub const fn with_output_size(mut self, width: usize, height: usize) -> Self {
        self.output_width = width;
        self.output_height = height;
        self
    }

    /// Enable adaptive smoothing.
    #[must_use]
    pub const fn with_adaptive(mut self, enable: bool) -> Self {
        self.adaptive = enable;
        self
    }

    /// Validate the configuration. Returns a list of issues (empty = valid).
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.smoothing_radius == 0 {
            issues.push("smoothing_radius must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.smoothing_strength) {
            issues.push(format!(
                "smoothing_strength out of range: {}",
                self.smoothing_strength
            ));
        }
        if !(0.0..=1.0).contains(&self.max_crop) {
            issues.push(format!("max_crop out of range: {}", self.max_crop));
        }
        issues
    }

    /// Whether this configuration is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

/// Named presets for common stabilization scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigPreset {
    /// Hand-held footage, moderate jitter.
    Handheld,
    /// Action camera (GoPro, etc.) with strong jitter.
    ActionCam,
    /// Drone footage with gentle drift.
    Drone,
    /// Tripod with slight vibrations.
    Tripod,
    /// Automotive dash-cam.
    Dashcam,
}

impl ConfigPreset {
    /// Build a `StabilizeConfigExt` for this preset.
    #[must_use]
    pub fn to_config(self) -> StabilizeConfigExt {
        match self {
            Self::Handheld => StabilizeConfigExt::new()
                .with_mode(StabilizeMode::Smooth)
                .with_smoothing_strength(0.75),
            Self::ActionCam => StabilizeConfigExt::new()
                .with_mode(StabilizeMode::Smooth)
                .with_smoothing_strength(0.9)
                .with_rolling_shutter(true)
                .with_adaptive(true),
            Self::Drone => StabilizeConfigExt::new()
                .with_mode(StabilizeMode::Follow)
                .with_smoothing_strength(0.6)
                .with_horizon_level(true),
            Self::Tripod => StabilizeConfigExt::new()
                .with_mode(StabilizeMode::LockOn)
                .with_smoothing_strength(0.95),
            Self::Dashcam => StabilizeConfigExt::new()
                .with_mode(StabilizeMode::Cinematic)
                .with_smoothing_strength(0.85)
                .with_rolling_shutter(true),
        }
    }

    /// Short description of the preset.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Handheld => "General-purpose hand-held footage stabilization",
            Self::ActionCam => "Action camera with aggressive jitter reduction",
            Self::Drone => "Aerial/drone footage with horizon correction",
            Self::Tripod => "Tripod shot with vibration removal",
            Self::Dashcam => "Automotive dashboard camera stabilization",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilize_mode_default() {
        assert_eq!(StabilizeMode::default(), StabilizeMode::Smooth);
    }

    #[test]
    fn test_suggested_radius() {
        assert_eq!(StabilizeMode::LockOn.suggested_radius(), 60);
        assert_eq!(StabilizeMode::Smooth.suggested_radius(), 30);
        assert_eq!(StabilizeMode::Follow.suggested_radius(), 15);
        assert_eq!(StabilizeMode::Cinematic.suggested_radius(), 45);
    }

    #[test]
    fn test_suggested_crop() {
        assert!((StabilizeMode::LockOn.suggested_crop() - 0.80).abs() < f64::EPSILON);
        assert!((StabilizeMode::Follow.suggested_crop() - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_ext_default() {
        let cfg = StabilizeConfigExt::new();
        assert_eq!(cfg.mode, StabilizeMode::Smooth);
        assert_eq!(cfg.smoothing_radius, 30);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_config_ext_builder() {
        let cfg = StabilizeConfigExt::new()
            .with_mode(StabilizeMode::LockOn)
            .with_smoothing_strength(0.95)
            .with_max_crop(0.8)
            .with_rolling_shutter(true)
            .with_horizon_level(true)
            .with_output_size(1920, 1080)
            .with_adaptive(true);
        assert_eq!(cfg.mode, StabilizeMode::LockOn);
        assert!((cfg.smoothing_strength - 0.95).abs() < f64::EPSILON);
        assert!(cfg.rolling_shutter);
        assert!(cfg.horizon_level);
        assert_eq!(cfg.output_width, 1920);
        assert!(cfg.adaptive);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_config_ext_clamps_strength() {
        let cfg = StabilizeConfigExt::new().with_smoothing_strength(5.0);
        assert!((cfg.smoothing_strength - 1.0).abs() < f64::EPSILON);
        let cfg2 = StabilizeConfigExt::new().with_smoothing_strength(-1.0);
        assert!((cfg2.smoothing_strength).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_ext_clamps_crop() {
        let cfg = StabilizeConfigExt::new().with_max_crop(2.0);
        assert!((cfg.max_crop - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_bad_radius() {
        let mut cfg = StabilizeConfigExt::new();
        cfg.smoothing_radius = 0;
        assert!(!cfg.is_valid());
        let issues = cfg.validate();
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn test_validate_bad_strength() {
        let mut cfg = StabilizeConfigExt::new();
        cfg.smoothing_strength = 1.5;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_preset_handheld() {
        let cfg = ConfigPreset::Handheld.to_config();
        assert_eq!(cfg.mode, StabilizeMode::Smooth);
        assert!((cfg.smoothing_strength - 0.75).abs() < f64::EPSILON);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_preset_action_cam() {
        let cfg = ConfigPreset::ActionCam.to_config();
        assert!(cfg.rolling_shutter);
        assert!(cfg.adaptive);
    }

    #[test]
    fn test_preset_drone() {
        let cfg = ConfigPreset::Drone.to_config();
        assert!(cfg.horizon_level);
        assert_eq!(cfg.mode, StabilizeMode::Follow);
    }

    #[test]
    fn test_preset_tripod() {
        let cfg = ConfigPreset::Tripod.to_config();
        assert_eq!(cfg.mode, StabilizeMode::LockOn);
    }

    #[test]
    fn test_preset_dashcam() {
        let cfg = ConfigPreset::Dashcam.to_config();
        assert_eq!(cfg.mode, StabilizeMode::Cinematic);
        assert!(cfg.rolling_shutter);
    }

    #[test]
    fn test_preset_descriptions() {
        assert!(!ConfigPreset::Handheld.description().is_empty());
        assert!(!ConfigPreset::ActionCam.description().is_empty());
        assert!(!ConfigPreset::Drone.description().is_empty());
        assert!(!ConfigPreset::Tripod.description().is_empty());
        assert!(!ConfigPreset::Dashcam.description().is_empty());
    }

    #[test]
    fn test_mode_with_sets_defaults() {
        let cfg = StabilizeConfigExt::new().with_mode(StabilizeMode::Cinematic);
        assert_eq!(cfg.smoothing_radius, 45);
        assert!((cfg.max_crop - 0.85).abs() < f64::EPSILON);
    }
}
