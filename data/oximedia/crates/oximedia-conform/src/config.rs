//! Configuration for media conforming.

use crate::types::FrameRate;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for a conform session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformConfig {
    /// Pre-roll frames (handles before edit point).
    pub pre_roll_frames: u64,

    /// Post-roll frames (handles after edit point).
    pub post_roll_frames: u64,

    /// Minimum match score threshold (0.0 - 1.0).
    pub match_threshold: f64,

    /// Enable fuzzy filename matching.
    pub fuzzy_matching: bool,

    /// Maximum Levenshtein distance for fuzzy matching.
    pub fuzzy_max_distance: usize,

    /// Enable timecode matching.
    pub timecode_matching: bool,

    /// Enable content hash matching.
    pub content_hash_matching: bool,

    /// Enable duration matching.
    pub duration_matching: bool,

    /// Duration tolerance in seconds.
    pub duration_tolerance: f64,

    /// Verify checksums during matching.
    pub verify_checksums: bool,

    /// Default frame rate if not specified.
    pub default_fps: FrameRate,

    /// Parallel processing thread count (0 = auto).
    pub parallel_threads: usize,

    /// Cache directory for temporary files.
    pub cache_dir: Option<PathBuf>,

    /// Enable strict validation.
    pub strict_validation: bool,

    /// Allow missing handles.
    pub allow_missing_handles: bool,

    /// Auto-relink to high-resolution sources.
    pub auto_relink: bool,

    /// Proxy filename patterns to high-res patterns.
    pub proxy_patterns: Vec<(String, String)>,
}

impl ConformConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a strict configuration.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            pre_roll_frames: 0,
            post_roll_frames: 0,
            match_threshold: 0.95,
            fuzzy_matching: false,
            fuzzy_max_distance: 2,
            timecode_matching: true,
            content_hash_matching: true,
            duration_matching: true,
            duration_tolerance: 0.1,
            verify_checksums: true,
            default_fps: FrameRate::Fps25,
            parallel_threads: 0,
            cache_dir: None,
            strict_validation: true,
            allow_missing_handles: false,
            auto_relink: false,
            proxy_patterns: Vec::new(),
        }
    }

    /// Create a lenient configuration.
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            pre_roll_frames: 0,
            post_roll_frames: 0,
            match_threshold: 0.6,
            fuzzy_matching: true,
            fuzzy_max_distance: 5,
            timecode_matching: true,
            content_hash_matching: false,
            duration_matching: true,
            duration_tolerance: 1.0,
            verify_checksums: false,
            default_fps: FrameRate::Fps25,
            parallel_threads: 0,
            cache_dir: None,
            strict_validation: false,
            allow_missing_handles: true,
            auto_relink: false,
            proxy_patterns: Vec::new(),
        }
    }

    /// Set pre-roll frames.
    pub fn with_pre_roll(&mut self, frames: u64) -> &mut Self {
        self.pre_roll_frames = frames;
        self
    }

    /// Set post-roll frames.
    pub fn with_post_roll(&mut self, frames: u64) -> &mut Self {
        self.post_roll_frames = frames;
        self
    }

    /// Set match threshold.
    pub fn with_match_threshold(&mut self, threshold: f64) -> &mut Self {
        self.match_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable fuzzy matching.
    pub fn with_fuzzy_matching(&mut self, enabled: bool) -> &mut Self {
        self.fuzzy_matching = enabled;
        self
    }

    /// Set default frame rate.
    pub fn with_default_fps(&mut self, fps: FrameRate) -> &mut Self {
        self.default_fps = fps;
        self
    }

    /// Add a proxy pattern mapping.
    pub fn add_proxy_pattern(&mut self, from: String, to: String) -> &mut Self {
        self.proxy_patterns.push((from, to));
        self
    }
}

impl Default for ConformConfig {
    fn default() -> Self {
        Self {
            pre_roll_frames: 10,
            post_roll_frames: 10,
            match_threshold: 0.8,
            fuzzy_matching: true,
            fuzzy_max_distance: 3,
            timecode_matching: true,
            content_hash_matching: false,
            duration_matching: true,
            duration_tolerance: 0.5,
            verify_checksums: false,
            default_fps: FrameRate::Fps25,
            parallel_threads: 0,
            cache_dir: None,
            strict_validation: false,
            allow_missing_handles: false,
            auto_relink: false,
            proxy_patterns: vec![
                ("_proxy".to_string(), String::new()),
                ("_low".to_string(), "_high".to_string()),
                ("_offline".to_string(), "_online".to_string()),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ConformConfig::default();
        assert_eq!(config.pre_roll_frames, 10);
        assert_eq!(config.post_roll_frames, 10);
        assert!((config.match_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_strict_config() {
        let config = ConformConfig::strict();
        assert!((config.match_threshold - 0.95).abs() < f64::EPSILON);
        assert!(!config.fuzzy_matching);
        assert!(config.strict_validation);
    }

    #[test]
    fn test_lenient_config() {
        let config = ConformConfig::lenient();
        assert!((config.match_threshold - 0.6).abs() < f64::EPSILON);
        assert!(config.fuzzy_matching);
        assert!(!config.strict_validation);
    }

    #[test]
    fn test_config_builder() {
        let mut config = ConformConfig::new();
        config.with_pre_roll(20).with_post_roll(30);
        assert_eq!(config.pre_roll_frames, 20);
        assert_eq!(config.post_roll_frames, 30);
    }

    #[test]
    fn test_match_threshold_clamping() {
        let mut config = ConformConfig::new();
        config.with_match_threshold(1.5);
        assert!((config.match_threshold - 1.0).abs() < f64::EPSILON);

        config.with_match_threshold(-0.5);
        assert!(config.match_threshold.abs() < f64::EPSILON);
    }
}
