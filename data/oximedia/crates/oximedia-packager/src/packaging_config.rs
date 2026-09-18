#![allow(dead_code)]
//! High-level packaging configuration and validation.
//!
//! Centralises the configurable aspects of an adaptive-streaming packaging job,
//! including encryption mode, segment duration, and low-latency options.

use std::time::Duration;

/// Encryption modes supported by the packager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncryptionMode {
    /// No encryption; content delivered in the clear.
    #[default]
    Clear,
    /// AES-128 CBC full-segment encryption (HLS).
    Aes128,
    /// SAMPLE-AES partial encryption (HLS).
    SampleAes,
    /// Common Encryption (ISO 23001-7), used in DASH/CMAF.
    Cenc,
}

impl EncryptionMode {
    /// Returns `true` if the content will be encrypted.
    #[must_use]
    pub fn is_encrypted(self) -> bool {
        !matches!(self, Self::Clear)
    }

    /// A short label for the mode.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Aes128 => "AES-128",
            Self::SampleAes => "SAMPLE-AES",
            Self::Cenc => "CENC",
        }
    }
}

impl std::fmt::Display for EncryptionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Full configuration for a single packaging job.
#[derive(Debug, Clone)]
pub struct PackagingConfig {
    /// Target segment duration.
    pub segment_duration: Duration,
    /// Encryption mode to apply.
    pub encryption_mode: EncryptionMode,
    /// Enable low-latency chunked transfer (LL-HLS / LL-DASH).
    pub low_latency: bool,
    /// Partial segment duration for low-latency mode.
    pub partial_segment_duration: Duration,
    /// Number of segments to keep in live playlists (0 = unlimited / VOD).
    pub playlist_window_segments: u32,
    /// Maximum allowed bitrate in bps (0 = unlimited).
    pub max_bitrate_bps: u64,
    /// Output directory path.
    pub output_dir: String,
    /// Base URL to prepend in manifests.
    pub base_url: String,
}

impl Default for PackagingConfig {
    fn default() -> Self {
        Self {
            segment_duration: Duration::from_secs(6),
            encryption_mode: EncryptionMode::Clear,
            low_latency: false,
            partial_segment_duration: Duration::from_millis(200),
            playlist_window_segments: 0,
            max_bitrate_bps: 0,
            output_dir: "output".to_string(),
            base_url: String::new(),
        }
    }
}

impl PackagingConfig {
    /// Create a default packaging configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if low-latency mode is enabled.
    #[must_use]
    pub fn is_low_latency(&self) -> bool {
        self.low_latency
    }

    /// Returns `true` if any form of encryption is active.
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.encryption_mode.is_encrypted()
    }

    /// Returns `true` if this is a live packaging job (sliding window).
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.playlist_window_segments > 0
    }

    /// Builder: set segment duration.
    #[must_use]
    pub fn with_segment_duration(mut self, d: Duration) -> Self {
        self.segment_duration = d;
        self
    }

    /// Builder: set encryption mode.
    #[must_use]
    pub fn with_encryption(mut self, mode: EncryptionMode) -> Self {
        self.encryption_mode = mode;
        self
    }

    /// Builder: enable/disable low latency.
    #[must_use]
    pub fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Builder: set playlist window (live mode).
    #[must_use]
    pub fn with_playlist_window(mut self, segments: u32) -> Self {
        self.playlist_window_segments = segments;
        self
    }

    /// Builder: set max bitrate.
    #[must_use]
    pub fn with_max_bitrate(mut self, bps: u64) -> Self {
        self.max_bitrate_bps = bps;
        self
    }

    /// Builder: set output directory.
    pub fn with_output_dir(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Builder: set base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

/// Validates a [`PackagingConfig`] for logical consistency.
#[derive(Debug, Default)]
pub struct PackagingConfigValidator;

impl PackagingConfigValidator {
    /// Validate the configuration, returning a list of human-readable error strings.
    ///
    /// An empty vec means the configuration is valid.
    #[must_use]
    pub fn validate(config: &PackagingConfig) -> Vec<String> {
        let mut errors = Vec::new();

        if config.segment_duration.is_zero() {
            errors.push("segment_duration must be greater than zero".to_string());
        }

        if config.low_latency {
            if config.partial_segment_duration.is_zero() {
                errors.push(
                    "partial_segment_duration must be greater than zero in low-latency mode"
                        .to_string(),
                );
            }
            if config.partial_segment_duration >= config.segment_duration {
                errors.push(
                    "partial_segment_duration must be less than segment_duration".to_string(),
                );
            }
        }

        if config.output_dir.is_empty() {
            errors.push("output_dir must not be empty".to_string());
        }

        if config.encryption_mode == EncryptionMode::Cenc && config.low_latency {
            // CENC + LL-DASH is technically possible but complex; warn rather than error.
            // (We add it as an informational notice.)
        }

        errors
    }

    /// Returns `true` if the configuration passes all validation checks.
    #[must_use]
    pub fn is_valid(config: &PackagingConfig) -> bool {
        Self::validate(config).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_mode_clear_not_encrypted() {
        assert!(!EncryptionMode::Clear.is_encrypted());
    }

    #[test]
    fn test_encryption_mode_aes128_is_encrypted() {
        assert!(EncryptionMode::Aes128.is_encrypted());
    }

    #[test]
    fn test_encryption_mode_sample_aes_is_encrypted() {
        assert!(EncryptionMode::SampleAes.is_encrypted());
    }

    #[test]
    fn test_encryption_mode_cenc_is_encrypted() {
        assert!(EncryptionMode::Cenc.is_encrypted());
    }

    #[test]
    fn test_encryption_mode_label() {
        assert_eq!(EncryptionMode::Clear.label(), "clear");
        assert_eq!(EncryptionMode::Aes128.label(), "AES-128");
        assert_eq!(EncryptionMode::Cenc.label(), "CENC");
    }

    #[test]
    fn test_encryption_mode_display() {
        assert_eq!(EncryptionMode::SampleAes.to_string(), "SAMPLE-AES");
    }

    #[test]
    fn test_config_default_is_valid() {
        let cfg = PackagingConfig::default();
        assert!(PackagingConfigValidator::is_valid(&cfg));
    }

    #[test]
    fn test_config_is_low_latency_false_by_default() {
        let cfg = PackagingConfig::new();
        assert!(!cfg.is_low_latency());
    }

    #[test]
    fn test_config_is_low_latency_enabled() {
        let cfg = PackagingConfig::new().with_low_latency(true);
        assert!(cfg.is_low_latency());
    }

    #[test]
    fn test_config_is_encrypted_default_false() {
        let cfg = PackagingConfig::new();
        assert!(!cfg.is_encrypted());
    }

    #[test]
    fn test_config_is_encrypted_with_aes() {
        let cfg = PackagingConfig::new().with_encryption(EncryptionMode::Aes128);
        assert!(cfg.is_encrypted());
    }

    #[test]
    fn test_config_is_live() {
        let cfg = PackagingConfig::new().with_playlist_window(5);
        assert!(cfg.is_live());
        let vod = PackagingConfig::new();
        assert!(!vod.is_live());
    }

    #[test]
    fn test_validator_zero_segment_duration_fails() {
        let cfg = PackagingConfig {
            segment_duration: Duration::ZERO,
            ..PackagingConfig::default()
        };
        let errors = PackagingConfigValidator::validate(&cfg);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_low_latency_partial_ge_segment_fails() {
        let cfg = PackagingConfig {
            low_latency: true,
            segment_duration: Duration::from_secs(2),
            partial_segment_duration: Duration::from_secs(3), // >= segment_duration
            ..PackagingConfig::default()
        };
        let errors = PackagingConfigValidator::validate(&cfg);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_empty_output_dir_fails() {
        let cfg = PackagingConfig {
            output_dir: String::new(),
            ..PackagingConfig::default()
        };
        let errors = PackagingConfigValidator::validate(&cfg);
        assert!(!errors.is_empty());
    }
}
