//! IMF essence constraints: per-application-profile limits on video/audio parameters.
//!
//! Application profiles (e.g. IMF App #2E, App #4) restrict the allowed codec
//! parameters, frame rates, and audio configurations.  This module provides a
//! simple constraint model that can be used for pre-delivery validation without
//! requiring an XML parser.

#![allow(dead_code)]

/// Allowed edit rates expressed as `(numerator, denominator)`.
///
/// Common values: `(24, 1)`, `(25, 1)`, `(30000, 1001)`.
pub type EditRate = (u32, u32);

/// Constraint set for picture (video) essence.
#[derive(Debug, Clone)]
pub struct PictureConstraints {
    /// Accepted edit rates for the picture track.
    pub allowed_edit_rates: Vec<EditRate>,
    /// Minimum frame width in pixels.
    pub min_width: u32,
    /// Maximum frame width in pixels.
    pub max_width: u32,
    /// Minimum frame height in pixels.
    pub min_height: u32,
    /// Maximum frame height in pixels.
    pub max_height: u32,
    /// Whether interlaced scanning is permitted.
    pub allow_interlaced: bool,
    /// Whether high-dynamic-range metadata is required.
    pub hdr_required: bool,
}

impl PictureConstraints {
    /// Check whether `width × height` satisfies the resolution limits.
    #[must_use]
    pub fn resolution_ok(&self, width: u32, height: u32) -> bool {
        width >= self.min_width
            && width <= self.max_width
            && height >= self.min_height
            && height <= self.max_height
    }

    /// Check whether the given edit rate is in the allowed set.
    #[must_use]
    pub fn edit_rate_ok(&self, rate: EditRate) -> bool {
        self.allowed_edit_rates.contains(&rate)
    }
}

/// Constraint set for audio essence.
#[derive(Debug, Clone)]
pub struct AudioConstraints {
    /// Allowed audio sampling rates in Hz (e.g. 48000, 96000).
    pub allowed_sample_rates: Vec<u32>,
    /// Allowed bit depths (e.g. 24, 32).
    pub allowed_bit_depths: Vec<u32>,
    /// Minimum number of audio channels.
    pub min_channels: u32,
    /// Maximum number of audio channels.
    pub max_channels: u32,
}

impl AudioConstraints {
    /// Check whether the given sample rate is allowed.
    #[must_use]
    pub fn sample_rate_ok(&self, rate: u32) -> bool {
        self.allowed_sample_rates.contains(&rate)
    }

    /// Check whether the given bit depth is allowed.
    #[must_use]
    pub fn bit_depth_ok(&self, depth: u32) -> bool {
        self.allowed_bit_depths.contains(&depth)
    }

    /// Check whether the channel count is within range.
    #[must_use]
    pub fn channels_ok(&self, count: u32) -> bool {
        count >= self.min_channels && count <= self.max_channels
    }
}

/// Combined essence constraints for an IMF application profile.
#[derive(Debug, Clone)]
pub struct EssenceConstraints {
    /// Name / label of the application profile.
    pub profile_name: String,
    /// Picture (video) constraints.
    pub picture: PictureConstraints,
    /// Audio constraints.
    pub audio: AudioConstraints,
}

impl EssenceConstraints {
    /// Create a new `EssenceConstraints` bundle.
    #[must_use]
    pub fn new(
        profile_name: impl Into<String>,
        picture: PictureConstraints,
        audio: AudioConstraints,
    ) -> Self {
        Self {
            profile_name: profile_name.into(),
            picture,
            audio,
        }
    }

    /// Validate a set of delivery parameters against this profile.
    ///
    /// Returns a list of human-readable violation messages (empty = compliant).
    #[must_use]
    pub fn validate(
        &self,
        width: u32,
        height: u32,
        edit_rate: EditRate,
        sample_rate: u32,
        bit_depth: u32,
        channels: u32,
    ) -> Vec<String> {
        let mut errors = Vec::new();

        if !self.picture.resolution_ok(width, height) {
            errors.push(format!(
                "Resolution {}x{} is outside the allowed range {}x{}–{}x{}",
                width,
                height,
                self.picture.min_width,
                self.picture.min_height,
                self.picture.max_width,
                self.picture.max_height,
            ));
        }
        if !self.picture.edit_rate_ok(edit_rate) {
            errors.push(format!(
                "Edit rate {}/{} is not allowed by profile '{}'",
                edit_rate.0, edit_rate.1, self.profile_name
            ));
        }
        if !self.audio.sample_rate_ok(sample_rate) {
            errors.push(format!("Sample rate {sample_rate} Hz is not allowed"));
        }
        if !self.audio.bit_depth_ok(bit_depth) {
            errors.push(format!("Bit depth {bit_depth} is not allowed"));
        }
        if !self.audio.channels_ok(channels) {
            errors.push(format!(
                "Channel count {channels} is outside the allowed range {}-{}",
                self.audio.min_channels, self.audio.max_channels
            ));
        }

        errors
    }
}

/// Pre-built constraint set approximating IMF Application #2 / #2E (SMPTE ST 2067-21).
#[must_use]
pub fn app2_constraints() -> EssenceConstraints {
    EssenceConstraints::new(
        "IMF App #2E",
        PictureConstraints {
            allowed_edit_rates: vec![(24, 1), (25, 1), (30, 1), (48, 1), (50, 1), (60, 1)],
            min_width: 1920,
            max_width: 3840,
            min_height: 1080,
            max_height: 2160,
            allow_interlaced: false,
            hdr_required: false,
        },
        AudioConstraints {
            allowed_sample_rates: vec![48_000, 96_000],
            allowed_bit_depths: vec![24, 32],
            min_channels: 2,
            max_channels: 16,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_audio_constraints() -> AudioConstraints {
        AudioConstraints {
            allowed_sample_rates: vec![48_000, 96_000],
            allowed_bit_depths: vec![24, 32],
            min_channels: 2,
            max_channels: 8,
        }
    }

    fn default_picture_constraints() -> PictureConstraints {
        PictureConstraints {
            allowed_edit_rates: vec![(24, 1), (25, 1)],
            min_width: 1920,
            max_width: 3840,
            min_height: 1080,
            max_height: 2160,
            allow_interlaced: false,
            hdr_required: false,
        }
    }

    // --- PictureConstraints ---

    #[test]
    fn test_resolution_ok_valid() {
        let pc = default_picture_constraints();
        assert!(pc.resolution_ok(1920, 1080));
        assert!(pc.resolution_ok(3840, 2160));
    }

    #[test]
    fn test_resolution_ok_too_small() {
        let pc = default_picture_constraints();
        assert!(!pc.resolution_ok(1280, 720));
    }

    #[test]
    fn test_resolution_ok_too_large() {
        let pc = default_picture_constraints();
        assert!(!pc.resolution_ok(7680, 4320));
    }

    #[test]
    fn test_edit_rate_ok_allowed() {
        let pc = default_picture_constraints();
        assert!(pc.edit_rate_ok((24, 1)));
        assert!(pc.edit_rate_ok((25, 1)));
    }

    #[test]
    fn test_edit_rate_ok_disallowed() {
        let pc = default_picture_constraints();
        assert!(!pc.edit_rate_ok((30, 1)));
        assert!(!pc.edit_rate_ok((30000, 1001)));
    }

    // --- AudioConstraints ---

    #[test]
    fn test_sample_rate_ok() {
        let ac = default_audio_constraints();
        assert!(ac.sample_rate_ok(48_000));
        assert!(ac.sample_rate_ok(96_000));
        assert!(!ac.sample_rate_ok(44_100));
    }

    #[test]
    fn test_bit_depth_ok() {
        let ac = default_audio_constraints();
        assert!(ac.bit_depth_ok(24));
        assert!(ac.bit_depth_ok(32));
        assert!(!ac.bit_depth_ok(16));
    }

    #[test]
    fn test_channels_ok_in_range() {
        let ac = default_audio_constraints();
        assert!(ac.channels_ok(2));
        assert!(ac.channels_ok(8));
    }

    #[test]
    fn test_channels_ok_out_of_range() {
        let ac = default_audio_constraints();
        assert!(!ac.channels_ok(1));
        assert!(!ac.channels_ok(9));
    }

    // --- EssenceConstraints::validate ---

    #[test]
    fn test_validate_fully_compliant() {
        let ec = EssenceConstraints::new(
            "Test",
            default_picture_constraints(),
            default_audio_constraints(),
        );
        let errors = ec.validate(1920, 1080, (24, 1), 48_000, 24, 2);
        assert!(errors.is_empty(), "Unexpected errors: {errors:?}");
    }

    #[test]
    fn test_validate_bad_resolution_produces_error() {
        let ec = EssenceConstraints::new(
            "Test",
            default_picture_constraints(),
            default_audio_constraints(),
        );
        let errors = ec.validate(1280, 720, (24, 1), 48_000, 24, 2);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Resolution")));
    }

    #[test]
    fn test_validate_bad_edit_rate_produces_error() {
        let ec = EssenceConstraints::new(
            "Test",
            default_picture_constraints(),
            default_audio_constraints(),
        );
        let errors = ec.validate(1920, 1080, (60, 1), 48_000, 24, 2);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Edit rate")));
    }

    #[test]
    fn test_validate_bad_sample_rate_produces_error() {
        let ec = EssenceConstraints::new(
            "Test",
            default_picture_constraints(),
            default_audio_constraints(),
        );
        let errors = ec.validate(1920, 1080, (24, 1), 44_100, 24, 2);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Sample rate")));
    }

    #[test]
    fn test_validate_multiple_errors() {
        let ec = EssenceConstraints::new(
            "Test",
            default_picture_constraints(),
            default_audio_constraints(),
        );
        // Bad resolution + bad edit rate + bad sample rate
        let errors = ec.validate(640, 480, (60, 1), 44_100, 24, 2);
        assert!(errors.len() >= 3);
    }

    // --- app2_constraints ---

    #[test]
    fn test_app2_allows_4k_24fps() {
        let ec = app2_constraints();
        let errors = ec.validate(3840, 2160, (24, 1), 48_000, 24, 2);
        assert!(errors.is_empty(), "Unexpected errors: {errors:?}");
    }

    #[test]
    fn test_app2_rejects_hd_30000_1001() {
        let ec = app2_constraints();
        // 30000/1001 is not in the allowed set for App #2E in this simplified model.
        let errors = ec.validate(1920, 1080, (30000, 1001), 48_000, 24, 2);
        assert!(!errors.is_empty());
    }
}
