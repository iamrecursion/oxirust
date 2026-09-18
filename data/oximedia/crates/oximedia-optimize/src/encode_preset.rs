//! Encode preset management for `oximedia-optimize`.
//!
//! Provides speed/quality presets, CRF adjustments, and bitrate estimation
//! for common encoding scenarios.

#![allow(dead_code)]

/// Encoding speed tier, trading quality for throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodeSpeed {
    /// Ultra-fast, lowest quality.
    Ultrafast,
    /// Very fast encoding.
    Superfast,
    /// Fast encoding with acceptable quality.
    Fast,
    /// Balanced speed and quality.
    Medium,
    /// Slower with better quality.
    Slow,
    /// Very slow, high quality.
    Slower,
    /// Exhaustive search, maximum quality.
    Veryslow,
}

impl EncodeSpeed {
    /// Returns the CRF adjustment relative to the baseline CRF.
    ///
    /// Faster presets add positive CRF offset (lower quality),
    /// slower presets subtract (higher quality).
    #[must_use]
    pub fn crf_adjustment(&self) -> i32 {
        match self {
            Self::Ultrafast => 6,
            Self::Superfast => 4,
            Self::Fast => 2,
            Self::Medium => 0,
            Self::Slow => -2,
            Self::Slower => -4,
            Self::Veryslow => -6,
        }
    }

    /// Returns a human-readable label for the speed tier.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ultrafast => "ultrafast",
            Self::Superfast => "superfast",
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::Slow => "slow",
            Self::Slower => "slower",
            Self::Veryslow => "veryslow",
        }
    }

    /// Returns the approximate encode-speed multiplier relative to `Medium`.
    ///
    /// A value > 1.0 means faster than medium; < 1.0 means slower.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn speed_multiplier(&self) -> f64 {
        match self {
            Self::Ultrafast => 8.0,
            Self::Superfast => 5.0,
            Self::Fast => 2.5,
            Self::Medium => 1.0,
            Self::Slow => 0.5,
            Self::Slower => 0.25,
            Self::Veryslow => 0.1,
        }
    }
}

impl Default for EncodeSpeed {
    fn default() -> Self {
        Self::Medium
    }
}

/// A complete encoding preset combining speed, base CRF, and codec hints.
#[derive(Debug, Clone)]
pub struct EncodePreset {
    /// Preset name (e.g. "web_h264_1080p").
    pub name: String,
    /// Speed tier.
    pub speed: EncodeSpeed,
    /// Baseline CRF before speed adjustment (typically 18–28).
    pub base_crf: u8,
    /// Target video width in pixels.
    pub width: u32,
    /// Target video height in pixels.
    pub height: u32,
    /// Target frame rate.
    pub fps: f64,
}

impl EncodePreset {
    /// Creates a new preset with the given parameters.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        speed: EncodeSpeed,
        base_crf: u8,
        width: u32,
        height: u32,
        fps: f64,
    ) -> Self {
        Self {
            name: name.into(),
            speed,
            base_crf,
            width,
            height,
            fps,
        }
    }

    /// Returns the effective CRF after applying the speed-tier adjustment,
    /// clamped to [0, 51].
    #[must_use]
    pub fn effective_crf(&self) -> u8 {
        let adjusted = i32::from(self.base_crf) + self.speed.crf_adjustment();
        adjusted.clamp(0, 51) as u8
    }

    /// Estimates the output bitrate in kilobits-per-second based on resolution,
    /// frame rate, and effective CRF using a simplified perceptual model.
    ///
    /// Formula: `bitrate_kbps ≈ (pixels_per_sec / complexity_factor) * crf_scale`
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_bitrate_kbps(&self) -> f64 {
        let pixels_per_sec = (self.width as f64) * (self.height as f64) * self.fps;
        // Empirical constants derived from x264/x265 rate-control curves.
        let crf = f64::from(self.effective_crf());
        let crf_scale = (-crf / 6.0_f64).exp2(); // quality factor: higher CRF → lower bitrate
        let complexity_factor = 500.0_f64; // pixels per bit at medium complexity
        (pixels_per_sec / complexity_factor) * crf_scale / 1000.0
    }

    /// Returns true when the preset is considered lossless (CRF == 0).
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.effective_crf() == 0
    }
}

impl Default for EncodePreset {
    fn default() -> Self {
        Self::new("default", EncodeSpeed::Medium, 23, 1920, 1080, 25.0)
    }
}

/// A library of named presets for quick lookup.
#[derive(Debug, Default)]
pub struct PresetLibrary {
    presets: Vec<EncodePreset>,
}

impl PresetLibrary {
    /// Creates an empty preset library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a preset library pre-populated with common web/broadcast presets.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut lib = Self::new();
        lib.add(EncodePreset::new(
            "web_1080p_fast",
            EncodeSpeed::Fast,
            23,
            1920,
            1080,
            30.0,
        ));
        lib.add(EncodePreset::new(
            "web_720p_medium",
            EncodeSpeed::Medium,
            24,
            1280,
            720,
            30.0,
        ));
        lib.add(EncodePreset::new(
            "archive_2160p_slow",
            EncodeSpeed::Slow,
            18,
            3840,
            2160,
            24.0,
        ));
        lib.add(EncodePreset::new(
            "streaming_480p_fast",
            EncodeSpeed::Fast,
            26,
            854,
            480,
            30.0,
        ));
        lib.add(EncodePreset::new(
            "broadcast_1080i_medium",
            EncodeSpeed::Medium,
            21,
            1920,
            1080,
            25.0,
        ));
        lib
    }

    /// Adds a preset to the library.
    pub fn add(&mut self, preset: EncodePreset) {
        self.presets.push(preset);
    }

    /// Finds a preset by exact name match.
    #[must_use]
    pub fn find_preset(&self, name: &str) -> Option<&EncodePreset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// Returns the preset whose estimated bitrate is closest to `target_kbps`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn best_for_target(&self, target_kbps: f64) -> Option<&EncodePreset> {
        self.presets.iter().min_by(|a, b| {
            let da = (a.estimated_bitrate_kbps() - target_kbps).abs();
            let db = (b.estimated_bitrate_kbps() - target_kbps).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns all presets in the library.
    #[must_use]
    pub fn all(&self) -> &[EncodePreset] {
        &self.presets
    }

    /// Returns the number of presets in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Returns true if the library contains no presets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_speed_crf_adjustment_medium_zero() {
        assert_eq!(EncodeSpeed::Medium.crf_adjustment(), 0);
    }

    #[test]
    fn test_encode_speed_crf_adjustment_fast_positive() {
        assert!(EncodeSpeed::Fast.crf_adjustment() > 0);
    }

    #[test]
    fn test_encode_speed_crf_adjustment_slow_negative() {
        assert!(EncodeSpeed::Slow.crf_adjustment() < 0);
    }

    #[test]
    fn test_encode_speed_label_round_trip() {
        assert_eq!(EncodeSpeed::Veryslow.label(), "veryslow");
        assert_eq!(EncodeSpeed::Ultrafast.label(), "ultrafast");
    }

    #[test]
    fn test_encode_speed_multiplier_ordering() {
        assert!(EncodeSpeed::Ultrafast.speed_multiplier() > EncodeSpeed::Fast.speed_multiplier());
        assert!(EncodeSpeed::Fast.speed_multiplier() > EncodeSpeed::Slow.speed_multiplier());
    }

    #[test]
    fn test_encode_preset_effective_crf_clamp_lower() {
        let p = EncodePreset::new("test", EncodeSpeed::Veryslow, 2, 1280, 720, 30.0);
        assert_eq!(p.effective_crf(), 0); // 2 - 6 = -4 → clamped to 0
    }

    #[test]
    fn test_encode_preset_effective_crf_clamp_upper() {
        let p = EncodePreset::new("test", EncodeSpeed::Ultrafast, 49, 1280, 720, 30.0);
        assert_eq!(p.effective_crf(), 51); // 49 + 6 = 55 → clamped to 51
    }

    #[test]
    fn test_encode_preset_estimated_bitrate_positive() {
        let p = EncodePreset::default();
        assert!(p.estimated_bitrate_kbps() > 0.0);
    }

    #[test]
    fn test_encode_preset_higher_crf_lower_bitrate() {
        let low_crf = EncodePreset::new("lo", EncodeSpeed::Medium, 18, 1920, 1080, 25.0);
        let high_crf = EncodePreset::new("hi", EncodeSpeed::Medium, 28, 1920, 1080, 25.0);
        assert!(low_crf.estimated_bitrate_kbps() > high_crf.estimated_bitrate_kbps());
    }

    #[test]
    fn test_encode_preset_lossless_when_crf_zero() {
        let p = EncodePreset::new("lossless", EncodeSpeed::Veryslow, 4, 1280, 720, 24.0);
        // 4 - 6 = -2 → clamped to 0
        assert!(p.is_lossless());
    }

    #[test]
    fn test_preset_library_empty_on_new() {
        let lib = PresetLibrary::new();
        assert!(lib.is_empty());
        assert_eq!(lib.len(), 0);
    }

    #[test]
    fn test_preset_library_with_defaults_non_empty() {
        let lib = PresetLibrary::with_defaults();
        assert!(!lib.is_empty());
        assert!(lib.len() >= 5);
    }

    #[test]
    fn test_preset_library_find_preset_found() {
        let lib = PresetLibrary::with_defaults();
        assert!(lib.find_preset("web_720p_medium").is_some());
    }

    #[test]
    fn test_preset_library_find_preset_not_found() {
        let lib = PresetLibrary::with_defaults();
        assert!(lib.find_preset("nonexistent_preset").is_none());
    }

    #[test]
    fn test_preset_library_best_for_target_returns_some() {
        let lib = PresetLibrary::with_defaults();
        let result = lib.best_for_target(2000.0);
        assert!(result.is_some());
    }
}
