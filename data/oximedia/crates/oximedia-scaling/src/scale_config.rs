//! Scale configuration and validation.
//!
//! Provides a rich `ScaleConfig` that bundles target dimensions, algorithm
//! choice, quality settings and sharpening parameters, plus a
//! `ScaleConfigValidator` that catches invalid combinations before work begins.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

// ── ScaleAlgorithm ────────────────────────────────────────────────────────────

/// Scaling algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleAlgorithm {
    /// Nearest-neighbour – fastest, lowest quality.
    NearestNeighbour,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation (Mitchell–Netravali by default).
    Bicubic,
    /// Lanczos resampling with configurable number of lobes.
    Lanczos,
    /// Area averaging – good for significant downscaling.
    Area,
    /// Super-resolution upscaling (ML-assisted).
    SuperResolution,
}

impl std::fmt::Display for ScaleAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::NearestNeighbour => "nearest-neighbour",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
            Self::Lanczos => "lanczos",
            Self::Area => "area",
            Self::SuperResolution => "super-resolution",
        };
        write!(f, "{s}")
    }
}

impl Default for ScaleAlgorithm {
    fn default() -> Self {
        Self::Lanczos
    }
}

// ── ScaleConfig ───────────────────────────────────────────────────────────────

/// Complete configuration for a scaling operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleConfig {
    /// Target width in pixels (must be > 0).
    pub width: u32,
    /// Target height in pixels (must be > 0).
    pub height: u32,
    /// Scaling algorithm to use.
    pub algorithm: ScaleAlgorithm,
    /// Lanczos lobe count (only used when `algorithm == Lanczos`).  Typical: 2–4.
    pub lanczos_lobes: u8,
    /// Mitchell B parameter for bicubic (0.0–1.0).
    pub bicubic_b: f32,
    /// Mitchell C parameter for bicubic (0.0–1.0).
    pub bicubic_c: f32,
    /// Post-scale unsharp-mask amount (0.0 = none, 1.0 = strong).
    pub sharpen_amount: f32,
    /// Enable multi-threaded processing.
    pub parallel: bool,
    /// Maximum number of worker threads (0 = use all logical cores).
    pub thread_count: u32,
}

impl ScaleConfig {
    /// Create a new configuration targeting `width × height` with `algorithm`.
    pub fn new(width: u32, height: u32, algorithm: ScaleAlgorithm) -> Self {
        Self {
            width,
            height,
            algorithm,
            lanczos_lobes: 3,
            bicubic_b: 1.0 / 3.0,
            bicubic_c: 1.0 / 3.0,
            sharpen_amount: 0.0,
            parallel: true,
            thread_count: 0,
        }
    }

    /// Create a high-quality 1080p preset.
    pub fn hd_1080p() -> Self {
        Self::new(1920, 1080, ScaleAlgorithm::Lanczos)
    }

    /// Create a 4K preset.
    pub fn uhd_4k() -> Self {
        Self::new(3840, 2160, ScaleAlgorithm::Lanczos)
    }

    /// Create a thumbnail preset (fast, bilinear).
    pub fn thumbnail(width: u32, height: u32) -> Self {
        Self::new(width, height, ScaleAlgorithm::Bilinear)
    }

    /// Set the Lanczos lobe count.
    pub fn with_lanczos_lobes(mut self, lobes: u8) -> Self {
        self.lanczos_lobes = lobes;
        self
    }

    /// Set bicubic B/C parameters.
    pub fn with_bicubic_params(mut self, b: f32, c: f32) -> Self {
        self.bicubic_b = b;
        self.bicubic_c = c;
        self
    }

    /// Set post-scale sharpening amount.
    pub fn with_sharpen(mut self, amount: f32) -> Self {
        self.sharpen_amount = amount;
        self
    }

    /// Enable or disable parallel processing.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Return the pixel count of the target frame.
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Return the aspect ratio of the target frame as a float.
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self::hd_1080p()
    }
}

// ── ScaleConfigValidator ──────────────────────────────────────────────────────

/// Error type for configuration validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Width or height is zero.
    ZeroDimension,
    /// Lanczos lobes are outside the 1–8 range.
    InvalidLanczosLobes(u8),
    /// A bicubic parameter is outside 0.0–1.0.
    InvalidBicubicParam(String),
    /// Sharpening amount is negative or greater than 2.0.
    InvalidSharpenAmount,
    /// Requested algorithm is `SuperResolution` but `width` or `height` is
    /// smaller than the (hypothetical) source — we can't determine upscale
    /// factor without source info, so the validator just flags zero targets.
    SuperResolutionZeroDim,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "width and height must be > 0"),
            Self::InvalidLanczosLobes(n) => {
                write!(f, "lanczos_lobes {n} is outside valid range [1, 8]")
            }
            Self::InvalidBicubicParam(p) => {
                write!(f, "bicubic parameter {p} is outside [0.0, 1.0]")
            }
            Self::InvalidSharpenAmount => {
                write!(f, "sharpen_amount must be in [0.0, 2.0]")
            }
            Self::SuperResolutionZeroDim => {
                write!(f, "super-resolution requires non-zero target dimensions")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Validates `ScaleConfig` instances before use.
#[derive(Debug, Default, Clone)]
pub struct ScaleConfigValidator;

impl ScaleConfigValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self
    }

    /// Validate a `ScaleConfig`, returning all errors found.
    /// An empty `Vec` means the configuration is valid.
    pub fn validate(&self, cfg: &ScaleConfig) -> Vec<ConfigError> {
        let mut errors = Vec::new();

        if cfg.width == 0 || cfg.height == 0 {
            errors.push(ConfigError::ZeroDimension);
        }

        if cfg.algorithm == ScaleAlgorithm::Lanczos
            && (cfg.lanczos_lobes < 1 || cfg.lanczos_lobes > 8)
        {
            errors.push(ConfigError::InvalidLanczosLobes(cfg.lanczos_lobes));
        }

        if cfg.algorithm == ScaleAlgorithm::Bicubic {
            if !(0.0..=1.0).contains(&cfg.bicubic_b) {
                errors.push(ConfigError::InvalidBicubicParam(format!(
                    "B={}",
                    cfg.bicubic_b
                )));
            }
            if !(0.0..=1.0).contains(&cfg.bicubic_c) {
                errors.push(ConfigError::InvalidBicubicParam(format!(
                    "C={}",
                    cfg.bicubic_c
                )));
            }
        }

        if !(0.0..=2.0).contains(&cfg.sharpen_amount) {
            errors.push(ConfigError::InvalidSharpenAmount);
        }

        errors
    }

    /// Return `true` if the configuration is valid.
    pub fn is_valid(&self, cfg: &ScaleConfig) -> bool {
        self.validate(cfg).is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn validator() -> ScaleConfigValidator {
        ScaleConfigValidator::new()
    }

    #[test]
    fn test_default_config_is_valid() {
        let cfg = ScaleConfig::default();
        assert!(validator().is_valid(&cfg));
    }

    #[test]
    fn test_hd_1080p_preset() {
        let cfg = ScaleConfig::hd_1080p();
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.algorithm, ScaleAlgorithm::Lanczos);
    }

    #[test]
    fn test_uhd_4k_preset() {
        let cfg = ScaleConfig::uhd_4k();
        assert_eq!(cfg.width, 3840);
        assert_eq!(cfg.height, 2160);
    }

    #[test]
    fn test_thumbnail_preset() {
        let cfg = ScaleConfig::thumbnail(320, 240);
        assert_eq!(cfg.algorithm, ScaleAlgorithm::Bilinear);
    }

    #[test]
    fn test_pixel_count() {
        let cfg = ScaleConfig::hd_1080p();
        assert_eq!(cfg.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_aspect_ratio() {
        let cfg = ScaleConfig::hd_1080p();
        let ar = cfg.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_width_rejected() {
        let cfg = ScaleConfig::new(0, 1080, ScaleAlgorithm::Bilinear);
        let errs = validator().validate(&cfg);
        assert!(errs.iter().any(|e| *e == ConfigError::ZeroDimension));
    }

    #[test]
    fn test_zero_height_rejected() {
        let cfg = ScaleConfig::new(1920, 0, ScaleAlgorithm::Bilinear);
        let errs = validator().validate(&cfg);
        assert!(errs.iter().any(|e| *e == ConfigError::ZeroDimension));
    }

    #[test]
    fn test_invalid_lanczos_lobes() {
        let cfg = ScaleConfig::new(1280, 720, ScaleAlgorithm::Lanczos).with_lanczos_lobes(0);
        let errs = validator().validate(&cfg);
        assert!(errs
            .iter()
            .any(|e| matches!(e, ConfigError::InvalidLanczosLobes(_))));
    }

    #[test]
    fn test_valid_lanczos_lobes_boundary() {
        let cfg = ScaleConfig::new(1280, 720, ScaleAlgorithm::Lanczos).with_lanczos_lobes(1);
        assert!(validator().is_valid(&cfg));
    }

    #[test]
    fn test_invalid_bicubic_b() {
        let cfg =
            ScaleConfig::new(1280, 720, ScaleAlgorithm::Bicubic).with_bicubic_params(1.5, 0.333);
        let errs = validator().validate(&cfg);
        assert!(errs
            .iter()
            .any(|e| matches!(e, ConfigError::InvalidBicubicParam(_))));
    }

    #[test]
    fn test_invalid_sharpen_amount() {
        let cfg = ScaleConfig::new(1280, 720, ScaleAlgorithm::Bilinear).with_sharpen(-0.1);
        let errs = validator().validate(&cfg);
        assert!(errs.iter().any(|e| *e == ConfigError::InvalidSharpenAmount));
    }

    #[test]
    fn test_sharpen_max_boundary_is_valid() {
        let cfg = ScaleConfig::new(1280, 720, ScaleAlgorithm::Bilinear).with_sharpen(2.0);
        assert!(validator().is_valid(&cfg));
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(ScaleAlgorithm::Lanczos.to_string(), "lanczos");
        assert_eq!(
            ScaleAlgorithm::NearestNeighbour.to_string(),
            "nearest-neighbour"
        );
        assert_eq!(
            ScaleAlgorithm::SuperResolution.to_string(),
            "super-resolution"
        );
    }

    #[test]
    fn test_config_error_display() {
        let e = ConfigError::ZeroDimension;
        assert!(e.to_string().contains("width and height"));
    }

    #[test]
    fn test_builder_parallel_false() {
        let cfg = ScaleConfig::hd_1080p().with_parallel(false);
        assert!(!cfg.parallel);
    }

    #[test]
    fn test_multiple_errors_accumulate() {
        let cfg = ScaleConfig {
            width: 0,
            height: 0,
            algorithm: ScaleAlgorithm::Lanczos,
            lanczos_lobes: 0,
            bicubic_b: 0.333,
            bicubic_c: 0.333,
            sharpen_amount: 3.0,
            parallel: true,
            thread_count: 0,
        };
        let errs = validator().validate(&cfg);
        assert!(errs.len() >= 3);
    }
}
