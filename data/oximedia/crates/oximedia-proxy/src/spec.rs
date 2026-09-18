//! Proxy specification: defines the desired proxy format, resolution, codec, and bitrate.
//!
//! A `ProxySpec` is the authoritative description of how a proxy should be created.
//! It decouples the "what" (spec) from the "how" (encoder) and "where" (registry).

use crate::generate::ProxyGenerationSettings;
use crate::{ProxyError, Result};
use serde::{Deserialize, Serialize};

/// Target video codec for proxy encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxyCodec {
    /// H.264 / AVC (best compatibility).
    H264,
    /// H.265 / HEVC (better compression).
    H265,
    /// VP9 (open, good quality).
    Vp9,
    /// Apple ProRes 422 Proxy (edit-optimized, Apple ecosystem).
    ProRes422Proxy,
    /// DNxHD 36 (Avid-compatible lightweight proxy).
    DnxHd36,
    /// Custom codec identified by name.
    Custom(String),
}

impl ProxyCodec {
    /// Get the codec identifier string used in settings.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::H264 => "h264",
            Self::H265 => "h265",
            Self::Vp9 => "vp9",
            Self::ProRes422Proxy => "prores_proxy",
            Self::DnxHd36 => "dnxhd36",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Recommended file container for this codec.
    #[must_use]
    pub fn recommended_container(&self) -> &'static str {
        match self {
            Self::H264 | Self::H265 | Self::Vp9 => "mp4",
            Self::ProRes422Proxy => "mov",
            Self::DnxHd36 => "mxf",
            Self::Custom(_) => "mp4",
        }
    }

    /// Whether this codec is typically hardware-acceleratable.
    #[must_use]
    pub const fn hw_accel_supported(&self) -> bool {
        matches!(self, Self::H264 | Self::H265)
    }
}

impl std::fmt::Display for ProxyCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Target resolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProxyResolutionMode {
    /// Scale to a fraction of the original (e.g., 0.25 = quarter res).
    ScaleFactor(f32),
    /// Specific pixel dimensions (width, height).
    Fixed(u32, u32),
    /// Fit within a bounding box preserving aspect ratio.
    FitWithin {
        /// Maximum allowed width in pixels.
        max_width: u32,
        /// Maximum allowed height in pixels.
        max_height: u32,
    },
}

impl ProxyResolutionMode {
    /// Compute the output dimensions given the original dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if scale factor is out of range.
    pub fn compute_output(&self, orig_w: u32, orig_h: u32) -> Result<(u32, u32)> {
        match self {
            Self::ScaleFactor(scale) => {
                if *scale <= 0.0 || *scale > 4.0 {
                    return Err(ProxyError::InvalidInput(format!(
                        "Scale factor must be in (0, 4], got {scale}"
                    )));
                }
                let w = ((orig_w as f32 * scale) as u32).max(2);
                let h = ((orig_h as f32 * scale) as u32).max(2);
                // Ensure even dimensions for codec compatibility
                Ok((w & !1, h & !1))
            }
            Self::Fixed(w, h) => {
                if *w == 0 || *h == 0 {
                    return Err(ProxyError::InvalidInput(
                        "Fixed dimensions must be > 0".to_string(),
                    ));
                }
                Ok((*w & !1, *h & !1))
            }
            Self::FitWithin {
                max_width,
                max_height,
            } => {
                if *max_width == 0 || *max_height == 0 {
                    return Err(ProxyError::InvalidInput(
                        "FitWithin bounds must be > 0".to_string(),
                    ));
                }
                let w_ratio = *max_width as f32 / orig_w as f32;
                let h_ratio = *max_height as f32 / orig_h as f32;
                let scale = w_ratio.min(h_ratio);
                let w = ((orig_w as f32 * scale) as u32).max(2);
                let h = ((orig_h as f32 * scale) as u32).max(2);
                Ok((w & !1, h & !1))
            }
        }
    }
}

/// Complete proxy specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxySpec {
    /// Human-readable name for this spec.
    pub name: String,
    /// Target resolution mode.
    pub resolution: ProxyResolutionMode,
    /// Video codec to use.
    pub codec: ProxyCodec,
    /// Video bitrate in bits per second.
    pub video_bitrate: u64,
    /// Audio codec (e.g. "aac").
    pub audio_codec: String,
    /// Audio bitrate in bits per second.
    pub audio_bitrate: u64,
    /// Container format override. If `None`, uses `codec.recommended_container()`.
    pub container: Option<String>,
    /// Whether to preserve source timecode.
    pub preserve_timecode: bool,
    /// Whether to preserve source metadata.
    pub preserve_metadata: bool,
    /// Whether to use hardware acceleration.
    pub use_hw_accel: bool,
    /// Encoding quality preset ("fast", "medium", "slow").
    pub quality_preset: String,
}

impl ProxySpec {
    /// Create a new proxy spec with required fields.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        resolution: ProxyResolutionMode,
        codec: ProxyCodec,
        video_bitrate: u64,
    ) -> Self {
        Self {
            name: name.into(),
            resolution,
            codec,
            video_bitrate,
            audio_codec: "aac".to_string(),
            audio_bitrate: 128_000,
            container: None,
            preserve_timecode: true,
            preserve_metadata: true,
            use_hw_accel: true,
            quality_preset: "fast".to_string(),
        }
    }

    /// Get the container format.
    #[must_use]
    pub fn container_format(&self) -> &str {
        self.container
            .as_deref()
            .unwrap_or_else(|| self.codec.recommended_container())
    }

    /// Validate the spec.
    ///
    /// # Errors
    ///
    /// Returns an error if any field has an invalid value.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(ProxyError::InvalidInput(
                "Spec name cannot be empty".to_string(),
            ));
        }
        if self.video_bitrate == 0 {
            return Err(ProxyError::InvalidInput(
                "Video bitrate must be > 0".to_string(),
            ));
        }
        if let ProxyResolutionMode::ScaleFactor(s) = self.resolution {
            if s <= 0.0 || s > 4.0 {
                return Err(ProxyError::InvalidInput(format!(
                    "Scale factor {s} out of range (0, 4]"
                )));
            }
        }
        Ok(())
    }

    /// Convert to `ProxyGenerationSettings`.
    #[must_use]
    pub fn to_generation_settings(&self) -> ProxyGenerationSettings {
        let scale_factor = match self.resolution {
            ProxyResolutionMode::ScaleFactor(s) => s,
            ProxyResolutionMode::Fixed(_, _) | ProxyResolutionMode::FitWithin { .. } => 0.5,
        };
        ProxyGenerationSettings {
            scale_factor,
            codec: self.codec.as_str().to_string(),
            bitrate: self.video_bitrate,
            audio_codec: self.audio_codec.clone(),
            audio_bitrate: self.audio_bitrate,
            preserve_frame_rate: true,
            preserve_timecode: self.preserve_timecode,
            preserve_metadata: self.preserve_metadata,
            container: self.container_format().to_string(),
            use_hw_accel: self.use_hw_accel,
            threads: 0,
            quality_preset: self.quality_preset.clone(),
        }
    }

    /// Predefined: quarter-resolution H.264 proxy at 2 Mbps.
    #[must_use]
    pub fn quarter_h264() -> Self {
        Self::new(
            "Quarter H.264",
            ProxyResolutionMode::ScaleFactor(0.25),
            ProxyCodec::H264,
            2_000_000,
        )
    }

    /// Predefined: half-resolution H.264 proxy at 5 Mbps.
    #[must_use]
    pub fn half_h264() -> Self {
        Self::new(
            "Half H.264",
            ProxyResolutionMode::ScaleFactor(0.5),
            ProxyCodec::H264,
            5_000_000,
        )
    }

    /// Predefined: 1080p FitWithin H.265 proxy at 4 Mbps.
    #[must_use]
    pub fn hd_h265() -> Self {
        Self::new(
            "HD H.265",
            ProxyResolutionMode::FitWithin {
                max_width: 1920,
                max_height: 1080,
            },
            ProxyCodec::H265,
            4_000_000,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_codec_as_str() {
        assert_eq!(ProxyCodec::H264.as_str(), "h264");
        assert_eq!(ProxyCodec::H265.as_str(), "h265");
        assert_eq!(ProxyCodec::Vp9.as_str(), "vp9");
        assert_eq!(ProxyCodec::ProRes422Proxy.as_str(), "prores_proxy");
        assert_eq!(ProxyCodec::DnxHd36.as_str(), "dnxhd36");
        assert_eq!(
            ProxyCodec::Custom("mycodec".to_string()).as_str(),
            "mycodec"
        );
    }

    #[test]
    fn test_proxy_codec_container() {
        assert_eq!(ProxyCodec::H264.recommended_container(), "mp4");
        assert_eq!(ProxyCodec::ProRes422Proxy.recommended_container(), "mov");
        assert_eq!(ProxyCodec::DnxHd36.recommended_container(), "mxf");
    }

    #[test]
    fn test_proxy_codec_hw_accel() {
        assert!(ProxyCodec::H264.hw_accel_supported());
        assert!(ProxyCodec::H265.hw_accel_supported());
        assert!(!ProxyCodec::Vp9.hw_accel_supported());
    }

    #[test]
    fn test_proxy_codec_display() {
        assert_eq!(ProxyCodec::H264.to_string(), "h264");
    }

    #[test]
    fn test_resolution_mode_scale_factor() {
        let (w, h) = ProxyResolutionMode::ScaleFactor(0.25)
            .compute_output(1920, 1080)
            .expect("should succeed in test");
        assert_eq!(w, 480);
        assert_eq!(h, 270);
    }

    #[test]
    fn test_resolution_mode_fixed() {
        let (w, h) = ProxyResolutionMode::Fixed(640, 360)
            .compute_output(1920, 1080)
            .expect("should succeed in test");
        assert_eq!(w, 640);
        assert_eq!(h, 360);
    }

    #[test]
    fn test_resolution_mode_fit_within() {
        let (w, h) = ProxyResolutionMode::FitWithin {
            max_width: 960,
            max_height: 540,
        }
        .compute_output(1920, 1080)
        .expect("should succeed in test");
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn test_resolution_mode_fit_within_portrait() {
        // 9:16 source (e.g. vertical video), fit in 1920x1080
        let (w, h) = ProxyResolutionMode::FitWithin {
            max_width: 1920,
            max_height: 1080,
        }
        .compute_output(1080, 1920)
        .expect("should succeed in test");
        assert!(h <= 1080, "Height {h} should not exceed 1080");
        assert!(w <= 1920, "Width {w} should not exceed 1920");
    }

    #[test]
    fn test_resolution_mode_scale_factor_invalid() {
        let result = ProxyResolutionMode::ScaleFactor(-0.1).compute_output(1920, 1080);
        assert!(result.is_err());
        let result2 = ProxyResolutionMode::ScaleFactor(5.0).compute_output(1920, 1080);
        assert!(result2.is_err());
    }

    #[test]
    fn test_resolution_mode_fixed_zero() {
        let result = ProxyResolutionMode::Fixed(0, 360).compute_output(1920, 1080);
        assert!(result.is_err());
    }

    #[test]
    fn test_proxy_spec_new() {
        let spec = ProxySpec::new(
            "Test",
            ProxyResolutionMode::ScaleFactor(0.5),
            ProxyCodec::H264,
            5_000_000,
        );
        assert_eq!(spec.name, "Test");
        assert_eq!(spec.video_bitrate, 5_000_000);
        assert!(spec.preserve_timecode);
    }

    #[test]
    fn test_proxy_spec_validate_ok() {
        let spec = ProxySpec::quarter_h264();
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn test_proxy_spec_validate_empty_name() {
        let mut spec = ProxySpec::quarter_h264();
        spec.name = String::new();
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_proxy_spec_validate_zero_bitrate() {
        let mut spec = ProxySpec::quarter_h264();
        spec.video_bitrate = 0;
        assert!(spec.validate().is_err());
    }

    #[test]
    fn test_proxy_spec_predefined() {
        let q = ProxySpec::quarter_h264();
        assert_eq!(q.codec, ProxyCodec::H264);
        assert_eq!(q.video_bitrate, 2_000_000);

        let h = ProxySpec::half_h264();
        assert_eq!(h.video_bitrate, 5_000_000);

        let hd = ProxySpec::hd_h265();
        assert_eq!(hd.codec, ProxyCodec::H265);
    }

    #[test]
    fn test_proxy_spec_container_format() {
        let spec = ProxySpec::quarter_h264();
        assert_eq!(spec.container_format(), "mp4");

        let mut prores_spec = spec.clone();
        prores_spec.codec = ProxyCodec::ProRes422Proxy;
        assert_eq!(prores_spec.container_format(), "mov");

        let mut custom_spec = spec.clone();
        custom_spec.container = Some("mkv".to_string());
        assert_eq!(custom_spec.container_format(), "mkv");
    }

    #[test]
    fn test_proxy_spec_to_generation_settings() {
        let spec = ProxySpec::quarter_h264();
        let settings = spec.to_generation_settings();
        assert!((settings.scale_factor - 0.25).abs() < f32::EPSILON);
        assert_eq!(settings.codec, "h264");
        assert_eq!(settings.bitrate, 2_000_000);
        assert_eq!(settings.container, "mp4");
    }

    #[test]
    fn test_resolution_output_even_dimensions() {
        // Should always produce even dimensions for codec compat
        let (w, h) = ProxyResolutionMode::ScaleFactor(0.33)
            .compute_output(1920, 1080)
            .expect("should succeed in test");
        assert_eq!(w % 2, 0, "Width must be even");
        assert_eq!(h % 2, 0, "Height must be even");
    }
}
