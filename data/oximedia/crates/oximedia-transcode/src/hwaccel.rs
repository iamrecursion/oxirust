//! Hardware acceleration configuration and simulation.
//!
//! Pure-Rust module providing backend enumeration, capability modelling,
//! codec–encoder name mapping, and best-backend selection — no actual
//! hardware calls are performed.

use serde::{Deserialize, Serialize};

// ─── HwAccelBackend ───────────────────────────────────────────────────────────

/// Hardware acceleration backends supported by OxiMedia.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HwAccelBackend {
    /// Software-only encoding/decoding.
    None,
    /// Video Acceleration API (Linux, Intel/AMD/Nvidia via VA-API).
    Vaapi,
    /// NVIDIA NVENC/NVDEC (CUDA-based).
    Nvenc,
    /// Apple VideoToolbox (macOS / iOS).
    Videotoolbox,
    /// Intel Quick Sync Video.
    Qsv,
    /// AMD Advanced Media Framework.
    Amf,
    /// Direct3D 11 Video Acceleration (Windows).
    D3d11Va,
}

impl HwAccelBackend {
    /// Returns a stable string identifier for this backend.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Vaapi => "vaapi",
            Self::Nvenc => "nvenc",
            Self::Videotoolbox => "videotoolbox",
            Self::Qsv => "qsv",
            Self::Amf => "amf",
            Self::D3d11Va => "d3d11va",
        }
    }

    /// Returns all non-`None` backends.
    #[must_use]
    pub fn all_hw() -> &'static [HwAccelBackend] {
        &[
            Self::Vaapi,
            Self::Nvenc,
            Self::Videotoolbox,
            Self::Qsv,
            Self::Amf,
            Self::D3d11Va,
        ]
    }
}

// ─── HwAccelCaps ─────────────────────────────────────────────────────────────

/// Capabilities advertised by a hardware acceleration backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HwAccelCaps {
    /// Which backend these capabilities describe.
    pub backend: HwAccelBackend,
    /// Whether this backend can accelerate decoding.
    pub can_decode: bool,
    /// Whether this backend can accelerate encoding.
    pub can_encode: bool,
    /// Maximum resolution supported (width, height).
    pub max_resolution: (u32, u32),
    /// Codec names supported by this backend (e.g., "hevc", "av1").
    pub supported_codecs: Vec<String>,
    /// Maximum concurrent encode/decode sessions.
    pub max_sessions: u8,
}

impl HwAccelCaps {
    /// Returns `true` if this backend supports the given codec.
    #[must_use]
    pub fn supports_codec(&self, codec: &str) -> bool {
        let codec_lower = codec.to_lowercase();
        self.supported_codecs
            .iter()
            .any(|c| c.to_lowercase() == codec_lower)
    }

    /// Returns `true` if this backend supports the given resolution.
    #[must_use]
    pub fn supports_resolution(&self, width: u32, height: u32) -> bool {
        width <= self.max_resolution.0 && height <= self.max_resolution.1
    }
}

// ─── HwAccelConfig ────────────────────────────────────────────────────────────

/// Runtime configuration for using a hardware acceleration backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HwAccelConfig {
    /// The backend to use.
    pub backend: HwAccelBackend,
    /// Device index (for multi-GPU setups).
    pub device_index: u8,
    /// Fall back to software encoding if the HW backend fails.
    pub fallback_to_software: bool,
}

impl HwAccelConfig {
    /// Creates a new config for the given backend.
    #[must_use]
    pub fn new(backend: HwAccelBackend) -> Self {
        Self {
            backend,
            device_index: 0,
            fallback_to_software: true,
        }
    }

    /// Software-only configuration.
    #[must_use]
    pub fn software() -> Self {
        Self::new(HwAccelBackend::None)
    }

    /// Sets the device index.
    #[must_use]
    pub fn with_device(mut self, index: u8) -> Self {
        self.device_index = index;
        self
    }

    /// Disables fallback to software.
    #[must_use]
    pub fn no_fallback(mut self) -> Self {
        self.fallback_to_software = false;
        self
    }
}

// ─── HwCodecMapping ───────────────────────────────────────────────────────────

/// Maps hardware backends and codecs to encoder/decoder program names.
#[derive(Debug, Clone, Default)]
pub struct HwCodecMapping;

impl HwCodecMapping {
    /// Creates a new mapping.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the platform-specific encoder name for a backend+codec pair,
    /// or `None` if the combination is not supported.
    #[must_use]
    pub fn get_encoder_name(backend: &HwAccelBackend, codec: &str) -> Option<&'static str> {
        let codec_lower = codec.to_lowercase();
        match backend {
            HwAccelBackend::None => None,
            HwAccelBackend::Vaapi => match codec_lower.as_str() {
                "hevc" | "h265" => Some("hevc_vaapi"),
                "h264" | "avc" => Some("h264_vaapi"),
                "vp9" => Some("vp9_vaapi"),
                "av1" => Some("av1_vaapi"),
                "vp8" => Some("vp8_vaapi"),
                "mjpeg" => Some("mjpeg_vaapi"),
                _ => None,
            },
            HwAccelBackend::Nvenc => match codec_lower.as_str() {
                "hevc" | "h265" => Some("hevc_nvenc"),
                "h264" | "avc" => Some("h264_nvenc"),
                "av1" => Some("av1_nvenc"),
                _ => None,
            },
            HwAccelBackend::Videotoolbox => match codec_lower.as_str() {
                "hevc" | "h265" => Some("hevc_videotoolbox"),
                "h264" | "avc" => Some("h264_videotoolbox"),
                _ => None,
            },
            HwAccelBackend::Qsv => match codec_lower.as_str() {
                "hevc" | "h265" => Some("hevc_qsv"),
                "h264" | "avc" => Some("h264_qsv"),
                "vp9" => Some("vp9_qsv"),
                "av1" => Some("av1_qsv"),
                "mjpeg" => Some("mjpeg_qsv"),
                _ => None,
            },
            HwAccelBackend::Amf => match codec_lower.as_str() {
                "hevc" | "h265" => Some("hevc_amf"),
                "h264" | "avc" => Some("h264_amf"),
                "av1" => Some("av1_amf"),
                _ => None,
            },
            HwAccelBackend::D3d11Va => match codec_lower.as_str() {
                "hevc" | "h265" => Some("hevc_d3d11va"),
                "h264" | "avc" => Some("h264_d3d11va"),
                _ => None,
            },
        }
    }

    /// Convenience wrapper for owned types.
    #[must_use]
    pub fn encoder_name(backend: &HwAccelBackend, codec: &str) -> Option<&'static str> {
        Self::get_encoder_name(backend, codec)
    }
}

// ─── simulate_hw_caps ─────────────────────────────────────────────────────────

/// Returns plausible simulated hardware capabilities for a given backend.
///
/// Intended for testing and simulation — no real device detection occurs.
#[must_use]
pub fn simulate_hw_caps(backend: HwAccelBackend) -> HwAccelCaps {
    match backend {
        HwAccelBackend::None => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: true,
            max_resolution: (7680, 4320),
            supported_codecs: vec![
                "h264".to_string(),
                "hevc".to_string(),
                "vp9".to_string(),
                "av1".to_string(),
                "vp8".to_string(),
                "opus".to_string(),
                "flac".to_string(),
            ],
            max_sessions: 32,
        },
        HwAccelBackend::Vaapi => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: true,
            max_resolution: (4096, 4096),
            supported_codecs: vec![
                "h264".to_string(),
                "hevc".to_string(),
                "vp9".to_string(),
                "av1".to_string(),
                "vp8".to_string(),
                "mjpeg".to_string(),
            ],
            max_sessions: 8,
        },
        HwAccelBackend::Nvenc => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: true,
            max_resolution: (7680, 4320),
            supported_codecs: vec!["h264".to_string(), "hevc".to_string(), "av1".to_string()],
            max_sessions: 3,
        },
        HwAccelBackend::Videotoolbox => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: true,
            max_resolution: (4096, 2160),
            supported_codecs: vec!["h264".to_string(), "hevc".to_string()],
            max_sessions: 4,
        },
        HwAccelBackend::Qsv => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: true,
            max_resolution: (8192, 8192),
            supported_codecs: vec![
                "h264".to_string(),
                "hevc".to_string(),
                "vp9".to_string(),
                "av1".to_string(),
                "mjpeg".to_string(),
            ],
            max_sessions: 6,
        },
        HwAccelBackend::Amf => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: true,
            max_resolution: (7680, 4320),
            supported_codecs: vec!["h264".to_string(), "hevc".to_string(), "av1".to_string()],
            max_sessions: 4,
        },
        HwAccelBackend::D3d11Va => HwAccelCaps {
            backend,
            can_decode: true,
            can_encode: false, // D3D11VA is decode-only
            max_resolution: (4096, 2160),
            supported_codecs: vec!["h264".to_string(), "hevc".to_string()],
            max_sessions: 2,
        },
    }
}

// ─── LatencyMode ─────────────────────────────────────────────────────────────

/// Latency mode for the transcoding pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatencyMode {
    /// Offline / batch processing — optimise for throughput and quality.
    Batch,
    /// Low-latency mode with a bounded output delay.
    LowLatency {
        /// Maximum acceptable end-to-end delay in milliseconds.
        max_delay_ms: u32,
    },
    /// Hard real-time mode — frames must arrive at wall-clock rate.
    Realtime,
}

impl LatencyMode {
    /// Returns `true` for real-time or near-real-time modes.
    #[must_use]
    pub fn is_live(&self) -> bool {
        matches!(self, Self::LowLatency { .. } | Self::Realtime)
    }
}

// ─── PipelineConfig ───────────────────────────────────────────────────────────

/// Pipeline-level hardware and threading configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineHwConfig {
    /// Hardware acceleration settings.
    pub hw_accel: HwAccelConfig,
    /// Number of worker threads.
    pub thread_count: u8,
    /// Number of frame/packet buffers in flight.
    pub buffer_count: u8,
    /// Latency mode.
    pub latency_mode: LatencyMode,
}

impl PipelineHwConfig {
    /// Creates a sensible default configuration using software encoding.
    #[must_use]
    pub fn default_software() -> Self {
        Self {
            hw_accel: HwAccelConfig::software(),
            thread_count: 4,
            buffer_count: 8,
            latency_mode: LatencyMode::Batch,
        }
    }

    /// Creates a low-latency configuration with the given backend.
    #[must_use]
    pub fn low_latency(backend: HwAccelBackend, max_delay_ms: u32) -> Self {
        Self {
            hw_accel: HwAccelConfig::new(backend),
            thread_count: 2,
            buffer_count: 4,
            latency_mode: LatencyMode::LowLatency { max_delay_ms },
        }
    }
}

// ─── HwAccelSelector ─────────────────────────────────────────────────────────

/// Selects the best available hardware backend for a given task.
#[derive(Debug, Clone, Default)]
pub struct HwAccelSelector;

impl HwAccelSelector {
    /// Creates a new selector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Selects the best backend from `available` that supports `codec` at
    /// `resolution`.
    ///
    /// Selection criterion: highest `max_sessions` among qualifying backends
    /// that can encode and support the requested codec and resolution.
    ///
    /// Returns `None` if no suitable backend is found.
    #[must_use]
    pub fn select_best(
        available: &[HwAccelCaps],
        codec: &str,
        resolution: (u32, u32),
    ) -> Option<HwAccelConfig> {
        let (width, height) = resolution;
        available
            .iter()
            .filter(|caps| {
                caps.can_encode
                    && caps.supports_codec(codec)
                    && caps.supports_resolution(width, height)
                    && caps.backend != HwAccelBackend::None
            })
            .max_by_key(|caps| caps.max_sessions)
            .map(|caps| HwAccelConfig::new(caps.backend))
    }

    /// Selects the best backend or falls back to software.
    #[must_use]
    pub fn select_best_or_software(
        available: &[HwAccelCaps],
        codec: &str,
        resolution: (u32, u32),
    ) -> HwAccelConfig {
        Self::select_best(available, codec, resolution).unwrap_or_else(HwAccelConfig::software)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HwAccelBackend ────────────────────────────────────────────────────────

    #[test]
    fn test_backend_as_str() {
        assert_eq!(HwAccelBackend::None.as_str(), "none");
        assert_eq!(HwAccelBackend::Vaapi.as_str(), "vaapi");
        assert_eq!(HwAccelBackend::Nvenc.as_str(), "nvenc");
        assert_eq!(HwAccelBackend::Videotoolbox.as_str(), "videotoolbox");
        assert_eq!(HwAccelBackend::Qsv.as_str(), "qsv");
        assert_eq!(HwAccelBackend::Amf.as_str(), "amf");
        assert_eq!(HwAccelBackend::D3d11Va.as_str(), "d3d11va");
    }

    #[test]
    fn test_all_hw_contains_six_backends() {
        assert_eq!(HwAccelBackend::all_hw().len(), 6);
        assert!(!HwAccelBackend::all_hw().contains(&HwAccelBackend::None));
    }

    // ── HwCodecMapping ────────────────────────────────────────────────────────

    #[test]
    fn test_vaapi_hevc_encoder_name() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::Vaapi, "hevc"),
            Some("hevc_vaapi")
        );
    }

    #[test]
    fn test_nvenc_hevc_encoder_name() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::Nvenc, "hevc"),
            Some("hevc_nvenc")
        );
    }

    #[test]
    fn test_videotoolbox_hevc_encoder_name() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::Videotoolbox, "hevc"),
            Some("hevc_videotoolbox")
        );
    }

    #[test]
    fn test_qsv_h264_encoder_name() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::Qsv, "h264"),
            Some("h264_qsv")
        );
    }

    #[test]
    fn test_amf_av1_encoder_name() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::Amf, "av1"),
            Some("av1_amf")
        );
    }

    #[test]
    fn test_none_backend_returns_none() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::None, "h264"),
            None
        );
    }

    #[test]
    fn test_unsupported_codec_returns_none() {
        assert_eq!(
            HwCodecMapping::get_encoder_name(&HwAccelBackend::Nvenc, "vp9"),
            None
        );
    }

    // ── simulate_hw_caps ──────────────────────────────────────────────────────

    #[test]
    fn test_simulate_vaapi_can_encode_and_decode() {
        let caps = simulate_hw_caps(HwAccelBackend::Vaapi);
        assert!(caps.can_encode);
        assert!(caps.can_decode);
    }

    #[test]
    fn test_simulate_d3d11va_is_decode_only() {
        let caps = simulate_hw_caps(HwAccelBackend::D3d11Va);
        assert!(!caps.can_encode);
        assert!(caps.can_decode);
    }

    #[test]
    fn test_simulate_nvenc_supports_hevc() {
        let caps = simulate_hw_caps(HwAccelBackend::Nvenc);
        assert!(caps.supports_codec("hevc"));
    }

    #[test]
    fn test_simulate_nvenc_does_not_support_vp9() {
        let caps = simulate_hw_caps(HwAccelBackend::Nvenc);
        assert!(!caps.supports_codec("vp9"));
    }

    #[test]
    fn test_simulate_resolution_check() {
        let caps = simulate_hw_caps(HwAccelBackend::Videotoolbox);
        assert!(caps.supports_resolution(1920, 1080));
        assert!(!caps.supports_resolution(7680, 4320)); // exceeds max
    }

    #[test]
    fn test_simulate_none_is_unlimited() {
        let caps = simulate_hw_caps(HwAccelBackend::None);
        assert!(caps.supports_resolution(7680, 4320));
    }

    // ── HwAccelConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_hw_accel_config_new() {
        let cfg = HwAccelConfig::new(HwAccelBackend::Vaapi);
        assert_eq!(cfg.backend, HwAccelBackend::Vaapi);
        assert_eq!(cfg.device_index, 0);
        assert!(cfg.fallback_to_software);
    }

    #[test]
    fn test_hw_accel_config_no_fallback() {
        let cfg = HwAccelConfig::new(HwAccelBackend::Nvenc).no_fallback();
        assert!(!cfg.fallback_to_software);
    }

    #[test]
    fn test_hw_accel_config_with_device() {
        let cfg = HwAccelConfig::new(HwAccelBackend::Nvenc).with_device(2);
        assert_eq!(cfg.device_index, 2);
    }

    // ── LatencyMode ───────────────────────────────────────────────────────────

    #[test]
    fn test_latency_mode_is_live() {
        assert!(!LatencyMode::Batch.is_live());
        assert!(LatencyMode::LowLatency { max_delay_ms: 100 }.is_live());
        assert!(LatencyMode::Realtime.is_live());
    }

    #[test]
    fn test_latency_mode_equality() {
        assert_eq!(
            LatencyMode::LowLatency { max_delay_ms: 200 },
            LatencyMode::LowLatency { max_delay_ms: 200 }
        );
        assert_ne!(
            LatencyMode::LowLatency { max_delay_ms: 100 },
            LatencyMode::LowLatency { max_delay_ms: 200 }
        );
    }

    // ── HwAccelSelector ───────────────────────────────────────────────────────

    #[test]
    fn test_selector_picks_highest_sessions() {
        let caps = vec![
            simulate_hw_caps(HwAccelBackend::Vaapi), // max_sessions = 8
            simulate_hw_caps(HwAccelBackend::Nvenc), // max_sessions = 3
            simulate_hw_caps(HwAccelBackend::Qsv),   // max_sessions = 6
        ];
        let cfg = HwAccelSelector::select_best(&caps, "h264", (1920, 1080));
        assert!(cfg.is_some());
        assert_eq!(
            cfg.expect("should have config").backend,
            HwAccelBackend::Vaapi
        );
    }

    #[test]
    fn test_selector_filters_unsupported_codec() {
        // D3D11VA doesn't support encode, so only Videotoolbox fits for hevc
        let caps = vec![
            simulate_hw_caps(HwAccelBackend::D3d11Va),
            simulate_hw_caps(HwAccelBackend::Videotoolbox),
        ];
        let cfg = HwAccelSelector::select_best(&caps, "hevc", (1920, 1080));
        assert!(cfg.is_some());
        assert_eq!(
            cfg.expect("should have config").backend,
            HwAccelBackend::Videotoolbox
        );
    }

    #[test]
    fn test_selector_returns_none_when_no_match() {
        let caps = vec![simulate_hw_caps(HwAccelBackend::D3d11Va)]; // decode only
        let cfg = HwAccelSelector::select_best(&caps, "h264", (1920, 1080));
        assert!(cfg.is_none());
    }

    #[test]
    fn test_selector_fallback_to_software() {
        let caps: Vec<HwAccelCaps> = vec![];
        let cfg = HwAccelSelector::select_best_or_software(&caps, "h264", (1920, 1080));
        assert_eq!(cfg.backend, HwAccelBackend::None);
    }

    #[test]
    fn test_pipeline_hw_config_default_software() {
        let cfg = PipelineHwConfig::default_software();
        assert_eq!(cfg.hw_accel.backend, HwAccelBackend::None);
        assert_eq!(cfg.latency_mode, LatencyMode::Batch);
    }

    #[test]
    fn test_pipeline_hw_config_low_latency() {
        let cfg = PipelineHwConfig::low_latency(HwAccelBackend::Nvenc, 50);
        assert!(cfg.latency_mode.is_live());
        assert_eq!(cfg.hw_accel.backend, HwAccelBackend::Nvenc);
    }
}
