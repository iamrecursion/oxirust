//! Forensic watermark embedding policy for DRM.
//!
//! Provides:
//! - [`WatermarkStrength`]: how aggressively the watermark affects signal quality
//! - [`WatermarkPayload`]: session/user/timestamp data packed into 64 bits
//! - [`ForensicWatermarkConfig`]: full watermark embedding configuration
//! - [`should_embed_watermark`]: decides whether a given frame should carry a watermark

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// WatermarkStrength
// ---------------------------------------------------------------------------

/// Strength of the embedded forensic watermark.
///
/// Higher strength means the watermark is more detectable (and robust) but
/// reduces perceptual signal quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WatermarkStrength {
    /// Completely invisible; SNR impact < 0.5 dB.
    Invisible,
    /// Barely perceptible; SNR impact ≈ 1 dB.
    Light,
    /// Moderate; SNR impact ≈ 3 dB.
    Medium,
    /// Strong; SNR impact ≈ 6 dB.
    Strong,
}

impl WatermarkStrength {
    /// Signal-to-noise ratio reduction in dB caused by this watermark strength.
    #[must_use]
    pub fn snr_reduction_db(self) -> f32 {
        match self {
            WatermarkStrength::Invisible => 0.0,
            WatermarkStrength::Light => 1.0,
            WatermarkStrength::Medium => 3.0,
            WatermarkStrength::Strong => 6.0,
        }
    }

    /// Returns `true` if this strength is completely invisible (no SNR impact).
    #[must_use]
    pub fn is_invisible(self) -> bool {
        matches!(self, WatermarkStrength::Invisible)
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            WatermarkStrength::Invisible => "invisible",
            WatermarkStrength::Light => "light",
            WatermarkStrength::Medium => "medium",
            WatermarkStrength::Strong => "strong",
        }
    }

    /// Returns `true` if the watermark is strong enough to survive re-encoding.
    ///
    /// Only `Medium` and `Strong` are considered robust.
    #[must_use]
    pub fn is_robust(self) -> bool {
        matches!(self, WatermarkStrength::Medium | WatermarkStrength::Strong)
    }
}

// ---------------------------------------------------------------------------
// WatermarkPayload
// ---------------------------------------------------------------------------

// Bit layout of the 64-bit packed value:
//   [63..48] = session_id  (16 bits)
//   [47..32] = user_id     (16 bits)
//   [31..0]  = timestamp   (32 bits)

const SESSION_BITS: u64 = 16;
const USER_BITS: u64 = 16;
// timestamp uses the remaining 32 bits

/// Payload embedded in a forensic watermark.
///
/// Packed bit layout (MSB → LSB):
/// `[15..0 session_id][15..0 user_id][31..0 timestamp_epoch]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatermarkPayload {
    /// Session identifier (truncated to 16 bits when encoding).
    pub session_id: u64,
    /// User identifier (truncated to 16 bits when encoding).
    pub user_id: u64,
    /// Unix timestamp in seconds (truncated to 32 bits when encoding).
    pub timestamp_epoch: u64,
}

impl WatermarkPayload {
    /// Create a new payload.
    #[must_use]
    pub const fn new(session_id: u64, user_id: u64, timestamp_epoch: u64) -> Self {
        Self {
            session_id,
            user_id,
            timestamp_epoch,
        }
    }

    /// Encode the payload into a 64-bit integer.
    ///
    /// Only the low bits of each field are preserved (16 / 16 / 32 respectively).
    #[must_use]
    pub fn encode(self) -> u64 {
        let session = (self.session_id & 0xFFFF) << (USER_BITS + 32);
        let user = (self.user_id & 0xFFFF) << 32;
        let ts = self.timestamp_epoch & 0xFFFF_FFFF;
        session | user | ts
    }

    /// Decode a 64-bit integer back into a `WatermarkPayload`.
    #[must_use]
    pub fn decode(bits: u64) -> Self {
        let session_id = (bits >> (USER_BITS + 32)) & 0xFFFF;
        let user_id = (bits >> 32) & 0xFFFF;
        let timestamp_epoch = bits & 0xFFFF_FFFF;
        Self {
            session_id,
            user_id,
            timestamp_epoch,
        }
    }

    /// Returns `true` if the payload is empty (all fields are zero).
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.session_id == 0 && self.user_id == 0 && self.timestamp_epoch == 0
    }
}

// ---------------------------------------------------------------------------
// ForensicWatermarkConfig
// ---------------------------------------------------------------------------

/// Configuration for forensic watermark embedding.
#[derive(Debug, Clone)]
pub struct ForensicWatermarkConfig {
    /// Watermark strength.
    pub strength: WatermarkStrength,
    /// Whether watermarking is enabled.
    pub enabled: bool,
    /// Embed a watermark every `interval_frames` frames.
    ///
    /// `0` means embed on every frame.
    pub interval_frames: u32,
}

impl ForensicWatermarkConfig {
    /// Default configuration for VOD (Video on Demand) content.
    ///
    /// Uses invisible strength, enabled, every 30 frames (~1 s at 30 fps).
    #[must_use]
    pub fn default_vod() -> Self {
        Self {
            strength: WatermarkStrength::Invisible,
            enabled: true,
            interval_frames: 30,
        }
    }

    /// Configuration for live streaming: light strength, every 60 frames.
    #[must_use]
    pub fn default_live() -> Self {
        Self {
            strength: WatermarkStrength::Light,
            enabled: true,
            interval_frames: 60,
        }
    }

    /// Configuration with watermarking disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            strength: WatermarkStrength::Invisible,
            enabled: false,
            interval_frames: 30,
        }
    }

    /// Returns `true` if the configuration will embed a watermark on any frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enabled
    }
}

// ---------------------------------------------------------------------------
// should_embed_watermark
// ---------------------------------------------------------------------------

/// Decide whether a watermark should be embedded into a specific video frame.
///
/// Returns `false` if:
/// - watermarking is disabled
/// - `interval_frames == 0` is handled by embedding every frame
///
/// Returns `true` when `frame_number % interval_frames == 0`.
#[must_use]
pub fn should_embed_watermark(config: &ForensicWatermarkConfig, frame_number: u64) -> bool {
    if !config.enabled {
        return false;
    }
    if config.interval_frames == 0 {
        return true;
    }
    frame_number % config.interval_frames as u64 == 0
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- WatermarkStrength ----

    #[test]
    fn test_snr_reduction_invisible() {
        assert_eq!(WatermarkStrength::Invisible.snr_reduction_db(), 0.0);
    }

    #[test]
    fn test_snr_reduction_values() {
        assert!((WatermarkStrength::Light.snr_reduction_db() - 1.0).abs() < f32::EPSILON);
        assert!((WatermarkStrength::Medium.snr_reduction_db() - 3.0).abs() < f32::EPSILON);
        assert!((WatermarkStrength::Strong.snr_reduction_db() - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_invisible() {
        assert!(WatermarkStrength::Invisible.is_invisible());
        assert!(!WatermarkStrength::Light.is_invisible());
    }

    #[test]
    fn test_is_robust() {
        assert!(!WatermarkStrength::Invisible.is_robust());
        assert!(!WatermarkStrength::Light.is_robust());
        assert!(WatermarkStrength::Medium.is_robust());
        assert!(WatermarkStrength::Strong.is_robust());
    }

    #[test]
    fn test_strength_ordering() {
        assert!(WatermarkStrength::Invisible < WatermarkStrength::Light);
        assert!(WatermarkStrength::Light < WatermarkStrength::Medium);
        assert!(WatermarkStrength::Medium < WatermarkStrength::Strong);
    }

    #[test]
    fn test_strength_name() {
        assert_eq!(WatermarkStrength::Invisible.name(), "invisible");
        assert_eq!(WatermarkStrength::Strong.name(), "strong");
    }

    // ---- WatermarkPayload ----

    #[test]
    fn test_payload_encode_decode_roundtrip() {
        let p = WatermarkPayload::new(0xABCD, 0x1234, 1_700_000_000);
        let bits = p.encode();
        let decoded = WatermarkPayload::decode(bits);
        // Low 16 bits of session (0xABCD), low 16 bits of user (0x1234),
        // low 32 bits of timestamp (1_700_000_000 fits in 32 bits)
        assert_eq!(decoded.session_id, p.session_id & 0xFFFF);
        assert_eq!(decoded.user_id, p.user_id & 0xFFFF);
        assert_eq!(decoded.timestamp_epoch, p.timestamp_epoch & 0xFFFF_FFFF);
    }

    #[test]
    fn test_payload_encode_zero() {
        let p = WatermarkPayload::new(0, 0, 0);
        assert_eq!(p.encode(), 0);
    }

    #[test]
    fn test_payload_is_empty() {
        let p = WatermarkPayload::new(0, 0, 0);
        assert!(p.is_empty());
    }

    #[test]
    fn test_payload_is_not_empty() {
        let p = WatermarkPayload::new(1, 0, 0);
        assert!(!p.is_empty());
    }

    #[test]
    fn test_payload_decode_encode_inverse() {
        let bits: u64 = 0x0012_0034_0000_ABCD;
        let p = WatermarkPayload::decode(bits);
        // Re-encode and compare
        assert_eq!(p.encode(), bits);
    }

    #[test]
    fn test_payload_field_isolation() {
        // Session in high bits, user in mid bits, ts in low bits
        let p = WatermarkPayload::new(1, 2, 3);
        let bits = p.encode();
        assert_ne!(bits, 0);
        let d = WatermarkPayload::decode(bits);
        assert_eq!(d.session_id, 1);
        assert_eq!(d.user_id, 2);
        assert_eq!(d.timestamp_epoch, 3);
    }

    // ---- ForensicWatermarkConfig ----

    #[test]
    fn test_default_vod_config() {
        let c = ForensicWatermarkConfig::default_vod();
        assert!(c.enabled);
        assert_eq!(c.interval_frames, 30);
        assert_eq!(c.strength, WatermarkStrength::Invisible);
    }

    #[test]
    fn test_disabled_config_is_not_active() {
        let c = ForensicWatermarkConfig::disabled();
        assert!(!c.is_active());
    }

    // ---- should_embed_watermark ----

    #[test]
    fn test_embed_disabled() {
        let c = ForensicWatermarkConfig::disabled();
        assert!(!should_embed_watermark(&c, 0));
        assert!(!should_embed_watermark(&c, 30));
    }

    #[test]
    fn test_embed_every_interval() {
        let c = ForensicWatermarkConfig::default_vod(); // interval = 30
        assert!(should_embed_watermark(&c, 0));
        assert!(!should_embed_watermark(&c, 1));
        assert!(!should_embed_watermark(&c, 29));
        assert!(should_embed_watermark(&c, 30));
        assert!(should_embed_watermark(&c, 60));
        assert!(!should_embed_watermark(&c, 61));
    }

    #[test]
    fn test_embed_zero_interval_every_frame() {
        let c = ForensicWatermarkConfig {
            strength: WatermarkStrength::Light,
            enabled: true,
            interval_frames: 0,
        };
        for frame in [0u64, 1, 2, 999, 100_000] {
            assert!(should_embed_watermark(&c, frame));
        }
    }

    #[test]
    fn test_embed_large_frame_number() {
        let c = ForensicWatermarkConfig::default_live(); // interval = 60
        assert!(should_embed_watermark(&c, 600));
        assert!(!should_embed_watermark(&c, 601));
    }
}
