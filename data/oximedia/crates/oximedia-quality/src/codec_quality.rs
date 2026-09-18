//! Codec-specific quality analysis.
//!
//! Provides types and utilities for assessing the quality of compressed video
//! as it relates to specific codec parameters such as quantiser, bitrate, and
//! blockiness artefacts.

#![allow(dead_code)]

/// Codec type enumeration for quality analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecType {
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// AV1
    Av1,
    /// VP9
    Vp9,
    /// Apple `ProRes`
    Prores,
    /// Avid `DNxHD`
    Dnxhd,
}

impl CodecType {
    /// Returns `true` if the codec uses lossy compression.
    #[must_use]
    pub fn is_lossy(&self) -> bool {
        match self {
            Self::H264 | Self::H265 | Self::Av1 | Self::Vp9 => true,
            // ProRes and DNxHD are visually lossless but technically lossy DCT codecs.
            Self::Prores | Self::Dnxhd => true,
        }
    }

    /// Returns a typical target bitrate in Mbit/s for the given resolution in mega-pixels.
    ///
    /// `resolution_mp` – resolution in mega-pixels (e.g. `2.07` for 1920×1080).
    #[must_use]
    pub fn typical_bitrate_mbps(&self, resolution_mp: f32) -> f32 {
        match self {
            Self::H264 => resolution_mp * 4.0,
            Self::H265 => resolution_mp * 2.5,
            Self::Av1 => resolution_mp * 1.8,
            Self::Vp9 => resolution_mp * 2.0,
            Self::Prores => resolution_mp * 50.0,
            Self::Dnxhd => resolution_mp * 40.0,
        }
    }
}

/// Valid CRF / QP range for a codec.
#[derive(Clone, Copy, Debug)]
pub struct QuantizerRange {
    /// Minimum quantiser value (best quality).
    pub min: u8,
    /// Maximum quantiser value (worst quality).
    pub max: u8,
}

impl QuantizerRange {
    /// Returns the standard H.264 CRF range (0–51).
    #[must_use]
    pub fn h264() -> Self {
        Self { min: 0, max: 51 }
    }

    /// Returns the standard H.265 CRF range (0–51).
    #[must_use]
    pub fn h265() -> Self {
        Self { min: 0, max: 51 }
    }

    /// Returns the AV1 CRF range (0–63).
    #[must_use]
    pub fn av1() -> Self {
        Self { min: 0, max: 63 }
    }

    /// Returns `true` if `crf` falls within an acceptable quality range (below midpoint).
    #[must_use]
    pub fn is_crf_acceptable(&self, crf: u8) -> bool {
        let midpoint = (u16::from(self.min) + u16::from(self.max)) / 2;
        u16::from(crf) <= midpoint
    }
}

/// Blockiness artefact score for a single frame.
#[derive(Clone, Copy, Debug)]
pub struct BlockinessScore {
    /// Overall blockiness score (0.0–1.0, higher = more blocking).
    pub score: f32,
    /// Dominant blocking frequency in cycles/pixel.
    pub blocking_frequency: f32,
}

impl BlockinessScore {
    /// Returns `true` when the blockiness is severe (score > 0.7).
    #[must_use]
    pub fn is_severe(&self) -> bool {
        self.score > 0.7
    }
}

/// Codec-specific quality report for a frame or segment.
#[derive(Clone, Debug)]
pub struct CodecQualityReport {
    /// Codec used for encoding.
    pub codec: CodecType,
    /// Encoded bitrate in kbit/s.
    pub bitrate_kbps: u32,
    /// Quantisation parameter used.
    pub qp: u8,
    /// Blockiness artefact score.
    pub blockiness: BlockinessScore,
}

impl CodecQualityReport {
    /// Returns `true` if the encoding meets typical broadcast quality criteria:
    /// - No severe blocking.
    /// - QP is in the acceptable half of the codec's range.
    /// - Bitrate > 1 000 kbit/s.
    #[must_use]
    pub fn is_broadcast_acceptable(&self) -> bool {
        if self.blockiness.is_severe() {
            return false;
        }
        if self.bitrate_kbps < 1_000 {
            return false;
        }
        let range = match self.codec {
            CodecType::H264 => QuantizerRange::h264(),
            CodecType::H265 => QuantizerRange::h265(),
            CodecType::Av1 => QuantizerRange::av1(),
            // ProRes / DNxHD don't use CRF in the same way – treat as acceptable.
            CodecType::Vp9 => QuantizerRange { min: 0, max: 63 },
            CodecType::Prores | CodecType::Dnxhd => return true,
        };
        range.is_crf_acceptable(self.qp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CodecType ──────────────────────────────────────────────────────────

    #[test]
    fn test_h264_is_lossy() {
        assert!(CodecType::H264.is_lossy());
    }

    #[test]
    fn test_av1_is_lossy() {
        assert!(CodecType::Av1.is_lossy());
    }

    #[test]
    fn test_prores_is_lossy() {
        assert!(CodecType::Prores.is_lossy());
    }

    #[test]
    fn test_typical_bitrate_h264_scales_with_resolution() {
        let hd = CodecType::H264.typical_bitrate_mbps(2.07);
        let uhd = CodecType::H264.typical_bitrate_mbps(8.29);
        assert!(uhd > hd);
    }

    #[test]
    fn test_av1_lower_bitrate_than_h264() {
        let mp = 2.07_f32;
        assert!(CodecType::Av1.typical_bitrate_mbps(mp) < CodecType::H264.typical_bitrate_mbps(mp));
    }

    #[test]
    fn test_prores_higher_bitrate_than_h264() {
        let mp = 2.07_f32;
        assert!(
            CodecType::Prores.typical_bitrate_mbps(mp) > CodecType::H264.typical_bitrate_mbps(mp)
        );
    }

    // ── QuantizerRange ─────────────────────────────────────────────────────

    #[test]
    fn test_h264_range() {
        let r = QuantizerRange::h264();
        assert_eq!(r.min, 0);
        assert_eq!(r.max, 51);
    }

    #[test]
    fn test_av1_range_max() {
        assert_eq!(QuantizerRange::av1().max, 63);
    }

    #[test]
    fn test_crf_acceptable_low_value() {
        assert!(QuantizerRange::h264().is_crf_acceptable(20));
    }

    #[test]
    fn test_crf_unacceptable_high_value() {
        assert!(!QuantizerRange::h264().is_crf_acceptable(40));
    }

    #[test]
    fn test_crf_acceptable_at_midpoint() {
        let r = QuantizerRange::h264(); // midpoint = 25
        assert!(r.is_crf_acceptable(25));
    }

    // ── BlockinessScore ────────────────────────────────────────────────────

    #[test]
    fn test_blockiness_severe_above_threshold() {
        let b = BlockinessScore {
            score: 0.8,
            blocking_frequency: 0.125,
        };
        assert!(b.is_severe());
    }

    #[test]
    fn test_blockiness_not_severe_below_threshold() {
        let b = BlockinessScore {
            score: 0.5,
            blocking_frequency: 0.125,
        };
        assert!(!b.is_severe());
    }

    #[test]
    fn test_blockiness_not_severe_at_threshold() {
        let b = BlockinessScore {
            score: 0.7,
            blocking_frequency: 0.0,
        };
        assert!(!b.is_severe());
    }

    // ── CodecQualityReport ─────────────────────────────────────────────────

    #[test]
    fn test_broadcast_acceptable_good_h264() {
        let report = CodecQualityReport {
            codec: CodecType::H264,
            bitrate_kbps: 8_000,
            qp: 20,
            blockiness: BlockinessScore {
                score: 0.1,
                blocking_frequency: 0.0,
            },
        };
        assert!(report.is_broadcast_acceptable());
    }

    #[test]
    fn test_broadcast_unacceptable_low_bitrate() {
        let report = CodecQualityReport {
            codec: CodecType::H264,
            bitrate_kbps: 500,
            qp: 20,
            blockiness: BlockinessScore {
                score: 0.1,
                blocking_frequency: 0.0,
            },
        };
        assert!(!report.is_broadcast_acceptable());
    }

    #[test]
    fn test_broadcast_unacceptable_severe_blocking() {
        let report = CodecQualityReport {
            codec: CodecType::H264,
            bitrate_kbps: 8_000,
            qp: 20,
            blockiness: BlockinessScore {
                score: 0.9,
                blocking_frequency: 0.125,
            },
        };
        assert!(!report.is_broadcast_acceptable());
    }

    #[test]
    fn test_broadcast_unacceptable_high_qp() {
        let report = CodecQualityReport {
            codec: CodecType::H264,
            bitrate_kbps: 8_000,
            qp: 40,
            blockiness: BlockinessScore {
                score: 0.1,
                blocking_frequency: 0.0,
            },
        };
        assert!(!report.is_broadcast_acceptable());
    }

    #[test]
    fn test_broadcast_acceptable_prores() {
        let report = CodecQualityReport {
            codec: CodecType::Prores,
            bitrate_kbps: 100_000,
            qp: 0,
            blockiness: BlockinessScore {
                score: 0.0,
                blocking_frequency: 0.0,
            },
        };
        assert!(report.is_broadcast_acceptable());
    }
}
