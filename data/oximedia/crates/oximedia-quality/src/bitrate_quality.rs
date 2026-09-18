#![allow(dead_code)]
//! Bitrate-based quality estimation and analysis.
//!
//! Provides tools for estimating video quality from bitrate information,
//! analyzing bitrate distributions, and determining whether a given bitrate
//! is adequate for a target resolution and codec.

/// Video codec family for bitrate estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecFamily {
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// AV1
    Av1,
    /// VP9
    Vp9,
    /// `ProRes` (constant quality, variable bitrate)
    ProRes,
}

/// Resolution tier for bitrate lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionTier {
    /// 640x360 or similar
    Sd360p,
    /// 854x480 or similar
    Sd480p,
    /// 1280x720
    Hd720p,
    /// 1920x1080
    Hd1080p,
    /// 2560x1440
    Qhd1440p,
    /// 3840x2160
    Uhd4k,
    /// 7680x4320
    Uhd8k,
}

impl ResolutionTier {
    /// Returns the total pixel count for this tier.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        match self {
            Self::Sd360p => 640 * 360,
            Self::Sd480p => 854 * 480,
            Self::Hd720p => 1280 * 720,
            Self::Hd1080p => 1920 * 1080,
            Self::Qhd1440p => 2560 * 1440,
            Self::Uhd4k => 3840 * 2160,
            Self::Uhd8k => 7680 * 4320,
        }
    }

    /// Detects the resolution tier from width and height.
    #[must_use]
    pub fn detect(width: u32, height: u32) -> Self {
        let pixels = u64::from(width) * u64::from(height);
        if pixels <= 300_000 {
            Self::Sd360p
        } else if pixels <= 500_000 {
            Self::Sd480p
        } else if pixels <= 1_100_000 {
            Self::Hd720p
        } else if pixels <= 2_500_000 {
            Self::Hd1080p
        } else if pixels <= 4_500_000 {
            Self::Qhd1440p
        } else if pixels <= 10_000_000 {
            Self::Uhd4k
        } else {
            Self::Uhd8k
        }
    }
}

/// Bitrate quality rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BitrateRating {
    /// Severely under-bitrated — visible artifacts expected
    Poor,
    /// Below recommended — some artifacts likely
    BelowAverage,
    /// Acceptable for streaming
    Acceptable,
    /// Good quality for the resolution and codec
    Good,
    /// Excellent / transparent quality
    Excellent,
}

/// Recommended bitrate ranges (in kbps) for a codec and resolution.
#[derive(Debug, Clone)]
pub struct BitrateRecommendation {
    /// Codec family
    pub codec: CodecFamily,
    /// Resolution tier
    pub resolution: ResolutionTier,
    /// Minimum acceptable bitrate (kbps)
    pub min_kbps: u32,
    /// Recommended bitrate (kbps)
    pub recommended_kbps: u32,
    /// High-quality bitrate (kbps)
    pub high_kbps: u32,
}

/// Returns recommended bitrate ranges for common codec/resolution combos.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn recommend_bitrate(codec: CodecFamily, resolution: ResolutionTier) -> BitrateRecommendation {
    let (min, rec, high) = match (codec, resolution) {
        (CodecFamily::H264, ResolutionTier::Sd360p) => (400, 800, 1500),
        (CodecFamily::H264, ResolutionTier::Sd480p) => (750, 1500, 2500),
        (CodecFamily::H264, ResolutionTier::Hd720p) => (1500, 3000, 5000),
        (CodecFamily::H264, ResolutionTier::Hd1080p) => (3000, 6000, 10000),
        (CodecFamily::H264, ResolutionTier::Qhd1440p) => (6000, 10000, 16000),
        (CodecFamily::H264, ResolutionTier::Uhd4k) => (13000, 20000, 35000),
        (CodecFamily::H264, ResolutionTier::Uhd8k) => (40000, 60000, 100000),

        (CodecFamily::H265, ResolutionTier::Sd360p) => (250, 500, 1000),
        (CodecFamily::H265, ResolutionTier::Sd480p) => (500, 1000, 1800),
        (CodecFamily::H265, ResolutionTier::Hd720p) => (1000, 2000, 3500),
        (CodecFamily::H265, ResolutionTier::Hd1080p) => (2000, 4000, 7000),
        (CodecFamily::H265, ResolutionTier::Qhd1440p) => (4000, 7000, 12000),
        (CodecFamily::H265, ResolutionTier::Uhd4k) => (8000, 15000, 25000),
        (CodecFamily::H265, ResolutionTier::Uhd8k) => (25000, 40000, 70000),

        (CodecFamily::Av1, ResolutionTier::Sd360p) => (200, 400, 800),
        (CodecFamily::Av1, ResolutionTier::Sd480p) => (400, 800, 1500),
        (CodecFamily::Av1, ResolutionTier::Hd720p) => (800, 1500, 2800),
        (CodecFamily::Av1, ResolutionTier::Hd1080p) => (1500, 3000, 5500),
        (CodecFamily::Av1, ResolutionTier::Qhd1440p) => (3000, 5500, 9000),
        (CodecFamily::Av1, ResolutionTier::Uhd4k) => (6000, 12000, 20000),
        (CodecFamily::Av1, ResolutionTier::Uhd8k) => (20000, 35000, 55000),

        (CodecFamily::Vp9, ResolutionTier::Sd360p) => (250, 500, 900),
        (CodecFamily::Vp9, ResolutionTier::Sd480p) => (500, 900, 1600),
        (CodecFamily::Vp9, ResolutionTier::Hd720p) => (900, 1800, 3200),
        (CodecFamily::Vp9, ResolutionTier::Hd1080p) => (1800, 3500, 6000),
        (CodecFamily::Vp9, ResolutionTier::Qhd1440p) => (3500, 6000, 10000),
        (CodecFamily::Vp9, ResolutionTier::Uhd4k) => (7000, 13000, 22000),
        (CodecFamily::Vp9, ResolutionTier::Uhd8k) => (22000, 38000, 60000),

        (CodecFamily::ProRes, ResolutionTier::Sd360p) => (10000, 20000, 40000),
        (CodecFamily::ProRes, ResolutionTier::Sd480p) => (20000, 35000, 60000),
        (CodecFamily::ProRes, ResolutionTier::Hd720p) => (35000, 60000, 100000),
        (CodecFamily::ProRes, ResolutionTier::Hd1080p) => (60000, 120000, 200000),
        (CodecFamily::ProRes, ResolutionTier::Qhd1440p) => (100000, 180000, 300000),
        (CodecFamily::ProRes, ResolutionTier::Uhd4k) => (200000, 350000, 600000),
        (CodecFamily::ProRes, ResolutionTier::Uhd8k) => (600000, 900000, 1500000),
    };
    BitrateRecommendation {
        codec,
        resolution,
        min_kbps: min,
        recommended_kbps: rec,
        high_kbps: high,
    }
}

/// Rates the quality of a given bitrate for a codec and resolution.
#[must_use]
pub fn rate_bitrate(
    bitrate_kbps: u32,
    codec: CodecFamily,
    resolution: ResolutionTier,
) -> BitrateRating {
    let rec = recommend_bitrate(codec, resolution);
    if bitrate_kbps < rec.min_kbps {
        BitrateRating::Poor
    } else if bitrate_kbps < rec.recommended_kbps {
        BitrateRating::BelowAverage
    } else if bitrate_kbps < rec.high_kbps {
        BitrateRating::Good
    } else {
        BitrateRating::Excellent
    }
}

/// Bitrate statistics computed from a stream of per-frame bitrate samples.
#[derive(Debug, Clone)]
pub struct BitrateStats {
    /// Average bitrate in kbps
    pub avg_kbps: f64,
    /// Peak bitrate in kbps
    pub peak_kbps: f64,
    /// Minimum bitrate in kbps
    pub min_kbps: f64,
    /// Standard deviation of bitrate in kbps
    pub stddev_kbps: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Ratio of peak to average (burstiness)
    pub peak_to_avg_ratio: f64,
}

/// Computes bitrate statistics from a series of per-frame sizes in bytes and a frame rate.
#[allow(clippy::cast_precision_loss)]
pub fn compute_bitrate_stats(frame_sizes_bytes: &[u32], fps: f64) -> BitrateStats {
    if frame_sizes_bytes.is_empty() {
        return BitrateStats {
            avg_kbps: 0.0,
            peak_kbps: 0.0,
            min_kbps: 0.0,
            stddev_kbps: 0.0,
            sample_count: 0,
            peak_to_avg_ratio: 0.0,
        };
    }

    let kbps_values: Vec<f64> = frame_sizes_bytes
        .iter()
        .map(|&sz| sz as f64 * 8.0 / 1000.0 * fps)
        .collect();

    let n = kbps_values.len() as f64;
    let sum: f64 = kbps_values.iter().sum();
    let avg = sum / n;

    let peak = kbps_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min = kbps_values.iter().copied().fold(f64::INFINITY, f64::min);

    let variance = kbps_values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    let ratio = if avg > 0.0 { peak / avg } else { 0.0 };

    BitrateStats {
        avg_kbps: avg,
        peak_kbps: peak,
        min_kbps: min,
        stddev_kbps: stddev,
        sample_count: frame_sizes_bytes.len(),
        peak_to_avg_ratio: ratio,
    }
}

/// Estimates VMAF-like quality score (0-100) from bitrate, codec, and resolution.
///
/// This is a rough heuristic approximation, not an actual VMAF computation.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_quality_from_bitrate(
    bitrate_kbps: u32,
    codec: CodecFamily,
    resolution: ResolutionTier,
) -> f64 {
    let rec = recommend_bitrate(codec, resolution);
    let ratio = bitrate_kbps as f64 / rec.recommended_kbps as f64;
    // Logarithmic curve saturating around 100
    let raw = 20.0 * ratio.ln() + 80.0;
    raw.clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_detect_1080p() {
        assert_eq!(ResolutionTier::detect(1920, 1080), ResolutionTier::Hd1080p);
    }

    #[test]
    fn test_resolution_detect_4k() {
        assert_eq!(ResolutionTier::detect(3840, 2160), ResolutionTier::Uhd4k);
    }

    #[test]
    fn test_resolution_pixel_count() {
        assert_eq!(ResolutionTier::Hd1080p.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_recommend_bitrate_h264_1080p() {
        let rec = recommend_bitrate(CodecFamily::H264, ResolutionTier::Hd1080p);
        assert!(rec.min_kbps < rec.recommended_kbps);
        assert!(rec.recommended_kbps < rec.high_kbps);
    }

    #[test]
    fn test_rate_bitrate_poor() {
        let rating = rate_bitrate(100, CodecFamily::H264, ResolutionTier::Hd1080p);
        assert_eq!(rating, BitrateRating::Poor);
    }

    #[test]
    fn test_rate_bitrate_good() {
        let rating = rate_bitrate(7000, CodecFamily::H264, ResolutionTier::Hd1080p);
        assert_eq!(rating, BitrateRating::Good);
    }

    #[test]
    fn test_rate_bitrate_excellent() {
        let rating = rate_bitrate(50000, CodecFamily::H264, ResolutionTier::Hd1080p);
        assert_eq!(rating, BitrateRating::Excellent);
    }

    #[test]
    fn test_compute_bitrate_stats_basic() {
        let sizes = vec![10000, 12000, 8000, 11000, 9000];
        let stats = compute_bitrate_stats(&sizes, 25.0);
        assert_eq!(stats.sample_count, 5);
        assert!(stats.avg_kbps > 0.0);
        assert!(stats.peak_kbps >= stats.avg_kbps);
        assert!(stats.min_kbps <= stats.avg_kbps);
    }

    #[test]
    fn test_compute_bitrate_stats_empty() {
        let stats = compute_bitrate_stats(&[], 25.0);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.avg_kbps, 0.0);
    }

    #[test]
    fn test_compute_bitrate_stats_stddev() {
        let sizes = vec![10000, 10000, 10000];
        let stats = compute_bitrate_stats(&sizes, 30.0);
        assert!(stats.stddev_kbps < 0.01);
    }

    #[test]
    fn test_estimate_quality_high_bitrate() {
        let q = estimate_quality_from_bitrate(20000, CodecFamily::H264, ResolutionTier::Hd1080p);
        assert!(q > 70.0);
    }

    #[test]
    fn test_estimate_quality_low_bitrate() {
        let q = estimate_quality_from_bitrate(500, CodecFamily::H264, ResolutionTier::Uhd4k);
        assert!(q < 50.0);
    }

    #[test]
    fn test_estimate_quality_clamped() {
        let q = estimate_quality_from_bitrate(1000000, CodecFamily::H264, ResolutionTier::Sd360p);
        assert!(q <= 100.0);
    }

    #[test]
    fn test_codec_efficiency_ordering() {
        // AV1 should recommend lower bitrates than H264 for same quality
        let h264 = recommend_bitrate(CodecFamily::H264, ResolutionTier::Hd1080p);
        let av1 = recommend_bitrate(CodecFamily::Av1, ResolutionTier::Hd1080p);
        assert!(av1.recommended_kbps < h264.recommended_kbps);
    }
}
