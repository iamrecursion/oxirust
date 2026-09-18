//! Bitrate control and rate estimation for transcoding pipelines.
//!
//! Provides rate-control mode selection, target bitrate specification,
//! and a rolling estimator for measuring live encoding throughput.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Rate-control algorithm used during encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Maintain a fixed output bitrate.
    ConstantBitrate,
    /// Allow bitrate to vary within min/peak bounds.
    VariableBitrate,
    /// Use a constant rate factor (quality-based).
    ConstantRateFactor,
    /// Encode to a constant perceptual quality level.
    ConstantQuality,
}

impl RateControlMode {
    /// Returns `true` for quality-based modes (`ConstantRateFactor`, `ConstantQuality`).
    #[must_use]
    pub fn is_quality_based(&self) -> bool {
        matches!(self, Self::ConstantRateFactor | Self::ConstantQuality)
    }

    /// Returns a short human-readable description of the mode.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::ConstantBitrate => "CBR – constant bitrate",
            Self::VariableBitrate => "VBR – variable bitrate",
            Self::ConstantRateFactor => "CRF – constant rate factor",
            Self::ConstantQuality => "CQ – constant quality",
        }
    }
}

/// Target bitrate specification with peak, average, and minimum limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetBitrate {
    /// Peak bitrate in kbps.
    pub peak_kbps: u32,
    /// Average bitrate in kbps.
    pub avg_kbps: u32,
    /// Minimum bitrate in kbps (may be 0 for unconstrained).
    pub min_kbps: u32,
}

impl TargetBitrate {
    /// Creates a VBR `TargetBitrate` with a peak and average; `min_kbps` is set
    /// to half of `avg_kbps`.
    #[must_use]
    pub fn with_peak(peak: u32, avg: u32) -> Self {
        Self {
            peak_kbps: peak,
            avg_kbps: avg,
            min_kbps: avg / 2,
        }
    }

    /// Creates a CBR `TargetBitrate` where peak, average, and minimum are all
    /// equal to `kbps`.
    #[must_use]
    pub fn cbr(kbps: u32) -> Self {
        Self {
            peak_kbps: kbps,
            avg_kbps: kbps,
            min_kbps: kbps,
        }
    }

    /// Returns the ratio of peak to average bitrate.
    ///
    /// Returns `1.0` if `avg_kbps` is zero.
    #[must_use]
    pub fn peak_to_avg_ratio(&self) -> f32 {
        if self.avg_kbps == 0 {
            return 1.0;
        }
        self.peak_kbps as f32 / self.avg_kbps as f32
    }

    /// Returns `true` if peak and average differ (VBR behaviour).
    #[must_use]
    pub fn is_vbr(&self) -> bool {
        self.peak_kbps != self.avg_kbps
    }
}

/// Tracks per-frame sizes and derives a rolling bitrate estimate.
#[derive(Debug, Clone)]
pub struct BitrateEstimator {
    /// Rate-control mode in use.
    pub mode: RateControlMode,
    /// History of frame sizes in bytes (most recent last).
    pub history: Vec<u32>,
}

impl BitrateEstimator {
    /// Creates a new estimator for the given mode.
    #[must_use]
    pub fn new(mode: RateControlMode) -> Self {
        Self {
            mode,
            history: Vec::new(),
        }
    }

    /// Appends a frame's encoded size (in bytes) to the history.
    pub fn add_frame_size_bytes(&mut self, size: u32) {
        self.history.push(size);
    }

    /// Estimates the current bitrate in kbps given the encoding frame rate.
    ///
    /// Returns `0.0` if no frames have been recorded or `fps` is zero/negative.
    #[must_use]
    pub fn current_kbps(&self, fps: f32) -> f32 {
        if self.history.is_empty() || fps <= 0.0 {
            return 0.0;
        }
        let avg_bytes: f32 =
            self.history.iter().map(|&b| b as f32).sum::<f32>() / self.history.len() as f32;
        // bytes/frame * frames/sec * 8 bits/byte / 1000 = kbps
        avg_bytes * fps * 8.0 / 1000.0
    }

    /// Returns the variance of frame sizes (in bytes²).
    ///
    /// Returns `0.0` if fewer than two frames have been recorded.
    #[must_use]
    pub fn variance(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean: f32 =
            self.history.iter().map(|&b| b as f32).sum::<f32>() / self.history.len() as f32;
        let var: f32 = self
            .history
            .iter()
            .map(|&b| {
                let diff = b as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / self.history.len() as f32;
        var
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RateControlMode ---

    #[test]
    fn test_cbr_not_quality_based() {
        assert!(!RateControlMode::ConstantBitrate.is_quality_based());
    }

    #[test]
    fn test_vbr_not_quality_based() {
        assert!(!RateControlMode::VariableBitrate.is_quality_based());
    }

    #[test]
    fn test_crf_is_quality_based() {
        assert!(RateControlMode::ConstantRateFactor.is_quality_based());
    }

    #[test]
    fn test_cq_is_quality_based() {
        assert!(RateControlMode::ConstantQuality.is_quality_based());
    }

    #[test]
    fn test_description_not_empty() {
        for mode in [
            RateControlMode::ConstantBitrate,
            RateControlMode::VariableBitrate,
            RateControlMode::ConstantRateFactor,
            RateControlMode::ConstantQuality,
        ] {
            assert!(!mode.description().is_empty());
        }
    }

    // --- TargetBitrate ---

    #[test]
    fn test_cbr_all_equal() {
        let tb = TargetBitrate::cbr(5000);
        assert_eq!(tb.peak_kbps, 5000);
        assert_eq!(tb.avg_kbps, 5000);
        assert_eq!(tb.min_kbps, 5000);
        assert!(!tb.is_vbr());
    }

    #[test]
    fn test_with_peak_is_vbr() {
        let tb = TargetBitrate::with_peak(8000, 5000);
        assert_eq!(tb.peak_kbps, 8000);
        assert_eq!(tb.avg_kbps, 5000);
        assert!(tb.is_vbr());
    }

    #[test]
    fn test_peak_to_avg_ratio() {
        let tb = TargetBitrate::with_peak(10000, 5000);
        assert!((tb.peak_to_avg_ratio() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_peak_to_avg_ratio_zero_avg() {
        let tb = TargetBitrate {
            peak_kbps: 100,
            avg_kbps: 0,
            min_kbps: 0,
        };
        assert!((tb.peak_to_avg_ratio() - 1.0).abs() < 1e-4);
    }

    // --- BitrateEstimator ---

    #[test]
    fn test_new_empty_history() {
        let est = BitrateEstimator::new(RateControlMode::ConstantBitrate);
        assert!(est.history.is_empty());
    }

    #[test]
    fn test_current_kbps_no_frames() {
        let est = BitrateEstimator::new(RateControlMode::ConstantBitrate);
        assert_eq!(est.current_kbps(30.0), 0.0);
    }

    #[test]
    fn test_current_kbps_single_frame() {
        let mut est = BitrateEstimator::new(RateControlMode::ConstantBitrate);
        // 125 bytes/frame at 30fps → 125 * 30 * 8 / 1000 = 30 kbps
        est.add_frame_size_bytes(125);
        assert!((est.current_kbps(30.0) - 30.0).abs() < 1e-3);
    }

    #[test]
    fn test_current_kbps_zero_fps() {
        let mut est = BitrateEstimator::new(RateControlMode::ConstantBitrate);
        est.add_frame_size_bytes(1000);
        assert_eq!(est.current_kbps(0.0), 0.0);
    }

    #[test]
    fn test_variance_zero_single_frame() {
        let mut est = BitrateEstimator::new(RateControlMode::VariableBitrate);
        est.add_frame_size_bytes(500);
        assert_eq!(est.variance(), 0.0);
    }

    #[test]
    fn test_variance_uniform_frames() {
        let mut est = BitrateEstimator::new(RateControlMode::VariableBitrate);
        for _ in 0..10 {
            est.add_frame_size_bytes(1000);
        }
        assert!((est.variance() - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_variance_non_zero() {
        let mut est = BitrateEstimator::new(RateControlMode::VariableBitrate);
        est.add_frame_size_bytes(100);
        est.add_frame_size_bytes(200);
        // mean = 150, variance = ((50)^2 + (50)^2) / 2 = 2500
        assert!((est.variance() - 2500.0).abs() < 1e-3);
    }
}
