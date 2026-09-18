//! Media quality metrics (bitrate, quality scores, encoding stats).

use serde::{Deserialize, Serialize};

/// Media quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityMetrics {
    /// Bitrate metrics.
    pub bitrate: BitrateMetrics,
    /// Quality scores.
    pub scores: QualityScore,
    /// Encoding time per frame.
    pub encoding_time_per_frame_ms: f64,
    /// Keyframe interval (in frames).
    pub keyframe_interval: u32,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Bitrate metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BitrateMetrics {
    /// Current video bitrate (bps).
    pub video_bitrate_bps: u64,
    /// Average video bitrate (bps).
    pub avg_video_bitrate_bps: u64,
    /// Peak video bitrate (bps).
    pub peak_video_bitrate_bps: u64,
    /// Current audio bitrate (bps).
    pub audio_bitrate_bps: u64,
    /// Average audio bitrate (bps).
    pub avg_audio_bitrate_bps: u64,
    /// Total bitrate (video + audio).
    pub total_bitrate_bps: u64,
}

impl BitrateMetrics {
    /// Get video bitrate in Mbps.
    #[must_use]
    pub fn video_bitrate_mbps(&self) -> f64 {
        self.video_bitrate_bps as f64 / 1_000_000.0
    }

    /// Get audio bitrate in kbps.
    #[must_use]
    pub fn audio_bitrate_kbps(&self) -> f64 {
        self.audio_bitrate_bps as f64 / 1000.0
    }

    /// Get total bitrate in Mbps.
    #[must_use]
    pub fn total_bitrate_mbps(&self) -> f64 {
        self.total_bitrate_bps as f64 / 1_000_000.0
    }
}

/// Quality assessment scores.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityScore {
    /// PSNR (Peak Signal-to-Noise Ratio) in dB.
    pub psnr: Option<f64>,
    /// SSIM (Structural Similarity Index) 0.0 to 1.0.
    pub ssim: Option<f64>,
    /// VMAF (Video Multimethod Assessment Fusion) 0-100.
    pub vmaf: Option<f64>,
    /// Custom quality score 0-100.
    pub custom_score: Option<f64>,
}

impl QualityScore {
    /// Check if PSNR is acceptable (>= 30 dB is generally good).
    #[must_use]
    pub fn is_psnr_acceptable(&self) -> bool {
        self.psnr.map_or(true, |psnr| psnr >= 30.0)
    }

    /// Check if SSIM is acceptable (>= 0.95 is generally good).
    #[must_use]
    pub fn is_ssim_acceptable(&self) -> bool {
        self.ssim.map_or(true, |ssim| ssim >= 0.95)
    }

    /// Check if VMAF is acceptable (>= 80 is generally good).
    #[must_use]
    pub fn is_vmaf_acceptable(&self) -> bool {
        self.vmaf.map_or(true, |vmaf| vmaf >= 80.0)
    }

    /// Check if overall quality is acceptable.
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.is_psnr_acceptable() && self.is_ssim_acceptable() && self.is_vmaf_acceptable()
    }
}

/// Quality metrics tracker.
pub struct QualityMetricsTracker {
    current: parking_lot::RwLock<QualityMetrics>,
}

impl QualityMetricsTracker {
    /// Create a new quality metrics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: parking_lot::RwLock::new(QualityMetrics::default()),
        }
    }

    /// Update bitrate metrics.
    pub fn update_bitrate(&self, video_bps: u64, audio_bps: u64) {
        let mut metrics = self.current.write();
        metrics.bitrate.video_bitrate_bps = video_bps;
        metrics.bitrate.audio_bitrate_bps = audio_bps;
        metrics.bitrate.total_bitrate_bps = video_bps + audio_bps;

        // Update averages (exponential moving average)
        let alpha = 0.1;
        metrics.bitrate.avg_video_bitrate_bps = (alpha * video_bps as f64
            + (1.0 - alpha) * metrics.bitrate.avg_video_bitrate_bps as f64)
            as u64;
        metrics.bitrate.avg_audio_bitrate_bps = (alpha * audio_bps as f64
            + (1.0 - alpha) * metrics.bitrate.avg_audio_bitrate_bps as f64)
            as u64;

        // Update peak
        if video_bps > metrics.bitrate.peak_video_bitrate_bps {
            metrics.bitrate.peak_video_bitrate_bps = video_bps;
        }

        metrics.timestamp = chrono::Utc::now();
    }

    /// Update quality scores.
    pub fn update_scores(&self, psnr: Option<f64>, ssim: Option<f64>, vmaf: Option<f64>) {
        let mut metrics = self.current.write();
        metrics.scores.psnr = psnr;
        metrics.scores.ssim = ssim;
        metrics.scores.vmaf = vmaf;
        metrics.timestamp = chrono::Utc::now();
    }

    /// Update encoding time per frame.
    pub fn update_encoding_time(&self, time_ms: f64) {
        let mut metrics = self.current.write();
        metrics.encoding_time_per_frame_ms = time_ms;
        metrics.timestamp = chrono::Utc::now();
    }

    /// Update keyframe interval.
    pub fn update_keyframe_interval(&self, interval: u32) {
        let mut metrics = self.current.write();
        metrics.keyframe_interval = interval;
        metrics.timestamp = chrono::Utc::now();
    }

    /// Get current metrics snapshot.
    #[must_use]
    pub fn snapshot(&self) -> QualityMetrics {
        self.current.read().clone()
    }
}

impl Default for QualityMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitrate_metrics_conversions() {
        let metrics = BitrateMetrics {
            video_bitrate_bps: 5_000_000,
            audio_bitrate_bps: 128_000,
            total_bitrate_bps: 5_128_000,
            ..Default::default()
        };

        assert_eq!(metrics.video_bitrate_mbps(), 5.0);
        assert_eq!(metrics.audio_bitrate_kbps(), 128.0);
        assert_eq!(metrics.total_bitrate_mbps(), 5.128);
    }

    #[test]
    fn test_quality_score_acceptance() {
        let mut score = QualityScore {
            psnr: Some(35.0),
            ssim: Some(0.98),
            vmaf: Some(85.0),
            custom_score: None,
        };

        assert!(score.is_psnr_acceptable());
        assert!(score.is_ssim_acceptable());
        assert!(score.is_vmaf_acceptable());
        assert!(score.is_acceptable());

        score.psnr = Some(25.0); // Below threshold
        assert!(!score.is_psnr_acceptable());
        assert!(!score.is_acceptable());
    }

    #[test]
    fn test_quality_score_none_values() {
        let score = QualityScore {
            psnr: None,
            ssim: None,
            vmaf: None,
            custom_score: None,
        };

        // None values should be considered acceptable
        assert!(score.is_psnr_acceptable());
        assert!(score.is_ssim_acceptable());
        assert!(score.is_vmaf_acceptable());
        assert!(score.is_acceptable());
    }

    #[test]
    fn test_quality_metrics_tracker() {
        let tracker = QualityMetricsTracker::new();

        tracker.update_bitrate(5_000_000, 128_000);
        tracker.update_scores(Some(35.0), Some(0.98), Some(85.0));
        tracker.update_encoding_time(16.67);
        tracker.update_keyframe_interval(60);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.bitrate.video_bitrate_bps, 5_000_000);
        assert_eq!(snapshot.bitrate.audio_bitrate_bps, 128_000);
        assert_eq!(snapshot.scores.psnr, Some(35.0));
        assert_eq!(snapshot.encoding_time_per_frame_ms, 16.67);
        assert_eq!(snapshot.keyframe_interval, 60);
    }

    #[test]
    fn test_bitrate_averaging() {
        let tracker = QualityMetricsTracker::new();

        tracker.update_bitrate(5_000_000, 128_000);
        tracker.update_bitrate(6_000_000, 128_000);
        tracker.update_bitrate(4_000_000, 128_000);

        let snapshot = tracker.snapshot();
        // Should be an exponential moving average
        assert!(snapshot.bitrate.avg_video_bitrate_bps > 0);
        assert_eq!(snapshot.bitrate.video_bitrate_bps, 4_000_000); // Last value
    }

    #[test]
    fn test_peak_bitrate_tracking() {
        let tracker = QualityMetricsTracker::new();

        tracker.update_bitrate(5_000_000, 128_000);
        tracker.update_bitrate(8_000_000, 128_000);
        tracker.update_bitrate(6_000_000, 128_000);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.bitrate.peak_video_bitrate_bps, 8_000_000);
    }
}
