//! Transcode optimization helpers for `oximedia-optimize`.
//!
//! Provides high-level goal-oriented optimization: given a target (file size,
//! quality, or streaming bitrate), suggest the best CRF and encoding settings.

#![allow(dead_code)]

/// High-level goal that drives the optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationGoal {
    /// Minimize file size while maintaining acceptable quality.
    MinimizeSize,
    /// Maximize perceptual quality regardless of file size.
    MaximizeQuality,
    /// Hit a specific streaming bitrate as closely as possible.
    TargetBitrate,
    /// Balance quality and file size equally.
    Balanced,
    /// Optimize for fast real-time encoding (e.g. live streaming).
    RealTime,
}

impl OptimizationGoal {
    /// Returns a human-readable description of the goal.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::MinimizeSize => "Minimize output file size",
            Self::MaximizeQuality => "Maximize perceptual quality",
            Self::TargetBitrate => "Hit a specific target bitrate",
            Self::Balanced => "Balance quality and file size",
            Self::RealTime => "Optimize for real-time encoding speed",
        }
    }

    /// Returns the default CRF value associated with this goal (H.264/H.265 scale).
    #[must_use]
    pub fn default_crf(&self) -> u8 {
        match self {
            Self::MinimizeSize => 30,
            Self::MaximizeQuality => 16,
            Self::TargetBitrate => 23,
            Self::Balanced => 23,
            Self::RealTime => 28,
        }
    }
}

impl Default for OptimizationGoal {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Configuration describing the input and desired output for a transcode job.
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    /// Input video width in pixels.
    pub width: u32,
    /// Input video height in pixels.
    pub height: u32,
    /// Frame rate.
    pub fps: f64,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Constant Rate Factor (0 = lossless, 51 = worst, 23 = default).
    pub crf: u8,
    /// Average audio bitrate in kbps (e.g. 128, 192, 320).
    pub audio_kbps: u32,
    /// Target bitrate override in kbps (used with `TargetBitrate` goal).
    pub target_bitrate_kbps: Option<f64>,
}

impl TranscodeConfig {
    /// Creates a new transcode config.
    #[must_use]
    pub fn new(width: u32, height: u32, fps: f64, duration_secs: f64, crf: u8) -> Self {
        Self {
            width,
            height,
            fps,
            duration_secs,
            crf,
            audio_kbps: 192,
            target_bitrate_kbps: None,
        }
    }

    /// Estimates the output file size in megabytes based on CRF, resolution and duration.
    ///
    /// Uses a simplified empirical formula. Actual sizes vary by content.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_size_mb(&self) -> f64 {
        let pixels_per_sec = self.width as f64 * self.height as f64 * self.fps;
        let crf_scale = (-f64::from(self.crf) / 6.0_f64).exp2();
        let video_kbps = (pixels_per_sec / 500.0) * crf_scale;
        let total_kbps = video_kbps + f64::from(self.audio_kbps);
        // Convert kbps × seconds to megabytes: kbps × sec / 8 / 1024
        total_kbps * self.duration_secs / 8.0 / 1024.0
    }

    /// Returns the pixel count (width × height).
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns `true` if this config describes a 4K or larger resolution.
    #[must_use]
    pub fn is_4k_or_above(&self) -> bool {
        self.width >= 3840 || self.height >= 2160
    }
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self::new(1920, 1080, 25.0, 60.0, 23)
    }
}

/// Optimizer that adjusts [`TranscodeConfig`] fields to meet an
/// [`OptimizationGoal`].
#[derive(Debug)]
pub struct TranscodeOptimizer {
    goal: OptimizationGoal,
    /// Maximum acceptable file size in MB (used for `MinimizeSize` goal).
    max_size_mb: Option<f64>,
    /// Target bitrate in kbps (used for `TargetBitrate` goal).
    target_kbps: Option<f64>,
}

impl TranscodeOptimizer {
    /// Creates a new optimizer for the given goal.
    #[must_use]
    pub fn new(goal: OptimizationGoal) -> Self {
        Self {
            goal,
            max_size_mb: None,
            target_kbps: None,
        }
    }

    /// Sets the maximum acceptable file size for `MinimizeSize` goal.
    #[must_use]
    pub fn with_max_size_mb(mut self, mb: f64) -> Self {
        self.max_size_mb = Some(mb);
        self
    }

    /// Sets the target bitrate for `TargetBitrate` goal.
    #[must_use]
    pub fn with_target_kbps(mut self, kbps: f64) -> Self {
        self.target_kbps = Some(kbps);
        self
    }

    /// Returns the active optimization goal.
    #[must_use]
    pub fn goal(&self) -> OptimizationGoal {
        self.goal
    }

    /// Returns an optimized [`TranscodeConfig`] derived from the input config
    /// that best satisfies the optimizer's goal.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn optimize_for_goal(&self, config: &TranscodeConfig) -> TranscodeConfig {
        let mut out = config.clone();
        match self.goal {
            OptimizationGoal::MinimizeSize => {
                out.crf = 30; // Higher CRF = smaller file
                out.audio_kbps = 128;
            }
            OptimizationGoal::MaximizeQuality => {
                out.crf = 16; // Lower CRF = better quality
                out.audio_kbps = 320;
            }
            OptimizationGoal::TargetBitrate => {
                if let Some(target) = self.target_kbps {
                    out.crf = self.suggest_crf(config, target);
                    out.target_bitrate_kbps = Some(target);
                }
            }
            OptimizationGoal::Balanced => {
                out.crf = 23;
                out.audio_kbps = 192;
            }
            OptimizationGoal::RealTime => {
                out.crf = 28;
                out.audio_kbps = 128;
            }
        }
        out
    }

    /// Suggests a CRF value that would produce output close to `target_kbps`.
    ///
    /// Uses binary-search style iteration over the CRF range [0, 51].
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn suggest_crf(&self, config: &TranscodeConfig, target_kbps: f64) -> u8 {
        let duration = config.duration_secs;
        if duration <= 0.0 {
            return 23;
        }
        let mut best_crf: u8 = 23;
        let mut best_delta = f64::MAX;
        for crf in 0u8..=51u8 {
            let probe = TranscodeConfig {
                crf,
                ..config.clone()
            };
            let size_mb = probe.estimated_size_mb();
            let kbps = size_mb * 8.0 * 1024.0 / duration;
            let delta = (kbps - target_kbps).abs();
            if delta < best_delta {
                best_delta = delta;
                best_crf = crf;
            }
        }
        best_crf
    }

    /// Returns the estimated output size in MB for a given config under the
    /// current optimization goal.
    #[must_use]
    pub fn estimated_output_size_mb(&self, config: &TranscodeConfig) -> f64 {
        self.optimize_for_goal(config).estimated_size_mb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_description_non_empty() {
        for g in [
            OptimizationGoal::MinimizeSize,
            OptimizationGoal::MaximizeQuality,
            OptimizationGoal::TargetBitrate,
            OptimizationGoal::Balanced,
            OptimizationGoal::RealTime,
        ] {
            assert!(!g.description().is_empty());
        }
    }

    #[test]
    fn test_goal_default_crf_in_range() {
        for g in [
            OptimizationGoal::MinimizeSize,
            OptimizationGoal::MaximizeQuality,
            OptimizationGoal::Balanced,
            OptimizationGoal::RealTime,
        ] {
            let crf = g.default_crf();
            assert!(crf <= 51, "CRF {crf} out of range for {g:?}");
        }
    }

    #[test]
    fn test_transcode_config_estimated_size_positive() {
        let cfg = TranscodeConfig::default();
        assert!(cfg.estimated_size_mb() > 0.0);
    }

    #[test]
    fn test_transcode_config_lower_crf_larger_size() {
        let low = TranscodeConfig::new(1920, 1080, 25.0, 60.0, 16);
        let high = TranscodeConfig::new(1920, 1080, 25.0, 60.0, 30);
        assert!(low.estimated_size_mb() > high.estimated_size_mb());
    }

    #[test]
    fn test_transcode_config_pixel_count() {
        let cfg = TranscodeConfig::new(1920, 1080, 25.0, 60.0, 23);
        assert_eq!(cfg.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_transcode_config_is_4k_above() {
        let uhd = TranscodeConfig::new(3840, 2160, 24.0, 120.0, 23);
        assert!(uhd.is_4k_or_above());
    }

    #[test]
    fn test_transcode_config_not_4k() {
        let hd = TranscodeConfig::new(1280, 720, 30.0, 60.0, 23);
        assert!(!hd.is_4k_or_above());
    }

    #[test]
    fn test_optimizer_goal_accessor() {
        let opt = TranscodeOptimizer::new(OptimizationGoal::MinimizeSize);
        assert_eq!(opt.goal(), OptimizationGoal::MinimizeSize);
    }

    #[test]
    fn test_optimize_for_minimize_size_increases_crf() {
        let cfg = TranscodeConfig::default(); // crf=23
        let opt = TranscodeOptimizer::new(OptimizationGoal::MinimizeSize);
        let out = opt.optimize_for_goal(&cfg);
        assert!(out.crf > cfg.crf);
    }

    #[test]
    fn test_optimize_for_max_quality_decreases_crf() {
        let cfg = TranscodeConfig::default(); // crf=23
        let opt = TranscodeOptimizer::new(OptimizationGoal::MaximizeQuality);
        let out = opt.optimize_for_goal(&cfg);
        assert!(out.crf < cfg.crf);
    }

    #[test]
    fn test_suggest_crf_in_range() {
        let cfg = TranscodeConfig::default();
        let opt = TranscodeOptimizer::new(OptimizationGoal::TargetBitrate);
        let crf = opt.suggest_crf(&cfg, 2000.0);
        assert!(crf <= 51);
    }

    #[test]
    fn test_suggest_crf_zero_duration_returns_default() {
        let mut cfg = TranscodeConfig::default();
        cfg.duration_secs = 0.0;
        let opt = TranscodeOptimizer::new(OptimizationGoal::TargetBitrate);
        assert_eq!(opt.suggest_crf(&cfg, 2000.0), 23);
    }

    #[test]
    fn test_optimize_for_target_bitrate_sets_crf() {
        let cfg = TranscodeConfig::default();
        let opt = TranscodeOptimizer::new(OptimizationGoal::TargetBitrate).with_target_kbps(4000.0);
        let out = opt.optimize_for_goal(&cfg);
        assert!(out.crf <= 51);
    }

    #[test]
    fn test_estimated_output_size_positive() {
        let cfg = TranscodeConfig::default();
        let opt = TranscodeOptimizer::new(OptimizationGoal::Balanced);
        assert!(opt.estimated_output_size_mb(&cfg) > 0.0);
    }
}
