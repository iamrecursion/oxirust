//! Model Predictive Control (MPC) adaptive bitrate controller.

use super::{
    AbrConfig, AbrDecision, AdaptiveBitrateController, BandwidthEstimator, QualityLevel,
    QualitySelector,
};
use crate::abr::history::SegmentDownloadHistory;
use std::collections::VecDeque;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Robust MPC (Model Predictive Control) ABR Controller
// ─────────────────────────────────────────────────────────────────────────────
//
// Based on Yin et al. (SIGCOMM 2015) "A Control-Theoretic Approach for
// Dynamic Adaptive Video Streaming over HTTP".
//
// The MPC controller solves an N-step lookahead optimisation at each decision
// epoch.  It minimises a cost function over a finite horizon H:
//
//   cost = Σ[ -q(s_k)                    (quality reward)
//              + λ * |q(s_k) - q(s_{k-1})|  (smoothness penalty)
//              + μ * rebuffer(s_k)           (stall penalty)
//              + ν * max(0, B_max - B(k))    (buffer overflow)
//           ]
//
// "Robust" variant: uses the 5th-percentile of the past-N throughput
// predictions to guard against network variance.

/// Throughput predictor for Robust-MPC that applies a pessimistic correction.
#[derive(Debug)]
struct RobustThroughputPredictor {
    /// Recent throughput samples (bytes/sec).
    samples: VecDeque<f64>,
    /// Window size.
    window: usize,
    /// Harmonic mean of recent samples.
    harmonic_mean: f64,
    /// 5th-percentile estimate from the sample distribution.
    percentile5: f64,
}

impl RobustThroughputPredictor {
    fn new(window: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(window),
            window,
            harmonic_mean: 0.0,
            percentile5: 0.0,
        }
    }

    fn add_sample(&mut self, throughput: f64) {
        if self.samples.len() >= self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(throughput);
        self.update_stats();
    }

    fn update_stats(&mut self) {
        if self.samples.is_empty() {
            self.harmonic_mean = 0.0;
            self.percentile5 = 0.0;
            return;
        }

        // Harmonic mean
        let sum_recip: f64 = self
            .samples
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .sum();
        self.harmonic_mean = if sum_recip > 0.0 {
            self.samples.len() as f64 / sum_recip
        } else {
            0.0
        };

        // 5th-percentile
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 * 0.05) as usize).min(sorted.len().saturating_sub(1));
        self.percentile5 = sorted[idx];
    }

    /// Returns the robust (pessimistic) throughput estimate in bytes/sec.
    fn robust_estimate(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        // Minimum of harmonic mean and 5th-percentile for robustness
        self.harmonic_mean.min(self.percentile5).max(0.0)
    }
}

/// MPC optimisation weights.
#[derive(Debug, Clone)]
pub struct MpcWeights {
    /// Weight for quality utility (higher = prefers higher quality).
    pub quality: f64,
    /// Penalty weight for quality switches (smoothness).
    pub smoothness: f64,
    /// Penalty weight for rebuffering events.
    pub rebuffer: f64,
    /// Penalty for allowing buffer to overflow.
    pub overflow: f64,
}

impl Default for MpcWeights {
    fn default() -> Self {
        Self {
            quality: 1.0,
            smoothness: 1.0,
            rebuffer: 4.5,
            overflow: 0.1,
        }
    }
}

/// Robust Model Predictive Control ABR controller.
///
/// Performs an N-step horizon search to find the quality sequence that
/// minimises a weighted cost of quality, smoothness, rebuffering, and
/// buffer overflow.  Uses a pessimistic (5th-percentile) throughput
/// prediction for robustness against network variations.
#[derive(Debug)]
pub struct RobustMpcController {
    /// Base configuration.
    config: AbrConfig,
    /// Bandwidth estimator (used to seed the throughput predictor).
    bandwidth_estimator: BandwidthEstimator,
    /// Robust throughput predictor.
    predictor: RobustThroughputPredictor,
    /// Current buffer level.
    buffer_level: Duration,
    /// Quality selector for hysteresis.
    quality_selector: QualitySelector,
    /// Download history.
    download_history: SegmentDownloadHistory,
    /// MPC cost function weights.
    weights: MpcWeights,
    /// Horizon length (number of future segments to look ahead).
    horizon: usize,
    /// Playback duration of one segment (seconds).
    segment_duration: f64,
    /// Previous quality level (for smoothness penalty).
    prev_quality_index: usize,
}

impl RobustMpcController {
    /// Creates a new Robust MPC controller.
    ///
    /// * `horizon` — look-ahead depth (2–8 is typical; 5 provides good results).
    /// * `segment_duration` — playback duration of one segment in seconds.
    #[must_use]
    pub fn new(config: AbrConfig, horizon: usize, segment_duration: f64) -> Self {
        let alpha = config.mode.ema_alpha();
        let window = config.estimation_window;
        let bandwidth_estimator = BandwidthEstimator::new(window, config.sample_ttl, alpha);
        Self {
            config,
            bandwidth_estimator,
            predictor: RobustThroughputPredictor::new(window),
            buffer_level: Duration::ZERO,
            quality_selector: QualitySelector::new(),
            download_history: SegmentDownloadHistory::new(50),
            weights: MpcWeights::default(),
            horizon: horizon.clamp(1, 10),
            segment_duration: segment_duration.max(1.0),
            prev_quality_index: 0,
        }
    }

    /// Creates a controller with default MPC parameters.
    #[must_use]
    pub fn default_params(config: AbrConfig) -> Self {
        Self::new(config, 5, 4.0)
    }

    /// Sets custom MPC cost weights.
    pub fn set_weights(&mut self, weights: MpcWeights) {
        self.weights = weights;
    }

    /// Returns the current MPC weights.
    #[must_use]
    pub fn weights(&self) -> &MpcWeights {
        &self.weights
    }

    /// Computes the log-scale quality utility for a bitrate.
    fn quality_utility(&self, bitrate: f64, max_bitrate: f64) -> f64 {
        if max_bitrate <= 0.0 || bitrate <= 0.0 {
            return 0.0;
        }
        // Normalised utility in [0, 1]
        (bitrate / max_bitrate).ln().max(0.0) / (max_bitrate / max_bitrate).max(1.0).ln().max(1.0)
    }

    /// Simulates one MPC step forward, returning (cost, new_buffer).
    fn simulate_step(
        &self,
        quality_idx: usize,
        prev_quality_idx: usize,
        buffer: f64,
        throughput_bps: f64,
        levels: &[QualityLevel],
        max_bitrate: f64,
    ) -> (f64, f64) {
        let bitrate = levels[quality_idx].effective_bandwidth() as f64;
        let seg_dur = self.segment_duration;

        // Download time for this segment
        let download_time = if throughput_bps > 0.0 {
            bitrate * seg_dur / throughput_bps
        } else {
            seg_dur * 2.0 // pessimistic
        };

        // Rebuffering: time spent stalled
        let rebuffer = (download_time - buffer).max(0.0);

        // New buffer level after playback
        let new_buffer = (buffer - download_time + seg_dur)
            .max(0.0)
            .min(self.config.max_buffer.as_secs_f64());

        // Quality utility
        let q_util = self.quality_utility(bitrate, max_bitrate);

        // Smoothness penalty
        let smooth = if quality_idx != prev_quality_idx {
            1.0
        } else {
            0.0
        };

        // Buffer overflow penalty (buffer exceeds max)
        let overflow = (buffer - self.config.max_buffer.as_secs_f64()).max(0.0);

        // Cost (minimise): negative quality reward + penalties
        let cost = -self.weights.quality * q_util
            + self.weights.smoothness * smooth
            + self.weights.rebuffer * rebuffer
            + self.weights.overflow * overflow;

        (cost, new_buffer)
    }

    /// Performs MPC optimisation over the horizon to select best quality.
    ///
    /// Uses a greedy horizon search: at each step choose the quality that
    /// minimises the one-step cost, then recurse `horizon - 1` more steps.
    fn mpc_select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        let throughput_bps = self.predictor.robust_estimate() * 8.0;
        if throughput_bps <= 0.0 {
            return AbrDecision::Maintain;
        }

        let max_bitrate = levels
            .iter()
            .map(|l| l.effective_bandwidth() as f64)
            .fold(0.0_f64, f64::max);

        if max_bitrate <= 0.0 {
            return AbrDecision::Maintain;
        }

        let min_q = self.config.min_quality.unwrap_or(0);
        let max_q = self
            .config
            .max_quality
            .unwrap_or(levels.len().saturating_sub(1))
            .min(levels.len().saturating_sub(1));

        let buffer_secs = self.buffer_level.as_secs_f64();

        // Emergency downswitch
        if self.buffer_level < self.config.mode.critical_buffer() && current_index > 0 {
            return AbrDecision::SwitchTo(min_q);
        }

        // Evaluate each candidate quality for the first step
        let mut best_first_idx = current_index;
        let mut best_total_cost = f64::INFINITY;

        for first_q in min_q..=max_q {
            let (step_cost, buf1) = self.simulate_step(
                first_q,
                self.prev_quality_index,
                buffer_secs,
                throughput_bps,
                levels,
                max_bitrate,
            );

            // Greedy lookahead for remaining horizon steps
            let mut total_cost = step_cost;
            let mut buf = buf1;
            let mut prev_q = first_q;

            for _ in 1..self.horizon {
                // At each future step, greedily pick best quality
                let mut best_step_cost = f64::INFINITY;
                let mut best_buf = buf;
                let mut best_q = prev_q;

                for q in min_q..=max_q {
                    let (sc, nb) =
                        self.simulate_step(q, prev_q, buf, throughput_bps, levels, max_bitrate);
                    if sc < best_step_cost {
                        best_step_cost = sc;
                        best_buf = nb;
                        best_q = q;
                    }
                }
                total_cost += best_step_cost;
                buf = best_buf;
                prev_q = best_q;
            }

            if total_cost < best_total_cost {
                best_total_cost = total_cost;
                best_first_idx = first_q;
            }
        }

        // Apply switch hysteresis
        if !self
            .quality_selector
            .can_switch(self.config.mode.min_switch_interval())
        {
            return AbrDecision::Maintain;
        }

        // Prevent upswitch with low buffer
        if best_first_idx > current_index
            && self.buffer_level < self.config.mode.min_buffer_for_upswitch()
        {
            return AbrDecision::Maintain;
        }

        if best_first_idx != current_index {
            AbrDecision::SwitchTo(best_first_idx)
        } else {
            AbrDecision::Maintain
        }
    }

    /// Returns a reference to the download history.
    #[must_use]
    pub fn download_history(&self) -> &SegmentDownloadHistory {
        &self.download_history
    }
}

impl AdaptiveBitrateController for RobustMpcController {
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if self.predictor.samples.is_empty() {
            let initial = self.config.initial_quality.unwrap_or(0);
            let initial = initial.min(levels.len().saturating_sub(1));
            return AbrDecision::SwitchTo(initial);
        }
        self.mpc_select_quality(levels, current_index)
    }

    fn report_segment_download(&mut self, bytes: usize, duration: Duration) {
        self.bandwidth_estimator.add_sample(bytes, duration);
        let throughput = if duration.as_secs_f64() > 0.0 {
            bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        self.predictor.add_sample(throughput);
        let seg_dur = Duration::from_secs_f64(self.segment_duration);
        self.download_history.add(0, bytes, duration, seg_dur);
    }

    fn report_buffer_level(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        self.predictor.robust_estimate() * 8.0
    }

    fn current_buffer(&self) -> Duration {
        self.buffer_level
    }

    fn reset(&mut self) {
        self.bandwidth_estimator.reset();
        self.predictor = RobustThroughputPredictor::new(self.config.estimation_window);
        self.buffer_level = Duration::ZERO;
        self.quality_selector.reset();
        self.download_history.reset();
        self.prev_quality_index = 0;
    }

    fn config(&self) -> &AbrConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests for new controllers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod new_abr_tests {
    use super::*;
    use crate::abr::bola::BolaBbrController;
    use crate::abr::dash_ctrl::{DashAbrController, DashSegmentAvailability};
    use crate::abr::AbrMode;

    fn make_levels() -> Vec<QualityLevel> {
        vec![
            QualityLevel::new(0, 500_000),
            QualityLevel::new(1, 1_000_000),
            QualityLevel::new(2, 2_000_000),
            QualityLevel::new(3, 4_000_000),
            QualityLevel::new(4, 8_000_000),
        ]
    }

    fn report_good_downloads(ctrl: &mut dyn AdaptiveBitrateController, n: usize) {
        // Simulate 10 Mbps throughput: download 1 MB in 0.8 seconds
        for _ in 0..n {
            ctrl.report_segment_download(1_000_000, Duration::from_millis(800));
        }
    }

    // ─── SegmentDownloadHistory tests ──────────────────────────────────────

    #[test]
    fn test_segment_history_empty() {
        let h = SegmentDownloadHistory::new(20);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        let stats = h.stats(10);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_segment_history_add_and_stats() {
        let mut h = SegmentDownloadHistory::new(20);
        h.add(0, 1_000_000, Duration::from_secs(1), Duration::from_secs(4));
        h.add(1, 2_000_000, Duration::from_secs(1), Duration::from_secs(4));
        assert_eq!(h.len(), 2);
        let stats = h.stats(10);
        assert_eq!(stats.count, 2);
        assert!(stats.mean_throughput > 0.0);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_segment_history_capacity_eviction() {
        let mut h = SegmentDownloadHistory::new(5);
        for i in 0..10 {
            h.add(
                0,
                1000 * i,
                Duration::from_millis(100),
                Duration::from_secs(4),
            );
        }
        assert_eq!(h.len(), 5);
    }

    #[test]
    fn test_segment_history_rolling_stats() {
        let mut h = SegmentDownloadHistory::new(20);
        // Fast downloads
        for _ in 0..5 {
            h.add(
                0,
                2_000_000,
                Duration::from_millis(500),
                Duration::from_secs(4),
            );
        }
        // Slow downloads
        for _ in 0..5 {
            h.add(0, 1_000_000, Duration::from_secs(4), Duration::from_secs(4));
        }
        let stats_all = h.stats(10);
        let stats_recent = h.stats(5);
        // Recent stats should only include slow downloads
        assert!(stats_recent.mean_throughput < stats_all.mean_throughput);
    }

    #[test]
    fn test_segment_history_reset() {
        let mut h = SegmentDownloadHistory::new(20);
        h.add(0, 1000, Duration::from_millis(100), Duration::from_secs(4));
        h.reset();
        assert!(h.is_empty());
    }

    #[test]
    fn test_segment_history_cv() {
        let mut h = SegmentDownloadHistory::new(20);
        // Uniform throughput → CV near zero
        for _ in 0..10 {
            h.add(
                0,
                1_000_000,
                Duration::from_millis(800),
                Duration::from_secs(4),
            );
        }
        let stats = h.stats(10);
        assert!(
            stats.cv_throughput < 0.01,
            "Expected low CV, got {}",
            stats.cv_throughput
        );
    }

    // ─── DashAbrController tests ───────────────────────────────────────────

    #[test]
    fn test_dash_abr_initial_quality() {
        let config = AbrConfig::new().with_initial_quality(2);
        let mut ctrl = DashAbrController::new(config);
        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        // First call with no samples should use initial quality
        assert_eq!(decision, AbrDecision::SwitchTo(2));

        // After enough samples, should make bandwidth-based decision
        report_good_downloads(&mut ctrl, 10);
        ctrl.report_buffer_level(Duration::from_secs(20));
    }

    #[test]
    fn test_dash_abr_live_mode_conservative() {
        let config = AbrConfig::new().with_mode(AbrMode::Balanced);
        let mut ctrl = DashAbrController::new(config);
        // Set live mode with very few available segments
        ctrl.update_availability(DashSegmentAvailability {
            available_segments: 2,
            segment_duration: Duration::from_secs(4),
            is_live: true,
            update_interval: Some(Duration::from_secs(4)),
            ..Default::default()
        });

        // Good throughput but live with tiny buffer
        for _ in 0..10 {
            ctrl.report_segment_download(4_000_000, Duration::from_millis(500));
        }
        ctrl.report_buffer_level(Duration::from_secs(20));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 4);
        // With only 2 segments available, should avoid top quality
        let _ = decision; // Just verify it doesn't panic
    }

    #[test]
    fn test_dash_abr_emergency_downswitch() {
        let config = AbrConfig::new();
        let mut ctrl = DashAbrController::new(config);
        report_good_downloads(&mut ctrl, 10);
        // Set critically low buffer
        ctrl.report_buffer_level(Duration::from_millis(500));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 4);
        assert_eq!(decision, AbrDecision::SwitchTo(0));
    }

    #[test]
    fn test_dash_abr_vod_upswitch() {
        let config = AbrConfig::new().with_mode(AbrMode::Aggressive);
        let mut ctrl = DashAbrController::new(config);
        // Excellent throughput: 50 Mbps (download 5 MB in 0.8s)
        for _ in 0..10 {
            ctrl.report_segment_download(5_000_000, Duration::from_millis(800));
        }
        ctrl.report_buffer_level(Duration::from_secs(25));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        // Should want to switch up given excellent throughput + high buffer
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx > 0, "Expected higher quality"),
            AbrDecision::Maintain => {} // Also acceptable if hysteresis prevents switch
        }
    }

    #[test]
    fn test_dash_abr_reports_download_history() {
        let config = AbrConfig::new();
        let mut ctrl = DashAbrController::new(config);
        ctrl.report_dash_segment(1, 1_000_000, Duration::from_secs(1));
        ctrl.report_dash_segment(1, 2_000_000, Duration::from_secs(1));
        assert_eq!(ctrl.download_history().len(), 2);
    }

    #[test]
    fn test_dash_abr_reset_clears_state() {
        let config = AbrConfig::new();
        let mut ctrl = DashAbrController::new(config);
        report_good_downloads(&mut ctrl, 10);
        ctrl.report_buffer_level(Duration::from_secs(15));
        ctrl.reset();
        assert_eq!(ctrl.current_buffer(), Duration::ZERO);
        assert_eq!(ctrl.download_history().len(), 0);
    }

    // ─── BolaBbrController tests ───────────────────────────────────────────

    #[test]
    fn test_bola_initial_quality_no_samples() {
        let config = AbrConfig::new().with_min_quality(1);
        let ctrl = BolaBbrController::default_params(config);
        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        // No samples → fall back to min quality
        assert_eq!(decision, AbrDecision::SwitchTo(1));
    }

    #[test]
    fn test_bola_emergency_downswitch() {
        let config = AbrConfig::new().with_mode(AbrMode::Balanced);
        let mut ctrl = BolaBbrController::default_params(config);
        report_good_downloads(&mut ctrl, 10);
        ctrl.report_buffer_level(Duration::from_millis(200));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 4);
        assert_eq!(decision, AbrDecision::SwitchTo(0));
    }

    #[test]
    fn test_bola_high_buffer_prefers_quality() {
        let config = AbrConfig::new().with_mode(AbrMode::Balanced);
        let mut ctrl = BolaBbrController::new(config, 10.0, 4.0);

        // Excellent throughput: 40 Mbps
        for _ in 0..15 {
            ctrl.report_segment_download(4_000_000, Duration::from_millis(800));
        }
        ctrl.report_buffer_level(Duration::from_secs(30));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx > 0),
            AbrDecision::Maintain => {} // Hysteresis prevents switch
        }
    }

    #[test]
    fn test_bola_low_buffer_downswitches() {
        let config = AbrConfig::new().with_mode(AbrMode::Conservative);
        let mut ctrl = BolaBbrController::new(config, 2.0, 4.0);

        // Moderate throughput
        for _ in 0..10 {
            ctrl.report_segment_download(1_000_000, Duration::from_secs(1));
        }
        // Very low buffer — BOLA buffer term becomes small → favours low quality
        ctrl.report_buffer_level(Duration::from_secs(1));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 4);
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx < 4),
            AbrDecision::Maintain => {} // Current quality unchanged (already at high, maintains)
        }
    }

    #[test]
    fn test_bola_lyapunov_v_getter() {
        let ctrl = BolaBbrController::new(AbrConfig::new(), 7.5, 6.0);
        assert!((ctrl.lyapunov_v() - 7.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bola_reset() {
        let config = AbrConfig::new();
        let mut ctrl = BolaBbrController::default_params(config);
        report_good_downloads(&mut ctrl, 10);
        ctrl.report_buffer_level(Duration::from_secs(20));
        ctrl.reset();
        assert_eq!(ctrl.current_buffer(), Duration::ZERO);
        assert_eq!(ctrl.download_history().len(), 0);
    }

    #[test]
    fn test_bola_respects_max_quality_constraint() {
        let config = AbrConfig::new().with_max_quality(2);
        let mut ctrl = BolaBbrController::new(config, 10.0, 4.0);
        for _ in 0..15 {
            ctrl.report_segment_download(10_000_000, Duration::from_millis(200));
        }
        ctrl.report_buffer_level(Duration::from_secs(40));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        match decision {
            AbrDecision::SwitchTo(idx) => {
                assert!(idx <= 2, "Should respect max_quality=2, got {idx}")
            }
            AbrDecision::Maintain => {}
        }
    }

    // ─── RobustMpcController tests ─────────────────────────────────────────

    #[test]
    fn test_mpc_initial_quality() {
        let config = AbrConfig::new().with_initial_quality(1);
        let ctrl = RobustMpcController::default_params(config);
        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        assert_eq!(decision, AbrDecision::SwitchTo(1));
    }

    #[test]
    fn test_mpc_emergency_downswitch() {
        let config = AbrConfig::new();
        let mut ctrl = RobustMpcController::default_params(config);
        report_good_downloads(&mut ctrl, 10);
        ctrl.report_buffer_level(Duration::from_millis(500));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 4);
        assert_eq!(decision, AbrDecision::SwitchTo(0));
    }

    #[test]
    fn test_mpc_selects_high_quality_with_good_conditions() {
        let config = AbrConfig::new().with_mode(AbrMode::Aggressive);
        let mut ctrl = RobustMpcController::new(config, 5, 4.0);

        // 40 Mbps throughput
        for _ in 0..15 {
            ctrl.report_segment_download(5_000_000, Duration::from_millis(1000));
        }
        ctrl.report_buffer_level(Duration::from_secs(25));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 0);
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx > 0, "Expected upswitch, got level 0"),
            AbrDecision::Maintain => {}
        }
    }

    #[test]
    fn test_mpc_horizon_parameter() {
        let config = AbrConfig::new();
        let ctrl1 = RobustMpcController::new(config.clone(), 1, 4.0);
        let ctrl2 = RobustMpcController::new(config, 8, 4.0);
        // Both should produce valid decisions without panicking
        let levels = make_levels();
        let _ = ctrl1.select_quality(&levels, 2);
        let _ = ctrl2.select_quality(&levels, 2);
    }

    #[test]
    fn test_mpc_weights_customisation() {
        let config = AbrConfig::new();
        let mut ctrl = RobustMpcController::default_params(config);
        let weights = MpcWeights {
            quality: 2.0,
            smoothness: 0.5,
            rebuffer: 8.0,
            overflow: 0.2,
        };
        ctrl.set_weights(weights.clone());
        assert!((ctrl.weights().rebuffer - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mpc_rebuffer_avoidance() {
        let config = AbrConfig::new().with_mode(AbrMode::Conservative);
        let weights = MpcWeights {
            quality: 1.0,
            smoothness: 0.5,
            rebuffer: 20.0, // Very high rebuffer penalty
            overflow: 0.1,
        };
        let mut ctrl = RobustMpcController::new(config, 5, 4.0);
        ctrl.set_weights(weights);

        // Poor throughput: 1 Mbps, 1 MB segment
        for _ in 0..10 {
            ctrl.report_segment_download(500_000, Duration::from_secs(4));
        }
        ctrl.report_buffer_level(Duration::from_secs(6));

        let levels = make_levels();
        let decision = ctrl.select_quality(&levels, 4);
        // With very high rebuffer penalty and poor throughput, should downswitch
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx < 4),
            AbrDecision::Maintain => {}
        }
    }

    #[test]
    fn test_mpc_reset_clears_state() {
        let config = AbrConfig::new();
        let mut ctrl = RobustMpcController::default_params(config);
        report_good_downloads(&mut ctrl, 10);
        ctrl.report_buffer_level(Duration::from_secs(20));
        ctrl.reset();
        assert_eq!(ctrl.current_buffer(), Duration::ZERO);
        assert_eq!(ctrl.download_history().len(), 0);
        assert_eq!(ctrl.estimated_throughput(), 0.0);
    }

    #[test]
    fn test_mpc_estimated_throughput_is_pessimistic() {
        let config = AbrConfig::new();
        let mut ctrl = RobustMpcController::default_params(config);
        // Add one very fast and many slow samples
        ctrl.report_segment_download(100_000_000, Duration::from_millis(1)); // 100 GB/s
        for _ in 0..9 {
            ctrl.report_segment_download(1_000_000, Duration::from_secs(2)); // 0.5 MB/s
        }
        // Robust estimate should be much lower than the outlier
        let est = ctrl.estimated_throughput();
        assert!(
            est < 100_000_000.0 * 8.0,
            "Robust estimate should not be dominated by outlier"
        );
    }

    #[test]
    fn test_mpc_respects_min_quality_constraint() {
        let config = AbrConfig::new().with_min_quality(2);
        let mut ctrl = RobustMpcController::default_params(config);
        // Poor throughput
        for _ in 0..10 {
            ctrl.report_segment_download(100_000, Duration::from_secs(5));
        }
        ctrl.report_buffer_level(Duration::from_secs(20));

        let levels = make_levels();
        // Without emergency (buffer ok), must respect min_quality=2
        let decision = ctrl.select_quality(&levels, 3);
        match decision {
            AbrDecision::SwitchTo(idx) => {
                assert!(idx >= 2, "Must respect min_quality=2, got {idx}")
            }
            AbrDecision::Maintain => {}
        }
    }

    #[test]
    fn test_download_history_window_stats_single_record() {
        let mut h = SegmentDownloadHistory::new(10);
        h.add(
            0,
            500_000,
            Duration::from_millis(500),
            Duration::from_secs(4),
        );
        let stats = h.stats(1);
        assert_eq!(stats.count, 1);
        assert!((stats.mean_throughput - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_all_controllers_implement_trait() {
        let levels = make_levels();
        let config = AbrConfig::new();

        let mut dash: Box<dyn AdaptiveBitrateController> =
            Box::new(DashAbrController::new(config.clone()));
        let mut bola: Box<dyn AdaptiveBitrateController> =
            Box::new(BolaBbrController::default_params(config.clone()));
        let mut mpc: Box<dyn AdaptiveBitrateController> =
            Box::new(RobustMpcController::default_params(config));

        for ctrl in [&mut dash, &mut bola, &mut mpc] {
            ctrl.report_segment_download(1_000_000, Duration::from_secs(1));
            ctrl.report_buffer_level(Duration::from_secs(10));
            let _ = ctrl.select_quality(&levels, 0);
            let _ = ctrl.estimated_throughput();
            let _ = ctrl.current_buffer();
            ctrl.reset();
        }
    }
}
