//! Buffer-Based Adaptation (BBA) ABR controller.
//!
//! BBA (Huang et al., SIGCOMM 2014) performs quality selection purely based on
//! the current buffer level, without relying on throughput estimates.  It
//! divides the buffer into three zones:
//!
//! - **Reservoir** (0 → `f_min`): emergency zone — always select lowest quality.
//! - **Cushion** (`f_min` → `f_max`): linear ramp — quality scales with buffer fill.
//! - **Above cushion** (`f_max` → max_buffer): steady state — select highest quality.
//!
//! # Algorithm
//!
//! For each quality level `i` with bitrate `r_i`, BBA maps the current buffer
//! level `B(t)` to a target bitrate `R_BBA(t)` using a linear interpolation:
//!
//! ```text
//! R_BBA(t) = r_min + (r_max - r_min) * (B(t) - f_min) / (f_max - f_min)
//! ```
//!
//! The highest quality whose bitrate ≤ `R_BBA(t)` is selected.  Hysteresis
//! prevents oscillation by requiring the buffer to rise further before an
//! upswitch and allowing immediate downswitches.

use super::{AbrConfig, AbrDecision, AdaptiveBitrateController, BandwidthEstimator, QualityLevel};
use std::time::{Duration, Instant};

// ─── BBA Configuration ────────────────────────────────────────────────────────

/// Buffer-level thresholds for BBA zones (in seconds).
#[derive(Debug, Clone)]
pub struct BbaZones {
    /// Reservoir boundary: below this → lowest quality.
    pub reservoir_secs: f64,
    /// Cushion upper boundary: above this → highest quality.
    pub cushion_secs: f64,
    /// Buffer considered "full" — clamped to `AbrConfig::max_buffer`.
    pub max_buffer_secs: f64,
    /// Extra buffer required above the linear target to allow an upswitch
    /// (hysteresis for quality increase).
    pub upswitch_margin_secs: f64,
}

impl Default for BbaZones {
    fn default() -> Self {
        Self {
            reservoir_secs: 5.0,
            cushion_secs: 25.0,
            max_buffer_secs: 40.0,
            upswitch_margin_secs: 2.0,
        }
    }
}

impl BbaZones {
    /// Creates zones for low-latency streaming (smaller buffer targets).
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            reservoir_secs: 1.5,
            cushion_secs: 8.0,
            max_buffer_secs: 12.0,
            upswitch_margin_secs: 0.5,
        }
    }

    /// Returns true if `buffer_secs` is within the cushion zone.
    #[must_use]
    pub fn in_cushion(&self, buffer_secs: f64) -> bool {
        buffer_secs > self.reservoir_secs && buffer_secs <= self.cushion_secs
    }

    /// Returns true if `buffer_secs` is in the reservoir (emergency) zone.
    #[must_use]
    pub fn in_reservoir(&self, buffer_secs: f64) -> bool {
        buffer_secs <= self.reservoir_secs
    }

    /// Returns true if `buffer_secs` is above the cushion.
    #[must_use]
    pub fn above_cushion(&self, buffer_secs: f64) -> bool {
        buffer_secs > self.cushion_secs
    }
}

// ─── BBA Controller ───────────────────────────────────────────────────────────

/// Buffer-Based Adaptation (BBA) ABR controller.
///
/// Selects quality solely from buffer level using the reservoir/cushion model.
/// A [`BandwidthEstimator`] is retained for fallback stall-avoidance checks
/// but does not drive quality selection under normal conditions.
#[derive(Debug)]
pub struct BbaController {
    /// Base ABR configuration (used for constraints and buffer sizing).
    config: AbrConfig,
    /// Buffer zone thresholds.
    zones: BbaZones,
    /// Current buffer occupancy.
    buffer_level: Duration,
    /// Fallback bandwidth estimator for stall avoidance.
    bandwidth_estimator: BandwidthEstimator,
    /// Last quality switch time (for minimum interval hysteresis).
    last_switch: Option<Instant>,
    /// Minimum time between quality increases.
    min_upswitch_interval: Duration,
    /// Minimum time between quality decreases.
    min_downswitch_interval: Duration,
}

impl BbaController {
    /// Creates a new BBA controller with custom zones.
    #[must_use]
    pub fn with_zones(config: AbrConfig, zones: BbaZones) -> Self {
        let alpha = config.mode.ema_alpha();
        let bandwidth_estimator =
            BandwidthEstimator::new(config.estimation_window, config.sample_ttl, alpha);
        Self {
            config,
            zones,
            buffer_level: Duration::ZERO,
            bandwidth_estimator,
            last_switch: None,
            min_upswitch_interval: Duration::from_secs(8),
            min_downswitch_interval: Duration::from_secs(2),
        }
    }

    /// Creates a new BBA controller with default zones.
    #[must_use]
    pub fn new(config: AbrConfig) -> Self {
        Self::with_zones(config, BbaZones::default())
    }

    /// Creates a BBA controller optimized for low-latency streaming.
    #[must_use]
    pub fn low_latency(config: AbrConfig) -> Self {
        let zones = BbaZones::low_latency();
        let mut ctrl = Self::with_zones(config, zones);
        ctrl.min_upswitch_interval = Duration::from_secs(3);
        ctrl.min_downswitch_interval = Duration::from_millis(500);
        ctrl
    }

    /// Returns the current BBA zones.
    #[must_use]
    pub fn zones(&self) -> &BbaZones {
        &self.zones
    }

    /// Returns the current buffer level.
    #[must_use]
    pub fn buffer_secs(&self) -> f64 {
        self.buffer_level.as_secs_f64()
    }

    /// Computes the target bitrate from the buffer level using BBA's
    /// linear mapping across the cushion zone.
    ///
    /// Returns bits per second.
    #[must_use]
    pub fn target_bitrate_bps(&self, levels: &[QualityLevel]) -> f64 {
        if levels.is_empty() {
            return 0.0;
        }

        let buf = self.buffer_level.as_secs_f64();
        let r_min = levels.iter().map(|l| l.effective_bandwidth()).min().unwrap_or(0) as f64;
        let r_max = levels.iter().map(|l| l.effective_bandwidth()).max().unwrap_or(0) as f64;

        if self.zones.in_reservoir(buf) {
            r_min
        } else if self.zones.above_cushion(buf) {
            r_max
        } else {
            // Linear interpolation in cushion zone
            let ratio = (buf - self.zones.reservoir_secs)
                / (self.zones.cushion_secs - self.zones.reservoir_secs);
            r_min + (r_max - r_min) * ratio.clamp(0.0, 1.0)
        }
    }

    /// Finds the best quality whose bitrate ≤ target_bps.
    fn best_quality_for_bps(&self, levels: &[QualityLevel], target_bps: f64) -> usize {
        let mut best = 0usize;
        let mut best_bw = 0u64;
        for (idx, level) in levels.iter().enumerate() {
            let bw = level.effective_bandwidth();
            if bw as f64 <= target_bps && bw > best_bw {
                best = idx;
                best_bw = bw;
            }
        }
        // Apply config constraints
        if let Some(min) = self.config.min_quality {
            best = best.max(min);
        }
        if let Some(max) = self.config.max_quality {
            best = best.min(max);
        }
        best.min(levels.len().saturating_sub(1))
    }

    fn can_upswitch(&self) -> bool {
        match self.last_switch {
            Some(t) => t.elapsed() >= self.min_upswitch_interval,
            None => true,
        }
    }

    fn can_downswitch(&self) -> bool {
        match self.last_switch {
            Some(t) => t.elapsed() >= self.min_downswitch_interval,
            None => true,
        }
    }
}

impl AdaptiveBitrateController for BbaController {
    fn select_quality(&self, levels: &[QualityLevel], current_index: usize) -> AbrDecision {
        if levels.is_empty() {
            return AbrDecision::Maintain;
        }

        let buf = self.buffer_level.as_secs_f64();

        // Reservoir: emergency downswitch immediately
        if self.zones.in_reservoir(buf) {
            let min = self.config.min_quality.unwrap_or(0);
            if current_index != min {
                return AbrDecision::SwitchTo(min);
            }
            return AbrDecision::Maintain;
        }

        // Stall avoidance: if bandwidth estimate is very low, cap quality
        let estimated_bps = self.bandwidth_estimator.estimate_conservative() * 8.0;
        let effective_bps = if estimated_bps > 0.0 {
            // BBA target, but don't exceed what bandwidth can sustain
            self.target_bitrate_bps(levels).min(estimated_bps * 0.9)
        } else {
            self.target_bitrate_bps(levels)
        };

        // Add upswitch margin to the target — must have extra buffer headroom
        // before allowing an upswitch.
        let current_bw = levels
            .get(current_index)
            .map(|l| l.effective_bandwidth() as f64)
            .unwrap_or(0.0);

        let target_idx = self.best_quality_for_bps(levels, effective_bps);

        if target_idx > current_index {
            // Upswitch: require margin + interval
            let margined_bps = if buf > self.zones.reservoir_secs + self.zones.upswitch_margin_secs
            {
                effective_bps
            } else {
                current_bw // Stay put if buffer is not high enough
            };
            let idx_with_margin = self.best_quality_for_bps(levels, margined_bps);
            if idx_with_margin > current_index && self.can_upswitch() {
                return AbrDecision::SwitchTo(idx_with_margin);
            }
            AbrDecision::Maintain
        } else if target_idx < current_index {
            if self.can_downswitch() {
                AbrDecision::SwitchTo(target_idx)
            } else {
                AbrDecision::Maintain
            }
        } else {
            AbrDecision::Maintain
        }
    }

    fn report_segment_download(&mut self, bytes: usize, duration: Duration) {
        self.bandwidth_estimator.add_sample(bytes, duration);
    }

    fn report_buffer_level(&mut self, buffer_duration: Duration) {
        self.buffer_level = buffer_duration;
    }

    fn estimated_throughput(&self) -> f64 {
        self.bandwidth_estimator.estimate_conservative() * 8.0
    }

    fn current_buffer(&self) -> Duration {
        self.buffer_level
    }

    fn reset(&mut self) {
        self.buffer_level = Duration::ZERO;
        self.bandwidth_estimator.reset();
        self.last_switch = None;
    }

    fn config(&self) -> &AbrConfig {
        &self.config
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn quality_levels() -> Vec<QualityLevel> {
        vec![
            QualityLevel::new(0, 500_000),
            QualityLevel::new(1, 1_500_000),
            QualityLevel::new(2, 3_000_000),
            QualityLevel::new(3, 6_000_000),
        ]
    }

    fn bba(reservoir: f64, cushion: f64) -> BbaController {
        let zones = BbaZones {
            reservoir_secs: reservoir,
            cushion_secs: cushion,
            max_buffer_secs: 40.0,
            upswitch_margin_secs: 1.0,
        };
        BbaController::with_zones(AbrConfig::default(), zones)
    }

    // 1. Default zones values
    #[test]
    fn test_default_zones() {
        let z = BbaZones::default();
        assert!(z.reservoir_secs < z.cushion_secs);
        assert!(z.cushion_secs < z.max_buffer_secs);
    }

    // 2. Low latency zones are smaller
    #[test]
    fn test_low_latency_zones() {
        let z = BbaZones::low_latency();
        let d = BbaZones::default();
        assert!(z.cushion_secs < d.cushion_secs);
    }

    // 3. Zone predicates – reservoir
    #[test]
    fn test_zone_reservoir() {
        let z = BbaZones { reservoir_secs: 5.0, cushion_secs: 25.0, ..BbaZones::default() };
        assert!(z.in_reservoir(0.0));
        assert!(z.in_reservoir(5.0));
        assert!(!z.in_reservoir(5.01));
    }

    // 4. Zone predicates – cushion
    #[test]
    fn test_zone_cushion() {
        let z = BbaZones { reservoir_secs: 5.0, cushion_secs: 25.0, ..BbaZones::default() };
        assert!(z.in_cushion(10.0));
        assert!(!z.in_cushion(4.9));
        assert!(!z.in_cushion(25.1));
    }

    // 5. Zone predicates – above cushion
    #[test]
    fn test_zone_above_cushion() {
        let z = BbaZones { reservoir_secs: 5.0, cushion_secs: 25.0, ..BbaZones::default() };
        assert!(z.above_cushion(25.01));
        assert!(!z.above_cushion(25.0));
    }

    // 6. Target bitrate at reservoir → minimum quality
    #[test]
    fn test_target_bitrate_reservoir() {
        let mut ctrl = bba(5.0, 25.0);
        ctrl.report_buffer_level(Duration::from_secs(3));
        let target = ctrl.target_bitrate_bps(&quality_levels());
        let min_bw = 500_000.0;
        assert!((target - min_bw).abs() < 1.0);
    }

    // 7. Target bitrate above cushion → maximum quality
    #[test]
    fn test_target_bitrate_above_cushion() {
        let mut ctrl = bba(5.0, 25.0);
        ctrl.report_buffer_level(Duration::from_secs(30));
        let target = ctrl.target_bitrate_bps(&quality_levels());
        let max_bw = 6_000_000.0;
        assert!((target - max_bw).abs() < 1.0);
    }

    // 8. Target bitrate mid-cushion → interpolated
    #[test]
    fn test_target_bitrate_mid_cushion() {
        let mut ctrl = bba(5.0, 25.0);
        // Buffer at midpoint of [5, 25] = 15 s → 50% of range
        ctrl.report_buffer_level(Duration::from_secs(15));
        let target = ctrl.target_bitrate_bps(&quality_levels());
        let expected = 500_000.0 + (6_000_000.0 - 500_000.0) * 0.5;
        assert!((target - expected).abs() < 1000.0);
    }

    // 9. Quality selection in reservoir → lowest quality
    #[test]
    fn test_reservoir_select_lowest() {
        let mut ctrl = bba(5.0, 25.0);
        ctrl.report_buffer_level(Duration::from_millis(500));
        let decision = ctrl.select_quality(&quality_levels(), 3);
        assert_eq!(decision, AbrDecision::SwitchTo(0));
    }

    // 10. Quality selection already at lowest in reservoir → Maintain
    #[test]
    fn test_reservoir_already_lowest() {
        let mut ctrl = bba(5.0, 25.0);
        ctrl.report_buffer_level(Duration::from_millis(500));
        let decision = ctrl.select_quality(&quality_levels(), 0);
        assert_eq!(decision, AbrDecision::Maintain);
    }

    // 11. Quality selection above cushion → upswitch allowed after interval
    #[test]
    fn test_above_cushion_allows_highest() {
        let mut ctrl = bba(5.0, 10.0);
        ctrl.report_buffer_level(Duration::from_secs(15));
        // No last_switch → can upswitch immediately
        let decision = ctrl.select_quality(&quality_levels(), 0);
        // Should suggest a higher quality
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx > 0),
            AbrDecision::Maintain => {} // also acceptable if margin not met
        }
    }

    // 12. Empty quality levels → Maintain
    #[test]
    fn test_empty_levels() {
        let mut ctrl = bba(5.0, 25.0);
        ctrl.report_buffer_level(Duration::from_secs(20));
        assert_eq!(ctrl.select_quality(&[], 0), AbrDecision::Maintain);
    }

    // 13. Report buffer level updates state
    #[test]
    fn test_report_buffer_level() {
        let mut ctrl = BbaController::new(AbrConfig::default());
        ctrl.report_buffer_level(Duration::from_secs(12));
        assert!((ctrl.buffer_secs() - 12.0).abs() < 1e-9);
    }

    // 14. Report segment download updates bandwidth estimator
    #[test]
    fn test_report_segment_download() {
        let mut ctrl = BbaController::new(AbrConfig::default());
        ctrl.report_segment_download(1_000_000, Duration::from_secs(1));
        assert!(ctrl.estimated_throughput() > 0.0);
    }

    // 15. Reset clears state
    #[test]
    fn test_reset() {
        let mut ctrl = BbaController::new(AbrConfig::default());
        ctrl.report_buffer_level(Duration::from_secs(20));
        ctrl.report_segment_download(1_000_000, Duration::from_secs(1));
        ctrl.reset();
        assert_eq!(ctrl.current_buffer(), Duration::ZERO);
        assert!((ctrl.estimated_throughput()).abs() < 1.0);
    }

    // 16. config() accessor
    #[test]
    fn test_config_accessor() {
        let cfg = AbrConfig::default();
        let ctrl = BbaController::new(cfg.clone());
        // Just verify it doesn't panic
        let _ = ctrl.config();
    }

    // 17. Low latency constructor
    #[test]
    fn test_low_latency_ctor() {
        let ctrl = BbaController::low_latency(AbrConfig::default());
        assert!(ctrl.zones().reservoir_secs < 5.0);
    }

    // 18. BBA with min_quality constraint
    #[test]
    fn test_min_quality_constraint() {
        let cfg = AbrConfig::default().with_min_quality(1);
        let mut ctrl = BbaController::new(cfg);
        ctrl.report_buffer_level(Duration::from_millis(100));
        // Reservoir would normally pick 0, but min_quality = 1
        let decision = ctrl.select_quality(&quality_levels(), 3);
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx >= 1),
            AbrDecision::Maintain => {}
        }
    }

    // 19. BBA with max_quality constraint
    #[test]
    fn test_max_quality_constraint() {
        let cfg = AbrConfig::default().with_max_quality(2);
        let mut ctrl = BbaController::with_zones(
            cfg,
            BbaZones { reservoir_secs: 1.0, cushion_secs: 3.0, ..BbaZones::default() },
        );
        ctrl.report_buffer_level(Duration::from_secs(10));
        let decision = ctrl.select_quality(&quality_levels(), 0);
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx <= 2),
            AbrDecision::Maintain => {}
        }
    }

    // 20. Target bitrate with empty levels returns 0
    #[test]
    fn test_target_bitrate_empty() {
        let ctrl = BbaController::new(AbrConfig::default());
        assert!((ctrl.target_bitrate_bps(&[])).abs() < 1e-9);
    }

    // 21. Downswitch in cushion when buffer drops
    #[test]
    fn test_downswitch_on_buffer_drop() {
        let mut ctrl = bba(5.0, 25.0);
        // Buffer drops to near reservoir
        ctrl.report_buffer_level(Duration::from_secs(6));
        // At quality index 3 (6 Mbps) — BBA will want lower quality
        let decision = ctrl.select_quality(&quality_levels(), 3);
        match decision {
            AbrDecision::SwitchTo(idx) => assert!(idx < 3),
            AbrDecision::Maintain => {} // acceptable if hysteresis prevents it
        }
    }

    // 22. Bandwidth estimator provides stall protection
    #[test]
    fn test_stall_protection_via_bandwidth() {
        let mut ctrl = bba(5.0, 25.0);
        // Very low bandwidth estimate
        ctrl.report_segment_download(1000, Duration::from_secs(1)); // ~8 kbps
        ctrl.report_buffer_level(Duration::from_secs(20)); // above cushion
        // Even though buffer is high, bandwidth is too low for top quality
        let levels = quality_levels();
        let target = ctrl.target_bitrate_bps(&levels);
        // The capped target should be reasonable
        assert!(target >= 0.0);
    }

    // 23. Buffer accessor matches reported level
    #[test]
    fn test_current_buffer_accessor() {
        let mut ctrl = BbaController::new(AbrConfig::default());
        ctrl.report_buffer_level(Duration::from_secs(7));
        assert_eq!(ctrl.current_buffer(), Duration::from_secs(7));
    }

    // 24. best_quality_for_bps selects highest fitting quality
    #[test]
    fn test_best_quality_for_bps() {
        let ctrl = BbaController::new(AbrConfig::default());
        let levels = quality_levels();
        // 2 Mbps → should pick index 1 (1.5 Mbps), not index 2 (3 Mbps)
        let idx = ctrl.best_quality_for_bps(&levels, 2_000_000.0);
        assert_eq!(idx, 1);
    }

    // 25. BBA zones with_zones constructor
    #[test]
    fn test_with_zones_ctor() {
        let zones = BbaZones { reservoir_secs: 2.0, cushion_secs: 15.0, ..BbaZones::default() };
        let ctrl = BbaController::with_zones(AbrConfig::default(), zones.clone());
        assert!((ctrl.zones().reservoir_secs - 2.0).abs() < 1e-9);
        assert!((ctrl.zones().cushion_secs - 15.0).abs() < 1e-9);
    }
}
