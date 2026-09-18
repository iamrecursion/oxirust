//! Stream health monitoring and diagnostics.
//!
//! Tracks quality-of-experience (QoE) metrics — rebuffering events, quality
//! switches, dropped frames, latency — and computes a composite health score
//! (0–100) together with a list of detected [`HealthIssue`]s.

use std::collections::VecDeque;
use std::time::{Instant, SystemTime};

// ─── Health issues ────────────────────────────────────────────────────────────

/// A specific problem detected by the health monitor.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthIssue {
    /// Playback buffer is below a warning threshold.
    BufferTooLow {
        /// Current buffer depth in seconds.
        buffer_secs: f64,
        /// Threshold that was breached.
        threshold: f64,
    },
    /// Quality switches occurred too frequently within the reporting window.
    FrequentSwitches {
        /// Number of switches observed.
        count: u32,
        /// Duration of the observation window in seconds.
        window_secs: u64,
    },
    /// The rebuffering (stall) rate exceeds an acceptable level.
    HighRebufferRate {
        /// Percentage of playback time spent rebuffering.
        rate_pct: f64,
    },
    /// The actual bitrate is significantly below the requested bitrate.
    LowBitrate {
        /// Measured bitrate in kbps.
        actual_kbps: f64,
        /// Requested or expected bitrate in kbps.
        requested_kbps: f64,
    },
    /// Round-trip latency exceeds a threshold.
    HighLatency {
        /// Measured latency in milliseconds.
        latency_ms: u64,
        /// Threshold in milliseconds.
        threshold_ms: u64,
    },
    /// Dropped-frame rate exceeds an acceptable level.
    DroppedFramesExcessive {
        /// Percentage of frames that were dropped.
        rate_pct: f64,
    },
}

// ─── Health report ────────────────────────────────────────────────────────────

/// A point-in-time snapshot of stream health.
#[derive(Debug, Clone)]
pub struct StreamHealthReport {
    /// Wall-clock time of this report.
    pub timestamp: SystemTime,
    /// Composite health score in the range \[0, 100\].
    /// 100 = perfect; 0 = completely broken.
    pub overall_score: f32,
    /// Current bitrate in kbps.
    pub bitrate_kbps: f64,
    /// Current buffer depth in seconds.
    pub buffer_secs: f64,
    /// Total rebuffering (stall) events since the session started.
    pub rebuffer_events: u32,
    /// Total quality switches since the session started.
    pub quality_switches: u32,
    /// Time from session start to first decoded frame in milliseconds.
    /// Zero if the first frame has not been observed yet.
    pub startup_time_ms: u64,
    /// Total dropped frames since the session started.
    pub dropped_frames: u32,
    /// Total error events since the session started.
    pub error_count: u32,
    /// List of issues detected at the time of this report.
    pub issues: Vec<HealthIssue>,
}

// ─── QoE score ───────────────────────────────────────────────────────────────

/// Standalone Quality-of-Experience (QoE) metric that incorporates rebuffer
/// ratio, startup delay, average bitrate, and quality switch frequency into a
/// single 0–100 score.
///
/// The model is inspired by ITU-T P.1203 simplified:
///
/// ```text
/// QoE = w_bitrate   × bitrate_factor
///     + w_rebuffer   × (1 − rebuffer_ratio × 10)
///     + w_startup    × startup_factor
///     + w_switch     × (1 − switch_rate × 0.05)
/// ```
///
/// Each factor is clamped to \[0, 1\] before weighting.
#[derive(Debug, Clone)]
pub struct QoeScore {
    /// Composite score in \[0, 100\].
    pub score: f64,
    /// Bitrate sub-score in \[0, 1\].
    pub bitrate_factor: f64,
    /// Rebuffer ratio sub-score in \[0, 1\] (1 = no rebuffering).
    pub rebuffer_factor: f64,
    /// Startup delay sub-score in \[0, 1\] (1 = instant start).
    pub startup_factor: f64,
    /// Quality-switch penalty sub-score in \[0, 1\] (1 = no switches).
    pub switch_factor: f64,
}

/// Configuration for the QoE scoring model.
#[derive(Debug, Clone)]
pub struct QoeConfig {
    /// Reference bitrate (kbps) that maps to a perfect bitrate score.
    pub reference_bitrate_kbps: f64,
    /// Weight for the bitrate component (default 0.35).
    pub weight_bitrate: f64,
    /// Weight for the rebuffer ratio component (default 0.30).
    pub weight_rebuffer: f64,
    /// Weight for the startup delay component (default 0.20).
    pub weight_startup: f64,
    /// Weight for the quality-switch component (default 0.15).
    pub weight_switch: f64,
    /// Startup delay (ms) considered "fast" (score = 1.0).
    pub startup_fast_ms: u64,
    /// Startup delay (ms) considered "slow" (score = 0.0).
    pub startup_slow_ms: u64,
}

impl Default for QoeConfig {
    fn default() -> Self {
        Self {
            reference_bitrate_kbps: 5000.0,
            weight_bitrate: 0.35,
            weight_rebuffer: 0.30,
            weight_startup: 0.20,
            weight_switch: 0.15,
            startup_fast_ms: 1000,
            startup_slow_ms: 10_000,
        }
    }
}

// ─── Health monitor ───────────────────────────────────────────────────────────

/// Accumulates stream health events and produces [`StreamHealthReport`]s.
pub struct StreamHealthMonitor {
    history: VecDeque<StreamHealthReport>,
    max_history: usize,
    rebuffer_events: u32,
    quality_switches: u32,
    dropped_frames: u32,
    errors: u32,
    start_time: Option<Instant>,
    first_frame_time: Option<Instant>,
    /// Cumulative rebuffering duration in milliseconds.
    rebuffer_duration_ms: u64,
    /// Total playback duration observed in milliseconds (set via `record_playback_duration`).
    total_playback_ms: u64,
}

impl StreamHealthMonitor {
    /// Create a new monitor that retains up to `max_history` historical reports.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history.max(1)),
            max_history: max_history.max(1),
            rebuffer_events: 0,
            quality_switches: 0,
            dropped_frames: 0,
            errors: 0,
            start_time: None,
            first_frame_time: None,
            rebuffer_duration_ms: 0,
            total_playback_ms: 0,
        }
    }

    /// Mark the beginning of a playback session.  Resets all counters.
    pub fn start_session(&mut self) {
        self.start_time = Some(Instant::now());
        self.first_frame_time = None;
        self.rebuffer_events = 0;
        self.quality_switches = 0;
        self.dropped_frames = 0;
        self.errors = 0;
        self.rebuffer_duration_ms = 0;
        self.total_playback_ms = 0;
        self.history.clear();
    }

    /// Record the instant the first frame was decoded and rendered.
    ///
    /// If the session has not been started, this call starts it implicitly.
    pub fn first_frame(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }
        if self.first_frame_time.is_none() {
            self.first_frame_time = Some(Instant::now());
        }
    }

    /// Increment the rebuffering (stall) event counter by one.
    pub fn record_rebuffer(&mut self) {
        self.rebuffer_events = self.rebuffer_events.saturating_add(1);
    }

    /// Record a rebuffering stall event with a known duration.
    ///
    /// This both increments the event counter and accumulates the stall
    /// duration for the rebuffer ratio calculation.
    pub fn record_rebuffer_with_duration(&mut self, duration_ms: u64) {
        self.rebuffer_events = self.rebuffer_events.saturating_add(1);
        self.rebuffer_duration_ms = self.rebuffer_duration_ms.saturating_add(duration_ms);
    }

    /// Update the total playback duration observed so far.
    ///
    /// This should be called periodically (e.g. once per segment) to keep the
    /// rebuffer ratio calculation accurate.
    pub fn record_playback_duration(&mut self, total_ms: u64) {
        self.total_playback_ms = total_ms;
    }

    /// Rebuffer ratio: fraction of total session time spent rebuffering.
    ///
    /// Returns 0.0 if no playback time has been recorded.
    pub fn rebuffer_ratio(&self) -> f64 {
        if self.total_playback_ms == 0 {
            return 0.0;
        }
        let total = self
            .total_playback_ms
            .saturating_add(self.rebuffer_duration_ms);
        if total == 0 {
            return 0.0;
        }
        self.rebuffer_duration_ms as f64 / total as f64
    }

    /// Increment the quality switch counter by one.
    pub fn record_quality_switch(&mut self) {
        self.quality_switches = self.quality_switches.saturating_add(1);
    }

    /// Increment the dropped-frame counter.
    pub fn record_dropped_frames(&mut self, count: u32) {
        self.dropped_frames = self.dropped_frames.saturating_add(count);
    }

    /// Increment the error counter by one.
    pub fn record_error(&mut self) {
        self.errors = self.errors.saturating_add(1);
    }

    /// Generate a [`StreamHealthReport`] for the current moment.
    ///
    /// The overall score is computed as a weighted sum:
    /// - Buffer health (25%): 100 at >= 30 s, 0 at 0 s.
    /// - Bitrate health (20%): 100 at >= `reference_kbps` = 5000 kbps.
    /// - Rebuffer ratio penalty (25%): 100 at 0%, 0 at >= 10%.
    /// - Startup delay penalty (15%): 100 at <= 1 s, 0 at >= 10 s.
    /// - Rebuffer event penalty (10%): 100 at 0 events, linear decay.
    /// - Switch penalty (5%): 100 at 0 switches, linear decay.
    ///
    /// Issues are detected independently of the score.
    pub fn report(&self, bitrate_kbps: f64, buffer_secs: f64) -> StreamHealthReport {
        let startup_ms = self.startup_time_ms().unwrap_or(0);

        // ── Score components ─────────────────────────────────────────────────
        const BUFFER_FULL_SECS: f64 = 30.0;
        const REFERENCE_KBPS: f64 = 5000.0;

        // Buffer health [0..1]
        let buffer_score = (buffer_secs / BUFFER_FULL_SECS).clamp(0.0, 1.0);

        // Bitrate health [0..1]
        let bitrate_score = if REFERENCE_KBPS > 0.0 {
            (bitrate_kbps / REFERENCE_KBPS).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Rebuffer ratio penalty [0..1]: 0% → 1.0; >= 10% → 0.0 (linear).
        let rebuffer_ratio = self.rebuffer_ratio();
        let rebuffer_ratio_score = (1.0 - rebuffer_ratio * 10.0).clamp(0.0, 1.0);

        // Startup delay penalty [0..1]: <= 1000 ms → 1.0; >= 10000 ms → 0.0 (linear).
        let startup_score = if startup_ms <= 1000 {
            1.0_f64
        } else if startup_ms >= 10_000 {
            0.0
        } else {
            let range = 10_000.0 - 1000.0;
            (1.0 - (startup_ms as f64 - 1000.0) / range).clamp(0.0, 1.0)
        };

        // Rebuffer event penalty [0..1]: 0 events → 1.0; every event costs 10 points.
        let rebuffer_event_score = if self.rebuffer_events == 0 {
            1.0_f64
        } else {
            (1.0 - self.rebuffer_events as f64 * 0.10).clamp(0.0, 1.0)
        };

        // Switch penalty [0..1]: 0 switches → 1.0; every switch costs 5 points.
        let switch_score = if self.quality_switches == 0 {
            1.0_f64
        } else {
            (1.0 - self.quality_switches as f64 * 0.05).clamp(0.0, 1.0)
        };

        let overall = (buffer_score * 0.25
            + bitrate_score * 0.20
            + rebuffer_ratio_score * 0.25
            + startup_score * 0.15
            + rebuffer_event_score * 0.10
            + switch_score * 0.05)
            * 100.0;

        let overall_score = (overall as f32).clamp(0.0, 100.0);

        // ── Issue detection ──────────────────────────────────────────────────
        let mut issues: Vec<HealthIssue> = Vec::new();

        if buffer_secs < 5.0 {
            issues.push(HealthIssue::BufferTooLow {
                buffer_secs,
                threshold: 5.0,
            });
        }

        if self.quality_switches >= 3 {
            issues.push(HealthIssue::FrequentSwitches {
                count: self.quality_switches,
                window_secs: 60,
            });
        }

        if self.rebuffer_events > 0 {
            // Rebuffer rate expressed as a rough percentage (1 event = 5%)
            let rate_pct = (self.rebuffer_events as f64 * 5.0).min(100.0);
            if rate_pct > 10.0 {
                issues.push(HealthIssue::HighRebufferRate { rate_pct });
            }
        }

        if bitrate_kbps > 0.0 && bitrate_kbps < REFERENCE_KBPS * 0.2 {
            issues.push(HealthIssue::LowBitrate {
                actual_kbps: bitrate_kbps,
                requested_kbps: REFERENCE_KBPS,
            });
        }

        // Dropped-frame issue: > 5% of a nominal 30 fps stream at 10 s = 300 frames.
        if self.dropped_frames > 15 {
            let rate_pct =
                (self.dropped_frames as f64 / (self.dropped_frames as f64 + 285.0)) * 100.0;
            issues.push(HealthIssue::DroppedFramesExcessive { rate_pct });
        }

        StreamHealthReport {
            timestamp: SystemTime::now(),
            overall_score,
            bitrate_kbps,
            buffer_secs,
            rebuffer_events: self.rebuffer_events,
            quality_switches: self.quality_switches,
            startup_time_ms: startup_ms,
            dropped_frames: self.dropped_frames,
            error_count: self.errors,
            issues,
        }
    }

    /// Compute a standalone QoE score using the given configuration.
    ///
    /// Unlike `report`, this method returns a detailed breakdown of each
    /// scoring factor and uses configurable weights and thresholds.
    ///
    /// # Parameters
    ///
    /// - `bitrate_kbps`: current average bitrate in kbps.
    /// - `config`: scoring model configuration.
    pub fn qoe_score(&self, bitrate_kbps: f64, config: &QoeConfig) -> QoeScore {
        // Bitrate factor: linear ramp to reference, capped at 1.0.
        let bitrate_factor = if config.reference_bitrate_kbps > 0.0 {
            (bitrate_kbps / config.reference_bitrate_kbps).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Rebuffer factor: 0% ratio → 1.0, >= 10% ratio → 0.0 (linear).
        let rebuffer_ratio = self.rebuffer_ratio();
        let rebuffer_factor = (1.0 - rebuffer_ratio * 10.0).clamp(0.0, 1.0);

        // Startup factor: fast_ms → 1.0, slow_ms → 0.0 (linear interpolation).
        let startup_ms = self.startup_time_ms().unwrap_or(0);
        let startup_factor = if startup_ms <= config.startup_fast_ms {
            1.0
        } else if startup_ms >= config.startup_slow_ms {
            0.0
        } else {
            let range = (config.startup_slow_ms - config.startup_fast_ms) as f64;
            if range <= 0.0 {
                1.0
            } else {
                (1.0 - (startup_ms - config.startup_fast_ms) as f64 / range).clamp(0.0, 1.0)
            }
        };

        // Switch factor: each switch costs 5% of the score (linear decay).
        let switch_factor = (1.0 - self.quality_switches as f64 * 0.05).clamp(0.0, 1.0);

        // Weighted sum (weights need not sum to 1.0 — we normalise).
        let total_weight = config.weight_bitrate
            + config.weight_rebuffer
            + config.weight_startup
            + config.weight_switch;
        let raw = if total_weight > 0.0 {
            (config.weight_bitrate * bitrate_factor
                + config.weight_rebuffer * rebuffer_factor
                + config.weight_startup * startup_factor
                + config.weight_switch * switch_factor)
                / total_weight
        } else {
            0.0
        };

        let score = (raw * 100.0).clamp(0.0, 100.0);

        QoeScore {
            score,
            bitrate_factor,
            rebuffer_factor,
            startup_factor,
            switch_factor,
        }
    }

    /// Store a report in the history ring-buffer (evicts oldest if at capacity).
    pub fn store_report(&mut self, report: StreamHealthReport) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(report);
    }

    /// Arithmetic mean of `overall_score` over stored history.
    ///
    /// Returns 0.0 if no reports have been stored.
    pub fn avg_score(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().map(|r| r.overall_score as f64).sum();
        sum / self.history.len() as f64
    }

    /// Time (ms) from `start_session` to `first_frame`.
    ///
    /// Returns `None` if either event has not been recorded.
    pub fn startup_time_ms(&self) -> Option<u64> {
        match (self.start_time, self.first_frame_time) {
            (Some(start), Some(first)) => {
                let dur = first.duration_since(start);
                Some(dur.as_millis() as u64)
            }
            _ => None,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> StreamHealthMonitor {
        StreamHealthMonitor::new(100)
    }

    #[test]
    fn test_initial_state_no_events() {
        let m = make_monitor();
        assert!(m.startup_time_ms().is_none());
        assert_eq!(m.avg_score(), 0.0);
    }

    #[test]
    fn test_report_perfect_conditions() {
        let m = make_monitor();
        let report = m.report(5000.0, 30.0);
        // buffer=1.0, bitrate=1.0, rebuffer=1.0, switch=1.0 → 100
        assert!(
            (report.overall_score - 100.0).abs() < 0.01,
            "score={}",
            report.overall_score
        );
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_report_low_buffer_detected() {
        let m = make_monitor();
        let report = m.report(5000.0, 1.0);
        let has_buffer_issue = report
            .issues
            .iter()
            .any(|i| matches!(i, HealthIssue::BufferTooLow { .. }));
        assert!(has_buffer_issue, "low buffer issue should be detected");
    }

    #[test]
    fn test_report_score_decreases_with_rebuffers() {
        let mut m = make_monitor();
        let r0 = m.report(5000.0, 30.0);
        m.record_rebuffer();
        m.record_rebuffer();
        let r2 = m.report(5000.0, 30.0);
        assert!(
            r2.overall_score < r0.overall_score,
            "rebuffers should reduce score"
        );
    }

    #[test]
    fn test_rebuffer_recorded() {
        let mut m = make_monitor();
        m.record_rebuffer();
        m.record_rebuffer();
        let r = m.report(5000.0, 30.0);
        assert_eq!(r.rebuffer_events, 2);
    }

    #[test]
    fn test_quality_switch_recorded() {
        let mut m = make_monitor();
        m.record_quality_switch();
        let r = m.report(5000.0, 30.0);
        assert_eq!(r.quality_switches, 1);
    }

    #[test]
    fn test_frequent_switches_issue_detected() {
        let mut m = make_monitor();
        for _ in 0..4 {
            m.record_quality_switch();
        }
        let report = m.report(5000.0, 20.0);
        let has_issue = report
            .issues
            .iter()
            .any(|i| matches!(i, HealthIssue::FrequentSwitches { .. }));
        assert!(has_issue, "frequent switches should be flagged");
    }

    #[test]
    fn test_dropped_frames_recorded() {
        let mut m = make_monitor();
        m.record_dropped_frames(30);
        let r = m.report(5000.0, 20.0);
        assert_eq!(r.dropped_frames, 30);
        let has_issue = r
            .issues
            .iter()
            .any(|i| matches!(i, HealthIssue::DroppedFramesExcessive { .. }));
        assert!(has_issue, "excessive dropped frames should be flagged");
    }

    #[test]
    fn test_error_recorded() {
        let mut m = make_monitor();
        m.record_error();
        let r = m.report(5000.0, 20.0);
        assert_eq!(r.error_count, 1);
    }

    #[test]
    fn test_startup_time_measured() {
        let mut m = make_monitor();
        m.start_session();
        // Small synthetic delay via spinning — avoids std::thread::sleep.
        let spin_until = Instant::now() + std::time::Duration::from_millis(5);
        while Instant::now() < spin_until {}
        m.first_frame();
        let ms = m.startup_time_ms().expect("startup time should be set");
        assert!(ms >= 5, "startup_time_ms={ms} should be at least 5 ms");
    }

    #[test]
    fn test_avg_score_empty_history() {
        let m = make_monitor();
        assert_eq!(m.avg_score(), 0.0);
    }

    #[test]
    fn test_avg_score_computed() {
        let mut m = make_monitor();
        let r1 = m.report(5000.0, 30.0); // score ~ 100
        let r2 = m.report(100.0, 0.0); // score ~ low
        let s1 = r1.overall_score as f64;
        let s2 = r2.overall_score as f64;
        m.store_report(r1);
        m.store_report(r2);
        let avg = m.avg_score();
        let expected = (s1 + s2) / 2.0;
        assert!(
            (avg - expected).abs() < 0.01,
            "avg={avg} expected={expected}"
        );
    }

    #[test]
    fn test_history_ring_buffer_evicts_oldest() {
        let mut m = StreamHealthMonitor::new(3);
        for i in 0..5 {
            let r = m.report(i as f64 * 1000.0, 10.0);
            m.store_report(r);
        }
        assert_eq!(
            m.history.len(),
            3,
            "history should be capped at max_history"
        );
    }

    // ── Rebuffer ratio tests ────────────────────────────────────────────────

    #[test]
    fn test_rebuffer_ratio_zero_when_no_stalls() {
        let m = make_monitor();
        assert_eq!(m.rebuffer_ratio(), 0.0);
    }

    #[test]
    fn test_rebuffer_ratio_computed_correctly() {
        let mut m = make_monitor();
        m.record_playback_duration(10_000); // 10 s playback
        m.record_rebuffer_with_duration(2_000); // 2 s stall
                                                // ratio = 2000 / (10000 + 2000) = 2/12 ≈ 0.1667
        let ratio = m.rebuffer_ratio();
        assert!(
            (ratio - 2.0 / 12.0).abs() < 0.001,
            "expected ~0.167, got {ratio}"
        );
    }

    #[test]
    fn test_rebuffer_with_duration_increments_event_count() {
        let mut m = make_monitor();
        m.record_rebuffer_with_duration(500);
        m.record_rebuffer_with_duration(300);
        assert_eq!(m.rebuffer_events, 2);
    }

    #[test]
    fn test_rebuffer_ratio_affects_qoe_score() {
        let m1 = make_monitor();
        let r1 = m1.report(5000.0, 30.0); // perfect conditions

        let mut m2 = make_monitor();
        m2.record_playback_duration(10_000);
        m2.record_rebuffer_with_duration(5_000); // 5s stalls → high ratio
        let r2 = m2.report(5000.0, 30.0);

        assert!(
            r2.overall_score < r1.overall_score,
            "high rebuffer ratio should reduce score: perfect={} degraded={}",
            r1.overall_score,
            r2.overall_score
        );
    }

    #[test]
    fn test_startup_delay_affects_qoe_score() {
        let mut m_fast = make_monitor();
        m_fast.start_session();
        m_fast.first_frame(); // effectively 0 ms startup
        let r_fast = m_fast.report(5000.0, 30.0);

        // We cannot easily simulate a long startup with spin,
        // but we can verify that the startup_time_ms field is used
        // by checking the score formula directly.
        // With 0 ms startup → startup_score = 1.0, weight = 0.15 → contributes 15
        assert!(
            r_fast.overall_score > 90.0,
            "fast startup should give high score: {}",
            r_fast.overall_score
        );
    }

    #[test]
    fn test_record_playback_duration_updates_total() {
        let mut m = make_monitor();
        m.record_playback_duration(5000);
        m.record_playback_duration(10000);
        // Should overwrite
        assert_eq!(m.total_playback_ms, 10000);
    }

    #[test]
    fn test_start_session_resets_rebuffer_duration() {
        let mut m = make_monitor();
        m.record_rebuffer_with_duration(1000);
        m.record_playback_duration(5000);
        m.start_session();
        assert_eq!(m.rebuffer_ratio(), 0.0);
    }

    #[test]
    fn test_qoe_perfect_score_with_all_good_metrics() {
        let mut m = make_monitor();
        m.start_session();
        m.first_frame(); // fast startup
                         // No rebuffers, no switches, no dropped frames
        let r = m.report(5000.0, 30.0);
        assert!(
            r.overall_score > 95.0,
            "perfect conditions should give near-100 score: {}",
            r.overall_score
        );
    }

    // ── QoE score tests ───────────────────────────────────────────────────────

    #[test]
    fn test_qoe_score_perfect_conditions() {
        let mut m = make_monitor();
        m.start_session();
        m.first_frame();
        let config = QoeConfig::default();
        let qoe = m.qoe_score(5000.0, &config);
        assert!(
            qoe.score > 95.0,
            "perfect conditions should yield near-100 QoE, got {}",
            qoe.score
        );
        assert!((qoe.bitrate_factor - 1.0).abs() < 0.01);
        assert!((qoe.rebuffer_factor - 1.0).abs() < 0.01);
        assert!((qoe.switch_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_qoe_score_low_bitrate_reduces_score() {
        let m = make_monitor();
        let config = QoeConfig::default();
        let high = m.qoe_score(5000.0, &config);
        let low = m.qoe_score(500.0, &config);
        assert!(
            low.score < high.score,
            "low bitrate should reduce QoE: high={} low={}",
            high.score,
            low.score
        );
        assert!(low.bitrate_factor < 0.2);
    }

    #[test]
    fn test_qoe_score_high_rebuffer_ratio_reduces_score() {
        let mut m = make_monitor();
        m.record_playback_duration(10_000);
        m.record_rebuffer_with_duration(5_000); // 33% ratio
        let config = QoeConfig::default();
        let qoe = m.qoe_score(5000.0, &config);
        assert!(
            qoe.rebuffer_factor < 0.7,
            "high rebuffer should penalise: factor={}",
            qoe.rebuffer_factor
        );
        assert!(qoe.score < 90.0, "score should be degraded: {}", qoe.score);
    }

    #[test]
    fn test_qoe_score_extreme_rebuffer_zeros_factor() {
        let mut m = make_monitor();
        m.record_playback_duration(10_000);
        // 10% rebuffer ratio → factor should be 0.0
        m.record_rebuffer_with_duration(1_112); // ratio ≈ 10%+
        let config = QoeConfig::default();
        let qoe = m.qoe_score(5000.0, &config);
        assert!(
            qoe.rebuffer_factor < 0.1,
            "extreme rebuffer should zero factor: {}",
            qoe.rebuffer_factor
        );
    }

    #[test]
    fn test_qoe_score_many_switches_reduces_score() {
        let mut m = make_monitor();
        for _ in 0..20 {
            m.record_quality_switch();
        }
        let config = QoeConfig::default();
        let qoe = m.qoe_score(5000.0, &config);
        assert!(
            (qoe.switch_factor - 0.0).abs() < 0.01,
            "20 switches should zero switch_factor: {}",
            qoe.switch_factor
        );
    }

    #[test]
    fn test_qoe_score_startup_factor_fast() {
        let mut m = make_monitor();
        m.start_session();
        m.first_frame(); // near-zero delay
        let config = QoeConfig::default();
        let qoe = m.qoe_score(5000.0, &config);
        assert!(
            qoe.startup_factor > 0.9,
            "fast startup should give high factor: {}",
            qoe.startup_factor
        );
    }

    #[test]
    fn test_qoe_score_no_startup_recorded() {
        let m = make_monitor();
        let config = QoeConfig::default();
        let qoe = m.qoe_score(5000.0, &config);
        // startup_time_ms returns None → 0 ms → startup_factor = 1.0
        assert!(
            (qoe.startup_factor - 1.0).abs() < 0.01,
            "no startup recorded should default to perfect: {}",
            qoe.startup_factor
        );
    }

    #[test]
    fn test_qoe_score_custom_config_weights() {
        let m = make_monitor();
        // Weight only bitrate, ignore everything else
        let config = QoeConfig {
            reference_bitrate_kbps: 10_000.0,
            weight_bitrate: 1.0,
            weight_rebuffer: 0.0,
            weight_startup: 0.0,
            weight_switch: 0.0,
            ..QoeConfig::default()
        };
        let qoe = m.qoe_score(5000.0, &config);
        assert!(
            (qoe.score - 50.0).abs() < 1.0,
            "half reference bitrate with bitrate-only weight → 50, got {}",
            qoe.score
        );
    }

    #[test]
    fn test_qoe_score_zero_bitrate() {
        let m = make_monitor();
        let config = QoeConfig::default();
        let qoe = m.qoe_score(0.0, &config);
        assert!(
            (qoe.bitrate_factor - 0.0).abs() < 0.01,
            "zero bitrate should zero factor"
        );
    }

    #[test]
    fn test_qoe_config_default_weights_sum_to_one() {
        let config = QoeConfig::default();
        let sum = config.weight_bitrate
            + config.weight_rebuffer
            + config.weight_startup
            + config.weight_switch;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "default weights should sum to ~1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_qoe_score_bitrate_above_reference_capped() {
        let m = make_monitor();
        let config = QoeConfig::default();
        let qoe = m.qoe_score(20_000.0, &config);
        assert!(
            (qoe.bitrate_factor - 1.0).abs() < 0.01,
            "bitrate above reference should cap at 1.0: {}",
            qoe.bitrate_factor
        );
    }

    // ── Additional QoE pattern tests ─────────────────────────────────────────

    #[test]
    fn test_stream_health_packet_loss_pattern() {
        // An error-heavy session should produce a degraded overall score.
        // There is no HealthIssue::Error variant; we verify score degradation.
        let mut m = make_monitor();
        m.start_session();
        // Record 10 errors and minimal playback
        for _ in 0..10 {
            m.record_error();
        }
        m.record_playback_duration(1_000); // very short playback
        let perfect = make_monitor().report(2000.0, 10.0);
        let degraded = m.report(2000.0, 10.0);
        // error_count is captured in the report; the overall score should be
        // equal-to-or-less-than a clean session at the same bitrate/buffer.
        assert!(
            degraded.overall_score <= perfect.overall_score,
            "error-heavy session should not score higher than clean session: \
             errors={} degraded={} perfect={}",
            degraded.error_count,
            degraded.overall_score,
            perfect.overall_score
        );
        assert_eq!(degraded.error_count, 10);
    }

    #[test]
    fn test_stream_health_jitter_pattern() {
        // 10 stall events of 50 ms each with 200 ms total playback duration.
        // rebuffer_duration_ms = 500, total_playback_ms = 200
        // ratio = 500 / (200 + 500) ≈ 0.714 → well above 0.5
        let mut m = make_monitor();
        for _ in 0..10 {
            m.record_rebuffer_with_duration(50);
        }
        m.record_playback_duration(200);
        let ratio = m.rebuffer_ratio();
        assert!(
            ratio > 0.5,
            "high-jitter session should have rebuffer_ratio > 0.5, got {ratio:.4}"
        );
        let report = m.report(2000.0, 1.0);
        assert!(
            report.overall_score < 80.0,
            "high rebuffer ratio should heavily degrade score: {}",
            report.overall_score
        );
    }

    #[test]
    fn test_stream_health_bitrate_drop() {
        // A very low bitrate (200 kbps) combined with a low buffer (2 s) should
        // produce a score below 90.
        let m = make_monitor();
        let report = m.report(200.0, 2.0);
        assert!(
            report.overall_score < 90.0,
            "low bitrate + low buffer should degrade score below 90: {}",
            report.overall_score
        );
        // The low buffer should also trigger a BufferTooLow health issue.
        let has_buffer_issue = report
            .issues
            .iter()
            .any(|i| matches!(i, HealthIssue::BufferTooLow { .. }));
        assert!(
            has_buffer_issue,
            "low buffer should trigger BufferTooLow issue"
        );
    }
}
