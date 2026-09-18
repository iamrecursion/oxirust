#![allow(dead_code)]
//! Stream health monitoring for video-over-IP connections.
//!
//! Provides real-time health scoring, quality-of-service tracking,
//! alert generation, and trend analysis for active streams. Designed
//! for control rooms and monitoring dashboards where operators need
//! at-a-glance stream status.

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Health status
// ---------------------------------------------------------------------------

/// Overall stream health grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HealthGrade {
    /// Stream is in critical condition (score 0..25).
    Critical,
    /// Stream has significant quality issues (score 25..50).
    Poor,
    /// Stream has minor issues but is usable (score 50..75).
    Fair,
    /// Stream is healthy (score 75..100).
    Good,
    /// Stream is in excellent condition (score 95..100).
    Excellent,
}

impl HealthGrade {
    /// Derive grade from a 0..100 score.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 95.0 {
            Self::Excellent
        } else if score >= 75.0 {
            Self::Good
        } else if score >= 50.0 {
            Self::Fair
        } else if score >= 25.0 {
            Self::Poor
        } else {
            Self::Critical
        }
    }
}

impl std::fmt::Display for HealthGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::Poor => write!(f, "POOR"),
            Self::Fair => write!(f, "FAIR"),
            Self::Good => write!(f, "GOOD"),
            Self::Excellent => write!(f, "EXCELLENT"),
        }
    }
}

// ---------------------------------------------------------------------------
// Alert
// ---------------------------------------------------------------------------

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertSeverity {
    /// Informational notice.
    Info,
    /// Warning — may require attention.
    Warning,
    /// Error — stream quality is degraded.
    Error,
    /// Critical — stream is likely unusable.
    Critical,
}

/// A health alert raised by the monitor.
#[derive(Debug, Clone)]
pub struct HealthAlert {
    /// Severity of the alert.
    pub severity: AlertSeverity,
    /// Human-readable description.
    pub message: String,
    /// Measurement index when the alert was raised.
    pub sample_index: usize,
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// A single health measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct HealthSample {
    /// Sample index (monotonically increasing).
    pub index: usize,
    /// Packet loss ratio for the measurement interval (0.0..1.0).
    pub packet_loss: f64,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f64,
    /// Jitter in milliseconds.
    pub jitter_ms: f64,
    /// Bitrate in bits per second.
    pub bitrate_bps: f64,
    /// Number of FEC recoveries in the interval.
    pub fec_recoveries: u32,
    /// Number of unrecoverable errors in the interval.
    pub unrecoverable_errors: u32,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Stream health monitor configuration.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Rolling window size for averaging.
    pub window_size: usize,
    /// Packet loss threshold for warning (0.0..1.0).
    pub loss_warn_threshold: f64,
    /// Packet loss threshold for error (0.0..1.0).
    pub loss_error_threshold: f64,
    /// RTT threshold for warning (ms).
    pub rtt_warn_ms: f64,
    /// RTT threshold for error (ms).
    pub rtt_error_ms: f64,
    /// Jitter threshold for warning (ms).
    pub jitter_warn_ms: f64,
    /// Expected nominal bitrate (bps) — zero disables bitrate checks.
    pub nominal_bitrate_bps: f64,
    /// Acceptable bitrate deviation ratio (0.0..1.0).
    pub bitrate_deviation_ratio: f64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            loss_warn_threshold: 0.001,
            loss_error_threshold: 0.01,
            rtt_warn_ms: 50.0,
            rtt_error_ms: 200.0,
            jitter_warn_ms: 10.0,
            nominal_bitrate_bps: 0.0,
            bitrate_deviation_ratio: 0.2,
        }
    }
}

// ---------------------------------------------------------------------------
// Health snapshot
// ---------------------------------------------------------------------------

/// Current health snapshot (summary of the rolling window).
#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    /// Number of samples in the window.
    pub sample_count: usize,
    /// Composite health score (0..100).
    pub score: f64,
    /// Derived grade.
    pub grade: HealthGrade,
    /// Average packet loss in the window.
    pub avg_loss: f64,
    /// Average RTT in the window (ms).
    pub avg_rtt_ms: f64,
    /// Average jitter in the window (ms).
    pub avg_jitter_ms: f64,
    /// Average bitrate in the window (bps).
    pub avg_bitrate_bps: f64,
    /// Peak packet loss in the window.
    pub peak_loss: f64,
    /// Peak RTT in the window (ms).
    pub peak_rtt_ms: f64,
    /// Total FEC recoveries in the window.
    pub total_fec_recoveries: u32,
    /// Total unrecoverable errors in the window.
    pub total_unrecoverable: u32,
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

/// Real-time stream health monitor.
#[derive(Debug)]
pub struct StreamHealthMonitor {
    /// Configuration.
    config: HealthConfig,
    /// Rolling window of recent samples.
    window: VecDeque<HealthSample>,
    /// All alerts raised so far.
    alerts: Vec<HealthAlert>,
    /// Total samples ingested.
    total_samples: usize,
}

impl StreamHealthMonitor {
    /// Create a new stream health monitor.
    #[must_use]
    pub fn new(config: HealthConfig) -> Self {
        Self {
            window: VecDeque::with_capacity(config.window_size),
            config,
            alerts: Vec::new(),
            total_samples: 0,
        }
    }

    /// Ingest a health measurement sample.
    pub fn push_sample(&mut self, sample: HealthSample) {
        // Maintain window size
        if self.window.len() == self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(sample);
        self.total_samples += 1;

        // Check for alerts
        self.check_alerts(&sample);
    }

    /// Build a convenience sample and push it.
    pub fn record(
        &mut self,
        packet_loss: f64,
        rtt_ms: f64,
        jitter_ms: f64,
        bitrate_bps: f64,
        fec_recoveries: u32,
        unrecoverable: u32,
    ) {
        let sample = HealthSample {
            index: self.total_samples,
            packet_loss,
            rtt_ms,
            jitter_ms,
            bitrate_bps,
            fec_recoveries,
            unrecoverable_errors: unrecoverable,
        };
        self.push_sample(sample);
    }

    /// Get the current health snapshot.
    pub fn snapshot(&self) -> HealthSnapshot {
        let n = self.window.len();
        if n == 0 {
            return HealthSnapshot {
                sample_count: 0,
                score: 100.0,
                grade: HealthGrade::Excellent,
                avg_loss: 0.0,
                avg_rtt_ms: 0.0,
                avg_jitter_ms: 0.0,
                avg_bitrate_bps: 0.0,
                peak_loss: 0.0,
                peak_rtt_ms: 0.0,
                total_fec_recoveries: 0,
                total_unrecoverable: 0,
            };
        }

        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;

        let avg_loss = self.window.iter().map(|s| s.packet_loss).sum::<f64>() / nf;
        let avg_rtt = self.window.iter().map(|s| s.rtt_ms).sum::<f64>() / nf;
        let avg_jitter = self.window.iter().map(|s| s.jitter_ms).sum::<f64>() / nf;
        let avg_bitrate = self.window.iter().map(|s| s.bitrate_bps).sum::<f64>() / nf;
        let peak_loss = self
            .window
            .iter()
            .map(|s| s.packet_loss)
            .fold(0.0_f64, f64::max);
        let peak_rtt = self.window.iter().map(|s| s.rtt_ms).fold(0.0_f64, f64::max);
        let total_fec: u32 = self.window.iter().map(|s| s.fec_recoveries).sum();
        let total_unrec: u32 = self.window.iter().map(|s| s.unrecoverable_errors).sum();

        let score = self.compute_score(avg_loss, avg_rtt, avg_jitter, avg_bitrate);
        let grade = HealthGrade::from_score(score);

        HealthSnapshot {
            sample_count: n,
            score,
            grade,
            avg_loss,
            avg_rtt_ms: avg_rtt,
            avg_jitter_ms: avg_jitter,
            avg_bitrate_bps: avg_bitrate,
            peak_loss,
            peak_rtt_ms: peak_rtt,
            total_fec_recoveries: total_fec,
            total_unrecoverable: total_unrec,
        }
    }

    /// Return all accumulated alerts.
    #[must_use]
    pub fn alerts(&self) -> &[HealthAlert] {
        &self.alerts
    }

    /// Total number of samples ingested.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Estimate uptime based on the ratio of non-critical samples.
    #[must_use]
    pub fn estimated_uptime_ratio(&self) -> f64 {
        if self.window.is_empty() {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let good = self
            .window
            .iter()
            .filter(|s| {
                s.packet_loss < self.config.loss_error_threshold && s.unrecoverable_errors == 0
            })
            .count() as f64;
        #[allow(clippy::cast_precision_loss)]
        let total = self.window.len() as f64;
        good / total
    }

    // -- internal --

    /// Compute composite score 0..100.
    fn compute_score(&self, loss: f64, rtt: f64, jitter: f64, bitrate: f64) -> f64 {
        // Loss penalty (most important)
        let loss_score = (1.0 - loss * 100.0).max(0.0) * 40.0; // 40 points

        // RTT penalty
        let rtt_score = (1.0 - (rtt / self.config.rtt_error_ms).min(1.0)) * 25.0; // 25 points

        // Jitter penalty
        let jitter_score = (1.0 - (jitter / (self.config.jitter_warn_ms * 5.0)).min(1.0)) * 20.0; // 20 points

        // Bitrate stability
        let bitrate_score = if self.config.nominal_bitrate_bps > 0.0 {
            let dev = ((bitrate - self.config.nominal_bitrate_bps)
                / self.config.nominal_bitrate_bps)
                .abs();
            (1.0 - (dev / self.config.bitrate_deviation_ratio).min(1.0)) * 15.0
        } else {
            15.0 // full marks if no nominal set
        };

        (loss_score + rtt_score + jitter_score + bitrate_score).clamp(0.0, 100.0)
    }

    /// Check a single sample against thresholds and raise alerts.
    fn check_alerts(&mut self, sample: &HealthSample) {
        if sample.packet_loss >= self.config.loss_error_threshold {
            self.alerts.push(HealthAlert {
                severity: AlertSeverity::Error,
                message: format!(
                    "High packet loss: {:.3}% (threshold {:.3}%)",
                    sample.packet_loss * 100.0,
                    self.config.loss_error_threshold * 100.0,
                ),
                sample_index: sample.index,
            });
        } else if sample.packet_loss >= self.config.loss_warn_threshold {
            self.alerts.push(HealthAlert {
                severity: AlertSeverity::Warning,
                message: format!("Elevated packet loss: {:.3}%", sample.packet_loss * 100.0),
                sample_index: sample.index,
            });
        }

        if sample.rtt_ms >= self.config.rtt_error_ms {
            self.alerts.push(HealthAlert {
                severity: AlertSeverity::Error,
                message: format!("High RTT: {:.1} ms", sample.rtt_ms),
                sample_index: sample.index,
            });
        } else if sample.rtt_ms >= self.config.rtt_warn_ms {
            self.alerts.push(HealthAlert {
                severity: AlertSeverity::Warning,
                message: format!("Elevated RTT: {:.1} ms", sample.rtt_ms),
                sample_index: sample.index,
            });
        }

        if sample.unrecoverable_errors > 0 {
            self.alerts.push(HealthAlert {
                severity: AlertSeverity::Critical,
                message: format!(
                    "{} unrecoverable packet errors",
                    sample.unrecoverable_errors,
                ),
                sample_index: sample.index,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: `Duration`-based SLA check
// ---------------------------------------------------------------------------

/// Check whether an average health snapshot meets an SLA.
#[must_use]
pub fn meets_sla(
    snapshot: &HealthSnapshot,
    max_loss: f64,
    max_rtt: Duration,
    min_score: f64,
) -> bool {
    snapshot.avg_loss <= max_loss
        && snapshot.avg_rtt_ms <= max_rtt.as_millis() as f64
        && snapshot.score >= min_score
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_sample(idx: usize) -> HealthSample {
        HealthSample {
            index: idx,
            packet_loss: 0.0,
            rtt_ms: 5.0,
            jitter_ms: 0.5,
            bitrate_bps: 10_000_000.0,
            fec_recoveries: 0,
            unrecoverable_errors: 0,
        }
    }

    fn lossy_sample(idx: usize) -> HealthSample {
        HealthSample {
            index: idx,
            packet_loss: 0.05,
            rtt_ms: 250.0,
            jitter_ms: 30.0,
            bitrate_bps: 5_000_000.0,
            fec_recoveries: 10,
            unrecoverable_errors: 3,
        }
    }

    #[test]
    fn test_health_grade_from_score() {
        assert_eq!(HealthGrade::from_score(100.0), HealthGrade::Excellent);
        assert_eq!(HealthGrade::from_score(95.0), HealthGrade::Excellent);
        assert_eq!(HealthGrade::from_score(80.0), HealthGrade::Good);
        assert_eq!(HealthGrade::from_score(60.0), HealthGrade::Fair);
        assert_eq!(HealthGrade::from_score(30.0), HealthGrade::Poor);
        assert_eq!(HealthGrade::from_score(10.0), HealthGrade::Critical);
    }

    #[test]
    fn test_empty_monitor_snapshot() {
        let m = StreamHealthMonitor::new(HealthConfig::default());
        let snap = m.snapshot();
        assert_eq!(snap.sample_count, 0);
        assert!((snap.score - 100.0).abs() < 0.01);
        assert_eq!(snap.grade, HealthGrade::Excellent);
    }

    #[test]
    fn test_healthy_stream() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        for i in 0..10 {
            m.push_sample(healthy_sample(i));
        }
        let snap = m.snapshot();
        assert!(snap.score >= 90.0);
        assert!(snap.grade >= HealthGrade::Good);
        assert!(snap.avg_loss < 0.001);
    }

    #[test]
    fn test_degraded_stream() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        for i in 0..10 {
            m.push_sample(lossy_sample(i));
        }
        let snap = m.snapshot();
        assert!(snap.score < 50.0);
        assert!(snap.avg_loss > 0.01);
    }

    #[test]
    fn test_alerts_on_loss() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        m.push_sample(lossy_sample(0));
        assert!(!m.alerts().is_empty());
        // Should have at least an error-level alert for packet loss
        assert!(m
            .alerts()
            .iter()
            .any(|a| a.severity == AlertSeverity::Error));
    }

    #[test]
    fn test_alerts_on_unrecoverable() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        m.push_sample(lossy_sample(0));
        assert!(m
            .alerts()
            .iter()
            .any(|a| a.severity == AlertSeverity::Critical));
    }

    #[test]
    fn test_no_alerts_healthy() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        m.push_sample(healthy_sample(0));
        assert!(m.alerts().is_empty());
    }

    #[test]
    fn test_window_eviction() {
        let cfg = HealthConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut m = StreamHealthMonitor::new(cfg);
        for i in 0..20 {
            m.push_sample(healthy_sample(i));
        }
        let snap = m.snapshot();
        assert_eq!(snap.sample_count, 5);
        assert_eq!(m.total_samples(), 20);
    }

    #[test]
    fn test_record_convenience() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        m.record(0.0, 5.0, 1.0, 10_000_000.0, 0, 0);
        assert_eq!(m.total_samples(), 1);
        let snap = m.snapshot();
        assert!(snap.score >= 90.0);
    }

    #[test]
    fn test_estimated_uptime() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        for i in 0..8 {
            m.push_sample(healthy_sample(i));
        }
        m.push_sample(lossy_sample(8));
        m.push_sample(lossy_sample(9));
        let uptime = m.estimated_uptime_ratio();
        assert!(uptime > 0.5 && uptime < 1.0);
    }

    #[test]
    fn test_meets_sla_pass() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        for i in 0..10 {
            m.push_sample(healthy_sample(i));
        }
        let snap = m.snapshot();
        assert!(meets_sla(&snap, 0.001, Duration::from_millis(100), 80.0));
    }

    #[test]
    fn test_meets_sla_fail() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        for i in 0..10 {
            m.push_sample(lossy_sample(i));
        }
        let snap = m.snapshot();
        assert!(!meets_sla(&snap, 0.001, Duration::from_millis(100), 80.0));
    }

    #[test]
    fn test_health_grade_display() {
        assert_eq!(HealthGrade::Critical.to_string(), "CRITICAL");
        assert_eq!(HealthGrade::Good.to_string(), "GOOD");
        assert_eq!(HealthGrade::Excellent.to_string(), "EXCELLENT");
    }

    #[test]
    fn test_peak_values() {
        let mut m = StreamHealthMonitor::new(HealthConfig::default());
        m.push_sample(healthy_sample(0));
        m.push_sample(lossy_sample(1));
        let snap = m.snapshot();
        assert!(snap.peak_loss >= 0.05);
        assert!(snap.peak_rtt_ms >= 250.0);
    }
}
