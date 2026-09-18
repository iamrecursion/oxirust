//! Synchronisation monitoring: drift tracking, alarm conditions, and statistics.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Severity of a synchronisation alarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlarmSeverity {
    /// Informational notice only.
    Info,
    /// Degraded performance, not yet critical.
    Warning,
    /// Synchronisation is unreliable.
    Critical,
    /// Complete loss of synchronisation.
    Emergency,
}

/// A specific alarm condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlarmKind {
    /// Offset has exceeded the warning threshold.
    OffsetExceeded,
    /// Clock frequency drift is too high.
    DriftExceeded,
    /// No synchronisation source available.
    SourceLost,
    /// Packet loss too high.
    PacketLoss,
    /// Holdover mode active (no external reference).
    HoldoverActive,
}

/// An alarm event recorded by the monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmEvent {
    /// The kind of alarm.
    pub kind: AlarmKind,
    /// Severity level.
    pub severity: AlarmSeverity,
    /// Description.
    pub message: String,
    /// Monotonic timestamp in seconds (relative to monitor creation).
    pub timestamp_secs: f64,
}

/// Configuration thresholds for the monitor.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Offset magnitude (ns) that triggers a Warning.
    pub offset_warning_ns: f64,
    /// Offset magnitude (ns) that triggers Critical.
    pub offset_critical_ns: f64,
    /// Frequency drift (ppb) that triggers Warning.
    pub drift_warning_ppb: f64,
    /// Frequency drift (ppb) that triggers Critical.
    pub drift_critical_ppb: f64,
    /// Number of recent samples to keep in the history window.
    pub history_size: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            offset_warning_ns: 1_000.0,
            offset_critical_ns: 10_000.0,
            drift_warning_ppb: 10_000.0,
            drift_critical_ppb: 100_000.0,
            history_size: 64,
        }
    }
}

/// Rolling statistics over the sample history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftStats {
    /// Number of samples in the window.
    pub sample_count: usize,
    /// Mean offset in nanoseconds.
    pub mean_offset_ns: f64,
    /// Maximum absolute offset observed.
    pub max_offset_ns: f64,
    /// Standard deviation of offsets.
    pub std_dev_ns: f64,
    /// Mean drift in ppb.
    pub mean_drift_ppb: f64,
}

/// Monitor that tracks drift and raises alarms.
#[derive(Debug)]
pub struct SyncMonitor {
    config: MonitorConfig,
    offset_history: VecDeque<f64>,
    drift_history: VecDeque<f64>,
    alarms: Vec<AlarmEvent>,
    elapsed_secs: f64,
}

impl SyncMonitor {
    /// Create a new monitor with the given configuration.
    #[must_use]
    pub fn new(config: MonitorConfig) -> Self {
        let capacity = config.history_size;
        Self {
            config,
            offset_history: VecDeque::with_capacity(capacity),
            drift_history: VecDeque::with_capacity(capacity),
            alarms: Vec::new(),
            elapsed_secs: 0.0,
        }
    }

    /// Create a monitor with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(MonitorConfig::default())
    }

    /// Feed a new measurement into the monitor.
    ///
    /// `offset_ns` – current clock offset in nanoseconds.
    /// `drift_ppb` – current frequency drift in ppb.
    /// `dt_secs`   – seconds elapsed since last call.
    pub fn update(&mut self, offset_ns: f64, drift_ppb: f64, dt_secs: f64) {
        self.elapsed_secs += dt_secs;

        // Maintain rolling windows
        if self.offset_history.len() >= self.config.history_size {
            self.offset_history.pop_front();
        }
        if self.drift_history.len() >= self.config.history_size {
            self.drift_history.pop_front();
        }
        self.offset_history.push_back(offset_ns);
        self.drift_history.push_back(drift_ppb);

        // Check offset thresholds
        let abs_offset = offset_ns.abs();
        if abs_offset >= self.config.offset_critical_ns {
            self.raise_alarm(
                AlarmKind::OffsetExceeded,
                AlarmSeverity::Critical,
                format!("Offset {:.0} ns exceeds critical threshold", abs_offset),
            );
        } else if abs_offset >= self.config.offset_warning_ns {
            self.raise_alarm(
                AlarmKind::OffsetExceeded,
                AlarmSeverity::Warning,
                format!("Offset {:.0} ns exceeds warning threshold", abs_offset),
            );
        }

        // Check drift thresholds
        let abs_drift = drift_ppb.abs();
        if abs_drift >= self.config.drift_critical_ppb {
            self.raise_alarm(
                AlarmKind::DriftExceeded,
                AlarmSeverity::Critical,
                format!("Drift {:.0} ppb exceeds critical threshold", abs_drift),
            );
        } else if abs_drift >= self.config.drift_warning_ppb {
            self.raise_alarm(
                AlarmKind::DriftExceeded,
                AlarmSeverity::Warning,
                format!("Drift {:.0} ppb exceeds warning threshold", abs_drift),
            );
        }
    }

    /// Signal that the sync source has been lost.
    pub fn signal_source_lost(&mut self) {
        self.raise_alarm(
            AlarmKind::SourceLost,
            AlarmSeverity::Emergency,
            "Synchronisation source lost".to_string(),
        );
    }

    /// Signal that holdover mode is active.
    pub fn signal_holdover(&mut self) {
        self.raise_alarm(
            AlarmKind::HoldoverActive,
            AlarmSeverity::Warning,
            "Operating in holdover mode".to_string(),
        );
    }

    /// Compute rolling statistics over the current history window.
    #[must_use]
    pub fn stats(&self) -> DriftStats {
        let n = self.offset_history.len();
        if n == 0 {
            return DriftStats {
                sample_count: 0,
                mean_offset_ns: 0.0,
                max_offset_ns: 0.0,
                std_dev_ns: 0.0,
                mean_drift_ppb: 0.0,
            };
        }

        let sum: f64 = self.offset_history.iter().sum();
        let mean = sum / n as f64;
        let max_abs = self
            .offset_history
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        let variance = self
            .offset_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        let drift_sum: f64 = self.drift_history.iter().sum();
        let mean_drift = if self.drift_history.is_empty() {
            0.0
        } else {
            drift_sum / self.drift_history.len() as f64
        };

        DriftStats {
            sample_count: n,
            mean_offset_ns: mean,
            max_offset_ns: max_abs,
            std_dev_ns: std_dev,
            mean_drift_ppb: mean_drift,
        }
    }

    /// All alarms recorded so far.
    #[must_use]
    pub fn alarms(&self) -> &[AlarmEvent] {
        &self.alarms
    }

    /// Most recent alarm, if any.
    #[must_use]
    pub fn latest_alarm(&self) -> Option<&AlarmEvent> {
        self.alarms.last()
    }

    /// Count of alarms at or above the given severity.
    #[must_use]
    pub fn alarm_count_above(&self, severity: AlarmSeverity) -> usize {
        self.alarms
            .iter()
            .filter(|a| a.severity >= severity)
            .count()
    }

    /// Clear all recorded alarms.
    pub fn clear_alarms(&mut self) {
        self.alarms.clear();
    }

    /// Elapsed monitoring time in seconds.
    #[must_use]
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed_secs
    }

    fn raise_alarm(&mut self, kind: AlarmKind, severity: AlarmSeverity, message: String) {
        self.alarms.push(AlarmEvent {
            kind,
            severity,
            message,
            timestamp_secs: self.elapsed_secs,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> SyncMonitor {
        SyncMonitor::default_config()
    }

    #[test]
    fn test_default_config_thresholds() {
        let cfg = MonitorConfig::default();
        assert!(cfg.offset_warning_ns < cfg.offset_critical_ns);
        assert!(cfg.drift_warning_ppb < cfg.drift_critical_ppb);
    }

    #[test]
    fn test_no_alarms_within_thresholds() {
        let mut m = make_monitor();
        m.update(500.0, 5_000.0, 1.0);
        assert!(m.alarms().is_empty());
    }

    #[test]
    fn test_offset_warning_alarm() {
        let mut m = make_monitor();
        m.update(2_000.0, 0.0, 1.0);
        assert!(m.alarm_count_above(AlarmSeverity::Warning) > 0);
    }

    #[test]
    fn test_offset_critical_alarm() {
        let mut m = make_monitor();
        m.update(50_000.0, 0.0, 1.0);
        assert!(m.alarm_count_above(AlarmSeverity::Critical) > 0);
    }

    #[test]
    fn test_drift_warning_alarm() {
        let mut m = make_monitor();
        m.update(0.0, 20_000.0, 1.0);
        assert!(m.alarm_count_above(AlarmSeverity::Warning) > 0);
    }

    #[test]
    fn test_drift_critical_alarm() {
        let mut m = make_monitor();
        m.update(0.0, 200_000.0, 1.0);
        assert!(m.alarm_count_above(AlarmSeverity::Critical) > 0);
    }

    #[test]
    fn test_signal_source_lost() {
        let mut m = make_monitor();
        m.signal_source_lost();
        let a = m.latest_alarm().expect("should succeed in test");
        assert_eq!(a.kind, AlarmKind::SourceLost);
        assert_eq!(a.severity, AlarmSeverity::Emergency);
    }

    #[test]
    fn test_signal_holdover() {
        let mut m = make_monitor();
        m.signal_holdover();
        let a = m.latest_alarm().expect("should succeed in test");
        assert_eq!(a.kind, AlarmKind::HoldoverActive);
    }

    #[test]
    fn test_clear_alarms() {
        let mut m = make_monitor();
        m.signal_source_lost();
        m.clear_alarms();
        assert!(m.alarms().is_empty());
    }

    #[test]
    fn test_stats_empty() {
        let m = make_monitor();
        let s = m.stats();
        assert_eq!(s.sample_count, 0);
    }

    #[test]
    fn test_stats_mean() {
        let mut m = make_monitor();
        m.update(100.0, 0.0, 0.5);
        m.update(200.0, 0.0, 0.5);
        let s = m.stats();
        assert!((s.mean_offset_ns - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_max_offset() {
        let mut m = make_monitor();
        m.update(100.0, 0.0, 0.5);
        m.update(-500.0, 0.0, 0.5);
        let s = m.stats();
        assert!((s.max_offset_ns - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_elapsed_secs_accumulates() {
        let mut m = make_monitor();
        m.update(0.0, 0.0, 1.5);
        m.update(0.0, 0.0, 2.5);
        assert!((m.elapsed_secs() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_alarm_severity_ordering() {
        assert!(AlarmSeverity::Emergency > AlarmSeverity::Critical);
        assert!(AlarmSeverity::Critical > AlarmSeverity::Warning);
        assert!(AlarmSeverity::Warning > AlarmSeverity::Info);
    }

    #[test]
    fn test_history_capped_at_config_size() {
        let cfg = MonitorConfig {
            history_size: 4,
            ..MonitorConfig::default()
        };
        let mut m = SyncMonitor::new(cfg);
        for i in 0..10 {
            m.update(i as f64, 0.0, 0.1);
        }
        assert!(m.offset_history.len() <= 4);
    }
}
