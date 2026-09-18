//! Holdover mode for maintaining accuracy without external reference.

use super::drift::DriftEstimator;
use std::time::{Duration, Instant};

/// Holdover quality metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldoverQuality {
    /// Excellent (< 1 microsecond/day)
    Excellent,
    /// Good (< 10 microseconds/day)
    Good,
    /// Fair (< 100 microseconds/day)
    Fair,
    /// Poor (> 100 microseconds/day)
    Poor,
}

/// Holdover state manager.
pub struct HoldoverManager {
    /// Drift estimator
    drift_estimator: DriftEstimator,
    /// Time when holdover started
    holdover_start: Option<Instant>,
    /// Last known good offset
    last_offset: i64,
    /// Maximum holdover duration before quality degrades
    max_holdover_duration: Duration,
}

impl HoldoverManager {
    /// Create a new holdover manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drift_estimator: DriftEstimator::new(),
            holdover_start: None,
            last_offset: 0,
            max_holdover_duration: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Enter holdover mode.
    pub fn enter_holdover(&mut self, current_offset: i64) {
        self.holdover_start = Some(Instant::now());
        self.last_offset = current_offset;
    }

    /// Exit holdover mode.
    pub fn exit_holdover(&mut self) {
        self.holdover_start = None;
    }

    /// Check if in holdover mode.
    #[must_use]
    pub fn is_in_holdover(&self) -> bool {
        self.holdover_start.is_some()
    }

    /// Get time in holdover.
    #[must_use]
    pub fn time_in_holdover(&self) -> Option<Duration> {
        self.holdover_start.map(|start| start.elapsed())
    }

    /// Get predicted current offset during holdover.
    #[must_use]
    pub fn predicted_offset(&self) -> i64 {
        if let Some(start) = self.holdover_start {
            let duration = start.elapsed();
            self.drift_estimator
                .predict_offset(self.last_offset, duration)
        } else {
            self.last_offset
        }
    }

    /// Get holdover quality.
    #[must_use]
    pub fn quality(&self) -> HoldoverQuality {
        let drift = self.drift_estimator.drift_ppb().abs();

        // Convert ppb to microseconds/day
        // ppb * 86400 seconds/day / 1e9 = microseconds/day
        let drift_per_day = (drift * 86400.0) / 1000.0;

        if drift_per_day < 1.0 {
            HoldoverQuality::Excellent
        } else if drift_per_day < 10.0 {
            HoldoverQuality::Good
        } else if drift_per_day < 100.0 {
            HoldoverQuality::Fair
        } else {
            HoldoverQuality::Poor
        }
    }

    /// Check if holdover has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(duration) = self.time_in_holdover() {
            duration > self.max_holdover_duration
        } else {
            false
        }
    }

    /// Update drift estimate (call when synchronized).
    pub fn update_drift(&mut self, offset_ns: i64) {
        self.drift_estimator.update(offset_ns, Instant::now());
    }

    /// Get drift estimate.
    #[must_use]
    pub fn drift_ppb(&self) -> f64 {
        self.drift_estimator.drift_ppb()
    }
}

impl Default for HoldoverManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holdover_manager() {
        let mut manager = HoldoverManager::new();
        assert!(!manager.is_in_holdover());

        manager.enter_holdover(1000);
        assert!(manager.is_in_holdover());

        manager.exit_holdover();
        assert!(!manager.is_in_holdover());
    }

    #[test]
    fn test_holdover_quality() {
        let manager = HoldoverManager::new();
        // With no drift updates, quality will be based on default drift (0)
        let quality = manager.quality();
        assert_eq!(quality, HoldoverQuality::Excellent);
    }

    #[test]
    fn test_holdover_prediction() {
        let mut manager = HoldoverManager::new();
        // Update drift with some measurements
        manager.update_drift(0);
        manager.update_drift(1000);
        manager.enter_holdover(0);

        std::thread::sleep(Duration::from_millis(10));

        let _predicted = manager.predicted_offset();
        // Prediction will be based on estimated drift
        // Just verify it doesn't panic
    }
}
