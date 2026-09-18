//! Service Level Objective (SLO) tracking.
//!
//! Tracks measurements against defined SLO targets and reports compliance.

/// The type of SLO being measured.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SloType {
    /// Percentage of time the service is available.
    Availability,
    /// Latency (lower is better).
    Latency,
    /// Operations per second (higher is better).
    Throughput,
    /// Fraction of requests that result in errors (lower is better).
    ErrorRate,
}

impl SloType {
    /// Return the natural unit for this SLO type.
    #[must_use]
    pub fn unit(&self) -> &str {
        match self {
            Self::Availability => "%",
            Self::Latency => "ms",
            Self::Throughput => "req/s",
            Self::ErrorRate => "%",
        }
    }
}

/// Definition of a service level objective.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SloDefinition {
    /// Human-readable SLO name.
    pub name: String,
    /// Type of SLO.
    pub slo_type: SloType,
    /// Target value (meaning depends on `slo_type`).
    pub target: f64,
    /// Measurement window in hours.
    pub window_hours: u32,
}

impl SloDefinition {
    /// Pre-built SLO: 99.9 % availability over 30 days.
    #[must_use]
    pub fn availability_99_9() -> Self {
        Self {
            name: "Availability 99.9%".to_string(),
            slo_type: SloType::Availability,
            target: 99.9,
            window_hours: 720,
        }
    }

    /// Pre-built SLO: p99 latency ≤ 100 ms.
    #[must_use]
    pub fn latency_p99_100ms() -> Self {
        Self {
            name: "Latency p99 100ms".to_string(),
            slo_type: SloType::Latency,
            target: 100.0,
            window_hours: 1,
        }
    }
}

/// A single data point recorded against an SLO.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SloMeasurement {
    /// Unix epoch timestamp of the measurement.
    pub timestamp_epoch: u64,
    /// Measured value.
    pub value: f64,
}

impl SloMeasurement {
    /// Returns `true` if this measurement meets the given SLO target.
    ///
    /// * `Availability` / `Throughput` – value must be **≥** target.
    /// * `Latency` / `ErrorRate` – value must be **≤** target.
    #[must_use]
    pub fn meets_target(&self, target: f64, slo_type: &SloType) -> bool {
        match slo_type {
            SloType::Availability | SloType::Throughput => self.value >= target,
            SloType::Latency | SloType::ErrorRate => self.value <= target,
        }
    }
}

/// Tracker that accumulates measurements for a single SLO.
#[derive(Debug)]
#[allow(dead_code)]
pub struct SloTracker {
    /// The SLO definition.
    pub definition: SloDefinition,
    measurements: Vec<SloMeasurement>,
}

impl SloTracker {
    /// Create a new tracker for the supplied definition.
    #[must_use]
    pub fn new(definition: SloDefinition) -> Self {
        Self {
            definition,
            measurements: Vec::new(),
        }
    }

    /// Record a new measurement.
    pub fn record(&mut self, epoch: u64, value: f64) {
        self.measurements.push(SloMeasurement {
            timestamp_epoch: epoch,
            value,
        });
    }

    /// Calculate the percentage of stored measurements that meet the SLO target.
    ///
    /// Returns `100.0` if there are no measurements yet.
    #[must_use]
    pub fn current_compliance_pct(&self) -> f64 {
        if self.measurements.is_empty() {
            return 100.0;
        }

        let passing = self
            .measurements
            .iter()
            .filter(|m| m.meets_target(self.definition.target, &self.definition.slo_type))
            .count();

        #[allow(clippy::cast_precision_loss)]
        let pct = (passing as f64 / self.measurements.len() as f64) * 100.0;
        pct
    }

    /// Returns `true` if the current compliance percentage is at or above `threshold_pct`.
    #[must_use]
    pub fn is_meeting_slo(&self, threshold_pct: f64) -> bool {
        self.current_compliance_pct() >= threshold_pct
    }

    /// Number of measurements recorded.
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ //
    // SloType
    // ------------------------------------------------------------------ //

    #[test]
    fn test_slo_type_unit_availability() {
        assert_eq!(SloType::Availability.unit(), "%");
    }

    #[test]
    fn test_slo_type_unit_latency() {
        assert_eq!(SloType::Latency.unit(), "ms");
    }

    #[test]
    fn test_slo_type_unit_throughput() {
        assert_eq!(SloType::Throughput.unit(), "req/s");
    }

    #[test]
    fn test_slo_type_unit_error_rate() {
        assert_eq!(SloType::ErrorRate.unit(), "%");
    }

    // ------------------------------------------------------------------ //
    // SloDefinition
    // ------------------------------------------------------------------ //

    #[test]
    fn test_availability_99_9_preset() {
        let slo = SloDefinition::availability_99_9();
        assert_eq!(slo.slo_type, SloType::Availability);
        assert!((slo.target - 99.9).abs() < f64::EPSILON);
        assert_eq!(slo.window_hours, 720);
    }

    #[test]
    fn test_latency_p99_100ms_preset() {
        let slo = SloDefinition::latency_p99_100ms();
        assert_eq!(slo.slo_type, SloType::Latency);
        assert!((slo.target - 100.0).abs() < f64::EPSILON);
        assert_eq!(slo.window_hours, 1);
    }

    // ------------------------------------------------------------------ //
    // SloMeasurement
    // ------------------------------------------------------------------ //

    #[test]
    fn test_measurement_meets_availability() {
        let m = SloMeasurement {
            timestamp_epoch: 0,
            value: 99.95,
        };
        assert!(m.meets_target(99.9, &SloType::Availability));
    }

    #[test]
    fn test_measurement_fails_availability() {
        let m = SloMeasurement {
            timestamp_epoch: 0,
            value: 99.0,
        };
        assert!(!m.meets_target(99.9, &SloType::Availability));
    }

    #[test]
    fn test_measurement_meets_latency() {
        let m = SloMeasurement {
            timestamp_epoch: 0,
            value: 80.0,
        };
        assert!(m.meets_target(100.0, &SloType::Latency));
    }

    #[test]
    fn test_measurement_fails_latency() {
        let m = SloMeasurement {
            timestamp_epoch: 0,
            value: 150.0,
        };
        assert!(!m.meets_target(100.0, &SloType::Latency));
    }

    #[test]
    fn test_measurement_meets_error_rate() {
        let m = SloMeasurement {
            timestamp_epoch: 0,
            value: 0.5,
        };
        assert!(m.meets_target(1.0, &SloType::ErrorRate));
    }

    #[test]
    fn test_measurement_meets_throughput() {
        let m = SloMeasurement {
            timestamp_epoch: 0,
            value: 500.0,
        };
        assert!(m.meets_target(400.0, &SloType::Throughput));
    }

    // ------------------------------------------------------------------ //
    // SloTracker
    // ------------------------------------------------------------------ //

    #[test]
    fn test_tracker_empty_is_100_pct() {
        let tracker = SloTracker::new(SloDefinition::availability_99_9());
        assert!((tracker.current_compliance_pct() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_record_increases_count() {
        let mut tracker = SloTracker::new(SloDefinition::latency_p99_100ms());
        tracker.record(1, 50.0);
        tracker.record(2, 60.0);
        assert_eq!(tracker.measurement_count(), 2);
    }

    #[test]
    fn test_tracker_full_compliance() {
        let mut tracker = SloTracker::new(SloDefinition::latency_p99_100ms());
        tracker.record(1, 50.0);
        tracker.record(2, 80.0);
        tracker.record(3, 99.0);
        assert!((tracker.current_compliance_pct() - 100.0).abs() < f64::EPSILON);
        assert!(tracker.is_meeting_slo(99.9));
    }

    #[test]
    fn test_tracker_partial_compliance() {
        let mut tracker = SloTracker::new(SloDefinition::latency_p99_100ms());
        tracker.record(1, 50.0); // pass
        tracker.record(2, 200.0); // fail
                                  // 50 % compliance
        let pct = tracker.current_compliance_pct();
        assert!((pct - 50.0).abs() < f64::EPSILON);
        assert!(!tracker.is_meeting_slo(99.9));
    }

    #[test]
    fn test_tracker_is_meeting_slo_exact() {
        let mut tracker = SloTracker::new(SloDefinition::availability_99_9());
        // 999 passing + 1 failing = 99.9 %
        for _ in 0..999 {
            tracker.record(0, 99.95); // pass
        }
        tracker.record(0, 50.0); // fail
        assert!(tracker.is_meeting_slo(99.9));
    }
}
