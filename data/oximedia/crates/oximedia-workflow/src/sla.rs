//! Service Level Agreement (SLA) tracking for workflow orchestration.
//!
//! Provides SLA target definitions, violation detection, and reporting
//! for workflow processing times, queue depth, and availability.

#![allow(dead_code)]

/// SLA target configuration.
#[derive(Debug, Clone)]
pub struct SlaTarget {
    /// Maximum acceptable processing time in seconds.
    pub max_processing_secs: u64,
    /// Maximum acceptable queue depth (number of pending workflows).
    pub max_queue_depth: u32,
    /// Required availability percentage (e.g. 99.9).
    pub availability_pct: f32,
    /// Percentage of target at which to start sending notifications.
    /// E.g. 0.8 means notify when processing time hits 80% of max.
    pub notification_threshold_pct: f32,
}

impl SlaTarget {
    /// Create a new SLA target.
    #[must_use]
    pub fn new(
        max_processing_secs: u64,
        max_queue_depth: u32,
        availability_pct: f32,
        notification_threshold_pct: f32,
    ) -> Self {
        Self {
            max_processing_secs,
            max_queue_depth,
            availability_pct,
            notification_threshold_pct,
        }
    }
}

impl Default for SlaTarget {
    fn default() -> Self {
        Self {
            max_processing_secs: 3600,
            max_queue_depth: 100,
            availability_pct: 99.0,
            notification_threshold_pct: 0.8,
        }
    }
}

/// Type of SLA violation.
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    /// Workflow took longer than the maximum allowed processing time.
    ProcessingTimeout,
    /// Queue depth exceeded the maximum allowed depth.
    QueueDepthExceeded,
    /// System availability fell below the required percentage.
    AvailabilityBreach,
}

/// A recorded SLA violation.
#[derive(Debug, Clone)]
pub struct SlaViolation {
    /// Workflow identifier associated with the violation.
    pub workflow_id: String,
    /// Type of violation.
    pub violation_type: ViolationType,
    /// Expected value (SLA limit).
    pub expected: f64,
    /// Actual measured value.
    pub actual: f64,
    /// Timestamp of the violation in milliseconds since epoch.
    pub timestamp_ms: u64,
}

impl SlaViolation {
    /// Create a new SLA violation record.
    #[must_use]
    pub fn new(
        workflow_id: impl Into<String>,
        violation_type: ViolationType,
        expected: f64,
        actual: f64,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            violation_type,
            expected,
            actual,
            timestamp_ms,
        }
    }
}

/// Completion record for a workflow.
#[derive(Debug, Clone)]
struct CompletionRecord {
    workflow_id: String,
    processing_secs: u64,
    timestamp_ms: u64,
    success: bool,
}

/// SLA tracker that records workflow completions and detects violations.
#[derive(Debug)]
pub struct SlaTracker {
    target: SlaTarget,
    completions: Vec<CompletionRecord>,
    violations: Vec<SlaViolation>,
    /// Total uptime periods (in ms)
    total_uptime_ms: u64,
    /// Total downtime periods (in ms)
    total_downtime_ms: u64,
    /// Epoch start of tracking window (ms)
    tracking_start_ms: u64,
}

impl SlaTracker {
    /// Create a new SLA tracker with the given target.
    #[must_use]
    pub fn new(target: SlaTarget) -> Self {
        Self {
            target,
            completions: Vec::new(),
            violations: Vec::new(),
            total_uptime_ms: 0,
            total_downtime_ms: 0,
            tracking_start_ms: current_time_ms(),
        }
    }

    /// Create an SLA tracker with default targets.
    #[must_use]
    pub fn with_default_target() -> Self {
        Self::new(SlaTarget::default())
    }

    /// Record a workflow completion.
    pub fn record_completion(&mut self, workflow_id: &str, processing_secs: u64) {
        let now = current_time_ms();

        self.completions.push(CompletionRecord {
            workflow_id: workflow_id.to_string(),
            processing_secs,
            timestamp_ms: now,
            success: true,
        });

        // Check for processing timeout violation
        if processing_secs > self.target.max_processing_secs {
            self.violations.push(SlaViolation::new(
                workflow_id,
                ViolationType::ProcessingTimeout,
                self.target.max_processing_secs as f64,
                processing_secs as f64,
                now,
            ));
        }
    }

    /// Check if current queue depth exceeds SLA, returning a violation if so.
    #[must_use]
    pub fn check_queue_depth(&mut self, depth: u32) -> Option<SlaViolation> {
        if depth > self.target.max_queue_depth {
            let violation = SlaViolation::new(
                "system",
                ViolationType::QueueDepthExceeded,
                f64::from(self.target.max_queue_depth),
                f64::from(depth),
                current_time_ms(),
            );
            self.violations.push(violation.clone());
            Some(violation)
        } else {
            None
        }
    }

    /// Get violations within a time window (last `window_ms` milliseconds).
    #[must_use]
    pub fn violations_in_window(&self, window_ms: u64) -> Vec<&SlaViolation> {
        let now = current_time_ms();
        let cutoff = now.saturating_sub(window_ms);
        self.violations
            .iter()
            .filter(|v| v.timestamp_ms >= cutoff)
            .collect()
    }

    /// Get all violations.
    #[must_use]
    pub fn all_violations(&self) -> &[SlaViolation] {
        &self.violations
    }

    /// Record downtime (system unavailable for `duration_ms` milliseconds).
    pub fn record_downtime(&mut self, duration_ms: u64) {
        self.total_downtime_ms += duration_ms;

        let now = current_time_ms();
        let availability = self.current_availability();
        if availability < self.target.availability_pct {
            self.violations.push(SlaViolation::new(
                "system",
                ViolationType::AvailabilityBreach,
                f64::from(self.target.availability_pct),
                f64::from(availability),
                now,
            ));
        }
    }

    /// Record uptime (system available for `duration_ms` milliseconds).
    pub fn record_uptime(&mut self, duration_ms: u64) {
        self.total_uptime_ms += duration_ms;
    }

    /// Calculate current availability as a percentage.
    #[must_use]
    pub fn current_availability(&self) -> f32 {
        let total = self.total_uptime_ms + self.total_downtime_ms;
        if total == 0 {
            return 100.0;
        }
        (self.total_uptime_ms as f32 / total as f32) * 100.0
    }

    /// Get all processing times recorded.
    #[must_use]
    pub fn processing_times(&self) -> Vec<u64> {
        self.completions.iter().map(|c| c.processing_secs).collect()
    }

    /// Get processing time at a given percentile (0.0 to 1.0).
    #[must_use]
    pub fn percentile_processing_secs(&self, percentile: f64) -> f64 {
        let mut times: Vec<u64> = self.processing_times();
        if times.is_empty() {
            return 0.0;
        }
        times.sort_unstable();
        let idx = ((times.len() as f64 * percentile).ceil() as usize).min(times.len()) - 1;
        times[idx] as f64
    }

    /// Get the SLA target.
    #[must_use]
    pub fn target(&self) -> &SlaTarget {
        &self.target
    }

    /// Total number of completions recorded.
    #[must_use]
    pub fn total_completions(&self) -> u64 {
        self.completions.len() as u64
    }
}

/// SLA report for a time window.
#[derive(Debug, Clone)]
pub struct SlaReport {
    /// Time window in milliseconds covered by this report.
    pub period_ms: u64,
    /// Total number of workflows processed.
    pub total_workflows: u64,
    /// Number of SLA violations.
    pub violations: u64,
    /// Average processing time in seconds.
    pub avg_processing_secs: f64,
    /// 95th percentile processing time in seconds.
    pub p95_processing_secs: f64,
    /// Availability percentage.
    pub availability_pct: f32,
}

impl SlaReport {
    /// Generate an SLA report for the given tracker over the last `window_ms`.
    #[must_use]
    pub fn generate(tracker: &SlaTracker, window_ms: u64) -> Self {
        let now = current_time_ms();
        let cutoff = now.saturating_sub(window_ms);

        let window_completions: Vec<&CompletionRecord> = tracker
            .completions
            .iter()
            .filter(|c| c.timestamp_ms >= cutoff)
            .collect();

        let total_workflows = window_completions.len() as u64;

        let avg_processing_secs = if total_workflows > 0 {
            window_completions
                .iter()
                .map(|c| c.processing_secs as f64)
                .sum::<f64>()
                / total_workflows as f64
        } else {
            0.0
        };

        // Calculate P95 from window completions
        let p95_processing_secs = if window_completions.is_empty() {
            0.0
        } else {
            let mut times: Vec<u64> = window_completions
                .iter()
                .map(|c| c.processing_secs)
                .collect();
            times.sort_unstable();
            let idx = ((times.len() as f64 * 0.95).ceil() as usize).min(times.len()) - 1;
            times[idx] as f64
        };

        let violations = tracker
            .violations
            .iter()
            .filter(|v| v.timestamp_ms >= cutoff)
            .count() as u64;

        SlaReport {
            period_ms: window_ms,
            total_workflows,
            violations,
            avg_processing_secs,
            p95_processing_secs,
            availability_pct: tracker.current_availability(),
        }
    }

    /// Whether the SLA was met (no violations).
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.violations == 0
    }
}

/// Get current time in milliseconds (monotonic-safe approximation via std).
fn current_time_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> SlaTracker {
        SlaTracker::new(SlaTarget::new(60, 10, 99.0, 0.8))
    }

    #[test]
    fn test_sla_target_defaults() {
        let target = SlaTarget::default();
        assert_eq!(target.max_processing_secs, 3600);
        assert_eq!(target.max_queue_depth, 100);
        assert!((target.availability_pct - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_record_completion_no_violation() {
        let mut tracker = make_tracker();
        tracker.record_completion("wf-1", 30); // under 60s limit
        assert_eq!(tracker.all_violations().len(), 0);
        assert_eq!(tracker.total_completions(), 1);
    }

    #[test]
    fn test_record_completion_with_violation() {
        let mut tracker = make_tracker();
        tracker.record_completion("wf-1", 120); // over 60s limit
        let violations = tracker.all_violations();
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::ProcessingTimeout
        );
        assert_eq!(violations[0].expected, 60.0);
        assert_eq!(violations[0].actual, 120.0);
    }

    #[test]
    fn test_check_queue_depth_no_violation() {
        let mut tracker = make_tracker();
        let result = tracker.check_queue_depth(5); // under 10 limit
        assert!(result.is_none());
        assert_eq!(tracker.all_violations().len(), 0);
    }

    #[test]
    fn test_check_queue_depth_violation() {
        let mut tracker = make_tracker();
        let result = tracker.check_queue_depth(15); // over 10 limit
        assert!(result.is_some());
        let violation = result.expect("should succeed in test");
        assert_eq!(violation.violation_type, ViolationType::QueueDepthExceeded);
        assert_eq!(violation.expected, 10.0);
        assert_eq!(violation.actual, 15.0);
    }

    #[test]
    fn test_violations_in_window() {
        let mut tracker = make_tracker();
        tracker.record_completion("wf-1", 120);
        tracker.record_completion("wf-2", 200);

        let violations = tracker.violations_in_window(60_000); // last 60 seconds
        assert_eq!(violations.len(), 2);
    }

    #[test]
    fn test_violations_in_empty_window() {
        let tracker = make_tracker();
        let violations = tracker.violations_in_window(0); // window of 0 ms
        assert_eq!(violations.len(), 0);
    }

    #[test]
    fn test_current_availability_no_data() {
        let tracker = make_tracker();
        assert!((tracker.current_availability() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_current_availability_with_downtime() {
        let mut tracker = make_tracker();
        tracker.record_uptime(90_000);
        tracker.record_downtime(10_000);
        let availability = tracker.current_availability();
        assert!((availability - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_processing() {
        let mut tracker = make_tracker();
        for i in 1..=100u64 {
            tracker.record_completion(&format!("wf-{}", i), i);
        }
        let p95 = tracker.percentile_processing_secs(0.95);
        assert!(p95 >= 90.0 && p95 <= 100.0);
    }

    #[test]
    fn test_sla_report_generation() {
        let mut tracker = make_tracker();
        tracker.record_completion("wf-1", 30);
        tracker.record_completion("wf-2", 90); // violation
        tracker.record_uptime(3_600_000);

        let report = SlaReport::generate(&tracker, 60_000);
        assert_eq!(report.total_workflows, 2);
        assert_eq!(report.violations, 1);
        assert!(report.avg_processing_secs > 0.0);
        assert_eq!(report.period_ms, 60_000);
    }

    #[test]
    fn test_sla_report_compliant() {
        let mut tracker = make_tracker();
        tracker.record_completion("wf-1", 30);
        tracker.record_completion("wf-2", 45);

        let report = SlaReport::generate(&tracker, 60_000);
        assert!(report.is_compliant());
    }

    #[test]
    fn test_sla_report_empty() {
        let tracker = make_tracker();
        let report = SlaReport::generate(&tracker, 60_000);
        assert_eq!(report.total_workflows, 0);
        assert_eq!(report.violations, 0);
        assert_eq!(report.avg_processing_secs, 0.0);
        assert!(report.is_compliant());
    }
}
