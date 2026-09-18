//! Cloud auto-scaling for media workloads
//!
//! Provides a policy-based auto-scaler that evaluates real-time metrics and
//! decides whether to scale the instance fleet up or down:
//! - CPU utilisation, queue depth, transcode backlog, and connection metrics
//! - Configurable scale-up / scale-down thresholds with min/max bounds
//! - Immutable event log for audit and analysis

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The metric observed by a scaling policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalingMetric {
    /// Percentage CPU utilisation across the fleet (0–100)
    CpuUtilization,
    /// Number of jobs waiting in a processing queue
    QueueDepth,
    /// Number of transcode tasks that have not yet started
    TranscodeBacklog,
    /// Number of active inbound/outbound connections
    ActiveConnections,
}

impl ScalingMetric {
    /// Returns a human-readable description of this metric.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            ScalingMetric::CpuUtilization => "CPU utilisation percentage (0–100)",
            ScalingMetric::QueueDepth => "Number of jobs waiting in the processing queue",
            ScalingMetric::TranscodeBacklog => "Number of pending transcode tasks",
            ScalingMetric::ActiveConnections => "Number of active inbound/outbound connections",
        }
    }
}

/// A policy that governs how the auto-scaler responds to metric changes.
#[derive(Debug, Clone)]
pub struct ScalingPolicy {
    /// The metric this policy tracks
    pub metric: ScalingMetric,
    /// If the metric exceeds this value, add instances
    pub scale_up_threshold: f32,
    /// If the metric falls below this value, remove instances
    pub scale_down_threshold: f32,
    /// Minimum number of instances to keep running
    pub min_instances: u32,
    /// Maximum number of instances allowed
    pub max_instances: u32,
}

impl ScalingPolicy {
    /// Returns a sensible default policy tuned for transcode workloads.
    #[must_use]
    pub fn default_transcode() -> Self {
        Self {
            metric: ScalingMetric::TranscodeBacklog,
            scale_up_threshold: 10.0,
            scale_down_threshold: 2.0,
            min_instances: 1,
            max_instances: 20,
        }
    }

    /// Evaluate whether the fleet size should change.
    ///
    /// Returns `Some(new_count)` if a scaling action is warranted, or `None`
    /// if the current instance count is already appropriate.
    ///
    /// Scale-up adds one instance; scale-down removes one, both clamped to
    /// `[min_instances, max_instances]`.
    #[must_use]
    pub fn evaluate(&self, current_value: f32, current_instances: u32) -> Option<u32> {
        if current_value > self.scale_up_threshold {
            let target = (current_instances + 1).min(self.max_instances);
            if target != current_instances {
                return Some(target);
            }
        } else if current_value < self.scale_down_threshold {
            let target = current_instances.saturating_sub(1).max(self.min_instances);
            if target != current_instances {
                return Some(target);
            }
        }
        None
    }
}

/// A record of a single scaling event.
#[derive(Debug, Clone)]
pub struct ScalingEvent {
    /// Unix epoch timestamp (seconds) when the event occurred
    pub timestamp_epoch: u64,
    /// Instance count before scaling
    pub from_count: u32,
    /// Instance count after scaling
    pub to_count: u32,
    /// Human-readable reason for the scaling action
    pub reason: String,
}

impl ScalingEvent {
    /// Returns `true` if this event increased the instance count.
    #[must_use]
    pub fn is_scale_up(&self) -> bool {
        self.to_count > self.from_count
    }

    /// Returns the signed change in instance count (positive = scale up).
    #[must_use]
    pub fn instance_delta(&self) -> i32 {
        self.to_count as i32 - self.from_count as i32
    }
}

/// An auto-scaler that applies a `ScalingPolicy` and records its decisions.
#[derive(Debug)]
pub struct AutoScaler {
    /// The active scaling policy
    pub policy: ScalingPolicy,
    /// Current number of running instances
    pub current_instances: u32,
    /// Chronological log of all scaling events
    pub events: Vec<ScalingEvent>,
}

impl AutoScaler {
    /// Create a new auto-scaler with the given policy and initial instance count.
    #[must_use]
    pub fn new(policy: ScalingPolicy, initial_instances: u32) -> Self {
        Self {
            policy,
            current_instances: initial_instances,
            events: Vec::new(),
        }
    }

    /// Feed a new metric value to the scaler.
    ///
    /// If the policy determines that a scaling action is needed, the instance
    /// count is updated and a `ScalingEvent` is appended.
    ///
    /// Returns `true` if a scaling action was taken.
    pub fn update(&mut self, metric_value: f32, epoch: u64) -> bool {
        if let Some(target) = self.policy.evaluate(metric_value, self.current_instances) {
            let reason = if target > self.current_instances {
                format!(
                    "Scale up: metric {metric_value:.1} exceeded threshold {:.1}",
                    self.policy.scale_up_threshold
                )
            } else {
                format!(
                    "Scale down: metric {metric_value:.1} below threshold {:.1}",
                    self.policy.scale_down_threshold
                )
            };
            self.events.push(ScalingEvent {
                timestamp_epoch: epoch,
                from_count: self.current_instances,
                to_count: target,
                reason,
            });
            self.current_instances = target;
            true
        } else {
            false
        }
    }

    /// Returns the total number of scaling events recorded.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy() -> ScalingPolicy {
        ScalingPolicy {
            metric: ScalingMetric::CpuUtilization,
            scale_up_threshold: 80.0,
            scale_down_threshold: 20.0,
            min_instances: 2,
            max_instances: 10,
        }
    }

    #[test]
    fn test_scaling_metric_description_non_empty() {
        assert!(!ScalingMetric::CpuUtilization.description().is_empty());
        assert!(!ScalingMetric::QueueDepth.description().is_empty());
        assert!(!ScalingMetric::TranscodeBacklog.description().is_empty());
        assert!(!ScalingMetric::ActiveConnections.description().is_empty());
    }

    #[test]
    fn test_default_transcode_policy() {
        let p = ScalingPolicy::default_transcode();
        assert_eq!(p.metric, ScalingMetric::TranscodeBacklog);
        assert_eq!(p.min_instances, 1);
        assert_eq!(p.max_instances, 20);
    }

    #[test]
    fn test_evaluate_scale_up() {
        let p = make_policy();
        let result = p.evaluate(90.0, 5);
        assert_eq!(result, Some(6));
    }

    #[test]
    fn test_evaluate_scale_down() {
        let p = make_policy();
        let result = p.evaluate(10.0, 5);
        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_evaluate_no_action() {
        let p = make_policy();
        let result = p.evaluate(50.0, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_evaluate_clamped_to_max() {
        let p = make_policy();
        // Already at max
        let result = p.evaluate(90.0, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_evaluate_clamped_to_min() {
        let p = make_policy();
        // Already at min
        let result = p.evaluate(5.0, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_scaling_event_is_scale_up() {
        let event = ScalingEvent {
            timestamp_epoch: 1000,
            from_count: 3,
            to_count: 4,
            reason: String::new(),
        };
        assert!(event.is_scale_up());
        assert_eq!(event.instance_delta(), 1);
    }

    #[test]
    fn test_scaling_event_is_scale_down() {
        let event = ScalingEvent {
            timestamp_epoch: 1000,
            from_count: 5,
            to_count: 4,
            reason: String::new(),
        };
        assert!(!event.is_scale_up());
        assert_eq!(event.instance_delta(), -1);
    }

    #[test]
    fn test_autoscaler_new_no_events() {
        let scaler = AutoScaler::new(make_policy(), 3);
        assert_eq!(scaler.current_instances, 3);
        assert_eq!(scaler.event_count(), 0);
    }

    #[test]
    fn test_autoscaler_update_scale_up() {
        let mut scaler = AutoScaler::new(make_policy(), 3);
        let scaled = scaler.update(90.0, 1000);
        assert!(scaled);
        assert_eq!(scaler.current_instances, 4);
        assert_eq!(scaler.event_count(), 1);
    }

    #[test]
    fn test_autoscaler_update_no_action() {
        let mut scaler = AutoScaler::new(make_policy(), 5);
        let scaled = scaler.update(50.0, 1000);
        assert!(!scaled);
        assert_eq!(scaler.event_count(), 0);
    }

    #[test]
    fn test_autoscaler_update_scale_down() {
        let mut scaler = AutoScaler::new(make_policy(), 6);
        let scaled = scaler.update(10.0, 2000);
        assert!(scaled);
        assert_eq!(scaler.current_instances, 5);
        assert!(scaler.events[0].instance_delta() < 0);
    }

    #[test]
    fn test_autoscaler_multiple_events() {
        let mut scaler = AutoScaler::new(make_policy(), 3);
        scaler.update(90.0, 1);
        scaler.update(90.0, 2);
        scaler.update(90.0, 3);
        assert_eq!(scaler.event_count(), 3);
    }
}
