#![allow(dead_code)]
//! Farm capacity planning and resource forecasting.
//!
//! Provides tools for predicting farm workload, estimating required
//! capacity, and making scaling decisions for the encoding farm:
//! - Workload tracking and historical analysis
//! - Resource utilization estimation
//! - Scaling recommendations (scale up/down/none)
//! - Worker slot calculation for target SLA
//! - Queue depth monitoring and alerting thresholds

use std::collections::VecDeque;
use std::time::Duration;

/// A single workload sample at a point in time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkloadSample {
    /// Timestamp of the sample (epoch seconds).
    pub timestamp_secs: u64,
    /// Number of active jobs at sample time.
    pub active_jobs: u32,
    /// Number of queued (waiting) jobs at sample time.
    pub queued_jobs: u32,
    /// Number of available worker slots.
    pub available_slots: u32,
    /// Number of busy worker slots.
    pub busy_slots: u32,
    /// Average job duration in seconds at sample time.
    pub avg_job_duration_secs: f64,
}

/// Scaling recommendation from the capacity planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScalingAction {
    /// No change needed.
    None,
    /// Scale up by the specified number of worker slots.
    ScaleUp(u32),
    /// Scale down by the specified number of worker slots.
    ScaleDown(u32),
}

impl std::fmt::Display for ScalingAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "NoChange"),
            Self::ScaleUp(n) => write!(f, "ScaleUp({n})"),
            Self::ScaleDown(n) => write!(f, "ScaleDown({n})"),
        }
    }
}

/// Configuration for the capacity planner.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapacityPlannerConfig {
    /// Maximum number of historical samples to retain.
    pub max_history: usize,
    /// Target utilization ratio (0.0 to 1.0). Above this triggers scale-up.
    pub target_utilization: f64,
    /// Low utilization threshold. Below this triggers scale-down.
    pub low_utilization: f64,
    /// Maximum queue depth before triggering urgent scale-up.
    pub max_queue_depth: u32,
    /// Minimum number of worker slots to maintain.
    pub min_slots: u32,
    /// Maximum number of worker slots allowed.
    pub max_slots: u32,
    /// Cooldown period between scaling actions.
    pub cooldown: Duration,
}

impl Default for CapacityPlannerConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            target_utilization: 0.80,
            low_utilization: 0.30,
            max_queue_depth: 50,
            min_slots: 2,
            max_slots: 200,
            cooldown: Duration::from_secs(300),
        }
    }
}

/// The capacity planner tracks workload history and recommends scaling actions.
#[derive(Debug)]
pub struct CapacityPlanner {
    /// Planner configuration.
    config: CapacityPlannerConfig,
    /// Historical workload samples.
    history: VecDeque<WorkloadSample>,
    /// Timestamp of the last scaling action (epoch seconds).
    last_scaling_action_secs: u64,
}

impl CapacityPlanner {
    /// Create a new capacity planner with the given configuration.
    #[must_use]
    pub fn new(config: CapacityPlannerConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            last_scaling_action_secs: 0,
        }
    }

    /// Create a capacity planner with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CapacityPlannerConfig::default())
    }

    /// Record a new workload sample.
    pub fn record_sample(&mut self, sample: WorkloadSample) {
        self.history.push_back(sample);
        while self.history.len() > self.config.max_history {
            self.history.pop_front();
        }
    }

    /// Get the number of recorded samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Calculate the current utilization ratio.
    ///
    /// Returns `busy_slots / total_slots` from the most recent sample, or 0.0
    /// if there are no samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn current_utilization(&self) -> f64 {
        self.history.back().map_or(0.0, |s| {
            let total = s.available_slots + s.busy_slots;
            if total == 0 {
                0.0
            } else {
                f64::from(s.busy_slots) / f64::from(total)
            }
        })
    }

    /// Calculate the average utilization over the last `n` samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_utilization(&self, n: usize) -> f64 {
        let samples: Vec<_> = self.history.iter().rev().take(n).collect();
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples
            .iter()
            .map(|s| {
                let total = s.available_slots + s.busy_slots;
                if total == 0 {
                    0.0
                } else {
                    f64::from(s.busy_slots) / f64::from(total)
                }
            })
            .sum();
        sum / samples.len() as f64
    }

    /// Calculate the average queue depth over the last `n` samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_queue_depth(&self, n: usize) -> f64 {
        let samples: Vec<_> = self.history.iter().rev().take(n).collect();
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|s| f64::from(s.queued_jobs)).sum();
        sum / samples.len() as f64
    }

    /// Estimate the number of worker slots needed to clear the current queue
    /// within `target_duration`.
    ///
    /// Uses the average job duration from the most recent sample.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn estimate_slots_for_queue_clear(&self, target_duration: Duration) -> u32 {
        let Some(latest) = self.history.back() else {
            return 0;
        };
        if latest.queued_jobs == 0 {
            return 0;
        }
        let avg_dur = latest.avg_job_duration_secs;
        if avg_dur <= 0.0 {
            return latest.queued_jobs;
        }
        let target_secs = target_duration.as_secs_f64();
        if target_secs <= 0.0 {
            return latest.queued_jobs;
        }
        let jobs_per_slot = target_secs / avg_dur;
        if jobs_per_slot <= 0.0 {
            return latest.queued_jobs;
        }

        (f64::from(latest.queued_jobs) / jobs_per_slot).ceil() as u32
    }

    /// Recommend a scaling action based on the current state.
    ///
    /// The planner considers:
    /// 1. Current utilization vs target/low thresholds
    /// 2. Queue depth vs maximum allowed
    /// 3. Cooldown period since last action
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn recommend_scaling(&mut self, current_time_secs: u64) -> ScalingAction {
        let Some(latest) = self.history.back() else {
            return ScalingAction::None;
        };

        // Enforce cooldown
        let elapsed = current_time_secs.saturating_sub(self.last_scaling_action_secs);
        if elapsed < self.config.cooldown.as_secs() && self.last_scaling_action_secs > 0 {
            return ScalingAction::None;
        }

        let total_slots = latest.available_slots + latest.busy_slots;
        let utilization = self.current_utilization();

        // Urgent: queue depth exceeded
        if latest.queued_jobs > self.config.max_queue_depth {
            let extra = self
                .estimate_slots_for_queue_clear(Duration::from_secs(600))
                .max(1);
            let capped = extra.min(self.config.max_slots.saturating_sub(total_slots));
            if capped > 0 {
                self.last_scaling_action_secs = current_time_secs;
                return ScalingAction::ScaleUp(capped);
            }
        }

        // High utilization: scale up
        if utilization > self.config.target_utilization {
            let desired = (f64::from(total_slots) * 1.25).ceil() as u32;
            let add = desired.saturating_sub(total_slots).max(1);
            let capped = add.min(self.config.max_slots.saturating_sub(total_slots));
            if capped > 0 {
                self.last_scaling_action_secs = current_time_secs;
                return ScalingAction::ScaleUp(capped);
            }
        }

        // Low utilization: scale down
        if utilization < self.config.low_utilization && total_slots > self.config.min_slots {
            let target = (f64::from(total_slots) * 0.75).floor() as u32;
            let target = target.max(self.config.min_slots);
            let remove = total_slots.saturating_sub(target);
            if remove > 0 {
                self.last_scaling_action_secs = current_time_secs;
                return ScalingAction::ScaleDown(remove);
            }
        }

        ScalingAction::None
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &CapacityPlannerConfig {
        &self.config
    }

    /// Get the full history of workload samples.
    #[must_use]
    pub fn history(&self) -> &VecDeque<WorkloadSample> {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(
        ts: u64,
        active: u32,
        queued: u32,
        avail: u32,
        busy: u32,
        avg_dur: f64,
    ) -> WorkloadSample {
        WorkloadSample {
            timestamp_secs: ts,
            active_jobs: active,
            queued_jobs: queued,
            available_slots: avail,
            busy_slots: busy,
            avg_job_duration_secs: avg_dur,
        }
    }

    #[test]
    fn test_new_planner_empty() {
        let planner = CapacityPlanner::with_defaults();
        assert_eq!(planner.sample_count(), 0);
        assert!((planner.current_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_sample() {
        let mut planner = CapacityPlanner::with_defaults();
        planner.record_sample(make_sample(100, 5, 2, 3, 7, 60.0));
        assert_eq!(planner.sample_count(), 1);
    }

    #[test]
    fn test_max_history_cap() {
        let config = CapacityPlannerConfig {
            max_history: 3,
            ..Default::default()
        };
        let mut planner = CapacityPlanner::new(config);
        for i in 0..10 {
            planner.record_sample(make_sample(i, 1, 0, 5, 5, 30.0));
        }
        assert_eq!(planner.sample_count(), 3);
    }

    #[test]
    fn test_current_utilization() {
        let mut planner = CapacityPlanner::with_defaults();
        planner.record_sample(make_sample(100, 5, 0, 2, 8, 60.0));
        let util = planner.current_utilization();
        assert!((util - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_current_utilization_zero_slots() {
        let mut planner = CapacityPlanner::with_defaults();
        planner.record_sample(make_sample(100, 0, 0, 0, 0, 0.0));
        assert!((planner.current_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_average_utilization() {
        let mut planner = CapacityPlanner::with_defaults();
        planner.record_sample(make_sample(100, 5, 0, 5, 5, 60.0)); // 50%
        planner.record_sample(make_sample(200, 8, 0, 2, 8, 60.0)); // 80%
        let avg = planner.average_utilization(2);
        assert!((avg - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_average_queue_depth() {
        let mut planner = CapacityPlanner::with_defaults();
        planner.record_sample(make_sample(100, 5, 10, 5, 5, 60.0));
        planner.record_sample(make_sample(200, 5, 20, 5, 5, 60.0));
        let avg = planner.average_queue_depth(2);
        assert!((avg - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_slots_for_queue_clear() {
        let mut planner = CapacityPlanner::with_defaults();
        // 10 queued jobs, avg 60s each, target clear in 300s
        planner.record_sample(make_sample(100, 5, 10, 5, 5, 60.0));
        let slots = planner.estimate_slots_for_queue_clear(Duration::from_secs(300));
        // 300/60 = 5 jobs per slot, 10/5 = 2 slots
        assert_eq!(slots, 2);
    }

    #[test]
    fn test_estimate_slots_empty_queue() {
        let mut planner = CapacityPlanner::with_defaults();
        planner.record_sample(make_sample(100, 5, 0, 5, 5, 60.0));
        assert_eq!(
            planner.estimate_slots_for_queue_clear(Duration::from_secs(300)),
            0
        );
    }

    #[test]
    fn test_recommend_no_action_normal() {
        let mut planner = CapacityPlanner::with_defaults();
        // 50% util, no queue — no action
        planner.record_sample(make_sample(100, 5, 0, 5, 5, 60.0));
        let action = planner.recommend_scaling(1000);
        assert_eq!(action, ScalingAction::None);
    }

    #[test]
    fn test_recommend_scale_up_high_util() {
        let config = CapacityPlannerConfig {
            target_utilization: 0.70,
            cooldown: Duration::from_secs(0),
            ..Default::default()
        };
        let mut planner = CapacityPlanner::new(config);
        // 90% utilization
        planner.record_sample(make_sample(100, 9, 5, 1, 9, 60.0));
        let action = planner.recommend_scaling(200);
        match action {
            ScalingAction::ScaleUp(n) => assert!(n >= 1),
            other => panic!("Expected ScaleUp, got {other:?}"),
        }
    }

    #[test]
    fn test_recommend_scale_down_low_util() {
        let config = CapacityPlannerConfig {
            low_utilization: 0.30,
            min_slots: 2,
            cooldown: Duration::from_secs(0),
            ..Default::default()
        };
        let mut planner = CapacityPlanner::new(config);
        // 10% utilization with 10 total slots
        planner.record_sample(make_sample(100, 1, 0, 9, 1, 60.0));
        let action = planner.recommend_scaling(200);
        match action {
            ScalingAction::ScaleDown(n) => assert!(n >= 1),
            other => panic!("Expected ScaleDown, got {other:?}"),
        }
    }

    #[test]
    fn test_scaling_action_display() {
        assert_eq!(ScalingAction::None.to_string(), "NoChange");
        assert_eq!(ScalingAction::ScaleUp(5).to_string(), "ScaleUp(5)");
        assert_eq!(ScalingAction::ScaleDown(3).to_string(), "ScaleDown(3)");
    }
}
