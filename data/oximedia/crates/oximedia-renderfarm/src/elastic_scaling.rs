// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Elastic cloud worker scaling.
//!
//! Provides policies and decision logic for dynamically scaling the number of
//! render-farm workers in response to queue depth, time-of-day schedules, or
//! hybrid combinations of both.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// WorkerType
// ---------------------------------------------------------------------------

/// The billing / availability class of a cloud worker instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkerType {
    /// Spot / preemptible instance — cheapest, may be interrupted.
    Spot,
    /// Standard on-demand instance — reliable, full price.
    OnDemand,
    /// Long-term reserved capacity — discounted, contractually committed.
    Reserved,
    /// GCP-style preemptible worker (similar to Spot but distinct billing).
    Preemptible,
}

impl WorkerType {
    /// Human-readable name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Spot => "Spot",
            Self::OnDemand => "On-Demand",
            Self::Reserved => "Reserved",
            Self::Preemptible => "Preemptible",
        }
    }

    /// Whether this worker type can be interrupted by the cloud provider.
    #[must_use]
    pub fn is_interruptible(&self) -> bool {
        matches!(self, Self::Spot | Self::Preemptible)
    }
}

// ---------------------------------------------------------------------------
// WorkerSpec
// ---------------------------------------------------------------------------

/// Hardware specification for a single cloud worker instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSpec {
    /// Number of CPU cores.
    pub cpu_cores: u8,
    /// System memory in gigabytes.
    pub memory_gb: u16,
    /// GPU VRAM in gigabytes, or `None` if no GPU.
    pub gpu_vram_gb: Option<u8>,
    /// Billing / availability class.
    pub worker_type: WorkerType,
    /// Approximate cost per hour in USD.
    pub cost_per_hour: f32,
}

impl WorkerSpec {
    /// Create a basic CPU-only worker spec.
    #[must_use]
    pub fn cpu_only(
        cpu_cores: u8,
        memory_gb: u16,
        worker_type: WorkerType,
        cost_per_hour: f32,
    ) -> Self {
        Self {
            cpu_cores,
            memory_gb,
            gpu_vram_gb: None,
            worker_type,
            cost_per_hour,
        }
    }

    /// Create a GPU-accelerated worker spec.
    #[must_use]
    pub fn with_gpu(
        cpu_cores: u8,
        memory_gb: u16,
        gpu_vram_gb: u8,
        worker_type: WorkerType,
        cost_per_hour: f32,
    ) -> Self {
        Self {
            cpu_cores,
            memory_gb,
            gpu_vram_gb: Some(gpu_vram_gb),
            worker_type,
            cost_per_hour,
        }
    }

    /// Whether this spec includes a GPU.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu_vram_gb.is_some()
    }
}

// ---------------------------------------------------------------------------
// TimeSlot
// ---------------------------------------------------------------------------

/// A time window within a week during which a certain worker count is desired.
///
/// `day_mask` uses one bit per weekday: bit 0 = Monday, …, bit 6 = Sunday.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSlot {
    /// Hour of day (0–23) when this slot starts (inclusive).
    pub start_hour: u8,
    /// Hour of day (0–23) when this slot ends (exclusive).
    pub end_hour: u8,
    /// Bitmask of weekdays this slot applies to (bit 0 = Mon, bit 6 = Sun).
    pub day_mask: u8,
    /// Number of workers desired during this slot.
    pub target_workers: u32,
}

impl TimeSlot {
    /// Returns `true` when the given `hour` on `weekday` (0 = Mon) falls inside this slot.
    #[must_use]
    pub fn is_active(&self, weekday: u8, hour: u8) -> bool {
        let day_active = (self.day_mask >> (weekday % 7)) & 1 == 1;
        let hour_active = if self.start_hour <= self.end_hour {
            hour >= self.start_hour && hour < self.end_hour
        } else {
            // Wraps midnight
            hour >= self.start_hour || hour < self.end_hour
        };
        day_active && hour_active
    }

    /// Convenience: create a weekday-business-hours slot (Mon–Fri 09:00–18:00).
    #[must_use]
    pub fn business_hours(target_workers: u32) -> Self {
        // Bits 0..4 = Mon..Fri
        Self {
            start_hour: 9,
            end_hour: 18,
            day_mask: 0b0001_1111,
            target_workers,
        }
    }

    /// Convenience: create an all-week overnight slot (00:00–06:00).
    #[must_use]
    pub fn overnight(target_workers: u32) -> Self {
        Self {
            start_hour: 0,
            end_hour: 6,
            day_mask: 0b0111_1111,
            target_workers,
        }
    }
}

// ---------------------------------------------------------------------------
// ScalingPolicy
// ---------------------------------------------------------------------------

/// Determines how many workers the farm should maintain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingPolicy {
    /// Always keep exactly `n` workers running.
    FixedSize(u32),

    /// Dynamically scale between `min` and `max` based on queue depth.
    Demand {
        /// Minimum number of workers to keep alive.
        min: u32,
        /// Maximum number of workers to provision.
        max: u32,
        /// Queue depth at which the scaler should aim to have `max` workers.
        target_queue_depth: u32,
    },

    /// Follow a weekly time-based schedule.
    Schedule {
        /// List of time slots; first matching slot wins.
        slots: Vec<TimeSlot>,
    },

    /// Combine demand-based and schedule-based policies, using whichever
    /// requests more workers (safe-side approach).
    HybridDemandSchedule {
        /// Demand sub-policy.
        demand: Box<ScalingPolicy>,
        /// Schedule sub-policy.
        schedule: Box<ScalingPolicy>,
    },
}

impl ScalingPolicy {
    /// Compute the desired worker count for the given context.
    ///
    /// `current_time_hour` is the local hour-of-day (0–23).
    /// `current_weekday` is 0-based weekday (0 = Monday).
    /// `current_queue` is the current job queue depth.
    #[must_use]
    pub fn desired_workers(
        &self,
        current_time_hour: u8,
        current_weekday: u8,
        current_queue: u32,
    ) -> u32 {
        match self {
            Self::FixedSize(n) => *n,

            Self::Demand {
                min,
                max,
                target_queue_depth,
            } => {
                if *target_queue_depth == 0 {
                    return *max;
                }
                let ratio = current_queue as f32 / *target_queue_depth as f32;
                let desired = (*min as f32 + ratio * (*max - *min) as f32).round() as u32;
                desired.clamp(*min, *max)
            }

            Self::Schedule { slots } => slots
                .iter()
                .find(|s| s.is_active(current_weekday, current_time_hour))
                .map(|s| s.target_workers)
                .unwrap_or(0),

            Self::HybridDemandSchedule { demand, schedule } => {
                let d = demand.desired_workers(current_time_hour, current_weekday, current_queue);
                let s = schedule.desired_workers(current_time_hour, current_weekday, current_queue);
                d.max(s)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScalingDecision
// ---------------------------------------------------------------------------

/// The result of a scaling evaluation, describing what should happen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    /// Total number of workers desired after this action.
    pub desired_workers: u32,
    /// Positive = scale up, negative = scale down, 0 = no change.
    pub scale_delta: i32,
    /// Human-readable description of why this decision was made.
    pub reason: String,
    /// Specs of workers to launch (only populated when `scale_delta > 0`).
    pub new_workers: Vec<WorkerSpec>,
}

// ---------------------------------------------------------------------------
// CooldownTracker
// ---------------------------------------------------------------------------

/// Prevents thrashing by enforcing minimum intervals between scale events.
#[derive(Debug, Clone)]
pub struct CooldownTracker {
    /// Unix-ms timestamp of the last scale-up event (-1 = never).
    pub last_scale_up_ms: i64,
    /// Unix-ms timestamp of the last scale-down event (-1 = never).
    pub last_scale_down_ms: i64,
    /// Minimum milliseconds between two consecutive scale-up events.
    pub scale_up_cooldown_ms: u64,
    /// Minimum milliseconds between two consecutive scale-down events.
    pub scale_down_cooldown_ms: u64,
}

impl CooldownTracker {
    /// Create a tracker with the given cooldown periods (milliseconds).
    #[must_use]
    pub fn new(scale_up_cooldown_ms: u64, scale_down_cooldown_ms: u64) -> Self {
        Self {
            last_scale_up_ms: -1,
            last_scale_down_ms: -1,
            scale_up_cooldown_ms,
            scale_down_cooldown_ms,
        }
    }

    /// Create a tracker with sensible defaults (2 min up / 5 min down).
    #[must_use]
    pub fn default_cooldowns() -> Self {
        Self::new(120_000, 300_000)
    }

    /// Returns `true` if enough time has elapsed since the last scale-up.
    #[must_use]
    pub fn can_scale_up(&self, now_ms: i64) -> bool {
        if self.last_scale_up_ms < 0 {
            return true;
        }
        let elapsed = (now_ms - self.last_scale_up_ms).max(0) as u64;
        elapsed >= self.scale_up_cooldown_ms
    }

    /// Returns `true` if enough time has elapsed since the last scale-down.
    #[must_use]
    pub fn can_scale_down(&self, now_ms: i64) -> bool {
        if self.last_scale_down_ms < 0 {
            return true;
        }
        let elapsed = (now_ms - self.last_scale_down_ms).max(0) as u64;
        elapsed >= self.scale_down_cooldown_ms
    }

    /// Record that a scale-up just occurred at `now_ms`.
    pub fn record_scale_up(&mut self, now_ms: i64) {
        self.last_scale_up_ms = now_ms;
    }

    /// Record that a scale-down just occurred at `now_ms`.
    pub fn record_scale_down(&mut self, now_ms: i64) {
        self.last_scale_down_ms = now_ms;
    }
}

impl Default for CooldownTracker {
    fn default() -> Self {
        Self::default_cooldowns()
    }
}

// ---------------------------------------------------------------------------
// ElasticScaler
// ---------------------------------------------------------------------------

/// Stateful elastic scaler that wraps a `ScalingPolicy` and tracks current
/// worker count plus live queue depth.
pub struct ElasticScaler {
    /// Current live worker count.
    pub current_workers: u32,
    /// Atomically-updated queue depth for concurrent use.
    pub queue_depth: AtomicU32,
    /// The active scaling policy.
    pub policy: ScalingPolicy,
    /// Default spec used when scale-up decisions need to spawn new workers.
    default_spec: WorkerSpec,
}

impl ElasticScaler {
    /// Create a new scaler with the given policy and a default worker spec.
    #[must_use]
    pub fn new(policy: ScalingPolicy, default_spec: WorkerSpec) -> Self {
        Self {
            current_workers: 0,
            queue_depth: AtomicU32::new(0),
            policy,
            default_spec,
        }
    }

    /// Update the queue depth atomically (can be called from any thread).
    pub fn update_queue_depth(&self, depth: u32) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Compute what the scaler recommends given the current wall-clock context.
    ///
    /// `current_time_hour` — hour of day (0–23).
    /// `current_weekday`   — 0-based weekday (0 = Monday).
    /// `current_queue`     — current job queue depth.
    #[must_use]
    pub fn compute_scaling_decision(
        &self,
        current_time_hour: u8,
        current_queue: u32,
    ) -> ScalingDecision {
        // Use 0 as weekday default when caller doesn't supply it (backward compat shim).
        self.compute_scaling_decision_with_day(current_time_hour, 0, current_queue)
    }

    /// Full version accepting weekday.
    #[must_use]
    pub fn compute_scaling_decision_with_day(
        &self,
        current_time_hour: u8,
        current_weekday: u8,
        current_queue: u32,
    ) -> ScalingDecision {
        let desired =
            self.policy
                .desired_workers(current_time_hour, current_weekday, current_queue);
        let delta = desired as i32 - self.current_workers as i32;

        let reason = if delta > 0 {
            format!(
                "Scale up: queue={current_queue}, desired={desired}, current={}",
                self.current_workers
            )
        } else if delta < 0 {
            format!(
                "Scale down: queue={current_queue}, desired={desired}, current={}",
                self.current_workers
            )
        } else {
            format!(
                "No change: current={} matches desired={desired}",
                self.current_workers
            )
        };

        let new_workers: Vec<WorkerSpec> = if delta > 0 {
            (0..delta as u32)
                .map(|_| self.default_spec.clone())
                .collect()
        } else {
            Vec::new()
        };

        ScalingDecision {
            desired_workers: desired,
            scale_delta: delta,
            reason,
            new_workers,
        }
    }

    /// Apply a scaling decision (updates `current_workers`).
    pub fn apply_decision(&mut self, decision: &ScalingDecision) {
        self.current_workers = decision.desired_workers;
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Estimate the percentage cost savings of using a spot/preemptible instance
/// compared to the on-demand hourly rate.
///
/// Returns `0.0` when `on_demand_hourly` is non-positive to avoid division by
/// zero.
#[must_use]
pub fn spot_savings_estimate(spec: &WorkerSpec, on_demand_hourly: f32) -> f32 {
    if on_demand_hourly <= 0.0 {
        return 0.0;
    }
    (on_demand_hourly - spec.cost_per_hour) * 100.0 / on_demand_hourly
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(worker_type: WorkerType, cost: f32) -> WorkerSpec {
        WorkerSpec::cpu_only(8, 32, worker_type, cost)
    }

    // --- WorkerType ---

    #[test]
    fn test_worker_type_display() {
        assert_eq!(WorkerType::Spot.display_name(), "Spot");
        assert_eq!(WorkerType::OnDemand.display_name(), "On-Demand");
        assert_eq!(WorkerType::Reserved.display_name(), "Reserved");
        assert_eq!(WorkerType::Preemptible.display_name(), "Preemptible");
    }

    #[test]
    fn test_worker_type_interruptible() {
        assert!(WorkerType::Spot.is_interruptible());
        assert!(WorkerType::Preemptible.is_interruptible());
        assert!(!WorkerType::OnDemand.is_interruptible());
        assert!(!WorkerType::Reserved.is_interruptible());
    }

    // --- WorkerSpec ---

    #[test]
    fn test_worker_spec_cpu_only() {
        let spec = WorkerSpec::cpu_only(16, 64, WorkerType::OnDemand, 0.5);
        assert_eq!(spec.cpu_cores, 16);
        assert!(!spec.has_gpu());
    }

    #[test]
    fn test_worker_spec_with_gpu() {
        let spec = WorkerSpec::with_gpu(8, 32, 16, WorkerType::Spot, 1.2);
        assert!(spec.has_gpu());
        assert_eq!(spec.gpu_vram_gb, Some(16));
    }

    // --- TimeSlot ---

    #[test]
    fn test_time_slot_business_hours_active() {
        let slot = TimeSlot::business_hours(10);
        // Monday (0), 10:00
        assert!(slot.is_active(0, 10));
    }

    #[test]
    fn test_time_slot_business_hours_inactive_weekend() {
        let slot = TimeSlot::business_hours(10);
        // Saturday (5), 10:00
        assert!(!slot.is_active(5, 10));
    }

    #[test]
    fn test_time_slot_business_hours_inactive_hour() {
        let slot = TimeSlot::business_hours(10);
        // Monday (0), 08:00 — before start
        assert!(!slot.is_active(0, 8));
    }

    #[test]
    fn test_time_slot_overnight() {
        let slot = TimeSlot::overnight(2);
        // Any day, 03:00
        assert!(slot.is_active(3, 3));
        // Any day, 08:00 — outside
        assert!(!slot.is_active(3, 8));
    }

    // --- ScalingPolicy::FixedSize ---

    #[test]
    fn test_scaling_policy_fixed() {
        let policy = ScalingPolicy::FixedSize(5);
        assert_eq!(policy.desired_workers(12, 0, 0), 5);
        assert_eq!(policy.desired_workers(12, 0, 9999), 5);
    }

    // --- ScalingPolicy::Demand ---

    #[test]
    fn test_scaling_policy_demand_min() {
        let policy = ScalingPolicy::Demand {
            min: 2,
            max: 20,
            target_queue_depth: 100,
        };
        // Empty queue → min workers
        assert_eq!(policy.desired_workers(12, 0, 0), 2);
    }

    #[test]
    fn test_scaling_policy_demand_max() {
        let policy = ScalingPolicy::Demand {
            min: 2,
            max: 20,
            target_queue_depth: 100,
        };
        // Queue >= target → max workers
        assert_eq!(policy.desired_workers(12, 0, 100), 20);
    }

    #[test]
    fn test_scaling_policy_demand_clamp() {
        let policy = ScalingPolicy::Demand {
            min: 2,
            max: 20,
            target_queue_depth: 100,
        };
        let d = policy.desired_workers(12, 0, 200);
        assert_eq!(d, 20);
    }

    #[test]
    fn test_scaling_policy_demand_zero_target() {
        let policy = ScalingPolicy::Demand {
            min: 1,
            max: 10,
            target_queue_depth: 0,
        };
        // Guard: should not divide by zero
        assert_eq!(policy.desired_workers(12, 0, 50), 10);
    }

    // --- ScalingPolicy::Schedule ---

    #[test]
    fn test_scaling_policy_schedule_match() {
        let slots = vec![TimeSlot::business_hours(8)];
        let policy = ScalingPolicy::Schedule { slots };
        // Monday 10:00
        assert_eq!(policy.desired_workers(10, 0, 0), 8);
    }

    #[test]
    fn test_scaling_policy_schedule_no_match() {
        let slots = vec![TimeSlot::business_hours(8)];
        let policy = ScalingPolicy::Schedule { slots };
        // Monday 23:00 — outside all slots → 0
        assert_eq!(policy.desired_workers(23, 0, 0), 0);
    }

    // --- ScalingPolicy::HybridDemandSchedule ---

    #[test]
    fn test_scaling_policy_hybrid_takes_max() {
        let demand = Box::new(ScalingPolicy::FixedSize(3));
        let schedule = Box::new(ScalingPolicy::FixedSize(7));
        let policy = ScalingPolicy::HybridDemandSchedule { demand, schedule };
        assert_eq!(policy.desired_workers(10, 0, 0), 7);
    }

    // --- ScalingDecision ---

    #[test]
    fn test_elastic_scaler_scale_up() {
        let policy = ScalingPolicy::FixedSize(5);
        let spec = make_spec(WorkerType::Spot, 0.25);
        let mut scaler = ElasticScaler::new(policy, spec);
        scaler.current_workers = 2;

        let decision = scaler.compute_scaling_decision(12, 0);
        assert_eq!(decision.desired_workers, 5);
        assert_eq!(decision.scale_delta, 3);
        assert_eq!(decision.new_workers.len(), 3);
    }

    #[test]
    fn test_elastic_scaler_scale_down() {
        let policy = ScalingPolicy::FixedSize(2);
        let spec = make_spec(WorkerType::OnDemand, 0.5);
        let mut scaler = ElasticScaler::new(policy, spec);
        scaler.current_workers = 8;

        let decision = scaler.compute_scaling_decision(12, 0);
        assert_eq!(decision.scale_delta, -6);
        assert!(decision.new_workers.is_empty());
    }

    #[test]
    fn test_elastic_scaler_no_change() {
        let policy = ScalingPolicy::FixedSize(4);
        let spec = make_spec(WorkerType::Reserved, 0.1);
        let mut scaler = ElasticScaler::new(policy, spec);
        scaler.current_workers = 4;

        let decision = scaler.compute_scaling_decision(12, 0);
        assert_eq!(decision.scale_delta, 0);
    }

    #[test]
    fn test_elastic_scaler_apply_decision() {
        let policy = ScalingPolicy::FixedSize(10);
        let spec = make_spec(WorkerType::Spot, 0.2);
        let mut scaler = ElasticScaler::new(policy, spec);

        let decision = scaler.compute_scaling_decision(12, 0);
        scaler.apply_decision(&decision);
        assert_eq!(scaler.current_workers, 10);
    }

    // --- CooldownTracker ---

    #[test]
    fn test_cooldown_tracker_initial_state() {
        let tracker = CooldownTracker::new(60_000, 120_000);
        assert!(tracker.can_scale_up(0));
        assert!(tracker.can_scale_down(0));
    }

    #[test]
    fn test_cooldown_tracker_scale_up_blocked() {
        let mut tracker = CooldownTracker::new(60_000, 120_000);
        tracker.record_scale_up(1_000);
        assert!(!tracker.can_scale_up(30_000));
        assert!(tracker.can_scale_up(70_000));
    }

    #[test]
    fn test_cooldown_tracker_scale_down_blocked() {
        let mut tracker = CooldownTracker::new(60_000, 120_000);
        tracker.record_scale_down(1_000);
        assert!(!tracker.can_scale_down(60_000));
        assert!(tracker.can_scale_down(130_000));
    }

    #[test]
    fn test_cooldown_tracker_independent() {
        let mut tracker = CooldownTracker::new(60_000, 120_000);
        tracker.record_scale_up(1_000);
        // scale-down cooldown not triggered
        assert!(tracker.can_scale_down(10_000));
    }

    // --- spot_savings_estimate ---

    #[test]
    fn test_spot_savings_estimate_basic() {
        let spec = make_spec(WorkerType::Spot, 0.30);
        let savings = spot_savings_estimate(&spec, 1.00);
        assert!((savings - 70.0).abs() < 0.001);
    }

    #[test]
    fn test_spot_savings_estimate_zero_on_demand() {
        let spec = make_spec(WorkerType::Spot, 0.10);
        assert_eq!(spot_savings_estimate(&spec, 0.0), 0.0);
    }

    #[test]
    fn test_spot_savings_estimate_negative_on_demand() {
        let spec = make_spec(WorkerType::Spot, 0.10);
        assert_eq!(spot_savings_estimate(&spec, -1.0), 0.0);
    }

    #[test]
    fn test_spot_savings_full_saving() {
        let spec = make_spec(WorkerType::Spot, 0.0);
        let savings = spot_savings_estimate(&spec, 2.0);
        assert!((savings - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_update_queue_depth_atomic() {
        let policy = ScalingPolicy::FixedSize(1);
        let spec = make_spec(WorkerType::Spot, 0.1);
        let scaler = ElasticScaler::new(policy, spec);
        scaler.update_queue_depth(42);
        assert_eq!(scaler.queue_depth.load(Ordering::Relaxed), 42);
    }
}
