#![allow(dead_code)]
//! Render quota management for fair resource allocation across projects and users.
//!
//! This module tracks per-user and per-project rendering quotas, enforces limits
//! on concurrent jobs, total render hours, and storage consumption.

use std::collections::HashMap;

/// Unique identifier for a quota holder (user or project).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuotaHolderId(String);

impl QuotaHolderId {
    /// Creates a new quota holder identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the underlying identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A set of resource limits for a quota holder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuotaLimits {
    /// Maximum number of concurrent render jobs.
    pub max_concurrent_jobs: u32,
    /// Maximum total render-hours per billing period.
    pub max_render_hours: f64,
    /// Maximum storage in bytes for output files.
    pub max_storage_bytes: u64,
    /// Maximum number of frames per single job submission.
    pub max_frames_per_job: u32,
}

impl Default for QuotaLimits {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 10,
            max_render_hours: 1000.0,
            max_storage_bytes: 500 * 1024 * 1024 * 1024, // 500 GB
            max_frames_per_job: 10_000,
        }
    }
}

/// Current usage counters for a quota holder.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QuotaUsage {
    /// Number of currently running jobs.
    pub current_jobs: u32,
    /// Total render-hours consumed this period.
    pub render_hours_used: f64,
    /// Total storage consumed in bytes.
    pub storage_bytes_used: u64,
}

impl QuotaUsage {
    /// Creates an empty usage record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The result of checking whether a resource request fits within quota.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaCheckResult {
    /// The request is within limits.
    Allowed,
    /// Denied because the concurrent-job limit is reached.
    DeniedConcurrentJobs,
    /// Denied because the render-hour budget would be exceeded.
    DeniedRenderHours,
    /// Denied because storage would be exceeded.
    DeniedStorage,
    /// Denied because the frame count exceeds per-job maximum.
    DeniedFrameCount,
    /// The holder has no quota record (unknown).
    UnknownHolder,
}

/// A pending resource request to be checked against quota.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuotaRequest {
    /// Number of additional concurrent jobs.
    pub jobs: u32,
    /// Estimated render-hours.
    pub estimated_hours: f64,
    /// Estimated storage bytes.
    pub estimated_storage: u64,
    /// Number of frames in the job.
    pub frame_count: u32,
}

/// Central quota manager tracking limits and usage.
#[derive(Debug, Clone)]
pub struct QuotaManager {
    /// Per-holder limits.
    limits: HashMap<QuotaHolderId, QuotaLimits>,
    /// Per-holder current usage.
    usage: HashMap<QuotaHolderId, QuotaUsage>,
    /// Default limits for holders without explicit configuration.
    default_limits: QuotaLimits,
}

impl QuotaManager {
    /// Creates a new quota manager with default limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
            usage: HashMap::new(),
            default_limits: QuotaLimits::default(),
        }
    }

    /// Creates a new quota manager with custom default limits.
    #[must_use]
    pub fn with_defaults(defaults: QuotaLimits) -> Self {
        Self {
            limits: HashMap::new(),
            usage: HashMap::new(),
            default_limits: defaults,
        }
    }

    /// Sets the limits for a specific holder, overriding defaults.
    pub fn set_limits(&mut self, holder: QuotaHolderId, limits: QuotaLimits) {
        self.limits.insert(holder, limits);
    }

    /// Returns the effective limits for a holder (explicit or default).
    #[must_use]
    pub fn effective_limits(&self, holder: &QuotaHolderId) -> QuotaLimits {
        self.limits
            .get(holder)
            .copied()
            .unwrap_or(self.default_limits)
    }

    /// Returns the current usage for a holder.
    #[must_use]
    pub fn usage(&self, holder: &QuotaHolderId) -> QuotaUsage {
        self.usage.get(holder).copied().unwrap_or_default()
    }

    /// Checks whether a request fits within the holder's quota.
    #[must_use]
    pub fn check(&self, holder: &QuotaHolderId, request: &QuotaRequest) -> QuotaCheckResult {
        let limits = self.effective_limits(holder);
        let usage = self.usage(holder);

        if usage.current_jobs + request.jobs > limits.max_concurrent_jobs {
            return QuotaCheckResult::DeniedConcurrentJobs;
        }
        if usage.render_hours_used + request.estimated_hours > limits.max_render_hours {
            return QuotaCheckResult::DeniedRenderHours;
        }
        if usage.storage_bytes_used + request.estimated_storage > limits.max_storage_bytes {
            return QuotaCheckResult::DeniedStorage;
        }
        if request.frame_count > limits.max_frames_per_job {
            return QuotaCheckResult::DeniedFrameCount;
        }

        QuotaCheckResult::Allowed
    }

    /// Records that new jobs have started for a holder.
    pub fn acquire_jobs(&mut self, holder: &QuotaHolderId, count: u32) {
        let u = self.usage.entry(holder.clone()).or_default();
        u.current_jobs += count;
    }

    /// Records that jobs have completed for a holder.
    pub fn release_jobs(&mut self, holder: &QuotaHolderId, count: u32) {
        let u = self.usage.entry(holder.clone()).or_default();
        u.current_jobs = u.current_jobs.saturating_sub(count);
    }

    /// Adds render-hours to a holder's consumption.
    pub fn record_hours(&mut self, holder: &QuotaHolderId, hours: f64) {
        let u = self.usage.entry(holder.clone()).or_default();
        u.render_hours_used += hours;
    }

    /// Adds storage to a holder's consumption.
    pub fn record_storage(&mut self, holder: &QuotaHolderId, bytes: u64) {
        let u = self.usage.entry(holder.clone()).or_default();
        u.storage_bytes_used += bytes;
    }

    /// Resets a holder's usage counters (e.g. at the start of a new billing period).
    pub fn reset_usage(&mut self, holder: &QuotaHolderId) {
        if let Some(u) = self.usage.get_mut(holder) {
            *u = QuotaUsage::default();
        }
    }

    /// Returns the number of tracked holders.
    #[must_use]
    pub fn holder_count(&self) -> usize {
        let mut all: std::collections::HashSet<&QuotaHolderId> = self.limits.keys().collect();
        for k in self.usage.keys() {
            all.insert(k);
        }
        all.len()
    }

    /// Returns the remaining concurrent job slots for a holder.
    #[must_use]
    pub fn remaining_jobs(&self, holder: &QuotaHolderId) -> u32 {
        let limits = self.effective_limits(holder);
        let usage = self.usage(holder);
        limits
            .max_concurrent_jobs
            .saturating_sub(usage.current_jobs)
    }

    /// Returns the remaining render-hours for a holder.
    #[must_use]
    pub fn remaining_hours(&self, holder: &QuotaHolderId) -> f64 {
        let limits = self.effective_limits(holder);
        let usage = self.usage(holder);
        (limits.max_render_hours - usage.render_hours_used).max(0.0)
    }
}

impl Default for QuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_holder() -> QuotaHolderId {
        QuotaHolderId::new("user-42")
    }

    fn small_limits() -> QuotaLimits {
        QuotaLimits {
            max_concurrent_jobs: 3,
            max_render_hours: 100.0,
            max_storage_bytes: 1024 * 1024, // 1 MB
            max_frames_per_job: 500,
        }
    }

    #[test]
    fn test_holder_id() {
        let h = QuotaHolderId::new("abc");
        assert_eq!(h.as_str(), "abc");
    }

    #[test]
    fn test_default_limits() {
        let d = QuotaLimits::default();
        assert_eq!(d.max_concurrent_jobs, 10);
        assert!(d.max_render_hours > 0.0);
    }

    #[test]
    fn test_usage_default() {
        let u = QuotaUsage::new();
        assert_eq!(u.current_jobs, 0);
        assert!((u.render_hours_used).abs() < f64::EPSILON);
    }

    #[test]
    fn test_manager_new() {
        let m = QuotaManager::new();
        assert_eq!(m.holder_count(), 0);
    }

    #[test]
    fn test_set_and_get_limits() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        let l = m.effective_limits(&h);
        assert_eq!(l.max_concurrent_jobs, 3);
    }

    #[test]
    fn test_check_allowed() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        let req = QuotaRequest {
            jobs: 1,
            estimated_hours: 10.0,
            estimated_storage: 512,
            frame_count: 100,
        };
        assert_eq!(m.check(&h, &req), QuotaCheckResult::Allowed);
    }

    #[test]
    fn test_check_denied_concurrent() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        m.acquire_jobs(&h, 3);
        let req = QuotaRequest {
            jobs: 1,
            estimated_hours: 1.0,
            estimated_storage: 0,
            frame_count: 10,
        };
        assert_eq!(m.check(&h, &req), QuotaCheckResult::DeniedConcurrentJobs);
    }

    #[test]
    fn test_check_denied_hours() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        m.record_hours(&h, 95.0);
        let req = QuotaRequest {
            jobs: 1,
            estimated_hours: 10.0,
            estimated_storage: 0,
            frame_count: 10,
        };
        assert_eq!(m.check(&h, &req), QuotaCheckResult::DeniedRenderHours);
    }

    #[test]
    fn test_check_denied_storage() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        m.record_storage(&h, 1024 * 1024);
        let req = QuotaRequest {
            jobs: 1,
            estimated_hours: 1.0,
            estimated_storage: 1,
            frame_count: 10,
        };
        assert_eq!(m.check(&h, &req), QuotaCheckResult::DeniedStorage);
    }

    #[test]
    fn test_check_denied_frames() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        let req = QuotaRequest {
            jobs: 1,
            estimated_hours: 1.0,
            estimated_storage: 0,
            frame_count: 501,
        };
        assert_eq!(m.check(&h, &req), QuotaCheckResult::DeniedFrameCount);
    }

    #[test]
    fn test_release_jobs() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.acquire_jobs(&h, 5);
        m.release_jobs(&h, 3);
        assert_eq!(m.usage(&h).current_jobs, 2);
    }

    #[test]
    fn test_reset_usage() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.acquire_jobs(&h, 5);
        m.record_hours(&h, 50.0);
        m.reset_usage(&h);
        let u = m.usage(&h);
        assert_eq!(u.current_jobs, 0);
        assert!((u.render_hours_used).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remaining_jobs() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        m.acquire_jobs(&h, 1);
        assert_eq!(m.remaining_jobs(&h), 2);
    }

    #[test]
    fn test_remaining_hours() {
        let mut m = QuotaManager::new();
        let h = test_holder();
        m.set_limits(h.clone(), small_limits());
        m.record_hours(&h, 40.0);
        assert!((m.remaining_hours(&h) - 60.0).abs() < f64::EPSILON);
    }
}
