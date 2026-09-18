//! Cloud replication policies for media assets.
//!
//! Defines how and when objects should be replicated across regions or vendors,
//! including replication lag budgets, priority tiers, and policy evaluation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

/// Geographical region identifier used for replication targeting.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Region(pub String);

impl Region {
    /// Constructs a new region from a string identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the string identifier for this region.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.0
    }
}

/// Priority at which a replication job is scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReplicationPriority {
    /// Low priority – replicates during off-peak windows.
    Low = 0,
    /// Normal priority – standard scheduling.
    Normal = 1,
    /// High priority – expedited replication, subject to higher cost.
    High = 2,
    /// Critical – synchronous or near-synchronous, maximum cost.
    Critical = 3,
}

impl ReplicationPriority {
    /// Returns the maximum allowed replication lag for this priority level.
    #[must_use]
    pub fn max_lag(&self) -> Duration {
        match self {
            Self::Low => Duration::from_secs(3600 * 24),   // 24 h
            Self::Normal => Duration::from_secs(3600 * 4), // 4 h
            Self::High => Duration::from_secs(900),        // 15 min
            Self::Critical => Duration::from_secs(60),     // 1 min
        }
    }
}

/// A single replication target specifying destination region and priority.
#[derive(Debug, Clone)]
pub struct ReplicationTarget {
    /// Destination region for this replica.
    pub region: Region,
    /// Scheduling priority for replication jobs to this target.
    pub priority: ReplicationPriority,
    /// Whether to verify the replica's integrity after copy.
    pub verify_checksum: bool,
}

impl ReplicationTarget {
    /// Creates a new replication target.
    #[must_use]
    pub fn new(region: Region, priority: ReplicationPriority) -> Self {
        Self {
            region,
            priority,
            verify_checksum: true,
        }
    }

    /// Disables checksum verification for this target (e.g., low-cost archival).
    #[must_use]
    pub fn without_verification(mut self) -> Self {
        self.verify_checksum = false;
        self
    }
}

/// A replication policy for a class of media objects.
#[derive(Debug, Clone)]
pub struct ReplicationPolicy {
    /// Descriptive name for this policy.
    pub name: String,
    /// Object key prefix this policy applies to (empty = apply to all).
    pub key_prefix: String,
    /// Ordered list of replication targets (replicated to all of them).
    pub targets: Vec<ReplicationTarget>,
    /// Minimum object size in bytes to trigger replication (0 = no minimum).
    pub min_object_size: u64,
    /// Maximum object size in bytes eligible for replication (0 = no limit).
    pub max_object_size: u64,
}

impl ReplicationPolicy {
    /// Creates a new policy with no targets yet.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            key_prefix: String::new(),
            targets: Vec::new(),
            min_object_size: 0,
            max_object_size: 0,
        }
    }

    /// Adds a replication target to this policy.
    pub fn add_target(&mut self, target: ReplicationTarget) {
        self.targets.push(target);
    }

    /// Sets the object key prefix filter.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    /// Sets minimum and maximum object size bounds (bytes).
    pub fn with_size_bounds(mut self, min: u64, max: u64) -> Self {
        self.min_object_size = min;
        self.max_object_size = max;
        self
    }

    /// Returns `true` if the given object key and size should be replicated
    /// according to this policy.
    #[must_use]
    pub fn applies_to(&self, key: &str, size_bytes: u64) -> bool {
        let prefix_ok = self.key_prefix.is_empty() || key.starts_with(&self.key_prefix);
        let min_ok = self.min_object_size == 0 || size_bytes >= self.min_object_size;
        let max_ok = self.max_object_size == 0 || size_bytes <= self.max_object_size;
        prefix_ok && min_ok && max_ok
    }

    /// Returns the highest priority among all targets in this policy.
    #[must_use]
    pub fn max_priority(&self) -> Option<ReplicationPriority> {
        self.targets.iter().map(|t| t.priority).max()
    }

    /// Returns the number of targets that require checksum verification.
    #[must_use]
    pub fn verified_target_count(&self) -> usize {
        self.targets.iter().filter(|t| t.verify_checksum).count()
    }
}

/// Evaluates a set of policies against an object and returns those that apply.
#[must_use]
pub fn evaluate_policies<'a>(
    policies: &'a [ReplicationPolicy],
    key: &str,
    size_bytes: u64,
) -> Vec<&'a ReplicationPolicy> {
    policies
        .iter()
        .filter(|p| p.applies_to(key, size_bytes))
        .collect()
}

/// Replication status for a single object replica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationStatus {
    /// Replication has not started yet.
    Pending,
    /// Replication is in progress.
    InProgress,
    /// Replication completed successfully.
    Completed,
    /// Replication failed; contains a reason.
    Failed(String),
}

impl ReplicationStatus {
    /// Returns `true` if this status represents a terminal success state.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Completed)
    }

    /// Returns `true` if this status represents a failure.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn us_east() -> Region {
        Region::new("us-east-1")
    }

    fn eu_west() -> Region {
        Region::new("eu-west-1")
    }

    #[test]
    fn test_region_id() {
        let r = Region::new("ap-southeast-1");
        assert_eq!(r.id(), "ap-southeast-1");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(ReplicationPriority::Critical > ReplicationPriority::High);
        assert!(ReplicationPriority::High > ReplicationPriority::Normal);
        assert!(ReplicationPriority::Normal > ReplicationPriority::Low);
    }

    #[test]
    fn test_priority_max_lag_decreases_with_priority() {
        assert!(ReplicationPriority::Low.max_lag() > ReplicationPriority::High.max_lag());
        assert!(ReplicationPriority::Critical.max_lag() < ReplicationPriority::Normal.max_lag());
    }

    #[test]
    fn test_policy_applies_to_prefix() {
        let policy = ReplicationPolicy::new("test").with_prefix("videos/");
        assert!(policy.applies_to("videos/clip.mp4", 1000));
        assert!(!policy.applies_to("audio/clip.mp3", 1000));
    }

    #[test]
    fn test_policy_applies_to_size_bounds() {
        let policy = ReplicationPolicy::new("size-test").with_size_bounds(1_000, 10_000);
        assert!(policy.applies_to("file.mp4", 5_000));
        assert!(!policy.applies_to("file.mp4", 500));
        assert!(!policy.applies_to("file.mp4", 20_000));
    }

    #[test]
    fn test_policy_no_prefix_matches_all_keys() {
        let policy = ReplicationPolicy::new("all");
        assert!(policy.applies_to("any/key/here.mp4", 0));
    }

    #[test]
    fn test_policy_no_size_bounds_matches_all_sizes() {
        let policy = ReplicationPolicy::new("all");
        assert!(policy.applies_to("f", 0));
        assert!(policy.applies_to("f", u64::MAX));
    }

    #[test]
    fn test_max_priority_returns_highest() {
        let mut policy = ReplicationPolicy::new("p");
        policy.add_target(ReplicationTarget::new(us_east(), ReplicationPriority::Low));
        policy.add_target(ReplicationTarget::new(
            eu_west(),
            ReplicationPriority::Critical,
        ));
        assert_eq!(policy.max_priority(), Some(ReplicationPriority::Critical));
    }

    #[test]
    fn test_max_priority_none_when_no_targets() {
        let policy = ReplicationPolicy::new("empty");
        assert_eq!(policy.max_priority(), None);
    }

    #[test]
    fn test_verified_target_count() {
        let mut policy = ReplicationPolicy::new("v");
        policy.add_target(ReplicationTarget::new(
            us_east(),
            ReplicationPriority::Normal,
        ));
        policy.add_target(
            ReplicationTarget::new(eu_west(), ReplicationPriority::Normal).without_verification(),
        );
        assert_eq!(policy.verified_target_count(), 1);
    }

    #[test]
    fn test_evaluate_policies_filters_correctly() {
        let mut video_policy = ReplicationPolicy::new("videos").with_prefix("videos/");
        video_policy.add_target(ReplicationTarget::new(us_east(), ReplicationPriority::High));

        let mut audio_policy = ReplicationPolicy::new("audio").with_prefix("audio/");
        audio_policy.add_target(ReplicationTarget::new(eu_west(), ReplicationPriority::Low));

        let policies = vec![video_policy, audio_policy];
        let matched = evaluate_policies(&policies, "videos/clip.mp4", 1000);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "videos");
    }

    #[test]
    fn test_replication_status_is_complete() {
        assert!(ReplicationStatus::Completed.is_complete());
        assert!(!ReplicationStatus::Pending.is_complete());
        assert!(!ReplicationStatus::InProgress.is_complete());
        assert!(!ReplicationStatus::Failed("err".into()).is_complete());
    }

    #[test]
    fn test_replication_status_is_failed() {
        assert!(ReplicationStatus::Failed("timeout".into()).is_failed());
        assert!(!ReplicationStatus::Completed.is_failed());
    }
}
