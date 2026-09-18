//! Per-user storage and bandwidth quotas with enforcement.
//!
//! Tracks storage usage, bandwidth consumption, and request counts per user.
//! Provides quota limits, enforcement, and usage reporting.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Types of quotas that can be tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuotaType {
    /// Storage in bytes.
    Storage,
    /// Bandwidth in bytes per period.
    Bandwidth,
    /// Number of API requests per period.
    RequestCount,
    /// Number of transcode operations per period.
    TranscodeCount,
    /// Number of uploads per period.
    UploadCount,
}

impl QuotaType {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Storage => "storage",
            Self::Bandwidth => "bandwidth",
            Self::RequestCount => "request_count",
            Self::TranscodeCount => "transcode_count",
            Self::UploadCount => "upload_count",
        }
    }

    /// Unit label for display.
    pub fn unit(self) -> &'static str {
        match self {
            Self::Storage | Self::Bandwidth => "bytes",
            Self::RequestCount | Self::TranscodeCount | Self::UploadCount => "count",
        }
    }
}

/// A quota limit definition.
#[derive(Debug, Clone)]
pub struct QuotaLimit {
    /// Type of quota.
    pub quota_type: QuotaType,
    /// Maximum allowed value.
    pub limit: u64,
    /// Soft limit (warning threshold, percentage 0-100).
    pub soft_limit_pct: u8,
    /// Period for rate-based quotas (None for cumulative like storage).
    pub period: Option<Duration>,
}

impl QuotaLimit {
    /// Creates a new quota limit.
    pub fn new(quota_type: QuotaType, limit: u64) -> Self {
        Self {
            quota_type,
            limit,
            soft_limit_pct: 80,
            period: None,
        }
    }

    /// Sets the period for rate-based quotas.
    pub fn with_period(mut self, period: Duration) -> Self {
        self.period = Some(period);
        self
    }

    /// Sets the soft limit percentage.
    pub fn with_soft_limit_pct(mut self, pct: u8) -> Self {
        self.soft_limit_pct = pct.min(100);
        self
    }

    /// Returns the soft limit value.
    pub fn soft_limit(&self) -> u64 {
        (self.limit as f64 * self.soft_limit_pct as f64 / 100.0) as u64
    }
}

/// Current usage for a specific quota.
#[derive(Debug, Clone)]
pub struct QuotaUsage {
    /// Current usage value.
    pub current: u64,
    /// Start of the current period.
    pub period_start: Instant,
    /// Last update timestamp.
    pub last_updated: Instant,
}

impl QuotaUsage {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            current: 0,
            period_start: now,
            last_updated: now,
        }
    }

    /// Checks if the period has expired.
    fn is_period_expired(&self, period: Duration) -> bool {
        self.period_start.elapsed() > period
    }
}

/// Result of a quota check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaCheckResult {
    /// Usage is within limits.
    Allowed,
    /// Usage is above soft limit but below hard limit.
    Warning {
        /// Current usage.
        current: u64,
        /// Hard limit.
        limit: u64,
        /// Percentage used.
        usage_pct: u8,
    },
    /// Hard limit exceeded.
    Exceeded {
        /// Current usage.
        current: u64,
        /// Hard limit.
        limit: u64,
        /// How much over the limit.
        excess: u64,
    },
}

impl QuotaCheckResult {
    /// Returns `true` if the request should be allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed | Self::Warning { .. })
    }

    /// Returns `true` if the quota is exceeded.
    pub fn is_exceeded(&self) -> bool {
        matches!(self, Self::Exceeded { .. })
    }
}

/// Quota plan for a user tier.
#[derive(Debug, Clone)]
pub struct QuotaPlan {
    /// Plan name.
    pub name: String,
    /// Quota limits.
    pub limits: Vec<QuotaLimit>,
}

impl QuotaPlan {
    /// Creates a free tier plan.
    pub fn free_tier() -> Self {
        Self {
            name: "free".to_string(),
            limits: vec![
                QuotaLimit::new(QuotaType::Storage, 1024 * 1024 * 1024), // 1 GB
                QuotaLimit::new(QuotaType::Bandwidth, 10 * 1024 * 1024 * 1024)
                    .with_period(Duration::from_secs(30 * 24 * 3600)), // 10 GB/month
                QuotaLimit::new(QuotaType::RequestCount, 10_000)
                    .with_period(Duration::from_secs(3600)), // 10K/hour
                QuotaLimit::new(QuotaType::TranscodeCount, 50)
                    .with_period(Duration::from_secs(24 * 3600)), // 50/day
                QuotaLimit::new(QuotaType::UploadCount, 100)
                    .with_period(Duration::from_secs(24 * 3600)), // 100/day
            ],
        }
    }

    /// Creates a pro tier plan.
    pub fn pro_tier() -> Self {
        Self {
            name: "pro".to_string(),
            limits: vec![
                QuotaLimit::new(QuotaType::Storage, 100 * 1024 * 1024 * 1024), // 100 GB
                QuotaLimit::new(QuotaType::Bandwidth, 1024 * 1024 * 1024 * 1024)
                    .with_period(Duration::from_secs(30 * 24 * 3600)), // 1 TB/month
                QuotaLimit::new(QuotaType::RequestCount, 100_000)
                    .with_period(Duration::from_secs(3600)), // 100K/hour
                QuotaLimit::new(QuotaType::TranscodeCount, 500)
                    .with_period(Duration::from_secs(24 * 3600)), // 500/day
                QuotaLimit::new(QuotaType::UploadCount, 1000)
                    .with_period(Duration::from_secs(24 * 3600)), // 1000/day
            ],
        }
    }

    /// Gets the limit for a specific quota type.
    pub fn get_limit(&self, quota_type: QuotaType) -> Option<&QuotaLimit> {
        self.limits.iter().find(|l| l.quota_type == quota_type)
    }
}

/// Per-user quota usage report.
#[derive(Debug, Clone)]
pub struct UserQuotaReport {
    /// User ID.
    pub user_id: String,
    /// Plan name.
    pub plan: String,
    /// Per-quota-type usage and limit information.
    pub entries: Vec<QuotaReportEntry>,
}

/// A single entry in a quota report.
#[derive(Debug, Clone)]
pub struct QuotaReportEntry {
    /// Quota type.
    pub quota_type: QuotaType,
    /// Current usage.
    pub current: u64,
    /// Hard limit.
    pub limit: u64,
    /// Usage percentage.
    pub usage_pct: f64,
    /// Remaining before limit.
    pub remaining: u64,
}

/// The quota manager.
pub struct QuotaManager {
    /// User → plan assignment.
    user_plans: HashMap<String, String>,
    /// Available plans.
    plans: HashMap<String, QuotaPlan>,
    /// User → quota type → usage.
    usage: HashMap<String, HashMap<QuotaType, QuotaUsage>>,
    /// Default plan for new users.
    default_plan: String,
}

impl QuotaManager {
    /// Creates a new quota manager with default plans.
    pub fn new() -> Self {
        let mut plans = HashMap::new();
        plans.insert("free".to_string(), QuotaPlan::free_tier());
        plans.insert("pro".to_string(), QuotaPlan::pro_tier());
        Self {
            user_plans: HashMap::new(),
            plans,
            usage: HashMap::new(),
            default_plan: "free".to_string(),
        }
    }

    /// Registers a custom plan.
    pub fn register_plan(&mut self, plan: QuotaPlan) {
        self.plans.insert(plan.name.clone(), plan);
    }

    /// Assigns a plan to a user.
    pub fn assign_plan(&mut self, user_id: &str, plan_name: &str) -> bool {
        if self.plans.contains_key(plan_name) {
            self.user_plans
                .insert(user_id.to_string(), plan_name.to_string());
            true
        } else {
            false
        }
    }

    /// Gets the plan assigned to a user.
    pub fn user_plan(&self, user_id: &str) -> &str {
        self.user_plans
            .get(user_id)
            .map(String::as_str)
            .unwrap_or(&self.default_plan)
    }

    /// Checks whether a user can consume `amount` of the given quota type.
    pub fn check_quota(
        &mut self,
        user_id: &str,
        quota_type: QuotaType,
        amount: u64,
    ) -> QuotaCheckResult {
        let plan_name = self.user_plan(user_id).to_string();
        let plan = match self.plans.get(&plan_name) {
            Some(p) => p,
            None => return QuotaCheckResult::Allowed, // no plan = no limits
        };

        let limit = match plan.get_limit(quota_type) {
            Some(l) => l.clone(),
            None => return QuotaCheckResult::Allowed,
        };

        let user_usage = self.usage.entry(user_id.to_string()).or_default();

        let usage = user_usage.entry(quota_type).or_insert_with(QuotaUsage::new);

        // Reset period if expired
        if let Some(period) = limit.period {
            if usage.is_period_expired(period) {
                usage.current = 0;
                usage.period_start = Instant::now();
            }
        }

        let projected = usage.current + amount;

        if projected > limit.limit {
            QuotaCheckResult::Exceeded {
                current: usage.current,
                limit: limit.limit,
                excess: projected - limit.limit,
            }
        } else if projected > limit.soft_limit() {
            let usage_pct = (projected as f64 / limit.limit as f64 * 100.0) as u8;
            QuotaCheckResult::Warning {
                current: usage.current,
                limit: limit.limit,
                usage_pct,
            }
        } else {
            QuotaCheckResult::Allowed
        }
    }

    /// Records usage of a quota (after the operation succeeds).
    pub fn record_usage(&mut self, user_id: &str, quota_type: QuotaType, amount: u64) {
        let user_usage = self.usage.entry(user_id.to_string()).or_default();

        let usage = user_usage.entry(quota_type).or_insert_with(QuotaUsage::new);

        usage.current += amount;
        usage.last_updated = Instant::now();
    }

    /// Gets current usage for a user and quota type.
    pub fn get_usage(&self, user_id: &str, quota_type: QuotaType) -> u64 {
        self.usage
            .get(user_id)
            .and_then(|u| u.get(&quota_type))
            .map(|u| u.current)
            .unwrap_or(0)
    }

    /// Generates a usage report for a user.
    pub fn user_report(&self, user_id: &str) -> UserQuotaReport {
        let plan_name = self.user_plan(user_id).to_string();
        let plan = self.plans.get(&plan_name);

        let entries = if let Some(plan) = plan {
            plan.limits
                .iter()
                .map(|limit| {
                    let current = self.get_usage(user_id, limit.quota_type);
                    let usage_pct = if limit.limit > 0 {
                        current as f64 / limit.limit as f64 * 100.0
                    } else {
                        0.0
                    };
                    QuotaReportEntry {
                        quota_type: limit.quota_type,
                        current,
                        limit: limit.limit,
                        usage_pct,
                        remaining: limit.limit.saturating_sub(current),
                    }
                })
                .collect()
        } else {
            vec![]
        };

        UserQuotaReport {
            user_id: user_id.to_string(),
            plan: plan_name,
            entries,
        }
    }

    /// Resets all usage for a user.
    pub fn reset_usage(&mut self, user_id: &str) {
        self.usage.remove(user_id);
    }

    /// Returns the number of tracked users.
    pub fn tracked_user_count(&self) -> usize {
        self.usage.len()
    }

    /// Returns the total storage used across all users.
    pub fn total_storage_used(&self) -> u64 {
        self.usage
            .values()
            .filter_map(|u| u.get(&QuotaType::Storage))
            .map(|u| u.current)
            .sum()
    }
}

impl Default for QuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Formats bytes into human-readable form.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;
    const TB: u64 = 1024 * 1024 * 1024 * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // QuotaType

    #[test]
    fn test_quota_type_labels() {
        assert_eq!(QuotaType::Storage.label(), "storage");
        assert_eq!(QuotaType::Bandwidth.label(), "bandwidth");
        assert_eq!(QuotaType::RequestCount.label(), "request_count");
    }

    #[test]
    fn test_quota_type_units() {
        assert_eq!(QuotaType::Storage.unit(), "bytes");
        assert_eq!(QuotaType::RequestCount.unit(), "count");
    }

    // QuotaLimit

    #[test]
    fn test_quota_limit_soft_limit() {
        let limit = QuotaLimit::new(QuotaType::Storage, 1000);
        assert_eq!(limit.soft_limit(), 800); // 80% default
    }

    #[test]
    fn test_quota_limit_custom_soft() {
        let limit = QuotaLimit::new(QuotaType::Storage, 1000).with_soft_limit_pct(90);
        assert_eq!(limit.soft_limit(), 900);
    }

    // QuotaCheckResult

    #[test]
    fn test_check_result_allowed() {
        assert!(QuotaCheckResult::Allowed.is_allowed());
        assert!(!QuotaCheckResult::Allowed.is_exceeded());
    }

    #[test]
    fn test_check_result_warning() {
        let w = QuotaCheckResult::Warning {
            current: 80,
            limit: 100,
            usage_pct: 80,
        };
        assert!(w.is_allowed());
        assert!(!w.is_exceeded());
    }

    #[test]
    fn test_check_result_exceeded() {
        let e = QuotaCheckResult::Exceeded {
            current: 100,
            limit: 100,
            excess: 10,
        };
        assert!(!e.is_allowed());
        assert!(e.is_exceeded());
    }

    // QuotaPlan

    #[test]
    fn test_free_tier() {
        let plan = QuotaPlan::free_tier();
        assert_eq!(plan.name, "free");
        assert!(plan.get_limit(QuotaType::Storage).is_some());
        assert!(plan.get_limit(QuotaType::Bandwidth).is_some());
    }

    #[test]
    fn test_pro_tier_larger_limits() {
        let free = QuotaPlan::free_tier();
        let pro = QuotaPlan::pro_tier();
        let free_storage = free.get_limit(QuotaType::Storage).expect("should exist");
        let pro_storage = pro.get_limit(QuotaType::Storage).expect("should exist");
        assert!(pro_storage.limit > free_storage.limit);
    }

    // QuotaManager

    #[test]
    fn test_default_plan_is_free() {
        let mgr = QuotaManager::new();
        assert_eq!(mgr.user_plan("new-user"), "free");
    }

    #[test]
    fn test_assign_plan() {
        let mut mgr = QuotaManager::new();
        assert!(mgr.assign_plan("user1", "pro"));
        assert_eq!(mgr.user_plan("user1"), "pro");
    }

    #[test]
    fn test_assign_unknown_plan() {
        let mut mgr = QuotaManager::new();
        assert!(!mgr.assign_plan("user1", "enterprise"));
    }

    #[test]
    fn test_check_quota_allowed() {
        let mut mgr = QuotaManager::new();
        let result = mgr.check_quota("user1", QuotaType::Storage, 1000);
        assert!(result.is_allowed());
    }

    #[test]
    fn test_check_quota_warning() {
        let mut mgr = QuotaManager::new();
        // Free tier storage = 1 GB
        let almost_full = 900 * 1024 * 1024; // 900 MB
        mgr.record_usage("user1", QuotaType::Storage, almost_full);
        let result = mgr.check_quota("user1", QuotaType::Storage, 1);
        assert!(matches!(result, QuotaCheckResult::Warning { .. }));
    }

    #[test]
    fn test_check_quota_exceeded() {
        let mut mgr = QuotaManager::new();
        let full = 1024 * 1024 * 1024; // 1 GB (free tier limit)
        mgr.record_usage("user1", QuotaType::Storage, full);
        let result = mgr.check_quota("user1", QuotaType::Storage, 1);
        assert!(result.is_exceeded());
    }

    #[test]
    fn test_record_and_get_usage() {
        let mut mgr = QuotaManager::new();
        mgr.record_usage("user1", QuotaType::Storage, 500);
        mgr.record_usage("user1", QuotaType::Storage, 300);
        assert_eq!(mgr.get_usage("user1", QuotaType::Storage), 800);
    }

    #[test]
    fn test_get_usage_no_data() {
        let mgr = QuotaManager::new();
        assert_eq!(mgr.get_usage("user1", QuotaType::Storage), 0);
    }

    #[test]
    fn test_user_report() {
        let mut mgr = QuotaManager::new();
        mgr.record_usage("user1", QuotaType::Storage, 1000);
        let report = mgr.user_report("user1");
        assert_eq!(report.user_id, "user1");
        assert_eq!(report.plan, "free");
        assert!(!report.entries.is_empty());
    }

    #[test]
    fn test_reset_usage() {
        let mut mgr = QuotaManager::new();
        mgr.record_usage("user1", QuotaType::Storage, 500);
        mgr.reset_usage("user1");
        assert_eq!(mgr.get_usage("user1", QuotaType::Storage), 0);
    }

    #[test]
    fn test_tracked_user_count() {
        let mut mgr = QuotaManager::new();
        mgr.record_usage("user1", QuotaType::Storage, 1);
        mgr.record_usage("user2", QuotaType::Storage, 1);
        assert_eq!(mgr.tracked_user_count(), 2);
    }

    #[test]
    fn test_total_storage_used() {
        let mut mgr = QuotaManager::new();
        mgr.record_usage("user1", QuotaType::Storage, 100);
        mgr.record_usage("user2", QuotaType::Storage, 200);
        assert_eq!(mgr.total_storage_used(), 300);
    }

    #[test]
    fn test_register_custom_plan() {
        let mut mgr = QuotaManager::new();
        let plan = QuotaPlan {
            name: "enterprise".to_string(),
            limits: vec![QuotaLimit::new(QuotaType::Storage, u64::MAX)],
        };
        mgr.register_plan(plan);
        assert!(mgr.assign_plan("user1", "enterprise"));
    }

    // format_bytes

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
