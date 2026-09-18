#![allow(dead_code)]
//! Scheduling rules for controlling when and how jobs are dispatched.
//!
//! Provides time-window constraints, resource-aware scheduling, priority-based
//! ordering, and affinity/anti-affinity rules for job placement.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Day of the week for time-window rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weekday {
    /// Monday.
    Monday,
    /// Tuesday.
    Tuesday,
    /// Wednesday.
    Wednesday,
    /// Thursday.
    Thursday,
    /// Friday.
    Friday,
    /// Saturday.
    Saturday,
    /// Sunday.
    Sunday,
}

impl fmt::Display for Weekday {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Weekday::Monday => "Mon",
            Weekday::Tuesday => "Tue",
            Weekday::Wednesday => "Wed",
            Weekday::Thursday => "Thu",
            Weekday::Friday => "Fri",
            Weekday::Saturday => "Sat",
            Weekday::Sunday => "Sun",
        };
        write!(f, "{s}")
    }
}

/// A time-of-day value in hours and minutes (24-hour clock).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeOfDay {
    /// Hour (0..23).
    pub hour: u8,
    /// Minute (0..59).
    pub minute: u8,
}

impl TimeOfDay {
    /// Create a new time-of-day value.
    ///
    /// # Panics
    ///
    /// Panics if hour >= 24 or minute >= 60.
    pub fn new(hour: u8, minute: u8) -> Self {
        assert!(hour < 24, "hour must be < 24");
        assert!(minute < 60, "minute must be < 60");
        Self { hour, minute }
    }

    /// Return the time as total minutes since midnight.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_minutes(&self) -> u16 {
        u16::from(self.hour) * 60 + u16::from(self.minute)
    }
}

impl fmt::Display for TimeOfDay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02}:{:02}", self.hour, self.minute)
    }
}

/// A time window during which jobs may or may not be scheduled.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeWindow {
    /// Start of the window.
    pub start: TimeOfDay,
    /// End of the window (exclusive).
    pub end: TimeOfDay,
    /// Days this window applies to.
    pub days: HashSet<Weekday>,
}

impl TimeWindow {
    /// Create a time window for the given range and days.
    pub fn new(start: TimeOfDay, end: TimeOfDay, days: HashSet<Weekday>) -> Self {
        Self { start, end, days }
    }

    /// Create a weekday-only window (Mon-Fri).
    pub fn weekdays(start: TimeOfDay, end: TimeOfDay) -> Self {
        let days = [
            Weekday::Monday,
            Weekday::Tuesday,
            Weekday::Wednesday,
            Weekday::Thursday,
            Weekday::Friday,
        ]
        .into_iter()
        .collect();
        Self { start, end, days }
    }

    /// Create a weekend-only window (Sat-Sun).
    pub fn weekends(start: TimeOfDay, end: TimeOfDay) -> Self {
        let days = [Weekday::Saturday, Weekday::Sunday].into_iter().collect();
        Self { start, end, days }
    }

    /// Create a window that applies every day of the week.
    pub fn every_day(start: TimeOfDay, end: TimeOfDay) -> Self {
        let days = [
            Weekday::Monday,
            Weekday::Tuesday,
            Weekday::Wednesday,
            Weekday::Thursday,
            Weekday::Friday,
            Weekday::Saturday,
            Weekday::Sunday,
        ]
        .into_iter()
        .collect();
        Self { start, end, days }
    }

    /// Check whether the given day and time fall within this window.
    pub fn contains(&self, day: Weekday, time: TimeOfDay) -> bool {
        if !self.days.contains(&day) {
            return false;
        }
        if self.start <= self.end {
            time >= self.start && time < self.end
        } else {
            // Wraps past midnight
            time >= self.start || time < self.end
        }
    }

    /// Duration of this window in minutes.
    pub fn duration_minutes(&self) -> u16 {
        let s = self.start.total_minutes();
        let e = self.end.total_minutes();
        if e >= s {
            e - s
        } else {
            (24 * 60 - s) + e
        }
    }
}

/// Resource requirement for a scheduling rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRequirement {
    /// Resource name (e.g., "gpu", "cpu_cores", "memory_mb").
    pub name: String,
    /// Minimum amount required.
    pub min_amount: f64,
    /// Whether this is a hard requirement (fail if not met) or soft (degrade).
    pub hard: bool,
}

/// Affinity type for job placement rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffinityKind {
    /// Prefer to co-locate with jobs having these tags.
    Affinity,
    /// Prefer to avoid co-location with jobs having these tags.
    AntiAffinity,
}

/// Affinity rule for job scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinityRule {
    /// The kind of affinity.
    pub kind: AffinityKind,
    /// Tags to match against.
    pub tags: HashSet<String>,
    /// Whether this is a hard or soft rule.
    pub hard: bool,
}

impl AffinityRule {
    /// Create a hard affinity rule.
    pub fn hard_affinity(tags: HashSet<String>) -> Self {
        Self {
            kind: AffinityKind::Affinity,
            tags,
            hard: true,
        }
    }

    /// Create a soft affinity rule.
    pub fn soft_affinity(tags: HashSet<String>) -> Self {
        Self {
            kind: AffinityKind::Affinity,
            tags,
            hard: false,
        }
    }

    /// Create a hard anti-affinity rule.
    pub fn hard_anti_affinity(tags: HashSet<String>) -> Self {
        Self {
            kind: AffinityKind::AntiAffinity,
            tags,
            hard: true,
        }
    }

    /// Check whether the given set of co-located job tags satisfies this rule.
    pub fn is_satisfied(&self, co_located_tags: &HashSet<String>) -> bool {
        let overlap = self.tags.intersection(co_located_tags).count() > 0;
        match self.kind {
            AffinityKind::Affinity => overlap,
            AffinityKind::AntiAffinity => !overlap,
        }
    }
}

/// A single scheduling rule.
#[derive(Debug, Clone)]
pub enum SchedulingRule {
    /// Allow scheduling only within the given time window.
    AllowWindow(TimeWindow),
    /// Deny scheduling within the given time window (blackout).
    DenyWindow(TimeWindow),
    /// Require specified resources.
    RequireResource(ResourceRequirement),
    /// Affinity/anti-affinity for co-location.
    Affinity(AffinityRule),
    /// Maximum concurrent jobs matching these tags.
    MaxConcurrent {
        /// Tags identifying the job group.
        tags: HashSet<String>,
        /// Maximum number of concurrent jobs.
        limit: u32,
    },
    /// Minimum delay between consecutive job starts (throttling).
    MinInterval {
        /// Minimum seconds between job starts.
        seconds: u64,
    },
}

/// Outcome of evaluating a scheduling rule.
#[derive(Debug, Clone, PartialEq)]
pub enum RuleOutcome {
    /// The rule is satisfied; scheduling may proceed.
    Allow,
    /// The rule is violated; scheduling should be blocked.
    Deny {
        /// Reason for denial.
        reason: String,
    },
    /// The rule is soft-violated; scheduling may proceed with a penalty score.
    SoftDeny {
        /// Penalty score (higher = worse).
        penalty: f64,
        /// Reason for the penalty.
        reason: String,
    },
}

/// Context provided when evaluating scheduling rules.
#[derive(Debug, Clone)]
pub struct SchedulingContext {
    /// Current day of the week.
    pub day: Weekday,
    /// Current time of day.
    pub time: TimeOfDay,
    /// Available resources on the target worker.
    pub available_resources: HashMap<String, f64>,
    /// Tags of jobs currently co-located on the target worker.
    pub co_located_tags: HashSet<String>,
    /// Number of currently running jobs matching specific tag groups.
    pub running_counts: HashMap<String, u32>,
}

/// Evaluate a single rule against the given context.
#[allow(clippy::cast_precision_loss)]
pub fn evaluate_rule(rule: &SchedulingRule, ctx: &SchedulingContext) -> RuleOutcome {
    match rule {
        SchedulingRule::AllowWindow(window) => {
            if window.contains(ctx.day, ctx.time) {
                RuleOutcome::Allow
            } else {
                RuleOutcome::Deny {
                    reason: format!("Outside allowed window ({} - {})", window.start, window.end),
                }
            }
        }
        SchedulingRule::DenyWindow(window) => {
            if window.contains(ctx.day, ctx.time) {
                RuleOutcome::Deny {
                    reason: format!("Inside blackout window ({} - {})", window.start, window.end),
                }
            } else {
                RuleOutcome::Allow
            }
        }
        SchedulingRule::RequireResource(req) => {
            let available = ctx
                .available_resources
                .get(&req.name)
                .copied()
                .unwrap_or(0.0);
            if available >= req.min_amount {
                RuleOutcome::Allow
            } else if req.hard {
                RuleOutcome::Deny {
                    reason: format!(
                        "Insufficient resource {}: need {}, have {}",
                        req.name, req.min_amount, available
                    ),
                }
            } else {
                RuleOutcome::SoftDeny {
                    penalty: (req.min_amount - available) / req.min_amount,
                    reason: format!(
                        "Low resource {}: {}/{}",
                        req.name, available, req.min_amount
                    ),
                }
            }
        }
        SchedulingRule::Affinity(affinity) => {
            let satisfied = affinity.is_satisfied(&ctx.co_located_tags);
            if satisfied {
                RuleOutcome::Allow
            } else if affinity.hard {
                RuleOutcome::Deny {
                    reason: format!("{:?} rule not satisfied", affinity.kind),
                }
            } else {
                RuleOutcome::SoftDeny {
                    penalty: 0.5,
                    reason: format!("Soft {:?} rule not satisfied", affinity.kind),
                }
            }
        }
        SchedulingRule::MaxConcurrent { tags, limit } => {
            let count: u32 = tags.iter().filter_map(|t| ctx.running_counts.get(t)).sum();
            if count < *limit {
                RuleOutcome::Allow
            } else {
                RuleOutcome::Deny {
                    reason: format!("Max concurrent limit {limit} reached (running: {count})"),
                }
            }
        }
        SchedulingRule::MinInterval { seconds: _ } => {
            // Interval enforcement requires external timestamp tracking;
            // we allow here and expect the caller to enforce timing.
            RuleOutcome::Allow
        }
    }
}

/// Evaluate a set of rules and return the aggregate outcome.
///
/// If any hard deny is encountered, returns `Deny`. Otherwise, returns
/// `Allow` or `SoftDeny` with the sum of penalties.
pub fn evaluate_rules(rules: &[SchedulingRule], ctx: &SchedulingContext) -> RuleOutcome {
    let mut total_penalty = 0.0_f64;
    let mut soft_reasons = Vec::new();

    for rule in rules {
        match evaluate_rule(rule, ctx) {
            RuleOutcome::Deny { reason } => return RuleOutcome::Deny { reason },
            RuleOutcome::SoftDeny { penalty, reason } => {
                total_penalty += penalty;
                soft_reasons.push(reason);
            }
            RuleOutcome::Allow => {}
        }
    }

    if total_penalty > 0.0 {
        RuleOutcome::SoftDeny {
            penalty: total_penalty,
            reason: soft_reasons.join("; "),
        }
    } else {
        RuleOutcome::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context() -> SchedulingContext {
        let mut resources = HashMap::new();
        resources.insert("gpu".to_string(), 2.0);
        resources.insert("cpu_cores".to_string(), 8.0);
        SchedulingContext {
            day: Weekday::Monday,
            time: TimeOfDay::new(10, 30),
            available_resources: resources,
            co_located_tags: HashSet::new(),
            running_counts: HashMap::new(),
        }
    }

    #[test]
    fn test_time_of_day_creation() {
        let t = TimeOfDay::new(14, 30);
        assert_eq!(t.hour, 14);
        assert_eq!(t.minute, 30);
        assert_eq!(t.total_minutes(), 870);
    }

    #[test]
    fn test_time_of_day_display() {
        assert_eq!(TimeOfDay::new(9, 5).to_string(), "09:05");
        assert_eq!(TimeOfDay::new(23, 59).to_string(), "23:59");
    }

    #[test]
    fn test_time_window_contains() {
        let w = TimeWindow::weekdays(TimeOfDay::new(9, 0), TimeOfDay::new(17, 0));
        assert!(w.contains(Weekday::Monday, TimeOfDay::new(10, 0)));
        assert!(!w.contains(Weekday::Monday, TimeOfDay::new(18, 0)));
        assert!(!w.contains(Weekday::Saturday, TimeOfDay::new(10, 0)));
    }

    #[test]
    fn test_time_window_overnight() {
        let w = TimeWindow::every_day(TimeOfDay::new(22, 0), TimeOfDay::new(6, 0));
        assert!(w.contains(Weekday::Monday, TimeOfDay::new(23, 0)));
        assert!(w.contains(Weekday::Monday, TimeOfDay::new(3, 0)));
        assert!(!w.contains(Weekday::Monday, TimeOfDay::new(12, 0)));
    }

    #[test]
    fn test_time_window_duration() {
        let w = TimeWindow::weekdays(TimeOfDay::new(9, 0), TimeOfDay::new(17, 0));
        assert_eq!(w.duration_minutes(), 480);
        let overnight = TimeWindow::every_day(TimeOfDay::new(22, 0), TimeOfDay::new(6, 0));
        assert_eq!(overnight.duration_minutes(), 480);
    }

    #[test]
    fn test_weekday_display() {
        assert_eq!(Weekday::Monday.to_string(), "Mon");
        assert_eq!(Weekday::Sunday.to_string(), "Sun");
    }

    #[test]
    fn test_allow_window_rule() {
        let ctx = make_context();
        let rule = SchedulingRule::AllowWindow(TimeWindow::weekdays(
            TimeOfDay::new(9, 0),
            TimeOfDay::new(17, 0),
        ));
        assert_eq!(evaluate_rule(&rule, &ctx), RuleOutcome::Allow);
    }

    #[test]
    fn test_deny_window_rule() {
        let ctx = make_context();
        let rule = SchedulingRule::DenyWindow(TimeWindow::weekdays(
            TimeOfDay::new(9, 0),
            TimeOfDay::new(17, 0),
        ));
        assert!(matches!(
            evaluate_rule(&rule, &ctx),
            RuleOutcome::Deny { .. }
        ));
    }

    #[test]
    fn test_require_resource_satisfied() {
        let ctx = make_context();
        let rule = SchedulingRule::RequireResource(ResourceRequirement {
            name: "gpu".to_string(),
            min_amount: 1.0,
            hard: true,
        });
        assert_eq!(evaluate_rule(&rule, &ctx), RuleOutcome::Allow);
    }

    #[test]
    fn test_require_resource_hard_deny() {
        let ctx = make_context();
        let rule = SchedulingRule::RequireResource(ResourceRequirement {
            name: "gpu".to_string(),
            min_amount: 4.0,
            hard: true,
        });
        assert!(matches!(
            evaluate_rule(&rule, &ctx),
            RuleOutcome::Deny { .. }
        ));
    }

    #[test]
    fn test_max_concurrent_allow() {
        let mut ctx = make_context();
        ctx.running_counts.insert("transcode".to_string(), 2);
        let rule = SchedulingRule::MaxConcurrent {
            tags: ["transcode".to_string()].into_iter().collect(),
            limit: 5,
        };
        assert_eq!(evaluate_rule(&rule, &ctx), RuleOutcome::Allow);
    }

    #[test]
    fn test_max_concurrent_deny() {
        let mut ctx = make_context();
        ctx.running_counts.insert("transcode".to_string(), 5);
        let rule = SchedulingRule::MaxConcurrent {
            tags: ["transcode".to_string()].into_iter().collect(),
            limit: 5,
        };
        assert!(matches!(
            evaluate_rule(&rule, &ctx),
            RuleOutcome::Deny { .. }
        ));
    }

    #[test]
    fn test_affinity_satisfied() {
        let mut ctx = make_context();
        ctx.co_located_tags.insert("gpu-job".to_string());
        let rule = SchedulingRule::Affinity(AffinityRule::hard_affinity(
            ["gpu-job".to_string()].into_iter().collect(),
        ));
        assert_eq!(evaluate_rule(&rule, &ctx), RuleOutcome::Allow);
    }

    #[test]
    fn test_evaluate_rules_all_allow() {
        let ctx = make_context();
        let rules = vec![
            SchedulingRule::AllowWindow(TimeWindow::weekdays(
                TimeOfDay::new(9, 0),
                TimeOfDay::new(17, 0),
            )),
            SchedulingRule::RequireResource(ResourceRequirement {
                name: "cpu_cores".to_string(),
                min_amount: 4.0,
                hard: true,
            }),
        ];
        assert_eq!(evaluate_rules(&rules, &ctx), RuleOutcome::Allow);
    }
}
