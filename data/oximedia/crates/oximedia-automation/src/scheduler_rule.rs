//! Scheduling rules: cron-like rules, calendar scheduling, and dependency chains.
//!
//! This module provides a rule-based scheduling layer for the automation system.
//! Rules can be time-of-day / day-of-week based (cron-like), calendar date
//! based, or depend on the successful completion of other jobs.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Time primitives
// ---------------------------------------------------------------------------

/// A time-of-day expressed in whole seconds from midnight (0..86400).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeOfDay(pub u32);

impl TimeOfDay {
    /// Create from hours, minutes, seconds.
    pub fn hms(h: u32, m: u32, s: u32) -> Self {
        Self(h * 3600 + m * 60 + s)
    }

    pub fn hour(self) -> u32 {
        self.0 / 3600
    }

    pub fn minute(self) -> u32 {
        (self.0 % 3600) / 60
    }

    pub fn second(self) -> u32 {
        self.0 % 60
    }
}

/// A calendar date (year, month 1-12, day 1-31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CalendarDate {
    pub year: u32,
    pub month: u8,
    pub day: u8,
}

impl CalendarDate {
    pub fn new(year: u32, month: u8, day: u8) -> Self {
        Self { year, month, day }
    }
}

/// Day-of-week bitmask (bit 0 = Monday, bit 6 = Sunday).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DayMask(pub u8);

impl DayMask {
    pub const MONDAY: Self = Self(0b0000001);
    pub const TUESDAY: Self = Self(0b0000010);
    pub const WEDNESDAY: Self = Self(0b0000100);
    pub const THURSDAY: Self = Self(0b0001000);
    pub const FRIDAY: Self = Self(0b0010000);
    pub const SATURDAY: Self = Self(0b0100000);
    pub const SUNDAY: Self = Self(0b1000000);
    pub const WEEKDAYS: Self = Self(0b0011111);
    pub const WEEKEND: Self = Self(0b1100000);
    pub const ALL: Self = Self(0b1111111);

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ---------------------------------------------------------------------------
// Cron-like rule
// ---------------------------------------------------------------------------

/// A cron-like scheduling rule.
#[derive(Debug, Clone)]
pub struct CronRule {
    pub id: u64,
    pub name: String,
    /// Days on which this rule is active.
    pub days: DayMask,
    /// Earliest time of day to fire.
    pub start_time: TimeOfDay,
    /// Latest time of day to fire (inclusive window).
    pub end_time: TimeOfDay,
    /// If `Some(interval)`, fire every `interval` seconds within the window.
    pub interval_secs: Option<u32>,
    pub enabled: bool,
}

impl CronRule {
    pub fn new(
        id: u64,
        name: impl Into<String>,
        days: DayMask,
        start: TimeOfDay,
        end: TimeOfDay,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            days,
            start_time: start,
            end_time: end,
            interval_secs: None,
            enabled: true,
        }
    }

    pub fn with_interval(mut self, secs: u32) -> Self {
        self.interval_secs = Some(secs);
        self
    }

    /// Returns `true` if the rule should fire at the given (day_mask, time_of_day).
    pub fn matches(&self, day: DayMask, time: TimeOfDay) -> bool {
        if !self.enabled {
            return false;
        }
        if !self.days.contains(day) {
            return false;
        }
        time >= self.start_time && time <= self.end_time
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

// ---------------------------------------------------------------------------
// Calendar-based rule
// ---------------------------------------------------------------------------

/// A calendar-based rule fires on specific calendar dates.
#[derive(Debug, Clone)]
pub struct CalendarRule {
    pub id: u64,
    pub name: String,
    pub dates: Vec<CalendarDate>,
    pub fire_time: TimeOfDay,
    pub enabled: bool,
}

impl CalendarRule {
    pub fn new(id: u64, name: impl Into<String>, fire_time: TimeOfDay) -> Self {
        Self {
            id,
            name: name.into(),
            dates: Vec::new(),
            fire_time,
            enabled: true,
        }
    }

    pub fn add_date(&mut self, date: CalendarDate) {
        self.dates.push(date);
    }

    pub fn matches(&self, date: CalendarDate, time: TimeOfDay) -> bool {
        if !self.enabled {
            return false;
        }
        self.dates.contains(&date) && time == self.fire_time
    }

    pub fn date_count(&self) -> usize {
        self.dates.len()
    }
}

// ---------------------------------------------------------------------------
// Dependency chain
// ---------------------------------------------------------------------------

/// Unique job identifier within a dependency chain.
pub type JobId = u64;

/// Status of a single job in the dependency chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Skipped,
}

/// A single node in a dependency chain.
#[derive(Debug, Clone)]
pub struct JobNode {
    pub id: JobId,
    pub name: String,
    pub deps: Vec<JobId>,
    pub status: JobStatus,
}

impl JobNode {
    pub fn new(id: JobId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            deps: Vec::new(),
            status: JobStatus::Pending,
        }
    }

    pub fn with_dep(mut self, dep: JobId) -> Self {
        self.deps.push(dep);
        self
    }
}

/// A dependency chain that can be topologically resolved.
#[derive(Debug, Default)]
pub struct DependencyChain {
    nodes: HashMap<JobId, JobNode>,
}

impl DependencyChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_job(&mut self, node: JobNode) {
        self.nodes.insert(node.id, node);
    }

    pub fn mark_succeeded(&mut self, id: JobId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.status = JobStatus::Succeeded;
        }
    }

    pub fn mark_failed(&mut self, id: JobId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.status = JobStatus::Failed;
        }
    }

    /// Returns IDs of jobs whose dependencies are all `Succeeded` and which
    /// are currently `Pending`.
    pub fn ready_jobs(&self) -> Vec<JobId> {
        let mut ready = Vec::new();
        for (id, node) in &self.nodes {
            if node.status != JobStatus::Pending {
                continue;
            }
            let all_done = node.deps.iter().all(|dep_id| {
                self.nodes
                    .get(dep_id)
                    .is_some_and(|d| d.status == JobStatus::Succeeded)
            });
            if all_done {
                ready.push(*id);
            }
        }
        ready.sort_unstable();
        ready
    }

    /// Perform a topological sort (Kahn's algorithm).
    /// Returns `Err(cycle)` if a cycle is detected.
    pub fn topological_order(&self) -> Result<Vec<JobId>, Vec<JobId>> {
        let mut in_degree: HashMap<JobId, usize> = self.nodes.keys().map(|k| (*k, 0)).collect();
        for node in self.nodes.values() {
            for dep in &node.deps {
                *in_degree.entry(*dep).or_insert(0) += 0; // ensure key exists
                                                          // The current node depends on dep, so dep must come first.
                                                          // We track in_degree of each node (how many nodes depend on it already handled)
            }
        }
        // Build adjacency: dep → [nodes that depend on dep]
        let mut successors: HashMap<JobId, Vec<JobId>> = HashMap::new();
        let mut in_deg: HashMap<JobId, usize> = self.nodes.keys().map(|k| (*k, 0)).collect();
        for node in self.nodes.values() {
            for &dep in &node.deps {
                successors.entry(dep).or_default().push(node.id);
                *in_deg.entry(node.id).or_insert(0) += 1;
            }
        }
        let mut queue: VecDeque<JobId> = in_deg
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::new();
        let mut visited: HashSet<JobId> = HashSet::new();
        while let Some(id) = queue.pop_front() {
            if !visited.insert(id) {
                continue;
            }
            order.push(id);
            if let Some(succs) = successors.get(&id) {
                for &s in succs {
                    let d = in_deg.entry(s).or_insert(1);
                    if *d > 0 {
                        *d -= 1;
                    }
                    if *d == 0 {
                        queue.push_back(s);
                    }
                }
            }
        }
        if order.len() == self.nodes.len() {
            Ok(order)
        } else {
            // Cycle: return nodes not yet visited
            let remaining: Vec<JobId> = self
                .nodes
                .keys()
                .filter(|id| !visited.contains(*id))
                .copied()
                .collect();
            Err(remaining)
        }
    }

    pub fn job_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Rule registry
// ---------------------------------------------------------------------------

/// Unified registry for all scheduling rules.
#[derive(Debug, Default)]
pub struct RuleRegistry {
    cron_rules: Vec<CronRule>,
    calendar_rules: Vec<CalendarRule>,
    chains: HashMap<String, DependencyChain>,
    next_id: u64,
}

impl RuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn add_cron(&mut self, rule: CronRule) {
        self.cron_rules.push(rule);
    }

    pub fn add_calendar(&mut self, rule: CalendarRule) {
        self.calendar_rules.push(rule);
    }

    pub fn add_chain(&mut self, name: impl Into<String>, chain: DependencyChain) {
        self.chains.insert(name.into(), chain);
    }

    pub fn matching_cron(&self, day: DayMask, time: TimeOfDay) -> Vec<u64> {
        self.cron_rules
            .iter()
            .filter(|r| r.matches(day, time))
            .map(|r| r.id)
            .collect()
    }

    pub fn matching_calendar(&self, date: CalendarDate, time: TimeOfDay) -> Vec<u64> {
        self.calendar_rules
            .iter()
            .filter(|r| r.matches(date, time))
            .map(|r| r.id)
            .collect()
    }

    pub fn cron_count(&self) -> usize {
        self.cron_rules.len()
    }

    pub fn calendar_count(&self) -> usize {
        self.calendar_rules.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_of_day_hms_roundtrip() {
        let t = TimeOfDay::hms(14, 30, 15);
        assert_eq!(t.hour(), 14);
        assert_eq!(t.minute(), 30);
        assert_eq!(t.second(), 15);
    }

    #[test]
    fn test_time_of_day_ordering() {
        assert!(TimeOfDay::hms(9, 0, 0) < TimeOfDay::hms(18, 0, 0));
    }

    #[test]
    fn test_day_mask_contains() {
        assert!(DayMask::WEEKDAYS.contains(DayMask::MONDAY));
        assert!(!DayMask::WEEKDAYS.contains(DayMask::SATURDAY));
    }

    #[test]
    fn test_day_mask_union() {
        let combined = DayMask::MONDAY.union(DayMask::TUESDAY);
        assert!(combined.contains(DayMask::MONDAY));
        assert!(combined.contains(DayMask::TUESDAY));
        assert!(!combined.contains(DayMask::WEDNESDAY));
    }

    #[test]
    fn test_cron_rule_matches_weekday_in_window() {
        let rule = CronRule::new(
            1,
            "news",
            DayMask::WEEKDAYS,
            TimeOfDay::hms(18, 0, 0),
            TimeOfDay::hms(19, 0, 0),
        );
        assert!(rule.matches(DayMask::MONDAY, TimeOfDay::hms(18, 30, 0)));
    }

    #[test]
    fn test_cron_rule_no_match_outside_window() {
        let rule = CronRule::new(
            2,
            "news",
            DayMask::WEEKDAYS,
            TimeOfDay::hms(18, 0, 0),
            TimeOfDay::hms(19, 0, 0),
        );
        assert!(!rule.matches(DayMask::MONDAY, TimeOfDay::hms(20, 0, 0)));
    }

    #[test]
    fn test_cron_rule_disabled_no_match() {
        let mut rule = CronRule::new(
            3,
            "r",
            DayMask::ALL,
            TimeOfDay::hms(0, 0, 0),
            TimeOfDay::hms(23, 59, 59),
        );
        rule.disable();
        assert!(!rule.matches(DayMask::MONDAY, TimeOfDay::hms(12, 0, 0)));
        rule.enable();
        assert!(rule.matches(DayMask::MONDAY, TimeOfDay::hms(12, 0, 0)));
    }

    #[test]
    fn test_cron_rule_with_interval() {
        let rule = CronRule::new(
            4,
            "r",
            DayMask::ALL,
            TimeOfDay::hms(0, 0, 0),
            TimeOfDay::hms(23, 59, 59),
        )
        .with_interval(3600);
        assert_eq!(rule.interval_secs, Some(3600));
    }

    #[test]
    fn test_calendar_rule_matches_specific_date() {
        let mut rule = CalendarRule::new(10, "holiday", TimeOfDay::hms(8, 0, 0));
        rule.add_date(CalendarDate::new(2026, 1, 1));
        assert!(rule.matches(CalendarDate::new(2026, 1, 1), TimeOfDay::hms(8, 0, 0)));
        assert!(!rule.matches(CalendarDate::new(2026, 1, 2), TimeOfDay::hms(8, 0, 0)));
    }

    #[test]
    fn test_calendar_rule_date_count() {
        let mut rule = CalendarRule::new(11, "r", TimeOfDay::hms(0, 0, 0));
        rule.add_date(CalendarDate::new(2026, 3, 1));
        rule.add_date(CalendarDate::new(2026, 3, 2));
        assert_eq!(rule.date_count(), 2);
    }

    #[test]
    fn test_dependency_chain_ready_jobs_no_deps() {
        let mut chain = DependencyChain::new();
        chain.add_job(JobNode::new(1, "ingest"));
        chain.add_job(JobNode::new(2, "encode"));
        let ready = chain.ready_jobs();
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_dependency_chain_ready_after_dep_succeeded() {
        let mut chain = DependencyChain::new();
        chain.add_job(JobNode::new(1, "ingest"));
        chain.add_job(JobNode::new(2, "encode").with_dep(1));
        // Before job 1 succeeds, job 2 is not ready
        let ready = chain.ready_jobs();
        assert!(!ready.contains(&2));
        chain.mark_succeeded(1);
        let ready = chain.ready_jobs();
        assert!(ready.contains(&2));
    }

    #[test]
    fn test_dependency_chain_topo_sort_linear() {
        let mut chain = DependencyChain::new();
        chain.add_job(JobNode::new(1, "a"));
        chain.add_job(JobNode::new(2, "b").with_dep(1));
        chain.add_job(JobNode::new(3, "c").with_dep(2));
        let order = chain.topological_order().expect("no cycle");
        // 1 must come before 2, 2 before 3
        let pos: HashMap<_, _> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        assert!(pos[&1] < pos[&2]);
        assert!(pos[&2] < pos[&3]);
    }

    #[test]
    fn test_rule_registry_cron_lookup() {
        let mut reg = RuleRegistry::new();
        let id = reg.next_id();
        reg.add_cron(CronRule::new(
            id,
            "news",
            DayMask::WEEKDAYS,
            TimeOfDay::hms(18, 0, 0),
            TimeOfDay::hms(19, 0, 0),
        ));
        let hits = reg.matching_cron(DayMask::MONDAY, TimeOfDay::hms(18, 30, 0));
        assert_eq!(hits, vec![id]);
        assert_eq!(reg.cron_count(), 1);
    }
}
