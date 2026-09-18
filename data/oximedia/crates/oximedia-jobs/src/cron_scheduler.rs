// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Cron-expression parser and scheduler for recurring job execution.
//!
//! Supports the standard 5-field POSIX cron syntax:
//!
//! ```text
//! ┌──────── minute        (0–59)
//! │ ┌────── hour          (0–23)
//! │ │ ┌──── day-of-month  (1–31)
//! │ │ │ ┌── month         (1–12)
//! │ │ │ │ ┌ day-of-week   (0–7, 0 and 7 both mean Sunday)
//! │ │ │ │ │
//! * * * * *
//! ```
//!
//! Supported field syntax:
//! - `*` — any value
//! - `n` — exact value
//! - `n-m` — inclusive range
//! - `*/s` — every `s` steps starting from the minimum
//! - `n-m/s` — every `s` steps within range `[n, m]`
//! - `a,b,c` — comma-separated list of the above

use chrono::{DateTime, Datelike, Duration, TimeZone, Timelike, Utc};
use std::collections::BTreeSet;
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

use crate::job_template::JobTemplate;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by cron parsing and scheduling.
#[derive(Debug, Error, PartialEq, Clone)]
pub enum CronError {
    /// The cron expression does not have exactly 5 fields.
    #[error("Expected 5 fields, got {0}")]
    WrongFieldCount(usize),
    /// A field contained an unrecognised token.
    #[error("Invalid cron field '{field}': {reason}")]
    InvalidField {
        /// The raw field string that caused the error.
        field: String,
        /// Human-readable explanation.
        reason: String,
    },
    /// The step value is zero (division by zero).
    #[error("Step value must be > 0 in field '{0}'")]
    ZeroStep(String),
    /// The scheduled job was not found.
    #[error("Cron job not found: {0}")]
    JobNotFound(Uuid),
}

// ─────────────────────────────────────────────────────────────────────────────
// CronField
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed, expanded cron field represented as the sorted set of allowed values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronField {
    /// The set of concrete values (e.g. minutes, hours, …) that match.
    pub values: BTreeSet<u32>,
}

impl CronField {
    /// Returns `true` if `v` is allowed by this field.
    pub fn matches(&self, v: u32) -> bool {
        self.values.contains(&v)
    }

    /// The smallest value in the field.
    pub fn min(&self) -> Option<u32> {
        self.values.iter().copied().next()
    }

    /// The next value ≥ `from`, wrapping around when none exists.
    ///
    /// Returns `(value, carry)` where `carry` is `true` when wrapping occurred.
    pub fn next_from(&self, from: u32) -> Option<(u32, bool)> {
        if let Some(&v) = self.values.range(from..).next() {
            return Some((v, false));
        }
        // wrap
        self.values.iter().next().map(|&v| (v, true))
    }
}

/// Parse a single cron field.
///
/// `field_str` is the raw string (e.g. `"*/5"`, `"1-5"`, `"3,7,9"`).
/// `min` and `max` are the inclusive bounds for the field.
fn parse_field(field_str: &str, min: u32, max: u32) -> Result<CronField, CronError> {
    let mut values = BTreeSet::new();

    for part in field_str.split(',') {
        parse_part(part.trim(), min, max, &mut values)?;
    }

    Ok(CronField { values })
}

fn parse_part(part: &str, min: u32, max: u32, out: &mut BTreeSet<u32>) -> Result<(), CronError> {
    let make_err = |reason: &str| CronError::InvalidField {
        field: part.to_string(),
        reason: reason.to_string(),
    };

    // Split off optional step: "range/step"
    let (range_part, step) = if let Some(slash) = part.find('/') {
        let step_str = &part[slash + 1..];
        let step: u32 = step_str
            .parse()
            .map_err(|_| make_err("step is not a number"))?;
        if step == 0 {
            return Err(CronError::ZeroStep(part.to_string()));
        }
        (&part[..slash], step)
    } else {
        (part, 1u32)
    };

    // Determine the range [lo, hi]
    let (lo, hi) = if range_part == "*" {
        (min, max)
    } else if let Some(dash) = range_part.find('-') {
        let lo: u32 = range_part[..dash]
            .parse()
            .map_err(|_| make_err("range start is not a number"))?;
        let hi: u32 = range_part[dash + 1..]
            .parse()
            .map_err(|_| make_err("range end is not a number"))?;
        if lo > hi {
            return Err(make_err("range start > range end"));
        }
        if lo < min || hi > max {
            return Err(make_err(&format!("values must be in [{min}, {max}]")));
        }
        (lo, hi)
    } else {
        // single value
        let v: u32 = range_part.parse().map_err(|_| make_err("not a number"))?;
        if v < min || v > max {
            return Err(make_err(&format!("value {v} is outside [{min}, {max}]")));
        }
        out.insert(v);
        return Ok(());
    };

    // Expand the range with step
    let mut v = lo;
    while v <= hi {
        out.insert(v);
        v = match v.checked_add(step) {
            Some(next) => next,
            None => break,
        };
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// CronExpression
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed 5-field cron expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronExpression {
    /// Minute field (0–59).
    pub minute: CronField,
    /// Hour field (0–23).
    pub hour: CronField,
    /// Day-of-month field (1–31).
    pub dom: CronField,
    /// Month field (1–12).
    pub month: CronField,
    /// Day-of-week field (0–6, Sunday = 0).
    pub dow: CronField,
    /// Original expression string, kept for display.
    raw: String,
}

impl CronExpression {
    /// Parse a 5-field cron expression string.
    ///
    /// # Errors
    ///
    /// Returns a [`CronError`] for any syntax or range problem.
    pub fn parse(expr: &str) -> Result<Self, CronError> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(CronError::WrongFieldCount(fields.len()));
        }

        let minute = parse_field(fields[0], 0, 59)?;
        let hour = parse_field(fields[1], 0, 23)?;
        let dom = parse_field(fields[2], 1, 31)?;
        let month = parse_field(fields[3], 1, 12)?;

        // day-of-week: 0 and 7 both mean Sunday; normalise 7 → 0
        let mut dow = parse_field(fields[4], 0, 7)?;
        if dow.values.remove(&7) {
            dow.values.insert(0);
        }

        Ok(Self {
            minute,
            hour,
            dom,
            month,
            dow,
            raw: expr.to_string(),
        })
    }

    /// Check whether `dt` matches this expression (all 5 fields match).
    pub fn matches(&self, dt: &DateTime<Utc>) -> bool {
        let dow_val = dt.weekday().num_days_from_sunday(); // 0=Sun, 6=Sat
        self.minute.matches(dt.minute())
            && self.hour.matches(dt.hour())
            && self.dom.matches(dt.day())
            && self.month.matches(dt.month())
            && self.dow.matches(dow_val)
    }

    /// Return the raw expression string.
    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// next_trigger
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the next trigger time for `expr` strictly after `from`.
///
/// Searches up to 4 years ahead (leap year cycles).  Returns `None` if no
/// matching time exists within that window (only possible with contradictory
/// expressions like `"30 2 29 2 1"` on a non-leap year or similar).
pub fn next_trigger(expr: &CronExpression, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
    // Start from the next minute.
    let start = from
        .with_second(0)
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(from)
        + Duration::minutes(1);

    // Upper bound: 4 years + 1 day
    let limit = from + Duration::days(365 * 4 + 2);

    let mut cur = start;
    while cur <= limit {
        // Fast-forward month
        if !expr.month.matches(cur.month()) {
            // Jump to the first day of the next matching month.
            cur = advance_to_next_month(expr, cur)?;
            continue;
        }

        // Fast-forward day-of-month and day-of-week together
        let dom_ok = expr.dom.matches(cur.day());
        let dow_val = cur.weekday().num_days_from_sunday();
        let dow_ok = expr.dow.matches(dow_val);

        if !dom_ok || !dow_ok {
            // Advance by one day, reset time to midnight.
            cur = Utc
                .with_ymd_and_hms(cur.year(), cur.month(), cur.day(), 0, 0, 0)
                .single()?
                + Duration::days(1);
            continue;
        }

        // Fast-forward hour
        if !expr.hour.matches(cur.hour()) {
            let (next_hour, carry) = expr.hour.next_from(cur.hour())?;
            if carry {
                // No matching hour today; go to next day.
                cur = Utc
                    .with_ymd_and_hms(cur.year(), cur.month(), cur.day(), 0, 0, 0)
                    .single()?
                    + Duration::days(1);
            } else {
                cur = Utc
                    .with_ymd_and_hms(cur.year(), cur.month(), cur.day(), next_hour, 0, 0)
                    .single()?;
            }
            continue;
        }

        // Fast-forward minute
        if !expr.minute.matches(cur.minute()) {
            let (next_min, carry) = expr.minute.next_from(cur.minute())?;
            if carry {
                // Roll to next hour.
                cur = Utc
                    .with_ymd_and_hms(cur.year(), cur.month(), cur.day(), cur.hour(), 0, 0)
                    .single()?
                    + Duration::hours(1);
            } else {
                cur = Utc
                    .with_ymd_and_hms(cur.year(), cur.month(), cur.day(), cur.hour(), next_min, 0)
                    .single()?;
            }
            continue;
        }

        // All fields match.
        return Some(cur);
    }

    None
}

/// Advance `cur` to the first day of the next month that is in `expr.month`,
/// resetting time to 00:00.  Returns `None` if that would exceed the 4-year
/// search window (caller already handles the limit check, but we guard here
/// too to avoid infinite loops).
fn advance_to_next_month(expr: &CronExpression, cur: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let mut year = cur.year();
    let mut month = cur.month() + 1;

    // Walk forward at most 12 months.
    for _ in 0..13 {
        if month > 12 {
            month = 1;
            year += 1;
        }
        if expr.month.matches(month) {
            return Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0).single();
        }
        month += 1;
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// CronJobId / CronJobEntry
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a scheduled cron job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CronJobId(Uuid);

impl CronJobId {
    fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Expose the underlying UUID.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl std::fmt::Display for CronJobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A registered cron job entry.
#[derive(Debug)]
pub struct CronJobEntry {
    /// Unique identifier.
    pub id: CronJobId,
    /// The parsed cron expression.
    pub expression: CronExpression,
    /// The job template to instantiate when the cron fires.
    pub template: JobTemplate,
    /// When the entry was registered.
    pub registered_at: DateTime<Utc>,
    /// When the cron was last triggered (if ever).
    pub last_triggered: Option<DateTime<Utc>>,
    /// Total number of times this cron has fired.
    pub trigger_count: u64,
    /// Whether this entry is active.
    pub enabled: bool,
}

impl CronJobEntry {
    fn new(expression: CronExpression, template: JobTemplate) -> Self {
        Self {
            id: CronJobId::new(),
            expression,
            template,
            registered_at: Utc::now(),
            last_triggered: None,
            trigger_count: 0,
            enabled: true,
        }
    }

    /// Compute the next fire time after `from`.
    pub fn next_trigger_after(&self, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
        next_trigger(&self.expression, from)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CronScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory cron scheduler.
///
/// Stores cron job registrations and provides utilities to query which jobs are
/// due to fire at any given point in time.
#[derive(Debug, Default)]
pub struct CronScheduler {
    entries: HashMap<CronJobId, CronJobEntry>,
}

impl CronScheduler {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new cron job.
    ///
    /// # Errors
    ///
    /// Returns [`CronError`] if the expression cannot be parsed.
    pub fn add_job(
        &mut self,
        cron_expr: &str,
        job_template: JobTemplate,
    ) -> Result<CronJobId, CronError> {
        let expression = CronExpression::parse(cron_expr)?;
        let entry = CronJobEntry::new(expression, job_template);
        let id = entry.id;
        self.entries.insert(id, entry);
        Ok(id)
    }

    /// Remove a cron job by ID.
    ///
    /// # Errors
    ///
    /// Returns [`CronError::JobNotFound`] if the ID is unknown.
    pub fn remove_job(&mut self, id: CronJobId) -> Result<CronJobEntry, CronError> {
        self.entries
            .remove(&id)
            .ok_or(CronError::JobNotFound(id.as_uuid()))
    }

    /// Enable or disable a cron job.
    ///
    /// # Errors
    ///
    /// Returns [`CronError::JobNotFound`] if the ID is unknown.
    pub fn set_enabled(&mut self, id: CronJobId, enabled: bool) -> Result<(), CronError> {
        self.entries
            .get_mut(&id)
            .ok_or(CronError::JobNotFound(id.as_uuid()))
            .map(|e| e.enabled = enabled)
    }

    /// Get a reference to a cron entry.
    pub fn get(&self, id: CronJobId) -> Option<&CronJobEntry> {
        self.entries.get(&id)
    }

    /// Return a list of job IDs that should fire at the given `at` datetime.
    ///
    /// Updates `last_triggered` and `trigger_count` for each matched entry.
    pub fn tick(&mut self, at: DateTime<Utc>) -> Vec<CronJobId> {
        let at_truncated = at
            .with_second(0)
            .and_then(|t| t.with_nanosecond(0))
            .unwrap_or(at);

        let mut fired = Vec::new();
        for entry in self.entries.values_mut() {
            if !entry.enabled {
                continue;
            }
            if entry.expression.matches(&at_truncated) {
                entry.last_triggered = Some(at_truncated);
                entry.trigger_count += 1;
                fired.push(entry.id);
            }
        }
        fired
    }

    /// Return the next scheduled fire time for `id` after `from`.
    ///
    /// # Errors
    ///
    /// Returns [`CronError::JobNotFound`] if the ID is unknown.
    pub fn next_trigger_for(
        &self,
        id: CronJobId,
        from: DateTime<Utc>,
    ) -> Result<Option<DateTime<Utc>>, CronError> {
        let entry = self
            .entries
            .get(&id)
            .ok_or(CronError::JobNotFound(id.as_uuid()))?;
        Ok(entry.next_trigger_after(from))
    }

    /// Number of registered cron jobs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no jobs are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn dt(year: i32, month: u32, day: u32, hour: u32, min: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, min, 0)
            .single()
            .expect("test: valid datetime")
    }

    fn sample_template() -> JobTemplate {
        JobTemplate::new("tmpl", "desc", "body {{input}}")
    }

    // ── parse errors ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_wrong_field_count() {
        let result = CronExpression::parse("* * * *");
        assert!(matches!(result, Err(CronError::WrongFieldCount(4))));
    }

    #[test]
    fn test_parse_invalid_value_out_of_range() {
        // minute 60 is out of range [0,59]
        let result = CronExpression::parse("60 * * * *");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_zero_step_error() {
        let result = CronExpression::parse("*/0 * * * *");
        assert!(matches!(result, Err(CronError::ZeroStep(_))));
    }

    // ── basic parsing ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_every_minute() {
        let expr = CronExpression::parse("* * * * *").expect("should parse");
        assert_eq!(expr.minute.values.len(), 60);
        assert_eq!(expr.hour.values.len(), 24);
    }

    #[test]
    fn test_parse_step_five_minutes() {
        let expr = CronExpression::parse("*/5 * * * *").expect("should parse");
        let expected: BTreeSet<u32> = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            .iter()
            .copied()
            .collect();
        assert_eq!(expr.minute.values, expected);
    }

    #[test]
    fn test_parse_range() {
        let expr = CronExpression::parse("0 9-17 * * *").expect("should parse");
        assert!(expr.hour.matches(9));
        assert!(expr.hour.matches(17));
        assert!(!expr.hour.matches(18));
    }

    #[test]
    fn test_parse_comma_list() {
        let expr = CronExpression::parse("0 8,12,20 * * *").expect("should parse");
        assert!(expr.hour.matches(8));
        assert!(expr.hour.matches(12));
        assert!(expr.hour.matches(20));
        assert!(!expr.hour.matches(9));
    }

    #[test]
    fn test_parse_dow_7_normalised_to_0() {
        let expr = CronExpression::parse("0 0 * * 7").expect("should parse");
        // 7 should be normalised to 0 (Sunday)
        assert!(expr.dow.matches(0));
        assert!(!expr.dow.matches(7));
    }

    // ── next_trigger ──────────────────────────────────────────────────────────

    #[test]
    fn test_next_trigger_every_minute() {
        let expr = CronExpression::parse("* * * * *").expect("should parse");
        let from = dt(2024, 6, 1, 12, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2024, 6, 1, 12, 1));
    }

    #[test]
    fn test_next_trigger_every_5_minutes() {
        let expr = CronExpression::parse("*/5 * * * *").expect("should parse");
        let from = dt(2024, 6, 1, 12, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2024, 6, 1, 12, 5));
    }

    #[test]
    fn test_next_trigger_daily_midnight() {
        let expr = CronExpression::parse("0 0 * * *").expect("should parse");
        let from = dt(2024, 6, 1, 12, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2024, 6, 2, 0, 0));
    }

    #[test]
    fn test_next_trigger_end_of_month_rollover() {
        let expr = CronExpression::parse("0 0 1 * *").expect("should parse");
        // From June 15th; next fire is July 1st.
        let from = dt(2024, 6, 15, 0, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2024, 7, 1, 0, 0));
    }

    #[test]
    fn test_next_trigger_end_of_year_rollover() {
        let expr = CronExpression::parse("0 0 1 1 *").expect("should parse");
        // From December 31st; next fire is Jan 1st next year.
        let from = dt(2024, 12, 31, 0, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2025, 1, 1, 0, 0));
    }

    #[test]
    fn test_next_trigger_leap_year_feb29() {
        // 2024 is a leap year.
        let expr = CronExpression::parse("0 0 29 2 *").expect("should parse");
        let from = dt(2024, 1, 1, 0, 0);
        let next = next_trigger(&expr, from).expect("should find next on leap year");
        assert_eq!(next, dt(2024, 2, 29, 0, 0));
    }

    #[test]
    fn test_next_trigger_non_leap_year_skips_feb29() {
        // 2025 is NOT a leap year.  The next Feb 29 is 2028.
        let expr = CronExpression::parse("0 0 29 2 *").expect("should parse");
        let from = dt(2025, 1, 1, 0, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2028, 2, 29, 0, 0));
    }

    #[test]
    fn test_next_trigger_hour_roll() {
        let expr = CronExpression::parse("0 9 * * *").expect("should parse");
        // Already past 09:00 today.
        let from = dt(2024, 6, 1, 10, 0);
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2024, 6, 2, 9, 0));
    }

    #[test]
    fn test_next_trigger_minute_boundary() {
        let expr = CronExpression::parse("30 * * * *").expect("should parse");
        let from = dt(2024, 6, 1, 12, 30); // exactly at :30 — next is :30 of the next hour
        let next = next_trigger(&expr, from).expect("should find next");
        assert_eq!(next, dt(2024, 6, 1, 13, 30));
    }

    // ── CronScheduler ─────────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_add_job() {
        let mut sched = CronScheduler::new();
        let id = sched
            .add_job("*/5 * * * *", sample_template())
            .expect("should add job");
        assert_eq!(sched.len(), 1);
        assert!(sched.get(id).is_some());
    }

    #[test]
    fn test_scheduler_remove_job() {
        let mut sched = CronScheduler::new();
        let id = sched
            .add_job("* * * * *", sample_template())
            .expect("should add job");
        let removed = sched.remove_job(id).expect("should remove");
        assert_eq!(removed.id, id);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_scheduler_tick_fires_matching_jobs() {
        let mut sched = CronScheduler::new();
        let id = sched
            .add_job("0 12 * * *", sample_template())
            .expect("should add job");

        let at = dt(2024, 6, 1, 12, 0);
        let fired = sched.tick(at);
        assert_eq!(fired, vec![id]);

        let entry = sched.get(id).expect("entry should exist");
        assert_eq!(entry.trigger_count, 1);
        assert_eq!(entry.last_triggered, Some(at));
    }

    #[test]
    fn test_scheduler_tick_does_not_fire_disabled_job() {
        let mut sched = CronScheduler::new();
        let id = sched
            .add_job("0 12 * * *", sample_template())
            .expect("should add job");
        sched.set_enabled(id, false).expect("should disable");

        let fired = sched.tick(dt(2024, 6, 1, 12, 0));
        assert!(fired.is_empty());
    }

    #[test]
    fn test_scheduler_next_trigger_for() {
        let mut sched = CronScheduler::new();
        let id = sched
            .add_job("0 0 1 1 *", sample_template())
            .expect("should add job");

        let from = dt(2024, 6, 1, 0, 0);
        let next = sched
            .next_trigger_for(id, from)
            .expect("should compute next")
            .expect("should have a next time");
        assert_eq!(next, dt(2025, 1, 1, 0, 0));
    }
}
