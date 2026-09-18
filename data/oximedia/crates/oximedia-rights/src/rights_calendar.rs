//! Calendar-based views of expiring rights windows.
//!
//! Organises rights records by their expiry date into a structured calendar
//! view.  Callers can query expiring rights for:
//!
//! - A specific day (expressed as a Unix-day number: `ts / 86400`)
//! - A specific calendar month
//! - An arbitrary date range
//! - Configurable look-ahead windows (e.g. "expiring in the next 30 days")
//!
//! All timestamps are Unix seconds; calendar conversion is done by integer
//! division (no timezone support: UTC assumed).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── CalendarEntry ─────────────────────────────────────────────────────────────

/// A single entry in the rights calendar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEntry {
    /// Rights record identifier.
    pub record_id: String,
    /// Asset this record applies to.
    pub asset_id: String,
    /// Rights holder name or ID.
    pub holder: String,
    /// Expiry timestamp (Unix seconds). Records without expiry are not included.
    pub expires_at: u64,
    /// Unix day number on which the right expires (`expires_at / 86400`).
    pub expiry_day: u64,
    /// Whether the record is currently active.
    pub active: bool,
    /// Human-readable notes.
    pub notes: String,
}

impl CalendarEntry {
    /// Days remaining until expiry, given a current timestamp.
    /// Returns `0` if already expired.
    #[must_use]
    pub fn days_remaining(&self, now: u64) -> u64 {
        if now >= self.expires_at {
            0
        } else {
            (self.expires_at - now) / 86_400
        }
    }

    /// Whether this entry has already expired at `now`.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

// ── MonthView ─────────────────────────────────────────────────────────────────

/// Calendar entries grouped by day-of-month for a single month.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthView {
    /// Year (e.g. 2025).
    pub year: i32,
    /// Month (1–12).
    pub month: u8,
    /// Entries grouped by day of month (1–31). Days without entries are absent.
    pub by_day: HashMap<u8, Vec<CalendarEntry>>,
    /// Total entries in this month.
    pub total: usize,
}

impl MonthView {
    /// Entries for a specific day of the month (1-based).
    #[must_use]
    pub fn entries_for_day(&self, day: u8) -> &[CalendarEntry] {
        self.by_day.get(&day).map(Vec::as_slice).unwrap_or(&[])
    }

    /// All days in this month that have at least one expiring right.
    #[must_use]
    pub fn days_with_expirations(&self) -> Vec<u8> {
        let mut days: Vec<u8> = self.by_day.keys().copied().collect();
        days.sort_unstable();
        days
    }
}

// ── LookaheadSummary ──────────────────────────────────────────────────────────

/// Summary of rights expiring within a look-ahead window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookaheadSummary {
    /// Start of the window (Unix seconds).
    pub window_start: u64,
    /// End of the window (Unix seconds, exclusive).
    pub window_end: u64,
    /// Total rights expiring in the window.
    pub expiring_count: usize,
    /// Count of currently-active rights expiring in the window.
    pub active_expiring_count: usize,
    /// Entries expiring in the window, sorted by expiry ascending.
    pub entries: Vec<CalendarEntry>,
}

// ── RightsCalendar ────────────────────────────────────────────────────────────

/// Calendar-based view of expiring rights records.
///
/// Build the calendar by calling [`add_record`](RightsCalendar::add_record)
/// for each rights record that has an expiry date, then query using the
/// provided methods.
#[derive(Debug, Default)]
pub struct RightsCalendar {
    /// All entries, keyed by record_id.
    entries: HashMap<String, CalendarEntry>,
}

impl RightsCalendar {
    /// Create an empty calendar.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rights record to the calendar.
    ///
    /// Records without an expiry timestamp are silently ignored.
    pub fn add_record(
        &mut self,
        record_id: impl Into<String>,
        asset_id: impl Into<String>,
        holder: impl Into<String>,
        expires_at: Option<u64>,
        active: bool,
        notes: impl Into<String>,
    ) {
        let Some(expires_at) = expires_at else { return };
        let record_id = record_id.into();
        let entry = CalendarEntry {
            record_id: record_id.clone(),
            asset_id: asset_id.into(),
            holder: holder.into(),
            expires_at,
            expiry_day: expires_at / 86_400,
            active,
            notes: notes.into(),
        };
        self.entries.insert(record_id, entry);
    }

    /// Remove a record from the calendar.
    pub fn remove_record(&mut self, record_id: &str) -> Option<CalendarEntry> {
        self.entries.remove(record_id)
    }

    /// Total number of entries in the calendar.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    // ── Day queries ───────────────────────────────────────────────────────────

    /// All entries expiring on a specific Unix-day (`ts / 86400`).
    #[must_use]
    pub fn expiring_on_day(&self, unix_day: u64) -> Vec<&CalendarEntry> {
        self.entries
            .values()
            .filter(|e| e.expiry_day == unix_day)
            .collect()
    }

    /// All entries expiring between `start` and `end` (both Unix seconds,
    /// inclusive start, exclusive end).
    #[must_use]
    pub fn expiring_in_range(&self, start: u64, end: u64) -> Vec<&CalendarEntry> {
        let mut v: Vec<&CalendarEntry> = self
            .entries
            .values()
            .filter(|e| e.expires_at >= start && e.expires_at < end)
            .collect();
        v.sort_by_key(|e| e.expires_at);
        v
    }

    // ── Look-ahead ────────────────────────────────────────────────────────────

    /// Summary of rights expiring within `days` days from `now`.
    #[must_use]
    pub fn lookahead(&self, now: u64, days: u64) -> LookaheadSummary {
        let window_start = now;
        let window_end = now + days * 86_400;

        let mut entries: Vec<CalendarEntry> = self
            .entries
            .values()
            .filter(|e| e.expires_at >= window_start && e.expires_at < window_end)
            .cloned()
            .collect();
        entries.sort_by_key(|e| e.expires_at);

        let active_expiring_count = entries.iter().filter(|e| e.active).count();
        let expiring_count = entries.len();

        LookaheadSummary {
            window_start,
            window_end,
            expiring_count,
            active_expiring_count,
            entries,
        }
    }

    // ── Already-expired ───────────────────────────────────────────────────────

    /// All entries that have already expired at `now`.
    #[must_use]
    pub fn already_expired(&self, now: u64) -> Vec<&CalendarEntry> {
        let mut v: Vec<&CalendarEntry> = self
            .entries
            .values()
            .filter(|e| e.is_expired(now))
            .collect();
        v.sort_by_key(|e| e.expires_at);
        v
    }

    // ── Month view ────────────────────────────────────────────────────────────

    /// Build a [`MonthView`] for the given year and month (1–12).
    ///
    /// Groups all entries whose expiry falls within the specified month.
    #[must_use]
    pub fn month_view(&self, year: i32, month: u8) -> MonthView {
        let (start_ts, end_ts) = month_range(year, month);
        let mut by_day: HashMap<u8, Vec<CalendarEntry>> = HashMap::new();

        for entry in self.entries.values() {
            if entry.expires_at >= start_ts && entry.expires_at < end_ts {
                let day_of_month = day_of_month(entry.expires_at, year, month);
                by_day.entry(day_of_month).or_default().push(entry.clone());
            }
        }

        // Sort each day's entries by expiry ascending.
        for day_entries in by_day.values_mut() {
            day_entries.sort_by_key(|e| e.expires_at);
        }

        let total = by_day.values().map(|v| v.len()).sum();

        MonthView {
            year,
            month,
            by_day,
            total,
        }
    }

    // ── Asset-specific ────────────────────────────────────────────────────────

    /// All calendar entries for a specific asset, sorted by expiry.
    #[must_use]
    pub fn entries_for_asset(&self, asset_id: &str) -> Vec<&CalendarEntry> {
        let mut v: Vec<&CalendarEntry> = self
            .entries
            .values()
            .filter(|e| e.asset_id == asset_id)
            .collect();
        v.sort_by_key(|e| e.expires_at);
        v
    }
}

// ── Calendar arithmetic helpers ───────────────────────────────────────────────

/// Number of days in a month (no leap-year correction for Feb in this impl;
/// February is treated as 28 days for simplicity since we need only rough
/// ranges for display purposes).
fn days_in_month(month: u8) -> u64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => 28, // conservative; use 28 to avoid crossing month boundary
        _ => 30,
    }
}

/// Compute a (very approximate) Unix timestamp for the first second of a
/// calendar month.  Uses a simplified formula (no DST, no timezone).
fn month_start_ts(year: i32, month: u8) -> u64 {
    // Compute days from Unix epoch (1970-01-01) to the first of `month/year`.
    // Using the standard formula for Gregorian calendar day numbers.
    let y = if month < 3 { year - 1 } else { year } as i64;
    let m = if month < 3 {
        month as i64 + 12
    } else {
        month as i64
    };
    // Julian Day Number of 1970-01-01 is 2440588
    let a = y / 100;
    let b = 2 - a + a / 4;
    let jdn = (365.25_f64 * (y + 4716) as f64) as i64
        + (30.6001_f64 * (m + 1) as f64) as i64
        + 1  // day of month = 1
        + b
        - 1524;
    let unix_day = jdn - 2440588; // days since Unix epoch
    (unix_day.max(0) as u64) * 86_400
}

/// Compute the Unix timestamp range `[start, end)` for a calendar month.
fn month_range(year: i32, month: u8) -> (u64, u64) {
    let start = month_start_ts(year, month);
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };
    let end = month_start_ts(next_year, next_month);
    (start, end)
}

/// Compute the day-of-month (1-based) of a Unix timestamp within a given
/// year/month by subtracting the month start and dividing by 86400.
fn day_of_month(ts: u64, year: i32, month: u8) -> u8 {
    let start = month_start_ts(year, month);
    let offset_days = (ts.saturating_sub(start)) / 86_400;
    // Days are 1-based and max 31
    ((offset_days % 31) + 1).min(31) as u8
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Base "now" for all day-scale tests: 30 days into the epoch
    const BASE_NOW: u64 = 30 * 86_400;

    fn build_calendar() -> RightsCalendar {
        let mut cal = RightsCalendar::new();
        // expires in 10 days from BASE_NOW
        cal.add_record(
            "r1",
            "asset-A",
            "Alice",
            Some(BASE_NOW + 10 * 86_400),
            true,
            "A",
        );
        // expires in 20 days from BASE_NOW
        cal.add_record(
            "r2",
            "asset-B",
            "Bob",
            Some(BASE_NOW + 20 * 86_400),
            true,
            "B",
        );
        // expired 5 days before BASE_NOW
        cal.add_record(
            "r3",
            "asset-C",
            "Carol",
            Some(BASE_NOW - 5 * 86_400),
            false,
            "C",
        );
        // no expiry – should be ignored
        cal.add_record("r4", "asset-D", "Dave", None, true, "D");
        cal
    }

    #[test]
    fn test_entry_count() {
        // r4 has no expiry → only 3 entries
        assert_eq!(build_calendar().entry_count(), 3);
    }

    #[test]
    fn test_expiring_in_range() {
        let cal = build_calendar();
        // r1 expires at BASE_NOW + 10 days; search range covers it
        let range = cal.expiring_in_range(BASE_NOW + 5 * 86_400, BASE_NOW + 15 * 86_400);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].record_id, "r1");
    }

    #[test]
    fn test_expiring_in_range_all() {
        let cal = build_calendar();
        // Full range from before the earliest to after the latest
        let range = cal.expiring_in_range(0, BASE_NOW + 25 * 86_400);
        assert_eq!(range.len(), 3);
    }

    #[test]
    fn test_already_expired() {
        let cal = build_calendar();
        // Only r3 (expires 5 days before BASE_NOW) is expired at BASE_NOW
        let expired = cal.already_expired(BASE_NOW);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].record_id, "r3");
    }

    #[test]
    fn test_lookahead_7_days() {
        let cal = build_calendar();
        // r1 expires 10 days from BASE_NOW → not in 7-day window
        let summary = cal.lookahead(BASE_NOW, 7);
        assert_eq!(summary.expiring_count, 0);
    }

    #[test]
    fn test_lookahead_15_days() {
        let cal = build_calendar();
        // r1 expires 10 days from BASE_NOW → in 15-day window
        let summary = cal.lookahead(BASE_NOW, 15);
        assert_eq!(summary.expiring_count, 1);
        assert_eq!(summary.active_expiring_count, 1);
    }

    #[test]
    fn test_lookahead_25_days() {
        let cal = build_calendar();
        // r1 (10 days) and r2 (20 days) both in 25-day window
        let summary = cal.lookahead(BASE_NOW, 25);
        assert_eq!(summary.expiring_count, 2);
        // r3 already expired before BASE_NOW → not in lookahead
        assert_eq!(summary.active_expiring_count, 2);
    }

    #[test]
    fn test_lookahead_sorted_by_expiry() {
        let cal = build_calendar();
        let summary = cal.lookahead(BASE_NOW, 30);
        let expiries: Vec<u64> = summary.entries.iter().map(|e| e.expires_at).collect();
        for pair in expiries.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn test_entries_for_asset() {
        let cal = build_calendar();
        assert_eq!(cal.entries_for_asset("asset-A").len(), 1);
        assert_eq!(cal.entries_for_asset("asset-Z").len(), 0);
    }

    #[test]
    fn test_days_remaining() {
        let cal = build_calendar();
        let entry = cal.entries.get("r1").expect("r1 should be in calendar");
        let expiry = BASE_NOW + 10 * 86_400;
        // 10 days remaining from BASE_NOW
        assert_eq!(entry.days_remaining(BASE_NOW), 10);
        assert_eq!(entry.days_remaining(BASE_NOW + 9 * 86_400), 1);
        // at expiry → 0
        assert_eq!(entry.days_remaining(expiry), 0);
    }

    #[test]
    fn test_is_expired() {
        let cal = build_calendar();
        let entry = cal.entries.get("r1").expect("r1");
        assert!(!entry.is_expired(BASE_NOW));
        assert!(entry.is_expired(BASE_NOW + 10 * 86_400));
    }

    #[test]
    fn test_remove_record() {
        let mut cal = build_calendar();
        let removed = cal.remove_record("r1");
        assert!(removed.is_some());
        assert_eq!(cal.entry_count(), 2);
    }

    #[test]
    fn test_expiring_on_day() {
        let cal = build_calendar();
        // r1 expires at BASE_NOW + 10 days; compute its unix_day
        let r1_expiry_day = (BASE_NOW + 10 * 86_400) / 86_400;
        let on_day = cal.expiring_on_day(r1_expiry_day);
        assert_eq!(on_day.len(), 1);
        assert_eq!(on_day[0].record_id, "r1");
    }

    #[test]
    fn test_month_view_total() {
        // Use a timestamp we can reason about: Jan 2025
        // 2025-01-01 00:00:00 UTC ≈ 1735689600
        let jan_2025_ts: u64 = 1_735_689_600;

        let mut cal = RightsCalendar::new();
        // expires mid-January 2025
        cal.add_record("m1", "a", "h", Some(jan_2025_ts + 14 * 86_400), true, "");
        // expires end-January 2025
        cal.add_record("m2", "b", "h", Some(jan_2025_ts + 28 * 86_400), true, "");
        // expires in February (should NOT appear in January view)
        cal.add_record("m3", "c", "h", Some(jan_2025_ts + 40 * 86_400), true, "");

        let view = cal.month_view(2025, 1);
        assert_eq!(view.total, 2);
        assert_eq!(view.year, 2025);
        assert_eq!(view.month, 1);
    }

    #[test]
    fn test_month_view_days_with_expirations_sorted() {
        let jan_2025_ts: u64 = 1_735_689_600;
        let mut cal = RightsCalendar::new();
        cal.add_record("x", "a", "h", Some(jan_2025_ts + 5 * 86_400), true, "");
        cal.add_record("y", "b", "h", Some(jan_2025_ts + 20 * 86_400), true, "");
        let view = cal.month_view(2025, 1);
        let days = view.days_with_expirations();
        for pair in days.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }
}
