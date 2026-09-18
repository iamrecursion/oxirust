//! Cohort analysis: group viewers by their first-view date and track
//! how each cohort's engagement evolves over subsequent time windows.
//!
//! ## Model
//!
//! A **cohort** is a set of viewers who started watching a content item (or any
//! content) for the first time within the same *cohort window* (e.g. the same
//! day or week).  For each cohort we then measure retention — what fraction of
//! the original cohort was still viewing in each subsequent period.
//!
//! Time is expressed as Unix epoch milliseconds throughout.

use std::collections::HashMap;

use crate::error::AnalyticsError;

// ─── Time window helpers ──────────────────────────────────────────────────────

/// Granularity used to bucket viewer first-view dates into cohorts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CohortWindow {
    /// Group viewers who first watched on the same calendar day.
    Day,
    /// Group viewers who first watched in the same ISO week (Monday-start).
    Week,
    /// Group viewers who first watched in the same calendar month.
    Month,
}

impl CohortWindow {
    /// Truncate a Unix epoch millisecond timestamp to the start of its
    /// cohort window (also in epoch ms).
    pub fn truncate_ms(&self, epoch_ms: i64) -> i64 {
        let secs = epoch_ms.div_euclid(1_000);
        match self {
            CohortWindow::Day => {
                let day_secs = 86_400i64;
                secs.div_euclid(day_secs) * day_secs * 1_000
            }
            CohortWindow::Week => {
                // Unix epoch (1970-01-01) was a Thursday; add 3 days to shift
                // epoch to Monday before bucketing.
                let day_secs = 86_400i64;
                let week_secs = 7 * day_secs;
                let shifted = secs + 3 * day_secs; // shift so Mon = 0
                let week_start = shifted.div_euclid(week_secs) * week_secs - 3 * day_secs;
                week_start * 1_000
            }
            CohortWindow::Month => {
                // Approximate: group by (year * 12 + month).
                // We reconstruct year/month from days since epoch.
                let days_since_epoch = secs.div_euclid(86_400);
                let (year, month) = days_to_year_month(days_since_epoch);
                // Return the first day of that month as epoch ms.
                year_month_to_epoch_ms(year, month)
            }
        }
    }
}

/// Compute (year, 1-indexed month) from days since Unix epoch.
fn days_to_year_month(days: i64) -> (i32, u32) {
    // Gregorian proleptic algorithm (valid for positive days).
    // Reference: https://www.researchgate.net/publication/316558298
    let z = days + 719_468;
    let era = z.div_euclid(146_097) as i32;
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_adj = if m <= 2 { y + 1 } else { y };
    (y_adj, m as u32)
}

/// Return epoch milliseconds for the first day of the given year and month.
fn year_month_to_epoch_ms(year: i32, month: u32) -> i64 {
    // Days from epoch to 1 Jan of the given year, then to the first of the month.
    let days = days_from_epoch(year, month, 1);
    days * 86_400_000
}

/// Days from the Unix epoch (1970-01-01) to the given date.
fn days_from_epoch(year: i32, month: u32, day: u32) -> i64 {
    // Rata Die algorithm (valid for Gregorian calendar).
    let y = if month <= 2 {
        year as i64 - 1
    } else {
        year as i64
    };
    let m = if month <= 2 { month + 12 } else { month } as i64;
    let d = day as i64;
    365 * y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400) + (153 * m + 8) / 5 + d
        - 719_469
}

// ─── Cohort model ─────────────────────────────────────────────────────────────

/// A single cohort: viewers who first engaged during the same time window.
#[derive(Debug, Clone)]
pub struct Cohort {
    /// Epoch-ms timestamp representing the *start* of this cohort window.
    pub window_start_ms: i64,
    /// Unique viewer IDs in this cohort.
    pub viewer_ids: Vec<String>,
}

impl Cohort {
    pub fn size(&self) -> usize {
        self.viewer_ids.len()
    }
}

/// One cell in the cohort retention matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortRetentionCell {
    /// Cohort window start (epoch ms).
    pub cohort_window_ms: i64,
    /// Periods-after-first-view (0 = same period as first view).
    pub period_offset: u32,
    /// Number of viewers from the cohort active in this period.
    pub active_viewers: u32,
    /// Fraction of the cohort still active (0.0–1.0).
    pub retention_rate: f32,
}

/// Full cohort-retention matrix for a set of viewer events.
#[derive(Debug, Clone)]
pub struct CohortMatrix {
    pub window: CohortWindow,
    /// Ordered list of cohorts (sorted by window_start_ms ascending).
    pub cohorts: Vec<Cohort>,
    /// Retention matrix cells: one row per cohort, columns are period offsets.
    pub cells: Vec<CohortRetentionCell>,
    /// Number of periods tracked after the first-view period.
    pub num_periods: u32,
}

impl CohortMatrix {
    /// Retrieve the retention rate for a specific cohort + period offset.
    ///
    /// Returns `None` if no matching cell exists.
    pub fn retention_at(&self, cohort_window_ms: i64, period_offset: u32) -> Option<f32> {
        self.cells
            .iter()
            .find(|c| c.cohort_window_ms == cohort_window_ms && c.period_offset == period_offset)
            .map(|c| c.retention_rate)
    }

    /// Return the average cohort retention at a given period offset across all
    /// cohorts (weighted by cohort size).
    pub fn average_retention_at_period(&self, period_offset: u32) -> f32 {
        let relevant: Vec<_> = self
            .cells
            .iter()
            .filter(|c| c.period_offset == period_offset)
            .collect();
        if relevant.is_empty() {
            return 0.0;
        }
        let total_cohort_size: u32 = relevant
            .iter()
            .filter_map(|c| {
                self.cohorts
                    .iter()
                    .find(|cohort| cohort.window_start_ms == c.cohort_window_ms)
                    .map(|cohort| cohort.size() as u32)
            })
            .sum();

        if total_cohort_size == 0 {
            return 0.0;
        }

        let weighted_sum: f32 = relevant
            .iter()
            .filter_map(|c| {
                self.cohorts
                    .iter()
                    .find(|cohort| cohort.window_start_ms == c.cohort_window_ms)
                    .map(|cohort| c.retention_rate * cohort.size() as f32)
            })
            .sum();

        weighted_sum / total_cohort_size as f32
    }
}

// ─── Viewer event model ───────────────────────────────────────────────────────

/// A minimal viewer activity record used as input for cohort analysis.
///
/// Each record says "viewer `viewer_id` was active at `event_ms`".
#[derive(Debug, Clone)]
pub struct ViewerEvent {
    pub viewer_id: String,
    /// Unix epoch millisecond timestamp of the activity.
    pub event_ms: i64,
}

// ─── Core function ────────────────────────────────────────────────────────────

/// Build a cohort-retention matrix from a stream of viewer events.
///
/// # Arguments
///
/// * `events`       — all viewer activity records (any order).
/// * `window`       — the time granularity used to group viewers into cohorts.
/// * `num_periods`  — how many periods after the first-view period to track
///   (e.g. 4 weeks for a weekly cohort with 4 follow-up periods).
///
/// # Algorithm
///
/// 1. Determine each viewer's *first-event period* (their cohort).
/// 2. For each cohort, determine which viewers were active in each subsequent
///    period (period offset 0 = same period as first view, offset 1 = next
///    period, etc.).
/// 3. Compute retention as `active_viewers / cohort_size`.
///
/// Returns an error if `events` is empty.
pub fn build_cohort_matrix(
    events: &[ViewerEvent],
    window: CohortWindow,
    num_periods: u32,
) -> Result<CohortMatrix, AnalyticsError> {
    if events.is_empty() {
        return Err(AnalyticsError::InsufficientData(
            "cannot build cohort matrix from empty event stream".to_string(),
        ));
    }

    // Step 1: Determine the first-event period for each viewer.
    let mut first_period_by_viewer: HashMap<&str, i64> = HashMap::new();
    for event in events {
        let period = window.truncate_ms(event.event_ms);
        let entry = first_period_by_viewer
            .entry(event.viewer_id.as_str())
            .or_insert(period);
        if period < *entry {
            *entry = period;
        }
    }

    // Step 2: Group viewers by their cohort period.
    let mut cohort_map: HashMap<i64, Vec<String>> = HashMap::new();
    for (viewer_id, first_period) in &first_period_by_viewer {
        cohort_map
            .entry(*first_period)
            .or_default()
            .push(viewer_id.to_string());
    }

    // Step 3: For each event, record which period offset it falls in relative to
    //         the viewer's cohort.
    // Build a map: (cohort_period, period_offset) → set of active viewers.
    let mut activity_map: HashMap<(i64, u32), std::collections::HashSet<String>> = HashMap::new();
    for event in events {
        let cohort_period = match first_period_by_viewer.get(event.viewer_id.as_str()) {
            Some(&p) => p,
            None => continue,
        };
        let event_period = window.truncate_ms(event.event_ms);
        // Calculate the offset in period units.
        let offset = period_offset(cohort_period, event_period, window);
        if offset <= num_periods {
            activity_map
                .entry((cohort_period, offset))
                .or_default()
                .insert(event.viewer_id.clone());
        }
    }

    // Step 4: Build the result structures.
    let mut cohort_keys: Vec<i64> = cohort_map.keys().copied().collect();
    cohort_keys.sort_unstable();

    let cohorts: Vec<Cohort> = cohort_keys
        .iter()
        .map(|&key| {
            let mut viewer_ids = cohort_map[&key].clone();
            viewer_ids.sort(); // deterministic ordering
            Cohort {
                window_start_ms: key,
                viewer_ids,
            }
        })
        .collect();

    let mut cells = Vec::new();
    for cohort in &cohorts {
        let cohort_size = cohort.size() as f32;
        for period_offset_val in 0..=num_periods {
            let active = activity_map
                .get(&(cohort.window_start_ms, period_offset_val))
                .map(|s| s.len() as u32)
                .unwrap_or(0);
            let retention_rate = if cohort_size > 0.0 {
                active as f32 / cohort_size
            } else {
                0.0
            };
            cells.push(CohortRetentionCell {
                cohort_window_ms: cohort.window_start_ms,
                period_offset: period_offset_val,
                active_viewers: active,
                retention_rate,
            });
        }
    }

    Ok(CohortMatrix {
        window,
        cohorts,
        cells,
        num_periods,
    })
}

/// Compute the integer period offset between two window-start timestamps.
///
/// For `Day` windows this is the number of days; for `Week` it is the number
/// of 7-day intervals; for `Month` it is the number of months.
fn period_offset(cohort_ms: i64, event_ms: i64, window: CohortWindow) -> u32 {
    if event_ms < cohort_ms {
        return 0; // clamp negative offsets (data anomalies)
    }
    let diff_ms = event_ms - cohort_ms;
    match window {
        CohortWindow::Day => {
            let day_ms = 86_400_000i64;
            (diff_ms / day_ms) as u32
        }
        CohortWindow::Week => {
            let week_ms = 7 * 86_400_000i64;
            (diff_ms / week_ms) as u32
        }
        CohortWindow::Month => {
            // Approximate months from the millisecond delta.
            // We count how many full months separate two Month-truncated timestamps.
            let cohort_days = cohort_ms.div_euclid(86_400_000);
            let event_days = event_ms.div_euclid(86_400_000);
            let (cy, cm) = days_to_year_month(cohort_days);
            let (ey, em) = days_to_year_month(event_days);
            let months = (ey - cy) * 12 + em as i32 - cm as i32;
            months.max(0) as u32
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a viewer event.
    fn ev(viewer_id: &str, epoch_ms: i64) -> ViewerEvent {
        ViewerEvent {
            viewer_id: viewer_id.to_string(),
            event_ms: epoch_ms,
        }
    }

    const DAY_MS: i64 = 86_400_000;
    const WEEK_MS: i64 = 7 * DAY_MS;

    // ── CohortWindow::truncate_ms ────────────────────────────────────────────

    #[test]
    fn truncate_ms_day_epoch_start() {
        let w = CohortWindow::Day;
        assert_eq!(w.truncate_ms(0), 0);
        assert_eq!(w.truncate_ms(DAY_MS - 1), 0);
        assert_eq!(w.truncate_ms(DAY_MS), DAY_MS);
    }

    #[test]
    fn truncate_ms_day_mid_day() {
        let w = CohortWindow::Day;
        // 2024-01-15 12:00 UTC = 1_705_320_000_000 ms
        let ts = 1_705_320_000_000i64;
        let truncated = w.truncate_ms(ts);
        // Should round down to start of day.
        assert_eq!(truncated % DAY_MS, 0);
        assert!(truncated <= ts);
        assert!(ts - truncated < DAY_MS);
    }

    #[test]
    fn truncate_ms_week_epoch_is_thursday() {
        // Epoch is 1970-01-01 (Thursday). Our Monday-start week for week 0 starts
        // on 1969-12-29 = -3 days = -259_200_000 ms.
        let w = CohortWindow::Week;
        let week0_start = w.truncate_ms(0);
        // The Monday of the epoch week.
        assert!(week0_start <= 0, "week0_start={week0_start}");
        assert!(
            week0_start > -WEEK_MS,
            "week0_start should be within one week before epoch"
        );
    }

    #[test]
    fn truncate_ms_week_stable_within_week() {
        let w = CohortWindow::Week;
        // All timestamps within the same week should map to the same value.
        // Use a Monday: 2024-01-15 = 1_705_276_800_000 ms.
        let monday = 1_705_276_800_000i64;
        let tuesday = monday + DAY_MS;
        let sunday = monday + 6 * DAY_MS;
        assert_eq!(w.truncate_ms(monday), w.truncate_ms(tuesday));
        assert_eq!(w.truncate_ms(monday), w.truncate_ms(sunday));
        assert_ne!(w.truncate_ms(monday), w.truncate_ms(monday + WEEK_MS));
    }

    #[test]
    fn truncate_ms_month_same_month() {
        let w = CohortWindow::Month;
        // 2024-01-01 and 2024-01-31 should map to the same month.
        let jan1 = 1_704_067_200_000i64; // 2024-01-01 UTC
        let jan31 = jan1 + 30 * DAY_MS;
        assert_eq!(w.truncate_ms(jan1), w.truncate_ms(jan31));
    }

    #[test]
    fn truncate_ms_month_different_months() {
        let w = CohortWindow::Month;
        let jan1 = 1_704_067_200_000i64; // 2024-01-01
        let feb1 = jan1 + 31 * DAY_MS; // 2024-02-01
        assert_ne!(w.truncate_ms(jan1), w.truncate_ms(feb1));
    }

    // ── build_cohort_matrix ──────────────────────────────────────────────────

    #[test]
    fn cohort_empty_events_returns_error() {
        let result = build_cohort_matrix(&[], CohortWindow::Day, 4);
        assert!(result.is_err());
    }

    #[test]
    fn cohort_single_cohort_period0_full_retention() {
        // All 3 viewers first see content on day 0 and re-engage on day 1.
        let events = vec![
            ev("alice", 0),
            ev("bob", 0),
            ev("charlie", 0),
            ev("alice", DAY_MS),
            ev("bob", DAY_MS),
            // charlie not seen on day 1
        ];
        let matrix = build_cohort_matrix(&events, CohortWindow::Day, 2)
            .expect("build cohort matrix should succeed");
        assert_eq!(matrix.cohorts.len(), 1);
        assert_eq!(matrix.cohorts[0].size(), 3);

        // Period 0: all 3 active → 100 %.
        let r0 = matrix
            .retention_at(0, 0)
            .expect("retention at should succeed");
        assert!((r0 - 1.0).abs() < 1e-6, "period 0 retention={r0}");

        // Period 1: alice + bob active → 2/3.
        let r1 = matrix
            .retention_at(0, 1)
            .expect("retention at should succeed");
        assert!((r1 - 2.0 / 3.0).abs() < 1e-6, "period 1 retention={r1}");

        // Period 2: no events → 0 %.
        let r2 = matrix
            .retention_at(0, 2)
            .expect("retention at should succeed");
        assert_eq!(r2, 0.0, "period 2 retention={r2}");
    }

    #[test]
    fn cohort_two_cohorts() {
        // Cohort 1 (day 0): alice, bob; Cohort 2 (day 1): charlie.
        let events = vec![
            ev("alice", 0),
            ev("bob", 0),
            ev("charlie", DAY_MS),
            ev("alice", DAY_MS), // alice re-engages in period 1
        ];
        let matrix = build_cohort_matrix(&events, CohortWindow::Day, 2)
            .expect("build cohort matrix should succeed");
        assert_eq!(matrix.cohorts.len(), 2, "expected 2 cohorts");

        // Cohort 1 at offset 0 = 2/2 = 1.0.
        let day0_start = CohortWindow::Day.truncate_ms(0);
        let r0 = matrix
            .retention_at(day0_start, 0)
            .expect("retention at should succeed");
        assert!((r0 - 1.0).abs() < 1e-6);

        // Cohort 1 at offset 1: only alice re-engaged → 1/2 = 0.5.
        let r1 = matrix
            .retention_at(day0_start, 1)
            .expect("retention at should succeed");
        assert!((r1 - 0.5).abs() < 1e-6, "cohort1 period1={r1}");
    }

    #[test]
    fn cohort_viewer_counted_once_per_period() {
        // alice fires 5 events on day 0 — should count as 1 unique viewer.
        let events = vec![
            ev("alice", 0),
            ev("alice", 1_000),
            ev("alice", 2_000),
            ev("alice", 3_000),
            ev("alice", 4_000),
        ];
        let matrix = build_cohort_matrix(&events, CohortWindow::Day, 0)
            .expect("build cohort matrix should succeed");
        assert_eq!(matrix.cohorts[0].size(), 1);
        let r0 = matrix
            .retention_at(0, 0)
            .expect("retention at should succeed");
        assert!((r0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cohort_average_retention_weighted() {
        // Cohort A: 4 viewers (100 % at period 1).
        // Cohort B: 2 viewers (0 % at period 1).
        // Weighted average at period 1: 4/6 ≈ 0.667.
        let events = vec![
            ev("a1", 0),
            ev("a2", 0),
            ev("a3", 0),
            ev("a4", 0),
            ev("b1", DAY_MS),
            ev("b2", DAY_MS),
            // Re-engage: all of cohort A, none of cohort B.
            ev("a1", DAY_MS),
            ev("a2", DAY_MS),
            ev("a3", DAY_MS),
            ev("a4", DAY_MS),
        ];
        let matrix = build_cohort_matrix(&events, CohortWindow::Day, 2)
            .expect("build cohort matrix should succeed");
        let avg = matrix.average_retention_at_period(1);
        assert!(
            (avg - 4.0 / 6.0).abs() < 0.05,
            "weighted avg retention at period 1 = {avg}"
        );
    }

    #[test]
    fn cohort_matrix_cells_cover_all_periods() {
        let events = vec![ev("u1", 0), ev("u1", DAY_MS)];
        let num_periods = 3u32;
        let matrix = build_cohort_matrix(&events, CohortWindow::Day, num_periods)
            .expect("build cohort matrix should succeed");
        // 1 cohort × (num_periods + 1) cells.
        assert_eq!(matrix.cells.len(), (num_periods + 1) as usize);
    }

    #[test]
    fn cohort_first_period_viewer_is_in_earliest_cohort() {
        // alice is first seen on day 2, then day 0 in event order.
        // Her cohort should be day 0 (the earliest).
        let events = vec![ev("alice", 2 * DAY_MS), ev("alice", 0), ev("alice", DAY_MS)];
        let matrix = build_cohort_matrix(&events, CohortWindow::Day, 4)
            .expect("build cohort matrix should succeed");
        // alice should belong to the day-0 cohort.
        let day0_cohort = matrix
            .cohorts
            .iter()
            .find(|c| c.window_start_ms == 0)
            .expect("day-0 cohort should exist");
        assert!(day0_cohort.viewer_ids.contains(&"alice".to_string()));
    }

    // ── period_offset ────────────────────────────────────────────────────────

    #[test]
    fn period_offset_same_period_is_zero() {
        assert_eq!(period_offset(0, 0, CohortWindow::Day), 0);
        assert_eq!(period_offset(0, DAY_MS - 1, CohortWindow::Day), 0);
    }

    #[test]
    fn period_offset_next_day_is_one() {
        assert_eq!(period_offset(0, DAY_MS, CohortWindow::Day), 1);
    }

    #[test]
    fn period_offset_next_week_is_one() {
        assert_eq!(period_offset(0, WEEK_MS, CohortWindow::Week), 1);
    }

    #[test]
    fn period_offset_event_before_cohort_clamps_to_zero() {
        // Negative offsets should be clamped to 0.
        assert_eq!(period_offset(DAY_MS, 0, CohortWindow::Day), 0);
    }
}

// ─── Simple cohort definition & analyzer ─────────────────────────────────────

/// A simple user cohort defined by a date and a list of users.
#[derive(Debug, Clone)]
pub struct CohortDefinition {
    /// Epoch milliseconds of the cohort's start-of-day.  All day boundaries
    /// are computed relative to this timestamp.
    pub cohort_date: u64,
    /// Unique user IDs belonging to this cohort.
    pub users: Vec<String>,
}

/// A raw user activity event for `CohortAnalyzer`.
#[derive(Debug, Clone)]
pub struct UserEvent {
    /// Unique user identifier.
    pub user_id: String,
    /// Unix epoch milliseconds when the user was active.
    pub timestamp_ms: u64,
}

/// Computes day-N retention curves from a [`CohortDefinition`] and a stream
/// of [`UserEvent`]s.
pub struct CohortAnalyzer;

impl CohortAnalyzer {
    /// Compute day-N retention rates for a cohort.
    ///
    /// Returns a `Vec<f64>` of length `periods + 1`, where index `i` is the
    /// fraction of `cohort.users` who fired at least one event on day `i`
    /// (i.e. in the time range `[cohort_date + i * DAY_MS, cohort_date + (i+1) * DAY_MS)`).
    ///
    /// Day 0 is the cohort date itself.
    ///
    /// Returns a vector of zeroes if the cohort has no users.
    pub fn retention_curve(
        cohort: &CohortDefinition,
        events: &[UserEvent],
        periods: u32,
    ) -> Vec<f64> {
        let n_periods = periods as usize + 1;
        let mut retention = vec![0f64; n_periods];

        if cohort.users.is_empty() {
            return retention;
        }

        let cohort_size = cohort.users.len() as f64;
        const DAY_MS: u64 = 86_400_000;

        // Build a set for fast cohort membership check.
        let cohort_set: std::collections::HashSet<&str> =
            cohort.users.iter().map(|u| u.as_str()).collect();

        // For each period, count distinct cohort users with an event in that day.
        let mut period_active: Vec<std::collections::HashSet<&str>> = (0..n_periods)
            .map(|_| std::collections::HashSet::new())
            .collect();

        for ev in events {
            if !cohort_set.contains(ev.user_id.as_str()) {
                continue;
            }
            if ev.timestamp_ms < cohort.cohort_date {
                continue;
            }
            let offset_ms = ev.timestamp_ms - cohort.cohort_date;
            let period = (offset_ms / DAY_MS) as usize;
            if period < n_periods {
                period_active[period].insert(ev.user_id.as_str());
            }
        }

        for (i, active_set) in period_active.iter().enumerate() {
            retention[i] = active_set.len() as f64 / cohort_size;
        }

        retention
    }
}

// ─── Tests for CohortDefinition / CohortAnalyzer ─────────────────────────────

#[cfg(test)]
mod cohort_analyzer_tests {
    use super::*;

    const DAY_MS: u64 = 86_400_000;

    fn uev(user: &str, ts: u64) -> UserEvent {
        UserEvent {
            user_id: user.to_string(),
            timestamp_ms: ts,
        }
    }

    #[test]
    fn retention_curve_empty_users_returns_zeros() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec![],
        };
        let curve = CohortAnalyzer::retention_curve(&cohort, &[uev("u1", 0)], 3);
        assert_eq!(curve.len(), 4);
        assert!(curve.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn retention_curve_empty_events_returns_zeros() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string(), "u2".to_string()],
        };
        let curve = CohortAnalyzer::retention_curve(&cohort, &[], 3);
        assert!(curve.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn retention_curve_period0_full_retention() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string(), "u2".to_string(), "u3".to_string()],
        };
        let events = vec![uev("u1", 0), uev("u2", 100), uev("u3", 200)];
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 0);
        assert_eq!(curve.len(), 1);
        assert!((curve[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn retention_curve_day1_partial_retention() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string(), "u2".to_string()],
        };
        let events = vec![
            uev("u1", 0),      // day 0
            uev("u1", DAY_MS), // day 1
            uev("u2", 0),      // day 0 only
        ];
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 2);
        assert!((curve[0] - 1.0).abs() < 1e-9); // both active day 0
        assert!((curve[1] - 0.5).abs() < 1e-9); // only u1 active day 1
        assert!(curve[2].abs() < 1e-9); // nobody on day 2
    }

    #[test]
    fn retention_curve_zero_retention_after_day0() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string()],
        };
        let events = vec![uev("u1", 0)]; // active only day 0
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 3);
        assert!((curve[0] - 1.0).abs() < 1e-9);
        assert!(curve[1].abs() < 1e-9);
        assert!(curve[2].abs() < 1e-9);
        assert!(curve[3].abs() < 1e-9);
    }

    #[test]
    fn retention_curve_non_cohort_events_ignored() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string()],
        };
        let events = vec![
            uev("u1", 0),
            uev("u_other", 0),      // not in cohort
            uev("u_other", DAY_MS), // not in cohort
        ];
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 1);
        assert!((curve[0] - 1.0).abs() < 1e-9);
        assert!(curve[1].abs() < 1e-9);
    }

    #[test]
    fn retention_curve_events_before_cohort_date_ignored() {
        let cohort = CohortDefinition {
            cohort_date: DAY_MS, // cohort starts on day 1
            users: vec!["u1".to_string()],
        };
        let events = vec![
            uev("u1", 0),      // before cohort date — should be ignored
            uev("u1", DAY_MS), // day 0 of cohort
        ];
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 1);
        assert!((curve[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn retention_curve_user_active_multiple_times_same_day_counted_once() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string()],
        };
        let events = vec![uev("u1", 100), uev("u1", 200), uev("u1", 300)]; // 3 events on day 0
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 0);
        assert!((curve[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn retention_curve_length_equals_periods_plus_one() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string()],
        };
        for p in [0u32, 1, 5, 10] {
            let curve = CohortAnalyzer::retention_curve(&cohort, &[], p);
            assert_eq!(curve.len(), p as usize + 1, "periods={p}");
        }
    }

    #[test]
    fn retention_curve_multi_period_full_retention() {
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: vec!["u1".to_string(), "u2".to_string()],
        };
        let events: Vec<UserEvent> = (0..5)
            .flat_map(|day| vec![uev("u1", day * DAY_MS + 100), uev("u2", day * DAY_MS + 200)])
            .collect();
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 4);
        for (i, r) in curve.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-9, "day {i} retention={r} expected 1.0");
        }
    }

    #[test]
    fn retention_curve_gradual_decay() {
        // 4 users, each drops off one day later.
        let users = vec![
            "u0".to_string(),
            "u1".to_string(),
            "u2".to_string(),
            "u3".to_string(),
        ];
        let cohort = CohortDefinition {
            cohort_date: 0,
            users: users.clone(),
        };
        // ui is active on days 0..=(3-i).
        let events: Vec<UserEvent> = users
            .iter()
            .enumerate()
            .flat_map(|(i, uid)| (0..=(3 - i) as u64).map(move |day| uev(uid, day * DAY_MS)))
            .collect();
        let curve = CohortAnalyzer::retention_curve(&cohort, &events, 3);
        assert!((curve[0] - 1.0).abs() < 1e-9);
        assert!((curve[1] - 0.75).abs() < 1e-9);
        assert!((curve[2] - 0.5).abs() < 1e-9);
        assert!((curve[3] - 0.25).abs() < 1e-9);
    }
}
