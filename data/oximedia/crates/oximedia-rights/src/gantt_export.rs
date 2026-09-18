//! Gantt-chart style rights-period visualization data export.
//!
//! Transforms a [`RightsTimeline`] into a structured
//! [`GanttChart`] that can be serialized to JSON or rendered in a UI.
//! Each asset becomes a Gantt row; each [`RightsWindow`] becomes a bar.
//!
//! The coordinate system uses the `u64` Unix-second epoch directly; callers
//! are responsible for mapping that to calendar dates for display.

#![allow(dead_code)]

use crate::rights_timeline::{RightsTimeline, RightsWindow, WindowStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── GanttBar ────────────────────────────────────────────────────────────────

/// A single bar (rights window) in a Gantt chart row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GanttBar {
    /// Window identifier.
    pub id: String,
    /// Start position on the timeline (Unix seconds).
    pub start: u64,
    /// End position on the timeline (Unix seconds). `u64::MAX` = open-ended.
    pub end: u64,
    /// Human-readable label / description.
    pub label: String,
    /// Status at the time the export was generated.
    pub status: GanttStatus,
    /// Territory restrictions (empty = worldwide).
    pub territories: Vec<String>,
    /// Duration in seconds, or `None` for open-ended windows.
    pub duration_secs: Option<u64>,
}

/// Simplified status enum for Gantt serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GanttStatus {
    /// Window is currently active.
    Active,
    /// Window starts in the future.
    Pending,
    /// Window has expired.
    Expired,
    /// Window has been suspended.
    Suspended,
}

impl From<WindowStatus> for GanttStatus {
    fn from(s: WindowStatus) -> Self {
        match s {
            WindowStatus::Active => Self::Active,
            WindowStatus::Pending => Self::Pending,
            WindowStatus::Expired => Self::Expired,
            WindowStatus::Suspended => Self::Suspended,
        }
    }
}

impl GanttBar {
    /// Build a `GanttBar` from a [`RightsWindow`] evaluated at `now`.
    #[must_use]
    pub fn from_window(window: &RightsWindow, now: u64) -> Self {
        Self {
            id: window.id.clone(),
            start: window.start,
            end: window.end,
            label: if window.description.is_empty() {
                window.id.clone()
            } else {
                window.description.clone()
            },
            status: window.derived_status(now).into(),
            territories: window.territories.clone(),
            duration_secs: window.duration_secs(),
        }
    }
}

// ── GanttRow ────────────────────────────────────────────────────────────────

/// A single row in the Gantt chart, corresponding to one asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GanttRow {
    /// Asset identifier.
    pub asset_id: String,
    /// Bars (rights windows) belonging to this asset.
    pub bars: Vec<GanttBar>,
    /// Earliest start across all bars (`u64::MAX` if no bars).
    pub earliest_start: u64,
    /// Latest end across all bars (`0` if no bars).
    pub latest_end: u64,
}

impl GanttRow {
    /// Build a row from a list of windows (all from the same asset).
    #[must_use]
    pub fn new(asset_id: &str, windows: &[&RightsWindow], now: u64) -> Self {
        let mut bars: Vec<GanttBar> = windows
            .iter()
            .map(|w| GanttBar::from_window(w, now))
            .collect();
        // Sort bars by start for consistent rendering.
        bars.sort_by_key(|b| b.start);

        let earliest_start = bars.iter().map(|b| b.start).min().unwrap_or(u64::MAX);
        let latest_end = bars.iter().map(|b| b.end).max().unwrap_or(0);

        Self {
            asset_id: asset_id.to_string(),
            bars,
            earliest_start,
            latest_end,
        }
    }

    /// Number of bars in this row.
    #[must_use]
    pub fn bar_count(&self) -> usize {
        self.bars.len()
    }

    /// Count bars with a given status.
    #[must_use]
    pub fn count_by_status(&self, status: GanttStatus) -> usize {
        self.bars.iter().filter(|b| b.status == status).count()
    }
}

// ── GanttChart ──────────────────────────────────────────────────────────────

/// The full Gantt chart export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GanttChart {
    /// Rows, one per asset, sorted by asset ID.
    pub rows: Vec<GanttRow>,
    /// Timestamp (Unix seconds) at which the chart was generated.
    pub generated_at: u64,
    /// Overall earliest start across all rows.
    pub chart_start: u64,
    /// Overall latest end across all rows.
    pub chart_end: u64,
    /// Total number of bars across all rows.
    pub total_bars: usize,
}

impl GanttChart {
    /// Build a `GanttChart` from a [`RightsTimeline`] evaluated at `now`.
    #[must_use]
    pub fn from_timeline(timeline: &RightsTimeline, now: u64) -> Self {
        // Group windows by asset_id.
        let sorted = timeline.sorted_by_start();
        let mut by_asset: HashMap<&str, Vec<&RightsWindow>> = HashMap::new();
        for window in &sorted {
            by_asset
                .entry(window.asset_id.as_str())
                .or_default()
                .push(window);
        }

        // Sort asset IDs for deterministic output.
        let mut asset_ids: Vec<&str> = by_asset.keys().copied().collect();
        asset_ids.sort_unstable();

        let rows: Vec<GanttRow> = asset_ids
            .into_iter()
            .map(|id| GanttRow::new(id, by_asset[id].as_slice(), now))
            .collect();

        let chart_start = rows.iter().map(|r| r.earliest_start).min().unwrap_or(0);
        let chart_end = rows.iter().map(|r| r.latest_end).max().unwrap_or(0);
        let total_bars = rows.iter().map(|r| r.bar_count()).sum();

        Self {
            rows,
            generated_at: now,
            chart_start,
            chart_end,
            total_bars,
        }
    }

    /// Export to JSON string.
    ///
    /// # Errors
    /// Returns a `String` description of the serialization error.
    pub fn to_json(&self) -> std::result::Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// Find the row for a specific asset.
    #[must_use]
    pub fn row_for_asset(&self, asset_id: &str) -> Option<&GanttRow> {
        self.rows.iter().find(|r| r.asset_id == asset_id)
    }

    /// Total number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Count all bars with a given status across all rows.
    #[must_use]
    pub fn total_by_status(&self, status: GanttStatus) -> usize {
        self.rows.iter().map(|r| r.count_by_status(status)).sum()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights_timeline::RightsWindow;

    fn build_timeline() -> RightsTimeline {
        let mut tl = RightsTimeline::new();
        // asset-A: two windows
        tl.add(RightsWindow::new("w1", "asset-A", 100, 200).with_description("Window 1"));
        tl.add(RightsWindow::new("w2", "asset-A", 300, 500).with_description("Window 2"));
        // asset-B: one window, suspended
        tl.add(
            RightsWindow::new("w3", "asset-B", 100, 400)
                .with_status(WindowStatus::Suspended)
                .with_territory("US"),
        );
        tl
    }

    #[test]
    fn test_gantt_chart_row_count() {
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        assert_eq!(chart.row_count(), 2);
    }

    #[test]
    fn test_gantt_chart_total_bars() {
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        assert_eq!(chart.total_bars, 3);
    }

    #[test]
    fn test_gantt_chart_asset_a_bars() {
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        let row = chart
            .row_for_asset("asset-A")
            .expect("asset-A row should exist");
        assert_eq!(row.bar_count(), 2);
    }

    #[test]
    fn test_gantt_bar_status_at_ts_250() {
        // At ts=250: w1(100..200) expired, w2(300..500) pending
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        let row = chart
            .row_for_asset("asset-A")
            .expect("asset-A row should exist");
        let expired_count = row.count_by_status(GanttStatus::Expired);
        let pending_count = row.count_by_status(GanttStatus::Pending);
        assert_eq!(expired_count, 1);
        assert_eq!(pending_count, 1);
    }

    #[test]
    fn test_gantt_bar_suspended() {
        let chart = GanttChart::from_timeline(&build_timeline(), 200);
        let row = chart
            .row_for_asset("asset-B")
            .expect("asset-B row should exist");
        assert_eq!(row.count_by_status(GanttStatus::Suspended), 1);
    }

    #[test]
    fn test_gantt_chart_start_and_end() {
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        assert_eq!(chart.chart_start, 100);
        assert_eq!(chart.chart_end, 500);
    }

    #[test]
    fn test_gantt_chart_generated_at() {
        let chart = GanttChart::from_timeline(&build_timeline(), 999);
        assert_eq!(chart.generated_at, 999);
    }

    #[test]
    fn test_gantt_chart_to_json() {
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        let json = chart.to_json();
        assert!(json.is_ok());
        let s = json.expect("serialization should succeed");
        assert!(s.contains("asset-A"));
        assert!(s.contains("asset-B"));
    }

    #[test]
    fn test_gantt_bar_label_from_description() {
        let chart = GanttChart::from_timeline(&build_timeline(), 150);
        let row = chart
            .row_for_asset("asset-A")
            .expect("asset-A row should exist");
        let bar = row.bars.iter().find(|b| b.id == "w1").expect("w1 bar");
        assert_eq!(bar.label, "Window 1");
    }

    #[test]
    fn test_gantt_bar_label_fallback_to_id() {
        // Window with no description
        let mut tl = RightsTimeline::new();
        tl.add(RightsWindow::new("wX", "asset-C", 0, 100));
        let chart = GanttChart::from_timeline(&tl, 50);
        let row = chart
            .row_for_asset("asset-C")
            .expect("asset-C row should exist");
        assert_eq!(row.bars[0].label, "wX");
    }

    #[test]
    fn test_gantt_bar_territories() {
        let chart = GanttChart::from_timeline(&build_timeline(), 200);
        let row = chart
            .row_for_asset("asset-B")
            .expect("asset-B row should exist");
        assert_eq!(row.bars[0].territories, vec!["US"]);
    }

    #[test]
    fn test_gantt_total_by_status_active() {
        // At ts=150: w1 active, w3 suspended (overrides active), w2 pending
        let chart = GanttChart::from_timeline(&build_timeline(), 150);
        assert_eq!(chart.total_by_status(GanttStatus::Active), 1);
        assert_eq!(chart.total_by_status(GanttStatus::Suspended), 1);
    }

    #[test]
    fn test_gantt_row_earliest_latest() {
        let chart = GanttChart::from_timeline(&build_timeline(), 250);
        let row = chart
            .row_for_asset("asset-A")
            .expect("asset-A row should exist");
        assert_eq!(row.earliest_start, 100);
        assert_eq!(row.latest_end, 500);
    }

    #[test]
    fn test_gantt_from_empty_timeline() {
        let tl = RightsTimeline::new();
        let chart = GanttChart::from_timeline(&tl, 0);
        assert_eq!(chart.row_count(), 0);
        assert_eq!(chart.total_bars, 0);
    }

    #[test]
    fn test_gantt_status_from_window_status() {
        assert_eq!(GanttStatus::from(WindowStatus::Active), GanttStatus::Active);
        assert_eq!(
            GanttStatus::from(WindowStatus::Pending),
            GanttStatus::Pending
        );
        assert_eq!(
            GanttStatus::from(WindowStatus::Expired),
            GanttStatus::Expired
        );
        assert_eq!(
            GanttStatus::from(WindowStatus::Suspended),
            GanttStatus::Suspended
        );
    }
}
