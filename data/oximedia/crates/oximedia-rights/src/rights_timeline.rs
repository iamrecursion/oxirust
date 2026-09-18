//! Rights-window timeline management.
//!
//! Models the temporal dimension of content rights as a series of
//! non-overlapping or overlapping windows on a timeline. Supports
//! querying which rights are active at a point in time, finding gaps
//! in coverage, and merging adjacent windows.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── WindowStatus ───────────────────────────────────────────────────────────

/// The status of a rights window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowStatus {
    /// Window is currently active / in effect.
    Active,
    /// Window is scheduled for the future.
    Pending,
    /// Window has expired.
    Expired,
    /// Window has been explicitly suspended.
    Suspended,
}

// ── RightsWindow ───────────────────────────────────────────────────────────

/// A single time window during which specific rights apply.
#[derive(Debug, Clone)]
pub struct RightsWindow {
    /// Unique identifier.
    pub id: String,
    /// Asset to which this window applies.
    pub asset_id: String,
    /// Description of the rights granted during this window.
    pub description: String,
    /// Start time (Unix seconds, inclusive).
    pub start: u64,
    /// End time (Unix seconds, exclusive). `u64::MAX` = open-ended.
    pub end: u64,
    /// Current status.
    pub status: WindowStatus,
    /// Optional territory restrictions (empty = worldwide).
    pub territories: Vec<String>,
}

impl RightsWindow {
    /// Create a new active window.
    #[must_use]
    pub fn new(id: &str, asset_id: &str, start: u64, end: u64) -> Self {
        Self {
            id: id.to_string(),
            asset_id: asset_id.to_string(),
            description: String::new(),
            start,
            end,
            status: WindowStatus::Active,
            territories: Vec::new(),
        }
    }

    /// Builder: set description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Builder: set status.
    #[must_use]
    pub fn with_status(mut self, status: WindowStatus) -> Self {
        self.status = status;
        self
    }

    /// Builder: add a territory restriction.
    #[must_use]
    pub fn with_territory(mut self, code: &str) -> Self {
        self.territories.push(code.to_uppercase());
        self
    }

    /// Duration of the window in seconds. Returns `None` for open-ended windows.
    #[must_use]
    pub fn duration_secs(&self) -> Option<u64> {
        if self.end == u64::MAX {
            None
        } else {
            Some(self.end.saturating_sub(self.start))
        }
    }

    /// Whether a given timestamp falls within this window.
    #[must_use]
    pub fn contains(&self, ts: u64) -> bool {
        ts >= self.start && ts < self.end
    }

    /// Whether this window overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Whether this window is adjacent to (and could be merged with) another.
    /// Two windows are adjacent if one's end equals the other's start.
    #[must_use]
    pub fn is_adjacent(&self, other: &Self) -> bool {
        self.end == other.start || other.end == self.start
    }

    /// Derive the status from a current timestamp.
    #[must_use]
    pub fn derived_status(&self, now: u64) -> WindowStatus {
        if self.status == WindowStatus::Suspended {
            return WindowStatus::Suspended;
        }
        if now < self.start {
            WindowStatus::Pending
        } else if now >= self.end {
            WindowStatus::Expired
        } else {
            WindowStatus::Active
        }
    }
}

// ── RightsTimeline ─────────────────────────────────────────────────────────

/// A collection of [`RightsWindow`]s for one or more assets, providing
/// timeline-aware queries.
#[derive(Debug, Clone, Default)]
pub struct RightsTimeline {
    windows: Vec<RightsWindow>,
}

impl RightsTimeline {
    /// Create an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a window to the timeline.
    pub fn add(&mut self, window: RightsWindow) {
        self.windows.push(window);
    }

    /// Total number of windows.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Return all windows that are active at the given timestamp.
    #[must_use]
    pub fn active_at(&self, ts: u64) -> Vec<&RightsWindow> {
        self.windows
            .iter()
            .filter(|w| w.contains(ts) && w.status != WindowStatus::Suspended)
            .collect()
    }

    /// Return all windows for a specific asset.
    #[must_use]
    pub fn for_asset(&self, asset_id: &str) -> Vec<&RightsWindow> {
        self.windows
            .iter()
            .filter(|w| w.asset_id == asset_id)
            .collect()
    }

    /// Return all windows for a specific asset that are active at a
    /// given timestamp.
    #[must_use]
    pub fn active_for_asset(&self, asset_id: &str, ts: u64) -> Vec<&RightsWindow> {
        self.windows
            .iter()
            .filter(|w| {
                w.asset_id == asset_id && w.contains(ts) && w.status != WindowStatus::Suspended
            })
            .collect()
    }

    /// Find gaps in coverage for an asset between `range_start` and `range_end`.
    ///
    /// Returns a list of `(gap_start, gap_end)` pairs.
    #[must_use]
    pub fn gaps(&self, asset_id: &str, range_start: u64, range_end: u64) -> Vec<(u64, u64)> {
        let mut windows: Vec<&RightsWindow> = self
            .windows
            .iter()
            .filter(|w| {
                w.asset_id == asset_id
                    && w.status != WindowStatus::Suspended
                    && w.start < range_end
                    && w.end > range_start
            })
            .collect();

        windows.sort_by_key(|w| w.start);

        let mut gaps = Vec::new();
        let mut cursor = range_start;

        for w in &windows {
            let effective_start = w.start.max(range_start);
            let effective_end = w.end.min(range_end);
            if effective_start > cursor {
                gaps.push((cursor, effective_start));
            }
            if effective_end > cursor {
                cursor = effective_end;
            }
        }

        if cursor < range_end {
            gaps.push((cursor, range_end));
        }

        gaps
    }

    /// Count windows by status at a given timestamp.
    #[must_use]
    pub fn status_counts(&self, now: u64) -> HashMap<WindowStatus, usize> {
        let mut counts: HashMap<WindowStatus, usize> = HashMap::new();
        for w in &self.windows {
            let status = w.derived_status(now);
            *counts.entry(status).or_insert(0) += 1;
        }
        counts
    }

    /// Return all windows sorted by start time.
    #[must_use]
    pub fn sorted_by_start(&self) -> Vec<&RightsWindow> {
        let mut sorted: Vec<&RightsWindow> = self.windows.iter().collect();
        sorted.sort_by_key(|w| w.start);
        sorted
    }

    /// Total covered duration (in seconds) for an asset within a range,
    /// accounting for overlapping windows.
    #[must_use]
    pub fn covered_duration(&self, asset_id: &str, range_start: u64, range_end: u64) -> u64 {
        let gap_total: u64 = self
            .gaps(asset_id, range_start, range_end)
            .iter()
            .map(|(s, e)| e - s)
            .sum();
        (range_end - range_start).saturating_sub(gap_total)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_timeline() -> RightsTimeline {
        let mut tl = RightsTimeline::new();
        // Window 1: 100..200
        tl.add(
            RightsWindow::new("w1", "asset-A", 100, 200).with_description("First broadcast window"),
        );
        // Window 2: 300..500
        tl.add(
            RightsWindow::new("w2", "asset-A", 300, 500)
                .with_description("Second broadcast window"),
        );
        // Window 3: 150..400 (overlaps with both w1 and w2)
        tl.add(RightsWindow::new("w3", "asset-B", 150, 400).with_description("Different asset"));
        tl
    }

    // ── RightsWindow ──

    #[test]
    fn test_window_contains() {
        let w = RightsWindow::new("w", "a", 100, 200);
        assert!(w.contains(100));
        assert!(w.contains(150));
        assert!(!w.contains(200)); // exclusive end
        assert!(!w.contains(50));
    }

    #[test]
    fn test_window_overlaps() {
        let a = RightsWindow::new("a", "x", 100, 300);
        let b = RightsWindow::new("b", "x", 200, 400);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_window_no_overlap() {
        let a = RightsWindow::new("a", "x", 100, 200);
        let b = RightsWindow::new("b", "x", 200, 300);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_window_adjacent() {
        let a = RightsWindow::new("a", "x", 100, 200);
        let b = RightsWindow::new("b", "x", 200, 300);
        assert!(a.is_adjacent(&b));
    }

    #[test]
    fn test_window_duration() {
        let w = RightsWindow::new("w", "a", 100, 250);
        assert_eq!(w.duration_secs(), Some(150));
    }

    #[test]
    fn test_window_open_ended_duration() {
        let w = RightsWindow::new("w", "a", 100, u64::MAX);
        assert!(w.duration_secs().is_none());
    }

    #[test]
    fn test_window_derived_status() {
        let w = RightsWindow::new("w", "a", 100, 200);
        assert_eq!(w.derived_status(50), WindowStatus::Pending);
        assert_eq!(w.derived_status(150), WindowStatus::Active);
        assert_eq!(w.derived_status(200), WindowStatus::Expired);
    }

    #[test]
    fn test_window_suspended_overrides() {
        let w = RightsWindow::new("w", "a", 100, 200).with_status(WindowStatus::Suspended);
        assert_eq!(w.derived_status(150), WindowStatus::Suspended);
    }

    // ── RightsTimeline ──

    #[test]
    fn test_timeline_window_count() {
        let tl = sample_timeline();
        assert_eq!(tl.window_count(), 3);
    }

    #[test]
    fn test_timeline_active_at() {
        let tl = sample_timeline();
        // At ts=150: w1(100..200) is active for asset-A, w3(150..400) for asset-B
        let active = tl.active_at(150);
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_timeline_for_asset() {
        let tl = sample_timeline();
        let a_windows = tl.for_asset("asset-A");
        assert_eq!(a_windows.len(), 2);
    }

    #[test]
    fn test_timeline_active_for_asset() {
        let tl = sample_timeline();
        let active = tl.active_for_asset("asset-A", 150);
        assert_eq!(active.len(), 1); // only w1
        assert_eq!(active[0].id, "w1");
    }

    #[test]
    fn test_timeline_gaps() {
        let tl = sample_timeline();
        // asset-A has w1=100..200 and w2=300..500
        // Range 0..600 => gaps: 0..100, 200..300, 500..600
        let gaps = tl.gaps("asset-A", 0, 600);
        assert_eq!(gaps.len(), 3);
        assert_eq!(gaps[0], (0, 100));
        assert_eq!(gaps[1], (200, 300));
        assert_eq!(gaps[2], (500, 600));
    }

    #[test]
    fn test_timeline_no_gaps() {
        let mut tl = RightsTimeline::new();
        tl.add(RightsWindow::new("w1", "a", 0, 100));
        tl.add(RightsWindow::new("w2", "a", 100, 200));
        let gaps = tl.gaps("a", 0, 200);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_timeline_covered_duration() {
        let tl = sample_timeline();
        // asset-A: windows cover 100..200 (100s) and 300..500 (200s) = 300s
        // Range 0..600 => covered = 300
        let covered = tl.covered_duration("asset-A", 0, 600);
        assert_eq!(covered, 300);
    }

    #[test]
    fn test_timeline_sorted_by_start() {
        let tl = sample_timeline();
        let sorted = tl.sorted_by_start();
        for pair in sorted.windows(2) {
            assert!(pair[0].start <= pair[1].start);
        }
    }

    #[test]
    fn test_timeline_status_counts() {
        let tl = sample_timeline();
        // At ts=350: w1 expired(100..200), w2 active(300..500), w3 active(150..400)
        let counts = tl.status_counts(350);
        assert_eq!(*counts.get(&WindowStatus::Active).unwrap_or(&0), 2);
        assert_eq!(*counts.get(&WindowStatus::Expired).unwrap_or(&0), 1);
    }
}
