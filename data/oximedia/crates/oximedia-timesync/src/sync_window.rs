//! Synchronisation windows — contiguous time ranges during which a clock is considered locked.
#![allow(dead_code)]

/// A contiguous time window `[start_ms, end_ms)` measured in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncWindow {
    /// Start of the window in milliseconds (inclusive).
    pub start_ms: u64,
    /// End of the window in milliseconds (exclusive).
    pub end_ms: u64,
}

impl SyncWindow {
    /// Create a new `SyncWindow`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `end_ms < start_ms`.
    pub fn new(start_ms: u64, end_ms: u64) -> Self {
        debug_assert!(end_ms >= start_ms, "end_ms must be >= start_ms");
        Self { start_ms, end_ms }
    }

    /// Duration of the window in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if `time_ms` falls within `[start_ms, end_ms)`.
    pub fn contains_ms(&self, time_ms: u64) -> bool {
        time_ms >= self.start_ms && time_ms < self.end_ms
    }

    /// Returns `true` if this window overlaps with `other`.
    pub fn overlaps(&self, other: &SyncWindow) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Merge this window with `other`, producing the smallest window containing both.
    pub fn merge(&self, other: &SyncWindow) -> SyncWindow {
        SyncWindow::new(
            self.start_ms.min(other.start_ms),
            self.end_ms.max(other.end_ms),
        )
    }
}

/// A collection of (possibly overlapping) `SyncWindow` instances.
#[derive(Debug, Clone, Default)]
pub struct SyncWindowSet {
    windows: Vec<SyncWindow>,
}

impl SyncWindowSet {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a window to the set.
    pub fn add(&mut self, window: SyncWindow) {
        self.windows.push(window);
    }

    /// Number of windows in the set (before merging).
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Returns `true` if the set contains no windows.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Returns all windows that overlap with `window`.
    pub fn overlapping(&self, window: &SyncWindow) -> Vec<SyncWindow> {
        self.windows
            .iter()
            .filter(|w| w.overlaps(window))
            .copied()
            .collect()
    }

    /// Merge all overlapping windows and return the consolidated list, sorted by `start_ms`.
    pub fn merge_overlapping(&self) -> Vec<SyncWindow> {
        let mut sorted = self.windows.clone();
        sorted.sort_by_key(|w| w.start_ms);

        let mut merged: Vec<SyncWindow> = Vec::new();
        for win in sorted {
            if let Some(last) = merged.last_mut() {
                if win.overlaps(last) || win.start_ms == last.end_ms {
                    *last = last.merge(&win);
                    continue;
                }
            }
            merged.push(win);
        }
        merged
    }

    /// Total coverage in milliseconds after merging overlapping windows.
    pub fn total_coverage_ms(&self) -> u64 {
        self.merge_overlapping()
            .iter()
            .map(SyncWindow::duration_ms)
            .sum()
    }

    /// Returns an iterator over the raw (unmerged) windows.
    pub fn iter(&self) -> impl Iterator<Item = &SyncWindow> {
        self.windows.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_window_duration() {
        let w = SyncWindow::new(100, 200);
        assert_eq!(w.duration_ms(), 100);
    }

    #[test]
    fn test_sync_window_zero_duration() {
        let w = SyncWindow::new(50, 50);
        assert_eq!(w.duration_ms(), 0);
    }

    #[test]
    fn test_contains_ms_inside() {
        let w = SyncWindow::new(0, 100);
        assert!(w.contains_ms(50));
    }

    #[test]
    fn test_contains_ms_at_start() {
        let w = SyncWindow::new(10, 20);
        assert!(w.contains_ms(10));
    }

    #[test]
    fn test_contains_ms_at_end_exclusive() {
        let w = SyncWindow::new(10, 20);
        assert!(!w.contains_ms(20));
    }

    #[test]
    fn test_contains_ms_outside() {
        let w = SyncWindow::new(10, 20);
        assert!(!w.contains_ms(5));
        assert!(!w.contains_ms(25));
    }

    #[test]
    fn test_overlaps_true() {
        let a = SyncWindow::new(0, 100);
        let b = SyncWindow::new(50, 150);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_adjacent_no_overlap() {
        let a = SyncWindow::new(0, 50);
        let b = SyncWindow::new(50, 100);
        // [0,50) and [50,100) are adjacent but not overlapping.
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_disjoint() {
        let a = SyncWindow::new(0, 10);
        let b = SyncWindow::new(20, 30);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_merge_windows() {
        let a = SyncWindow::new(0, 50);
        let b = SyncWindow::new(30, 80);
        let m = a.merge(&b);
        assert_eq!(m.start_ms, 0);
        assert_eq!(m.end_ms, 80);
    }

    #[test]
    fn test_sync_window_set_add_and_len() {
        let mut set = SyncWindowSet::new();
        set.add(SyncWindow::new(0, 100));
        set.add(SyncWindow::new(50, 150));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_sync_window_set_overlapping() {
        let mut set = SyncWindowSet::new();
        set.add(SyncWindow::new(0, 100));
        set.add(SyncWindow::new(200, 300));
        let query = SyncWindow::new(50, 250);
        let hits = set.overlapping(&query);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_merge_overlapping_consolidates() {
        let mut set = SyncWindowSet::new();
        set.add(SyncWindow::new(0, 100));
        set.add(SyncWindow::new(50, 150));
        set.add(SyncWindow::new(200, 300));
        let merged = set.merge_overlapping();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], SyncWindow::new(0, 150));
        assert_eq!(merged[1], SyncWindow::new(200, 300));
    }

    #[test]
    fn test_total_coverage_ms() {
        let mut set = SyncWindowSet::new();
        set.add(SyncWindow::new(0, 100));
        set.add(SyncWindow::new(50, 150)); // overlaps => merged to [0,150)
        set.add(SyncWindow::new(200, 300));
        // coverage = 150 + 100 = 250
        assert_eq!(set.total_coverage_ms(), 250);
    }

    #[test]
    fn test_empty_set() {
        let set = SyncWindowSet::new();
        assert!(set.is_empty());
        assert_eq!(set.total_coverage_ms(), 0);
        assert!(set.merge_overlapping().is_empty());
    }
}
