//! Distribution window scheduling for content release management.

#![allow(dead_code)]

use std::collections::HashMap;

/// Primary distribution channel type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WindowType {
    /// Theatrical (cinema) release.
    Theatrical,
    /// Home video (DVD, Blu-ray, EST).
    HomeVideo,
    /// Streaming (SVOD / TVOD / AVOD).
    Streaming,
    /// Linear broadcast (TV channels).
    Broadcast,
}

impl WindowType {
    /// Typical exclusivity window length in weeks for this channel.
    pub fn typical_weeks(&self) -> u32 {
        match self {
            WindowType::Theatrical => 12,
            WindowType::HomeVideo => 16,
            WindowType::Streaming => 26,
            WindowType::Broadcast => 52,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            WindowType::Theatrical => "Theatrical",
            WindowType::HomeVideo => "Home Video",
            WindowType::Streaming => "Streaming",
            WindowType::Broadcast => "Broadcast",
        }
    }
}

/// A single exclusive distribution window with start/end epochs (Unix seconds).
#[derive(Debug, Clone)]
pub struct DistributionWindow {
    /// Identifier for this window.
    pub id: String,
    /// Channel type.
    pub window_type: WindowType,
    /// Start of the window (Unix timestamp in seconds).
    pub start_ts: i64,
    /// End of the window (Unix timestamp in seconds).
    pub end_ts: i64,
    /// Optional territory code (ISO 3166-1 alpha-2), `None` means worldwide.
    pub territory: Option<String>,
}

impl DistributionWindow {
    /// Create a new distribution window.
    pub fn new(id: impl Into<String>, window_type: WindowType, start_ts: i64, end_ts: i64) -> Self {
        Self {
            id: id.into(),
            window_type,
            start_ts,
            end_ts,
            territory: None,
        }
    }

    /// Constrain this window to a specific territory.
    pub fn with_territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = Some(territory.into());
        self
    }

    /// Returns `true` when the window is active at the given Unix timestamp.
    pub fn is_active_at(&self, ts: i64) -> bool {
        ts >= self.start_ts && ts < self.end_ts
    }

    /// Duration of the window in seconds.
    pub fn duration_seconds(&self) -> i64 {
        (self.end_ts - self.start_ts).max(0)
    }

    /// Duration of the window in whole weeks (rounded down).
    pub fn duration_weeks(&self) -> u32 {
        (self.duration_seconds() / (7 * 24 * 3600)) as u32
    }
}

/// Manages a set of `DistributionWindow` entries for a title.
#[derive(Debug, Default)]
pub struct WindowSchedule {
    windows: HashMap<String, DistributionWindow>,
}

impl WindowSchedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or replace) a window in the schedule.
    pub fn add_window(&mut self, window: DistributionWindow) {
        self.windows.insert(window.id.clone(), window);
    }

    /// Return all windows active at the given timestamp.
    pub fn active_windows(&self, ts: i64) -> Vec<&DistributionWindow> {
        self.windows
            .values()
            .filter(|w| w.is_active_at(ts))
            .collect()
    }

    /// Find a window by ID.
    pub fn find(&self, id: &str) -> Option<&DistributionWindow> {
        self.windows.get(id)
    }

    /// Remove a window by ID.
    pub fn remove(&mut self, id: &str) -> Option<DistributionWindow> {
        self.windows.remove(id)
    }

    /// Return all windows of a given type.
    pub fn by_type(&self, window_type: &WindowType) -> Vec<&DistributionWindow> {
        self.windows
            .values()
            .filter(|w| &w.window_type == window_type)
            .collect()
    }

    /// Total number of windows in the schedule.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Returns `true` when the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: i64 = 1_700_000_000; // arbitrary fixed epoch

    fn theatrical_window() -> DistributionWindow {
        DistributionWindow::new("w-theat", WindowType::Theatrical, BASE, BASE + 7_257_600)
        // 84 days ≈ 12 weeks
    }

    fn streaming_window() -> DistributionWindow {
        // starts after theatrical
        DistributionWindow::new(
            "w-stream",
            WindowType::Streaming,
            BASE + 7_257_600,
            BASE + 7_257_600 + 15_724_800,
        )
    }

    #[test]
    fn test_theatrical_typical_weeks() {
        assert_eq!(WindowType::Theatrical.typical_weeks(), 12);
    }

    #[test]
    fn test_home_video_typical_weeks() {
        assert_eq!(WindowType::HomeVideo.typical_weeks(), 16);
    }

    #[test]
    fn test_streaming_typical_weeks() {
        assert_eq!(WindowType::Streaming.typical_weeks(), 26);
    }

    #[test]
    fn test_broadcast_typical_weeks() {
        assert_eq!(WindowType::Broadcast.typical_weeks(), 52);
    }

    #[test]
    fn test_window_labels() {
        assert_eq!(WindowType::Theatrical.label(), "Theatrical");
        assert_eq!(WindowType::HomeVideo.label(), "Home Video");
        assert_eq!(WindowType::Streaming.label(), "Streaming");
        assert_eq!(WindowType::Broadcast.label(), "Broadcast");
    }

    #[test]
    fn test_is_active_at_within_range() {
        let w = theatrical_window();
        assert!(w.is_active_at(BASE + 1000));
    }

    #[test]
    fn test_is_not_active_before_start() {
        let w = theatrical_window();
        assert!(!w.is_active_at(BASE - 1));
    }

    #[test]
    fn test_is_not_active_at_end() {
        let w = theatrical_window();
        assert!(!w.is_active_at(BASE + 7_257_600)); // end is exclusive
    }

    #[test]
    fn test_duration_weeks() {
        let w = theatrical_window();
        assert_eq!(w.duration_weeks(), 12);
    }

    #[test]
    fn test_with_territory() {
        let w = theatrical_window().with_territory("US");
        assert_eq!(w.territory.as_deref(), Some("US"));
    }

    #[test]
    fn test_schedule_add_and_find() {
        let mut sched = WindowSchedule::new();
        sched.add_window(theatrical_window());
        assert!(sched.find("w-theat").is_some());
    }

    #[test]
    fn test_schedule_active_windows() {
        let mut sched = WindowSchedule::new();
        sched.add_window(theatrical_window());
        sched.add_window(streaming_window());
        let active = sched.active_windows(BASE + 1000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "w-theat");
    }

    #[test]
    fn test_schedule_by_type() {
        let mut sched = WindowSchedule::new();
        sched.add_window(theatrical_window());
        sched.add_window(streaming_window());
        let streaming = sched.by_type(&WindowType::Streaming);
        assert_eq!(streaming.len(), 1);
    }

    #[test]
    fn test_schedule_remove() {
        let mut sched = WindowSchedule::new();
        sched.add_window(theatrical_window());
        sched.remove("w-theat");
        assert!(sched.is_empty());
    }

    #[test]
    fn test_schedule_len() {
        let mut sched = WindowSchedule::new();
        sched.add_window(theatrical_window());
        sched.add_window(streaming_window());
        assert_eq!(sched.len(), 2);
    }
}
