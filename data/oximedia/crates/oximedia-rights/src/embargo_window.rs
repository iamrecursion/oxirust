//! Release-window and territorial embargo management.
//!
//! This module handles time-based and territory-based embargo windows,
//! including scheduled unlocking of content for specific regions.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── ReleaseWindow ─────────────────────────────────────────────────────────────

/// A time window during which content is available.
#[derive(Debug, Clone)]
pub struct ReleaseWindow {
    /// Unique window identifier.
    pub id: u32,
    /// Human-readable name (e.g. "Theatrical", "Home Video", "Streaming").
    pub name: String,
    /// Unix timestamp (seconds) at which the window opens.
    pub opens_at: i64,
    /// Unix timestamp (seconds) at which the window closes.
    /// `None` means the window is perpetually open once started.
    pub closes_at: Option<i64>,
}

impl ReleaseWindow {
    /// Create a new `ReleaseWindow`.
    pub fn new(id: u32, name: impl Into<String>, opens_at: i64, closes_at: Option<i64>) -> Self {
        Self {
            id,
            name: name.into(),
            opens_at,
            closes_at,
        }
    }

    /// Return `true` if the window is open at `timestamp`.
    pub fn is_open_at(&self, timestamp: i64) -> bool {
        if timestamp < self.opens_at {
            return false;
        }
        if let Some(close) = self.closes_at {
            return timestamp <= close;
        }
        true
    }

    /// Duration of the window in seconds.  Returns `None` for perpetual windows.
    pub fn duration_seconds(&self) -> Option<i64> {
        self.closes_at.map(|c| c - self.opens_at)
    }
}

// ── TerritorialEmbargoStatus ──────────────────────────────────────────────────

/// Whether content is currently embargoed in a territory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerritorialEmbargoStatus {
    /// Content is embargoed — no distribution allowed.
    Embargoed,
    /// Embargo is scheduled to lift at a future time.
    ScheduledLift,
    /// Content is cleared for distribution.
    Cleared,
}

// ── TerritorialEmbargo ────────────────────────────────────────────────────────

/// An embargo applied to a specific territory.
#[derive(Debug, Clone)]
pub struct TerritorialEmbargo {
    /// ISO 3166-1 alpha-2 territory code (e.g. `"US"`, `"DE"`).
    pub territory_code: String,
    /// Current embargo status.
    pub status: TerritorialEmbargoStatus,
    /// Unix timestamp at which the embargo lifts.  `None` means indefinite.
    pub lift_at: Option<i64>,
    /// Optional reason / note for the embargo.
    pub reason: Option<String>,
}

impl TerritorialEmbargo {
    /// Create a new `TerritorialEmbargo`.
    pub fn new(
        territory_code: impl Into<String>,
        status: TerritorialEmbargoStatus,
        lift_at: Option<i64>,
        reason: Option<String>,
    ) -> Self {
        Self {
            territory_code: territory_code.into(),
            status,
            lift_at,
            reason,
        }
    }

    /// Evaluate the embargo at `now` and return the effective status.
    ///
    /// If a `lift_at` time has passed, the status is considered `Cleared`
    /// regardless of the stored `status`.
    pub fn effective_status(&self, now: i64) -> TerritorialEmbargoStatus {
        if let Some(lift) = self.lift_at {
            if now >= lift {
                return TerritorialEmbargoStatus::Cleared;
            }
        }
        self.status
    }

    /// Return `true` if this embargo blocks distribution at `now`.
    pub fn is_blocking(&self, now: i64) -> bool {
        matches!(
            self.effective_status(now),
            TerritorialEmbargoStatus::Embargoed | TerritorialEmbargoStatus::ScheduledLift
        )
    }
}

// ── EmbargoWindowSchedule ─────────────────────────────────────────────────────

/// Scheduled sequence of release windows for a piece of content.
#[derive(Debug, Default)]
pub struct EmbargoWindowSchedule {
    windows: Vec<ReleaseWindow>,
}

impl EmbargoWindowSchedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a release window to the schedule.
    pub fn add_window(&mut self, window: ReleaseWindow) {
        self.windows.push(window);
    }

    /// Find all windows that are open at `timestamp`.
    pub fn open_windows(&self, timestamp: i64) -> Vec<&ReleaseWindow> {
        self.windows
            .iter()
            .filter(|w| w.is_open_at(timestamp))
            .collect()
    }

    /// Find all windows that have not yet opened at `timestamp`.
    pub fn upcoming_windows(&self, timestamp: i64) -> Vec<&ReleaseWindow> {
        self.windows
            .iter()
            .filter(|w| w.opens_at > timestamp)
            .collect()
    }

    /// Total number of windows.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// `true` if no windows are defined.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

// ── TerritorialEmbargoRegistry ────────────────────────────────────────────────

/// Registry of territorial embargoes for a single asset.
#[derive(Debug, Default)]
pub struct TerritorialEmbargoRegistry {
    embargoes: HashMap<String, TerritorialEmbargo>,
}

impl TerritorialEmbargoRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or replace) the embargo for a territory.
    pub fn set(&mut self, embargo: TerritorialEmbargo) {
        self.embargoes
            .insert(embargo.territory_code.clone(), embargo);
    }

    /// Get the embargo for a territory.
    pub fn get(&self, territory_code: &str) -> Option<&TerritorialEmbargo> {
        self.embargoes.get(territory_code)
    }

    /// Check if distribution is blocked in `territory_code` at `now`.
    /// Returns `true` if the territory has an active embargo or is unknown.
    pub fn is_blocked(&self, territory_code: &str, now: i64) -> bool {
        match self.embargoes.get(territory_code) {
            Some(e) => e.is_blocking(now),
            None => false,
        }
    }

    /// Return all territory codes that are cleared at `now`.
    pub fn cleared_territories(&self, now: i64) -> Vec<&str> {
        self.embargoes
            .iter()
            .filter(|(_, e)| e.effective_status(now) == TerritorialEmbargoStatus::Cleared)
            .map(|(code, _)| code.as_str())
            .collect()
    }

    /// Number of territories in the registry.
    pub fn len(&self) -> usize {
        self.embargoes.len()
    }

    /// `true` if no territories are registered.
    pub fn is_empty(&self) -> bool {
        self.embargoes.is_empty()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ReleaseWindow ────────────────────────────────────────────────────────

    #[test]
    fn test_release_window_is_open_before_open() {
        let w = ReleaseWindow::new(1, "Theatrical", 1000, Some(2000));
        assert!(!w.is_open_at(999));
    }

    #[test]
    fn test_release_window_is_open_within() {
        let w = ReleaseWindow::new(1, "Theatrical", 1000, Some(2000));
        assert!(w.is_open_at(1500));
    }

    #[test]
    fn test_release_window_is_closed_after_close() {
        let w = ReleaseWindow::new(1, "Theatrical", 1000, Some(2000));
        assert!(!w.is_open_at(2001));
    }

    #[test]
    fn test_release_window_perpetual_open_after_open() {
        let w = ReleaseWindow::new(1, "Streaming", 1000, None);
        assert!(w.is_open_at(99_999_999));
    }

    #[test]
    fn test_release_window_duration_seconds() {
        let w = ReleaseWindow::new(1, "Window", 0, Some(3600));
        assert_eq!(w.duration_seconds(), Some(3600));
    }

    #[test]
    fn test_release_window_duration_perpetual() {
        let w = ReleaseWindow::new(1, "Window", 0, None);
        assert!(w.duration_seconds().is_none());
    }

    // ── TerritorialEmbargo ───────────────────────────────────────────────────

    #[test]
    fn test_territorial_embargo_still_active() {
        let e =
            TerritorialEmbargo::new("DE", TerritorialEmbargoStatus::Embargoed, Some(5000), None);
        assert_eq!(
            e.effective_status(1000),
            TerritorialEmbargoStatus::Embargoed
        );
    }

    #[test]
    fn test_territorial_embargo_lifted_by_time() {
        let e =
            TerritorialEmbargo::new("DE", TerritorialEmbargoStatus::Embargoed, Some(5000), None);
        assert_eq!(e.effective_status(5000), TerritorialEmbargoStatus::Cleared);
    }

    #[test]
    fn test_territorial_embargo_is_blocking() {
        let e = TerritorialEmbargo::new(
            "FR",
            TerritorialEmbargoStatus::ScheduledLift,
            Some(9000),
            None,
        );
        assert!(e.is_blocking(8000));
        assert!(!e.is_blocking(10000));
    }

    // ── EmbargoWindowSchedule ────────────────────────────────────────────────

    #[test]
    fn test_embargo_window_schedule_open_windows() {
        let mut sched = EmbargoWindowSchedule::new();
        sched.add_window(ReleaseWindow::new(1, "A", 0, Some(1000)));
        sched.add_window(ReleaseWindow::new(2, "B", 500, Some(1500)));
        let open = sched.open_windows(600);
        assert_eq!(open.len(), 2);
    }

    #[test]
    fn test_embargo_window_schedule_upcoming_windows() {
        let mut sched = EmbargoWindowSchedule::new();
        sched.add_window(ReleaseWindow::new(1, "Now", 0, Some(100)));
        sched.add_window(ReleaseWindow::new(2, "Future", 5000, None));
        let upcoming = sched.upcoming_windows(1000);
        assert_eq!(upcoming.len(), 1);
        assert_eq!(upcoming[0].id, 2);
    }

    #[test]
    fn test_embargo_window_schedule_empty() {
        let sched = EmbargoWindowSchedule::new();
        assert!(sched.is_empty());
        assert_eq!(sched.open_windows(100).len(), 0);
    }

    // ── TerritorialEmbargoRegistry ───────────────────────────────────────────

    #[test]
    fn test_registry_set_and_get() {
        let mut reg = TerritorialEmbargoRegistry::new();
        reg.set(TerritorialEmbargo::new(
            "US",
            TerritorialEmbargoStatus::Cleared,
            None,
            None,
        ));
        assert!(reg.get("US").is_some());
        assert!(reg.get("JP").is_none());
    }

    #[test]
    fn test_registry_is_blocked_embargoed() {
        let mut reg = TerritorialEmbargoRegistry::new();
        reg.set(TerritorialEmbargo::new(
            "CN",
            TerritorialEmbargoStatus::Embargoed,
            None,
            None,
        ));
        assert!(reg.is_blocked("CN", 99999));
        assert!(!reg.is_blocked("US", 99999)); // unknown → not blocked
    }

    #[test]
    fn test_registry_cleared_territories() {
        let mut reg = TerritorialEmbargoRegistry::new();
        reg.set(TerritorialEmbargo::new(
            "US",
            TerritorialEmbargoStatus::Cleared,
            None,
            None,
        ));
        reg.set(TerritorialEmbargo::new(
            "DE",
            TerritorialEmbargoStatus::Embargoed,
            None,
            None,
        ));
        let cleared = reg.cleared_territories(1000);
        assert_eq!(cleared.len(), 1);
        assert_eq!(cleared[0], "US");
    }
}
