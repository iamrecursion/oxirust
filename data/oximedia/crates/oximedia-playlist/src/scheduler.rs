//! Broadcast scheduling engine for managing time-based broadcast events.
//!
//! Provides a [`BroadcastScheduler`] that manages scheduled events on a timeline,
//! detects conflicts, fills gaps with filler content, and generates schedule reports.

use crate::{PlaylistError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Status of a scheduled broadcast event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventStatus {
    /// Waiting to be played.
    Pending,
    /// Currently playing.
    Playing,
    /// Finished playing.
    Completed,
    /// Failed to play.
    Failed,
    /// Skipped (e.g., due to a higher-priority event).
    Skipped,
}

impl std::fmt::Display for EventStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Playing => write!(f, "Playing"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Skipped => write!(f, "Skipped"),
        }
    }
}

/// A scheduled broadcast event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// Unique identifier for this event.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Identifier of the media asset to play.
    pub asset_id: String,
    /// Scheduled start time as a Unix timestamp (seconds).
    pub scheduled_start: u64,
    /// Scheduled end time as a Unix timestamp (seconds).
    pub scheduled_end: u64,
    /// Actual start time (set when the event begins playing).
    pub actual_start: Option<u64>,
    /// Current status of this event.
    pub status: EventStatus,
    /// Priority (higher values take precedence on conflict).
    pub priority: u8,
    /// Number of times to loop: 0 = play once, [`u32::MAX`] = loop forever.
    pub loop_count: u32,
    /// Arbitrary tags for filtering and categorisation.
    pub tags: Vec<String>,
}

impl ScheduledEvent {
    /// Creates a new scheduled event.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_playlist::scheduler::{ScheduledEvent, EventStatus};
    ///
    /// let event = ScheduledEvent::new(
    ///     "evt_001",
    ///     "Morning News",
    ///     "asset_news_001",
    ///     1_700_000_000,
    ///     1_700_003_600,
    /// );
    /// assert_eq!(event.status, EventStatus::Pending);
    /// ```
    #[must_use]
    pub fn new<S: Into<String>>(
        id: S,
        title: S,
        asset_id: S,
        scheduled_start: u64,
        scheduled_end: u64,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            asset_id: asset_id.into(),
            scheduled_start,
            scheduled_end,
            actual_start: None,
            status: EventStatus::Pending,
            priority: 128,
            loop_count: 0,
            tags: Vec::new(),
        }
    }

    /// Sets the priority of this event.
    #[must_use]
    pub const fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the loop count (0 = once, [`u32::MAX`] = forever).
    #[must_use]
    pub const fn with_loop_count(mut self, count: u32) -> Self {
        self.loop_count = count;
        self
    }

    /// Adds a tag to this event.
    #[must_use]
    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Returns the nominal duration of the event in seconds.
    #[must_use]
    pub const fn duration_secs(&self) -> u64 {
        self.scheduled_end.saturating_sub(self.scheduled_start)
    }

    /// Returns `true` if this event overlaps with another event.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.scheduled_start < other.scheduled_end && other.scheduled_start < self.scheduled_end
    }
}

/// Errors specific to the broadcast scheduler.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// An event with the given ID already exists.
    #[error("Duplicate event ID: {0}")]
    DuplicateId(String),
    /// The event time range is invalid (end <= start).
    #[error("Invalid time range for event {0}: end must be after start")]
    InvalidTimeRange(String),
    /// The event was not found.
    #[error("Event not found: {0}")]
    NotFound(String),
}

impl From<SchedulerError> for PlaylistError {
    fn from(e: SchedulerError) -> Self {
        PlaylistError::SchedulingConflict(e.to_string())
    }
}

/// Summary of a schedule report for a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleReport {
    /// Start of the reporting window (Unix timestamp).
    pub from_ts: u64,
    /// End of the reporting window (Unix timestamp).
    pub to_ts: u64,
    /// All events within the window, ordered by scheduled start.
    pub events: Vec<ScheduledEvent>,
    /// Total scheduled duration in seconds (sum of all event durations).
    pub total_scheduled_secs: u64,
    /// Total gap time in seconds within the window.
    pub total_gap_secs: u64,
    /// Number of conflicts detected.
    pub conflict_count: usize,
    /// Pairs of conflicting event IDs.
    pub conflicts: Vec<(String, String)>,
}

/// Broadcast scheduling engine.
///
/// Manages a timeline of [`ScheduledEvent`]s, provides conflict detection,
/// gap filling, and reporting.
pub struct BroadcastScheduler {
    /// All registered events keyed by ID.
    events: HashMap<String, ScheduledEvent>,
    /// Index: scheduled_start timestamp → list of event IDs starting at that time.
    timeline: BTreeMap<u64, Vec<String>>,
}

impl BroadcastScheduler {
    /// Creates an empty scheduler.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_playlist::scheduler::BroadcastScheduler;
    ///
    /// let scheduler = BroadcastScheduler::new();
    /// assert_eq!(scheduler.event_count(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: HashMap::new(),
            timeline: BTreeMap::new(),
        }
    }

    /// Returns the total number of registered events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Adds an event to the schedule.
    ///
    /// Returns [`SchedulerError::DuplicateId`] if an event with the same ID already
    /// exists, and [`SchedulerError::InvalidTimeRange`] if `scheduled_end <= scheduled_start`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_playlist::scheduler::{BroadcastScheduler, ScheduledEvent};
    ///
    /// let mut scheduler = BroadcastScheduler::new();
    /// let event = ScheduledEvent::new("e1", "Show", "asset1", 1000, 2000);
    /// scheduler.add_event(event).expect("valid event");
    /// assert_eq!(scheduler.event_count(), 1);
    /// ```
    pub fn add_event(&mut self, event: ScheduledEvent) -> Result<()> {
        if event.scheduled_end <= event.scheduled_start {
            return Err(SchedulerError::InvalidTimeRange(event.id).into());
        }
        if self.events.contains_key(&event.id) {
            return Err(SchedulerError::DuplicateId(event.id).into());
        }
        let start = event.scheduled_start;
        let id = event.id.clone();
        self.events.insert(id.clone(), event);
        self.timeline.entry(start).or_default().push(id);
        Ok(())
    }

    /// Removes the event with the given ID.
    ///
    /// Returns `true` if the event was found and removed, `false` otherwise.
    pub fn remove_event(&mut self, id: &str) -> bool {
        if let Some(event) = self.events.remove(id) {
            if let Some(ids) = self.timeline.get_mut(&event.scheduled_start) {
                ids.retain(|eid| eid != id);
                if ids.is_empty() {
                    self.timeline.remove(&event.scheduled_start);
                }
            }
            true
        } else {
            false
        }
    }

    /// Returns a reference to an event by ID.
    #[must_use]
    pub fn get_event(&self, id: &str) -> Option<&ScheduledEvent> {
        self.events.get(id)
    }

    /// Returns the event that should be playing at `now_ts`, or `None` if no event
    /// covers that timestamp.
    ///
    /// When multiple events cover the same time, the one with the highest priority
    /// is returned.
    #[must_use]
    pub fn current_event(&self, now_ts: u64) -> Option<&ScheduledEvent> {
        // Collect all events whose window contains now_ts
        let candidates: Vec<&ScheduledEvent> = self
            .events
            .values()
            .filter(|e| e.scheduled_start <= now_ts && now_ts < e.scheduled_end)
            .collect();

        candidates
            .into_iter()
            .max_by_key(|e| (e.priority, e.scheduled_start))
    }

    /// Returns all events that start within `[now_ts, now_ts + window_secs)`,
    /// ordered by `scheduled_start`.
    #[must_use]
    pub fn upcoming(&self, now_ts: u64, window_secs: u64) -> Vec<&ScheduledEvent> {
        let window_end = now_ts.saturating_add(window_secs);
        let mut result: Vec<&ScheduledEvent> = self
            .timeline
            .range(now_ts..window_end)
            .flat_map(|(_, ids)| ids.iter().filter_map(|id| self.events.get(id)))
            .collect();
        result.sort_by_key(|e| e.scheduled_start);
        result
    }

    /// Fills gaps in the schedule with filler content.
    ///
    /// A gap is a period of at least `gap_secs_min` seconds between consecutive
    /// events where no event is scheduled.  A new filler event is inserted for
    /// each such gap.
    pub fn fill_gaps(&mut self, filler_asset_id: &str, gap_secs_min: u64) {
        // Gather a sorted list of (start, end) pairs from existing events.
        let mut windows: Vec<(u64, u64)> = self
            .events
            .values()
            .map(|e| (e.scheduled_start, e.scheduled_end))
            .collect();
        windows.sort_by_key(|&(s, _)| s);

        let mut fillers: Vec<ScheduledEvent> = Vec::new();
        let mut filler_idx: u32 = 0;
        let mut prev_end: Option<u64> = None;

        for (start, end) in &windows {
            if let Some(pe) = prev_end {
                if *start > pe {
                    let gap = *start - pe;
                    if gap >= gap_secs_min {
                        let fid = format!("filler_{filler_idx}");
                        filler_idx += 1;
                        let filler = ScheduledEvent {
                            id: fid,
                            title: "Filler".to_string(),
                            asset_id: filler_asset_id.to_string(),
                            scheduled_start: pe,
                            scheduled_end: *start,
                            actual_start: None,
                            status: EventStatus::Pending,
                            priority: 0,
                            loop_count: 0,
                            tags: vec!["filler".to_string()],
                        };
                        fillers.push(filler);
                    }
                }
            }
            let new_end = (*end).max(prev_end.unwrap_or(0));
            prev_end = Some(new_end);
        }

        for filler in fillers {
            let start = filler.scheduled_start;
            let id = filler.id.clone();
            self.events.insert(id.clone(), filler);
            self.timeline.entry(start).or_default().push(id);
        }
    }

    /// Detects overlapping (conflicting) events.
    ///
    /// Returns a list of ID pairs `(a, b)` where `a` and `b` overlap.  Each
    /// conflicting pair is returned exactly once.
    #[must_use]
    pub fn conflicts(&self) -> Vec<(String, String)> {
        let mut events: Vec<&ScheduledEvent> = self.events.values().collect();
        events.sort_by_key(|e| e.scheduled_start);

        let mut result = Vec::new();
        for i in 0..events.len() {
            for j in (i + 1)..events.len() {
                let a = events[i];
                let b = events[j];
                // Events are sorted by start; if b starts after a ends, no further overlap.
                if b.scheduled_start >= a.scheduled_end {
                    break;
                }
                if a.overlaps(b) {
                    result.push((a.id.clone(), b.id.clone()));
                }
            }
        }
        result
    }

    /// Marks an event as started.
    ///
    /// Sets `actual_start` and transitions status to [`EventStatus::Playing`].
    /// Returns `true` if the event was found.
    pub fn mark_started(&mut self, id: &str, actual_start: u64) -> bool {
        if let Some(event) = self.events.get_mut(id) {
            event.actual_start = Some(actual_start);
            event.status = EventStatus::Playing;
            true
        } else {
            false
        }
    }

    /// Marks an event as completed.
    ///
    /// Transitions status to [`EventStatus::Completed`].
    /// Returns `true` if the event was found.
    pub fn mark_completed(&mut self, id: &str) -> bool {
        if let Some(event) = self.events.get_mut(id) {
            event.status = EventStatus::Completed;
            true
        } else {
            false
        }
    }

    /// Marks an event as failed.
    ///
    /// Returns `true` if the event was found.
    pub fn mark_failed(&mut self, id: &str) -> bool {
        if let Some(event) = self.events.get_mut(id) {
            event.status = EventStatus::Failed;
            true
        } else {
            false
        }
    }

    /// Marks an event as skipped.
    ///
    /// Returns `true` if the event was found.
    pub fn mark_skipped(&mut self, id: &str) -> bool {
        if let Some(event) = self.events.get_mut(id) {
            event.status = EventStatus::Skipped;
            true
        } else {
            false
        }
    }

    /// Generates a [`ScheduleReport`] for events in `[from_ts, to_ts]`.
    #[must_use]
    pub fn report(&self, from_ts: u64, to_ts: u64) -> ScheduleReport {
        // Collect events that overlap the requested window
        let mut events: Vec<ScheduledEvent> = self
            .events
            .values()
            .filter(|e| e.scheduled_start < to_ts && e.scheduled_end > from_ts)
            .cloned()
            .collect();
        events.sort_by_key(|e| e.scheduled_start);

        let total_scheduled_secs: u64 = events
            .iter()
            .map(|e| {
                let clamped_start = e.scheduled_start.max(from_ts);
                let clamped_end = e.scheduled_end.min(to_ts);
                clamped_end.saturating_sub(clamped_start)
            })
            .sum();

        let window = to_ts.saturating_sub(from_ts);
        let total_gap_secs = window.saturating_sub(total_scheduled_secs);

        let conflicts = self.conflicts();
        let conflict_count = conflicts.len();

        ScheduleReport {
            from_ts,
            to_ts,
            events,
            total_scheduled_secs,
            total_gap_secs,
            conflict_count,
            conflicts,
        }
    }

    /// Returns all events sorted by scheduled start time.
    #[must_use]
    pub fn all_events_sorted(&self) -> Vec<&ScheduledEvent> {
        let mut events: Vec<&ScheduledEvent> = self.events.values().collect();
        events.sort_by_key(|e| e.scheduled_start);
        events
    }

    /// Returns events filtered by tag.
    #[must_use]
    pub fn events_with_tag(&self, tag: &str) -> Vec<&ScheduledEvent> {
        self.events
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Returns events filtered by status.
    #[must_use]
    pub fn events_with_status(&self, status: &EventStatus) -> Vec<&ScheduledEvent> {
        self.events
            .values()
            .filter(|e| &e.status == status)
            .collect()
    }

    /// Clears all events from the schedule.
    pub fn clear(&mut self) {
        self.events.clear();
        self.timeline.clear();
    }

    /// Exports the full schedule as a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails.
    pub fn export_json(&self) -> Result<String> {
        let events = self.all_events_sorted();
        serde_json::to_string_pretty(&events)
            .map_err(|e| PlaylistError::MetadataError(format!("JSON export failed: {e}")))
    }
}

impl Default for BroadcastScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: &str, start: u64, end: u64) -> ScheduledEvent {
        ScheduledEvent::new(id, "Test Event", "asset_001", start, end)
    }

    #[test]
    fn test_new_scheduler() {
        let sched = BroadcastScheduler::new();
        assert_eq!(sched.event_count(), 0);
    }

    #[test]
    fn test_add_and_count() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 2000, 3000))
            .expect("should succeed in test");
        assert_eq!(sched.event_count(), 2);
    }

    #[test]
    fn test_add_duplicate_id_fails() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        let err = sched.add_event(make_event("e1", 3000, 4000));
        assert!(err.is_err());
    }

    #[test]
    fn test_add_invalid_range_fails() {
        let mut sched = BroadcastScheduler::new();
        let err = sched.add_event(make_event("e1", 2000, 1000));
        assert!(err.is_err());
    }

    #[test]
    fn test_remove_event() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        assert!(sched.remove_event("e1"));
        assert_eq!(sched.event_count(), 0);
        assert!(!sched.remove_event("e1")); // already gone
    }

    #[test]
    fn test_current_event() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 2000, 3000))
            .expect("should succeed in test");

        assert!(sched.current_event(500).is_none());
        assert_eq!(sched.current_event(1000).map(|e| e.id.as_str()), Some("e1"));
        assert_eq!(sched.current_event(1999).map(|e| e.id.as_str()), Some("e1"));
        assert_eq!(sched.current_event(2000).map(|e| e.id.as_str()), Some("e2"));
        assert!(sched.current_event(3000).is_none());
    }

    #[test]
    fn test_upcoming() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 1500, 2500))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e3", 3000, 4000))
            .expect("should succeed in test");

        let up = sched.upcoming(900, 700); // window [900..1600)
        let ids: Vec<&str> = up.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e1"));
        assert!(ids.contains(&"e2"));
        assert!(!ids.contains(&"e3"));
    }

    #[test]
    fn test_conflicts_none() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 2000, 3000))
            .expect("should succeed in test");
        assert!(sched.conflicts().is_empty());
    }

    #[test]
    fn test_conflicts_detected() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 3000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 2000, 4000))
            .expect("should succeed in test");
        let conflicts = sched.conflicts();
        assert_eq!(conflicts.len(), 1);
        let (a, b) = &conflicts[0];
        assert!((a == "e1" && b == "e2") || (a == "e2" && b == "e1"));
    }

    #[test]
    fn test_fill_gaps() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 0, 1000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 2000, 3000))
            .expect("should succeed in test");
        sched.fill_gaps("filler_asset", 100);
        // A filler event covering [1000, 2000) should have been inserted
        let filler_events = sched.events_with_tag("filler");
        assert!(!filler_events.is_empty());
        let filler = filler_events[0];
        assert_eq!(filler.scheduled_start, 1000);
        assert_eq!(filler.scheduled_end, 2000);
    }

    #[test]
    fn test_fill_gaps_ignores_small_gaps() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 0, 1000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 1050, 2000))
            .expect("should succeed in test");
        // gap is 50s, minimum is 100s → should NOT fill
        sched.fill_gaps("filler_asset", 100);
        assert!(sched.events_with_tag("filler").is_empty());
    }

    #[test]
    fn test_mark_started_and_completed() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 1000, 2000))
            .expect("should succeed in test");
        assert!(sched.mark_started("e1", 1005));
        let event = sched.get_event("e1").expect("should succeed in test");
        assert_eq!(event.status, EventStatus::Playing);
        assert_eq!(event.actual_start, Some(1005));

        assert!(sched.mark_completed("e1"));
        let event = sched.get_event("e1").expect("should succeed in test");
        assert_eq!(event.status, EventStatus::Completed);
    }

    #[test]
    fn test_mark_nonexistent_returns_false() {
        let mut sched = BroadcastScheduler::new();
        assert!(!sched.mark_started("nonexistent", 100));
        assert!(!sched.mark_completed("nonexistent"));
    }

    #[test]
    fn test_report() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 0, 1000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 1500, 2000))
            .expect("should succeed in test");
        let report = sched.report(0, 2000);
        assert_eq!(report.events.len(), 2);
        assert_eq!(report.total_scheduled_secs, 1500); // 1000 + 500
        assert_eq!(report.total_gap_secs, 500); // 2000 - 1500
        assert_eq!(report.conflict_count, 0);
    }

    #[test]
    fn test_events_with_status() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 0, 1000))
            .expect("should succeed in test");
        sched
            .add_event(make_event("e2", 1000, 2000))
            .expect("should succeed in test");
        sched.mark_completed("e1");
        let completed = sched.events_with_status(&EventStatus::Completed);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, "e1");
    }

    #[test]
    fn test_priority_resolution() {
        let mut sched = BroadcastScheduler::new();
        let low = ScheduledEvent::new("low", "Low", "a1", 1000, 2000).with_priority(10);
        let high = ScheduledEvent::new("high", "High", "a2", 1000, 2000).with_priority(200);
        sched.add_event(low).expect("should succeed in test");
        sched.add_event(high).expect("should succeed in test");
        let current = sched.current_event(1500).expect("should succeed in test");
        assert_eq!(current.id, "high");
    }

    #[test]
    fn test_export_json() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 0, 1000))
            .expect("should succeed in test");
        let json = sched.export_json().expect("should succeed in test");
        assert!(json.contains("e1"));
    }

    #[test]
    fn test_clear() {
        let mut sched = BroadcastScheduler::new();
        sched
            .add_event(make_event("e1", 0, 1000))
            .expect("should succeed in test");
        sched.clear();
        assert_eq!(sched.event_count(), 0);
    }

    #[test]
    fn test_event_duration() {
        let event = make_event("e1", 1000, 4600);
        assert_eq!(event.duration_secs(), 3600);
    }

    #[test]
    fn test_event_overlaps() {
        let a = make_event("a", 1000, 3000);
        let b = make_event("b", 2000, 4000);
        let c = make_event("c", 3000, 5000);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }
}
