//! Commercial break management.

use super::scte35::Scte35Marker;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Commercial break configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommercialBreak {
    /// Unique identifier.
    pub id: String,

    /// Display name.
    pub name: String,

    /// Scheduled time for the break.
    pub scheduled_time: DateTime<Utc>,

    /// Duration of the break.
    pub duration: Duration,

    /// SCTE-35 marker to insert.
    pub scte35_marker: Option<Scte35Marker>,

    /// Number of commercial spots.
    pub spot_count: u32,

    /// Whether this break is mandatory.
    pub mandatory: bool,

    /// Priority (higher = more important).
    pub priority: u32,

    /// Whether this break has been executed.
    pub executed: bool,
}

impl CommercialBreak {
    /// Creates a new commercial break.
    #[must_use]
    pub fn new<S: Into<String>>(
        name: S,
        scheduled_time: DateTime<Utc>,
        duration: Duration,
    ) -> Self {
        Self {
            id: generate_id(),
            name: name.into(),
            scheduled_time,
            duration,
            scte35_marker: None,
            spot_count: 0,
            mandatory: false,
            priority: 0,
            executed: false,
        }
    }

    /// Sets the SCTE-35 marker.
    #[must_use]
    pub fn with_scte35_marker(mut self, marker: Scte35Marker) -> Self {
        self.scte35_marker = Some(marker);
        self
    }

    /// Sets the number of spots.
    #[must_use]
    pub const fn with_spot_count(mut self, count: u32) -> Self {
        self.spot_count = count;
        self
    }

    /// Makes this break mandatory.
    #[must_use]
    pub const fn as_mandatory(mut self) -> Self {
        self.mandatory = true;
        self
    }

    /// Sets the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Marks this break as executed.
    pub fn mark_executed(&mut self) {
        self.executed = true;
    }

    /// Checks if this break should execute at the given time.
    #[must_use]
    pub fn should_execute(&self, time: &DateTime<Utc>) -> bool {
        !self.executed && time >= &self.scheduled_time
    }
}

/// Manager for commercial breaks.
#[derive(Debug, Default)]
pub struct BreakManager {
    breaks: Vec<CommercialBreak>,
}

impl BreakManager {
    /// Creates a new break manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a commercial break.
    pub fn add_break(&mut self, commercial_break: CommercialBreak) {
        self.breaks.push(commercial_break);
        self.sort_by_time();
    }

    /// Removes a break by ID.
    pub fn remove_break(&mut self, break_id: &str) {
        self.breaks.retain(|b| b.id != break_id);
    }

    /// Gets all scheduled breaks.
    #[must_use]
    pub fn get_scheduled_breaks(&self) -> Vec<&CommercialBreak> {
        self.breaks.iter().filter(|b| !b.executed).collect()
    }

    /// Gets breaks that should execute at the given time.
    #[must_use]
    pub fn get_breaks_to_execute(&self, time: &DateTime<Utc>) -> Vec<&CommercialBreak> {
        self.breaks
            .iter()
            .filter(|b| b.should_execute(time))
            .collect()
    }

    /// Marks a break as executed.
    pub fn mark_executed(&mut self, break_id: &str) {
        if let Some(commercial_break) = self.breaks.iter_mut().find(|b| b.id == break_id) {
            commercial_break.mark_executed();
        }
    }

    /// Gets the next scheduled break.
    #[must_use]
    pub fn get_next_break(&self, after: &DateTime<Utc>) -> Option<&CommercialBreak> {
        self.breaks
            .iter()
            .filter(|b| !b.executed && b.scheduled_time > *after)
            .min_by_key(|b| b.scheduled_time)
    }

    /// Calculates total break duration in a time range.
    #[must_use]
    pub fn total_break_duration(&self, start: &DateTime<Utc>, end: &DateTime<Utc>) -> Duration {
        self.breaks
            .iter()
            .filter(|b| !b.executed && b.scheduled_time >= *start && b.scheduled_time < *end)
            .map(|b| b.duration)
            .sum()
    }

    /// Sorts breaks by scheduled time.
    fn sort_by_time(&mut self) {
        self.breaks
            .sort_by(|a, b| a.scheduled_time.cmp(&b.scheduled_time));
    }

    /// Returns the number of breaks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.breaks.len()
    }

    /// Returns true if there are no breaks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.breaks.is_empty()
    }

    /// Clears all executed breaks.
    pub fn clear_executed(&mut self) {
        self.breaks.retain(|b| !b.executed);
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("break_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commercial_break() {
        let mut commercial_break =
            CommercialBreak::new("Prime Time Break", Utc::now(), Duration::from_secs(120))
                .with_spot_count(4)
                .as_mandatory();

        assert!(!commercial_break.executed);
        commercial_break.mark_executed();
        assert!(commercial_break.executed);
    }

    #[test]
    fn test_break_manager() {
        let mut manager = BreakManager::new();

        let now = Utc::now();
        let break1 = CommercialBreak::new("Break 1", now, Duration::from_secs(60));
        let break2 = CommercialBreak::new(
            "Break 2",
            now + chrono::Duration::minutes(30),
            Duration::from_secs(90),
        );

        manager.add_break(break1);
        manager.add_break(break2);

        assert_eq!(manager.len(), 2);

        let to_execute = manager.get_breaks_to_execute(&now);
        assert_eq!(to_execute.len(), 1);
    }

    #[test]
    fn test_next_break() {
        let mut manager = BreakManager::new();
        let now = Utc::now();

        let break1 = CommercialBreak::new(
            "Future Break",
            now + chrono::Duration::hours(1),
            Duration::from_secs(120),
        );

        manager.add_break(break1);

        let next = manager.get_next_break(&now);
        assert!(next.is_some());
    }
}
