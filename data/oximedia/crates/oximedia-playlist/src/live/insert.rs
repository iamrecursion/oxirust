//! Live content insertion into playlists.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Source for live content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiveSource {
    /// NDI source.
    Ndi {
        /// NDI source name.
        name: String,
    },

    /// SDI input.
    Sdi {
        /// SDI input number.
        input: u32,
    },

    /// RTMP stream.
    Rtmp {
        /// RTMP URL.
        url: String,
    },

    /// SRT stream.
    Srt {
        /// SRT URL.
        url: String,
    },

    /// Custom source.
    Custom {
        /// Source identifier.
        id: String,
    },
}

/// Live insertion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveInsert {
    /// Unique identifier.
    pub id: String,

    /// Display name.
    pub name: String,

    /// Live source.
    pub source: LiveSource,

    /// Scheduled start time (None = manual trigger).
    pub start_time: Option<DateTime<Utc>>,

    /// Maximum duration (None = manual stop).
    pub max_duration: Option<Duration>,

    /// Priority level (higher = more important).
    pub priority: u32,

    /// Whether to interrupt current item.
    pub interrupt: bool,

    /// Whether to return to playlist after live insert.
    pub return_to_playlist: bool,

    /// Pre-roll before live content.
    pub preroll_duration: Option<Duration>,

    /// Post-roll after live content.
    pub postroll_duration: Option<Duration>,

    /// Whether this insert is active.
    pub active: bool,
}

impl LiveInsert {
    /// Creates a new live insert.
    #[must_use]
    pub fn new<S: Into<String>>(name: S, source: LiveSource) -> Self {
        Self {
            id: generate_id(),
            name: name.into(),
            source,
            start_time: None,
            max_duration: None,
            priority: 0,
            interrupt: false,
            return_to_playlist: true,
            preroll_duration: None,
            postroll_duration: None,
            active: false,
        }
    }

    /// Sets the start time.
    #[must_use]
    pub const fn with_start_time(mut self, time: DateTime<Utc>) -> Self {
        self.start_time = Some(time);
        self
    }

    /// Sets the maximum duration.
    #[must_use]
    pub const fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Sets the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Makes this insert interrupt current playback.
    #[must_use]
    pub const fn as_interrupt(mut self) -> Self {
        self.interrupt = true;
        self
    }

    /// Sets whether to return to playlist after.
    #[must_use]
    pub const fn with_return_to_playlist(mut self, return_to: bool) -> Self {
        self.return_to_playlist = return_to;
        self
    }

    /// Activates this live insert.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivates this live insert.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Checks if this insert should be active at the given time.
    #[must_use]
    pub fn should_activate(&self, time: &DateTime<Utc>) -> bool {
        if let Some(start) = self.start_time {
            if time < &start {
                return false;
            }

            if let Some(max_dur) = self.max_duration {
                let end = start
                    + chrono::Duration::from_std(max_dur)
                        .unwrap_or_else(|_| chrono::Duration::zero());
                if time >= &end {
                    return false;
                }
            }

            return true;
        }

        false
    }
}

/// Manager for live inserts.
#[derive(Debug, Default)]
pub struct LiveInsertManager {
    inserts: Vec<LiveInsert>,
}

impl LiveInsertManager {
    /// Creates a new live insert manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a live insert.
    pub fn add_insert(&mut self, insert: LiveInsert) {
        self.inserts.push(insert);
        self.sort_by_priority();
    }

    /// Removes a live insert by ID.
    pub fn remove_insert(&mut self, insert_id: &str) {
        self.inserts.retain(|i| i.id != insert_id);
    }

    /// Activates a live insert by ID.
    pub fn activate_insert(&mut self, insert_id: &str) {
        if let Some(insert) = self.inserts.iter_mut().find(|i| i.id == insert_id) {
            insert.activate();
        }
    }

    /// Deactivates a live insert by ID.
    pub fn deactivate_insert(&mut self, insert_id: &str) {
        if let Some(insert) = self.inserts.iter_mut().find(|i| i.id == insert_id) {
            insert.deactivate();
        }
    }

    /// Gets all active inserts.
    #[must_use]
    pub fn get_active_inserts(&self) -> Vec<&LiveInsert> {
        self.inserts.iter().filter(|i| i.active).collect()
    }

    /// Gets the highest priority active insert.
    #[must_use]
    pub fn get_highest_priority_active(&self) -> Option<&LiveInsert> {
        self.inserts
            .iter()
            .filter(|i| i.active)
            .max_by_key(|i| i.priority)
    }

    /// Checks for scheduled inserts at a given time.
    #[must_use]
    pub fn check_scheduled_inserts(&self, time: &DateTime<Utc>) -> Vec<&LiveInsert> {
        self.inserts
            .iter()
            .filter(|i| !i.active && i.should_activate(time))
            .collect()
    }

    /// Sorts inserts by priority (highest first).
    fn sort_by_priority(&mut self) {
        self.inserts.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Returns the number of inserts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inserts.len()
    }

    /// Returns true if there are no inserts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inserts.is_empty()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("live_insert_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_insert() {
        let mut insert = LiveInsert::new(
            "breaking_news",
            LiveSource::Ndi {
                name: "NEWS_CAM".to_string(),
            },
        )
        .with_priority(10)
        .as_interrupt();

        assert!(!insert.active);
        insert.activate();
        assert!(insert.active);
        assert!(insert.interrupt);
    }

    #[test]
    fn test_live_insert_manager() {
        let mut manager = LiveInsertManager::new();

        let insert1 = LiveInsert::new("insert1", LiveSource::Sdi { input: 1 }).with_priority(5);
        let insert2 = LiveInsert::new("insert2", LiveSource::Sdi { input: 2 }).with_priority(10);

        let insert1_id = insert1.id.clone();
        let insert2_id = insert2.id.clone();

        manager.add_insert(insert1);
        manager.add_insert(insert2);

        manager.activate_insert(&insert1_id);
        manager.activate_insert(&insert2_id);

        let highest = manager
            .get_highest_priority_active()
            .expect("should succeed in test");
        assert_eq!(highest.priority, 10);
    }
}
