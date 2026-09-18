//! Scheduling engine with clock synchronization.

use crate::{Playlist, PlaylistError, Result};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

/// Event emitted by the scheduling engine.
#[derive(Debug, Clone)]
pub enum ScheduleEvent {
    /// Playlist started.
    PlaylistStarted {
        /// Playlist ID.
        playlist_id: String,
        /// Start time.
        start_time: DateTime<Utc>,
    },

    /// Playlist ended.
    PlaylistEnded {
        /// Playlist ID.
        playlist_id: String,
        /// End time.
        end_time: DateTime<Utc>,
    },

    /// Item started.
    ItemStarted {
        /// Playlist ID.
        playlist_id: String,
        /// Item index.
        item_index: usize,
        /// Start time.
        start_time: DateTime<Utc>,
    },

    /// Item ended.
    ItemEnded {
        /// Playlist ID.
        playlist_id: String,
        /// Item index.
        item_index: usize,
        /// End time.
        end_time: DateTime<Utc>,
    },

    /// Schedule conflict detected.
    ConflictDetected {
        /// Conflict message.
        message: String,
    },
}

/// Scheduled playlist entry.
#[derive(Debug, Clone)]
pub struct ScheduledPlaylist {
    /// Playlist to play.
    pub playlist: Playlist,

    /// Scheduled start time.
    pub start_time: DateTime<Utc>,

    /// Priority (higher values have priority).
    pub priority: u32,

    /// Whether this is a recurring schedule.
    pub recurring: bool,
}

/// Scheduling engine for time-based playlist playback.
///
/// Manages scheduled playlists and emits events when playlists
/// and items start/end.
pub struct ScheduleEngine {
    scheduled_playlists: Arc<RwLock<Vec<ScheduledPlaylist>>>,
    event_tx: mpsc::UnboundedSender<ScheduleEvent>,
    event_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<ScheduleEvent>>>>,
}

impl ScheduleEngine {
    /// Creates a new scheduling engine.
    #[must_use]
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            scheduled_playlists: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
        }
    }

    /// Schedules a playlist for playback.
    pub fn schedule_playlist(
        &self,
        playlist: Playlist,
        start_time: DateTime<Utc>,
        priority: u32,
    ) -> Result<()> {
        let mut playlists = self
            .scheduled_playlists
            .write()
            .map_err(|e| PlaylistError::SchedulingConflict(format!("Lock error: {e}")))?;

        // Check for conflicts
        if let Some(conflict) = self.check_conflict(&playlists, &start_time, &playlist) {
            let _ = self.event_tx.send(ScheduleEvent::ConflictDetected {
                message: conflict.clone(),
            });
            return Err(PlaylistError::SchedulingConflict(conflict));
        }

        playlists.push(ScheduledPlaylist {
            playlist,
            start_time,
            priority,
            recurring: false,
        });

        // Sort by start time and priority
        playlists.sort_by(|a, b| {
            a.start_time
                .cmp(&b.start_time)
                .then(b.priority.cmp(&a.priority))
        });

        Ok(())
    }

    /// Removes a scheduled playlist.
    pub fn unschedule_playlist(&self, playlist_id: &str) -> Result<()> {
        let mut playlists = self
            .scheduled_playlists
            .write()
            .map_err(|e| PlaylistError::SchedulingConflict(format!("Lock error: {e}")))?;

        playlists.retain(|sp| sp.playlist.id != playlist_id);
        Ok(())
    }

    /// Gets all scheduled playlists.
    pub fn get_scheduled(&self) -> Result<Vec<ScheduledPlaylist>> {
        let playlists = self
            .scheduled_playlists
            .read()
            .map_err(|e| PlaylistError::SchedulingConflict(format!("Lock error: {e}")))?;

        Ok(playlists.clone())
    }

    /// Gets the currently active playlist.
    pub fn get_active(&self, now: DateTime<Utc>) -> Result<Option<ScheduledPlaylist>> {
        let playlists = self
            .scheduled_playlists
            .read()
            .map_err(|e| PlaylistError::SchedulingConflict(format!("Lock error: {e}")))?;

        // Find the playlist that should be playing now
        for scheduled in playlists.iter() {
            if scheduled.start_time <= now {
                let end_time = scheduled.start_time
                    + chrono::Duration::from_std(scheduled.playlist.total_duration)
                        .unwrap_or_else(|_| chrono::Duration::zero());

                if now < end_time {
                    return Ok(Some(scheduled.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Takes the event receiver (can only be called once).
    pub fn take_event_receiver(&self) -> Option<mpsc::UnboundedReceiver<ScheduleEvent>> {
        self.event_rx.write().ok()?.take()
    }

    /// Emits a schedule event.
    pub fn emit_event(&self, event: ScheduleEvent) {
        let _ = self.event_tx.send(event);
    }

    fn check_conflict(
        &self,
        playlists: &[ScheduledPlaylist],
        start_time: &DateTime<Utc>,
        playlist: &Playlist,
    ) -> Option<String> {
        let end_time = *start_time
            + chrono::Duration::from_std(playlist.total_duration)
                .unwrap_or_else(|_| chrono::Duration::zero());

        for scheduled in playlists {
            let scheduled_end = scheduled.start_time
                + chrono::Duration::from_std(scheduled.playlist.total_duration)
                    .unwrap_or_else(|_| chrono::Duration::zero());

            // Check for overlap
            if start_time < &scheduled_end && end_time > scheduled.start_time {
                return Some(format!(
                    "Playlist '{}' conflicts with '{}'",
                    playlist.name, scheduled.playlist.name
                ));
            }
        }

        None
    }
}

impl Default for ScheduleEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Schedule statistics.
#[derive(Debug, Default)]
pub struct ScheduleStats {
    /// Total scheduled playlists.
    pub total_scheduled: usize,

    /// Active playlists.
    pub active: usize,

    /// Completed playlists.
    pub completed: usize,

    /// Conflicts detected.
    pub conflicts: usize,
}

/// Schedule manager for multiple channels.
pub struct ScheduleManager {
    engines: HashMap<String, ScheduleEngine>,
}

impl ScheduleManager {
    /// Creates a new schedule manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            engines: HashMap::new(),
        }
    }

    /// Adds a channel.
    pub fn add_channel(&mut self, channel_id: String) {
        self.engines.insert(channel_id, ScheduleEngine::new());
    }

    /// Gets the engine for a channel.
    pub fn get_engine(&self, channel_id: &str) -> Option<&ScheduleEngine> {
        self.engines.get(channel_id)
    }

    /// Gets a mutable engine for a channel.
    pub fn get_engine_mut(&mut self, channel_id: &str) -> Option<&mut ScheduleEngine> {
        self.engines.get_mut(channel_id)
    }
}

impl Default for ScheduleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playlist::PlaylistType;

    #[test]
    fn test_schedule_engine() {
        let engine = ScheduleEngine::new();
        let playlist = Playlist::new("test", PlaylistType::Linear);
        let start_time = Utc::now();

        engine
            .schedule_playlist(playlist, start_time, 1)
            .expect("should succeed in test");
        let scheduled = engine.get_scheduled().expect("should succeed in test");
        assert_eq!(scheduled.len(), 1);
    }

    #[test]
    fn test_schedule_manager() {
        let mut manager = ScheduleManager::new();
        manager.add_channel("channel1".to_string());
        assert!(manager.get_engine("channel1").is_some());
    }
}
