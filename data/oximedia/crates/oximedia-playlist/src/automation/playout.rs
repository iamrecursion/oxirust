//! Automated playout engine.

use crate::{Playlist, PlaylistError, Result};
use chrono::{DateTime, Utc};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;

/// State of the playout engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayoutState {
    /// Engine is stopped.
    Stopped,

    /// Engine is playing.
    Playing,

    /// Engine is paused.
    Paused,

    /// Engine is in fast-forward mode.
    FastForward,

    /// Engine is in rewind mode.
    Rewind,
}

/// Event emitted by the playout engine.
#[derive(Debug, Clone)]
pub enum PlayoutEvent {
    /// State changed.
    StateChanged {
        /// Old state.
        old_state: PlayoutState,
        /// New state.
        new_state: PlayoutState,
    },

    /// Item started playing.
    ItemStarted {
        /// Item index.
        item_index: usize,
        /// Timestamp.
        timestamp: DateTime<Utc>,
    },

    /// Item finished playing.
    ItemFinished {
        /// Item index.
        item_index: usize,
        /// Timestamp.
        timestamp: DateTime<Utc>,
    },

    /// Playlist finished.
    PlaylistFinished {
        /// Timestamp.
        timestamp: DateTime<Utc>,
    },

    /// Error occurred.
    Error {
        /// Error message.
        message: String,
    },
}

/// Configuration for playout engine.
#[derive(Debug, Clone)]
pub struct PlayoutConfig {
    /// Whether to loop the playlist.
    pub loop_playlist: bool,

    /// Pre-roll duration.
    pub preroll_duration: Duration,

    /// Post-roll duration.
    pub postroll_duration: Duration,

    /// Whether to enable frame-accurate timing.
    pub frame_accurate: bool,

    /// Target frame rate for accurate timing.
    pub target_fps: f64,
}

impl Default for PlayoutConfig {
    fn default() -> Self {
        Self {
            loop_playlist: false,
            preroll_duration: Duration::ZERO,
            postroll_duration: Duration::ZERO,
            frame_accurate: true,
            target_fps: 25.0,
        }
    }
}

/// Automated playout engine for broadcast playlists.
///
/// Handles frame-accurate playback, transitions, and event generation.
pub struct PlayoutEngine {
    playlist: Arc<RwLock<Option<Playlist>>>,
    state: Arc<RwLock<PlayoutState>>,
    #[allow(dead_code)]
    config: PlayoutConfig,
    current_position: Arc<RwLock<Duration>>,
    event_tx: mpsc::UnboundedSender<PlayoutEvent>,
    event_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<PlayoutEvent>>>>,
}

impl PlayoutEngine {
    /// Creates a new playout engine.
    #[must_use]
    pub fn new(config: PlayoutConfig) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            playlist: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(PlayoutState::Stopped)),
            config,
            current_position: Arc::new(RwLock::new(Duration::ZERO)),
            event_tx,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
        }
    }

    /// Loads a playlist for playback.
    pub fn load_playlist(&self, playlist: Playlist) -> Result<()> {
        let mut pl = self
            .playlist
            .write()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        *pl = Some(playlist);
        Ok(())
    }

    /// Starts playback.
    pub fn play(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        let old_state = *state;
        *state = PlayoutState::Playing;

        let _ = self.event_tx.send(PlayoutEvent::StateChanged {
            old_state,
            new_state: PlayoutState::Playing,
        });

        Ok(())
    }

    /// Pauses playback.
    pub fn pause(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        let old_state = *state;
        *state = PlayoutState::Paused;

        let _ = self.event_tx.send(PlayoutEvent::StateChanged {
            old_state,
            new_state: PlayoutState::Paused,
        });

        Ok(())
    }

    /// Stops playback.
    pub fn stop(&self) -> Result<()> {
        let mut state = self
            .state
            .write()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        let old_state = *state;
        *state = PlayoutState::Stopped;

        // Reset position
        if let Ok(mut pos) = self.current_position.write() {
            *pos = Duration::ZERO;
        }

        let _ = self.event_tx.send(PlayoutEvent::StateChanged {
            old_state,
            new_state: PlayoutState::Stopped,
        });

        Ok(())
    }

    /// Gets the current playback state.
    pub fn get_state(&self) -> Result<PlayoutState> {
        let state = self
            .state
            .read()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        Ok(*state)
    }

    /// Gets the current playback position.
    pub fn get_position(&self) -> Result<Duration> {
        let pos = self
            .current_position
            .read()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        Ok(*pos)
    }

    /// Seeks to a specific position.
    pub fn seek(&self, position: Duration) -> Result<()> {
        let mut pos = self
            .current_position
            .write()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        *pos = position;
        Ok(())
    }

    /// Takes the event receiver (can only be called once).
    pub fn take_event_receiver(&self) -> Option<mpsc::UnboundedReceiver<PlayoutEvent>> {
        self.event_rx.write().ok()?.take()
    }

    /// Emits a playout event.
    pub fn emit_event(&self, event: PlayoutEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Gets the loaded playlist.
    pub fn get_playlist(&self) -> Result<Option<Playlist>> {
        let pl = self
            .playlist
            .read()
            .map_err(|e| PlaylistError::InvalidItem(format!("Lock error: {e}")))?;

        Ok(pl.clone())
    }
}

impl Default for PlayoutEngine {
    fn default() -> Self {
        Self::new(PlayoutConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playlist::PlaylistType;

    #[test]
    fn test_playout_engine() {
        let engine = PlayoutEngine::new(PlayoutConfig::default());
        assert_eq!(
            engine.get_state().expect("should succeed in test"),
            PlayoutState::Stopped
        );

        engine.play().expect("should succeed in test");
        assert_eq!(
            engine.get_state().expect("should succeed in test"),
            PlayoutState::Playing
        );

        engine.pause().expect("should succeed in test");
        assert_eq!(
            engine.get_state().expect("should succeed in test"),
            PlayoutState::Paused
        );

        engine.stop().expect("should succeed in test");
        assert_eq!(
            engine.get_state().expect("should succeed in test"),
            PlayoutState::Stopped
        );
    }

    #[test]
    fn test_load_playlist() {
        let engine = PlayoutEngine::new(PlayoutConfig::default());
        let playlist = Playlist::new("test", PlaylistType::Linear);

        engine
            .load_playlist(playlist)
            .expect("should succeed in test");
        assert!(engine
            .get_playlist()
            .expect("should succeed in test")
            .is_some());
    }
}
