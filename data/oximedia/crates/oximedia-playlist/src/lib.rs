//! Broadcast playlist and scheduling system for OxiMedia.
//!
//! This crate provides comprehensive broadcast automation, including:
//!
//! - **Playlist Management**: Create and manage broadcast playlists with frame-accurate timing
//! - **Scheduling Engine**: Time-based playback scheduling with calendar and recurrence support
//! - **Automation**: Automated playout with pre-roll, post-roll, and event triggers
//! - **Secondary Events**: Graphics overlays, station logos, and scrolling tickers
//! - **Transitions**: Smooth transitions with audio/video crossfades
//! - **Live Integration**: Insert live content seamlessly into scheduled playlists
//! - **Failover**: Automatic backup content and filler management
//! - **Clock Sync**: Synchronization to wall clock or external timecode
//! - **Commercial Breaks**: SCTE-35 marker generation and break management
//! - **EPG Generation**: Electronic Program Guide generation with XMLTV export
//! - **As-run Logs**: Metadata tracking and as-run log generation
//! - **Multi-channel**: Support for multiple simultaneous broadcast channels
//!
//! # Playlist Types
//!
//! - **Linear Playout**: Traditional broadcast playlist with scheduled items
//! - **Loop**: Continuous loop of content for fill channels
//! - **Interstitial**: Filler content between programs
//! - **Live-to-Tape**: Scheduled live insertion points
//!
//! # Example
//!
//! ```
//! use oximedia_playlist::playlist::{Playlist, PlaylistItem, PlaylistType};
//! use oximedia_playlist::schedule::ScheduleEngine;
//! use std::time::Duration;
//!
//! // Create a new playlist
//! let mut playlist = Playlist::new("prime_time", PlaylistType::Linear);
//!
//! // Add items to the playlist
//! let item = PlaylistItem::new("show_001.mxf")
//!     .with_duration(Duration::from_secs(3600));
//!
//! playlist.add_item(item);
//!
//! // Create a scheduling engine
//! let engine = ScheduleEngine::new();
//! ```
//!
//! # Features
//!
//! - **Frame-accurate timing**: Precise playback timing for broadcast operations
//! - **Seamless transitions**: Gapless playback with customizable crossfades
//! - **Live integration**: Dynamic insertion of live feeds
//! - **Automatic failover**: Backup content on failure
//! - **SCTE-35 support**: Standard commercial break signaling
//! - **Multi-channel**: Manage multiple channels simultaneously

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_errors_doc)]

pub mod ab_test;
pub mod automation;
pub mod backup;
pub mod clock;
pub mod commercial;
pub mod continuity;
pub mod crossfade;
pub mod crossfade_playlist;
pub mod duration_calc;
pub mod epg;
pub mod gap_filler;
pub mod history;
pub mod interstitial;
/// Augmented interval tree for O(log n) schedule overlap detection.
pub mod interval_tree;
pub mod live;
pub mod m3u;
pub mod m3u8;
pub mod metadata;
pub mod multichannel;
pub mod play_history;
pub mod playlist;
pub mod playlist_archive;
pub mod playlist_diff;
pub mod playlist_export;
pub mod playlist_filter;
pub mod playlist_health;
pub mod playlist_history;
pub mod playlist_merge;
pub mod playlist_notify;
pub mod playlist_priority;
pub mod playlist_rotation;
pub mod playlist_rules;
pub mod playlist_segment;
pub mod playlist_stats;
pub mod playlist_sync;
pub mod playlist_tempo;
pub mod playlist_validator;
pub mod queue_manager;
pub mod recommendation_engine;
pub mod repeat_policy;
pub mod schedule;
pub mod scheduler;
pub mod secondary;
pub mod shuffle;
pub mod smart;
pub mod smart_play;
pub mod track_metadata;
pub mod track_order;
pub mod transition;

pub use automation::PlayoutEngine;
pub use clock::ClockSync;
pub use playlist::{Playlist, PlaylistItem, PlaylistType};
pub use schedule::ScheduleEngine;

use thiserror::Error;

/// Result type for playlist operations.
pub type Result<T> = std::result::Result<T, PlaylistError>;

/// Errors that can occur during playlist operations.
#[derive(Debug, Error)]
pub enum PlaylistError {
    /// Invalid playlist item.
    #[error("Invalid playlist item: {0}")]
    InvalidItem(String),

    /// Scheduling conflict.
    #[error("Scheduling conflict: {0}")]
    SchedulingConflict(String),

    /// Clock synchronization error.
    #[error("Clock sync error: {0}")]
    ClockSyncError(String),

    /// Transition error.
    #[error("Transition error: {0}")]
    TransitionError(String),

    /// Live insertion error.
    #[error("Live insertion error: {0}")]
    LiveInsertionError(String),

    /// Backup failover error.
    #[error("Backup failover error: {0}")]
    FailoverError(String),

    /// SCTE-35 error.
    #[error("SCTE-35 error: {0}")]
    Scte35Error(String),

    /// EPG generation error.
    #[error("EPG generation error: {0}")]
    EpgError(String),

    /// Metadata tracking error.
    #[error("Metadata error: {0}")]
    MetadataError(String),

    /// Multi-channel routing error.
    #[error("Multi-channel routing error: {0}")]
    RoutingError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Core error from oximedia-core.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::error::OxiError),
}
