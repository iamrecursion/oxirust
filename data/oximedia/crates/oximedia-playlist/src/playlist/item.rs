//! Playlist item with in/out points and metadata.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// A single item in a broadcast playlist.
///
/// Each item represents a media asset with optional in/out points,
/// transitions, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistItem {
    /// Unique identifier for this item.
    pub id: String,

    /// Path to the media file.
    pub path: PathBuf,

    /// Display name for this item.
    pub name: String,

    /// In point (start position within the media).
    pub in_point: Option<Duration>,

    /// Out point (end position within the media).
    pub out_point: Option<Duration>,

    /// Duration of the item (calculated from in/out or actual duration).
    pub duration: Duration,

    /// Fade in duration.
    pub fade_in: Option<Duration>,

    /// Fade out duration.
    pub fade_out: Option<Duration>,

    /// Audio level adjustment (in dB).
    pub audio_level: f64,

    /// Whether this item loops.
    pub looping: bool,

    /// Number of times to loop (None = infinite).
    pub loop_count: Option<u32>,

    /// Metadata associated with this item.
    pub metadata: ItemMetadata,

    /// Whether this item is enabled.
    pub enabled: bool,
}

/// Metadata associated with a playlist item.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ItemMetadata {
    /// Title of the content.
    pub title: Option<String>,

    /// Episode number.
    pub episode: Option<u32>,

    /// Season number.
    pub season: Option<u32>,

    /// Content rating (e.g., "TV-PG", "TV-14").
    pub rating: Option<String>,

    /// Genre tags.
    pub genre: Vec<String>,

    /// Description.
    pub description: Option<String>,

    /// Copyright information.
    pub copyright: Option<String>,

    /// Custom metadata fields.
    pub custom: std::collections::HashMap<String, String>,
}

impl PlaylistItem {
    /// Creates a new playlist item from a file path.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_playlist::playlist::PlaylistItem;
    ///
    /// let item = PlaylistItem::new("content/show_001.mxf");
    /// ```
    #[must_use]
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        let path = path.into();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Untitled")
            .to_string();

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            path,
            name,
            in_point: None,
            out_point: None,
            duration: Duration::ZERO,
            fade_in: None,
            fade_out: None,
            audio_level: 0.0,
            looping: false,
            loop_count: None,
            metadata: ItemMetadata::default(),
            enabled: true,
        }
    }

    /// Sets the duration of this item.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Sets the in and out points.
    #[must_use]
    pub fn with_in_out(mut self, in_point: Duration, out_point: Duration) -> Self {
        self.in_point = Some(in_point);
        self.out_point = Some(out_point);
        self.duration = out_point.saturating_sub(in_point);
        self
    }

    /// Sets fade in duration.
    #[must_use]
    pub fn with_fade_in(mut self, duration: Duration) -> Self {
        self.fade_in = Some(duration);
        self
    }

    /// Sets fade out duration.
    #[must_use]
    pub fn with_fade_out(mut self, duration: Duration) -> Self {
        self.fade_out = Some(duration);
        self
    }

    /// Sets audio level adjustment in dB.
    #[must_use]
    pub fn with_audio_level(mut self, db: f64) -> Self {
        self.audio_level = db;
        self
    }

    /// Makes this item loop.
    #[must_use]
    pub fn with_looping(mut self, count: Option<u32>) -> Self {
        self.looping = true;
        self.loop_count = count;
        self
    }

    /// Sets the item title.
    #[must_use]
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.metadata.title = Some(title.into());
        self
    }

    /// Calculates the effective duration including loops.
    #[must_use]
    pub fn effective_duration(&self) -> Duration {
        if self.looping {
            if let Some(count) = self.loop_count {
                self.duration * count
            } else {
                // Infinite loop
                Duration::MAX
            }
        } else {
            self.duration
        }
    }

    /// Returns true if this item is currently enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables this item.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

// We need to add uuid dependency, but for now let's use a simple ID generator
mod uuid {
    pub struct Uuid;

    impl Uuid {
        pub fn new_v4() -> Self {
            Self
        }
    }

    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // Simple ID generation using timestamp and random number
            use std::time::{SystemTime, UNIX_EPOCH};
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            write!(f, "item_{timestamp}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_playlist_item() {
        let item = PlaylistItem::new("test.mxf");
        assert_eq!(item.name, "test");
        assert!(item.is_enabled());
    }

    #[test]
    fn test_item_with_duration() {
        let item = PlaylistItem::new("test.mxf").with_duration(Duration::from_secs(60));
        assert_eq!(item.duration, Duration::from_secs(60));
    }

    #[test]
    fn test_item_with_in_out() {
        let item = PlaylistItem::new("test.mxf")
            .with_in_out(Duration::from_secs(10), Duration::from_secs(70));
        assert_eq!(item.duration, Duration::from_secs(60));
    }

    #[test]
    fn test_effective_duration() {
        let item = PlaylistItem::new("test.mxf")
            .with_duration(Duration::from_secs(60))
            .with_looping(Some(3));
        assert_eq!(item.effective_duration(), Duration::from_secs(180));
    }
}
