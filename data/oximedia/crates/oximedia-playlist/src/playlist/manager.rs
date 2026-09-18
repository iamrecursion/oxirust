//! Playlist management and types.

use super::item::PlaylistItem;
use crate::{PlaylistError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Type of playlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaylistType {
    /// Traditional linear broadcast playlist.
    Linear,

    /// Continuous loop of content.
    Loop,

    /// Interstitial filler between programs.
    Interstitial,

    /// Scheduled live insertion points.
    LiveToTape,
}

/// A broadcast playlist containing multiple items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playlist {
    /// Unique identifier for this playlist.
    pub id: String,

    /// Name of the playlist.
    pub name: String,

    /// Type of playlist.
    pub playlist_type: PlaylistType,

    /// Items in the playlist.
    pub items: Vec<PlaylistItem>,

    /// Current playback position (item index).
    pub current_position: usize,

    /// Whether the playlist loops.
    pub looping: bool,

    /// Total duration of the playlist.
    pub total_duration: Duration,

    /// Metadata for the playlist.
    pub metadata: PlaylistMetadata,
}

/// Metadata for a playlist.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlaylistMetadata {
    /// Description of the playlist.
    pub description: Option<String>,

    /// Creator/author.
    pub creator: Option<String>,

    /// Creation timestamp.
    pub created: Option<chrono::DateTime<chrono::Utc>>,

    /// Last modified timestamp.
    pub modified: Option<chrono::DateTime<chrono::Utc>>,

    /// Tags for categorization.
    pub tags: Vec<String>,

    /// Custom metadata fields.
    pub custom: std::collections::HashMap<String, String>,
}

impl Playlist {
    /// Creates a new empty playlist.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_playlist::playlist::{Playlist, PlaylistType};
    ///
    /// let playlist = Playlist::new("prime_time", PlaylistType::Linear);
    /// assert_eq!(playlist.name, "prime_time");
    /// ```
    #[must_use]
    pub fn new<S: Into<String>>(name: S, playlist_type: PlaylistType) -> Self {
        Self {
            id: generate_id(),
            name: name.into(),
            playlist_type,
            items: Vec::new(),
            current_position: 0,
            looping: false,
            total_duration: Duration::ZERO,
            metadata: PlaylistMetadata {
                created: Some(chrono::Utc::now()),
                modified: Some(chrono::Utc::now()),
                ..Default::default()
            },
        }
    }

    /// Adds an item to the playlist.
    pub fn add_item(&mut self, item: PlaylistItem) {
        self.total_duration += item.effective_duration();
        self.items.push(item);
        self.metadata.modified = Some(chrono::Utc::now());
    }

    /// Inserts an item at a specific position.
    pub fn insert_item(&mut self, index: usize, item: PlaylistItem) -> Result<()> {
        if index > self.items.len() {
            return Err(PlaylistError::InvalidItem(
                "Index out of bounds".to_string(),
            ));
        }
        self.total_duration += item.effective_duration();
        self.items.insert(index, item);
        self.metadata.modified = Some(chrono::Utc::now());
        Ok(())
    }

    /// Removes an item at a specific position.
    pub fn remove_item(&mut self, index: usize) -> Result<PlaylistItem> {
        if index >= self.items.len() {
            return Err(PlaylistError::InvalidItem(
                "Index out of bounds".to_string(),
            ));
        }
        let item = self.items.remove(index);
        self.total_duration = self
            .total_duration
            .saturating_sub(item.effective_duration());
        self.metadata.modified = Some(chrono::Utc::now());

        // Adjust current position if needed
        if self.current_position > index {
            self.current_position = self.current_position.saturating_sub(1);
        }

        Ok(item)
    }

    /// Gets the current item being played.
    #[must_use]
    pub fn current_item(&self) -> Option<&PlaylistItem> {
        self.items.get(self.current_position)
    }

    /// Advances to the next item.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&PlaylistItem> {
        if self.current_position + 1 < self.items.len() {
            self.current_position += 1;
            self.current_item()
        } else if self.looping {
            self.current_position = 0;
            self.current_item()
        } else {
            None
        }
    }

    /// Goes to the previous item.
    pub fn previous(&mut self) -> Option<&PlaylistItem> {
        if self.current_position > 0 {
            self.current_position -= 1;
            self.current_item()
        } else if self.looping && !self.items.is_empty() {
            self.current_position = self.items.len() - 1;
            self.current_item()
        } else {
            None
        }
    }

    /// Seeks to a specific item by index.
    pub fn seek(&mut self, index: usize) -> Result<&PlaylistItem> {
        if index >= self.items.len() {
            return Err(PlaylistError::InvalidItem(
                "Index out of bounds".to_string(),
            ));
        }
        self.current_position = index;
        Ok(&self.items[index])
    }

    /// Returns the number of items in the playlist.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the playlist is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Clears all items from the playlist.
    pub fn clear(&mut self) {
        self.items.clear();
        self.current_position = 0;
        self.total_duration = Duration::ZERO;
        self.metadata.modified = Some(chrono::Utc::now());
    }

    /// Sets whether the playlist loops.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Recalculates total duration from all items.
    pub fn recalculate_duration(&mut self) {
        self.total_duration = self
            .items
            .iter()
            .map(PlaylistItem::effective_duration)
            .sum();
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("playlist_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_playlist() {
        let playlist = Playlist::new("test", PlaylistType::Linear);
        assert_eq!(playlist.name, "test");
        assert!(playlist.is_empty());
    }

    #[test]
    fn test_add_items() {
        let mut playlist = Playlist::new("test", PlaylistType::Linear);
        playlist.add_item(PlaylistItem::new("item1.mxf").with_duration(Duration::from_secs(60)));
        playlist.add_item(PlaylistItem::new("item2.mxf").with_duration(Duration::from_secs(30)));
        assert_eq!(playlist.len(), 2);
        assert_eq!(playlist.total_duration, Duration::from_secs(90));
    }

    #[test]
    fn test_navigation() {
        let mut playlist = Playlist::new("test", PlaylistType::Linear);
        playlist.add_item(PlaylistItem::new("item1.mxf"));
        playlist.add_item(PlaylistItem::new("item2.mxf"));
        playlist.add_item(PlaylistItem::new("item3.mxf"));

        assert_eq!(playlist.current_position, 0);
        playlist.next();
        assert_eq!(playlist.current_position, 1);
        playlist.next();
        assert_eq!(playlist.current_position, 2);
        playlist.previous();
        assert_eq!(playlist.current_position, 1);
    }

    #[test]
    fn test_looping() {
        let mut playlist = Playlist::new("test", PlaylistType::Loop);
        playlist.set_looping(true);
        playlist.add_item(PlaylistItem::new("item1.mxf"));
        playlist.add_item(PlaylistItem::new("item2.mxf"));

        playlist.current_position = 1;
        playlist.next();
        assert_eq!(playlist.current_position, 0);
    }
}
