//! Fluent builder for creating playlists.

use super::{Playlist, PlaylistItem, PlaylistType};

/// Builder for creating playlists with a fluent API.
///
/// # Example
///
/// ```
/// use oximedia_playlist::playlist::{PlaylistBuilder, PlaylistType};
/// use std::time::Duration;
///
/// let playlist = PlaylistBuilder::new("morning_block", PlaylistType::Linear)
///     .with_looping(false)
///     .with_description("Morning programming block")
///     .add_item_path("news.mxf", Duration::from_secs(1800))
///     .add_item_path("weather.mxf", Duration::from_secs(300))
///     .build();
///
/// assert_eq!(playlist.len(), 2);
/// ```
pub struct PlaylistBuilder {
    playlist: Playlist,
}

impl PlaylistBuilder {
    /// Creates a new playlist builder.
    #[must_use]
    pub fn new<S: Into<String>>(name: S, playlist_type: PlaylistType) -> Self {
        Self {
            playlist: Playlist::new(name, playlist_type),
        }
    }

    /// Sets whether the playlist loops.
    #[must_use]
    pub fn with_looping(mut self, looping: bool) -> Self {
        self.playlist.looping = looping;
        self
    }

    /// Sets the playlist description.
    #[must_use]
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.playlist.metadata.description = Some(description.into());
        self
    }

    /// Sets the playlist creator.
    #[must_use]
    pub fn with_creator<S: Into<String>>(mut self, creator: S) -> Self {
        self.playlist.metadata.creator = Some(creator.into());
        self
    }

    /// Adds a tag to the playlist.
    #[must_use]
    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.playlist.metadata.tags.push(tag.into());
        self
    }

    /// Adds an existing playlist item.
    #[must_use]
    pub fn add_item(mut self, item: PlaylistItem) -> Self {
        self.playlist.add_item(item);
        self
    }

    /// Adds a new item from a path with a specific duration.
    #[must_use]
    pub fn add_item_path<P: Into<std::path::PathBuf>>(
        mut self,
        path: P,
        duration: std::time::Duration,
    ) -> Self {
        let item = PlaylistItem::new(path).with_duration(duration);
        self.playlist.add_item(item);
        self
    }

    /// Builds the final playlist.
    #[must_use]
    pub fn build(self) -> Playlist {
        self.playlist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_builder() {
        let playlist = PlaylistBuilder::new("test", PlaylistType::Linear)
            .with_looping(true)
            .with_description("Test playlist")
            .with_creator("Test User")
            .with_tag("test")
            .add_item_path("item1.mxf", Duration::from_secs(60))
            .add_item_path("item2.mxf", Duration::from_secs(30))
            .build();

        assert_eq!(playlist.name, "test");
        assert_eq!(playlist.len(), 2);
        assert!(playlist.looping);
        assert_eq!(
            playlist.metadata.description,
            Some("Test playlist".to_string())
        );
    }
}
