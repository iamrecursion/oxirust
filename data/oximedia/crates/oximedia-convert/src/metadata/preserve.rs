// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Metadata preservation across format conversions.

use crate::Result;
use std::collections::HashMap;
use std::path::Path;

/// Preserver for maintaining metadata across conversions.
#[derive(Debug, Clone)]
pub struct MetadataPreserver {
    preserve_all: bool,
    allowed_keys: Option<Vec<String>>,
}

impl MetadataPreserver {
    /// Create a new metadata preserver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            preserve_all: true,
            allowed_keys: None,
        }
    }

    /// Set whether to preserve all metadata.
    #[must_use]
    pub fn with_preserve_all(mut self, preserve: bool) -> Self {
        self.preserve_all = preserve;
        self
    }

    /// Set specific metadata keys to preserve.
    #[must_use]
    pub fn with_allowed_keys(mut self, keys: Vec<String>) -> Self {
        self.allowed_keys = Some(keys);
        self.preserve_all = false;
        self
    }

    /// Extract metadata from a file.
    pub fn extract<P: AsRef<Path>>(&self, path: P) -> Result<MediaMetadata> {
        let _path = path.as_ref();

        // Placeholder for actual metadata extraction
        // In a real implementation, this would use oximedia-metadata
        Ok(MediaMetadata {
            title: None,
            artist: None,
            album: None,
            date: None,
            comment: None,
            genre: None,
            track: None,
            custom: HashMap::new(),
        })
    }

    /// Restore metadata to a file.
    pub fn restore<P: AsRef<Path>>(&self, path: P, metadata: &MediaMetadata) -> Result<()> {
        let _path = path.as_ref();
        let _filtered = self.filter_metadata(metadata);

        // Placeholder for actual metadata restoration
        // In a real implementation, this would use oximedia-metadata
        Ok(())
    }

    /// Copy metadata from one file to another.
    pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(&self, source: P, dest: Q) -> Result<()> {
        let metadata = self.extract(source)?;
        self.restore(dest, &metadata)
    }

    fn filter_metadata(&self, metadata: &MediaMetadata) -> MediaMetadata {
        if self.preserve_all {
            return metadata.clone();
        }

        let allowed_keys = match &self.allowed_keys {
            Some(keys) => keys,
            None => return MediaMetadata::default(),
        };

        let mut filtered = MediaMetadata {
            title: if allowed_keys.contains(&"title".to_string()) {
                metadata.title.clone()
            } else {
                None
            },
            artist: if allowed_keys.contains(&"artist".to_string()) {
                metadata.artist.clone()
            } else {
                None
            },
            album: if allowed_keys.contains(&"album".to_string()) {
                metadata.album.clone()
            } else {
                None
            },
            date: if allowed_keys.contains(&"date".to_string()) {
                metadata.date.clone()
            } else {
                None
            },
            comment: if allowed_keys.contains(&"comment".to_string()) {
                metadata.comment.clone()
            } else {
                None
            },
            genre: if allowed_keys.contains(&"genre".to_string()) {
                metadata.genre.clone()
            } else {
                None
            },
            track: if allowed_keys.contains(&"track".to_string()) {
                metadata.track
            } else {
                None
            },
            custom: HashMap::new(),
        };

        for (key, value) in &metadata.custom {
            if allowed_keys.contains(key) {
                filtered.custom.insert(key.clone(), value.clone());
            }
        }

        filtered
    }
}

impl Default for MetadataPreserver {
    fn default() -> Self {
        Self::new()
    }
}

/// Media file metadata.
#[derive(Debug, Clone, Default)]
pub struct MediaMetadata {
    /// Title
    pub title: Option<String>,
    /// Artist/Author
    pub artist: Option<String>,
    /// Album
    pub album: Option<String>,
    /// Date/Year
    pub date: Option<String>,
    /// Comment
    pub comment: Option<String>,
    /// Genre
    pub genre: Option<String>,
    /// Track number
    pub track: Option<u32>,
    /// Custom metadata fields
    pub custom: HashMap<String, String>,
}

impl MediaMetadata {
    /// Create empty metadata.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if metadata is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.artist.is_none()
            && self.album.is_none()
            && self.date.is_none()
            && self.comment.is_none()
            && self.genre.is_none()
            && self.track.is_none()
            && self.custom.is_empty()
    }

    /// Get the number of metadata fields set.
    #[must_use]
    pub fn field_count(&self) -> usize {
        let mut count = 0;
        if self.title.is_some() {
            count += 1;
        }
        if self.artist.is_some() {
            count += 1;
        }
        if self.album.is_some() {
            count += 1;
        }
        if self.date.is_some() {
            count += 1;
        }
        if self.comment.is_some() {
            count += 1;
        }
        if self.genre.is_some() {
            count += 1;
        }
        if self.track.is_some() {
            count += 1;
        }
        count + self.custom.len()
    }

    /// Merge with another metadata, preferring non-None values.
    pub fn merge(&mut self, other: &MediaMetadata) {
        if self.title.is_none() && other.title.is_some() {
            self.title = other.title.clone();
        }
        if self.artist.is_none() && other.artist.is_some() {
            self.artist = other.artist.clone();
        }
        if self.album.is_none() && other.album.is_some() {
            self.album = other.album.clone();
        }
        if self.date.is_none() && other.date.is_some() {
            self.date = other.date.clone();
        }
        if self.comment.is_none() && other.comment.is_some() {
            self.comment = other.comment.clone();
        }
        if self.genre.is_none() && other.genre.is_some() {
            self.genre = other.genre.clone();
        }
        if self.track.is_none() && other.track.is_some() {
            self.track = other.track;
        }

        for (key, value) in &other.custom {
            self.custom
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preserver_creation() {
        let preserver = MetadataPreserver::new();
        assert!(preserver.preserve_all);
    }

    #[test]
    fn test_metadata_empty() {
        let metadata = MediaMetadata::empty();
        assert!(metadata.is_empty());
        assert_eq!(metadata.field_count(), 0);
    }

    #[test]
    fn test_metadata_field_count() {
        let mut metadata = MediaMetadata::empty();
        metadata.title = Some("Test".to_string());
        metadata.artist = Some("Artist".to_string());

        assert_eq!(metadata.field_count(), 2);
        assert!(!metadata.is_empty());
    }

    #[test]
    fn test_metadata_merge() {
        let mut meta1 = MediaMetadata {
            title: Some("Title1".to_string()),
            artist: None,
            ..Default::default()
        };

        let meta2 = MediaMetadata {
            title: Some("Title2".to_string()),
            artist: Some("Artist2".to_string()),
            ..Default::default()
        };

        meta1.merge(&meta2);

        // Title should keep original value
        assert_eq!(meta1.title, Some("Title1".to_string()));
        // Artist should be filled from meta2
        assert_eq!(meta1.artist, Some("Artist2".to_string()));
    }

    #[test]
    fn test_filter_metadata() {
        let preserver = MetadataPreserver::new()
            .with_allowed_keys(vec!["title".to_string(), "artist".to_string()]);

        let metadata = MediaMetadata {
            title: Some("Title".to_string()),
            artist: Some("Artist".to_string()),
            album: Some("Album".to_string()),
            ..Default::default()
        };

        let filtered = preserver.filter_metadata(&metadata);

        assert_eq!(filtered.title, Some("Title".to_string()));
        assert_eq!(filtered.artist, Some("Artist".to_string()));
        assert_eq!(filtered.album, None);
    }

    #[test]
    fn test_custom_metadata() {
        let mut metadata = MediaMetadata::empty();
        metadata
            .custom
            .insert("key1".to_string(), "value1".to_string());
        metadata
            .custom
            .insert("key2".to_string(), "value2".to_string());

        assert_eq!(metadata.field_count(), 2);
        assert!(!metadata.is_empty());
    }
}
