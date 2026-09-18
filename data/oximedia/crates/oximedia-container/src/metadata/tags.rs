//! Common metadata tag types and utilities.
//!
//! Provides a unified representation of metadata tags across different
//! container formats.

use std::collections::HashMap;
use std::fmt;

/// Standard tag field names used across formats.
///
/// These are the most common metadata fields found in audio and video files.
/// The names follow Vorbis comment conventions (uppercase) for consistency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardTag {
    /// Track or file title.
    Title,
    /// Artist or author name.
    Artist,
    /// Album name (audio) or collection name.
    Album,
    /// Album artist (may differ from track artist).
    AlbumArtist,
    /// Release date (ISO 8601 format recommended).
    Date,
    /// Genre classification.
    Genre,
    /// Track number within album.
    TrackNumber,
    /// Total number of tracks in album.
    TotalTracks,
    /// Disc number for multi-disc releases.
    DiscNumber,
    /// Total number of discs.
    TotalDiscs,
    /// Composer name.
    Composer,
    /// Performer name.
    Performer,
    /// Copyright information.
    Copyright,
    /// License information.
    License,
    /// Recording organization or label.
    Organization,
    /// Description or comment.
    Description,
    /// Comment field.
    Comment,
    /// Lyrics or subtitle text.
    Lyrics,
    /// International Standard Recording Code.
    Isrc,
    /// Encoding software or application.
    Encoder,
    /// Encoded by (person or organization).
    EncodedBy,
    /// Language code (ISO 639-2 or BCP 47).
    Language,
}

impl StandardTag {
    /// Returns the canonical Vorbis comment field name.
    ///
    /// These are uppercase names used in Vorbis comments (Ogg, FLAC).
    #[must_use]
    pub const fn vorbis_name(self) -> &'static str {
        match self {
            Self::Title => "TITLE",
            Self::Artist => "ARTIST",
            Self::Album => "ALBUM",
            Self::AlbumArtist => "ALBUMARTIST",
            Self::Date => "DATE",
            Self::Genre => "GENRE",
            Self::TrackNumber => "TRACKNUMBER",
            Self::TotalTracks => "TOTALTRACKS",
            Self::DiscNumber => "DISCNUMBER",
            Self::TotalDiscs => "TOTALDISCS",
            Self::Composer => "COMPOSER",
            Self::Performer => "PERFORMER",
            Self::Copyright => "COPYRIGHT",
            Self::License => "LICENSE",
            Self::Organization => "ORGANIZATION",
            Self::Description => "DESCRIPTION",
            Self::Comment => "COMMENT",
            Self::Lyrics => "LYRICS",
            Self::Isrc => "ISRC",
            Self::Encoder => "ENCODER",
            Self::EncodedBy => "ENCODED-BY",
            Self::Language => "LANGUAGE",
        }
    }

    /// Returns the Matroska tag name equivalent.
    #[must_use]
    pub const fn matroska_name(self) -> &'static str {
        match self {
            Self::Title => "TITLE",
            Self::Artist => "ARTIST",
            Self::Album => "ALBUM",
            Self::AlbumArtist => "ALBUM_ARTIST",
            Self::Date => "DATE_RELEASED",
            Self::Genre => "GENRE",
            Self::TrackNumber => "PART_NUMBER",
            Self::TotalTracks => "TOTAL_PARTS",
            Self::DiscNumber => "DISC",
            Self::TotalDiscs => "TOTAL_DISCS",
            Self::Composer => "COMPOSER",
            Self::Performer => "PERFORMER",
            Self::Copyright => "COPYRIGHT",
            Self::License => "LICENSE",
            Self::Organization => "PUBLISHER",
            Self::Description => "DESCRIPTION",
            Self::Comment => "COMMENT",
            Self::Lyrics => "LYRICS",
            Self::Isrc => "ISRC",
            Self::Encoder => "ENCODER",
            Self::EncodedBy => "ENCODED_BY",
            Self::Language => "LANGUAGE",
        }
    }

    /// Attempts to parse a standard tag from a string field name.
    ///
    /// Case-insensitive matching for both Vorbis and Matroska conventions.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let upper = s.to_uppercase();
        let normalized = upper.replace(['-', ' '], "_");

        match normalized.as_str() {
            "TITLE" => Some(Self::Title),
            "ARTIST" => Some(Self::Artist),
            "ALBUM" => Some(Self::Album),
            "ALBUMARTIST" | "ALBUM_ARTIST" => Some(Self::AlbumArtist),
            "DATE" | "DATE_RELEASED" | "YEAR" => Some(Self::Date),
            "GENRE" => Some(Self::Genre),
            "TRACKNUMBER" | "TRACK" | "PART_NUMBER" => Some(Self::TrackNumber),
            "TOTALTRACKS" | "TOTAL_PARTS" | "TRACKTOTAL" => Some(Self::TotalTracks),
            "DISCNUMBER" | "DISC" => Some(Self::DiscNumber),
            "TOTALDISCS" | "DISCTOTAL" | "TOTAL_DISCS" => Some(Self::TotalDiscs),
            "COMPOSER" => Some(Self::Composer),
            "PERFORMER" => Some(Self::Performer),
            "COPYRIGHT" => Some(Self::Copyright),
            "LICENSE" => Some(Self::License),
            "ORGANIZATION" | "PUBLISHER" | "LABEL" => Some(Self::Organization),
            "DESCRIPTION" => Some(Self::Description),
            "COMMENT" => Some(Self::Comment),
            "LYRICS" | "UNSYNCEDLYRICS" => Some(Self::Lyrics),
            "ISRC" => Some(Self::Isrc),
            "ENCODER" | "ENCODING" => Some(Self::Encoder),
            "ENCODED_BY" | "ENCODEDBY" => Some(Self::EncodedBy),
            "LANGUAGE" => Some(Self::Language),
            _ => None,
        }
    }
}

impl fmt::Display for StandardTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.vorbis_name())
    }
}

/// A metadata tag value.
///
/// Tags can contain either text or binary data.
#[derive(Clone, Debug, PartialEq)]
pub enum TagValue {
    /// UTF-8 text value.
    Text(String),
    /// Binary data (e.g., embedded images).
    Binary(Vec<u8>),
}

impl TagValue {
    /// Returns the text value if this is a text tag.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Binary(_) => None,
        }
    }

    /// Returns the binary value if this is a binary tag.
    #[must_use]
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Self::Text(_) => None,
            Self::Binary(b) => Some(b),
        }
    }

    /// Returns true if this is a text value.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns true if this is a binary value.
    #[must_use]
    pub const fn is_binary(&self) -> bool {
        matches!(self, Self::Binary(_))
    }
}

impl From<String> for TagValue {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for TagValue {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<Vec<u8>> for TagValue {
    fn from(b: Vec<u8>) -> Self {
        Self::Binary(b)
    }
}

impl fmt::Display for TagValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "{s}"),
            Self::Binary(b) => write!(f, "<binary {} bytes>", b.len()),
        }
    }
}

/// A map of metadata tags.
///
/// Stores tag names (case-insensitive) and their values.
/// Multiple values per tag are supported (as in Vorbis comments).
#[derive(Clone, Debug, Default)]
pub struct TagMap {
    /// Internal storage using uppercase keys for case-insensitive access.
    tags: HashMap<String, Vec<TagValue>>,
}

impl TagMap {
    /// Creates a new empty tag map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
        }
    }

    /// Sets a tag value, replacing any existing values.
    pub fn set(&mut self, key: impl AsRef<str>, value: impl Into<TagValue>) {
        let key = key.as_ref().to_uppercase();
        self.tags.insert(key, vec![value.into()]);
    }

    /// Adds a tag value without removing existing values.
    ///
    /// Useful for tags that can have multiple values (e.g., multiple artists).
    pub fn add(&mut self, key: impl AsRef<str>, value: impl Into<TagValue>) {
        let key = key.as_ref().to_uppercase();
        self.tags.entry(key).or_default().push(value.into());
    }

    /// Gets the first value for a tag.
    ///
    /// Returns `None` if the tag doesn't exist or has no values.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&TagValue> {
        let key = key.to_uppercase();
        self.tags.get(&key).and_then(|values| values.first())
    }

    /// Gets the first text value for a tag.
    ///
    /// Returns `None` if the tag doesn't exist, has no values, or is binary.
    #[must_use]
    pub fn get_text(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(TagValue::as_text)
    }

    /// Gets all values for a tag.
    ///
    /// Returns an empty slice if the tag doesn't exist.
    #[must_use]
    pub fn get_all(&self, key: &str) -> &[TagValue] {
        let key = key.to_uppercase();
        self.tags.get(&key).map_or(&[], |v| v.as_slice())
    }

    /// Removes a tag and all its values.
    ///
    /// Returns true if the tag existed.
    pub fn remove(&mut self, key: &str) -> bool {
        let key = key.to_uppercase();
        self.tags.remove(&key).is_some()
    }

    /// Returns true if the tag map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Returns the number of distinct tags (not counting multiple values).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Returns an iterator over all tag names.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tags.keys().map(std::string::String::as_str)
    }

    /// Returns an iterator over all (key, value) pairs.
    ///
    /// Tags with multiple values will appear multiple times.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TagValue)> {
        self.tags
            .iter()
            .flat_map(|(k, values)| values.iter().map(move |v| (k.as_str(), v)))
    }

    /// Clears all tags.
    pub fn clear(&mut self) {
        self.tags.clear();
    }

    /// Merges another tag map into this one.
    ///
    /// Existing tags are replaced by values from `other`.
    pub fn merge(&mut self, other: &TagMap) {
        for (key, values) in &other.tags {
            self.tags.insert(key.clone(), values.clone());
        }
    }

    /// Gets a standard tag value if present.
    #[must_use]
    pub fn get_standard(&self, tag: StandardTag) -> Option<&TagValue> {
        self.get(tag.vorbis_name())
    }

    /// Sets a standard tag value.
    pub fn set_standard(&mut self, tag: StandardTag, value: impl Into<TagValue>) {
        self.set(tag.vorbis_name(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_tag_names() {
        assert_eq!(StandardTag::Title.vorbis_name(), "TITLE");
        assert_eq!(StandardTag::Artist.vorbis_name(), "ARTIST");
        assert_eq!(StandardTag::Album.vorbis_name(), "ALBUM");
    }

    #[test]
    fn test_standard_tag_from_str() {
        assert_eq!(StandardTag::parse("TITLE"), Some(StandardTag::Title));
        assert_eq!(StandardTag::parse("title"), Some(StandardTag::Title));
        assert_eq!(StandardTag::parse("Artist"), Some(StandardTag::Artist));
        assert_eq!(
            StandardTag::parse("ALBUMARTIST"),
            Some(StandardTag::AlbumArtist)
        );
        assert_eq!(
            StandardTag::parse("ALBUM_ARTIST"),
            Some(StandardTag::AlbumArtist)
        );
        assert_eq!(StandardTag::parse("UNKNOWN"), None);
    }

    #[test]
    fn test_tag_value() {
        let text = TagValue::Text("test".to_string());
        assert!(text.is_text());
        assert!(!text.is_binary());
        assert_eq!(text.as_text(), Some("test"));
        assert_eq!(text.as_binary(), None);

        let binary = TagValue::Binary(vec![1, 2, 3]);
        assert!(!binary.is_text());
        assert!(binary.is_binary());
        assert_eq!(binary.as_text(), None);
        assert_eq!(binary.as_binary(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_tag_value_from() {
        let v1: TagValue = "test".into();
        assert_eq!(v1.as_text(), Some("test"));

        let v2: TagValue = "test".to_string().into();
        assert_eq!(v2.as_text(), Some("test"));

        let v3: TagValue = vec![1, 2, 3].into();
        assert_eq!(v3.as_binary(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_tag_map_basic() {
        let mut map = TagMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.set("TITLE", "Test Title");
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
        assert_eq!(map.get_text("TITLE"), Some("Test Title"));
    }

    #[test]
    fn test_tag_map_case_insensitive() {
        let mut map = TagMap::new();
        map.set("Title", "Test");
        assert_eq!(map.get_text("TITLE"), Some("Test"));
        assert_eq!(map.get_text("title"), Some("Test"));
        assert_eq!(map.get_text("TiTlE"), Some("Test"));
    }

    #[test]
    fn test_tag_map_multiple_values() {
        let mut map = TagMap::new();
        map.add("ARTIST", "Artist 1");
        map.add("ARTIST", "Artist 2");

        let values = map.get_all("ARTIST");
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].as_text(), Some("Artist 1"));
        assert_eq!(values[1].as_text(), Some("Artist 2"));

        // get() returns first value
        assert_eq!(map.get_text("ARTIST"), Some("Artist 1"));
    }

    #[test]
    fn test_tag_map_set_replaces() {
        let mut map = TagMap::new();
        map.add("TITLE", "Old");
        map.set("TITLE", "New");

        let values = map.get_all("TITLE");
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].as_text(), Some("New"));
    }

    #[test]
    fn test_tag_map_remove() {
        let mut map = TagMap::new();
        map.set("TITLE", "Test");
        assert!(map.remove("TITLE"));
        assert!(!map.remove("TITLE"));
        assert!(map.is_empty());
    }

    #[test]
    fn test_tag_map_clear() {
        let mut map = TagMap::new();
        map.set("TITLE", "Test");
        map.set("ARTIST", "Test");
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn test_tag_map_merge() {
        let mut map1 = TagMap::new();
        map1.set("TITLE", "Title1");
        map1.set("ARTIST", "Artist1");

        let mut map2 = TagMap::new();
        map2.set("ARTIST", "Artist2");
        map2.set("ALBUM", "Album2");

        map1.merge(&map2);
        assert_eq!(map1.get_text("TITLE"), Some("Title1"));
        assert_eq!(map1.get_text("ARTIST"), Some("Artist2")); // Replaced
        assert_eq!(map1.get_text("ALBUM"), Some("Album2")); // Added
    }

    #[test]
    fn test_tag_map_standard() {
        let mut map = TagMap::new();
        map.set_standard(StandardTag::Title, "Test");
        assert_eq!(
            map.get_standard(StandardTag::Title)
                .and_then(TagValue::as_text),
            Some("Test")
        );
    }

    #[test]
    fn test_tag_map_iter() {
        let mut map = TagMap::new();
        map.set("TITLE", "Title");
        map.set("ARTIST", "Artist");

        let entries: Vec<_> = map.iter().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_tag_map_keys() {
        let mut map = TagMap::new();
        map.set("TITLE", "Title");
        map.set("ARTIST", "Artist");

        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"TITLE"));
        assert!(keys.contains(&"ARTIST"));
    }
}
