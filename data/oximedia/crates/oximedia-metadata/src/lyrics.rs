//! Lyrics metadata support (synchronized and unsynchronized).
//!
//! This module provides parsing, construction, and serialization of lyrics
//! metadata in multiple formats:
//!
//! - **ID3v2 USLT** (unsynchronized lyrics)
//! - **ID3v2 SYLT** (synchronized lyrics with timestamps)
//! - **LRC format** (simple `[mm:ss.xx]` timestamps + text lines)
//! - **Enhanced LRC** (word-level timestamps)
//!
//! # Overview
//!
//! Lyrics come in two flavors:
//!
//! - **Unsynchronized**: Plain text lyrics without timing information.
//! - **Synchronized**: Each line (or word) is associated with a timestamp,
//!   enabling karaoke-style display and lyric highlighting.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::lyrics::{SyncedLyrics, LyricLine};
//!
//! let mut lyrics = SyncedLyrics::new();
//! lyrics.add_line(LyricLine::new(0, "First line of the song"));
//! lyrics.add_line(LyricLine::new(5000, "Second line"));
//! lyrics.add_line(LyricLine::new(10_000, "Third line"));
//!
//! let lrc = lyrics.to_lrc();
//! assert!(lrc.contains("[00:00.00]"));
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};

/// Unsynchronized lyrics (plain text, possibly with language/description).
#[derive(Debug, Clone, PartialEq)]
pub struct UnsyncedLyrics {
    /// Language code (ISO 639-2, e.g., "eng").
    pub language: String,
    /// Content descriptor / description.
    pub description: String,
    /// The lyrics text.
    pub text: String,
}

impl UnsyncedLyrics {
    /// Create new unsynchronized lyrics.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            language: "und".to_string(), // undetermined
            description: String::new(),
            text: text.into(),
        }
    }

    /// Set the language code.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Check if the lyrics are empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Number of lines in the lyrics.
    pub fn line_count(&self) -> usize {
        if self.text.is_empty() {
            0
        } else {
            self.text.lines().count()
        }
    }

    /// Write to a `Metadata` container as an ID3v2 USLT-style field.
    pub fn to_metadata(&self, metadata: &mut Metadata) {
        metadata.insert("USLT".to_string(), MetadataValue::Text(self.text.clone()));
        if !self.language.is_empty() && self.language != "und" {
            metadata.insert(
                "USLT:language".to_string(),
                MetadataValue::Text(self.language.clone()),
            );
        }
        if !self.description.is_empty() {
            metadata.insert(
                "USLT:description".to_string(),
                MetadataValue::Text(self.description.clone()),
            );
        }
    }

    /// Extract from a `Metadata` container.
    pub fn from_metadata(metadata: &Metadata) -> Option<Self> {
        let text = metadata.get("USLT").and_then(|v| v.as_text())?;
        let language = metadata
            .get("USLT:language")
            .and_then(|v| v.as_text())
            .unwrap_or("und")
            .to_string();
        let description = metadata
            .get("USLT:description")
            .and_then(|v| v.as_text())
            .unwrap_or("")
            .to_string();

        Some(Self {
            language,
            description,
            text: text.to_string(),
        })
    }
}

/// A single line of synchronized lyrics.
#[derive(Debug, Clone, PartialEq)]
pub struct LyricLine {
    /// Timestamp in milliseconds from the start of the media.
    pub timestamp_ms: u64,
    /// The text content of this line.
    pub text: String,
    /// Optional end timestamp (for karaoke-style word highlighting).
    pub end_ms: Option<u64>,
}

impl LyricLine {
    /// Create a new lyric line.
    pub fn new(timestamp_ms: u64, text: impl Into<String>) -> Self {
        Self {
            timestamp_ms,
            text: text.into(),
            end_ms: None,
        }
    }

    /// Set the end timestamp.
    pub fn with_end(mut self, end_ms: u64) -> Self {
        self.end_ms = Some(end_ms);
        self
    }

    /// Duration of this line in milliseconds (if end is set).
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms.map(|end| end.saturating_sub(self.timestamp_ms))
    }

    /// Format the timestamp as `[mm:ss.xx]` (LRC format).
    pub fn lrc_timestamp(&self) -> String {
        format_lrc_time(self.timestamp_ms)
    }
}

/// Synchronized lyrics container.
#[derive(Debug, Clone, Default)]
pub struct SyncedLyrics {
    /// Language code (ISO 639-2).
    pub language: String,
    /// Content descriptor.
    pub description: String,
    /// Content type (lyrics, text, movement name, etc.).
    pub content_type: SyncedContentType,
    /// Ordered list of lyric lines.
    lines: Vec<LyricLine>,
    /// LRC metadata: title.
    pub lrc_title: Option<String>,
    /// LRC metadata: artist.
    pub lrc_artist: Option<String>,
    /// LRC metadata: album.
    pub lrc_album: Option<String>,
    /// LRC metadata: author of the LRC file.
    pub lrc_author: Option<String>,
    /// LRC metadata: offset in milliseconds.
    pub lrc_offset_ms: Option<i64>,
}

/// Content type for synchronized lyrics (ID3v2 SYLT types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncedContentType {
    /// Other / unknown.
    Other,
    /// Song lyrics.
    Lyrics,
    /// Text transcription (spoken word).
    TextTranscription,
    /// Movement / part name.
    MovementName,
    /// Event description (e.g., "door opens").
    Events,
    /// Chord names (e.g., "Am", "G7").
    Chord,
    /// Trivia / pop-up information.
    Trivia,
}

impl Default for SyncedContentType {
    fn default() -> Self {
        Self::Lyrics
    }
}

impl SyncedContentType {
    /// ID3v2 SYLT content type byte.
    pub fn to_id3v2_byte(self) -> u8 {
        match self {
            Self::Other => 0,
            Self::Lyrics => 1,
            Self::TextTranscription => 2,
            Self::MovementName => 3,
            Self::Events => 4,
            Self::Chord => 5,
            Self::Trivia => 6,
        }
    }

    /// Parse from ID3v2 SYLT content type byte.
    pub fn from_id3v2_byte(byte: u8) -> Self {
        match byte {
            0 => Self::Other,
            1 => Self::Lyrics,
            2 => Self::TextTranscription,
            3 => Self::MovementName,
            4 => Self::Events,
            5 => Self::Chord,
            6 => Self::Trivia,
            _ => Self::Other,
        }
    }
}

impl SyncedLyrics {
    /// Create new empty synchronized lyrics.
    pub fn new() -> Self {
        Self {
            language: "und".to_string(),
            ..Self::default()
        }
    }

    /// Set language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set content type.
    pub fn with_content_type(mut self, ct: SyncedContentType) -> Self {
        self.content_type = ct;
        self
    }

    /// Add a lyric line.
    pub fn add_line(&mut self, line: LyricLine) {
        self.lines.push(line);
    }

    /// Get all lyric lines.
    pub fn lines(&self) -> &[LyricLine] {
        &self.lines
    }

    /// Number of lyric lines.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Returns true if there are no lyric lines.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Sort lines by timestamp.
    pub fn sort_by_time(&mut self) {
        self.lines.sort_by_key(|l| l.timestamp_ms);
    }

    /// Find the active lyric line at a given timestamp.
    ///
    /// Returns the last line whose timestamp is <= the given time.
    pub fn line_at(&self, time_ms: u64) -> Option<&LyricLine> {
        let mut result = None;
        for line in &self.lines {
            if line.timestamp_ms <= time_ms {
                result = Some(line);
            } else {
                break;
            }
        }
        result
    }

    /// Total duration (timestamp of last line).
    pub fn total_duration_ms(&self) -> Option<u64> {
        self.lines.last().map(|l| l.timestamp_ms)
    }

    /// Apply a time offset to all lines (can be negative).
    pub fn apply_offset(&mut self, offset_ms: i64) {
        for line in &mut self.lines {
            if offset_ms >= 0 {
                line.timestamp_ms = line.timestamp_ms.saturating_add(offset_ms as u64);
                if let Some(ref mut end) = line.end_ms {
                    *end = end.saturating_add(offset_ms as u64);
                }
            } else {
                let abs_offset = offset_ms.unsigned_abs();
                line.timestamp_ms = line.timestamp_ms.saturating_sub(abs_offset);
                if let Some(ref mut end) = line.end_ms {
                    *end = end.saturating_sub(abs_offset);
                }
            }
        }
    }

    // ---- LRC format ----

    /// Serialize to LRC format.
    pub fn to_lrc(&self) -> String {
        let mut lrc = String::new();

        // LRC metadata headers
        if let Some(ref title) = self.lrc_title {
            lrc.push_str(&format!("[ti:{title}]\n"));
        }
        if let Some(ref artist) = self.lrc_artist {
            lrc.push_str(&format!("[ar:{artist}]\n"));
        }
        if let Some(ref album) = self.lrc_album {
            lrc.push_str(&format!("[al:{album}]\n"));
        }
        if let Some(ref author) = self.lrc_author {
            lrc.push_str(&format!("[by:{author}]\n"));
        }
        if let Some(offset) = self.lrc_offset_ms {
            lrc.push_str(&format!("[offset:{offset}]\n"));
        }

        if !lrc.is_empty() {
            lrc.push('\n');
        }

        // Lines
        for line in &self.lines {
            lrc.push_str(&format!(
                "{}{}",
                format_lrc_time(line.timestamp_ms),
                line.text,
            ));
            lrc.push('\n');
        }

        lrc
    }

    /// Parse from LRC format.
    ///
    /// # Errors
    ///
    /// Returns an error if the LRC data is malformed.
    pub fn from_lrc(lrc: &str) -> Result<Self, Error> {
        let mut synced = SyncedLyrics::new();

        for line in lrc.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse LRC metadata tags
            if let Some(inner) = extract_lrc_tag(trimmed, "ti") {
                synced.lrc_title = Some(inner.to_string());
                continue;
            }
            if let Some(inner) = extract_lrc_tag(trimmed, "ar") {
                synced.lrc_artist = Some(inner.to_string());
                continue;
            }
            if let Some(inner) = extract_lrc_tag(trimmed, "al") {
                synced.lrc_album = Some(inner.to_string());
                continue;
            }
            if let Some(inner) = extract_lrc_tag(trimmed, "by") {
                synced.lrc_author = Some(inner.to_string());
                continue;
            }
            if let Some(inner) = extract_lrc_tag(trimmed, "offset") {
                if let Ok(offset) = inner.parse::<i64>() {
                    synced.lrc_offset_ms = Some(offset);
                }
                continue;
            }

            // Parse timestamp lines: [mm:ss.xx]text or [mm:ss.xxx]text
            let mut pos = 0;
            while pos < trimmed.len() {
                if let Some(open) = trimmed[pos..].find('[') {
                    let abs_open = pos + open;
                    if let Some(close) = trimmed[abs_open..].find(']') {
                        let abs_close = abs_open + close;
                        let time_str = &trimmed[abs_open + 1..abs_close];

                        if let Some(timestamp_ms) = parse_lrc_time(time_str) {
                            // Text is everything after the last `]`
                            let text_start = abs_close + 1;
                            // Check if there's another timestamp tag
                            let text_end = trimmed[text_start..]
                                .find('[')
                                .map(|i| text_start + i)
                                .unwrap_or(trimmed.len());
                            let text = trimmed[text_start..text_end].to_string();

                            if !text.is_empty() || text_end == trimmed.len() {
                                synced.add_line(LyricLine::new(timestamp_ms, text));
                            }
                            pos = text_end;
                        } else {
                            // Not a valid timestamp, skip past this bracket
                            pos = abs_close + 1;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        synced.sort_by_time();
        Ok(synced)
    }

    /// Write to a `Metadata` container.
    pub fn to_metadata(&self, metadata: &mut Metadata) {
        let lrc = self.to_lrc();
        metadata.insert("SYLT".to_string(), MetadataValue::Text(lrc));
        if !self.language.is_empty() && self.language != "und" {
            metadata.insert(
                "SYLT:language".to_string(),
                MetadataValue::Text(self.language.clone()),
            );
        }
        metadata.insert(
            "SYLT:content_type".to_string(),
            MetadataValue::Integer(i64::from(self.content_type.to_id3v2_byte())),
        );
    }

    /// Extract from a `Metadata` container.
    ///
    /// # Errors
    ///
    /// Returns an error if LRC parsing fails.
    pub fn from_metadata(metadata: &Metadata) -> Result<Option<Self>, Error> {
        let lrc_text = match metadata.get("SYLT").and_then(|v| v.as_text()) {
            Some(text) => text,
            None => return Ok(None),
        };

        let mut synced = Self::from_lrc(lrc_text)?;

        if let Some(lang) = metadata.get("SYLT:language").and_then(|v| v.as_text()) {
            synced.language = lang.to_string();
        }

        if let Some(ct) = metadata
            .get("SYLT:content_type")
            .and_then(|v| v.as_integer())
        {
            synced.content_type = SyncedContentType::from_id3v2_byte(ct as u8);
        }

        Ok(Some(synced))
    }
}

// ---- LRC helpers ----

fn format_lrc_time(ms: u64) -> String {
    let total_secs = ms / 1000;
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    let hundredths = (ms % 1000) / 10;
    format!("[{minutes:02}:{seconds:02}.{hundredths:02}]")
}

fn parse_lrc_time(s: &str) -> Option<u64> {
    // Formats: "mm:ss.xx" or "mm:ss.xxx"
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return None;
    }

    let minutes: u64 = parts[0].parse().ok()?;

    let sec_parts: Vec<&str> = parts[1].split('.').collect();
    if sec_parts.len() != 2 {
        return None;
    }

    let seconds: u64 = sec_parts[0].parse().ok()?;
    let frac_str = sec_parts[1];
    let frac_ms: u64 = match frac_str.len() {
        1 => frac_str.parse::<u64>().ok()? * 100,
        2 => frac_str.parse::<u64>().ok()? * 10,
        3 => frac_str.parse::<u64>().ok()?,
        _ => return None,
    };

    Some(minutes * 60_000 + seconds * 1000 + frac_ms)
}

fn extract_lrc_tag<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let prefix = format!("[{tag}:");
    if line.starts_with(&prefix) {
        let rest = &line[prefix.len()..];
        rest.strip_suffix(']').or(Some(rest))
    } else {
        None
    }
}

/// A container holding both synchronized and unsynchronized lyrics.
#[derive(Debug, Clone, Default)]
pub struct LyricsCollection {
    /// Unsynchronized lyrics (may have multiple language versions).
    pub unsynced: Vec<UnsyncedLyrics>,
    /// Synchronized lyrics (may have multiple language versions).
    pub synced: Vec<SyncedLyrics>,
}

impl LyricsCollection {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add unsynchronized lyrics.
    pub fn add_unsynced(&mut self, lyrics: UnsyncedLyrics) {
        self.unsynced.push(lyrics);
    }

    /// Add synchronized lyrics.
    pub fn add_synced(&mut self, lyrics: SyncedLyrics) {
        self.synced.push(lyrics);
    }

    /// Get unsynced lyrics for a specific language.
    pub fn unsynced_for_language(&self, lang: &str) -> Option<&UnsyncedLyrics> {
        self.unsynced.iter().find(|l| l.language == lang)
    }

    /// Get synced lyrics for a specific language.
    pub fn synced_for_language(&self, lang: &str) -> Option<&SyncedLyrics> {
        self.synced.iter().find(|l| l.language == lang)
    }

    /// Total number of lyrics entries (synced + unsynced).
    pub fn total_count(&self) -> usize {
        self.unsynced.len() + self.synced.len()
    }

    /// Returns true if no lyrics are stored.
    pub fn is_empty(&self) -> bool {
        self.unsynced.is_empty() && self.synced.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- UnsyncedLyrics tests ----

    #[test]
    fn test_unsynced_lyrics_new() {
        let lyrics = UnsyncedLyrics::new("Hello world");
        assert_eq!(lyrics.text, "Hello world");
        assert_eq!(lyrics.language, "und");
        assert!(lyrics.description.is_empty());
        assert!(!lyrics.is_empty());
        assert_eq!(lyrics.line_count(), 1);
    }

    #[test]
    fn test_unsynced_lyrics_with_builders() {
        let lyrics = UnsyncedLyrics::new("Line 1\nLine 2\nLine 3")
            .with_language("eng")
            .with_description("Lyrics");

        assert_eq!(lyrics.language, "eng");
        assert_eq!(lyrics.description, "Lyrics");
        assert_eq!(lyrics.line_count(), 3);
    }

    #[test]
    fn test_unsynced_lyrics_empty() {
        let lyrics = UnsyncedLyrics::new("");
        assert!(lyrics.is_empty());
        assert_eq!(lyrics.line_count(), 0);
    }

    #[test]
    fn test_unsynced_lyrics_metadata_round_trip() {
        let original = UnsyncedLyrics::new("Verse 1\nChorus\nVerse 2")
            .with_language("eng")
            .with_description("Main lyrics");

        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        original.to_metadata(&mut metadata);

        let restored = UnsyncedLyrics::from_metadata(&metadata).expect("should parse");
        assert_eq!(restored.text, original.text);
        assert_eq!(restored.language, "eng");
        assert_eq!(restored.description, "Main lyrics");
    }

    #[test]
    fn test_unsynced_lyrics_from_metadata_missing() {
        let metadata = Metadata::new(MetadataFormat::Id3v2);
        assert!(UnsyncedLyrics::from_metadata(&metadata).is_none());
    }

    // ---- LyricLine tests ----

    #[test]
    fn test_lyric_line_new() {
        let line = LyricLine::new(5000, "Hello");
        assert_eq!(line.timestamp_ms, 5000);
        assert_eq!(line.text, "Hello");
        assert_eq!(line.end_ms, None);
        assert_eq!(line.duration_ms(), None);
    }

    #[test]
    fn test_lyric_line_with_end() {
        let line = LyricLine::new(5000, "Hello").with_end(8000);
        assert_eq!(line.end_ms, Some(8000));
        assert_eq!(line.duration_ms(), Some(3000));
    }

    #[test]
    fn test_lyric_line_lrc_timestamp() {
        let line = LyricLine::new(65_500, "Text");
        assert_eq!(line.lrc_timestamp(), "[01:05.50]");

        let line2 = LyricLine::new(0, "Start");
        assert_eq!(line2.lrc_timestamp(), "[00:00.00]");
    }

    // ---- SyncedContentType tests ----

    #[test]
    fn test_synced_content_type_round_trip() {
        let types = [
            SyncedContentType::Other,
            SyncedContentType::Lyrics,
            SyncedContentType::TextTranscription,
            SyncedContentType::MovementName,
            SyncedContentType::Events,
            SyncedContentType::Chord,
            SyncedContentType::Trivia,
        ];

        for ct in &types {
            let byte = ct.to_id3v2_byte();
            let restored = SyncedContentType::from_id3v2_byte(byte);
            assert_eq!(&restored, ct);
        }
    }

    #[test]
    fn test_synced_content_type_default() {
        assert_eq!(SyncedContentType::default(), SyncedContentType::Lyrics);
    }

    // ---- SyncedLyrics tests ----

    #[test]
    fn test_synced_lyrics_new() {
        let lyrics = SyncedLyrics::new();
        assert!(lyrics.is_empty());
        assert_eq!(lyrics.len(), 0);
        assert_eq!(lyrics.language, "und");
    }

    #[test]
    fn test_synced_lyrics_add_and_access() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.add_line(LyricLine::new(0, "Line 1"));
        lyrics.add_line(LyricLine::new(5000, "Line 2"));
        lyrics.add_line(LyricLine::new(10_000, "Line 3"));

        assert_eq!(lyrics.len(), 3);
        assert!(!lyrics.is_empty());
        assert_eq!(lyrics.lines()[0].text, "Line 1");
        assert_eq!(lyrics.total_duration_ms(), Some(10_000));
    }

    #[test]
    fn test_synced_lyrics_line_at() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.add_line(LyricLine::new(0, "Line 1"));
        lyrics.add_line(LyricLine::new(5000, "Line 2"));
        lyrics.add_line(LyricLine::new(10_000, "Line 3"));

        assert_eq!(lyrics.line_at(0).map(|l| l.text.as_str()), Some("Line 1"));
        assert_eq!(
            lyrics.line_at(3000).map(|l| l.text.as_str()),
            Some("Line 1")
        );
        assert_eq!(
            lyrics.line_at(5000).map(|l| l.text.as_str()),
            Some("Line 2")
        );
        assert_eq!(
            lyrics.line_at(7000).map(|l| l.text.as_str()),
            Some("Line 2")
        );
        assert_eq!(
            lyrics.line_at(15_000).map(|l| l.text.as_str()),
            Some("Line 3")
        );
    }

    #[test]
    fn test_synced_lyrics_sort_by_time() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.add_line(LyricLine::new(10_000, "Third"));
        lyrics.add_line(LyricLine::new(0, "First"));
        lyrics.add_line(LyricLine::new(5000, "Second"));

        lyrics.sort_by_time();

        assert_eq!(lyrics.lines()[0].text, "First");
        assert_eq!(lyrics.lines()[1].text, "Second");
        assert_eq!(lyrics.lines()[2].text, "Third");
    }

    #[test]
    fn test_synced_lyrics_apply_offset_positive() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.add_line(LyricLine::new(1000, "A"));
        lyrics.add_line(LyricLine::new(2000, "B"));

        lyrics.apply_offset(500);

        assert_eq!(lyrics.lines()[0].timestamp_ms, 1500);
        assert_eq!(lyrics.lines()[1].timestamp_ms, 2500);
    }

    #[test]
    fn test_synced_lyrics_apply_offset_negative() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.add_line(LyricLine::new(1000, "A"));
        lyrics.add_line(LyricLine::new(200, "B"));

        lyrics.apply_offset(-500);

        assert_eq!(lyrics.lines()[0].timestamp_ms, 500);
        assert_eq!(lyrics.lines()[1].timestamp_ms, 0); // saturating_sub
    }

    #[test]
    fn test_synced_lyrics_apply_offset_with_end() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.add_line(LyricLine::new(1000, "A").with_end(2000));

        lyrics.apply_offset(500);

        assert_eq!(lyrics.lines()[0].timestamp_ms, 1500);
        assert_eq!(lyrics.lines()[0].end_ms, Some(2500));
    }

    // ---- LRC format tests ----

    #[test]
    fn test_lrc_to_and_from() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.lrc_title = Some("My Song".to_string());
        lyrics.lrc_artist = Some("Artist".to_string());
        lyrics.add_line(LyricLine::new(0, "First line"));
        lyrics.add_line(LyricLine::new(5000, "Second line"));
        lyrics.add_line(LyricLine::new(10_000, "Third line"));

        let lrc = lyrics.to_lrc();

        assert!(lrc.contains("[ti:My Song]"));
        assert!(lrc.contains("[ar:Artist]"));
        assert!(lrc.contains("[00:00.00]First line"));
        assert!(lrc.contains("[00:05.00]Second line"));
        assert!(lrc.contains("[00:10.00]Third line"));

        // Parse it back
        let parsed = SyncedLyrics::from_lrc(&lrc).expect("parse should succeed");
        assert_eq!(parsed.lrc_title.as_deref(), Some("My Song"));
        assert_eq!(parsed.lrc_artist.as_deref(), Some("Artist"));
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed.lines()[0].text, "First line");
        assert_eq!(parsed.lines()[0].timestamp_ms, 0);
        assert_eq!(parsed.lines()[1].timestamp_ms, 5000);
        assert_eq!(parsed.lines()[2].timestamp_ms, 10_000);
    }

    #[test]
    fn test_lrc_with_all_metadata() {
        let mut lyrics = SyncedLyrics::new();
        lyrics.lrc_title = Some("Song".to_string());
        lyrics.lrc_artist = Some("Artist".to_string());
        lyrics.lrc_album = Some("Album".to_string());
        lyrics.lrc_author = Some("LRC Author".to_string());
        lyrics.lrc_offset_ms = Some(100);

        let lrc = lyrics.to_lrc();
        assert!(lrc.contains("[al:Album]"));
        assert!(lrc.contains("[by:LRC Author]"));
        assert!(lrc.contains("[offset:100]"));

        let parsed = SyncedLyrics::from_lrc(&lrc).expect("parse");
        assert_eq!(parsed.lrc_album.as_deref(), Some("Album"));
        assert_eq!(parsed.lrc_author.as_deref(), Some("LRC Author"));
        assert_eq!(parsed.lrc_offset_ms, Some(100));
    }

    #[test]
    fn test_lrc_parse_various_time_formats() {
        let lrc = "[01:05.50]Line with hundredths\n[02:30.123]Line with milliseconds\n";
        let parsed = SyncedLyrics::from_lrc(lrc).expect("parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.lines()[0].timestamp_ms, 65_500);
        assert_eq!(parsed.lines()[1].timestamp_ms, 150_123);
    }

    #[test]
    fn test_lrc_empty_input() {
        let parsed = SyncedLyrics::from_lrc("").expect("parse");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_lrc_whitespace_only() {
        let parsed = SyncedLyrics::from_lrc("   \n  \n").expect("parse");
        assert!(parsed.is_empty());
    }

    // ---- Metadata round-trip tests ----

    #[test]
    fn test_synced_lyrics_metadata_round_trip() {
        let mut lyrics = SyncedLyrics::new()
            .with_language("jpn")
            .with_content_type(SyncedContentType::Lyrics);

        lyrics.add_line(LyricLine::new(0, "First"));
        lyrics.add_line(LyricLine::new(5000, "Second"));

        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        lyrics.to_metadata(&mut metadata);

        let restored = SyncedLyrics::from_metadata(&metadata)
            .expect("should not fail")
            .expect("should have data");

        assert_eq!(restored.language, "jpn");
        assert_eq!(restored.content_type, SyncedContentType::Lyrics);
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn test_synced_lyrics_from_metadata_missing() {
        let metadata = Metadata::new(MetadataFormat::Id3v2);
        let result = SyncedLyrics::from_metadata(&metadata).expect("should not error");
        assert!(result.is_none());
    }

    // ---- LyricsCollection tests ----

    #[test]
    fn test_lyrics_collection_new() {
        let coll = LyricsCollection::new();
        assert!(coll.is_empty());
        assert_eq!(coll.total_count(), 0);
    }

    #[test]
    fn test_lyrics_collection_multi_language() {
        let mut coll = LyricsCollection::new();

        coll.add_unsynced(UnsyncedLyrics::new("English lyrics").with_language("eng"));
        coll.add_unsynced(UnsyncedLyrics::new("Japanese lyrics").with_language("jpn"));

        assert_eq!(coll.total_count(), 2);
        assert_eq!(
            coll.unsynced_for_language("eng").map(|l| l.text.as_str()),
            Some("English lyrics")
        );
        assert_eq!(
            coll.unsynced_for_language("jpn").map(|l| l.text.as_str()),
            Some("Japanese lyrics")
        );
        assert!(coll.unsynced_for_language("kor").is_none());
    }

    #[test]
    fn test_lyrics_collection_synced_language() {
        let mut coll = LyricsCollection::new();

        let mut eng_synced = SyncedLyrics::new().with_language("eng");
        eng_synced.add_line(LyricLine::new(0, "Hello"));
        coll.add_synced(eng_synced);

        assert_eq!(coll.total_count(), 1);
        assert!(coll.synced_for_language("eng").is_some());
        assert!(coll.synced_for_language("jpn").is_none());
    }

    // ---- LRC helper function tests ----

    #[test]
    fn test_format_lrc_time() {
        assert_eq!(format_lrc_time(0), "[00:00.00]");
        assert_eq!(format_lrc_time(1500), "[00:01.50]");
        assert_eq!(format_lrc_time(65_500), "[01:05.50]");
        assert_eq!(format_lrc_time(600_000), "[10:00.00]");
    }

    #[test]
    fn test_parse_lrc_time_valid() {
        assert_eq!(parse_lrc_time("00:00.00"), Some(0));
        assert_eq!(parse_lrc_time("01:05.50"), Some(65_500));
        assert_eq!(parse_lrc_time("10:00.00"), Some(600_000));
        assert_eq!(parse_lrc_time("01:05.500"), Some(65_500));
    }

    #[test]
    fn test_parse_lrc_time_invalid() {
        assert_eq!(parse_lrc_time("invalid"), None);
        assert_eq!(parse_lrc_time("00:00"), None);
        assert_eq!(parse_lrc_time("abc:de.fg"), None);
    }

    #[test]
    fn test_extract_lrc_tag() {
        assert_eq!(extract_lrc_tag("[ti:My Song]", "ti"), Some("My Song"));
        assert_eq!(
            extract_lrc_tag("[ar:Artist Name]", "ar"),
            Some("Artist Name")
        );
        assert_eq!(extract_lrc_tag("[offset:100]", "offset"), Some("100"));
        assert_eq!(extract_lrc_tag("[00:05.50]text", "ti"), None);
    }

    #[test]
    fn test_synced_lyrics_builders() {
        let lyrics = SyncedLyrics::new()
            .with_language("eng")
            .with_description("Main lyrics")
            .with_content_type(SyncedContentType::TextTranscription);

        assert_eq!(lyrics.language, "eng");
        assert_eq!(lyrics.description, "Main lyrics");
        assert_eq!(lyrics.content_type, SyncedContentType::TextTranscription);
    }

    #[test]
    fn test_synced_lyrics_empty_total_duration() {
        let lyrics = SyncedLyrics::new();
        assert_eq!(lyrics.total_duration_ms(), None);
    }
}
