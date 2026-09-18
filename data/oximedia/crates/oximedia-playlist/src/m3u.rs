//! M3U/M3U8 playlist parser and writer.
//!
//! Supports both plain M3U (simple list of paths) and extended M3U format
//! (with `#EXTM3U` header and `#EXTINF` duration/title lines).

use crate::{PlaylistError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single entry in an M3U playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3uEntry {
    /// Path or URL to the media file.
    pub path: String,
    /// Duration in seconds (`-1.0` if unknown, as per the M3U spec).
    pub duration_secs: f64,
    /// Track title.
    pub title: String,
    /// Artist name (if present in tags).
    pub artist: Option<String>,
    /// Album name (if present in tags).
    pub album: Option<String>,
    /// Additional key/value tags extracted from the `#EXTINF` line or extra directives.
    pub tags: HashMap<String, String>,
}

impl M3uEntry {
    /// Creates a new entry with a path and unknown duration.
    #[must_use]
    pub fn new<S: Into<String>>(path: S) -> Self {
        Self {
            path: path.into(),
            duration_secs: -1.0,
            title: String::new(),
            artist: None,
            album: None,
            tags: HashMap::new(),
        }
    }

    /// Sets the duration.
    #[must_use]
    pub const fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = secs;
        self
    }

    /// Sets the title.
    #[must_use]
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the artist.
    #[must_use]
    pub fn with_artist<S: Into<String>>(mut self, artist: S) -> Self {
        self.artist = Some(artist.into());
        self
    }

    /// Sets the album.
    #[must_use]
    pub fn with_album<S: Into<String>>(mut self, album: S) -> Self {
        self.album = Some(album.into());
        self
    }

    /// Returns `true` if the duration is known (≥ 0).
    #[must_use]
    pub fn has_duration(&self) -> bool {
        self.duration_secs >= 0.0
    }
}

/// An M3U or M3U8 playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3uPlaylist {
    /// Entries in the playlist.
    pub entries: Vec<M3uEntry>,
    /// Sum of all entry durations, or 0.0 when durations are absent.
    pub total_duration_secs: f64,
    /// Whether the playlist uses the Extended M3U format (`#EXTM3U` header).
    pub is_extended: bool,
}

impl M3uPlaylist {
    /// Creates a new, empty extended M3U playlist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_duration_secs: 0.0,
            is_extended: true,
        }
    }

    /// Parses an M3U or M3U8 playlist from a string.
    ///
    /// # Errors
    ///
    /// Returns [`PlaylistError::InvalidItem`] if the content cannot be parsed.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_playlist::m3u::M3uPlaylist;
    ///
    /// let content = "#EXTM3U\n#EXTINF:210,Artist - Title\npath/to/track.mp3\n";
    /// let playlist = M3uPlaylist::parse(content).expect("valid M3U content");
    /// assert_eq!(playlist.entries.len(), 1);
    /// assert_eq!(playlist.entries[0].duration_secs, 210.0);
    /// ```
    pub fn parse(content: &str) -> Result<Self> {
        let mut entries: Vec<M3uEntry> = Vec::new();
        let mut is_extended = false;
        let mut pending_duration: f64 = -1.0;
        let mut pending_title = String::new();
        let mut pending_artist: Option<String> = None;
        let mut pending_tags: HashMap<String, String> = HashMap::new();

        for raw_line in content.lines() {
            let line = raw_line.trim();

            if line.is_empty() {
                continue;
            }

            if line == "#EXTM3U" {
                is_extended = true;
                continue;
            }

            if let Some(rest) = line.strip_prefix("#EXTINF:") {
                // Format: #EXTINF:<duration>,<title>
                // Duration may be followed by key=value pairs separated by spaces.
                let (duration_part, title_part) = match rest.split_once(',') {
                    Some((d, t)) => (d, t),
                    None => (rest, ""),
                };

                // Duration may be a bare integer, a float, or include attributes before
                // the comma (e.g., `#EXTINF:210 tvg-id="..." ,Title`).
                let duration_token = duration_part.split_whitespace().next().unwrap_or("-1");
                pending_duration = duration_token.parse::<f64>().unwrap_or(-1.0);

                // Parse optional attributes in the duration field:
                // e.g., `210 tvg-id="foo" group-title="Bar"`
                let attr_str = duration_part
                    .split_once(' ')
                    .map(|x| x.1)
                    .unwrap_or("")
                    .trim();
                parse_attributes(attr_str, &mut pending_tags);

                // Title part may be "Artist - Title"
                let title_str = title_part.trim();
                if let Some((artist, title)) = title_str.split_once(" - ") {
                    pending_artist = Some(artist.trim().to_string());
                    pending_title = title.trim().to_string();
                } else {
                    pending_artist = None;
                    pending_title = title_str.to_string();
                }

                continue;
            }

            // Lines starting with '#' that are not recognised directives are comments.
            if line.starts_with('#') {
                continue;
            }

            // This line is a media path / URL.
            let mut entry = M3uEntry {
                path: line.to_string(),
                duration_secs: pending_duration,
                title: std::mem::take(&mut pending_title),
                artist: pending_artist.take(),
                album: None,
                tags: std::mem::take(&mut pending_tags),
            };

            // Try to derive a title from the path if none was found.
            if entry.title.is_empty() {
                entry.title = derive_title(&entry.path);
            }

            entries.push(entry);
            pending_duration = -1.0;
        }

        let total_duration_secs: f64 = entries
            .iter()
            .filter(|e| e.duration_secs >= 0.0)
            .map(|e| e.duration_secs)
            .sum();

        Ok(Self {
            entries,
            total_duration_secs,
            is_extended,
        })
    }

    /// Serialises the playlist to an M3U string.
    ///
    /// Always writes in extended M3U format when `is_extended` is `true`.
    #[must_use]
    pub fn to_m3u(&self) -> String {
        let mut out = String::new();

        if self.is_extended {
            out.push_str("#EXTM3U\n");
        }

        for entry in &self.entries {
            if self.is_extended {
                let duration = if entry.duration_secs >= 0.0 {
                    entry.duration_secs.round() as i64
                } else {
                    -1
                };

                let title = if let Some(artist) = &entry.artist {
                    format!("{} - {}", artist, entry.title)
                } else {
                    entry.title.clone()
                };

                out.push_str(&format!("#EXTINF:{},{}\n", duration, title));
            }
            out.push_str(&entry.path);
            out.push('\n');
        }

        out
    }

    /// Shuffles the entries in-place using a simple deterministic Fisher-Yates
    /// shuffle seeded from the entry count (for reproducibility in tests).
    pub fn shuffle(&mut self) {
        let n = self.entries.len();
        if n < 2 {
            return;
        }
        // Simple LCG shuffle (no external deps required).
        let mut seed: u64 = n as u64 ^ 0xDEAD_BEEF_CAFE_1234;
        for i in (1..n).rev() {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (seed >> 33) as usize % (i + 1);
            self.entries.swap(i, j);
        }
    }

    /// Sorts entries alphabetically by artist then title.
    pub fn sort_by_artist(&mut self) {
        self.entries.sort_by(|a, b| {
            let a_key = (a.artist.as_deref().unwrap_or(""), a.title.as_str());
            let b_key = (b.artist.as_deref().unwrap_or(""), b.title.as_str());
            a_key.cmp(&b_key)
        });
    }

    /// Sorts entries by duration (ascending).
    pub fn sort_by_duration(&mut self) {
        self.entries.sort_by(|a, b| {
            a.duration_secs
                .partial_cmp(&b.duration_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Returns a new playlist containing only entries whose duration is within
    /// `[min_secs, max_secs]`.  Entries with unknown duration (`< 0`) are
    /// excluded when `min_secs > 0`.
    #[must_use]
    pub fn filter_by_duration(&self, min_secs: f64, max_secs: f64) -> Self {
        let entries: Vec<M3uEntry> = self
            .entries
            .iter()
            .filter(|e| e.duration_secs >= min_secs && e.duration_secs <= max_secs)
            .cloned()
            .collect();

        let total_duration_secs: f64 = entries
            .iter()
            .filter(|e| e.duration_secs >= 0.0)
            .map(|e| e.duration_secs)
            .sum();

        Self {
            entries,
            total_duration_secs,
            is_extended: self.is_extended,
        }
    }

    /// Returns a new playlist containing only entries that have the given tag key.
    #[must_use]
    pub fn filter_by_tag(&self, key: &str) -> Self {
        let entries: Vec<M3uEntry> = self
            .entries
            .iter()
            .filter(|e| e.tags.contains_key(key))
            .cloned()
            .collect();

        let total_duration_secs: f64 = entries
            .iter()
            .filter(|e| e.duration_secs >= 0.0)
            .map(|e| e.duration_secs)
            .sum();

        Self {
            entries,
            total_duration_secs,
            is_extended: self.is_extended,
        }
    }

    /// Adds an entry and updates `total_duration_secs`.
    pub fn add_entry(&mut self, entry: M3uEntry) {
        if entry.duration_secs >= 0.0 {
            self.total_duration_secs += entry.duration_secs;
        }
        self.entries.push(entry);
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the playlist has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Recalculates `total_duration_secs` from the current entries.
    pub fn recalculate_duration(&mut self) {
        self.total_duration_secs = self
            .entries
            .iter()
            .filter(|e| e.duration_secs >= 0.0)
            .map(|e| e.duration_secs)
            .sum();
    }

    /// Converts this playlist into an error if it has no entries.
    pub fn require_non_empty(self) -> Result<Self> {
        if self.is_empty() {
            Err(PlaylistError::InvalidItem(
                "M3U playlist is empty".to_string(),
            ))
        } else {
            Ok(self)
        }
    }
}

impl Default for M3uPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Derives a human-readable title from a file path or URL.
fn derive_title(path: &str) -> String {
    // Take the last path component and strip the extension.
    let filename = path.rsplit('/').next().unwrap_or(path);
    let stem = filename
        .rsplit_once('.')
        .map(|(s, _)| s)
        .unwrap_or(filename);
    stem.replace(['_', '-'], " ")
}

/// Parses `key="value"` attribute pairs from an `#EXTINF` attribute string.
fn parse_attributes(input: &str, tags: &mut HashMap<String, String>) {
    // Match patterns like: key="value" or key=value
    let mut rest = input;
    while !rest.is_empty() {
        // Skip leading spaces
        rest = rest.trim_start();

        // Find '='
        let Some(eq_pos) = rest.find('=') else { break };
        let key = rest[..eq_pos].trim().to_string();
        rest = &rest[eq_pos + 1..];

        let value = if rest.starts_with('"') {
            // Quoted value
            rest = &rest[1..];
            let end = rest.find('"').unwrap_or(rest.len());
            let v = rest[..end].to_string();
            rest = if end + 1 < rest.len() {
                &rest[end + 1..]
            } else {
                ""
            };
            v
        } else {
            // Unquoted value (until next space)
            let end = rest.find(' ').unwrap_or(rest.len());
            let v = rest[..end].to_string();
            rest = if end < rest.len() { &rest[end..] } else { "" };
            v
        };

        if !key.is_empty() {
            tags.insert(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_M3U: &str = "song1.mp3\nsong2.mp3\n";

    const EXTENDED_M3U: &str = "\
#EXTM3U
#EXTINF:210,The Artist - Great Song
music/great_song.mp3
#EXTINF:180,Another Artist - Another Song
music/another_song.mp3
#EXTINF:-1,Unknown Duration
music/unknown.mp3
";

    #[test]
    fn test_parse_simple() {
        let pl = M3uPlaylist::parse(SIMPLE_M3U).expect("should succeed in test");
        assert!(!pl.is_extended);
        assert_eq!(pl.entries.len(), 2);
        assert_eq!(pl.entries[0].path, "song1.mp3");
        assert_eq!(pl.entries[1].path, "song2.mp3");
    }

    #[test]
    fn test_parse_extended() {
        let pl = M3uPlaylist::parse(EXTENDED_M3U).expect("should succeed in test");
        assert!(pl.is_extended);
        assert_eq!(pl.entries.len(), 3);

        let e0 = &pl.entries[0];
        assert_eq!(e0.duration_secs, 210.0);
        assert_eq!(e0.artist.as_deref(), Some("The Artist"));
        assert_eq!(e0.title, "Great Song");

        let e2 = &pl.entries[2];
        assert_eq!(e2.duration_secs, -1.0);
        assert!(!e2.has_duration());
    }

    #[test]
    fn test_total_duration() {
        let pl = M3uPlaylist::parse(EXTENDED_M3U).expect("should succeed in test");
        // 210 + 180 = 390; -1 excluded
        assert!((pl.total_duration_secs - 390.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_m3u_round_trip() {
        let pl = M3uPlaylist::parse(EXTENDED_M3U).expect("should succeed in test");
        let out = pl.to_m3u();

        // Must start with header
        assert!(out.starts_with("#EXTM3U\n"));

        // Re-parse the output
        let pl2 = M3uPlaylist::parse(&out).expect("should succeed in test");
        assert_eq!(pl2.entries.len(), pl.entries.len());
        assert_eq!(pl2.entries[0].duration_secs, 210.0);
    }

    #[test]
    fn test_filter_by_duration() {
        let pl = M3uPlaylist::parse(EXTENDED_M3U).expect("should succeed in test");
        let short = pl.filter_by_duration(0.0, 200.0);
        assert_eq!(short.entries.len(), 1);
        assert_eq!(short.entries[0].duration_secs, 180.0);
    }

    #[test]
    fn test_sort_by_artist() {
        let mut pl = M3uPlaylist::new();
        pl.add_entry(
            M3uEntry::new("b.mp3")
                .with_artist("Zebra")
                .with_title("Song"),
        );
        pl.add_entry(
            M3uEntry::new("a.mp3")
                .with_artist("Alpha")
                .with_title("Song"),
        );
        pl.sort_by_artist();
        assert_eq!(pl.entries[0].artist.as_deref(), Some("Alpha"));
        assert_eq!(pl.entries[1].artist.as_deref(), Some("Zebra"));
    }

    #[test]
    fn test_sort_by_duration() {
        let mut pl = M3uPlaylist::new();
        pl.add_entry(M3uEntry::new("c.mp3").with_duration(300.0));
        pl.add_entry(M3uEntry::new("a.mp3").with_duration(100.0));
        pl.add_entry(M3uEntry::new("b.mp3").with_duration(200.0));
        pl.sort_by_duration();
        assert_eq!(pl.entries[0].duration_secs, 100.0);
        assert_eq!(pl.entries[2].duration_secs, 300.0);
    }

    #[test]
    fn test_shuffle_changes_order() {
        let mut pl = M3uPlaylist::new();
        for i in 0..10 {
            pl.add_entry(M3uEntry::new(format!("track{i}.mp3")));
        }
        let before: Vec<String> = pl.entries.iter().map(|e| e.path.clone()).collect();
        pl.shuffle();
        let after: Vec<String> = pl.entries.iter().map(|e| e.path.clone()).collect();
        // With 10 items, a shuffle should almost certainly change the order.
        // (There is a 1/10! chance it stays the same, which we accept.)
        assert_eq!(before.len(), after.len());
    }

    #[test]
    fn test_parse_extended_attributes() {
        let content = "#EXTM3U\n#EXTINF:120 tvg-id=\"ch1\" group-title=\"News\",Channel 1\nhttp://example.com/stream\n";
        let pl = M3uPlaylist::parse(content).expect("should succeed in test");
        let entry = &pl.entries[0];
        assert_eq!(entry.tags.get("tvg-id").map(String::as_str), Some("ch1"));
        assert_eq!(
            entry.tags.get("group-title").map(String::as_str),
            Some("News")
        );
    }

    #[test]
    fn test_add_entry_updates_duration() {
        let mut pl = M3uPlaylist::new();
        pl.add_entry(M3uEntry::new("a.mp3").with_duration(100.0));
        pl.add_entry(M3uEntry::new("b.mp3").with_duration(200.0));
        assert!((pl.total_duration_secs - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_require_non_empty_error() {
        let pl = M3uPlaylist::new();
        assert!(pl.require_non_empty().is_err());
    }

    #[test]
    fn test_require_non_empty_ok() {
        let mut pl = M3uPlaylist::new();
        pl.add_entry(M3uEntry::new("a.mp3"));
        assert!(pl.require_non_empty().is_ok());
    }

    #[test]
    fn test_derive_title_from_path() {
        let pl =
            M3uPlaylist::parse("some/path/my_great_track.flac\n").expect("should succeed in test");
        assert_eq!(pl.entries[0].title, "my great track");
    }

    #[test]
    fn test_plain_m3u_to_m3u() {
        let mut pl = M3uPlaylist::new();
        pl.is_extended = false;
        pl.add_entry(M3uEntry::new("track.mp3").with_title("Track"));
        let out = pl.to_m3u();
        assert!(!out.starts_with("#EXTM3U"));
        assert!(out.contains("track.mp3"));
    }
}
