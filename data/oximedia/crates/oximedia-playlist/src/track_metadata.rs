//! Track metadata types and in-memory cache for the playlist system.
//!
//! Provides `TrackGenre`, `TrackMetadata`, and `TrackMetadataCache`.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Genre classification for playlist tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackGenre {
    /// Pop music.
    Pop,
    /// Rock / indie.
    Rock,
    /// Jazz and blues.
    Jazz,
    /// Classical / orchestral.
    Classical,
    /// Electronic / EDM.
    Electronic,
    /// Hip-hop / R&B.
    HipHop,
    /// Country / folk.
    Country,
    /// Latin music.
    Latin,
    /// World / roots music.
    World,
    /// Genre not yet classified.
    Unknown,
}

impl TrackGenre {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pop => "Pop",
            Self::Rock => "Rock",
            Self::Jazz => "Jazz",
            Self::Classical => "Classical",
            Self::Electronic => "Electronic",
            Self::HipHop => "Hip-Hop",
            Self::Country => "Country",
            Self::Latin => "Latin",
            Self::World => "World",
            Self::Unknown => "Unknown",
        }
    }

    /// Parse a genre from a string (case-insensitive).
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "pop" => Self::Pop,
            "rock" | "indie" => Self::Rock,
            "jazz" | "blues" => Self::Jazz,
            "classical" | "orchestral" => Self::Classical,
            "electronic" | "edm" | "dance" => Self::Electronic,
            "hip-hop" | "hiphop" | "r&b" | "rnb" => Self::HipHop,
            "country" | "folk" => Self::Country,
            "latin" => Self::Latin,
            "world" | "roots" => Self::World,
            _ => Self::Unknown,
        }
    }
}

/// Rich metadata for a single track.
#[derive(Debug, Clone)]
pub struct TrackMetadata {
    /// Unique track identifier (e.g. ISRC or internal ID).
    pub id: String,
    /// Track title.
    pub title: String,
    /// Primary artist name.
    pub artist: String,
    /// Album name.
    pub album: String,
    /// Genre classification.
    pub genre: TrackGenre,
    /// Duration in seconds.
    pub duration_s: f64,
    /// Release year (if known).
    pub year: Option<u16>,
    /// BPM as detected or tagged.
    pub bpm: Option<f32>,
    /// Whether this track contains explicit content.
    pub explicit: bool,
    /// Language code (ISO 639-1, e.g. "en").
    pub language: Option<String>,
    /// Play count (for statistics / recommendation).
    pub play_count: u64,
}

impl TrackMetadata {
    /// Create a minimal track metadata record.
    #[must_use]
    pub fn new(id: &str, title: &str, artist: &str, genre: TrackGenre, duration_s: f64) -> Self {
        Self {
            id: id.to_owned(),
            title: title.to_owned(),
            artist: artist.to_owned(),
            album: String::new(),
            genre,
            duration_s,
            year: None,
            bpm: None,
            explicit: false,
            language: None,
            play_count: 0,
        }
    }

    /// Whether this track is marked as explicit.
    #[must_use]
    pub fn is_explicit(&self) -> bool {
        self.explicit
    }

    /// Whether the stored duration is in a reasonable broadcast range (5s–3h).
    #[must_use]
    pub fn duration_valid(&self) -> bool {
        self.duration_s >= 5.0 && self.duration_s <= 10_800.0
    }

    /// Return a display string "Artist – Title".
    #[must_use]
    pub fn display(&self) -> String {
        format!("{} – {}", self.artist, self.title)
    }

    /// Increment the play count.
    pub fn increment_plays(&mut self) {
        self.play_count += 1;
    }

    /// Duration in whole minutes.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    pub fn duration_minutes(&self) -> u32 {
        (self.duration_s / 60.0) as u32
    }
}

/// In-memory cache for track metadata, keyed by track ID.
#[derive(Debug, Clone, Default)]
pub struct TrackMetadataCache {
    store: HashMap<String, TrackMetadata>,
}

impl TrackMetadataCache {
    /// Create a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a track record.
    pub fn insert(&mut self, meta: TrackMetadata) {
        self.store.insert(meta.id.clone(), meta);
    }

    /// Look up a track by ID.
    #[must_use]
    pub fn lookup(&self, id: &str) -> Option<&TrackMetadata> {
        self.store.get(id)
    }

    /// Look up a track mutably by ID.
    #[must_use]
    pub fn lookup_mut(&mut self, id: &str) -> Option<&mut TrackMetadata> {
        self.store.get_mut(id)
    }

    /// Return the number of entries in the cache.
    #[must_use]
    pub fn count(&self) -> usize {
        self.store.len()
    }

    /// Remove a track by ID.  Returns the record if it existed.
    pub fn remove(&mut self, id: &str) -> Option<TrackMetadata> {
        self.store.remove(id)
    }

    /// Return all tracks with a given genre.
    #[must_use]
    pub fn tracks_by_genre(&self, genre: TrackGenre) -> Vec<&TrackMetadata> {
        self.store.values().filter(|t| t.genre == genre).collect()
    }

    /// Return all tracks by a given artist.
    #[must_use]
    pub fn tracks_by_artist(&self, artist: &str) -> Vec<&TrackMetadata> {
        self.store
            .values()
            .filter(|t| t.artist.eq_ignore_ascii_case(artist))
            .collect()
    }

    /// Return IDs of all explicit tracks.
    #[must_use]
    pub fn explicit_ids(&self) -> Vec<&str> {
        self.store
            .values()
            .filter(|t| t.explicit)
            .map(|t| t.id.as_str())
            .collect()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_track(id: &str) -> TrackMetadata {
        TrackMetadata::new(id, "Test Track", "Artist A", TrackGenre::Pop, 210.0)
    }

    // TrackGenre tests

    #[test]
    fn test_genre_labels_non_empty() {
        let genres = [
            TrackGenre::Pop,
            TrackGenre::Rock,
            TrackGenre::Jazz,
            TrackGenre::Classical,
            TrackGenre::Electronic,
            TrackGenre::HipHop,
            TrackGenre::Country,
            TrackGenre::Latin,
            TrackGenre::World,
            TrackGenre::Unknown,
        ];
        for g in genres {
            assert!(!g.label().is_empty());
        }
    }

    #[test]
    fn test_genre_parse_pop() {
        assert_eq!(TrackGenre::from_str("pop"), TrackGenre::Pop);
        assert_eq!(TrackGenre::from_str("POP"), TrackGenre::Pop);
    }

    #[test]
    fn test_genre_parse_electronic_aliases() {
        assert_eq!(TrackGenre::from_str("edm"), TrackGenre::Electronic);
        assert_eq!(TrackGenre::from_str("dance"), TrackGenre::Electronic);
    }

    #[test]
    fn test_genre_parse_unknown_fallback() {
        assert_eq!(TrackGenre::from_str("ska"), TrackGenre::Unknown);
    }

    // TrackMetadata tests

    #[test]
    fn test_track_is_explicit_default_false() {
        let t = sample_track("t1");
        assert!(!t.is_explicit());
    }

    #[test]
    fn test_track_is_explicit_when_set() {
        let mut t = sample_track("t2");
        t.explicit = true;
        assert!(t.is_explicit());
    }

    #[test]
    fn test_duration_valid_normal() {
        let t = sample_track("t3");
        assert!(t.duration_valid());
    }

    #[test]
    fn test_duration_invalid_too_short() {
        let mut t = sample_track("t4");
        t.duration_s = 2.0;
        assert!(!t.duration_valid());
    }

    #[test]
    fn test_duration_invalid_too_long() {
        let mut t = sample_track("t5");
        t.duration_s = 20_000.0;
        assert!(!t.duration_valid());
    }

    #[test]
    fn test_display_format() {
        let t = sample_track("t6");
        assert!(t.display().contains("Artist A"));
        assert!(t.display().contains("Test Track"));
    }

    #[test]
    fn test_increment_plays() {
        let mut t = sample_track("t7");
        t.increment_plays();
        t.increment_plays();
        assert_eq!(t.play_count, 2);
    }

    // TrackMetadataCache tests

    #[test]
    fn test_cache_empty_initially() {
        let cache = TrackMetadataCache::new();
        assert_eq!(cache.count(), 0);
    }

    #[test]
    fn test_cache_insert_and_lookup() {
        let mut cache = TrackMetadataCache::new();
        cache.insert(sample_track("t1"));
        assert!(cache.lookup("t1").is_some());
        assert_eq!(cache.count(), 1);
    }

    #[test]
    fn test_cache_lookup_missing_returns_none() {
        let cache = TrackMetadataCache::new();
        assert!(cache.lookup("missing").is_none());
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = TrackMetadataCache::new();
        cache.insert(sample_track("t1"));
        let removed = cache.remove("t1");
        assert!(removed.is_some());
        assert_eq!(cache.count(), 0);
    }

    #[test]
    fn test_cache_tracks_by_genre() {
        let mut cache = TrackMetadataCache::new();
        cache.insert(sample_track("pop1"));
        let mut jazz = sample_track("jazz1");
        jazz.genre = TrackGenre::Jazz;
        cache.insert(jazz);
        let pops = cache.tracks_by_genre(TrackGenre::Pop);
        assert_eq!(pops.len(), 1);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = TrackMetadataCache::new();
        for i in 0..5 {
            cache.insert(sample_track(&format!("t{i}")));
        }
        cache.clear();
        assert_eq!(cache.count(), 0);
    }
}
