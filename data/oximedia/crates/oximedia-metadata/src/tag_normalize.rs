#![allow(dead_code)]
//! Tag normalization utilities for harmonizing metadata field names across formats.
//!
//! Different metadata standards use different naming conventions for the same
//! conceptual field (e.g., "TIT2" in ID3v2, "TITLE" in Vorbis, "\u{00a9}nam" in iTunes).
//! This module provides canonical normalization so downstream code can work with a
//! single unified namespace.

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

/// The forward and reverse default-mapping tables, built exactly once.
///
/// `with_defaults()` / `default()` previously rebuilt all ~63 default mappings
/// on *every* construction. Building them once into this process-wide
/// `OnceLock` and cloning the (small, owned) maps keeps behaviour identical
/// while eliminating the repeated insert work and string allocations.
type ForwardMap = HashMap<(TagFormat, String), CanonicalTag>;
type ReverseMap = HashMap<(TagFormat, CanonicalTag), String>;
static DEFAULT_MAPPINGS: OnceLock<(ForwardMap, ReverseMap)> = OnceLock::new();

/// Return the shared default mapping tables, initializing them on first use.
fn default_mappings() -> &'static (ForwardMap, ReverseMap) {
    DEFAULT_MAPPINGS.get_or_init(|| {
        let mut n = TagNormalizer::new();
        n.register_defaults();
        (n.forward, n.reverse)
    })
}

/// Canonical tag categories for cross-format normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalTag {
    /// Track/song title.
    Title,
    /// Artist or performer.
    Artist,
    /// Album name.
    Album,
    /// Album artist (may differ from track artist).
    AlbumArtist,
    /// Composer.
    Composer,
    /// Genre.
    Genre,
    /// Track number.
    TrackNumber,
    /// Total number of tracks.
    TrackTotal,
    /// Disc number.
    DiscNumber,
    /// Total number of discs.
    DiscTotal,
    /// Year or date of release.
    Year,
    /// Comment.
    Comment,
    /// Lyrics.
    Lyrics,
    /// Encoder software.
    Encoder,
    /// Copyright notice.
    Copyright,
    /// Publisher or label.
    Publisher,
    /// BPM (beats per minute).
    Bpm,
    /// ISRC (International Standard Recording Code).
    Isrc,
}

impl fmt::Display for CanonicalTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Title => write!(f, "Title"),
            Self::Artist => write!(f, "Artist"),
            Self::Album => write!(f, "Album"),
            Self::AlbumArtist => write!(f, "Album Artist"),
            Self::Composer => write!(f, "Composer"),
            Self::Genre => write!(f, "Genre"),
            Self::TrackNumber => write!(f, "Track Number"),
            Self::TrackTotal => write!(f, "Track Total"),
            Self::DiscNumber => write!(f, "Disc Number"),
            Self::DiscTotal => write!(f, "Disc Total"),
            Self::Year => write!(f, "Year"),
            Self::Comment => write!(f, "Comment"),
            Self::Lyrics => write!(f, "Lyrics"),
            Self::Encoder => write!(f, "Encoder"),
            Self::Copyright => write!(f, "Copyright"),
            Self::Publisher => write!(f, "Publisher"),
            Self::Bpm => write!(f, "BPM"),
            Self::Isrc => write!(f, "ISRC"),
        }
    }
}

/// Format-specific tag name registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagFormat {
    /// ID3v2 four-character frame IDs.
    Id3v2,
    /// Vorbis comment field names (case-insensitive by spec).
    Vorbis,
    /// iTunes/MP4 atom names.
    ITunes,
    /// APEv2 tag keys.
    Ape,
}

impl fmt::Display for TagFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id3v2 => write!(f, "ID3v2"),
            Self::Vorbis => write!(f, "Vorbis"),
            Self::ITunes => write!(f, "iTunes"),
            Self::Ape => write!(f, "APE"),
        }
    }
}

/// Mapping entry that associates a format-specific tag key with a canonical tag.
#[derive(Debug, Clone)]
pub struct TagMapping {
    /// The source format.
    pub format: TagFormat,
    /// The format-specific key string.
    pub key: String,
    /// The canonical tag it maps to.
    pub canonical: CanonicalTag,
}

impl TagMapping {
    /// Create a new tag mapping.
    pub fn new(format: TagFormat, key: &str, canonical: CanonicalTag) -> Self {
        Self {
            format,
            key: key.to_string(),
            canonical,
        }
    }
}

/// The normalizer engine that translates between format-specific keys and canonical tags.
#[derive(Debug, Clone)]
pub struct TagNormalizer {
    /// Map from (format, lowercase_key) -> canonical tag.
    forward: HashMap<(TagFormat, String), CanonicalTag>,
    /// Map from (format, canonical) -> preferred key string.
    reverse: HashMap<(TagFormat, CanonicalTag), String>,
}

impl TagNormalizer {
    /// Create a new empty normalizer.
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Create a normalizer pre-populated with standard mappings.
    ///
    /// The default mapping tables are built once per process (see
    /// `default_mappings`) and cloned here, rather than re-registering all
    /// ~63 default entries on every call.
    pub fn with_defaults() -> Self {
        let (forward, reverse) = default_mappings();
        Self {
            forward: forward.clone(),
            reverse: reverse.clone(),
        }
    }

    /// Register a single mapping.
    pub fn register(&mut self, mapping: TagMapping) {
        let lower = mapping.key.to_lowercase();
        self.forward
            .insert((mapping.format, lower), mapping.canonical);
        self.reverse
            .insert((mapping.format, mapping.canonical), mapping.key);
    }

    /// Register all well-known default mappings for ID3v2, Vorbis, iTunes, APE.
    pub fn register_defaults(&mut self) {
        // ID3v2 mappings
        let id3_mappings = [
            ("TIT2", CanonicalTag::Title),
            ("TPE1", CanonicalTag::Artist),
            ("TALB", CanonicalTag::Album),
            ("TPE2", CanonicalTag::AlbumArtist),
            ("TCOM", CanonicalTag::Composer),
            ("TCON", CanonicalTag::Genre),
            ("TRCK", CanonicalTag::TrackNumber),
            ("TPOS", CanonicalTag::DiscNumber),
            ("TDRC", CanonicalTag::Year),
            ("COMM", CanonicalTag::Comment),
            ("USLT", CanonicalTag::Lyrics),
            ("TSSE", CanonicalTag::Encoder),
            ("TCOP", CanonicalTag::Copyright),
            ("TPUB", CanonicalTag::Publisher),
            ("TBPM", CanonicalTag::Bpm),
            ("TSRC", CanonicalTag::Isrc),
        ];
        for (key, canonical) in &id3_mappings {
            self.register(TagMapping::new(TagFormat::Id3v2, key, *canonical));
        }

        // Vorbis comment mappings
        let vorbis_mappings = [
            ("TITLE", CanonicalTag::Title),
            ("ARTIST", CanonicalTag::Artist),
            ("ALBUM", CanonicalTag::Album),
            ("ALBUMARTIST", CanonicalTag::AlbumArtist),
            ("COMPOSER", CanonicalTag::Composer),
            ("GENRE", CanonicalTag::Genre),
            ("TRACKNUMBER", CanonicalTag::TrackNumber),
            ("TRACKTOTAL", CanonicalTag::TrackTotal),
            ("DISCNUMBER", CanonicalTag::DiscNumber),
            ("DISCTOTAL", CanonicalTag::DiscTotal),
            ("DATE", CanonicalTag::Year),
            ("COMMENT", CanonicalTag::Comment),
            ("LYRICS", CanonicalTag::Lyrics),
            ("ENCODER", CanonicalTag::Encoder),
            ("COPYRIGHT", CanonicalTag::Copyright),
            ("LABEL", CanonicalTag::Publisher),
            ("BPM", CanonicalTag::Bpm),
            ("ISRC", CanonicalTag::Isrc),
        ];
        for (key, canonical) in &vorbis_mappings {
            self.register(TagMapping::new(TagFormat::Vorbis, key, *canonical));
        }

        // iTunes/MP4 mappings
        let itunes_mappings = [
            ("\u{00a9}nam", CanonicalTag::Title),
            ("\u{00a9}ART", CanonicalTag::Artist),
            ("\u{00a9}alb", CanonicalTag::Album),
            ("aART", CanonicalTag::AlbumArtist),
            ("\u{00a9}wrt", CanonicalTag::Composer),
            ("\u{00a9}gen", CanonicalTag::Genre),
            ("trkn", CanonicalTag::TrackNumber),
            ("disk", CanonicalTag::DiscNumber),
            ("\u{00a9}day", CanonicalTag::Year),
            ("\u{00a9}cmt", CanonicalTag::Comment),
            ("\u{00a9}too", CanonicalTag::Encoder),
            ("cprt", CanonicalTag::Copyright),
            ("tmpo", CanonicalTag::Bpm),
        ];
        for (key, canonical) in &itunes_mappings {
            self.register(TagMapping::new(TagFormat::ITunes, key, *canonical));
        }

        // APE mappings
        let ape_mappings = [
            ("Title", CanonicalTag::Title),
            ("Artist", CanonicalTag::Artist),
            ("Album", CanonicalTag::Album),
            ("Album Artist", CanonicalTag::AlbumArtist),
            ("Composer", CanonicalTag::Composer),
            ("Genre", CanonicalTag::Genre),
            ("Track", CanonicalTag::TrackNumber),
            ("Disc", CanonicalTag::DiscNumber),
            ("Year", CanonicalTag::Year),
            ("Comment", CanonicalTag::Comment),
            ("Lyrics", CanonicalTag::Lyrics),
            ("Encoder", CanonicalTag::Encoder),
            ("Copyright", CanonicalTag::Copyright),
            ("Publisher", CanonicalTag::Publisher),
            ("BPM", CanonicalTag::Bpm),
            ("ISRC", CanonicalTag::Isrc),
        ];
        for (key, canonical) in &ape_mappings {
            self.register(TagMapping::new(TagFormat::Ape, key, *canonical));
        }
    }

    /// Normalize a format-specific key to its canonical tag.
    ///
    /// The lookup is case-insensitive.
    pub fn normalize(&self, format: TagFormat, key: &str) -> Option<CanonicalTag> {
        let lower = key.to_lowercase();
        self.forward.get(&(format, lower)).copied()
    }

    /// Get the preferred format-specific key for a canonical tag.
    pub fn denormalize(&self, format: TagFormat, canonical: CanonicalTag) -> Option<&str> {
        self.reverse.get(&(format, canonical)).map(|s| s.as_str())
    }

    /// Number of registered forward mappings.
    pub fn mapping_count(&self) -> usize {
        self.forward.len()
    }

    /// Check if a specific mapping exists.
    pub fn has_mapping(&self, format: TagFormat, key: &str) -> bool {
        let lower = key.to_lowercase();
        self.forward.contains_key(&(format, lower))
    }
}

impl Default for TagNormalizer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Normalize a raw string tag key by trimming whitespace and lowercasing.
pub fn sanitize_key(key: &str) -> String {
    key.trim().to_lowercase()
}

/// Split a combined track number field like "3/12" into (track, total).
pub fn split_track_field(value: &str) -> (Option<u32>, Option<u32>) {
    let parts: Vec<&str> = value.split('/').collect();
    let track = parts.first().and_then(|s| s.trim().parse::<u32>().ok());
    let total = parts.get(1).and_then(|s| s.trim().parse::<u32>().ok());
    (track, total)
}

/// Join track number and total into a combined string like "3/12".
pub fn join_track_field(track: u32, total: Option<u32>) -> String {
    match total {
        Some(t) => format!("{track}/{t}"),
        None => format!("{track}"),
    }
}

/// Normalize a genre string by stripping parenthesized numeric ID3v1 genre codes.
///
/// ID3v2 genre frames sometimes contain numeric references like "(17)" for "Rock".
pub fn normalize_genre(raw: &str) -> String {
    let trimmed = raw.trim();

    // Handle pure numeric like "(17)"
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        if let Some(inner) = trimmed.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            if let Ok(code) = inner.parse::<u8>() {
                return id3v1_genre_name(code).to_string();
            }
        }
    }

    // Handle mixed like "(17)Rock" — keep the text portion
    if trimmed.starts_with('(') {
        if let Some(close) = trimmed.find(')') {
            let rest = &trimmed[close + 1..];
            if !rest.is_empty() {
                return rest.trim().to_string();
            }
            // If only "(17)" with no trailing text, try the code
            let inner = &trimmed[1..close];
            if let Ok(code) = inner.parse::<u8>() {
                return id3v1_genre_name(code).to_string();
            }
        }
    }

    trimmed.to_string()
}

/// Lookup table for ID3v1 genre codes (subset of the most common).
fn id3v1_genre_name(code: u8) -> &'static str {
    match code {
        0 => "Blues",
        1 => "Classic Rock",
        2 => "Country",
        3 => "Dance",
        4 => "Disco",
        5 => "Funk",
        6 => "Grunge",
        7 => "Hip-Hop",
        8 => "Jazz",
        9 => "Metal",
        10 => "New Age",
        11 => "Oldies",
        12 => "Other",
        13 => "Pop",
        14 => "Rhythm and Blues",
        15 => "Rap",
        16 => "Reggae",
        17 => "Rock",
        18 => "Techno",
        19 => "Industrial",
        20 => "Alternative",
        21 => "Ska",
        22 => "Death Metal",
        23 => "Pranks",
        24 => "Soundtrack",
        25 => "Euro-Techno",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_tag_display() {
        assert_eq!(CanonicalTag::Title.to_string(), "Title");
        assert_eq!(CanonicalTag::AlbumArtist.to_string(), "Album Artist");
        assert_eq!(CanonicalTag::Bpm.to_string(), "BPM");
        assert_eq!(CanonicalTag::Isrc.to_string(), "ISRC");
    }

    #[test]
    fn test_tag_format_display() {
        assert_eq!(TagFormat::Id3v2.to_string(), "ID3v2");
        assert_eq!(TagFormat::Vorbis.to_string(), "Vorbis");
        assert_eq!(TagFormat::ITunes.to_string(), "iTunes");
        assert_eq!(TagFormat::Ape.to_string(), "APE");
    }

    #[test]
    fn test_normalizer_id3v2_title() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(
            n.normalize(TagFormat::Id3v2, "TIT2"),
            Some(CanonicalTag::Title)
        );
    }

    #[test]
    fn test_normalizer_vorbis_artist() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(
            n.normalize(TagFormat::Vorbis, "ARTIST"),
            Some(CanonicalTag::Artist)
        );
    }

    #[test]
    fn test_normalizer_case_insensitive() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(
            n.normalize(TagFormat::Vorbis, "artist"),
            Some(CanonicalTag::Artist)
        );
        assert_eq!(
            n.normalize(TagFormat::Vorbis, "ArTiSt"),
            Some(CanonicalTag::Artist)
        );
    }

    #[test]
    fn test_normalizer_unknown_key() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(n.normalize(TagFormat::Id3v2, "ZZZZ"), None);
    }

    #[test]
    fn test_denormalize_id3v2() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(
            n.denormalize(TagFormat::Id3v2, CanonicalTag::Title),
            Some("TIT2")
        );
    }

    #[test]
    fn test_denormalize_vorbis() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(
            n.denormalize(TagFormat::Vorbis, CanonicalTag::Album),
            Some("ALBUM")
        );
    }

    #[test]
    fn test_denormalize_itunes() {
        let n = TagNormalizer::with_defaults();
        assert_eq!(
            n.denormalize(TagFormat::ITunes, CanonicalTag::Title),
            Some("\u{00a9}nam")
        );
    }

    #[test]
    fn test_mapping_count() {
        let n = TagNormalizer::with_defaults();
        assert!(n.mapping_count() > 50);
    }

    #[test]
    fn test_has_mapping() {
        let n = TagNormalizer::with_defaults();
        assert!(n.has_mapping(TagFormat::Id3v2, "TIT2"));
        assert!(!n.has_mapping(TagFormat::Id3v2, "FAKE"));
    }

    #[test]
    fn test_sanitize_key() {
        assert_eq!(sanitize_key("  Title  "), "title");
        assert_eq!(sanitize_key("ARTIST"), "artist");
    }

    #[test]
    fn test_split_track_field() {
        assert_eq!(split_track_field("3/12"), (Some(3), Some(12)));
        assert_eq!(split_track_field("7"), (Some(7), None));
        assert_eq!(split_track_field("abc"), (None, None));
        assert_eq!(split_track_field("5 / 10"), (Some(5), Some(10)));
    }

    #[test]
    fn test_join_track_field() {
        assert_eq!(join_track_field(3, Some(12)), "3/12");
        assert_eq!(join_track_field(7, None), "7");
    }

    #[test]
    fn test_normalize_genre_numeric() {
        assert_eq!(normalize_genre("(17)"), "Rock");
        assert_eq!(normalize_genre("(0)"), "Blues");
    }

    #[test]
    fn test_normalize_genre_mixed() {
        assert_eq!(normalize_genre("(17)Rock"), "Rock");
        assert_eq!(normalize_genre("(13)Pop"), "Pop");
    }

    #[test]
    fn test_normalize_genre_plain() {
        assert_eq!(normalize_genre("Electronic"), "Electronic");
        assert_eq!(normalize_genre("  Jazz  "), "Jazz");
    }

    #[test]
    fn test_custom_mapping_registration() {
        let mut n = TagNormalizer::new();
        n.register(TagMapping::new(
            TagFormat::Id3v2,
            "TXXX:MOOD",
            CanonicalTag::Genre,
        ));
        assert_eq!(
            n.normalize(TagFormat::Id3v2, "TXXX:MOOD"),
            Some(CanonicalTag::Genre)
        );
        assert_eq!(n.mapping_count(), 1);
    }

    #[test]
    fn test_normalizer_default_trait() {
        let n = TagNormalizer::default();
        assert!(n.mapping_count() > 0);
    }

    #[test]
    fn test_default_mapping_count_is_stable() {
        // The shared (OnceLock) default tables must always yield the same number
        // of forward mappings: ID3v2 (16) + Vorbis (18) + iTunes (13) + APE (16).
        const EXPECTED_FORWARD: usize = 63;
        let a = TagNormalizer::with_defaults();
        let b = TagNormalizer::with_defaults();
        assert_eq!(a.mapping_count(), EXPECTED_FORWARD);
        assert_eq!(b.mapping_count(), EXPECTED_FORWARD);
        // Two independent constructions must agree on both directions.
        assert_eq!(a.forward.len(), b.forward.len());
        assert_eq!(a.reverse.len(), b.reverse.len());
    }

    #[test]
    fn test_default_normalizer_constructed_twice_equal_behavior() {
        // Build the default normalizer twice; the OnceLock-shared tables must
        // produce identical normalization for a known input across both.
        let first = TagNormalizer::with_defaults();
        let second = TagNormalizer::default();

        // Forward: a known ID3v2 frame key -> canonical tag, on both.
        assert_eq!(
            first.normalize(TagFormat::Id3v2, "TPE2"),
            Some(CanonicalTag::AlbumArtist)
        );
        assert_eq!(
            second.normalize(TagFormat::Id3v2, "TPE2"),
            Some(CanonicalTag::AlbumArtist)
        );

        // Case-insensitive Vorbis lookup, on both.
        assert_eq!(
            first.normalize(TagFormat::Vorbis, "albumartist"),
            Some(CanonicalTag::AlbumArtist)
        );
        assert_eq!(
            second.normalize(TagFormat::Vorbis, "ALBUMARTIST"),
            Some(CanonicalTag::AlbumArtist)
        );

        // Reverse: canonical -> preferred key, on both.
        assert_eq!(
            first.denormalize(TagFormat::ITunes, CanonicalTag::Title),
            Some("\u{00a9}nam")
        );
        assert_eq!(
            second.denormalize(TagFormat::ITunes, CanonicalTag::Title),
            Some("\u{00a9}nam")
        );
    }

    #[test]
    fn test_with_defaults_clone_is_independent() {
        // A normalizer cloned from the shared tables must be independently
        // mutable: registering a custom mapping on one must not affect another.
        let mut a = TagNormalizer::with_defaults();
        let base = a.mapping_count();
        a.register(TagMapping::new(
            TagFormat::Id3v2,
            "TXXX:CUSTOM",
            CanonicalTag::Comment,
        ));
        let b = TagNormalizer::with_defaults();
        assert_eq!(a.mapping_count(), base + 1);
        assert_eq!(b.mapping_count(), base);
        assert!(b.normalize(TagFormat::Id3v2, "TXXX:CUSTOM").is_none());
    }
}
