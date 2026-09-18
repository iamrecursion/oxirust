//! Podcast-specific metadata support.
//!
//! Provides types and parsing for podcast RSS feed metadata, including
//! Apple Podcasts (iTunes) tags, Google Podcasts extensions, and
//! the Podcast Namespace 2.0 standard.
//!
//! # iTunes Podcast Tags
//!
//! - `itunes:author` - The podcast author
//! - `itunes:category` - One or more categories
//! - `itunes:duration` - Episode duration (HH:MM:SS or seconds)
//! - `itunes:explicit` - Whether the episode contains explicit content
//! - `itunes:summary` - Episode or show summary
//! - `itunes:subtitle` - Short description
//! - `itunes:keywords` - Comma-separated keywords
//! - `itunes:image` - Artwork URL
//! - `itunes:owner` - Podcast owner (name and email)
//! - `itunes:type` - Show type (episodic or serial)
//! - `itunes:episode` - Episode number
//! - `itunes:season` - Season number
//! - `itunes:episodeType` - Episode type (full, trailer, bonus)
//!
//! # RSS Field Mapping
//!
//! Standard RSS 2.0 fields are mapped to podcast metadata:
//! - `title` -> `PodcastMetadata.title`
//! - `description` -> `PodcastMetadata.description`
//! - `link` -> `PodcastMetadata.link`
//! - `pubDate` -> `PodcastMetadata.pub_date`
//! - `enclosure` -> `PodcastEpisode.enclosure_url`, `enclosure_type`, `enclosure_length`

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::collections::HashMap;

// ---- Duration Parsing ----

/// Parse a podcast duration string into total seconds.
///
/// Accepts formats:
/// - `HH:MM:SS` (e.g., "1:23:45")
/// - `MM:SS` (e.g., "45:30")
/// - Bare seconds (e.g., "3600")
pub fn parse_duration(duration: &str) -> Result<u64, Error> {
    let trimmed = duration.trim();
    if trimmed.is_empty() {
        return Err(Error::ParseError("Empty duration string".to_string()));
    }

    let parts: Vec<&str> = trimmed.split(':').collect();
    match parts.len() {
        1 => {
            // Bare seconds
            parts[0]
                .parse::<u64>()
                .map_err(|e| Error::ParseError(format!("Invalid duration seconds: {e}")))
        }
        2 => {
            // MM:SS
            let minutes = parts[0]
                .parse::<u64>()
                .map_err(|e| Error::ParseError(format!("Invalid duration minutes: {e}")))?;
            let seconds = parts[1]
                .parse::<u64>()
                .map_err(|e| Error::ParseError(format!("Invalid duration seconds: {e}")))?;
            Ok(minutes * 60 + seconds)
        }
        3 => {
            // HH:MM:SS
            let hours = parts[0]
                .parse::<u64>()
                .map_err(|e| Error::ParseError(format!("Invalid duration hours: {e}")))?;
            let minutes = parts[1]
                .parse::<u64>()
                .map_err(|e| Error::ParseError(format!("Invalid duration minutes: {e}")))?;
            let seconds = parts[2]
                .parse::<u64>()
                .map_err(|e| Error::ParseError(format!("Invalid duration seconds: {e}")))?;
            Ok(hours * 3600 + minutes * 60 + seconds)
        }
        _ => Err(Error::ParseError(format!(
            "Invalid duration format: {trimmed}"
        ))),
    }
}

/// Format total seconds as HH:MM:SS.
pub fn format_duration(total_seconds: u64) -> String {
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

// ---- Explicit Content ----

/// Explicit content indicator per Apple Podcasts specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplicitFlag {
    /// No explicit content.
    No,
    /// Contains explicit content.
    Yes,
    /// Content has been cleaned of explicit material.
    Clean,
}

impl ExplicitFlag {
    /// Parse from RSS/iTunes string values.
    pub fn from_str_value(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "yes" | "true" | "1" | "explicit" => Self::Yes,
            "clean" => Self::Clean,
            _ => Self::No,
        }
    }

    /// Serialize to the standard string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::No => "no",
            Self::Yes => "yes",
            Self::Clean => "clean",
        }
    }
}

// ---- Show Type ----

/// Podcast show type (Apple Podcasts `itunes:type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShowType {
    /// Episodic: episodes can be listened to in any order (default).
    Episodic,
    /// Serial: episodes should be listened to in order.
    Serial,
}

impl ShowType {
    /// Parse from string.
    pub fn from_str_value(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "serial" => Self::Serial,
            _ => Self::Episodic,
        }
    }

    /// Serialize to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Serial => "serial",
        }
    }
}

// ---- Episode Type ----

/// Podcast episode type (Apple Podcasts `itunes:episodeType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeType {
    /// A full episode.
    Full,
    /// A trailer or preview.
    Trailer,
    /// Bonus content.
    Bonus,
}

impl EpisodeType {
    /// Parse from string.
    pub fn from_str_value(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "trailer" => Self::Trailer,
            "bonus" => Self::Bonus,
            _ => Self::Full,
        }
    }

    /// Serialize to string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Trailer => "trailer",
            Self::Bonus => "bonus",
        }
    }
}

// ---- Podcast Owner ----

/// Podcast owner information (`itunes:owner`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodcastOwner {
    /// Owner name (`itunes:name`).
    pub name: String,
    /// Owner email (`itunes:email`).
    pub email: String,
}

// ---- Podcast Category ----

/// A podcast category, optionally with a subcategory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodcastCategory {
    /// Primary category name (e.g., "Technology").
    pub category: String,
    /// Optional subcategory (e.g., "Software How-To").
    pub subcategory: Option<String>,
}

impl PodcastCategory {
    /// Create a primary category.
    pub fn new(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            subcategory: None,
        }
    }

    /// Create a category with a subcategory.
    pub fn with_subcategory(category: impl Into<String>, subcategory: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            subcategory: Some(subcategory.into()),
        }
    }
}

// ---- Podcast Episode ----

/// Metadata for a single podcast episode.
#[derive(Debug, Clone)]
pub struct PodcastEpisode {
    /// Episode title (RSS `title`).
    pub title: String,
    /// Episode description (RSS `description`).
    pub description: Option<String>,
    /// Episode publication date (RSS `pubDate`).
    pub pub_date: Option<String>,
    /// Duration in seconds.
    pub duration_seconds: Option<u64>,
    /// Explicit content flag.
    pub explicit: ExplicitFlag,
    /// Episode type.
    pub episode_type: EpisodeType,
    /// Season number (`itunes:season`).
    pub season: Option<u32>,
    /// Episode number (`itunes:episode`).
    pub episode_number: Option<u32>,
    /// Episode subtitle (`itunes:subtitle`).
    pub subtitle: Option<String>,
    /// Episode summary (`itunes:summary`).
    pub summary: Option<String>,
    /// Episode-specific image URL (`itunes:image`).
    pub image_url: Option<String>,
    /// Enclosure URL (audio/video file URL).
    pub enclosure_url: Option<String>,
    /// Enclosure MIME type (e.g., "audio/mpeg").
    pub enclosure_type: Option<String>,
    /// Enclosure file size in bytes.
    pub enclosure_length: Option<u64>,
    /// GUID (globally unique identifier).
    pub guid: Option<String>,
    /// Episode link URL.
    pub link: Option<String>,
    /// Additional custom tags.
    pub custom: HashMap<String, String>,
}

impl PodcastEpisode {
    /// Create a new episode with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: None,
            pub_date: None,
            duration_seconds: None,
            explicit: ExplicitFlag::No,
            episode_type: EpisodeType::Full,
            season: None,
            episode_number: None,
            subtitle: None,
            summary: None,
            image_url: None,
            enclosure_url: None,
            enclosure_type: None,
            enclosure_length: None,
            guid: None,
            link: None,
            custom: HashMap::new(),
        }
    }

    /// Set the enclosure (audio/video file).
    pub fn with_enclosure(
        mut self,
        url: impl Into<String>,
        mime_type: impl Into<String>,
        length: u64,
    ) -> Self {
        self.enclosure_url = Some(url.into());
        self.enclosure_type = Some(mime_type.into());
        self.enclosure_length = Some(length);
        self
    }

    /// Get the formatted duration string.
    pub fn duration_string(&self) -> Option<String> {
        self.duration_seconds.map(format_duration)
    }
}

// ---- Podcast Metadata (Show-level) ----

/// Complete podcast (show-level) metadata.
#[derive(Debug, Clone)]
pub struct PodcastMetadata {
    /// Podcast title (RSS `title`).
    pub title: String,
    /// Podcast description (RSS `description`).
    pub description: Option<String>,
    /// Podcast author (`itunes:author`).
    pub author: Option<String>,
    /// Podcast link URL (RSS `link`).
    pub link: Option<String>,
    /// Language code (RSS `language`, e.g., "en-us").
    pub language: Option<String>,
    /// Copyright notice (RSS `copyright`).
    pub copyright: Option<String>,
    /// Podcast categories.
    pub categories: Vec<PodcastCategory>,
    /// Explicit content flag.
    pub explicit: ExplicitFlag,
    /// Show type (episodic or serial).
    pub show_type: ShowType,
    /// Artwork image URL (`itunes:image`).
    pub image_url: Option<String>,
    /// Podcast owner.
    pub owner: Option<PodcastOwner>,
    /// Whether the podcast is complete (no new episodes expected).
    pub complete: bool,
    /// Whether the podcast should be blocked from directories.
    pub block: bool,
    /// Keywords (comma-separated in RSS, stored as Vec here).
    pub keywords: Vec<String>,
    /// Episodes.
    pub episodes: Vec<PodcastEpisode>,
    /// Additional custom fields.
    pub custom: HashMap<String, String>,
}

impl PodcastMetadata {
    /// Create new podcast metadata with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: None,
            author: None,
            link: None,
            language: None,
            copyright: None,
            categories: Vec::new(),
            explicit: ExplicitFlag::No,
            show_type: ShowType::Episodic,
            image_url: None,
            owner: None,
            complete: false,
            block: false,
            keywords: Vec::new(),
            episodes: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Add a category.
    pub fn add_category(&mut self, category: PodcastCategory) {
        self.categories.push(category);
    }

    /// Add an episode.
    pub fn add_episode(&mut self, episode: PodcastEpisode) {
        self.episodes.push(episode);
    }

    /// Return the number of episodes.
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Total duration of all episodes (sum of known durations).
    pub fn total_duration_seconds(&self) -> u64 {
        self.episodes
            .iter()
            .filter_map(|e| e.duration_seconds)
            .sum()
    }
}

// ---- RSS Field Mapping ----

/// Standard RSS 2.0 field names used in podcast feeds.
pub mod rss_fields {
    /// RSS channel title.
    pub const TITLE: &str = "title";
    /// RSS channel description.
    pub const DESCRIPTION: &str = "description";
    /// RSS channel link.
    pub const LINK: &str = "link";
    /// RSS channel language.
    pub const LANGUAGE: &str = "language";
    /// RSS channel copyright.
    pub const COPYRIGHT: &str = "copyright";
    /// RSS item publication date.
    pub const PUB_DATE: &str = "pubDate";
    /// RSS item GUID.
    pub const GUID: &str = "guid";
    /// RSS item enclosure.
    pub const ENCLOSURE: &str = "enclosure";
}

/// iTunes namespace tag names used in podcast feeds.
pub mod itunes_tags {
    /// Author of the podcast.
    pub const AUTHOR: &str = "itunes:author";
    /// Category.
    pub const CATEGORY: &str = "itunes:category";
    /// Episode or show duration.
    pub const DURATION: &str = "itunes:duration";
    /// Explicit content flag.
    pub const EXPLICIT: &str = "itunes:explicit";
    /// Summary text.
    pub const SUMMARY: &str = "itunes:summary";
    /// Short subtitle.
    pub const SUBTITLE: &str = "itunes:subtitle";
    /// Keywords (comma-separated, deprecated but still used).
    pub const KEYWORDS: &str = "itunes:keywords";
    /// Image URL.
    pub const IMAGE: &str = "itunes:image";
    /// Podcast owner.
    pub const OWNER: &str = "itunes:owner";
    /// Show type.
    pub const TYPE: &str = "itunes:type";
    /// Episode number.
    pub const EPISODE: &str = "itunes:episode";
    /// Season number.
    pub const SEASON: &str = "itunes:season";
    /// Episode type.
    pub const EPISODE_TYPE: &str = "itunes:episodeType";
    /// Block from directories.
    pub const BLOCK: &str = "itunes:block";
    /// Podcast is complete.
    pub const COMPLETE: &str = "itunes:complete";
    /// Owner name (inside itunes:owner).
    pub const OWNER_NAME: &str = "itunes:name";
    /// Owner email (inside itunes:owner).
    pub const OWNER_EMAIL: &str = "itunes:email";
}

// ---- Conversion to/from Metadata ----

/// Convert `PodcastMetadata` to a generic `Metadata` container.
///
/// Uses `MetadataFormat::iTunes` and stores podcast fields under
/// their standard RSS / iTunes tag names.
pub fn to_metadata(podcast: &PodcastMetadata) -> Metadata {
    let mut metadata = Metadata::new(MetadataFormat::iTunes);

    metadata.insert(
        rss_fields::TITLE.to_string(),
        MetadataValue::Text(podcast.title.clone()),
    );

    if let Some(ref desc) = podcast.description {
        metadata.insert(
            rss_fields::DESCRIPTION.to_string(),
            MetadataValue::Text(desc.clone()),
        );
    }

    if let Some(ref author) = podcast.author {
        metadata.insert(
            itunes_tags::AUTHOR.to_string(),
            MetadataValue::Text(author.clone()),
        );
    }

    if let Some(ref link) = podcast.link {
        metadata.insert(
            rss_fields::LINK.to_string(),
            MetadataValue::Text(link.clone()),
        );
    }

    if let Some(ref lang) = podcast.language {
        metadata.insert(
            rss_fields::LANGUAGE.to_string(),
            MetadataValue::Text(lang.clone()),
        );
    }

    metadata.insert(
        itunes_tags::EXPLICIT.to_string(),
        MetadataValue::Text(podcast.explicit.as_str().to_string()),
    );

    metadata.insert(
        itunes_tags::TYPE.to_string(),
        MetadataValue::Text(podcast.show_type.as_str().to_string()),
    );

    if let Some(ref img) = podcast.image_url {
        metadata.insert(
            itunes_tags::IMAGE.to_string(),
            MetadataValue::Text(img.clone()),
        );
    }

    if !podcast.categories.is_empty() {
        let cat_strings: Vec<String> = podcast
            .categories
            .iter()
            .map(|c| {
                if let Some(ref sub) = c.subcategory {
                    format!("{} > {sub}", c.category)
                } else {
                    c.category.clone()
                }
            })
            .collect();
        metadata.insert(
            itunes_tags::CATEGORY.to_string(),
            MetadataValue::TextList(cat_strings),
        );
    }

    if !podcast.keywords.is_empty() {
        metadata.insert(
            itunes_tags::KEYWORDS.to_string(),
            MetadataValue::Text(podcast.keywords.join(",")),
        );
    }

    metadata.insert(
        "episode_count".to_string(),
        MetadataValue::Integer(podcast.episodes.len() as i64),
    );

    metadata
}

/// Extract `PodcastMetadata` from a generic `Metadata` container.
pub fn from_metadata(metadata: &Metadata) -> PodcastMetadata {
    let title = metadata
        .get(rss_fields::TITLE)
        .and_then(|v| v.as_text())
        .unwrap_or("Untitled Podcast")
        .to_string();

    let mut podcast = PodcastMetadata::new(title);

    podcast.description = metadata
        .get(rss_fields::DESCRIPTION)
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    podcast.author = metadata
        .get(itunes_tags::AUTHOR)
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    podcast.link = metadata
        .get(rss_fields::LINK)
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    podcast.language = metadata
        .get(rss_fields::LANGUAGE)
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    if let Some(explicit_str) = metadata
        .get(itunes_tags::EXPLICIT)
        .and_then(|v| v.as_text())
    {
        podcast.explicit = ExplicitFlag::from_str_value(explicit_str);
    }

    if let Some(show_type_str) = metadata.get(itunes_tags::TYPE).and_then(|v| v.as_text()) {
        podcast.show_type = ShowType::from_str_value(show_type_str);
    }

    podcast.image_url = metadata
        .get(itunes_tags::IMAGE)
        .and_then(|v| v.as_text())
        .map(|s| s.to_string());

    if let Some(kw) = metadata
        .get(itunes_tags::KEYWORDS)
        .and_then(|v| v.as_text())
    {
        podcast.keywords = kw.split(',').map(|s| s.trim().to_string()).collect();
    }

    // Parse categories
    if let Some(cats) = metadata.get(itunes_tags::CATEGORY) {
        let cat_strings: Vec<&str> = match cats {
            MetadataValue::TextList(list) => list.iter().map(|s| s.as_str()).collect(),
            MetadataValue::Text(s) => vec![s.as_str()],
            _ => Vec::new(),
        };
        for cat_str in cat_strings {
            if let Some((main, sub)) = cat_str.split_once(" > ") {
                podcast.add_category(PodcastCategory::with_subcategory(main, sub));
            } else {
                podcast.add_category(PodcastCategory::new(cat_str));
            }
        }
    }

    podcast
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_hhmmss() {
        assert_eq!(parse_duration("1:23:45").expect("valid"), 5025);
        assert_eq!(parse_duration("0:00:00").expect("valid"), 0);
        assert_eq!(parse_duration("2:00:00").expect("valid"), 7200);
    }

    #[test]
    fn test_parse_duration_mmss() {
        assert_eq!(parse_duration("45:30").expect("valid"), 2730);
        assert_eq!(parse_duration("0:30").expect("valid"), 30);
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("3600").expect("valid"), 3600);
        assert_eq!(parse_duration("0").expect("valid"), 0);
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration("").is_err());
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("1:2:3:4").is_err());
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(5025), "1:23:45");
        assert_eq!(format_duration(0), "0:00");
        assert_eq!(format_duration(90), "1:30");
        assert_eq!(format_duration(7200), "2:00:00");
    }

    #[test]
    fn test_explicit_flag() {
        assert_eq!(ExplicitFlag::from_str_value("yes"), ExplicitFlag::Yes);
        assert_eq!(ExplicitFlag::from_str_value("true"), ExplicitFlag::Yes);
        assert_eq!(ExplicitFlag::from_str_value("1"), ExplicitFlag::Yes);
        assert_eq!(ExplicitFlag::from_str_value("explicit"), ExplicitFlag::Yes);
        assert_eq!(ExplicitFlag::from_str_value("clean"), ExplicitFlag::Clean);
        assert_eq!(ExplicitFlag::from_str_value("no"), ExplicitFlag::No);
        assert_eq!(ExplicitFlag::from_str_value("false"), ExplicitFlag::No);
        assert_eq!(ExplicitFlag::from_str_value(""), ExplicitFlag::No);

        assert_eq!(ExplicitFlag::Yes.as_str(), "yes");
        assert_eq!(ExplicitFlag::No.as_str(), "no");
        assert_eq!(ExplicitFlag::Clean.as_str(), "clean");
    }

    #[test]
    fn test_show_type() {
        assert_eq!(ShowType::from_str_value("serial"), ShowType::Serial);
        assert_eq!(ShowType::from_str_value("episodic"), ShowType::Episodic);
        assert_eq!(ShowType::from_str_value("unknown"), ShowType::Episodic);
        assert_eq!(ShowType::Serial.as_str(), "serial");
        assert_eq!(ShowType::Episodic.as_str(), "episodic");
    }

    #[test]
    fn test_episode_type() {
        assert_eq!(EpisodeType::from_str_value("full"), EpisodeType::Full);
        assert_eq!(EpisodeType::from_str_value("trailer"), EpisodeType::Trailer);
        assert_eq!(EpisodeType::from_str_value("bonus"), EpisodeType::Bonus);
        assert_eq!(EpisodeType::from_str_value("other"), EpisodeType::Full);
        assert_eq!(EpisodeType::Full.as_str(), "full");
        assert_eq!(EpisodeType::Trailer.as_str(), "trailer");
        assert_eq!(EpisodeType::Bonus.as_str(), "bonus");
    }

    #[test]
    fn test_podcast_category() {
        let cat = PodcastCategory::new("Technology");
        assert_eq!(cat.category, "Technology");
        assert_eq!(cat.subcategory, None);

        let cat2 = PodcastCategory::with_subcategory("Technology", "Software How-To");
        assert_eq!(cat2.category, "Technology");
        assert_eq!(cat2.subcategory.as_deref(), Some("Software How-To"));
    }

    #[test]
    fn test_podcast_episode() {
        let ep = PodcastEpisode::new("Episode 1").with_enclosure(
            "https://example.com/ep1.mp3",
            "audio/mpeg",
            12345678,
        );

        assert_eq!(ep.title, "Episode 1");
        assert_eq!(
            ep.enclosure_url.as_deref(),
            Some("https://example.com/ep1.mp3")
        );
        assert_eq!(ep.enclosure_type.as_deref(), Some("audio/mpeg"));
        assert_eq!(ep.enclosure_length, Some(12345678));
        assert_eq!(ep.explicit, ExplicitFlag::No);
        assert_eq!(ep.episode_type, EpisodeType::Full);
    }

    #[test]
    fn test_podcast_episode_duration_string() {
        let mut ep = PodcastEpisode::new("Ep");
        assert_eq!(ep.duration_string(), None);
        ep.duration_seconds = Some(3661);
        assert_eq!(ep.duration_string(), Some("1:01:01".to_string()));
    }

    #[test]
    fn test_podcast_metadata_basic() {
        let mut podcast = PodcastMetadata::new("My Podcast");
        podcast.author = Some("Alice".to_string());
        podcast.description = Some("A great show".to_string());
        podcast.language = Some("en-us".to_string());
        podcast.explicit = ExplicitFlag::No;
        podcast.show_type = ShowType::Episodic;

        podcast.add_category(PodcastCategory::new("Technology"));
        podcast.add_category(PodcastCategory::with_subcategory("Arts", "Design"));

        assert_eq!(podcast.title, "My Podcast");
        assert_eq!(podcast.categories.len(), 2);
        assert_eq!(podcast.episode_count(), 0);
    }

    #[test]
    fn test_podcast_metadata_episodes() {
        let mut podcast = PodcastMetadata::new("Show");

        let mut ep1 = PodcastEpisode::new("Episode 1");
        ep1.duration_seconds = Some(1800);

        let mut ep2 = PodcastEpisode::new("Episode 2");
        ep2.duration_seconds = Some(2400);

        podcast.add_episode(ep1);
        podcast.add_episode(ep2);

        assert_eq!(podcast.episode_count(), 2);
        assert_eq!(podcast.total_duration_seconds(), 4200);
    }

    #[test]
    fn test_podcast_to_metadata_round_trip() {
        let mut podcast = PodcastMetadata::new("Test Podcast");
        podcast.author = Some("Bob".to_string());
        podcast.description = Some("Description here".to_string());
        podcast.link = Some("https://example.com".to_string());
        podcast.language = Some("en".to_string());
        podcast.explicit = ExplicitFlag::Yes;
        podcast.show_type = ShowType::Serial;
        podcast.image_url = Some("https://example.com/art.jpg".to_string());
        podcast.keywords = vec!["tech".to_string(), "news".to_string()];
        podcast.add_category(PodcastCategory::new("Technology"));
        podcast.add_category(PodcastCategory::with_subcategory(
            "Society & Culture",
            "Philosophy",
        ));

        let metadata = to_metadata(&podcast);

        // Verify fields
        assert_eq!(
            metadata.get(rss_fields::TITLE).and_then(|v| v.as_text()),
            Some("Test Podcast")
        );
        assert_eq!(
            metadata.get(itunes_tags::AUTHOR).and_then(|v| v.as_text()),
            Some("Bob")
        );
        assert_eq!(
            metadata
                .get(itunes_tags::EXPLICIT)
                .and_then(|v| v.as_text()),
            Some("yes")
        );
        assert_eq!(
            metadata.get(itunes_tags::TYPE).and_then(|v| v.as_text()),
            Some("serial")
        );

        // Round-trip back
        let podcast2 = from_metadata(&metadata);
        assert_eq!(podcast2.title, "Test Podcast");
        assert_eq!(podcast2.author.as_deref(), Some("Bob"));
        assert_eq!(podcast2.explicit, ExplicitFlag::Yes);
        assert_eq!(podcast2.show_type, ShowType::Serial);
        assert_eq!(podcast2.categories.len(), 2);
        assert_eq!(podcast2.categories[0].category, "Technology");
        assert_eq!(podcast2.keywords.len(), 2);
    }

    #[test]
    fn test_from_metadata_defaults() {
        let metadata = Metadata::new(MetadataFormat::iTunes);
        let podcast = from_metadata(&metadata);
        assert_eq!(podcast.title, "Untitled Podcast");
        assert_eq!(podcast.explicit, ExplicitFlag::No);
        assert_eq!(podcast.show_type, ShowType::Episodic);
    }

    #[test]
    fn test_podcast_owner() {
        let owner = PodcastOwner {
            name: "Alice".to_string(),
            email: "alice@example.com".to_string(),
        };
        assert_eq!(owner.name, "Alice");
        assert_eq!(owner.email, "alice@example.com");
    }

    #[test]
    fn test_podcast_episode_custom_fields() {
        let mut ep = PodcastEpisode::new("Special Episode");
        ep.custom.insert(
            "podcast:transcript".to_string(),
            "https://example.com/transcript.srt".to_string(),
        );
        assert_eq!(
            ep.custom.get("podcast:transcript").map(|s| s.as_str()),
            Some("https://example.com/transcript.srt")
        );
    }

    #[test]
    fn test_podcast_metadata_custom_fields() {
        let mut podcast = PodcastMetadata::new("Show");
        podcast
            .custom
            .insert("podcast:locked".to_string(), "yes".to_string());
        assert_eq!(
            podcast.custom.get("podcast:locked").map(|s| s.as_str()),
            Some("yes")
        );
    }
}
