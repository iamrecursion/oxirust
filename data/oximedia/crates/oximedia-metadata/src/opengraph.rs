//! Open Graph protocol and Twitter Card metadata support.
//!
//! Generates and parses `<meta property="og:...">` HTML tags for rich link previews
//! on social media platforms. Also supports Twitter Card meta tags.
//!
//! # Open Graph Protocol
//!
//! The Open Graph protocol enables any web page to become a rich object in a social graph.
//! Facebook, LinkedIn, and many other platforms use these tags for link previews.
//!
//! # Twitter Cards
//!
//! Twitter Cards allow you to attach rich photos, videos and media experiences to Tweets.
//! They use `<meta name="twitter:...">` tags.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::opengraph::{OpenGraph, OgType, to_html_meta_tags, from_html_meta};
//!
//! let og = OpenGraph {
//!     og_type: OgType::Video,
//!     title: "My Video".to_string(),
//!     url: Some("https://example.com/video/1".to_string()),
//!     image: Some("https://example.com/thumb.jpg".to_string()),
//!     video: Some("https://example.com/video.mp4".to_string()),
//!     description: Some("A great video".to_string()),
//!     site_name: Some("Example Site".to_string()),
//!     locale: None,
//!     video_width: None,
//!     video_height: None,
//!     video_type: None,
//!     image_width: None,
//!     image_height: None,
//!     image_type: None,
//!     audio: None,
//!     determiner: None,
//! };
//!
//! let html = to_html_meta_tags(&og);
//! assert!(html.contains("og:type"));
//! ```

#![allow(dead_code)]

// ────────────────────────────────────────────────────────────────────────────
// Open Graph types
// ────────────────────────────────────────────────────────────────────────────

/// Open Graph object type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OgType {
    /// og:type = "video"
    Video,
    /// og:type = "video.movie"
    VideoMovie,
    /// og:type = "video.episode"
    VideoEpisode,
    /// og:type = "video.tv_show"
    VideoTvShow,
    /// og:type = "video.other"
    VideoOther,
    /// og:type = "music.song"
    MusicSong,
    /// og:type = "music.album"
    MusicAlbum,
    /// og:type = "music.playlist"
    MusicPlaylist,
    /// og:type = "article"
    Article,
    /// og:type = "website"
    Website,
    /// og:type = "profile"
    Profile,
    /// og:type = "book"
    Book,
}

impl OgType {
    /// Return the Open Graph type string value.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::VideoMovie => "video.movie",
            Self::VideoEpisode => "video.episode",
            Self::VideoTvShow => "video.tv_show",
            Self::VideoOther => "video.other",
            Self::MusicSong => "music.song",
            Self::MusicAlbum => "music.album",
            Self::MusicPlaylist => "music.playlist",
            Self::Article => "article",
            Self::Website => "website",
            Self::Profile => "profile",
            Self::Book => "book",
        }
    }

    /// Parse an OgType from a string value.
    #[must_use]
    pub fn from_str_value(s: &str) -> Option<Self> {
        match s {
            "video" => Some(Self::Video),
            "video.movie" => Some(Self::VideoMovie),
            "video.episode" => Some(Self::VideoEpisode),
            "video.tv_show" => Some(Self::VideoTvShow),
            "video.other" => Some(Self::VideoOther),
            "music.song" => Some(Self::MusicSong),
            "music.album" => Some(Self::MusicAlbum),
            "music.playlist" => Some(Self::MusicPlaylist),
            "article" => Some(Self::Article),
            "website" => Some(Self::Website),
            "profile" => Some(Self::Profile),
            "book" => Some(Self::Book),
            _ => None,
        }
    }
}

impl std::fmt::Display for OgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// OpenGraph struct
// ────────────────────────────────────────────────────────────────────────────

/// Open Graph metadata for a web page or media object.
///
/// Represents the essential Open Graph Protocol properties plus common
/// media-related extensions (video dimensions, audio URL, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct OpenGraph {
    /// The type of object (og:type).
    pub og_type: OgType,
    /// Title of the object (og:title). Required.
    pub title: String,
    /// Canonical URL of the object (og:url).
    pub url: Option<String>,
    /// URL to an image representing the object (og:image).
    pub image: Option<String>,
    /// URL to a video for this object (og:video).
    pub video: Option<String>,
    /// A one- to two-sentence description (og:description).
    pub description: Option<String>,
    /// Site name (og:site_name).
    pub site_name: Option<String>,
    /// Locale in language_TERRITORY format (og:locale).
    pub locale: Option<String>,
    /// Video width in pixels (og:video:width).
    pub video_width: Option<u32>,
    /// Video height in pixels (og:video:height).
    pub video_height: Option<u32>,
    /// Video MIME type (og:video:type).
    pub video_type: Option<String>,
    /// Image width in pixels (og:image:width).
    pub image_width: Option<u32>,
    /// Image height in pixels (og:image:height).
    pub image_height: Option<u32>,
    /// Image MIME type (og:image:type).
    pub image_type: Option<String>,
    /// Audio URL (og:audio).
    pub audio: Option<String>,
    /// Determiner (og:determiner): "a", "an", "the", "", "auto".
    pub determiner: Option<String>,
}

impl OpenGraph {
    /// Create a new OpenGraph with required fields only.
    #[must_use]
    pub fn new(og_type: OgType, title: String) -> Self {
        Self {
            og_type,
            title,
            url: None,
            image: None,
            video: None,
            description: None,
            site_name: None,
            locale: None,
            video_width: None,
            video_height: None,
            video_type: None,
            image_width: None,
            image_height: None,
            image_type: None,
            audio: None,
            determiner: None,
        }
    }

    /// Builder: set the URL.
    #[must_use]
    pub fn with_url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    /// Builder: set the image URL.
    #[must_use]
    pub fn with_image(mut self, image: String) -> Self {
        self.image = Some(image);
        self
    }

    /// Builder: set the video URL.
    #[must_use]
    pub fn with_video(mut self, video: String) -> Self {
        self.video = Some(video);
        self
    }

    /// Builder: set the description.
    #[must_use]
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Builder: set the site name.
    #[must_use]
    pub fn with_site_name(mut self, name: String) -> Self {
        self.site_name = Some(name);
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// HTML meta tag generation
// ────────────────────────────────────────────────────────────────────────────

/// Escape a string for use in an HTML attribute value.
fn html_attr_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c => out.push(c),
        }
    }
    out
}

/// Append a single `<meta property="..." content="..."/>` tag to the output buffer.
fn append_og_tag(out: &mut String, property: &str, content: &str) {
    out.push_str("<meta property=\"");
    out.push_str(property);
    out.push_str("\" content=\"");
    out.push_str(&html_attr_escape(content));
    out.push_str("\"/>\n");
}

/// Generate HTML `<meta>` tags from an [`OpenGraph`] object.
///
/// Each property becomes a separate `<meta property="og:..." content="..."/>` tag.
/// Tags are separated by newlines.
#[must_use]
pub fn to_html_meta_tags(og: &OpenGraph) -> String {
    let mut out = String::new();
    append_og_tag(&mut out, "og:type", og.og_type.as_str());
    append_og_tag(&mut out, "og:title", &og.title);

    if let Some(ref url) = og.url {
        append_og_tag(&mut out, "og:url", url);
    }
    if let Some(ref image) = og.image {
        append_og_tag(&mut out, "og:image", image);
    }
    if let Some(ref w) = og.image_width {
        append_og_tag(&mut out, "og:image:width", &w.to_string());
    }
    if let Some(ref h) = og.image_height {
        append_og_tag(&mut out, "og:image:height", &h.to_string());
    }
    if let Some(ref t) = og.image_type {
        append_og_tag(&mut out, "og:image:type", t);
    }
    if let Some(ref video) = og.video {
        append_og_tag(&mut out, "og:video", video);
    }
    if let Some(ref w) = og.video_width {
        append_og_tag(&mut out, "og:video:width", &w.to_string());
    }
    if let Some(ref h) = og.video_height {
        append_og_tag(&mut out, "og:video:height", &h.to_string());
    }
    if let Some(ref t) = og.video_type {
        append_og_tag(&mut out, "og:video:type", t);
    }
    if let Some(ref desc) = og.description {
        append_og_tag(&mut out, "og:description", desc);
    }
    if let Some(ref name) = og.site_name {
        append_og_tag(&mut out, "og:site_name", name);
    }
    if let Some(ref locale) = og.locale {
        append_og_tag(&mut out, "og:locale", locale);
    }
    if let Some(ref audio) = og.audio {
        append_og_tag(&mut out, "og:audio", audio);
    }
    if let Some(ref det) = og.determiner {
        append_og_tag(&mut out, "og:determiner", det);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// HTML meta tag parsing
// ────────────────────────────────────────────────────────────────────────────

/// Extract an attribute value from a tag string.
///
/// Searches for `attr_name="value"` or `attr_name='value'` and returns the
/// unescaped content between the quotes.
fn extract_attr(tag: &str, attr_name: &str) -> Option<String> {
    // Try double quotes first
    let needle_dq = format!("{attr_name}=\"");
    if let Some(start) = tag.find(&needle_dq) {
        let after = start + needle_dq.len();
        if let Some(end) = tag[after..].find('"') {
            let raw = &tag[after..after + end];
            return Some(html_unescape(raw));
        }
    }
    // Try single quotes
    let needle_sq = format!("{attr_name}='");
    if let Some(start) = tag.find(&needle_sq) {
        let after = start + needle_sq.len();
        if let Some(end) = tag[after..].find('\'') {
            let raw = &tag[after..after + end];
            return Some(html_unescape(raw));
        }
    }
    None
}

/// Basic HTML entity unescaping.
fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

/// Parse Open Graph metadata from an HTML document string.
///
/// Scans for `<meta property="og:..." content="..."/>` tags in the HTML.
/// Unknown og: properties are silently ignored.
#[must_use]
pub fn from_html_meta(html: &str) -> OpenGraph {
    let mut og = OpenGraph::new(OgType::Website, String::new());

    // Use a case-insensitive scan for <meta tags
    let lower = html.to_lowercase();
    let mut search_pos = 0;

    while let Some(tag_start) = lower[search_pos..].find("<meta") {
        let abs_start = search_pos + tag_start;
        // Find the end of this tag
        let tag_end = match lower[abs_start..].find('>') {
            Some(e) => abs_start + e + 1,
            None => break,
        };
        let tag_lower = &lower[abs_start..tag_end];
        let tag_orig = &html[abs_start..tag_end];

        // Check for og: property
        if tag_lower.contains("property=") && tag_lower.contains("og:") {
            // Try extracting from original case first (preserves content case)
            if let (Some(prop), Some(content)) = (
                extract_attr(tag_orig, "property"),
                extract_attr(tag_orig, "content"),
            ) {
                apply_og_property(&mut og, &prop, &content);
            } else if let (Some(prop), Some(_content)) = (
                extract_attr(tag_lower, "property"),
                extract_attr(tag_lower, "content"),
            ) {
                // Fallback: use lowercase property name but extract content from original
                // to preserve casing of content values
                if let Some(content_orig) = extract_attr(tag_orig, "content") {
                    apply_og_property(&mut og, &prop, &content_orig);
                }
            }
        }

        search_pos = tag_end;
    }

    og
}

/// Apply a single og: property value to an OpenGraph struct.
fn apply_og_property(og: &mut OpenGraph, property: &str, value: &str) {
    match property {
        "og:type" => {
            if let Some(t) = OgType::from_str_value(value) {
                og.og_type = t;
            }
        }
        "og:title" => {
            og.title = value.to_string();
        }
        "og:url" => {
            og.url = Some(value.to_string());
        }
        "og:image" => {
            og.image = Some(value.to_string());
        }
        "og:image:width" => {
            og.image_width = value.parse().ok();
        }
        "og:image:height" => {
            og.image_height = value.parse().ok();
        }
        "og:image:type" => {
            og.image_type = Some(value.to_string());
        }
        "og:video" => {
            og.video = Some(value.to_string());
        }
        "og:video:width" => {
            og.video_width = value.parse().ok();
        }
        "og:video:height" => {
            og.video_height = value.parse().ok();
        }
        "og:video:type" => {
            og.video_type = Some(value.to_string());
        }
        "og:description" => {
            og.description = Some(value.to_string());
        }
        "og:site_name" => {
            og.site_name = Some(value.to_string());
        }
        "og:locale" => {
            og.locale = Some(value.to_string());
        }
        "og:audio" => {
            og.audio = Some(value.to_string());
        }
        "og:determiner" => {
            og.determiner = Some(value.to_string());
        }
        _ => {}
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Twitter Card
// ────────────────────────────────────────────────────────────────────────────

/// Twitter Card type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TwitterCardType {
    /// summary — default card with small thumbnail.
    Summary,
    /// summary_large_image — card with large image.
    SummaryLargeImage,
    /// player — video/audio player card.
    Player,
    /// app — card for mobile applications.
    App,
}

impl TwitterCardType {
    /// Return the twitter:card value string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::SummaryLargeImage => "summary_large_image",
            Self::Player => "player",
            Self::App => "app",
        }
    }

    /// Parse from a string.
    #[must_use]
    pub fn from_str_value(s: &str) -> Option<Self> {
        match s {
            "summary" => Some(Self::Summary),
            "summary_large_image" => Some(Self::SummaryLargeImage),
            "player" => Some(Self::Player),
            "app" => Some(Self::App),
            _ => None,
        }
    }
}

impl std::fmt::Display for TwitterCardType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Twitter Card metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TwitterCard {
    /// Card type (twitter:card).
    pub card_type: TwitterCardType,
    /// Title (twitter:title).
    pub title: String,
    /// Description (twitter:description).
    pub description: Option<String>,
    /// Image URL (twitter:image).
    pub image: Option<String>,
    /// Image alt text (twitter:image:alt).
    pub image_alt: Option<String>,
    /// Site @username (twitter:site).
    pub site: Option<String>,
    /// Creator @username (twitter:creator).
    pub creator: Option<String>,
    /// Player URL for player cards (twitter:player).
    pub player: Option<String>,
    /// Player width (twitter:player:width).
    pub player_width: Option<u32>,
    /// Player height (twitter:player:height).
    pub player_height: Option<u32>,
    /// Player stream URL (twitter:player:stream).
    pub player_stream: Option<String>,
}

impl TwitterCard {
    /// Create a new Twitter Card with required fields.
    #[must_use]
    pub fn new(card_type: TwitterCardType, title: String) -> Self {
        Self {
            card_type,
            title,
            description: None,
            image: None,
            image_alt: None,
            site: None,
            creator: None,
            player: None,
            player_width: None,
            player_height: None,
            player_stream: None,
        }
    }

    /// Builder: set the description.
    #[must_use]
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Builder: set the image URL.
    #[must_use]
    pub fn with_image(mut self, image: String) -> Self {
        self.image = Some(image);
        self
    }

    /// Builder: set the site username.
    #[must_use]
    pub fn with_site(mut self, site: String) -> Self {
        self.site = Some(site);
        self
    }

    /// Builder: set the creator username.
    #[must_use]
    pub fn with_creator(mut self, creator: String) -> Self {
        self.creator = Some(creator);
        self
    }
}

/// Append a twitter meta tag to the output buffer.
fn append_twitter_tag(out: &mut String, name: &str, content: &str) {
    out.push_str("<meta name=\"");
    out.push_str(name);
    out.push_str("\" content=\"");
    out.push_str(&html_attr_escape(content));
    out.push_str("\"/>\n");
}

/// Generate HTML `<meta>` tags for a [`TwitterCard`].
///
/// Each property becomes a `<meta name="twitter:..." content="..."/>` tag.
#[must_use]
pub fn to_twitter_meta_tags(card: &TwitterCard) -> String {
    let mut out = String::new();
    append_twitter_tag(&mut out, "twitter:card", card.card_type.as_str());
    append_twitter_tag(&mut out, "twitter:title", &card.title);

    if let Some(ref desc) = card.description {
        append_twitter_tag(&mut out, "twitter:description", desc);
    }
    if let Some(ref image) = card.image {
        append_twitter_tag(&mut out, "twitter:image", image);
    }
    if let Some(ref alt) = card.image_alt {
        append_twitter_tag(&mut out, "twitter:image:alt", alt);
    }
    if let Some(ref site) = card.site {
        append_twitter_tag(&mut out, "twitter:site", site);
    }
    if let Some(ref creator) = card.creator {
        append_twitter_tag(&mut out, "twitter:creator", creator);
    }
    if let Some(ref player) = card.player {
        append_twitter_tag(&mut out, "twitter:player", player);
    }
    if let Some(ref w) = card.player_width {
        append_twitter_tag(&mut out, "twitter:player:width", &w.to_string());
    }
    if let Some(ref h) = card.player_height {
        append_twitter_tag(&mut out, "twitter:player:height", &h.to_string());
    }
    if let Some(ref stream) = card.player_stream {
        append_twitter_tag(&mut out, "twitter:player:stream", stream);
    }
    out
}

/// Convert an [`OpenGraph`] object to a basic [`TwitterCard`].
///
/// Maps og fields to twitter card fields. Uses `SummaryLargeImage` for video
/// types and `Summary` for other types.
#[must_use]
pub fn og_to_twitter_card(og: &OpenGraph) -> TwitterCard {
    let card_type = match og.og_type {
        OgType::Video
        | OgType::VideoMovie
        | OgType::VideoEpisode
        | OgType::VideoTvShow
        | OgType::VideoOther => TwitterCardType::SummaryLargeImage,
        _ => TwitterCardType::Summary,
    };
    TwitterCard {
        card_type,
        title: og.title.clone(),
        description: og.description.clone(),
        image: og.image.clone(),
        image_alt: None,
        site: og.site_name.clone(),
        creator: None,
        player: og.video.clone(),
        player_width: og.video_width,
        player_height: og.video_height,
        player_stream: None,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OgType ──────────────────────────────────────────────────────────

    #[test]
    fn test_og_type_as_str() {
        assert_eq!(OgType::Video.as_str(), "video");
        assert_eq!(OgType::VideoMovie.as_str(), "video.movie");
        assert_eq!(OgType::MusicSong.as_str(), "music.song");
        assert_eq!(OgType::Article.as_str(), "article");
        assert_eq!(OgType::Website.as_str(), "website");
    }

    #[test]
    fn test_og_type_roundtrip() {
        let types = [
            OgType::Video,
            OgType::VideoMovie,
            OgType::VideoEpisode,
            OgType::VideoTvShow,
            OgType::VideoOther,
            OgType::MusicSong,
            OgType::MusicAlbum,
            OgType::MusicPlaylist,
            OgType::Article,
            OgType::Website,
            OgType::Profile,
            OgType::Book,
        ];
        for t in &types {
            let s = t.as_str();
            let parsed =
                OgType::from_str_value(s).unwrap_or_else(|| panic!("failed to parse OgType: {s}"));
            assert_eq!(&parsed, t);
        }
    }

    #[test]
    fn test_og_type_from_str_unknown() {
        assert!(OgType::from_str_value("unknown.type").is_none());
    }

    #[test]
    fn test_og_type_display() {
        assert_eq!(format!("{}", OgType::Article), "article");
    }

    // ── to_html_meta_tags ───────────────────────────────────────────────

    #[test]
    fn test_to_html_meta_tags_minimal() {
        let og = OpenGraph::new(OgType::Website, "My Page".to_string());
        let html = to_html_meta_tags(&og);
        assert!(html.contains("og:type"));
        assert!(html.contains("website"));
        assert!(html.contains("og:title"));
        assert!(html.contains("My Page"));
    }

    #[test]
    fn test_to_html_meta_tags_full() {
        let og = OpenGraph {
            og_type: OgType::VideoMovie,
            title: "Test Movie".to_string(),
            url: Some("https://example.com/movie".to_string()),
            image: Some("https://example.com/poster.jpg".to_string()),
            video: Some("https://example.com/trailer.mp4".to_string()),
            description: Some("A test movie".to_string()),
            site_name: Some("Example".to_string()),
            locale: Some("en_US".to_string()),
            video_width: Some(1920),
            video_height: Some(1080),
            video_type: Some("video/mp4".to_string()),
            image_width: Some(800),
            image_height: Some(600),
            image_type: Some("image/jpeg".to_string()),
            audio: Some("https://example.com/audio.mp3".to_string()),
            determiner: Some("the".to_string()),
        };
        let html = to_html_meta_tags(&og);
        assert!(html.contains("og:type\" content=\"video.movie\""));
        assert!(html.contains("og:title\" content=\"Test Movie\""));
        assert!(html.contains("og:url\" content=\"https://example.com/movie\""));
        assert!(html.contains("og:video:width\" content=\"1920\""));
        assert!(html.contains("og:video:height\" content=\"1080\""));
        assert!(html.contains("og:locale\" content=\"en_US\""));
        assert!(html.contains("og:audio\" content=\"https://example.com/audio.mp3\""));
        assert!(html.contains("og:determiner\" content=\"the\""));
    }

    #[test]
    fn test_to_html_meta_tags_escaping() {
        let og = OpenGraph::new(
            OgType::Website,
            "Title with \"quotes\" & <brackets>".to_string(),
        );
        let html = to_html_meta_tags(&og);
        assert!(html.contains("&amp;"));
        assert!(html.contains("&quot;"));
        assert!(html.contains("&lt;"));
        assert!(html.contains("&gt;"));
    }

    // ── from_html_meta ──────────────────────────────────────────────────

    #[test]
    fn test_from_html_meta_basic() {
        let html = r#"<html><head>
<meta property="og:type" content="article"/>
<meta property="og:title" content="My Article"/>
<meta property="og:url" content="https://example.com/article"/>
<meta property="og:description" content="An article"/>
</head></html>"#;
        let og = from_html_meta(html);
        assert_eq!(og.og_type, OgType::Article);
        assert_eq!(og.title, "My Article");
        assert_eq!(og.url.as_deref(), Some("https://example.com/article"));
        assert_eq!(og.description.as_deref(), Some("An article"));
    }

    #[test]
    fn test_from_html_meta_video_with_dimensions() {
        let html = r#"
<meta property="og:type" content="video.movie"/>
<meta property="og:title" content="Movie"/>
<meta property="og:video" content="https://cdn.example.com/v.mp4"/>
<meta property="og:video:width" content="1920"/>
<meta property="og:video:height" content="1080"/>
"#;
        let og = from_html_meta(html);
        assert_eq!(og.og_type, OgType::VideoMovie);
        assert_eq!(og.video.as_deref(), Some("https://cdn.example.com/v.mp4"));
        assert_eq!(og.video_width, Some(1920));
        assert_eq!(og.video_height, Some(1080));
    }

    #[test]
    fn test_from_html_meta_empty() {
        let og = from_html_meta("<html><head></head></html>");
        assert_eq!(og.og_type, OgType::Website);
        assert_eq!(og.title, "");
    }

    #[test]
    fn test_from_html_meta_escaped_content() {
        let html = r#"<meta property="og:title" content="Title &amp; More"/>"#;
        let og = from_html_meta(html);
        assert_eq!(og.title, "Title & More");
    }

    #[test]
    fn test_roundtrip_og_html() {
        let original = OpenGraph {
            og_type: OgType::MusicSong,
            title: "My Song".to_string(),
            url: Some("https://example.com/song".to_string()),
            image: Some("https://example.com/cover.jpg".to_string()),
            video: None,
            description: Some("A great song".to_string()),
            site_name: Some("MusicSite".to_string()),
            locale: None,
            video_width: None,
            video_height: None,
            video_type: None,
            image_width: None,
            image_height: None,
            image_type: None,
            audio: Some("https://example.com/song.mp3".to_string()),
            determiner: None,
        };
        let html = to_html_meta_tags(&original);
        let parsed = from_html_meta(&html);
        assert_eq!(parsed.og_type, original.og_type);
        assert_eq!(parsed.title, original.title);
        assert_eq!(parsed.url, original.url);
        assert_eq!(parsed.image, original.image);
        assert_eq!(parsed.description, original.description);
        assert_eq!(parsed.site_name, original.site_name);
        assert_eq!(parsed.audio, original.audio);
    }

    // ── Twitter Card ────────────────────────────────────────────────────

    #[test]
    fn test_twitter_card_type_roundtrip() {
        let types = [
            TwitterCardType::Summary,
            TwitterCardType::SummaryLargeImage,
            TwitterCardType::Player,
            TwitterCardType::App,
        ];
        for t in &types {
            let s = t.as_str();
            let parsed = TwitterCardType::from_str_value(s)
                .unwrap_or_else(|| panic!("failed to parse TwitterCardType: {s}"));
            assert_eq!(&parsed, t);
        }
    }

    #[test]
    fn test_to_twitter_meta_tags_basic() {
        let card = TwitterCard::new(TwitterCardType::Summary, "My Tweet".to_string())
            .with_description("A description".to_string())
            .with_image("https://example.com/img.jpg".to_string())
            .with_site("@example".to_string());
        let html = to_twitter_meta_tags(&card);
        assert!(html.contains("twitter:card\" content=\"summary\""));
        assert!(html.contains("twitter:title\" content=\"My Tweet\""));
        assert!(html.contains("twitter:description\" content=\"A description\""));
        assert!(html.contains("twitter:image\" content=\"https://example.com/img.jpg\""));
        assert!(html.contains("twitter:site\" content=\"@example\""));
    }

    #[test]
    fn test_to_twitter_meta_tags_player() {
        let card = TwitterCard {
            card_type: TwitterCardType::Player,
            title: "Video Card".to_string(),
            description: None,
            image: None,
            image_alt: Some("Alt text".to_string()),
            site: None,
            creator: Some("@creator".to_string()),
            player: Some("https://example.com/player".to_string()),
            player_width: Some(480),
            player_height: Some(360),
            player_stream: Some("https://example.com/stream.m3u8".to_string()),
        };
        let html = to_twitter_meta_tags(&card);
        assert!(html.contains("twitter:card\" content=\"player\""));
        assert!(html.contains("twitter:player\" content=\"https://example.com/player\""));
        assert!(html.contains("twitter:player:width\" content=\"480\""));
        assert!(html.contains("twitter:player:height\" content=\"360\""));
        assert!(html.contains("twitter:player:stream"));
        assert!(html.contains("twitter:creator\" content=\"@creator\""));
        assert!(html.contains("twitter:image:alt\" content=\"Alt text\""));
    }

    #[test]
    fn test_og_to_twitter_card_video() {
        let og = OpenGraph {
            og_type: OgType::VideoMovie,
            title: "Movie".to_string(),
            url: None,
            image: Some("https://example.com/poster.jpg".to_string()),
            video: Some("https://example.com/v.mp4".to_string()),
            description: Some("Desc".to_string()),
            site_name: Some("MySite".to_string()),
            locale: None,
            video_width: Some(1920),
            video_height: Some(1080),
            video_type: None,
            image_width: None,
            image_height: None,
            image_type: None,
            audio: None,
            determiner: None,
        };
        let card = og_to_twitter_card(&og);
        assert_eq!(card.card_type, TwitterCardType::SummaryLargeImage);
        assert_eq!(card.title, "Movie");
        assert_eq!(
            card.image.as_deref(),
            Some("https://example.com/poster.jpg")
        );
        assert_eq!(card.player.as_deref(), Some("https://example.com/v.mp4"));
        assert_eq!(card.player_width, Some(1920));
    }

    #[test]
    fn test_og_to_twitter_card_article() {
        let og = OpenGraph::new(OgType::Article, "Article Title".to_string());
        let card = og_to_twitter_card(&og);
        assert_eq!(card.card_type, TwitterCardType::Summary);
    }

    #[test]
    fn test_twitter_card_builder() {
        let card = TwitterCard::new(TwitterCardType::Summary, "Test".to_string())
            .with_description("Desc".to_string())
            .with_image("img.jpg".to_string())
            .with_site("@site".to_string())
            .with_creator("@me".to_string());
        assert_eq!(card.description.as_deref(), Some("Desc"));
        assert_eq!(card.image.as_deref(), Some("img.jpg"));
        assert_eq!(card.site.as_deref(), Some("@site"));
        assert_eq!(card.creator.as_deref(), Some("@me"));
    }

    #[test]
    fn test_og_builder_methods() {
        let og = OpenGraph::new(OgType::Video, "V".to_string())
            .with_url("https://example.com".to_string())
            .with_image("img.jpg".to_string())
            .with_video("vid.mp4".to_string())
            .with_description("D".to_string())
            .with_site_name("S".to_string());
        assert_eq!(og.url.as_deref(), Some("https://example.com"));
        assert_eq!(og.image.as_deref(), Some("img.jpg"));
        assert_eq!(og.video.as_deref(), Some("vid.mp4"));
        assert_eq!(og.description.as_deref(), Some("D"));
        assert_eq!(og.site_name.as_deref(), Some("S"));
    }

    #[test]
    fn test_twitter_card_type_display() {
        assert_eq!(
            format!("{}", TwitterCardType::SummaryLargeImage),
            "summary_large_image"
        );
    }

    #[test]
    fn test_html_attr_escape_passthrough() {
        assert_eq!(html_attr_escape("plain text"), "plain text");
    }
}
