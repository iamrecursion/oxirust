//! Schema.org structured metadata for media objects.
//!
//! Provides serialization/deserialization of Schema.org JSON-LD metadata for
//! VideoObject, AudioObject, MusicRecording, Movie, TVEpisode, Podcast,
//! NewsArticle, and ImageObject types.

use crate::Error;

/// Schema.org type for a media object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaType {
    /// Schema.org/VideoObject
    VideoObject,
    /// Schema.org/AudioObject
    AudioObject,
    /// Schema.org/MusicRecording
    MusicRecording,
    /// Schema.org/Movie
    Movie,
    /// Schema.org/TVEpisode
    TvEpisode,
    /// Schema.org/PodcastEpisode (often represented as Podcast)
    Podcast,
    /// Schema.org/NewsArticle
    NewsArticle,
    /// Schema.org/ImageObject
    ImageObject,
}

impl SchemaType {
    /// Return the Schema.org type name string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VideoObject => "VideoObject",
            Self::AudioObject => "AudioObject",
            Self::MusicRecording => "MusicRecording",
            Self::Movie => "Movie",
            Self::TvEpisode => "TVEpisode",
            Self::Podcast => "Podcast",
            Self::NewsArticle => "NewsArticle",
            Self::ImageObject => "ImageObject",
        }
    }

    /// Parse from a string. Returns None if unrecognized.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "VideoObject" => Some(Self::VideoObject),
            "AudioObject" => Some(Self::AudioObject),
            "MusicRecording" => Some(Self::MusicRecording),
            "Movie" => Some(Self::Movie),
            "TVEpisode" => Some(Self::TvEpisode),
            "Podcast" => Some(Self::Podcast),
            "NewsArticle" => Some(Self::NewsArticle),
            "ImageObject" => Some(Self::ImageObject),
            _ => None,
        }
    }
}

impl std::fmt::Display for SchemaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Core Schema.org object properties shared across all media types.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaOrgObject {
    /// The Schema.org type.
    pub schema_type: SchemaType,
    /// `@id` — URL uniquely identifying this item.
    pub id: Option<String>,
    /// Human-readable name.
    pub name: String,
    /// Description text.
    pub description: Option<String>,
    /// Canonical URL for this item.
    pub url: Option<String>,
    /// ISO 8601 duration string (e.g. "PT1H2M3S").
    pub duration_iso: Option<String>,
    /// URL of a thumbnail image.
    pub thumbnail_url: Option<String>,
    /// ISO 8601 upload/publish date (e.g. "2024-01-15").
    pub upload_date: Option<String>,
    /// Author or creator name.
    pub author: Option<String>,
}

impl SchemaOrgObject {
    /// Create a new `SchemaOrgObject` with required fields.
    #[must_use]
    pub fn new(schema_type: SchemaType, name: String) -> Self {
        Self {
            schema_type,
            id: None,
            name,
            description: None,
            url: None,
            duration_iso: None,
            thumbnail_url: None,
            upload_date: None,
            author: None,
        }
    }
}

/// Schema.org VideoObject with extended video-specific fields.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoObjectMeta {
    /// Base Schema.org object.
    pub schema_object: SchemaOrgObject,
    /// URL to the actual video content.
    pub content_url: String,
    /// URL for embedding the video.
    pub embed_url: Option<String>,
    /// Video height in pixels.
    pub height_px: Option<u32>,
    /// Video width in pixels.
    pub width_px: Option<u32>,
    /// Bitrate in kilobits per second.
    pub bitrate_kbps: Option<u32>,
    /// URLs of clip or chapter sub-items.
    pub has_parts: Vec<String>,
}

impl VideoObjectMeta {
    /// Create a new `VideoObjectMeta`.
    #[must_use]
    pub fn new(name: String, content_url: String) -> Self {
        let mut obj = SchemaOrgObject::new(SchemaType::VideoObject, name);
        obj.url = Some(content_url.clone());
        Self {
            schema_object: obj,
            content_url,
            embed_url: None,
            height_px: None,
            width_px: None,
            bitrate_kbps: None,
            has_parts: Vec::new(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Duration helpers
// ────────────────────────────────────────────────────────────────────────────

/// Convert a duration expressed in seconds to an ISO 8601 duration string.
///
/// Produces output of the form `PT1H2M3S`, omitting zero components except
/// when the total duration is zero (returns `PT0S`).
///
/// # Examples
/// ```
/// # use oximedia_metadata::schema_org::duration_to_iso8601;
/// assert_eq!(duration_to_iso8601(3723), "PT1H2M3S");
/// assert_eq!(duration_to_iso8601(90),   "PT1M30S");
/// assert_eq!(duration_to_iso8601(0),    "PT0S");
/// ```
#[must_use]
pub fn duration_to_iso8601(seconds: u64) -> String {
    if seconds == 0 {
        return "PT0S".to_string();
    }
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    let mut out = String::from("PT");
    if hours > 0 {
        out.push_str(&hours.to_string());
        out.push('H');
    }
    if minutes > 0 {
        out.push_str(&minutes.to_string());
        out.push('M');
    }
    if secs > 0 {
        out.push_str(&secs.to_string());
        out.push('S');
    }
    out
}

/// Parse an ISO 8601 duration string (PT format) into total seconds.
///
/// Accepts strings of the form `PT[nH][nM][nS]` where each component is
/// optional. Only integer values are supported (no decimal seconds).
///
/// # Errors
///
/// Returns [`Error::ParseError`] when the string is not a valid PT duration.
///
/// # Examples
/// ```
/// # use oximedia_metadata::schema_org::iso8601_to_seconds;
/// assert_eq!(iso8601_to_seconds("PT1H2M3S").unwrap(), 3723);
/// assert_eq!(iso8601_to_seconds("PT90S").unwrap(), 90);
/// assert_eq!(iso8601_to_seconds("PT0S").unwrap(), 0);
/// ```
pub fn iso8601_to_seconds(iso: &str) -> Result<u64, Error> {
    if !iso.starts_with("PT") {
        return Err(Error::ParseError(format!(
            "ISO 8601 duration must start with 'PT', got: {iso}"
        )));
    }
    let body = &iso[2..];
    let mut total: u64 = 0;
    let mut num_buf = String::new();

    for ch in body.chars() {
        if ch.is_ascii_digit() {
            num_buf.push(ch);
        } else {
            let n: u64 = num_buf.parse().map_err(|_| {
                Error::ParseError(format!("Invalid number in duration '{iso}': '{num_buf}'"))
            })?;
            num_buf.clear();
            match ch {
                'H' => total += n * 3600,
                'M' => total += n * 60,
                'S' => total += n,
                other => {
                    return Err(Error::ParseError(format!(
                        "Unexpected character '{other}' in duration '{iso}'"
                    )))
                }
            }
        }
    }
    if !num_buf.is_empty() {
        return Err(Error::ParseError(format!(
            "Trailing digits without unit in duration '{iso}'"
        )));
    }
    Ok(total)
}

// ────────────────────────────────────────────────────────────────────────────
// JSON-LD serialization
// ────────────────────────────────────────────────────────────────────────────

/// Escape a string value for embedding inside a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Append a JSON key/value pair (string value) to an output buffer.
fn append_kv(out: &mut String, key: &str, value: &str) {
    out.push_str(",\"");
    out.push_str(key);
    out.push_str("\":\"");
    out.push_str(&json_escape(value));
    out.push('"');
}

/// Serialize a [`SchemaOrgObject`] as a JSON-LD document.
///
/// The output is a compact (single-line) JSON-LD object.
#[must_use]
pub fn to_json_ld(obj: &SchemaOrgObject) -> String {
    let mut out = String::new();
    out.push('{');
    out.push_str("\"@context\":\"https://schema.org\"");
    out.push_str(",\"@type\":\"");
    out.push_str(obj.schema_type.as_str());
    out.push('"');
    append_kv(&mut out, "name", &obj.name);
    if let Some(ref id) = obj.id {
        append_kv(&mut out, "@id", id);
    }
    if let Some(ref desc) = obj.description {
        append_kv(&mut out, "description", desc);
    }
    if let Some(ref url) = obj.url {
        append_kv(&mut out, "url", url);
    }
    if let Some(ref dur) = obj.duration_iso {
        append_kv(&mut out, "duration", dur);
    }
    if let Some(ref thumb) = obj.thumbnail_url {
        append_kv(&mut out, "thumbnailUrl", thumb);
    }
    if let Some(ref date) = obj.upload_date {
        append_kv(&mut out, "uploadDate", date);
    }
    if let Some(ref author) = obj.author {
        out.push_str(",\"author\":{\"@type\":\"Person\",\"name\":\"");
        out.push_str(&json_escape(author));
        out.push_str("\"}");
    }
    out.push('}');
    out
}

// ────────────────────────────────────────────────────────────────────────────
// JSON-LD parsing  (no external JSON crate — pure string scanning)
// ────────────────────────────────────────────────────────────────────────────

/// Extract the string value that follows `"key":` in a JSON text.
///
/// Returns `None` when the key is absent or the value is not a quoted string.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..];
    let mut value = String::new();
    let mut chars = inner.chars();
    loop {
        match chars.next()? {
            '\\' => match chars.next()? {
                '"' => value.push('"'),
                '\\' => value.push('\\'),
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                other => {
                    value.push('\\');
                    value.push(other);
                }
            },
            '"' => break,
            c => value.push(c),
        }
    }
    Some(value)
}

/// Parse a JSON-LD document into a [`SchemaOrgObject`].
///
/// Uses lightweight string scanning without an external JSON library.
/// Only top-level string-valued keys are extracted; nested objects (e.g.
/// `author`) are handled specially.
///
/// Returns `None` if the `@type` field is missing or unrecognised.
#[must_use]
pub fn parse_json_ld(json: &str) -> Option<SchemaOrgObject> {
    let type_str = extract_json_string(json, "@type")?;
    let schema_type = SchemaType::from_str(&type_str)?;
    let name = extract_json_string(json, "name").unwrap_or_default();

    // Extract author name from nested {"@type":"Person","name":"..."} object.
    let author = extract_json_string(json, "name").and_then(|_| {
        // Look specifically inside an author object.
        let author_needle = "\"author\":";
        if let Some(pos) = json.find(author_needle) {
            let after = &json[pos + author_needle.len()..];
            extract_json_string(after, "name")
        } else {
            None
        }
    });
    // Avoid setting author equal to name when no author key exists.
    let author = if json.contains("\"author\":") {
        author
    } else {
        None
    };

    Some(SchemaOrgObject {
        schema_type,
        id: extract_json_string(json, "@id"),
        name,
        description: extract_json_string(json, "description"),
        url: extract_json_string(json, "url"),
        duration_iso: extract_json_string(json, "duration"),
        thumbnail_url: extract_json_string(json, "thumbnailUrl"),
        upload_date: extract_json_string(json, "uploadDate"),
        author,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── duration_to_iso8601 ──────────────────────────────────────────────

    #[test]
    fn test_duration_zero() {
        assert_eq!(duration_to_iso8601(0), "PT0S");
    }

    #[test]
    fn test_duration_seconds_only() {
        assert_eq!(duration_to_iso8601(45), "PT45S");
    }

    #[test]
    fn test_duration_minutes_and_seconds() {
        assert_eq!(duration_to_iso8601(90), "PT1M30S");
    }

    #[test]
    fn test_duration_hours_minutes_seconds() {
        assert_eq!(duration_to_iso8601(3723), "PT1H2M3S");
    }

    #[test]
    fn test_duration_exact_hour() {
        assert_eq!(duration_to_iso8601(3600), "PT1H");
    }

    #[test]
    fn test_duration_large_value() {
        // 10 hours exactly
        assert_eq!(duration_to_iso8601(36000), "PT10H");
    }

    // ── iso8601_to_seconds ───────────────────────────────────────────────

    #[test]
    fn test_parse_pt0s() {
        assert_eq!(iso8601_to_seconds("PT0S").expect("PT0S should parse"), 0);
    }

    #[test]
    fn test_parse_seconds_only() {
        assert_eq!(iso8601_to_seconds("PT45S").expect("PT45S should parse"), 45);
    }

    #[test]
    fn test_parse_minutes_seconds() {
        assert_eq!(
            iso8601_to_seconds("PT1M30S").expect("PT1M30S should parse"),
            90
        );
    }

    #[test]
    fn test_parse_hours_minutes_seconds() {
        assert_eq!(
            iso8601_to_seconds("PT1H2M3S").expect("PT1H2M3S should parse"),
            3723
        );
    }

    #[test]
    fn test_parse_hours_only() {
        assert_eq!(iso8601_to_seconds("PT2H").expect("PT2H should parse"), 7200);
    }

    #[test]
    fn test_parse_invalid_prefix() {
        assert!(iso8601_to_seconds("P1H").is_err());
        assert!(iso8601_to_seconds("1H2M").is_err());
    }

    #[test]
    fn test_parse_invalid_unit() {
        assert!(iso8601_to_seconds("PT1X").is_err());
    }

    #[test]
    fn test_roundtrip_duration() {
        for secs in [0u64, 1, 59, 60, 3600, 3723, 86399, 90061] {
            let iso = duration_to_iso8601(secs);
            let parsed = iso8601_to_seconds(&iso).expect("roundtrip duration should parse");
            assert_eq!(parsed, secs, "roundtrip failed for {secs}s → {iso}");
        }
    }

    // ── to_json_ld ───────────────────────────────────────────────────────

    #[test]
    fn test_to_json_ld_minimal() {
        let obj = SchemaOrgObject::new(SchemaType::VideoObject, "My Video".to_string());
        let ld = to_json_ld(&obj);
        assert!(ld.contains("\"@context\":\"https://schema.org\""));
        assert!(ld.contains("\"@type\":\"VideoObject\""));
        assert!(ld.contains("\"name\":\"My Video\""));
    }

    #[test]
    fn test_to_json_ld_with_all_fields() {
        let mut obj = SchemaOrgObject::new(SchemaType::Movie, "Interstellar".to_string());
        obj.id = Some("https://example.com/movie/1".to_string());
        obj.description = Some("A space epic".to_string());
        obj.url = Some("https://example.com/watch/1".to_string());
        obj.duration_iso = Some("PT2H49M".to_string());
        obj.thumbnail_url = Some("https://example.com/thumb.jpg".to_string());
        obj.upload_date = Some("2014-11-05".to_string());
        obj.author = Some("Christopher Nolan".to_string());

        let ld = to_json_ld(&obj);
        assert!(ld.contains("\"@type\":\"Movie\""));
        assert!(ld.contains("\"duration\":\"PT2H49M\""));
        assert!(ld.contains("\"uploadDate\":\"2014-11-05\""));
        assert!(ld.contains("Christopher Nolan"));
    }

    #[test]
    fn test_to_json_ld_escaping() {
        let mut obj =
            SchemaOrgObject::new(SchemaType::NewsArticle, "Test \"quoted\" name".to_string());
        obj.description = Some("Line1\nLine2".to_string());
        let ld = to_json_ld(&obj);
        assert!(ld.contains("\\\"quoted\\\""));
        assert!(ld.contains("\\n"));
    }

    // ── parse_json_ld ────────────────────────────────────────────────────

    #[test]
    fn test_parse_json_ld_minimal() {
        let json =
            r#"{"@context":"https://schema.org","@type":"VideoObject","name":"Sample Video"}"#;
        let obj = parse_json_ld(json).expect("should parse");
        assert_eq!(obj.schema_type, SchemaType::VideoObject);
        assert_eq!(obj.name, "Sample Video");
    }

    #[test]
    fn test_parse_json_ld_unknown_type() {
        let json = r#"{"@type":"UnknownType","name":"X"}"#;
        assert!(parse_json_ld(json).is_none());
    }

    #[test]
    fn test_parse_json_ld_missing_type() {
        let json = r#"{"name":"No Type"}"#;
        assert!(parse_json_ld(json).is_none());
    }

    #[test]
    fn test_roundtrip_json_ld() {
        let mut obj = SchemaOrgObject::new(SchemaType::AudioObject, "Podcast Episode".to_string());
        obj.description = Some("Great episode".to_string());
        obj.duration_iso = Some("PT45M".to_string());
        obj.upload_date = Some("2025-06-01".to_string());

        let ld = to_json_ld(&obj);
        let parsed = parse_json_ld(&ld).expect("roundtrip should parse");
        assert_eq!(parsed.schema_type, SchemaType::AudioObject);
        assert_eq!(parsed.name, "Podcast Episode");
        assert_eq!(parsed.description.as_deref(), Some("Great episode"));
        assert_eq!(parsed.duration_iso.as_deref(), Some("PT45M"));
    }

    // ── SchemaType ───────────────────────────────────────────────────────

    #[test]
    fn test_schema_type_roundtrip() {
        let types = [
            SchemaType::VideoObject,
            SchemaType::AudioObject,
            SchemaType::MusicRecording,
            SchemaType::Movie,
            SchemaType::TvEpisode,
            SchemaType::Podcast,
            SchemaType::NewsArticle,
            SchemaType::ImageObject,
        ];
        for t in &types {
            let s = t.as_str();
            let parsed = SchemaType::from_str(s).expect("should parse");
            assert_eq!(&parsed, t, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_video_object_meta_new() {
        let meta = VideoObjectMeta::new(
            "Test Video".to_string(),
            "https://cdn.example.com/video.mp4".to_string(),
        );
        assert_eq!(meta.schema_object.schema_type, SchemaType::VideoObject);
        assert_eq!(meta.content_url, "https://cdn.example.com/video.mp4");
        assert!(meta.has_parts.is_empty());
    }
}
