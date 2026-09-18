//! Facet aggregation.
//!
//! This module defines [`Facets`] — the aggregated facet summary that is
//! returned alongside every search result set — together with a free-standing
//! [`aggregate_facets`] function that computes it from a slice of
//! [`crate::SearchResultItem`]s.
//!
//! # Facet groups
//!
//! | Field | Source | Buckets / values |
//! |---|---|---|
//! | `formats` | MIME-type prefix (`video/`, `audio/`, `image/`) | `"video"`, `"audio"`, `"image"`, `"other"` |
//! | `codecs` | MIME-type subtype (the part after `/`) | raw value, e.g. `"mp4"`, `"webm"` |
//! | `duration_ranges` | `duration_ms` | `<1min`, `1-5min`, `5-30min`, `>30min` |
//! | `resolutions` | width/height embedded in `file_path` suffix | `SD`, `HD`, `4K`, `8K` |
//! | `date_ranges` | `created_at` (Unix seconds) | `today`, `this_week`, `this_month`, `this_year`, `older` |
//! | `mime_types` | `mime_type` raw value | counted as-is |
//! | `categories` | title-derived coarse category | `"video"`, `"audio"`, `"image"`, `"document"`, `"other"` |
//! | `tags` | keywords extracted from title / description | individual words |

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public data types
// ──────────────────────────────────────────────────────────────────────────────

/// Facet aggregations returned alongside a set of search results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Facets {
    /// MIME type facets (raw MIME strings, e.g. `"video/mp4"`).
    pub mime_types: Vec<FacetCount>,
    /// Media format facets (`"video"`, `"audio"`, `"image"`, `"other"`).
    pub formats: Vec<FacetCount>,
    /// Codec / container facets derived from the MIME subtype.
    pub codecs: Vec<FacetCount>,
    /// Resolution tier facets (`"SD"`, `"HD"`, `"4K"`, `"8K"`).
    pub resolutions: Vec<FacetCount>,
    /// Category facets (coarse classification).
    pub categories: Vec<FacetCount>,
    /// Duration range facets (`"<1min"`, `"1-5min"`, `"5-30min"`, `">30min"`).
    pub duration_ranges: Vec<FacetCount>,
    /// Date range facets (`"today"`, `"this_week"`, `"this_month"`, `"this_year"`, `"older"`).
    pub date_ranges: Vec<FacetCount>,
    /// Tag facets (keywords derived from titles).
    pub tags: Vec<FacetCount>,
}

/// A single facet value with its document count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetCount {
    /// Facet value (human-readable label or raw value).
    pub value: String,
    /// Number of documents that carry this facet value.
    pub count: usize,
}

impl Facets {
    /// Create new empty facets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Duration bucket boundaries in milliseconds.
const ONE_MIN_MS: i64 = 60_000;
const FIVE_MIN_MS: i64 = 5 * ONE_MIN_MS;
const THIRTY_MIN_MS: i64 = 30 * ONE_MIN_MS;

/// Date boundary offsets in seconds.
const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_WEEK: i64 = 7 * SECS_PER_DAY;
const SECS_PER_MONTH: i64 = 30 * SECS_PER_DAY;
const SECS_PER_YEAR: i64 = 365 * SECS_PER_DAY;

/// Classify a duration in milliseconds into a human-readable bucket label.
fn duration_bucket(duration_ms: i64) -> &'static str {
    if duration_ms < ONE_MIN_MS {
        "<1min"
    } else if duration_ms < FIVE_MIN_MS {
        "1-5min"
    } else if duration_ms < THIRTY_MIN_MS {
        "5-30min"
    } else {
        ">30min"
    }
}

/// Classify a Unix-second timestamp relative to `now_secs` into a date bucket.
fn date_bucket(created_at_secs: i64, now_secs: i64) -> &'static str {
    let age = now_secs.saturating_sub(created_at_secs);
    if age < SECS_PER_DAY {
        "today"
    } else if age < SECS_PER_WEEK {
        "this_week"
    } else if age < SECS_PER_MONTH {
        "this_month"
    } else if age < SECS_PER_YEAR {
        "this_year"
    } else {
        "older"
    }
}

/// Derive the coarse media format from a MIME type string.
///
/// Returns `"video"`, `"audio"`, `"image"`, or `"other"`.
fn format_from_mime(mime: &str) -> &'static str {
    if mime.starts_with("video/") {
        "video"
    } else if mime.starts_with("audio/") {
        "audio"
    } else if mime.starts_with("image/") {
        "image"
    } else {
        "other"
    }
}

/// Extract the codec/container label from a MIME type string.
///
/// For `"video/mp4"` this returns `"mp4"`.  If the MIME type is absent or
/// malformed, returns `"unknown"`.
fn codec_from_mime(mime: &str) -> &str {
    mime.split_once('/')
        .map(|(_, sub)| sub)
        .unwrap_or("unknown")
}

/// Attempt to classify a resolution tier from a file path.
///
/// The heuristic looks for well-known suffix patterns (e.g. `_4k`, `_1080p`,
/// `_720p`, `_8k`) that are commonly appended by media management tools.
/// Returns `None` when no recognisable pattern is found so that callers can
/// skip the result rather than emit a misleading bucket.
fn resolution_from_path(path: &str) -> Option<&'static str> {
    let lower = path.to_ascii_lowercase();
    // Check from highest resolution downwards so that "_8k" wins over "_4k".
    if lower.contains("8k") || lower.contains("7680") || lower.contains("4320") {
        Some("8K")
    } else if lower.contains("4k") || lower.contains("2160") || lower.contains("uhd") {
        Some("4K")
    } else if lower.contains("1080") || lower.contains("fhd") || lower.contains("fullhd") {
        Some("HD")
    } else if lower.contains("720p")
        || lower.contains("720 ")
        || lower.contains("_720")
        || lower.contains("hd_")
        || lower.contains("-hd")
    {
        Some("HD")
    } else if lower.contains("480")
        || lower.contains("360")
        || lower.contains("240")
        || lower.contains("sd")
    {
        Some("SD")
    } else {
        None
    }
}

/// Extract simple tag words from an optional title string.
///
/// Stop-words (short, very common English words) are filtered out to prevent
/// them from dominating the tag facet counts.
fn tags_from_title(title: Option<&str>) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "it", "as", "be", "was", "are", "has", "had", "not", "no", "up", "do", "if",
        "so", "we", "i",
    ];

    let raw = match title {
        Some(t) => t,
        None => return Vec::new(),
    };

    raw.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

/// Convert a `HashMap<String, usize>` of counts into a sorted `Vec<FacetCount>`.
///
/// Results are ordered by descending count; ties are broken alphabetically.
fn counts_to_vec(map: HashMap<String, usize>) -> Vec<FacetCount> {
    let mut vec: Vec<FacetCount> = map
        .into_iter()
        .map(|(value, count)| FacetCount { value, count })
        .collect();
    vec.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
    vec
}

// ──────────────────────────────────────────────────────────────────────────────
// Public aggregation entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Compute [`Facets`] from a slice of search result items.
///
/// `now_secs` should be the current Unix timestamp in **seconds** (use
/// `std::time::SystemTime::now()` or a clock abstraction).  Passing the
/// current time as an explicit parameter makes the function deterministic and
/// easy to unit-test.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_search::facet::aggregation::{aggregate_facets, Facets};
/// use oximedia_search::SearchResultItem;
/// use uuid::Uuid;
///
/// let items: Vec<SearchResultItem> = vec![];
/// let now_secs = std::time::SystemTime::now()
///     .duration_since(std::time::UNIX_EPOCH)
///     .map(|d| d.as_secs() as i64)
///     .unwrap_or(0);
/// let facets: Facets = aggregate_facets(&items, now_secs);
/// ```
#[must_use]
pub fn aggregate_facets(items: &[crate::SearchResultItem], now_secs: i64) -> Facets {
    // Accumulator maps  (value → count) for each facet dimension.
    let mut mime_counts: HashMap<String, usize> = HashMap::new();
    let mut format_counts: HashMap<String, usize> = HashMap::new();
    let mut codec_counts: HashMap<String, usize> = HashMap::new();
    let mut resolution_counts: HashMap<String, usize> = HashMap::new();
    let mut category_counts: HashMap<String, usize> = HashMap::new();
    let mut duration_counts: HashMap<String, usize> = HashMap::new();
    let mut date_counts: HashMap<String, usize> = HashMap::new();
    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for item in items {
        // ── MIME type ────────────────────────────────────────────────────────
        let mime_str: &str = item
            .mime_type
            .as_deref()
            .unwrap_or("application/octet-stream");

        *mime_counts.entry(mime_str.to_string()).or_insert(0) += 1;

        // ── Format (video / audio / image / other) ───────────────────────────
        let fmt = format_from_mime(mime_str);
        *format_counts.entry(fmt.to_string()).or_insert(0) += 1;

        // ── Codec / container ────────────────────────────────────────────────
        let codec = codec_from_mime(mime_str);
        *codec_counts.entry(codec.to_string()).or_insert(0) += 1;

        // ── Resolution tier ──────────────────────────────────────────────────
        if let Some(res) = resolution_from_path(&item.file_path) {
            *resolution_counts.entry(res.to_string()).or_insert(0) += 1;
        }

        // ── Category (coarse, same logic as format for media assets) ─────────
        // For richer classification the file extension is checked as well.
        let ext = std::path::Path::new(&item.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());

        let category = match ext.as_deref() {
            Some("mp4" | "mkv" | "mov" | "avi" | "webm" | "flv" | "ts" | "m2ts" | "mxf") => "video",
            Some("mp3" | "flac" | "opus" | "ogg" | "wav" | "aac" | "m4a" | "aiff") => "audio",
            Some(
                "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp" | "avif" | "heic"
                | "jxl" | "dng",
            ) => "image",
            Some("pdf" | "doc" | "docx" | "txt" | "md" | "rst" | "html" | "htm") => "document",
            _ => {
                // Fall back to MIME-derived format.
                fmt
            }
        };
        *category_counts.entry(category.to_string()).or_insert(0) += 1;

        // ── Duration range ───────────────────────────────────────────────────
        if let Some(dur_ms) = item.duration_ms {
            let bucket = duration_bucket(dur_ms);
            *duration_counts.entry(bucket.to_string()).or_insert(0) += 1;
        }

        // ── Date range ───────────────────────────────────────────────────────
        let dbucket = date_bucket(item.created_at, now_secs);
        *date_counts.entry(dbucket.to_string()).or_insert(0) += 1;

        // ── Tags from title ──────────────────────────────────────────────────
        for tag in tags_from_title(item.title.as_deref()) {
            *tag_counts.entry(tag).or_insert(0) += 1;
        }
    }

    Facets {
        mime_types: counts_to_vec(mime_counts),
        formats: counts_to_vec(format_counts),
        codecs: counts_to_vec(codec_counts),
        resolutions: counts_to_vec(resolution_counts),
        categories: counts_to_vec(category_counts),
        duration_ranges: counts_to_vec(duration_counts),
        date_ranges: counts_to_vec(date_counts),
        tags: counts_to_vec(tag_counts),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Hierarchical facets
// ──────────────────────────────────────────────────────────────────────────────

/// A hierarchical facet node that can contain nested children.
///
/// This enables drilldown navigation such as:
/// - Format > Codec > Profile  (e.g., Video > AV1 > Main)
/// - Category > Subcategory    (e.g., Music > Jazz > BeBop)
/// - Date > Year > Month       (e.g., 2025 > January)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalFacet {
    /// The facet value at this level.
    pub value: String,
    /// Number of documents matching at this level (including children).
    pub count: usize,
    /// Child facets at the next level of the hierarchy.
    pub children: Vec<HierarchicalFacet>,
}

impl HierarchicalFacet {
    /// Create a new leaf facet with no children.
    #[must_use]
    pub fn leaf(value: impl Into<String>, count: usize) -> Self {
        Self {
            value: value.into(),
            count,
            children: Vec::new(),
        }
    }

    /// Create a new facet node with children.
    #[must_use]
    pub fn node(value: impl Into<String>, count: usize, children: Vec<Self>) -> Self {
        Self {
            value: value.into(),
            count,
            children,
        }
    }

    /// Total count of all descendants (recursive).
    #[must_use]
    pub fn total_descendant_count(&self) -> usize {
        let mut total = self.count;
        for child in &self.children {
            total += child.total_descendant_count();
        }
        total
    }

    /// Depth of the hierarchy (1 for a leaf).
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(Self::depth).max().unwrap_or(0)
        }
    }

    /// Find a child by value (non-recursive, immediate children only).
    #[must_use]
    pub fn find_child(&self, value: &str) -> Option<&Self> {
        self.children.iter().find(|c| c.value == value)
    }
}

/// Result of hierarchical facet aggregation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HierarchicalFacets {
    /// Format hierarchy: format > codec > profile
    pub format_hierarchy: Vec<HierarchicalFacet>,
    /// Date hierarchy: year > month
    pub date_hierarchy: Vec<HierarchicalFacet>,
    /// Category hierarchy: top-level category > extension-derived subcategory
    pub category_hierarchy: Vec<HierarchicalFacet>,
}

/// Build hierarchical facets from a slice of search result items.
///
/// Produces a three-level format hierarchy (format > codec), a date
/// hierarchy (year > month), and a category hierarchy (category > extension).
#[must_use]
pub fn aggregate_hierarchical_facets(
    items: &[crate::SearchResultItem],
    now_secs: i64,
) -> HierarchicalFacets {
    // ── Format > Codec hierarchy ──
    // Accumulate: format -> codec -> count
    let mut format_codec: HashMap<String, HashMap<String, usize>> = HashMap::new();
    // ── Date: year -> month -> count ──
    let mut date_year_month: HashMap<i32, HashMap<u32, usize>> = HashMap::new();
    // ── Category > extension ──
    let mut category_ext: HashMap<String, HashMap<String, usize>> = HashMap::new();

    let _ = now_secs; // used for consistency with aggregate_facets signature

    for item in items {
        let mime_str = item
            .mime_type
            .as_deref()
            .unwrap_or("application/octet-stream");

        // Format > Codec
        let fmt = format_from_mime(mime_str).to_string();
        let codec = codec_from_mime(mime_str).to_string();
        *format_codec
            .entry(fmt.clone())
            .or_default()
            .entry(codec)
            .or_insert(0) += 1;

        // Date hierarchy (year > month from created_at)
        let (year, month) = year_month_from_timestamp(item.created_at);
        *date_year_month
            .entry(year)
            .or_default()
            .entry(month)
            .or_insert(0) += 1;

        // Category > extension
        let ext = std::path::Path::new(&item.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        let category = match ext.as_str() {
            "mp4" | "mkv" | "mov" | "avi" | "webm" | "flv" | "ts" | "m2ts" | "mxf" => "video",
            "mp3" | "flac" | "opus" | "ogg" | "wav" | "aac" | "m4a" | "aiff" => "audio",
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp" | "avif" | "heic"
            | "jxl" | "dng" => "image",
            "pdf" | "doc" | "docx" | "txt" | "md" | "rst" | "html" | "htm" => "document",
            _ => format_from_mime(mime_str),
        };
        *category_ext
            .entry(category.to_string())
            .or_default()
            .entry(if ext.is_empty() {
                "unknown".to_string()
            } else {
                ext
            })
            .or_insert(0) += 1;
    }

    // Build format hierarchy
    let mut format_hierarchy: Vec<HierarchicalFacet> = format_codec
        .into_iter()
        .map(|(fmt, codecs)| {
            let total: usize = codecs.values().sum();
            let mut children: Vec<HierarchicalFacet> = codecs
                .into_iter()
                .map(|(codec, count)| HierarchicalFacet::leaf(codec, count))
                .collect();
            children.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
            HierarchicalFacet::node(fmt, total, children)
        })
        .collect();
    format_hierarchy.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));

    // Build date hierarchy
    let mut date_hierarchy: Vec<HierarchicalFacet> = date_year_month
        .into_iter()
        .map(|(year, months)| {
            let total: usize = months.values().sum();
            let mut children: Vec<HierarchicalFacet> = months
                .into_iter()
                .map(|(month, count)| {
                    let month_name = month_name(month);
                    HierarchicalFacet::leaf(month_name, count)
                })
                .collect();
            children.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
            HierarchicalFacet::node(year.to_string(), total, children)
        })
        .collect();
    date_hierarchy.sort_by(|a, b| b.value.cmp(&a.value)); // Most recent year first

    // Build category hierarchy
    let mut category_hierarchy: Vec<HierarchicalFacet> = category_ext
        .into_iter()
        .map(|(cat, exts)| {
            let total: usize = exts.values().sum();
            let mut children: Vec<HierarchicalFacet> = exts
                .into_iter()
                .map(|(ext, count)| HierarchicalFacet::leaf(ext, count))
                .collect();
            children.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
            HierarchicalFacet::node(cat, total, children)
        })
        .collect();
    category_hierarchy.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));

    HierarchicalFacets {
        format_hierarchy,
        date_hierarchy,
        category_hierarchy,
    }
}

/// Extract year and month from a Unix timestamp (seconds).
fn year_month_from_timestamp(secs: i64) -> (i32, u32) {
    // Simple calendar calculation without external dependencies.
    // Days since Unix epoch.
    let days = secs / SECS_PER_DAY;
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m)
}

/// Convert a month number (1-12) to a short name.
fn month_name(month: u32) -> String {
    match month {
        1 => "January".to_string(),
        2 => "February".to_string(),
        3 => "March".to_string(),
        4 => "April".to_string(),
        5 => "May".to_string(),
        6 => "June".to_string(),
        7 => "July".to_string(),
        8 => "August".to_string(),
        9 => "September".to_string(),
        10 => "October".to_string(),
        11 => "November".to_string(),
        12 => "December".to_string(),
        _ => format!("Month-{month}"),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchResultItem;
    use uuid::Uuid;

    /// Helper: build a minimal `SearchResultItem` with the given fields.
    fn make_item(
        mime_type: Option<&str>,
        duration_ms: Option<i64>,
        created_at: i64,
        file_path: &str,
        title: Option<&str>,
    ) -> SearchResultItem {
        SearchResultItem {
            asset_id: Uuid::new_v4(),
            score: 1.0,
            title: title.map(str::to_string),
            description: None,
            file_path: file_path.to_string(),
            mime_type: mime_type.map(str::to_string),
            duration_ms,
            created_at,
            modified_at: None,
            file_size: None,
            matched_fields: Vec::new(),
            thumbnail_url: None,
        }
    }

    // Fixed "now" for all tests: 2025-01-01 00:00:00 UTC
    const NOW: i64 = 1_735_689_600_i64;

    #[test]
    fn test_facets_new() {
        let facets = Facets::new();
        assert!(facets.mime_types.is_empty());
        assert!(facets.formats.is_empty());
        assert!(facets.codecs.is_empty());
        assert!(facets.duration_ranges.is_empty());
        assert!(facets.date_ranges.is_empty());
        assert!(facets.tags.is_empty());
    }

    #[test]
    fn test_aggregate_empty() {
        let facets = aggregate_facets(&[], NOW);
        assert!(facets.mime_types.is_empty());
        assert!(facets.formats.is_empty());
        assert!(facets.duration_ranges.is_empty());
    }

    #[test]
    fn test_format_facets() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW, "a.mp4", None),
            make_item(Some("video/webm"), None, NOW, "b.webm", None),
            make_item(Some("audio/flac"), None, NOW, "c.flac", None),
            make_item(Some("image/png"), None, NOW, "d.png", None),
        ];
        let facets = aggregate_facets(&items, NOW);
        // Two video items → "video" count = 2
        let video = facets.formats.iter().find(|f| f.value == "video");
        assert!(video.is_some());
        assert_eq!(video.map(|f| f.count), Some(2));
        // One audio item
        let audio = facets.formats.iter().find(|f| f.value == "audio");
        assert_eq!(audio.map(|f| f.count), Some(1));
    }

    #[test]
    fn test_codec_facets() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW, "a.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "b.mp4", None),
            make_item(Some("video/webm"), None, NOW, "c.webm", None),
        ];
        let facets = aggregate_facets(&items, NOW);
        let mp4 = facets.codecs.iter().find(|f| f.value == "mp4");
        assert_eq!(mp4.map(|f| f.count), Some(2));
        let webm = facets.codecs.iter().find(|f| f.value == "webm");
        assert_eq!(webm.map(|f| f.count), Some(1));
        // mp4 should appear before webm (higher count)
        assert_eq!(facets.codecs[0].value, "mp4");
    }

    #[test]
    fn test_duration_range_facets() {
        let items = vec![
            make_item(Some("video/mp4"), Some(30_000), NOW, "a.mp4", None), // <1min (30 s)
            make_item(Some("video/mp4"), Some(90_000), NOW, "b.mp4", None), // 1-5min (90 s)
            make_item(Some("video/mp4"), Some(360_000), NOW, "c.mp4", None), // 5-30min (6 min)
            make_item(Some("video/mp4"), Some(2_000_000), NOW, "d.mp4", None), // >30min
            make_item(Some("video/mp4"), None, NOW, "e.mp4", None),         // no duration → omitted
        ];
        let facets = aggregate_facets(&items, NOW);
        let find = |v: &str| {
            facets
                .duration_ranges
                .iter()
                .find(|f| f.value == v)
                .map(|f| f.count)
        };
        assert_eq!(find("<1min"), Some(1));
        assert_eq!(find("1-5min"), Some(1));
        assert_eq!(find("5-30min"), Some(1));
        assert_eq!(find(">30min"), Some(1));
        // Item with no duration should not contribute any bucket.
        assert_eq!(
            facets
                .duration_ranges
                .iter()
                .map(|f| f.count)
                .sum::<usize>(),
            4
        );
    }

    #[test]
    fn test_date_range_facets() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW - 3600, "a.mp4", None), // today
            make_item(
                Some("video/mp4"),
                None,
                NOW - SECS_PER_DAY - 1,
                "b.mp4",
                None,
            ), // this_week
            make_item(
                Some("video/mp4"),
                None,
                NOW - SECS_PER_WEEK - 1,
                "c.mp4",
                None,
            ), // this_month
            make_item(
                Some("video/mp4"),
                None,
                NOW - SECS_PER_MONTH - 1,
                "d.mp4",
                None,
            ), // this_year
            make_item(
                Some("video/mp4"),
                None,
                NOW - SECS_PER_YEAR - 1,
                "e.mp4",
                None,
            ), // older
        ];
        let facets = aggregate_facets(&items, NOW);
        let find = |v: &str| {
            facets
                .date_ranges
                .iter()
                .find(|f| f.value == v)
                .map(|f| f.count)
        };
        assert_eq!(find("today"), Some(1));
        assert_eq!(find("this_week"), Some(1));
        assert_eq!(find("this_month"), Some(1));
        assert_eq!(find("this_year"), Some(1));
        assert_eq!(find("older"), Some(1));
    }

    #[test]
    fn test_resolution_facets() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW, "movie_4k.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "clip_1080p.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "short_8k.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "unknown.mp4", None), // no tier
        ];
        let facets = aggregate_facets(&items, NOW);
        assert!(facets
            .resolutions
            .iter()
            .any(|f| f.value == "4K" && f.count == 1));
        assert!(facets
            .resolutions
            .iter()
            .any(|f| f.value == "HD" && f.count == 1));
        assert!(facets
            .resolutions
            .iter()
            .any(|f| f.value == "8K" && f.count == 1));
        // "unknown.mp4" contributes no resolution facet.
        assert_eq!(facets.resolutions.iter().map(|f| f.count).sum::<usize>(), 3);
    }

    #[test]
    fn test_tag_facets() {
        let items = vec![
            make_item(
                Some("video/mp4"),
                None,
                NOW,
                "a.mp4",
                Some("Nature Wildlife Documentary"),
            ),
            make_item(Some("video/mp4"), None, NOW, "b.mp4", Some("Nature Park")),
            make_item(
                Some("video/mp4"),
                None,
                NOW,
                "c.mp4",
                Some("City Documentary"),
            ),
        ];
        let facets = aggregate_facets(&items, NOW);
        let find = |v: &str| facets.tags.iter().find(|f| f.value == v).map(|f| f.count);
        assert_eq!(find("nature"), Some(2));
        assert_eq!(find("documentary"), Some(2));
        assert_eq!(find("wildlife"), Some(1));
        assert_eq!(find("park"), Some(1));
        assert_eq!(find("city"), Some(1));
    }

    #[test]
    fn test_mime_type_no_type_defaults() {
        // Items with no MIME type should be bucketed under
        // "application/octet-stream" and format "other".
        let items = vec![make_item(None, None, NOW, "mystery.bin", None)];
        let facets = aggregate_facets(&items, NOW);
        assert!(facets
            .mime_types
            .iter()
            .any(|f| f.value == "application/octet-stream"));
        assert!(facets.formats.iter().any(|f| f.value == "other"));
    }

    #[test]
    fn test_facet_counts_sorted_descending() {
        // Ensure the most common codec appears first.
        let items = vec![
            make_item(Some("video/webm"), None, NOW, "a.webm", None),
            make_item(Some("video/mp4"), None, NOW, "b.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "c.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "d.mp4", None),
        ];
        let facets = aggregate_facets(&items, NOW);
        assert_eq!(facets.codecs[0].value, "mp4");
        assert_eq!(facets.codecs[0].count, 3);
    }

    // ── Hierarchical facet tests ──

    #[test]
    fn test_hierarchical_facet_leaf() {
        let leaf = HierarchicalFacet::leaf("mp4", 5);
        assert_eq!(leaf.value, "mp4");
        assert_eq!(leaf.count, 5);
        assert!(leaf.children.is_empty());
        assert_eq!(leaf.depth(), 1);
    }

    #[test]
    fn test_hierarchical_facet_node() {
        let node = HierarchicalFacet::node(
            "video",
            10,
            vec![
                HierarchicalFacet::leaf("mp4", 7),
                HierarchicalFacet::leaf("webm", 3),
            ],
        );
        assert_eq!(node.value, "video");
        assert_eq!(node.children.len(), 2);
        assert_eq!(node.depth(), 2);
    }

    #[test]
    fn test_hierarchical_facet_find_child() {
        let node = HierarchicalFacet::node(
            "video",
            10,
            vec![
                HierarchicalFacet::leaf("mp4", 7),
                HierarchicalFacet::leaf("webm", 3),
            ],
        );
        let mp4 = node.find_child("mp4");
        assert!(mp4.is_some());
        assert_eq!(mp4.map(|c| c.count), Some(7));
        assert!(node.find_child("avi").is_none());
    }

    #[test]
    fn test_hierarchical_facet_total_descendant_count() {
        let node = HierarchicalFacet::node(
            "video",
            10,
            vec![
                HierarchicalFacet::leaf("mp4", 7),
                HierarchicalFacet::leaf("webm", 3),
            ],
        );
        // 10 (self) + 7 + 3
        assert_eq!(node.total_descendant_count(), 20);
    }

    #[test]
    fn test_hierarchical_format_aggregation() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW, "a.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "b.mp4", None),
            make_item(Some("video/webm"), None, NOW, "c.webm", None),
            make_item(Some("audio/flac"), None, NOW, "d.flac", None),
        ];
        let h = aggregate_hierarchical_facets(&items, NOW);

        // Should have "video" and "audio" at top level
        let video = h.format_hierarchy.iter().find(|f| f.value == "video");
        assert!(video.is_some());
        let video = video.expect("video present");
        assert_eq!(video.count, 3); // 2 mp4 + 1 webm

        // Video should have mp4 and webm children
        let mp4_child = video.find_child("mp4");
        assert_eq!(mp4_child.map(|c| c.count), Some(2));
        let webm_child = video.find_child("webm");
        assert_eq!(webm_child.map(|c| c.count), Some(1));

        let audio = h.format_hierarchy.iter().find(|f| f.value == "audio");
        assert!(audio.is_some());
        assert_eq!(audio.map(|f| f.count), Some(1));
    }

    #[test]
    fn test_hierarchical_date_aggregation() {
        // 2025-01-01 00:00:00 UTC = 1735689600
        // 2024-06-15 00:00:00 UTC ~ 1718409600
        let items = vec![
            make_item(Some("video/mp4"), None, 1_735_689_600, "a.mp4", None), // 2025-01
            make_item(
                Some("video/mp4"),
                None,
                1_735_689_600 + 86400,
                "b.mp4",
                None,
            ), // 2025-01
            make_item(Some("video/mp4"), None, 1_718_409_600, "c.mp4", None), // 2024-06
        ];
        let h = aggregate_hierarchical_facets(&items, NOW);

        // Should have at least year 2025 and 2024
        assert!(!h.date_hierarchy.is_empty());
        let y2025 = h.date_hierarchy.iter().find(|f| f.value == "2025");
        assert!(y2025.is_some());
        let y2025 = y2025.expect("2025 present");
        assert_eq!(y2025.count, 2); // two items in January 2025
        assert!(!y2025.children.is_empty());
        let jan = y2025.find_child("January");
        assert!(jan.is_some());
    }

    #[test]
    fn test_hierarchical_category_aggregation() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW, "movie.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "clip.mkv", None),
            make_item(Some("audio/flac"), None, NOW, "song.flac", None),
            make_item(Some("image/png"), None, NOW, "photo.png", None),
            make_item(Some("image/png"), None, NOW, "icon.jpg", None),
        ];
        let h = aggregate_hierarchical_facets(&items, NOW);

        let video_cat = h.category_hierarchy.iter().find(|f| f.value == "video");
        assert!(video_cat.is_some());
        let video_cat = video_cat.expect("video category present");
        assert_eq!(video_cat.count, 2);
        // Should have mp4 and mkv as children
        assert!(video_cat.find_child("mp4").is_some());
        assert!(video_cat.find_child("mkv").is_some());

        let image_cat = h.category_hierarchy.iter().find(|f| f.value == "image");
        assert!(image_cat.is_some());
        let image_cat = image_cat.expect("image category present");
        assert_eq!(image_cat.count, 2);
    }

    #[test]
    fn test_hierarchical_empty_input() {
        let h = aggregate_hierarchical_facets(&[], NOW);
        assert!(h.format_hierarchy.is_empty());
        assert!(h.date_hierarchy.is_empty());
        assert!(h.category_hierarchy.is_empty());
    }

    #[test]
    fn test_hierarchical_format_sorted_by_count() {
        let items = vec![
            make_item(Some("video/mp4"), None, NOW, "a.mp4", None),
            make_item(Some("video/mp4"), None, NOW, "b.mp4", None),
            make_item(Some("audio/flac"), None, NOW, "c.flac", None),
        ];
        let h = aggregate_hierarchical_facets(&items, NOW);
        // "video" (count=2) should come before "audio" (count=1)
        assert_eq!(h.format_hierarchy[0].value, "video");
    }

    #[test]
    fn test_hierarchical_depth_nested() {
        let nested = HierarchicalFacet::node(
            "root",
            100,
            vec![HierarchicalFacet::node(
                "level1",
                50,
                vec![HierarchicalFacet::leaf("level2", 25)],
            )],
        );
        assert_eq!(nested.depth(), 3);
    }

    #[test]
    fn test_year_month_from_known_timestamp() {
        // 2025-01-01 00:00:00 UTC = 1735689600
        let (year, month) = year_month_from_timestamp(1_735_689_600);
        assert_eq!(year, 2025);
        assert_eq!(month, 1);
    }
}
