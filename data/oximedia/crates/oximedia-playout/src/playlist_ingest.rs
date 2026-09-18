//! Playlist ingest for the playout server.
//!
//! Provides `IngestFormat`, `PlaylistItem`, and `PlaylistIngest` for
//! ingesting and validating playlist content before playout.

#![allow(dead_code)]

use std::path::PathBuf;

// ── IngestFormat ──────────────────────────────────────────────────────────────

/// Supported ingest formats for playlist items.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestFormat {
    /// MXF (Material eXchange Format) — professional broadcast container.
    Mxf,
    /// Matroska video container.
    Mkv,
    /// MP4 / ISOBMFF.
    Mp4,
    /// Transport Stream (MPEG-2 TS).
    Ts,
    /// QuickTime MOV.
    Mov,
    /// BXF (Broadcast eXchange Format) XML playlist.
    Bxf,
    /// Plain text file list.
    TextList,
}

impl IngestFormat {
    /// Returns `true` when this format can carry embedded metadata.
    pub fn supports_metadata(&self) -> bool {
        matches!(
            self,
            Self::Mxf | Self::Mp4 | Self::Mkv | Self::Mov | Self::Bxf
        )
    }

    /// Returns `true` when this is a container format (wraps A/V streams).
    pub fn is_container(&self) -> bool {
        matches!(
            self,
            Self::Mxf | Self::Mkv | Self::Mp4 | Self::Ts | Self::Mov
        )
    }

    /// Returns the conventional file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mxf => "mxf",
            Self::Mkv => "mkv",
            Self::Mp4 => "mp4",
            Self::Ts => "ts",
            Self::Mov => "mov",
            Self::Bxf => "xml",
            Self::TextList => "txt",
        }
    }

    /// Try to infer the format from a file extension string (case-insensitive).
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.trim().to_lowercase().as_str() {
            "mxf" => Some(Self::Mxf),
            "mkv" => Some(Self::Mkv),
            "mp4" | "m4v" => Some(Self::Mp4),
            "ts" | "m2ts" => Some(Self::Ts),
            "mov" => Some(Self::Mov),
            "xml" => Some(Self::Bxf),
            "txt" => Some(Self::TextList),
            _ => None,
        }
    }
}

// ── PlaylistItem ──────────────────────────────────────────────────────────────

/// A single content item queued for playout ingest.
#[derive(Debug, Clone)]
pub struct PlaylistItem {
    /// Unique identifier for this item within the ingest session.
    pub id: String,
    /// File system path to the source media.
    pub path: PathBuf,
    /// Detected or specified ingest format.
    pub format: IngestFormat,
    /// Clip start offset in milliseconds (for sub-clip ingest).
    pub in_point_ms: u64,
    /// Clip end offset in milliseconds.  `0` means use the full duration.
    pub out_point_ms: u64,
    /// Optional human-readable title.
    pub title: Option<String>,
    /// Whether this item has passed the pre-flight validation check.
    pub validated: bool,
}

impl PlaylistItem {
    /// Create a new playlist item.
    pub fn new(id: impl Into<String>, path: impl Into<PathBuf>, format: IngestFormat) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            format,
            in_point_ms: 0,
            out_point_ms: 0,
            title: None,
            validated: false,
        }
    }

    /// Set in/out points.
    pub fn with_points(mut self, in_ms: u64, out_ms: u64) -> Self {
        self.in_point_ms = in_ms;
        self.out_point_ms = out_ms;
        self
    }

    /// Attach a title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Checks basic validity of the item:
    /// - Path must not be empty.
    /// - If `out_point_ms > 0`, it must be greater than `in_point_ms`.
    pub fn is_valid(&self) -> bool {
        let path_ok = !self.path.as_os_str().is_empty();
        let points_ok = self.out_point_ms == 0 || self.out_point_ms > self.in_point_ms;
        path_ok && points_ok
    }

    /// Duration of the trimmed clip in milliseconds.
    /// Returns `0` when no out-point is set (full duration).
    pub fn clip_duration_ms(&self) -> u64 {
        self.out_point_ms.saturating_sub(self.in_point_ms)
    }

    /// Mark the item as validated.
    pub fn mark_validated(&mut self) {
        self.validated = true;
    }
}

// ── PlaylistIngest ─────────────────────────────────────────────────────────────

/// An ordered list of items to be ingested and scheduled for playout.
pub struct PlaylistIngest {
    /// Ingest session identifier.
    pub session_id: String,
    items: Vec<PlaylistItem>,
}

impl PlaylistIngest {
    /// Create a new ingest session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            items: Vec::new(),
        }
    }

    /// Append an item to the ingest queue.
    pub fn add_item(&mut self, item: PlaylistItem) {
        self.items.push(item);
    }

    /// Total number of items in the session.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` when there are no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Run a basic validation pass over all items.
    ///
    /// Returns a list of IDs that failed validation.
    pub fn validate(&self) -> Vec<String> {
        self.items
            .iter()
            .filter(|item| !item.is_valid())
            .map(|item| item.id.clone())
            .collect()
    }

    /// Find an item by ID.
    pub fn find(&self, id: &str) -> Option<&PlaylistItem> {
        self.items.iter().find(|i| i.id == id)
    }

    /// Find a mutable item by ID.
    pub fn find_mut(&mut self, id: &str) -> Option<&mut PlaylistItem> {
        self.items.iter_mut().find(|i| i.id == id)
    }

    /// Remove an item by ID, returning it if present.
    pub fn remove(&mut self, id: &str) -> Option<PlaylistItem> {
        if let Some(pos) = self.items.iter().position(|i| i.id == id) {
            Some(self.items.remove(pos))
        } else {
            None
        }
    }

    /// Count items that have been successfully validated.
    pub fn validated_count(&self) -> usize {
        self.items.iter().filter(|i| i.validated).count()
    }

    /// Count items by format type.
    pub fn count_by_format(&self, format: &IngestFormat) -> usize {
        self.items.iter().filter(|i| &i.format == format).count()
    }

    /// All items in ingest order.
    pub fn items(&self) -> &[PlaylistItem] {
        &self.items
    }

    /// Total duration of all trimmed clips (items with explicit out-points).
    pub fn total_clip_duration_ms(&self) -> u64 {
        self.items.iter().map(PlaylistItem::clip_duration_ms).sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mxf_item(id: &str) -> PlaylistItem {
        PlaylistItem::new(id, format!("/media/{id}.mxf"), IngestFormat::Mxf)
    }

    fn mp4_item(id: &str) -> PlaylistItem {
        PlaylistItem::new(id, format!("/media/{id}.mp4"), IngestFormat::Mp4)
    }

    // IngestFormat

    #[test]
    fn format_supports_metadata_mxf() {
        assert!(IngestFormat::Mxf.supports_metadata());
        assert!(IngestFormat::Mp4.supports_metadata());
        assert!(!IngestFormat::Ts.supports_metadata());
        assert!(!IngestFormat::TextList.supports_metadata());
    }

    #[test]
    fn format_is_container() {
        assert!(IngestFormat::Mxf.is_container());
        assert!(IngestFormat::Ts.is_container());
        assert!(!IngestFormat::Bxf.is_container());
        assert!(!IngestFormat::TextList.is_container());
    }

    #[test]
    fn format_extension() {
        assert_eq!(IngestFormat::Mxf.extension(), "mxf");
        assert_eq!(IngestFormat::Mp4.extension(), "mp4");
        assert_eq!(IngestFormat::Bxf.extension(), "xml");
    }

    #[test]
    fn format_from_extension_known() {
        assert_eq!(IngestFormat::from_extension("mxf"), Some(IngestFormat::Mxf));
        assert_eq!(IngestFormat::from_extension("MKV"), Some(IngestFormat::Mkv));
        assert_eq!(IngestFormat::from_extension("m2ts"), Some(IngestFormat::Ts));
        assert_eq!(IngestFormat::from_extension("m4v"), Some(IngestFormat::Mp4));
    }

    #[test]
    fn format_from_extension_unknown() {
        assert_eq!(IngestFormat::from_extension("avi"), None);
    }

    // PlaylistItem

    #[test]
    fn item_is_valid_basic() {
        let item = mxf_item("i1");
        assert!(item.is_valid());
    }

    #[test]
    fn item_is_valid_with_good_points() {
        let item = mxf_item("i1").with_points(1000, 5000);
        assert!(item.is_valid());
    }

    #[test]
    fn item_is_invalid_bad_points() {
        let item = mxf_item("i1").with_points(5000, 1000);
        assert!(!item.is_valid());
    }

    #[test]
    fn item_clip_duration_with_points() {
        let item = mxf_item("i1").with_points(1000, 4000);
        assert_eq!(item.clip_duration_ms(), 3000);
    }

    #[test]
    fn item_clip_duration_no_out_point() {
        let item = mxf_item("i1");
        assert_eq!(item.clip_duration_ms(), 0);
    }

    #[test]
    fn item_mark_validated() {
        let mut item = mxf_item("i1");
        assert!(!item.validated);
        item.mark_validated();
        assert!(item.validated);
    }

    #[test]
    fn item_with_title() {
        let item = mxf_item("i1").with_title("News Bulletin");
        assert_eq!(item.title.as_deref(), Some("News Bulletin"));
    }

    // PlaylistIngest

    #[test]
    fn ingest_add_and_count() {
        let mut ingest = PlaylistIngest::new("session1");
        assert!(ingest.is_empty());
        ingest.add_item(mxf_item("i1"));
        ingest.add_item(mp4_item("i2"));
        assert_eq!(ingest.item_count(), 2);
    }

    #[test]
    fn ingest_validate_all_valid() {
        let mut ingest = PlaylistIngest::new("session1");
        ingest.add_item(mxf_item("i1"));
        ingest.add_item(mxf_item("i2").with_points(0, 3000));
        let failures = ingest.validate();
        assert!(failures.is_empty(), "unexpected failures: {failures:?}");
    }

    #[test]
    fn ingest_validate_catches_invalid() {
        let mut ingest = PlaylistIngest::new("session1");
        ingest.add_item(mxf_item("i1")); // valid
        ingest.add_item(mxf_item("bad").with_points(5000, 1000)); // out < in
        let failures = ingest.validate();
        assert_eq!(failures, vec!["bad".to_string()]);
    }

    #[test]
    fn ingest_find_item() {
        let mut ingest = PlaylistIngest::new("s");
        ingest.add_item(mxf_item("i1"));
        assert!(ingest.find("i1").is_some());
        assert!(ingest.find("i99").is_none());
    }

    #[test]
    fn ingest_remove_item() {
        let mut ingest = PlaylistIngest::new("s");
        ingest.add_item(mxf_item("i1"));
        let removed = ingest.remove("i1");
        assert!(removed.is_some());
        assert_eq!(ingest.item_count(), 0);
    }

    #[test]
    fn ingest_validated_count() {
        let mut ingest = PlaylistIngest::new("s");
        ingest.add_item(mxf_item("i1"));
        ingest.add_item(mxf_item("i2"));
        ingest
            .find_mut("i1")
            .expect("should succeed in test")
            .mark_validated();
        assert_eq!(ingest.validated_count(), 1);
    }

    #[test]
    fn ingest_count_by_format() {
        let mut ingest = PlaylistIngest::new("s");
        ingest.add_item(mxf_item("i1"));
        ingest.add_item(mxf_item("i2"));
        ingest.add_item(mp4_item("i3"));
        assert_eq!(ingest.count_by_format(&IngestFormat::Mxf), 2);
        assert_eq!(ingest.count_by_format(&IngestFormat::Mp4), 1);
        assert_eq!(ingest.count_by_format(&IngestFormat::Mkv), 0);
    }

    #[test]
    fn ingest_total_clip_duration() {
        let mut ingest = PlaylistIngest::new("s");
        ingest.add_item(mxf_item("i1").with_points(0, 2000));
        ingest.add_item(mxf_item("i2").with_points(0, 3000));
        ingest.add_item(mxf_item("i3")); // no out-point → 0
        assert_eq!(ingest.total_clip_duration_ms(), 5000);
    }
}
