//! In-memory media catalogue for the MAM system.
//!
//! Provides a flat catalogue of media records with creation/update
//! timestamps, format metadata, and basic query helpers.

#![allow(dead_code)]

/// High-level media type classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MediaType {
    /// Video asset (MP4, MXF, MOV, …).
    Video,
    /// Audio asset (WAV, FLAC, …).
    Audio,
    /// Still image (JPEG, TIFF, …).
    Image,
    /// Document (PDF, DOCX, …).
    Document,
    /// Unrecognised format.
    Unknown,
}

impl MediaType {
    /// Returns the string label for this media type.
    pub fn label(&self) -> &'static str {
        match self {
            MediaType::Video => "video",
            MediaType::Audio => "audio",
            MediaType::Image => "image",
            MediaType::Document => "document",
            MediaType::Unknown => "unknown",
        }
    }
}

/// A catalogue record for a single media asset.
#[derive(Debug, Clone)]
pub struct CatalogRecord {
    /// Unique asset identifier.
    pub asset_id: String,
    /// Human-readable title.
    pub title: String,
    /// Media type classification.
    pub media_type: MediaType,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Duration in milliseconds (for time-based media).
    pub duration_ms: Option<u64>,
    /// Format / container string (e.g. "video/mp4").
    pub format: String,
    /// Unix epoch timestamp (ms) when the record was created.
    pub created_at_ms: u64,
    /// Unix epoch timestamp (ms) of the last update to this record.
    pub updated_at_ms: u64,
    /// Whether the asset has been soft-deleted.
    pub deleted: bool,
}

impl CatalogRecord {
    /// Creates a new catalogue record with `created_at_ms == updated_at_ms`.
    pub fn new(
        asset_id: impl Into<String>,
        title: impl Into<String>,
        media_type: MediaType,
        file_size_bytes: u64,
        format: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            title: title.into(),
            media_type,
            file_size_bytes,
            duration_ms: None,
            format: format.into(),
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            deleted: false,
        }
    }

    /// Attaches a duration to a time-based asset.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Returns `true` when this is a video or audio asset (time-based).
    pub fn is_time_based(&self) -> bool {
        matches!(self.media_type, MediaType::Video | MediaType::Audio)
    }

    /// Returns file size in mebibytes.
    pub fn size_mib(&self) -> f64 {
        self.file_size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Marks the record as deleted and updates the timestamp.
    pub fn soft_delete(&mut self, now_ms: u64) {
        self.deleted = true;
        self.updated_at_ms = now_ms;
    }

    /// Updates the title and stamps the record.
    pub fn update_title(&mut self, new_title: impl Into<String>, now_ms: u64) {
        self.title = new_title.into();
        self.updated_at_ms = now_ms;
    }
}

/// Criteria for querying the catalogue.
#[derive(Debug, Clone, Default)]
pub struct CatalogQuery {
    /// Restrict to the given media type.
    pub media_type: Option<MediaType>,
    /// Title must contain this substring (case-insensitive).
    pub title_contains: Option<String>,
    /// Minimum file size in bytes.
    pub min_size_bytes: Option<u64>,
    /// Maximum file size in bytes.
    pub max_size_bytes: Option<u64>,
    /// Include soft-deleted records.
    pub include_deleted: bool,
}

impl CatalogQuery {
    /// Creates an empty query (matches all active records).
    pub fn new() -> Self {
        Self::default()
    }

    /// Tests whether a record satisfies the query.
    pub fn matches(&self, r: &CatalogRecord) -> bool {
        if !self.include_deleted && r.deleted {
            return false;
        }
        if let Some(ref mt) = self.media_type {
            if r.media_type != *mt {
                return false;
            }
        }
        if let Some(ref substr) = self.title_contains {
            if !r.title.to_lowercase().contains(&substr.to_lowercase()) {
                return false;
            }
        }
        if let Some(min) = self.min_size_bytes {
            if r.file_size_bytes < min {
                return false;
            }
        }
        if let Some(max) = self.max_size_bytes {
            if r.file_size_bytes > max {
                return false;
            }
        }
        true
    }
}

/// Storage statistics derived from the catalogue.
#[derive(Debug, Clone, Default)]
pub struct CatalogStats {
    /// Total active records.
    pub total_records: usize,
    /// Total bytes across all active assets.
    pub total_bytes: u64,
    /// Number of video assets.
    pub video_count: usize,
    /// Number of audio assets.
    pub audio_count: usize,
    /// Number of image assets.
    pub image_count: usize,
    /// Number of soft-deleted records.
    pub deleted_count: usize,
}

/// An in-memory catalogue of media records.
#[derive(Debug, Default)]
pub struct MediaCatalog {
    records: Vec<CatalogRecord>,
}

impl MediaCatalog {
    /// Creates an empty catalogue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a new record.  Replaces an existing record with the same asset ID.
    pub fn insert(&mut self, record: CatalogRecord) {
        if let Some(pos) = self
            .records
            .iter()
            .position(|r| r.asset_id == record.asset_id)
        {
            self.records[pos] = record;
        } else {
            self.records.push(record);
        }
    }

    /// Returns a reference to the record with the given asset ID, if present.
    pub fn get(&self, asset_id: &str) -> Option<&CatalogRecord> {
        self.records.iter().find(|r| r.asset_id == asset_id)
    }

    /// Returns a mutable reference to the record with the given asset ID.
    pub fn get_mut(&mut self, asset_id: &str) -> Option<&mut CatalogRecord> {
        self.records.iter_mut().find(|r| r.asset_id == asset_id)
    }

    /// Hard-removes the record for `asset_id`.
    ///
    /// Returns `true` if a record was removed.
    pub fn remove(&mut self, asset_id: &str) -> bool {
        let before = self.records.len();
        self.records.retain(|r| r.asset_id != asset_id);
        self.records.len() < before
    }

    /// Queries the catalogue and returns matching records sorted by `created_at_ms`.
    pub fn query(&self, q: &CatalogQuery) -> Vec<&CatalogRecord> {
        let mut results: Vec<&CatalogRecord> =
            self.records.iter().filter(|r| q.matches(r)).collect();
        results.sort_by_key(|r| r.created_at_ms);
        results
    }

    /// Returns the total number of records (including deleted).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` when the catalogue has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Computes aggregate statistics for the catalogue.
    pub fn stats(&self) -> CatalogStats {
        let mut s = CatalogStats::default();
        for r in &self.records {
            if r.deleted {
                s.deleted_count += 1;
                continue;
            }
            s.total_records += 1;
            s.total_bytes += r.file_size_bytes;
            match r.media_type {
                MediaType::Video => s.video_count += 1,
                MediaType::Audio => s.audio_count += 1,
                MediaType::Image => s.image_count += 1,
                _ => {}
            }
        }
        s
    }

    /// Returns the top N records by file size (largest first).
    pub fn largest(&self, n: usize) -> Vec<&CatalogRecord> {
        let mut active: Vec<&CatalogRecord> = self.records.iter().filter(|r| !r.deleted).collect();
        active.sort_by(|a, b| b.file_size_bytes.cmp(&a.file_size_bytes));
        active.truncate(n);
        active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(id: &str, title: &str, size: u64, now: u64) -> CatalogRecord {
        CatalogRecord::new(id, title, MediaType::Video, size, "video/mp4", now)
    }

    fn populated_catalog() -> MediaCatalog {
        let mut cat = MediaCatalog::new();
        cat.insert(video("v1", "Breaking News", 500_000, 100));
        cat.insert(video("v2", "Sports Highlights", 1_000_000, 200));
        cat.insert(CatalogRecord::new(
            "i1",
            "Thumbnail",
            MediaType::Image,
            50_000,
            "image/jpeg",
            150,
        ));
        cat
    }

    // --- MediaType ---

    #[test]
    fn test_media_type_label() {
        assert_eq!(MediaType::Video.label(), "video");
        assert_eq!(MediaType::Audio.label(), "audio");
        assert_eq!(MediaType::Image.label(), "image");
        assert_eq!(MediaType::Unknown.label(), "unknown");
    }

    // --- CatalogRecord ---

    #[test]
    fn test_record_is_time_based() {
        let v = video("v1", "Test", 100, 0);
        assert!(v.is_time_based());
        let img = CatalogRecord::new("i1", "Img", MediaType::Image, 100, "image/jpeg", 0);
        assert!(!img.is_time_based());
    }

    #[test]
    fn test_record_size_mib() {
        let r = video("v1", "Test", 1_048_576, 0);
        assert!((r.size_mib() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_record_soft_delete() {
        let mut r = video("v1", "Test", 1000, 100);
        r.soft_delete(200);
        assert!(r.deleted);
        assert_eq!(r.updated_at_ms, 200);
    }

    #[test]
    fn test_record_update_title() {
        let mut r = video("v1", "Old Title", 1000, 100);
        r.update_title("New Title", 300);
        assert_eq!(r.title, "New Title");
        assert_eq!(r.updated_at_ms, 300);
    }

    #[test]
    fn test_record_with_duration() {
        let r = video("v1", "Test", 1000, 0).with_duration(120_000);
        assert_eq!(r.duration_ms, Some(120_000));
    }

    // --- CatalogQuery ---

    #[test]
    fn test_query_by_media_type() {
        let cat = populated_catalog();
        let mut q = CatalogQuery::new();
        q.media_type = Some(MediaType::Video);
        let results = cat.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_title_contains() {
        let cat = populated_catalog();
        let mut q = CatalogQuery::new();
        q.title_contains = Some("news".to_string());
        let results = cat.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "v1");
    }

    #[test]
    fn test_query_excludes_deleted_by_default() {
        let mut cat = populated_catalog();
        cat.get_mut("v1")
            .expect("should succeed in test")
            .soft_delete(500);
        let results = cat.query(&CatalogQuery::new());
        assert!(results.iter().all(|r| r.asset_id != "v1"));
    }

    #[test]
    fn test_query_include_deleted() {
        let mut cat = populated_catalog();
        cat.get_mut("v1")
            .expect("should succeed in test")
            .soft_delete(500);
        let mut q = CatalogQuery::new();
        q.include_deleted = true;
        let results = cat.query(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_size_range() {
        let cat = populated_catalog();
        let mut q = CatalogQuery::new();
        q.min_size_bytes = Some(400_000);
        q.max_size_bytes = Some(600_000);
        let results = cat.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, "v1");
    }

    // --- MediaCatalog ---

    #[test]
    fn test_insert_and_get() {
        let mut cat = MediaCatalog::new();
        cat.insert(video("v1", "Test", 100, 0));
        assert!(cat.get("v1").is_some());
    }

    #[test]
    fn test_insert_replaces_existing() {
        let mut cat = MediaCatalog::new();
        cat.insert(video("v1", "Old", 100, 0));
        cat.insert(video("v1", "New", 200, 10));
        assert_eq!(cat.len(), 1);
        assert_eq!(cat.get("v1").expect("should succeed in test").title, "New");
    }

    #[test]
    fn test_remove_existing() {
        let mut cat = populated_catalog();
        assert!(cat.remove("v1"));
        assert!(cat.get("v1").is_none());
    }

    #[test]
    fn test_remove_missing_returns_false() {
        let mut cat = MediaCatalog::new();
        assert!(!cat.remove("ghost"));
    }

    #[test]
    fn test_stats_aggregation() {
        let cat = populated_catalog();
        let s = cat.stats();
        assert_eq!(s.total_records, 3);
        assert_eq!(s.video_count, 2);
        assert_eq!(s.image_count, 1);
        assert_eq!(s.deleted_count, 0);
    }

    #[test]
    fn test_stats_excludes_deleted() {
        let mut cat = populated_catalog();
        cat.get_mut("v1")
            .expect("should succeed in test")
            .soft_delete(1000);
        let s = cat.stats();
        assert_eq!(s.total_records, 2);
        assert_eq!(s.deleted_count, 1);
    }

    #[test]
    fn test_largest_returns_sorted() {
        let cat = populated_catalog();
        let top = cat.largest(2);
        assert!(top[0].file_size_bytes >= top[1].file_size_bytes);
    }
}
