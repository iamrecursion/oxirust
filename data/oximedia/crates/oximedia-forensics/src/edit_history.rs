#![allow(dead_code)]
//! Edit history reconstruction for forensic analysis.
//!
//! Models the sequence of operations that may have been applied to a media
//! asset, distinguishing between destructive and non-destructive changes.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A discrete operation that may have been performed on a media asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOperation {
    /// Crop or trim – removes pixels/frames permanently.
    Crop,
    /// Colour grading or LUT application.
    ColorGrade,
    /// Audio normalisation.
    AudioNormalize,
    /// Re-encoding with a (possibly lossy) codec.
    Transcode,
    /// Upscaling or downscaling.
    Resize,
    /// Rotation or flip.
    Rotate,
    /// Overlay composite (watermark, logo, subtitle burn-in).
    Overlay,
    /// Metadata-only change (EXIF, XMP, etc.).
    MetadataEdit,
    /// Unknown or unrecognised operation.
    Unknown(String),
}

impl EditOperation {
    /// Returns `true` for operations that permanently alter the original content.
    pub fn is_destructive(&self) -> bool {
        matches!(
            self,
            EditOperation::Crop | EditOperation::Transcode | EditOperation::Overlay
        )
    }

    /// Short display name.
    pub fn name(&self) -> &str {
        match self {
            EditOperation::Crop => "Crop",
            EditOperation::ColorGrade => "ColorGrade",
            EditOperation::AudioNormalize => "AudioNormalize",
            EditOperation::Transcode => "Transcode",
            EditOperation::Resize => "Resize",
            EditOperation::Rotate => "Rotate",
            EditOperation::Overlay => "Overlay",
            EditOperation::MetadataEdit => "MetadataEdit",
            EditOperation::Unknown(s) => s.as_str(),
        }
    }
}

/// A single entry in the reconstructed edit history.
#[derive(Debug, Clone)]
pub struct EditHistoryEntry {
    /// The operation performed.
    pub operation: EditOperation,
    /// Unix timestamp in seconds (estimated or extracted from metadata).
    pub timestamp_secs: u64,
    /// Name of the software tool that performed the edit (if known).
    pub tool: Option<String>,
    /// Extra notes about this edit.
    pub notes: String,
    /// Confidence that this entry is accurate (0.0–1.0).
    pub confidence: f64,
}

impl EditHistoryEntry {
    /// Create a new entry.
    pub fn new(operation: EditOperation, timestamp_secs: u64) -> Self {
        Self {
            operation,
            timestamp_secs,
            tool: None,
            notes: String::new(),
            confidence: 1.0,
        }
    }

    /// Attach a tool name.
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }

    /// Attach notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Human-readable description of this history entry.
    pub fn description(&self) -> String {
        let tool_str = self
            .tool
            .as_deref()
            .map(|t| format!(" by {t}"))
            .unwrap_or_default();
        let destr = if self.operation.is_destructive() {
            " [DESTRUCTIVE]"
        } else {
            ""
        };
        format!(
            "t={}: {}{}{}{}",
            self.timestamp_secs,
            self.operation.name(),
            tool_str,
            destr,
            if self.notes.is_empty() {
                String::new()
            } else {
                format!(" – {}", self.notes)
            }
        )
    }
}

/// A chronologically ordered sequence of edit operations on an asset.
#[derive(Debug, Default)]
pub struct EditHistory {
    entries: Vec<EditHistoryEntry>,
}

impl EditHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry.  Entries need not be inserted in order;
    /// `timeline()` will sort them.
    pub fn add(&mut self, entry: EditHistoryEntry) {
        self.entries.push(entry);
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return only the destructive operations.
    pub fn destructive_ops(&self) -> Vec<&EditHistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.operation.is_destructive())
            .collect()
    }

    /// Return all entries sorted ascending by timestamp.
    pub fn timeline(&self) -> Vec<&EditHistoryEntry> {
        let mut sorted: Vec<&EditHistoryEntry> = self.entries.iter().collect();
        sorted.sort_by_key(|e| e.timestamp_secs);
        sorted
    }

    /// Whether any destructive operation is in the history.
    pub fn has_destructive_ops(&self) -> bool {
        self.entries.iter().any(|e| e.operation.is_destructive())
    }

    /// The earliest timestamp in the history (or `None` if empty).
    pub fn earliest_timestamp(&self) -> Option<u64> {
        self.entries.iter().map(|e| e.timestamp_secs).min()
    }

    /// The latest timestamp in the history (or `None` if empty).
    pub fn latest_timestamp(&self) -> Option<u64> {
        self.entries.iter().map(|e| e.timestamp_secs).max()
    }

    /// Duration spanned by the history (latest - earliest).
    pub fn span(&self) -> Option<Duration> {
        let earliest = self.earliest_timestamp()?;
        let latest = self.latest_timestamp()?;
        Some(Duration::from_secs(latest.saturating_sub(earliest)))
    }

    /// Mean confidence across all entries.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_confidence(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        self.entries.iter().map(|e| e.confidence).sum::<f64>() / self.entries.len() as f64
    }

    /// Produce a plain-text report.
    pub fn report(&self) -> String {
        let mut out = format!("Edit history ({} entries):\n", self.entries.len());
        for entry in self.timeline() {
            out.push_str("  ");
            out.push_str(&entry.description());
            out.push('\n');
        }
        out
    }
}

/// Convenience: get a plausible "now" unix timestamp for tests.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_operation_crop_is_destructive() {
        assert!(EditOperation::Crop.is_destructive());
    }

    #[test]
    fn test_edit_operation_transcode_is_destructive() {
        assert!(EditOperation::Transcode.is_destructive());
    }

    #[test]
    fn test_edit_operation_overlay_is_destructive() {
        assert!(EditOperation::Overlay.is_destructive());
    }

    #[test]
    fn test_edit_operation_color_grade_not_destructive() {
        assert!(!EditOperation::ColorGrade.is_destructive());
    }

    #[test]
    fn test_edit_operation_metadata_not_destructive() {
        assert!(!EditOperation::MetadataEdit.is_destructive());
    }

    #[test]
    fn test_edit_operation_names() {
        assert_eq!(EditOperation::Crop.name(), "Crop");
        assert_eq!(EditOperation::Transcode.name(), "Transcode");
        assert_eq!(EditOperation::Unknown("foo".to_string()).name(), "foo");
    }

    #[test]
    fn test_history_entry_description_contains_operation() {
        let e = EditHistoryEntry::new(EditOperation::Crop, 1_000_000)
            .with_tool("Resolve")
            .with_notes("removed first 30 frames");
        let desc = e.description();
        assert!(desc.contains("Crop"));
        assert!(desc.contains("Resolve"));
        assert!(desc.contains("DESTRUCTIVE"));
        assert!(desc.contains("removed first 30 frames"));
    }

    #[test]
    fn test_history_entry_non_destructive_description() {
        let e = EditHistoryEntry::new(EditOperation::MetadataEdit, 2_000_000);
        let desc = e.description();
        assert!(!desc.contains("DESTRUCTIVE"));
    }

    #[test]
    fn test_history_add_and_len() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Crop, 100));
        h.add(EditHistoryEntry::new(EditOperation::ColorGrade, 200));
        assert_eq!(h.len(), 2);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_history_destructive_ops() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Crop, 100));
        h.add(EditHistoryEntry::new(EditOperation::ColorGrade, 200));
        h.add(EditHistoryEntry::new(EditOperation::Transcode, 300));
        let d = h.destructive_ops();
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_history_has_destructive_ops_false() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Resize, 100));
        assert!(!h.has_destructive_ops());
    }

    #[test]
    fn test_history_timeline_sorted() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Transcode, 300));
        h.add(EditHistoryEntry::new(EditOperation::Crop, 100));
        h.add(EditHistoryEntry::new(EditOperation::ColorGrade, 200));
        let timeline = h.timeline();
        assert_eq!(timeline[0].timestamp_secs, 100);
        assert_eq!(timeline[1].timestamp_secs, 200);
        assert_eq!(timeline[2].timestamp_secs, 300);
    }

    #[test]
    fn test_history_earliest_latest() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Crop, 500));
        h.add(EditHistoryEntry::new(EditOperation::Resize, 100));
        h.add(EditHistoryEntry::new(EditOperation::Rotate, 300));
        assert_eq!(h.earliest_timestamp(), Some(100));
        assert_eq!(h.latest_timestamp(), Some(500));
    }

    #[test]
    fn test_history_span() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Crop, 1000));
        h.add(EditHistoryEntry::new(EditOperation::Transcode, 4600));
        assert_eq!(h.span(), Some(Duration::from_hours(1)));
    }

    #[test]
    fn test_history_mean_confidence() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Crop, 100).with_confidence(0.8));
        h.add(EditHistoryEntry::new(EditOperation::ColorGrade, 200).with_confidence(0.6));
        let mean = h.mean_confidence();
        assert!((mean - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_history_report_nonempty() {
        let mut h = EditHistory::new();
        h.add(EditHistoryEntry::new(EditOperation::Transcode, 1000).with_tool("ffmpeg"));
        let report = h.report();
        assert!(report.contains("1 entries"));
        assert!(report.contains("Transcode"));
    }

    #[test]
    fn test_history_empty_defaults() {
        let h = EditHistory::new();
        assert!(h.is_empty());
        assert!(h.earliest_timestamp().is_none());
        assert!(h.latest_timestamp().is_none());
        assert!(h.span().is_none());
        assert_eq!(h.mean_confidence(), 0.0);
    }
}
