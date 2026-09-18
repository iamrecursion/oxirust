#![allow(dead_code)]
//! Conform diff engine for comparing timeline versions.
//!
//! This module provides tools for computing differences between two conform
//! sessions or timeline versions, identifying added, removed, modified, and
//! moved clips for version control and change tracking workflows.

use std::collections::HashMap;
use std::fmt;

/// Type of change detected between two timeline versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffChangeType {
    /// Clip was added in the new version.
    Added,
    /// Clip was removed from the new version.
    Removed,
    /// Clip was modified (timing, source, etc.).
    Modified,
    /// Clip was moved to a different position.
    Moved,
    /// Clip is unchanged.
    Unchanged,
}

impl fmt::Display for DiffChangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Added => write!(f, "ADDED"),
            Self::Removed => write!(f, "REMOVED"),
            Self::Modified => write!(f, "MODIFIED"),
            Self::Moved => write!(f, "MOVED"),
            Self::Unchanged => write!(f, "UNCHANGED"),
        }
    }
}

/// What aspect of a clip was modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModifiedField {
    /// Source file path changed.
    SourcePath,
    /// Timeline in point changed.
    TimelineIn,
    /// Timeline out point changed.
    TimelineOut,
    /// Source in point changed.
    SourceIn,
    /// Source out point changed.
    SourceOut,
    /// Track assignment changed.
    Track,
    /// Speed/retime changed.
    Speed,
    /// Transition changed.
    Transition,
    /// Effects changed.
    Effects,
}

impl fmt::Display for ModifiedField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourcePath => write!(f, "source_path"),
            Self::TimelineIn => write!(f, "timeline_in"),
            Self::TimelineOut => write!(f, "timeline_out"),
            Self::SourceIn => write!(f, "source_in"),
            Self::SourceOut => write!(f, "source_out"),
            Self::Track => write!(f, "track"),
            Self::Speed => write!(f, "speed"),
            Self::Transition => write!(f, "transition"),
            Self::Effects => write!(f, "effects"),
        }
    }
}

/// A simplified clip representation for diff comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffClip {
    /// Unique clip identifier.
    pub clip_id: String,
    /// Source file path.
    pub source_path: String,
    /// Timeline in point in frames.
    pub timeline_in: i64,
    /// Timeline out point in frames.
    pub timeline_out: i64,
    /// Source in point in frames.
    pub source_in: i64,
    /// Source out point in frames.
    pub source_out: i64,
    /// Track index.
    pub track_index: usize,
    /// Speed factor (1.0 = normal).
    pub speed: f64,
    /// Transition type (if any).
    pub transition: Option<String>,
    /// Active effects list.
    pub effects: Vec<String>,
}

impl DiffClip {
    /// Create a new diff clip.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clip_id: impl Into<String>,
        source_path: impl Into<String>,
        timeline_in: i64,
        timeline_out: i64,
        source_in: i64,
        source_out: i64,
        track_index: usize,
        speed: f64,
    ) -> Self {
        Self {
            clip_id: clip_id.into(),
            source_path: source_path.into(),
            timeline_in,
            timeline_out,
            source_in,
            source_out,
            track_index,
            speed,
            transition: None,
            effects: Vec::new(),
        }
    }

    /// Get the duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> i64 {
        self.timeline_out - self.timeline_in
    }

    /// Compare fields with another clip and return which fields differ.
    #[must_use]
    pub fn diff_fields(&self, other: &Self) -> Vec<ModifiedField> {
        let mut changes = Vec::new();

        if self.source_path != other.source_path {
            changes.push(ModifiedField::SourcePath);
        }
        if self.timeline_in != other.timeline_in {
            changes.push(ModifiedField::TimelineIn);
        }
        if self.timeline_out != other.timeline_out {
            changes.push(ModifiedField::TimelineOut);
        }
        if self.source_in != other.source_in {
            changes.push(ModifiedField::SourceIn);
        }
        if self.source_out != other.source_out {
            changes.push(ModifiedField::SourceOut);
        }
        if self.track_index != other.track_index {
            changes.push(ModifiedField::Track);
        }
        if (self.speed - other.speed).abs() > 0.001 {
            changes.push(ModifiedField::Speed);
        }
        if self.transition != other.transition {
            changes.push(ModifiedField::Transition);
        }
        if self.effects != other.effects {
            changes.push(ModifiedField::Effects);
        }

        changes
    }
}

/// A single change record in the diff.
#[derive(Debug, Clone)]
pub struct DiffEntry {
    /// Type of change.
    pub change_type: DiffChangeType,
    /// Clip ID.
    pub clip_id: String,
    /// The clip in the old version (if present).
    pub old_clip: Option<DiffClip>,
    /// The clip in the new version (if present).
    pub new_clip: Option<DiffClip>,
    /// Which fields were modified (for Modified type).
    pub modified_fields: Vec<ModifiedField>,
}

impl DiffEntry {
    /// Create an "added" entry.
    #[must_use]
    pub fn added(clip: DiffClip) -> Self {
        Self {
            change_type: DiffChangeType::Added,
            clip_id: clip.clip_id.clone(),
            old_clip: None,
            new_clip: Some(clip),
            modified_fields: Vec::new(),
        }
    }

    /// Create a "removed" entry.
    #[must_use]
    pub fn removed(clip: DiffClip) -> Self {
        Self {
            change_type: DiffChangeType::Removed,
            clip_id: clip.clip_id.clone(),
            old_clip: Some(clip),
            new_clip: None,
            modified_fields: Vec::new(),
        }
    }

    /// Create a "modified" entry.
    #[must_use]
    pub fn modified(old: DiffClip, new: DiffClip, fields: Vec<ModifiedField>) -> Self {
        Self {
            change_type: DiffChangeType::Modified,
            clip_id: old.clip_id.clone(),
            old_clip: Some(old),
            new_clip: Some(new),
            modified_fields: fields,
        }
    }

    /// Create an "unchanged" entry.
    #[must_use]
    pub fn unchanged(clip: DiffClip) -> Self {
        Self {
            change_type: DiffChangeType::Unchanged,
            clip_id: clip.clip_id.clone(),
            old_clip: Some(clip.clone()),
            new_clip: Some(clip),
            modified_fields: Vec::new(),
        }
    }

    /// Format as a human-readable summary line.
    #[must_use]
    pub fn summary_line(&self) -> String {
        match self.change_type {
            DiffChangeType::Added => format!("+ [{}] added", self.clip_id),
            DiffChangeType::Removed => format!("- [{}] removed", self.clip_id),
            DiffChangeType::Modified => {
                let fields: Vec<String> = self
                    .modified_fields
                    .iter()
                    .map(|f| format!("{f}"))
                    .collect();
                format!("~ [{}] modified: {}", self.clip_id, fields.join(", "))
            }
            DiffChangeType::Moved => format!("> [{}] moved", self.clip_id),
            DiffChangeType::Unchanged => format!("  [{}] unchanged", self.clip_id),
        }
    }
}

/// Statistics for a diff operation.
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    /// Total clips in old version.
    pub old_total: usize,
    /// Total clips in new version.
    pub new_total: usize,
    /// Number of added clips.
    pub added: usize,
    /// Number of removed clips.
    pub removed: usize,
    /// Number of modified clips.
    pub modified: usize,
    /// Number of moved clips.
    pub moved: usize,
    /// Number of unchanged clips.
    pub unchanged: usize,
    /// Count of changes by field type.
    pub field_changes: HashMap<ModifiedField, usize>,
}

impl DiffStats {
    /// Calculate a similarity score (0.0 - 1.0) between the two versions.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn similarity(&self) -> f64 {
        let total = self.old_total.max(self.new_total);
        if total == 0 {
            return 1.0;
        }
        self.unchanged as f64 / total as f64
    }

    /// Check if the two versions are identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.added == 0 && self.removed == 0 && self.modified == 0 && self.moved == 0
    }
}

/// The diff report containing all changes.
#[derive(Debug, Clone)]
pub struct DiffReport {
    /// Label for the old version.
    pub old_label: String,
    /// Label for the new version.
    pub new_label: String,
    /// All diff entries.
    pub entries: Vec<DiffEntry>,
    /// Aggregate statistics.
    pub stats: DiffStats,
}

impl DiffReport {
    /// Get entries of a specific change type.
    #[must_use]
    pub fn filter_by_type(&self, change_type: DiffChangeType) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| e.change_type == change_type)
            .collect()
    }

    /// Format the entire report as summary lines.
    #[must_use]
    pub fn format_summary(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!("Diff: {} -> {}", self.old_label, self.new_label));
        lines.push(format!(
            "  Added: {}, Removed: {}, Modified: {}, Unchanged: {}",
            self.stats.added, self.stats.removed, self.stats.modified, self.stats.unchanged
        ));
        for entry in &self.entries {
            if entry.change_type != DiffChangeType::Unchanged {
                lines.push(format!("  {}", entry.summary_line()));
            }
        }
        lines
    }
}

/// The conform diff engine.
#[derive(Debug, Clone)]
pub struct ConformDiffEngine {
    /// Whether to include unchanged clips in the report.
    pub include_unchanged: bool,
    /// Whether to detect moved clips (more expensive).
    pub detect_moves: bool,
}

impl Default for ConformDiffEngine {
    fn default() -> Self {
        Self {
            include_unchanged: false,
            detect_moves: true,
        }
    }
}

impl ConformDiffEngine {
    /// Create a new diff engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the diff between two sets of clips.
    pub fn diff(
        &self,
        old_clips: &[DiffClip],
        new_clips: &[DiffClip],
        old_label: impl Into<String>,
        new_label: impl Into<String>,
    ) -> DiffReport {
        let old_map: HashMap<&str, &DiffClip> =
            old_clips.iter().map(|c| (c.clip_id.as_str(), c)).collect();
        let new_map: HashMap<&str, &DiffClip> =
            new_clips.iter().map(|c| (c.clip_id.as_str(), c)).collect();

        let mut entries = Vec::new();
        let mut stats = DiffStats {
            old_total: old_clips.len(),
            new_total: new_clips.len(),
            ..Default::default()
        };

        // Check old clips: removed or modified
        for old_clip in old_clips {
            if let Some(new_clip) = new_map.get(old_clip.clip_id.as_str()) {
                let fields = old_clip.diff_fields(new_clip);
                if fields.is_empty() {
                    stats.unchanged += 1;
                    if self.include_unchanged {
                        entries.push(DiffEntry::unchanged(old_clip.clone()));
                    }
                } else {
                    stats.modified += 1;
                    for field in &fields {
                        *stats.field_changes.entry(*field).or_insert(0) += 1;
                    }
                    entries.push(DiffEntry::modified(
                        old_clip.clone(),
                        (*new_clip).clone(),
                        fields,
                    ));
                }
            } else {
                stats.removed += 1;
                entries.push(DiffEntry::removed(old_clip.clone()));
            }
        }

        // Check new clips: added
        for new_clip in new_clips {
            if !old_map.contains_key(new_clip.clip_id.as_str()) {
                stats.added += 1;
                entries.push(DiffEntry::added(new_clip.clone()));
            }
        }

        DiffReport {
            old_label: old_label.into(),
            new_label: new_label.into(),
            entries,
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip_a() -> DiffClip {
        DiffClip::new("clip_001", "/media/a.mxf", 0, 100, 0, 100, 0, 1.0)
    }

    fn clip_b() -> DiffClip {
        DiffClip::new("clip_002", "/media/b.mxf", 100, 250, 0, 150, 0, 1.0)
    }

    #[test]
    fn test_diff_clip_duration() {
        let c = clip_a();
        assert_eq!(c.duration_frames(), 100);
    }

    #[test]
    fn test_diff_clip_no_changes() {
        let c1 = clip_a();
        let c2 = clip_a();
        let fields = c1.diff_fields(&c2);
        assert!(fields.is_empty());
    }

    #[test]
    fn test_diff_clip_source_changed() {
        let c1 = clip_a();
        let mut c2 = clip_a();
        c2.source_path = "/media/new_a.mxf".to_string();
        let fields = c1.diff_fields(&c2);
        assert!(fields.contains(&ModifiedField::SourcePath));
    }

    #[test]
    fn test_diff_clip_timing_changed() {
        let c1 = clip_a();
        let mut c2 = clip_a();
        c2.timeline_out = 120;
        let fields = c1.diff_fields(&c2);
        assert!(fields.contains(&ModifiedField::TimelineOut));
    }

    #[test]
    fn test_diff_clip_speed_changed() {
        let c1 = clip_a();
        let mut c2 = clip_a();
        c2.speed = 2.0;
        let fields = c1.diff_fields(&c2);
        assert!(fields.contains(&ModifiedField::Speed));
    }

    #[test]
    fn test_diff_entry_added() {
        let entry = DiffEntry::added(clip_a());
        assert_eq!(entry.change_type, DiffChangeType::Added);
        assert!(entry.old_clip.is_none());
        assert!(entry.new_clip.is_some());
    }

    #[test]
    fn test_diff_entry_removed() {
        let entry = DiffEntry::removed(clip_a());
        assert_eq!(entry.change_type, DiffChangeType::Removed);
        assert!(entry.old_clip.is_some());
        assert!(entry.new_clip.is_none());
    }

    #[test]
    fn test_diff_entry_summary_line() {
        let entry = DiffEntry::added(clip_a());
        let line = entry.summary_line();
        assert!(line.contains("added"));
        assert!(line.contains("clip_001"));
    }

    #[test]
    fn test_diff_engine_identical() {
        let engine = ConformDiffEngine::new();
        let old = vec![clip_a(), clip_b()];
        let new = vec![clip_a(), clip_b()];
        let report = engine.diff(&old, &new, "v1", "v2");
        assert!(report.stats.is_identical());
        assert!((report.stats.similarity() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diff_engine_added_clip() {
        let engine = ConformDiffEngine::new();
        let old = vec![clip_a()];
        let new = vec![clip_a(), clip_b()];
        let report = engine.diff(&old, &new, "v1", "v2");
        assert_eq!(report.stats.added, 1);
        assert_eq!(report.stats.unchanged, 1);
    }

    #[test]
    fn test_diff_engine_removed_clip() {
        let engine = ConformDiffEngine::new();
        let old = vec![clip_a(), clip_b()];
        let new = vec![clip_a()];
        let report = engine.diff(&old, &new, "v1", "v2");
        assert_eq!(report.stats.removed, 1);
        assert_eq!(report.stats.unchanged, 1);
    }

    #[test]
    fn test_diff_engine_modified_clip() {
        let engine = ConformDiffEngine::new();
        let old = vec![clip_a()];
        let mut modified = clip_a();
        modified.timeline_out = 200;
        let new = vec![modified];
        let report = engine.diff(&old, &new, "v1", "v2");
        assert_eq!(report.stats.modified, 1);
    }

    #[test]
    fn test_diff_engine_empty_timelines() {
        let engine = ConformDiffEngine::new();
        let report = engine.diff(&[], &[], "v1", "v2");
        assert!(report.stats.is_identical());
        assert!((report.stats.similarity() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diff_stats_similarity() {
        let stats = DiffStats {
            old_total: 10,
            new_total: 10,
            unchanged: 8,
            modified: 2,
            ..Default::default()
        };
        assert!((stats.similarity() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diff_report_filter_by_type() {
        let engine = ConformDiffEngine::new();
        let old = vec![clip_a()];
        let new = vec![clip_a(), clip_b()];
        let report = engine.diff(&old, &new, "v1", "v2");
        let added = report.filter_by_type(DiffChangeType::Added);
        assert_eq!(added.len(), 1);
    }

    #[test]
    fn test_diff_report_format_summary() {
        let engine = ConformDiffEngine::new();
        let old = vec![clip_a()];
        let new = vec![clip_b()];
        let report = engine.diff(&old, &new, "v1", "v2");
        let lines = report.format_summary();
        assert!(!lines.is_empty());
        assert!(lines[0].contains("v1"));
    }

    #[test]
    fn test_change_type_display() {
        assert_eq!(format!("{}", DiffChangeType::Added), "ADDED");
        assert_eq!(format!("{}", DiffChangeType::Removed), "REMOVED");
        assert_eq!(format!("{}", DiffChangeType::Modified), "MODIFIED");
    }

    #[test]
    fn test_modified_field_display() {
        assert_eq!(format!("{}", ModifiedField::SourcePath), "source_path");
        assert_eq!(format!("{}", ModifiedField::Speed), "speed");
    }
}
