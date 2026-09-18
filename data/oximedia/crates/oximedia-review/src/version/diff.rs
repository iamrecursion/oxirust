//! Visual diff between versions.

use serde::{Deserialize, Serialize};

/// Type of difference between versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffType {
    /// Frame added.
    Added,
    /// Frame removed.
    Removed,
    /// Frame modified.
    Modified,
    /// Metadata changed.
    Metadata,
    /// Audio changed.
    Audio,
    /// Timing changed.
    Timing,
}

/// Difference between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Type of difference.
    pub diff_type: DiffType,
    /// Frame number (if applicable).
    pub frame: Option<i64>,
    /// Description of the change.
    pub description: String,
    /// Severity (0.0-1.0, higher = more significant).
    pub severity: f32,
}

impl VersionDiff {
    /// Create a new version diff.
    #[must_use]
    pub fn new(diff_type: DiffType, description: String) -> Self {
        Self {
            diff_type,
            frame: None,
            description,
            severity: 0.5,
        }
    }

    /// Set the frame number.
    #[must_use]
    pub fn with_frame(mut self, frame: i64) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Set the severity.
    #[must_use]
    pub fn with_severity(mut self, severity: f32) -> Self {
        self.severity = severity.clamp(0.0, 1.0);
        self
    }

    /// Check if this is a high severity change.
    #[must_use]
    pub fn is_high_severity(&self) -> bool {
        self.severity > 0.7
    }
}

/// Diff statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffStats {
    /// Number of added frames.
    pub frames_added: usize,
    /// Number of removed frames.
    pub frames_removed: usize,
    /// Number of modified frames.
    pub frames_modified: usize,
    /// Number of metadata changes.
    pub metadata_changes: usize,
    /// Total changes.
    pub total_changes: usize,
}

impl DiffStats {
    /// Create new diff stats from a list of diffs.
    #[must_use]
    pub fn from_diffs(diffs: &[VersionDiff]) -> Self {
        let mut stats = Self::default();

        for diff in diffs {
            match diff.diff_type {
                DiffType::Added => stats.frames_added += 1,
                DiffType::Removed => stats.frames_removed += 1,
                DiffType::Modified => stats.frames_modified += 1,
                DiffType::Metadata => stats.metadata_changes += 1,
                DiffType::Audio | DiffType::Timing => {}
            }
        }

        stats.total_changes = diffs.len();
        stats
    }

    /// Calculate change percentage.
    #[must_use]
    pub fn change_percentage(&self, total_frames: usize) -> f64 {
        if total_frames == 0 {
            return 0.0;
        }

        let changed_frames = self.frames_added + self.frames_removed + self.frames_modified;
        (changed_frames as f64 / total_frames as f64) * 100.0
    }
}

/// Side-by-side diff view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideBySideDiff {
    /// Frame number.
    pub frame: i64,
    /// Left side (version A) image URL.
    pub left_image: String,
    /// Right side (version B) image URL.
    pub right_image: String,
    /// Diff overlay image URL.
    pub diff_image: Option<String>,
    /// Difference score (0.0-1.0).
    pub difference_score: f64,
}

/// Overlay diff view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayDiff {
    /// Frame number.
    pub frame: i64,
    /// Base image URL (version A).
    pub base_image: String,
    /// Overlay image URL (version B with transparency).
    pub overlay_image: String,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Opacity of overlay (0.0-1.0).
    pub opacity: f32,
}

/// Blend mode for overlay diffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendMode {
    /// Normal blending.
    Normal,
    /// Difference blending.
    Difference,
    /// Multiply blending.
    Multiply,
    /// Screen blending.
    Screen,
    /// Overlay blending.
    Overlay,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_diff_creation() {
        let diff = VersionDiff::new(DiffType::Modified, "Color corrected".to_string())
            .with_frame(100)
            .with_severity(0.8);

        assert_eq!(diff.diff_type, DiffType::Modified);
        assert_eq!(diff.frame, Some(100));
        assert!(diff.is_high_severity());
    }

    #[test]
    fn test_diff_stats() {
        let diffs = vec![
            VersionDiff::new(DiffType::Added, "Added frame".to_string()),
            VersionDiff::new(DiffType::Modified, "Modified frame".to_string()),
            VersionDiff::new(DiffType::Modified, "Modified frame".to_string()),
            VersionDiff::new(DiffType::Removed, "Removed frame".to_string()),
        ];

        let stats = DiffStats::from_diffs(&diffs);
        assert_eq!(stats.frames_added, 1);
        assert_eq!(stats.frames_modified, 2);
        assert_eq!(stats.frames_removed, 1);
        assert_eq!(stats.total_changes, 4);
    }

    #[test]
    fn test_change_percentage() {
        let stats = DiffStats {
            frames_added: 5,
            frames_removed: 3,
            frames_modified: 12,
            metadata_changes: 0,
            total_changes: 20,
        };

        let percentage = stats.change_percentage(100);
        assert!((percentage - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_diff_type_equality() {
        assert_eq!(DiffType::Added, DiffType::Added);
        assert_ne!(DiffType::Added, DiffType::Removed);
    }
}
