//! Diff utilities for comparing two presets and identifying changed fields.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single entry in a preset diff describing one changed field.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffEntry {
    /// Name of the changed field.
    pub field: String,
    /// Value in the "before" (base) preset, as a display string.
    pub before: String,
    /// Value in the "after" (modified) preset, as a display string.
    pub after: String,
    /// Whether the change is classified as significant (i.e. materially affects output).
    pub significant: bool,
}

impl DiffEntry {
    /// Create a new diff entry.
    #[must_use]
    pub fn new(
        field: impl Into<String>,
        before: impl Into<String>,
        after: impl Into<String>,
        significant: bool,
    ) -> Self {
        Self {
            field: field.into(),
            before: before.into(),
            after: after.into(),
            significant,
        }
    }

    /// Return `true` if this change is classified as significant.
    ///
    /// Significant changes are those that would materially affect the encoded output
    /// quality, compatibility, or file size (e.g. codec, resolution, or bitrate changes).
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.significant
    }

    /// Return a human-readable summary line.
    #[must_use]
    pub fn summary(&self) -> String {
        format!("{}: {} -> {}", self.field, self.before, self.after)
    }
}

/// The complete diff between two preset configurations, as a list of `DiffEntry` records.
#[derive(Debug, Clone, Default)]
pub struct PresetDiff {
    entries: Vec<DiffEntry>,
}

impl PresetDiff {
    /// Create an empty diff.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a diff entry.
    pub fn push(&mut self, entry: DiffEntry) {
        self.entries.push(entry);
    }

    /// Return all diff entries (changed fields).
    #[must_use]
    pub fn entries(&self) -> &[DiffEntry] {
        &self.entries
    }

    /// Return only the entries for fields that changed.
    ///
    /// (All entries in a `PresetDiff` represent changes, so this returns all of them,
    /// but the method name follows the spec interface.)
    #[must_use]
    pub fn changed_fields(&self) -> Vec<&DiffEntry> {
        self.entries.iter().collect()
    }

    /// Return only the entries that are marked as significant.
    #[must_use]
    pub fn significant_changes(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.significant).collect()
    }

    /// Total number of changed fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no fields changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return `true` if there is at least one significant change.
    #[must_use]
    pub fn has_significant_changes(&self) -> bool {
        self.entries.iter().any(|e| e.significant)
    }
}

/// Fields that are considered breaking when changed (codec / container switches).
const BREAKING_FIELDS: &[&str] = &["video_codec", "audio_codec", "container"];

/// Significance threshold: changing a numeric field by more than this fraction
/// of the original value is considered significant (5 %).
const SIGNIFICANCE_THRESHOLD: f64 = 0.05;

/// Compares two preset parameter maps and produces a `PresetDiff`.
pub struct PresetDiffCompare;

impl PresetDiffCompare {
    /// Compare `base` and `modified` parameter maps (string->string), returning a diff.
    ///
    /// Fields present in `base` but absent from `modified` are treated as removals.
    /// Fields present in `modified` but absent from `base` are treated as additions.
    #[must_use]
    pub fn compare(
        base: &HashMap<String, String>,
        modified: &HashMap<String, String>,
    ) -> PresetDiff {
        let mut diff = PresetDiff::new();

        // Changed or removed fields
        for (field, before_val) in base {
            match modified.get(field.as_str()) {
                Some(after_val) if after_val != before_val => {
                    let significant = Self::is_significant(field, before_val, after_val);
                    diff.push(DiffEntry::new(field, before_val, after_val, significant));
                }
                None => {
                    // Field removed
                    diff.push(DiffEntry::new(field, before_val, "<removed>", true));
                }
                _ => {}
            }
        }

        // Newly added fields
        for (field, after_val) in modified {
            if !base.contains_key(field.as_str()) {
                diff.push(DiffEntry::new(field, "<added>", after_val, false));
            }
        }

        diff
    }

    /// Determine if a change to `field` from `before` to `after` is significant.
    #[must_use]
    pub fn is_significant(field: &str, before: &str, after: &str) -> bool {
        // Breaking fields are always significant
        if BREAKING_FIELDS.contains(&field) {
            return true;
        }

        // Numeric fields: significant if the relative change exceeds the threshold
        if let (Ok(b), Ok(a)) = (before.parse::<f64>(), after.parse::<f64>()) {
            if b == 0.0 {
                return a != 0.0;
            }
            #[allow(clippy::cast_precision_loss)]
            let rel = (a - b).abs() / b.abs();
            return rel > SIGNIFICANCE_THRESHOLD;
        }

        // Non-numeric string change: always significant
        before != after
    }

    /// Return `true` if the diff contains any breaking changes (codec or container switch).
    #[must_use]
    pub fn has_breaking_changes(diff: &PresetDiff) -> bool {
        diff.entries()
            .iter()
            .any(|e| BREAKING_FIELDS.contains(&e.field.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DiffEntry ---

    #[test]
    fn test_diff_entry_is_significant_true() {
        let e = DiffEntry::new("video_codec", "h264", "av1", true);
        assert!(e.is_significant());
    }

    #[test]
    fn test_diff_entry_is_significant_false() {
        let e = DiffEntry::new("description", "old", "new", false);
        assert!(!e.is_significant());
    }

    #[test]
    fn test_diff_entry_summary() {
        let e = DiffEntry::new("width", "1920", "1280", true);
        assert_eq!(e.summary(), "width: 1920 -> 1280");
    }

    // --- PresetDiff ---

    #[test]
    fn test_preset_diff_empty() {
        let diff = PresetDiff::new();
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn test_preset_diff_push_and_len() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("crf", "18", "23", true));
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn test_preset_diff_changed_fields() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("width", "1920", "1280", true));
        diff.push(DiffEntry::new("height", "1080", "720", true));
        assert_eq!(diff.changed_fields().len(), 2);
    }

    #[test]
    fn test_preset_diff_significant_changes_filter() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("video_codec", "h264", "av1", true));
        diff.push(DiffEntry::new("tag", "old", "new", false));
        let sig = diff.significant_changes();
        assert_eq!(sig.len(), 1);
        assert_eq!(sig[0].field, "video_codec");
    }

    #[test]
    fn test_preset_diff_has_significant_changes() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("width", "1920", "1280", true));
        assert!(diff.has_significant_changes());
    }

    #[test]
    fn test_preset_diff_no_significant_changes() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("label", "a", "b", false));
        assert!(!diff.has_significant_changes());
    }

    // --- PresetDiffCompare ---

    #[test]
    fn test_compare_identical_maps() {
        let mut m = HashMap::new();
        m.insert("width".to_string(), "1920".to_string());
        let diff = PresetDiffCompare::compare(&m, &m);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_compare_changed_field() {
        let mut base = HashMap::new();
        base.insert("width".to_string(), "1920".to_string());
        let mut modified = HashMap::new();
        modified.insert("width".to_string(), "1280".to_string());
        let diff = PresetDiffCompare::compare(&base, &modified);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff.entries()[0].field, "width");
    }

    #[test]
    fn test_compare_removed_field() {
        let mut base = HashMap::new();
        base.insert("label".to_string(), "hi".to_string());
        let modified = HashMap::new();
        let diff = PresetDiffCompare::compare(&base, &modified);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff.entries()[0].after, "<removed>");
    }

    #[test]
    fn test_compare_added_field() {
        let base = HashMap::new();
        let mut modified = HashMap::new();
        modified.insert("new_field".to_string(), "val".to_string());
        let diff = PresetDiffCompare::compare(&base, &modified);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff.entries()[0].before, "<added>");
    }

    #[test]
    fn test_is_significant_codec_change() {
        assert!(PresetDiffCompare::is_significant(
            "video_codec",
            "h264",
            "av1"
        ));
    }

    #[test]
    fn test_is_significant_small_numeric_change() {
        // 1% change is below threshold → not significant
        assert!(!PresetDiffCompare::is_significant("crf", "1000", "1010"));
    }

    #[test]
    fn test_is_significant_large_numeric_change() {
        // 50% change is above threshold → significant
        assert!(PresetDiffCompare::is_significant(
            "bitrate", "4000000", "6000000"
        ));
    }

    #[test]
    fn test_has_breaking_changes_true() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("video_codec", "h264", "hevc", true));
        assert!(PresetDiffCompare::has_breaking_changes(&diff));
    }

    #[test]
    fn test_has_breaking_changes_false() {
        let mut diff = PresetDiff::new();
        diff.push(DiffEntry::new("crf", "18", "23", true));
        assert!(!PresetDiffCompare::has_breaking_changes(&diff));
    }
}
