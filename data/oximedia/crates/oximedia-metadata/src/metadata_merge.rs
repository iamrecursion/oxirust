#![allow(dead_code)]
//! Metadata merging utilities for combining fields from multiple metadata sources.
//!
//! When working with media files that have metadata in several formats (e.g., both
//! ID3v2 and Vorbis tags on a FLAC file, or XMP + EXIF on a JPEG), this module
//! provides strategies for merging them into a single coherent set of fields.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Strategy for resolving conflicts when the same key appears in multiple sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Keep the value from the first (higher-priority) source.
    KeepFirst,
    /// Keep the value from the last (lower-priority) source.
    KeepLast,
    /// Concatenate values with a separator.
    Concatenate,
    /// Skip the field entirely if there is a conflict.
    Skip,
    /// Keep the longer of the two values.
    KeepLonger,
    /// Keep the shorter of the two values.
    KeepShorter,
}

impl fmt::Display for ConflictStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeepFirst => write!(f, "Keep First"),
            Self::KeepLast => write!(f, "Keep Last"),
            Self::Concatenate => write!(f, "Concatenate"),
            Self::Skip => write!(f, "Skip"),
            Self::KeepLonger => write!(f, "Keep Longer"),
            Self::KeepShorter => write!(f, "Keep Shorter"),
        }
    }
}

/// A single metadata field value used in merging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldValue {
    /// The string representation of the value.
    pub value: String,
    /// The source identifier (e.g., format name or file path).
    pub source: String,
    /// Priority (lower number = higher priority).
    pub priority: u32,
}

impl FieldValue {
    /// Create a new field value.
    pub fn new(value: &str, source: &str, priority: u32) -> Self {
        Self {
            value: value.to_string(),
            source: source.to_string(),
            priority,
        }
    }
}

/// Record of a conflict encountered during merge.
#[derive(Debug, Clone)]
pub struct MergeConflict {
    /// The key that had conflicting values.
    pub key: String,
    /// The values from different sources.
    pub values: Vec<FieldValue>,
    /// How the conflict was resolved.
    pub resolution: ConflictStrategy,
    /// The resolved value (if any).
    pub resolved_value: Option<String>,
}

impl MergeConflict {
    /// Create a new merge conflict record.
    pub fn new(key: &str, values: Vec<FieldValue>, resolution: ConflictStrategy) -> Self {
        Self {
            key: key.to_string(),
            values,
            resolution,
            resolved_value: None,
        }
    }

    /// Set the resolved value.
    pub fn with_resolved(mut self, value: String) -> Self {
        self.resolved_value = Some(value);
        self
    }

    /// Number of conflicting sources.
    pub fn source_count(&self) -> usize {
        self.values.len()
    }
}

/// Result of a metadata merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged fields (key -> value).
    pub fields: HashMap<String, String>,
    /// Any conflicts that were encountered and resolved.
    pub conflicts: Vec<MergeConflict>,
    /// Keys that were skipped due to conflict strategy.
    pub skipped_keys: HashSet<String>,
    /// Total number of input fields processed.
    pub total_inputs: usize,
}

impl MergeResult {
    /// Create a new empty merge result.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            conflicts: Vec::new(),
            skipped_keys: HashSet::new(),
            total_inputs: 0,
        }
    }

    /// Number of merged fields in the output.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Number of conflicts encountered.
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Whether the merge was conflict-free.
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }

    /// Get a merged field value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    /// Check if a key was skipped.
    pub fn was_skipped(&self, key: &str) -> bool {
        self.skipped_keys.contains(key)
    }
}

impl Default for MergeResult {
    fn default() -> Self {
        Self::new()
    }
}

/// A metadata source to be fed into the merger.
#[derive(Debug, Clone)]
pub struct MetadataSource {
    /// Human-readable label for this source.
    pub label: String,
    /// Priority of this source (lower = higher priority).
    pub priority: u32,
    /// The key-value fields from this source.
    pub fields: HashMap<String, String>,
}

impl MetadataSource {
    /// Create a new metadata source.
    pub fn new(label: &str, priority: u32) -> Self {
        Self {
            label: label.to_string(),
            priority,
            fields: HashMap::new(),
        }
    }

    /// Insert a field.
    pub fn insert(&mut self, key: &str, value: &str) {
        self.fields.insert(key.to_string(), value.to_string());
    }

    /// Number of fields in this source.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// The metadata merger engine.
#[derive(Debug, Clone)]
pub struct MetadataMerger {
    /// Default conflict resolution strategy.
    pub default_strategy: ConflictStrategy,
    /// Per-key overrides for conflict resolution.
    key_strategies: HashMap<String, ConflictStrategy>,
    /// Separator used for Concatenate strategy.
    pub separator: String,
    /// Keys to exclude from merging.
    exclude_keys: HashSet<String>,
}

impl MetadataMerger {
    /// Create a new merger with the given default strategy.
    pub fn new(default_strategy: ConflictStrategy) -> Self {
        Self {
            default_strategy,
            key_strategies: HashMap::new(),
            separator: "; ".to_string(),
            exclude_keys: HashSet::new(),
        }
    }

    /// Set a per-key conflict strategy.
    pub fn set_key_strategy(&mut self, key: &str, strategy: ConflictStrategy) {
        self.key_strategies.insert(key.to_string(), strategy);
    }

    /// Set the concatenation separator.
    pub fn set_separator(&mut self, sep: &str) {
        self.separator = sep.to_string();
    }

    /// Exclude a key from the merge output.
    pub fn exclude_key(&mut self, key: &str) {
        self.exclude_keys.insert(key.to_string());
    }

    /// Get the effective strategy for a given key.
    pub fn strategy_for(&self, key: &str) -> ConflictStrategy {
        self.key_strategies
            .get(key)
            .copied()
            .unwrap_or(self.default_strategy)
    }

    /// Merge multiple metadata sources.
    pub fn merge(&self, sources: &[MetadataSource]) -> MergeResult {
        let mut result = MergeResult::new();

        // Collect all field values grouped by key
        let mut grouped: HashMap<String, Vec<FieldValue>> = HashMap::new();

        for source in sources {
            for (key, value) in &source.fields {
                if self.exclude_keys.contains(key) {
                    continue;
                }
                result.total_inputs += 1;
                grouped
                    .entry(key.clone())
                    .or_default()
                    .push(FieldValue::new(value, &source.label, source.priority));
            }
        }

        // Resolve each key
        for (key, mut values) in grouped {
            // Sort by priority (ascending = higher priority first)
            values.sort_by_key(|v| v.priority);

            // Deduplicate: if all values are the same, no conflict
            let unique: HashSet<&str> = values.iter().map(|v| v.value.as_str()).collect();
            if unique.len() == 1 {
                result.fields.insert(key, values[0].value.clone());
                continue;
            }

            // Conflict!
            let strategy = self.strategy_for(&key);
            let resolved = self.resolve_conflict(&values, strategy);

            let mut conflict = MergeConflict::new(&key, values, strategy);
            if let Some(ref val) = resolved {
                conflict = conflict.with_resolved(val.clone());
                result.fields.insert(key.clone(), val.clone());
            } else {
                result.skipped_keys.insert(key.clone());
            }
            result.conflicts.push(conflict);
        }

        result
    }

    /// Resolve a conflict according to the given strategy.
    fn resolve_conflict(
        &self,
        values: &[FieldValue],
        strategy: ConflictStrategy,
    ) -> Option<String> {
        if values.is_empty() {
            return None;
        }
        match strategy {
            ConflictStrategy::KeepFirst => Some(values[0].value.clone()),
            ConflictStrategy::KeepLast => Some(values[values.len() - 1].value.clone()),
            ConflictStrategy::Concatenate => {
                let parts: Vec<&str> = values.iter().map(|v| v.value.as_str()).collect();
                // Deduplicate while preserving order
                let mut seen = HashSet::new();
                let deduped: Vec<&str> = parts.into_iter().filter(|p| seen.insert(*p)).collect();
                Some(deduped.join(&self.separator))
            }
            ConflictStrategy::Skip => None,
            ConflictStrategy::KeepLonger => values
                .iter()
                .max_by_key(|v| v.value.len())
                .map(|v| v.value.clone()),
            ConflictStrategy::KeepShorter => values
                .iter()
                .min_by_key(|v| v.value.len())
                .map(|v| v.value.clone()),
        }
    }
}

impl Default for MetadataMerger {
    fn default() -> Self {
        Self::new(ConflictStrategy::KeepFirst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_strategy_display() {
        assert_eq!(ConflictStrategy::KeepFirst.to_string(), "Keep First");
        assert_eq!(ConflictStrategy::Concatenate.to_string(), "Concatenate");
        assert_eq!(ConflictStrategy::Skip.to_string(), "Skip");
    }

    #[test]
    fn test_field_value_creation() {
        let fv = FieldValue::new("test", "source1", 0);
        assert_eq!(fv.value, "test");
        assert_eq!(fv.source, "source1");
        assert_eq!(fv.priority, 0);
    }

    #[test]
    fn test_merge_conflict_source_count() {
        let v1 = FieldValue::new("a", "s1", 0);
        let v2 = FieldValue::new("b", "s2", 1);
        let conflict = MergeConflict::new("key", vec![v1, v2], ConflictStrategy::KeepFirst);
        assert_eq!(conflict.source_count(), 2);
    }

    #[test]
    fn test_merge_no_conflict() {
        let merger = MetadataMerger::new(ConflictStrategy::KeepFirst);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("title", "My Song");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("title", "My Song");

        let result = merger.merge(&[s1, s2]);
        assert!(result.is_clean());
        assert_eq!(result.get("title"), Some("My Song"));
    }

    #[test]
    fn test_merge_keep_first() {
        let merger = MetadataMerger::new(ConflictStrategy::KeepFirst);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("artist", "Artist A");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("artist", "Artist B");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.get("artist"), Some("Artist A"));
    }

    #[test]
    fn test_merge_keep_last() {
        let merger = MetadataMerger::new(ConflictStrategy::KeepLast);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("artist", "Artist A");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("artist", "Artist B");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.get("artist"), Some("Artist B"));
    }

    #[test]
    fn test_merge_concatenate() {
        let merger = MetadataMerger::new(ConflictStrategy::Concatenate);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("genre", "Rock");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("genre", "Alternative");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.get("genre"), Some("Rock; Alternative"));
    }

    #[test]
    fn test_merge_skip() {
        let merger = MetadataMerger::new(ConflictStrategy::Skip);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("comment", "A");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("comment", "B");

        let result = merger.merge(&[s1, s2]);
        assert!(result.was_skipped("comment"));
        assert_eq!(result.get("comment"), None);
    }

    #[test]
    fn test_merge_keep_longer() {
        let merger = MetadataMerger::new(ConflictStrategy::KeepLonger);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("title", "Hi");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("title", "Hello World");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.get("title"), Some("Hello World"));
    }

    #[test]
    fn test_merge_keep_shorter() {
        let merger = MetadataMerger::new(ConflictStrategy::KeepShorter);
        let mut s1 = MetadataSource::new("id3v2", 0);
        s1.insert("title", "Hi");
        let mut s2 = MetadataSource::new("vorbis", 1);
        s2.insert("title", "Hello World");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.get("title"), Some("Hi"));
    }

    #[test]
    fn test_per_key_strategy() {
        let mut merger = MetadataMerger::new(ConflictStrategy::KeepFirst);
        merger.set_key_strategy("genre", ConflictStrategy::Concatenate);

        let mut s1 = MetadataSource::new("s1", 0);
        s1.insert("genre", "Rock");
        s1.insert("artist", "Artist A");
        let mut s2 = MetadataSource::new("s2", 1);
        s2.insert("genre", "Pop");
        s2.insert("artist", "Artist B");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.get("genre"), Some("Rock; Pop"));
        assert_eq!(result.get("artist"), Some("Artist A"));
    }

    #[test]
    fn test_exclude_keys() {
        let mut merger = MetadataMerger::new(ConflictStrategy::KeepFirst);
        merger.exclude_key("encoder");

        let mut s1 = MetadataSource::new("s1", 0);
        s1.insert("title", "Song");
        s1.insert("encoder", "LAME 3.100");

        let result = merger.merge(&[s1]);
        assert_eq!(result.get("title"), Some("Song"));
        assert_eq!(result.get("encoder"), None);
    }

    #[test]
    fn test_merge_result_default() {
        let result = MergeResult::default();
        assert_eq!(result.field_count(), 0);
        assert_eq!(result.conflict_count(), 0);
        assert!(result.is_clean());
    }

    #[test]
    fn test_metadata_source_field_count() {
        let mut src = MetadataSource::new("test", 0);
        assert_eq!(src.field_count(), 0);
        src.insert("a", "1");
        src.insert("b", "2");
        assert_eq!(src.field_count(), 2);
    }

    #[test]
    fn test_custom_separator() {
        let mut merger = MetadataMerger::new(ConflictStrategy::Concatenate);
        merger.set_separator(" | ");

        let mut s1 = MetadataSource::new("s1", 0);
        s1.insert("genre", "Rock");
        let mut s2 = MetadataSource::new("s2", 1);
        s2.insert("genre", "Pop");

        let result = merger.merge(&[s1, s2]);
        assert_eq!(result.get("genre"), Some("Rock | Pop"));
    }

    #[test]
    fn test_merger_default_trait() {
        let merger = MetadataMerger::default();
        assert_eq!(merger.default_strategy, ConflictStrategy::KeepFirst);
    }

    #[test]
    fn test_merge_disjoint_sources() {
        let merger = MetadataMerger::new(ConflictStrategy::KeepFirst);
        let mut s1 = MetadataSource::new("s1", 0);
        s1.insert("title", "Song");
        let mut s2 = MetadataSource::new("s2", 1);
        s2.insert("artist", "Artist");

        let result = merger.merge(&[s1, s2]);
        assert!(result.is_clean());
        assert_eq!(result.field_count(), 2);
        assert_eq!(result.get("title"), Some("Song"));
        assert_eq!(result.get("artist"), Some("Artist"));
    }
}
