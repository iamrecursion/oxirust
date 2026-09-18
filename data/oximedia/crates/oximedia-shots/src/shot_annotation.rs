//! Shot annotation: attach free-form metadata and tags to detected shots.
//!
//! `ShotAnnotation` is an extensible record that associates free-form text
//! notes, arbitrary key–value metadata, and a set of semantic tags with a
//! detected shot. An `AnnotationStore` provides CRUD operations plus
//! tag-based and range-based querying so that callers can efficiently
//! retrieve subsets of annotated shots.
//!
//! # Design
//!
//! - **No external dependencies** beyond `std` and `serde`.
//! - **No `unwrap()`** – all fallible operations return `ShotResult`.
//! - Tags are stored as `BTreeSet<String>` for deterministic iteration and
//!   efficient prefix queries.
//! - Metadata values are arbitrary UTF-8 strings; typed access helpers are
//!   provided for `f64`, `i64`, and `bool`.

use crate::error::{ShotError, ShotResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

// ---------------------------------------------------------------------------
// Core Types
// ---------------------------------------------------------------------------

/// A single annotation attached to a shot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotAnnotation {
    /// The ID of the shot this annotation belongs to.
    pub shot_id: u64,
    /// Free-form text notes written by the annotator.
    pub notes: String,
    /// Semantic tags (e.g. `"dialogue"`, `"action"`, `"vfx-ref"`, `"favourite"`).
    pub tags: BTreeSet<String>,
    /// Arbitrary key–value metadata (e.g. `"scene" => "INT. OFFICE - DAY"`).
    pub metadata: BTreeMap<String, String>,
    /// Author/creator of this annotation (optional).
    pub author: Option<String>,
    /// ISO-8601 timestamp string when this annotation was created (optional).
    pub created_at: Option<String>,
    /// Colour label for visual differentiation (hex string like `"#FF6B6B"`).
    pub color_label: Option<String>,
    /// Star rating in the range 0–5 (0 = unrated).
    pub rating: u8,
}

impl ShotAnnotation {
    /// Create a new blank annotation for the given shot.
    #[must_use]
    pub fn new(shot_id: u64) -> Self {
        Self {
            shot_id,
            notes: String::new(),
            tags: BTreeSet::new(),
            metadata: BTreeMap::new(),
            author: None,
            created_at: None,
            color_label: None,
            rating: 0,
        }
    }

    /// Builder: set the free-form notes.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Builder: add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }

    /// Builder: add a metadata key–value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builder: set the author.
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Builder: set the creation timestamp (ISO-8601 string).
    #[must_use]
    pub fn with_created_at(mut self, ts: impl Into<String>) -> Self {
        self.created_at = Some(ts.into());
        self
    }

    /// Builder: set a colour label (hex colour string, e.g. `"#FF6B6B"`).
    ///
    /// # Errors
    ///
    /// Returns `ShotError::InvalidMetadata` if the string is not a valid
    /// 7-character hex colour (`#RRGGBB`).
    pub fn with_color_label(mut self, color: impl Into<String>) -> ShotResult<Self> {
        let c = color.into();
        validate_hex_color(&c)?;
        self.color_label = Some(c);
        Ok(self)
    }

    /// Builder: set the star rating (clamped to 0–5).
    #[must_use]
    pub fn with_rating(mut self, rating: u8) -> Self {
        self.rating = rating.min(5);
        self
    }

    /// Add a tag to this annotation.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.insert(tag.into());
    }

    /// Remove a tag from this annotation. Returns `true` if the tag existed.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        self.tags.remove(tag)
    }

    /// Check whether a tag is present.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Get a metadata value parsed as `f64`.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::InvalidMetadata` if the key is absent or the value
    /// cannot be parsed as a floating-point number.
    pub fn get_metadata_f64(&self, key: &str) -> ShotResult<f64> {
        let raw = self
            .metadata
            .get(key)
            .ok_or_else(|| ShotError::InvalidMetadata(format!("key '{key}' not found")))?;
        raw.parse::<f64>().map_err(|_| {
            ShotError::InvalidMetadata(format!("cannot parse '{raw}' as f64 for key '{key}'"))
        })
    }

    /// Get a metadata value parsed as `i64`.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::InvalidMetadata` if the key is absent or the value
    /// cannot be parsed as an integer.
    pub fn get_metadata_i64(&self, key: &str) -> ShotResult<i64> {
        let raw = self
            .metadata
            .get(key)
            .ok_or_else(|| ShotError::InvalidMetadata(format!("key '{key}' not found")))?;
        raw.parse::<i64>().map_err(|_| {
            ShotError::InvalidMetadata(format!("cannot parse '{raw}' as i64 for key '{key}'"))
        })
    }

    /// Get a metadata value parsed as `bool`.
    ///
    /// Accepts `"true"` / `"1"` as `true` and `"false"` / `"0"` as `false`
    /// (case-insensitive).
    ///
    /// # Errors
    ///
    /// Returns `ShotError::InvalidMetadata` if the key is absent or the value
    /// is not a recognised boolean string.
    pub fn get_metadata_bool(&self, key: &str) -> ShotResult<bool> {
        let raw = self
            .metadata
            .get(key)
            .ok_or_else(|| ShotError::InvalidMetadata(format!("key '{key}' not found")))?;
        match raw.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            _ => Err(ShotError::InvalidMetadata(format!(
                "cannot parse '{raw}' as bool for key '{key}'"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Annotation Store
// ---------------------------------------------------------------------------

/// CRUD store for shot annotations with tag-based and range-based querying.
///
/// Internally stores annotations indexed by `shot_id` and maintains a
/// reverse-index from tag → set of shot IDs for O(log n) tag lookups.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnnotationStore {
    /// Primary index: shot_id → annotation.
    annotations: HashMap<u64, ShotAnnotation>,
    /// Reverse tag index: tag → set of shot_ids.
    tag_index: HashMap<String, BTreeSet<u64>>,
}

impl AnnotationStore {
    /// Create an empty annotation store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace the annotation for a shot.
    ///
    /// If an annotation for the same `shot_id` already exists it is replaced
    /// and the tag index is updated atomically.
    pub fn insert(&mut self, annotation: ShotAnnotation) {
        let shot_id = annotation.shot_id;

        // Remove the shot from all existing tag index entries before replacing.
        if let Some(old) = self.annotations.get(&shot_id) {
            for tag in &old.tags {
                if let Some(set) = self.tag_index.get_mut(tag) {
                    set.remove(&shot_id);
                }
            }
        }

        // Add to new tag index entries.
        for tag in &annotation.tags {
            self.tag_index
                .entry(tag.clone())
                .or_default()
                .insert(shot_id);
        }

        self.annotations.insert(shot_id, annotation);
    }

    /// Get an immutable reference to the annotation for a shot.
    #[must_use]
    pub fn get(&self, shot_id: u64) -> Option<&ShotAnnotation> {
        self.annotations.get(&shot_id)
    }

    /// Get a mutable reference to the annotation for a shot.
    #[must_use]
    pub fn get_mut(&mut self, shot_id: u64) -> Option<&mut ShotAnnotation> {
        self.annotations.get_mut(&shot_id)
    }

    /// Remove the annotation for a shot. Returns the removed annotation if present.
    pub fn remove(&mut self, shot_id: u64) -> Option<ShotAnnotation> {
        if let Some(ann) = self.annotations.remove(&shot_id) {
            for tag in &ann.tags {
                if let Some(set) = self.tag_index.get_mut(tag) {
                    set.remove(&shot_id);
                }
            }
            Some(ann)
        } else {
            None
        }
    }

    /// Return `true` if there is an annotation for the given shot.
    #[must_use]
    pub fn contains(&self, shot_id: u64) -> bool {
        self.annotations.contains_key(&shot_id)
    }

    /// Total number of annotations stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// Return `true` if no annotations are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }

    /// Iterate over all annotations in undefined order.
    pub fn iter(&self) -> impl Iterator<Item = &ShotAnnotation> {
        self.annotations.values()
    }

    // ---- Query API --------------------------------------------------------

    /// Return references to all annotations that have the given tag.
    #[must_use]
    pub fn by_tag(&self, tag: &str) -> Vec<&ShotAnnotation> {
        let Some(ids) = self.tag_index.get(tag) else {
            return Vec::new();
        };
        ids.iter()
            .filter_map(|id| self.annotations.get(id))
            .collect()
    }

    /// Return references to all annotations whose tag set contains **all**
    /// of the provided tags (AND query).
    #[must_use]
    pub fn by_all_tags(&self, tags: &[&str]) -> Vec<&ShotAnnotation> {
        if tags.is_empty() {
            return self.annotations.values().collect();
        }

        // Start with the smallest candidate set.
        let mut candidate_ids: Option<BTreeSet<u64>> = None;
        for &tag in tags {
            let ids = self.tag_index.get(tag).cloned().unwrap_or_default();
            candidate_ids = Some(match candidate_ids {
                None => ids,
                Some(existing) => existing.intersection(&ids).copied().collect(),
            });
        }

        candidate_ids
            .unwrap_or_default()
            .iter()
            .filter_map(|id| self.annotations.get(id))
            .collect()
    }

    /// Return references to all annotations whose tag set contains **any**
    /// of the provided tags (OR query).
    #[must_use]
    pub fn by_any_tag(&self, tags: &[&str]) -> Vec<&ShotAnnotation> {
        let mut ids: BTreeSet<u64> = BTreeSet::new();
        for &tag in tags {
            if let Some(set) = self.tag_index.get(tag) {
                ids.extend(set.iter().copied());
            }
        }
        ids.iter()
            .filter_map(|id| self.annotations.get(id))
            .collect()
    }

    /// Return all annotations for shots whose IDs fall within
    /// `[start_shot_id, end_shot_id]` (inclusive).
    #[must_use]
    pub fn by_shot_range(&self, start_shot_id: u64, end_shot_id: u64) -> Vec<&ShotAnnotation> {
        self.annotations
            .values()
            .filter(|a| a.shot_id >= start_shot_id && a.shot_id <= end_shot_id)
            .collect()
    }

    /// Return all annotations with a star rating of at least `min_rating`.
    #[must_use]
    pub fn by_min_rating(&self, min_rating: u8) -> Vec<&ShotAnnotation> {
        self.annotations
            .values()
            .filter(|a| a.rating >= min_rating)
            .collect()
    }

    /// Return all annotations authored by the given author.
    #[must_use]
    pub fn by_author(&self, author: &str) -> Vec<&ShotAnnotation> {
        self.annotations
            .values()
            .filter(|a| a.author.as_deref() == Some(author))
            .collect()
    }

    /// Add a tag to an existing annotation, updating the reverse index.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::NotFound` if no annotation exists for `shot_id`.
    pub fn add_tag(&mut self, shot_id: u64, tag: impl Into<String>) -> ShotResult<()> {
        let tag = tag.into();
        let ann = self
            .annotations
            .get_mut(&shot_id)
            .ok_or_else(|| ShotError::NotFound(format!("shot {shot_id}")))?;
        ann.tags.insert(tag.clone());
        self.tag_index.entry(tag).or_default().insert(shot_id);
        Ok(())
    }

    /// Remove a tag from an existing annotation, updating the reverse index.
    ///
    /// Returns `true` if the tag existed and was removed.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::NotFound` if no annotation exists for `shot_id`.
    pub fn remove_tag(&mut self, shot_id: u64, tag: &str) -> ShotResult<bool> {
        let ann = self
            .annotations
            .get_mut(&shot_id)
            .ok_or_else(|| ShotError::NotFound(format!("shot {shot_id}")))?;
        let removed = ann.tags.remove(tag);
        if removed {
            if let Some(set) = self.tag_index.get_mut(tag) {
                set.remove(&shot_id);
            }
        }
        Ok(removed)
    }

    /// Update the notes of an existing annotation.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::NotFound` if no annotation exists for `shot_id`.
    pub fn update_notes(&mut self, shot_id: u64, notes: impl Into<String>) -> ShotResult<()> {
        let ann = self
            .annotations
            .get_mut(&shot_id)
            .ok_or_else(|| ShotError::NotFound(format!("shot {shot_id}")))?;
        ann.notes = notes.into();
        Ok(())
    }

    /// Set a metadata key–value pair on an existing annotation.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::NotFound` if no annotation exists for `shot_id`.
    pub fn set_metadata(
        &mut self,
        shot_id: u64,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> ShotResult<()> {
        let ann = self
            .annotations
            .get_mut(&shot_id)
            .ok_or_else(|| ShotError::NotFound(format!("shot {shot_id}")))?;
        ann.metadata.insert(key.into(), value.into());
        Ok(())
    }

    /// Serialise the entire store to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::SerializationError` if serialization fails.
    pub fn to_json(&self) -> ShotResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| ShotError::SerializationError(e.to_string()))
    }

    /// Deserialise an annotation store from a JSON string.
    ///
    /// The tag index is rebuilt from the loaded annotations to ensure
    /// consistency.
    ///
    /// # Errors
    ///
    /// Returns `ShotError::SerializationError` if parsing fails.
    pub fn from_json(json: &str) -> ShotResult<Self> {
        // Deserialise only the annotation map; rebuild the index.
        #[derive(Deserialize)]
        struct Raw {
            annotations: HashMap<u64, ShotAnnotation>,
        }
        let raw: Raw =
            serde_json::from_str(json).map_err(|e| ShotError::SerializationError(e.to_string()))?;

        let mut store = Self::new();
        for ann in raw.annotations.into_values() {
            store.insert(ann);
        }
        Ok(store)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate a hex colour string of the form `#RRGGBB`.
fn validate_hex_color(color: &str) -> ShotResult<()> {
    if color.len() == 7
        && color.starts_with('#')
        && color[1..].chars().all(|c| c.is_ascii_hexdigit())
    {
        Ok(())
    } else {
        Err(ShotError::InvalidMetadata(format!(
            "invalid hex colour '{color}'; expected format #RRGGBB"
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ann(shot_id: u64) -> ShotAnnotation {
        ShotAnnotation::new(shot_id)
    }

    // ---- ShotAnnotation construction ----

    #[test]
    fn test_new_annotation_defaults() {
        let ann = ShotAnnotation::new(42);
        assert_eq!(ann.shot_id, 42);
        assert!(ann.notes.is_empty());
        assert!(ann.tags.is_empty());
        assert!(ann.metadata.is_empty());
        assert_eq!(ann.rating, 0);
    }

    #[test]
    fn test_builder_with_notes() {
        let ann = ShotAnnotation::new(1).with_notes("A wide establishing shot.");
        assert_eq!(ann.notes, "A wide establishing shot.");
    }

    #[test]
    fn test_builder_with_tag() {
        let ann = ShotAnnotation::new(1)
            .with_tag("dialogue")
            .with_tag("favourite");
        assert!(ann.has_tag("dialogue"));
        assert!(ann.has_tag("favourite"));
        assert!(!ann.has_tag("action"));
    }

    #[test]
    fn test_builder_with_metadata() {
        let ann = ShotAnnotation::new(1).with_metadata("scene", "INT. OFFICE - DAY");
        assert_eq!(ann.get_metadata("scene"), Some("INT. OFFICE - DAY"));
        assert_eq!(ann.get_metadata("missing"), None);
    }

    #[test]
    fn test_builder_with_rating_clamped() {
        let ann = ShotAnnotation::new(1).with_rating(10);
        assert_eq!(ann.rating, 5, "rating should be clamped to 5");
    }

    #[test]
    fn test_builder_with_author() {
        let ann = ShotAnnotation::new(1).with_author("Alice");
        assert_eq!(ann.author.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_builder_with_color_label_valid() {
        let ann = ShotAnnotation::new(1)
            .with_color_label("#FF6B6B")
            .expect("valid hex colour");
        assert_eq!(ann.color_label.as_deref(), Some("#FF6B6B"));
    }

    #[test]
    fn test_builder_with_color_label_invalid() {
        let result = ShotAnnotation::new(1).with_color_label("red");
        assert!(result.is_err(), "non-hex colour should fail");
    }

    #[test]
    fn test_builder_with_color_label_wrong_length() {
        let result = ShotAnnotation::new(1).with_color_label("#FFF");
        assert!(result.is_err());
    }

    // ---- Metadata typed getters ----

    #[test]
    fn test_get_metadata_f64() {
        let ann = ShotAnnotation::new(1).with_metadata("duration", "3.14");
        let v = ann.get_metadata_f64("duration").expect("valid f64");
        assert!((v - 3.14).abs() < 1e-9);
    }

    #[test]
    fn test_get_metadata_f64_missing_key() {
        let ann = ShotAnnotation::new(1);
        assert!(ann.get_metadata_f64("missing").is_err());
    }

    #[test]
    fn test_get_metadata_f64_invalid_value() {
        let ann = ShotAnnotation::new(1).with_metadata("x", "not_a_number");
        assert!(ann.get_metadata_f64("x").is_err());
    }

    #[test]
    fn test_get_metadata_i64() {
        let ann = ShotAnnotation::new(1).with_metadata("frame", "120");
        assert_eq!(ann.get_metadata_i64("frame").expect("valid i64"), 120);
    }

    #[test]
    fn test_get_metadata_bool_true_variants() {
        for v in &["true", "True", "TRUE", "1", "yes"] {
            let ann = ShotAnnotation::new(1).with_metadata("flag", *v);
            assert!(
                ann.get_metadata_bool("flag").expect("valid bool"),
                "expected true for '{v}'"
            );
        }
    }

    #[test]
    fn test_get_metadata_bool_false_variants() {
        for v in &["false", "False", "0", "no"] {
            let ann = ShotAnnotation::new(1).with_metadata("flag", *v);
            assert!(
                !ann.get_metadata_bool("flag").expect("valid bool"),
                "expected false for '{v}'"
            );
        }
    }

    #[test]
    fn test_get_metadata_bool_invalid() {
        let ann = ShotAnnotation::new(1).with_metadata("flag", "maybe");
        assert!(ann.get_metadata_bool("flag").is_err());
    }

    // ---- Mutable tag ops ----

    #[test]
    fn test_add_remove_tag() {
        let mut ann = ShotAnnotation::new(1);
        ann.add_tag("vfx");
        assert!(ann.has_tag("vfx"));
        let removed = ann.remove_tag("vfx");
        assert!(removed);
        assert!(!ann.has_tag("vfx"));
    }

    #[test]
    fn test_remove_nonexistent_tag_returns_false() {
        let mut ann = ShotAnnotation::new(1);
        assert!(!ann.remove_tag("missing"));
    }

    // ---- AnnotationStore CRUD ----

    #[test]
    fn test_store_insert_and_get() {
        let mut store = AnnotationStore::new();
        let ann = ShotAnnotation::new(10).with_notes("wide");
        store.insert(ann);
        assert!(store.contains(10));
        let got = store.get(10).expect("should exist");
        assert_eq!(got.notes, "wide");
    }

    #[test]
    fn test_store_len_is_empty() {
        let mut store = AnnotationStore::new();
        assert!(store.is_empty());
        store.insert(make_ann(1));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_store_replace_updates_tag_index() {
        let mut store = AnnotationStore::new();
        let ann1 = ShotAnnotation::new(1).with_tag("old-tag");
        store.insert(ann1);
        // Replace with new annotation that has a different tag.
        let ann2 = ShotAnnotation::new(1).with_tag("new-tag");
        store.insert(ann2);

        // Old tag should no longer appear.
        assert!(store.by_tag("old-tag").is_empty());
        assert_eq!(store.by_tag("new-tag").len(), 1);
    }

    #[test]
    fn test_store_remove() {
        let mut store = AnnotationStore::new();
        store.insert(make_ann(5));
        let removed = store.remove(5);
        assert!(removed.is_some());
        assert!(!store.contains(5));
    }

    #[test]
    fn test_store_remove_nonexistent() {
        let mut store = AnnotationStore::new();
        assert!(store.remove(999).is_none());
    }

    // ---- Query API ----

    #[test]
    fn test_by_tag_empty_store() {
        let store = AnnotationStore::new();
        assert!(store.by_tag("dialogue").is_empty());
    }

    #[test]
    fn test_by_tag_returns_correct_shots() {
        let mut store = AnnotationStore::new();
        store.insert(ShotAnnotation::new(1).with_tag("dialogue"));
        store.insert(ShotAnnotation::new(2).with_tag("action"));
        store.insert(ShotAnnotation::new(3).with_tag("dialogue"));

        let hits = store.by_tag("dialogue");
        let ids: Vec<u64> = {
            let mut v: Vec<u64> = hits.iter().map(|a| a.shot_id).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn test_by_all_tags_and_query() {
        let mut store = AnnotationStore::new();
        store.insert(ShotAnnotation::new(1).with_tag("a").with_tag("b"));
        store.insert(ShotAnnotation::new(2).with_tag("a"));
        store.insert(ShotAnnotation::new(3).with_tag("b"));

        let hits = store.by_all_tags(&["a", "b"]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].shot_id, 1);
    }

    #[test]
    fn test_by_all_tags_empty_tag_list_returns_all() {
        let mut store = AnnotationStore::new();
        store.insert(make_ann(1));
        store.insert(make_ann(2));
        let hits = store.by_all_tags(&[]);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_by_any_tag_or_query() {
        let mut store = AnnotationStore::new();
        store.insert(ShotAnnotation::new(1).with_tag("vfx"));
        store.insert(ShotAnnotation::new(2).with_tag("sfx"));
        store.insert(ShotAnnotation::new(3).with_tag("music"));

        let hits = store.by_any_tag(&["vfx", "sfx"]);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_by_shot_range() {
        let mut store = AnnotationStore::new();
        for id in 0..10 {
            store.insert(make_ann(id));
        }
        let hits = store.by_shot_range(3, 6);
        assert_eq!(hits.len(), 4);
    }

    #[test]
    fn test_by_min_rating() {
        let mut store = AnnotationStore::new();
        store.insert(ShotAnnotation::new(1).with_rating(5));
        store.insert(ShotAnnotation::new(2).with_rating(3));
        store.insert(ShotAnnotation::new(3).with_rating(1));
        let hits = store.by_min_rating(3);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_by_author() {
        let mut store = AnnotationStore::new();
        store.insert(ShotAnnotation::new(1).with_author("Alice"));
        store.insert(ShotAnnotation::new(2).with_author("Bob"));
        store.insert(ShotAnnotation::new(3).with_author("Alice"));
        let hits = store.by_author("Alice");
        assert_eq!(hits.len(), 2);
    }

    // ---- Mutating via store API ----

    #[test]
    fn test_add_tag_via_store() {
        let mut store = AnnotationStore::new();
        store.insert(make_ann(1));
        store.add_tag(1, "new").expect("should succeed in test");
        assert!(store.get(1).expect("exists").has_tag("new"));
        // Tag index should be updated.
        assert_eq!(store.by_tag("new").len(), 1);
    }

    #[test]
    fn test_add_tag_missing_shot() {
        let mut store = AnnotationStore::new();
        assert!(store.add_tag(999, "x").is_err());
    }

    #[test]
    fn test_remove_tag_via_store() {
        let mut store = AnnotationStore::new();
        store.insert(ShotAnnotation::new(1).with_tag("vfx"));
        let removed = store.remove_tag(1, "vfx").expect("should succeed in test");
        assert!(removed);
        assert!(store.by_tag("vfx").is_empty());
    }

    #[test]
    fn test_remove_tag_missing_shot() {
        let mut store = AnnotationStore::new();
        assert!(store.remove_tag(999, "vfx").is_err());
    }

    #[test]
    fn test_update_notes() {
        let mut store = AnnotationStore::new();
        store.insert(make_ann(1));
        store
            .update_notes(1, "Updated notes")
            .expect("should succeed in test");
        assert_eq!(store.get(1).expect("exists").notes, "Updated notes");
    }

    #[test]
    fn test_update_notes_missing_shot() {
        let mut store = AnnotationStore::new();
        assert!(store.update_notes(99, "x").is_err());
    }

    #[test]
    fn test_set_metadata_via_store() {
        let mut store = AnnotationStore::new();
        store.insert(make_ann(1));
        store
            .set_metadata(1, "scene", "EXT. PARK - NOON")
            .expect("should succeed in test");
        assert_eq!(
            store.get(1).expect("exists").get_metadata("scene"),
            Some("EXT. PARK - NOON")
        );
    }

    // ---- Serialisation round-trip ----

    #[test]
    fn test_json_round_trip() {
        let mut store = AnnotationStore::new();
        store.insert(
            ShotAnnotation::new(1)
                .with_notes("Test note")
                .with_tag("dialogue")
                .with_metadata("scene", "INT. CAFE - DAY")
                .with_rating(4),
        );
        store.insert(ShotAnnotation::new(2).with_tag("action"));

        let json = store.to_json().expect("serialisation should succeed");
        let restored = AnnotationStore::from_json(&json).expect("deserialisation should succeed");

        assert_eq!(restored.len(), 2);
        let ann1 = restored.get(1).expect("shot 1 should exist");
        assert_eq!(ann1.notes, "Test note");
        assert!(ann1.has_tag("dialogue"));
        assert_eq!(ann1.get_metadata("scene"), Some("INT. CAFE - DAY"));
        assert_eq!(ann1.rating, 4);

        // Tag index should be reconstructed correctly.
        assert_eq!(restored.by_tag("dialogue").len(), 1);
        assert_eq!(restored.by_tag("action").len(), 1);
    }

    #[test]
    fn test_from_json_invalid() {
        assert!(AnnotationStore::from_json("{invalid json}").is_err());
    }

    // ---- iter ----

    #[test]
    fn test_store_iter() {
        let mut store = AnnotationStore::new();
        for id in 0..5 {
            store.insert(make_ann(id));
        }
        assert_eq!(store.iter().count(), 5);
    }

    // ---- validate_hex_color helper ----

    #[test]
    fn test_validate_hex_color_valid() {
        assert!(validate_hex_color("#000000").is_ok());
        assert!(validate_hex_color("#FFFFFF").is_ok());
        assert!(validate_hex_color("#1a2B3c").is_ok());
    }

    #[test]
    fn test_validate_hex_color_invalid_no_hash() {
        assert!(validate_hex_color("FF0000").is_err());
    }

    #[test]
    fn test_validate_hex_color_invalid_length() {
        assert!(validate_hex_color("#FFF").is_err());
        assert!(validate_hex_color("#FFFFFFFF").is_err());
    }

    #[test]
    fn test_validate_hex_color_invalid_chars() {
        assert!(validate_hex_color("#GGGGGG").is_err());
    }
}
