//! Faceted search: facet counting, filter combinations, and drill-down.
//!
//! This module provides standalone facet computation that complements the
//! `facet` directory module.  It operates on a flat list of document attribute
//! maps and computes bucket counts for each attribute value.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Document attribute
// ---------------------------------------------------------------------------

/// A single attribute (field + value) describing a document dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Attribute {
    /// Attribute field name (e.g. "genre", "codec", "resolution").
    pub field: String,
    /// Attribute value (e.g. "action", "h264", "1080p").
    pub value: String,
}

impl Attribute {
    /// Create a new attribute.
    #[must_use]
    pub fn new(field: &str, value: &str) -> Self {
        Self {
            field: field.to_owned(),
            value: value.to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// Document record (for faceting)
// ---------------------------------------------------------------------------

/// A document with multi-valued attributes used during facet computation.
#[derive(Debug, Clone)]
pub struct FacetDocument {
    /// Document identifier.
    pub id: Uuid,
    /// Attributes associated with this document.
    pub attributes: Vec<Attribute>,
}

impl FacetDocument {
    /// Construct a new facet document.
    #[must_use]
    pub fn new(id: Uuid, attributes: Vec<Attribute>) -> Self {
        Self { id, attributes }
    }

    /// Returns `true` if the document has the given attribute.
    #[must_use]
    pub fn has_attribute(&self, attr: &Attribute) -> bool {
        self.attributes.contains(attr)
    }
}

// ---------------------------------------------------------------------------
// Facet bucket
// ---------------------------------------------------------------------------

/// A single bucket in a facet: a value and the count of matching documents.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetBucket {
    /// The facet value.
    pub value: String,
    /// Number of documents with this value for the parent field.
    pub count: usize,
}

impl FacetBucket {
    /// Create a new facet bucket.
    #[must_use]
    pub fn new(value: String, count: usize) -> Self {
        Self { value, count }
    }
}

// ---------------------------------------------------------------------------
// Facet (one field)
// ---------------------------------------------------------------------------

/// Computed facet for a single field: an ordered list of value buckets.
#[derive(Debug, Clone)]
pub struct Facet {
    /// Field name.
    pub field: String,
    /// Buckets, sorted by count descending.
    pub buckets: Vec<FacetBucket>,
}

impl Facet {
    /// Build a `Facet` from a map of `value → count` pairs.
    #[must_use]
    pub fn from_counts(field: &str, counts: HashMap<String, usize>) -> Self {
        let mut buckets: Vec<FacetBucket> = counts
            .into_iter()
            .map(|(v, c)| FacetBucket::new(v, c))
            .collect();
        buckets.sort_by(|a, b| b.count.cmp(&a.count).then(a.value.cmp(&b.value)));
        Self {
            field: field.to_owned(),
            buckets,
        }
    }

    /// Total number of documents covered by this facet.
    #[must_use]
    pub fn total(&self) -> usize {
        self.buckets.iter().map(|b| b.count).sum()
    }

    /// Find the count for a specific value.
    #[must_use]
    pub fn count_for(&self, value: &str) -> usize {
        self.buckets
            .iter()
            .find(|b| b.value == value)
            .map_or(0, |b| b.count)
    }
}

// ---------------------------------------------------------------------------
// Facet computer
// ---------------------------------------------------------------------------

/// Compute facets from a slice of documents.
///
/// Returns one `Facet` per distinct field found across all documents.
#[must_use]
pub fn compute_facets(docs: &[FacetDocument]) -> HashMap<String, Facet> {
    // Accumulate counts: field → value → count.
    let mut raw: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for doc in docs {
        for attr in &doc.attributes {
            *raw.entry(attr.field.clone())
                .or_default()
                .entry(attr.value.clone())
                .or_insert(0) += 1;
        }
    }
    raw.into_iter()
        .map(|(field, counts)| {
            let facet = Facet::from_counts(&field, counts);
            (field, facet)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Facet filter
// ---------------------------------------------------------------------------

/// A set of facet constraints used to filter a document collection.
#[derive(Debug, Clone, Default)]
pub struct FacetFilter {
    /// Required attribute values: field → set of acceptable values.
    /// A document must match at least one value in the set for each field.
    pub required: HashMap<String, Vec<String>>,
    /// Excluded attribute values: field → set of excluded values.
    /// A document must NOT have any of these.
    pub excluded: HashMap<String, Vec<String>>,
}

impl FacetFilter {
    /// Create an empty filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Require that the document has at least one of `values` for `field`.
    pub fn require(mut self, field: &str, values: Vec<&str>) -> Self {
        self.required.insert(
            field.to_owned(),
            values.into_iter().map(str::to_owned).collect(),
        );
        self
    }

    /// Exclude documents that have any of `values` for `field`.
    pub fn exclude(mut self, field: &str, values: Vec<&str>) -> Self {
        self.excluded.insert(
            field.to_owned(),
            values.into_iter().map(str::to_owned).collect(),
        );
        self
    }

    /// Returns `true` if `doc` satisfies this filter.
    #[must_use]
    pub fn matches(&self, doc: &FacetDocument) -> bool {
        // Check required constraints.
        for (field, acceptable) in &self.required {
            let doc_values: Vec<&str> = doc
                .attributes
                .iter()
                .filter(|a| &a.field == field)
                .map(|a| a.value.as_str())
                .collect();
            if !acceptable.iter().any(|v| doc_values.contains(&v.as_str())) {
                return false;
            }
        }

        // Check excluded constraints.
        for (field, bad_values) in &self.excluded {
            let doc_values: Vec<&str> = doc
                .attributes
                .iter()
                .filter(|a| &a.field == field)
                .map(|a| a.value.as_str())
                .collect();
            if bad_values.iter().any(|v| doc_values.contains(&v.as_str())) {
                return false;
            }
        }

        true
    }

    /// Apply this filter to a slice, returning matching document IDs.
    #[must_use]
    pub fn apply<'a>(&self, docs: &'a [FacetDocument]) -> Vec<&'a FacetDocument> {
        docs.iter().filter(|d| self.matches(d)).collect()
    }
}

// ---------------------------------------------------------------------------
// Drill-down helper
// ---------------------------------------------------------------------------

/// Compute facets for the subset of `docs` satisfying `filter`.
///
/// This enables "drill-down": apply a filter, then re-compute facets for the
/// filtered result set.
#[must_use]
pub fn drill_down(docs: &[FacetDocument], filter: &FacetFilter) -> HashMap<String, Facet> {
    let filtered: Vec<FacetDocument> = docs.iter().filter(|d| filter.matches(d)).cloned().collect();
    compute_facets(&filtered)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    fn attr(f: &str, v: &str) -> Attribute {
        Attribute::new(f, v)
    }

    fn doc(attrs: &[(&str, &str)]) -> FacetDocument {
        FacetDocument::new(uid(), attrs.iter().map(|(f, v)| attr(f, v)).collect())
    }

    // --- Attribute ---

    #[test]
    fn test_attribute_equality() {
        let a = attr("genre", "action");
        let b = attr("genre", "action");
        assert_eq!(a, b);
    }

    #[test]
    fn test_attribute_inequality() {
        let a = attr("genre", "action");
        let b = attr("genre", "comedy");
        assert_ne!(a, b);
    }

    // --- FacetDocument ---

    #[test]
    fn test_facet_document_has_attribute() {
        let d = doc(&[("genre", "action"), ("codec", "h264")]);
        assert!(d.has_attribute(&attr("genre", "action")));
        assert!(!d.has_attribute(&attr("genre", "comedy")));
    }

    // --- FacetBucket ---

    #[test]
    fn test_facet_bucket_new() {
        let b = FacetBucket::new("action".to_owned(), 42);
        assert_eq!(b.value, "action");
        assert_eq!(b.count, 42);
    }

    // --- Facet ---

    #[test]
    fn test_facet_from_counts_sorted() {
        let mut counts = HashMap::new();
        counts.insert("comedy".to_owned(), 5usize);
        counts.insert("action".to_owned(), 10usize);
        counts.insert("drama".to_owned(), 3usize);
        let facet = Facet::from_counts("genre", counts);
        assert_eq!(facet.buckets[0].value, "action");
        assert_eq!(facet.buckets[0].count, 10);
    }

    #[test]
    fn test_facet_total() {
        let mut counts = HashMap::new();
        counts.insert("a".to_owned(), 3usize);
        counts.insert("b".to_owned(), 7usize);
        let facet = Facet::from_counts("x", counts);
        assert_eq!(facet.total(), 10);
    }

    #[test]
    fn test_facet_count_for() {
        let mut counts = HashMap::new();
        counts.insert("h264".to_owned(), 8usize);
        let facet = Facet::from_counts("codec", counts);
        assert_eq!(facet.count_for("h264"), 8);
        assert_eq!(facet.count_for("hevc"), 0);
    }

    // --- compute_facets ---

    #[test]
    fn test_compute_facets_basic() {
        let docs = vec![
            doc(&[("genre", "action")]),
            doc(&[("genre", "action")]),
            doc(&[("genre", "comedy")]),
        ];
        let facets = compute_facets(&docs);
        let genre = facets.get("genre").expect("genre facet should exist");
        assert_eq!(genre.count_for("action"), 2);
        assert_eq!(genre.count_for("comedy"), 1);
    }

    #[test]
    fn test_compute_facets_empty() {
        let facets = compute_facets(&[]);
        assert!(facets.is_empty());
    }

    #[test]
    fn test_compute_facets_multiple_fields() {
        let docs = vec![
            doc(&[("genre", "action"), ("codec", "h264")]),
            doc(&[("genre", "drama"), ("codec", "hevc")]),
        ];
        let facets = compute_facets(&docs);
        assert!(facets.contains_key("genre"));
        assert!(facets.contains_key("codec"));
    }

    // --- FacetFilter ---

    #[test]
    fn test_filter_require_passes() {
        let d = doc(&[("genre", "action")]);
        let f = FacetFilter::new().require("genre", vec!["action", "comedy"]);
        assert!(f.matches(&d));
    }

    #[test]
    fn test_filter_require_fails() {
        let d = doc(&[("genre", "drama")]);
        let f = FacetFilter::new().require("genre", vec!["action"]);
        assert!(!f.matches(&d));
    }

    #[test]
    fn test_filter_exclude_passes() {
        let d = doc(&[("genre", "comedy")]);
        let f = FacetFilter::new().exclude("genre", vec!["action"]);
        assert!(f.matches(&d));
    }

    #[test]
    fn test_filter_exclude_fails() {
        let d = doc(&[("genre", "action")]);
        let f = FacetFilter::new().exclude("genre", vec!["action"]);
        assert!(!f.matches(&d));
    }

    #[test]
    fn test_filter_apply() {
        let docs = vec![
            doc(&[("genre", "action")]),
            doc(&[("genre", "drama")]),
            doc(&[("genre", "action")]),
        ];
        let f = FacetFilter::new().require("genre", vec!["action"]);
        let results = f.apply(&docs);
        assert_eq!(results.len(), 2);
    }

    // --- drill_down ---

    #[test]
    fn test_drill_down_refines_facets() {
        let docs = vec![
            doc(&[("genre", "action"), ("codec", "h264")]),
            doc(&[("genre", "drama"), ("codec", "hevc")]),
            doc(&[("genre", "action"), ("codec", "hevc")]),
        ];
        let filter = FacetFilter::new().require("genre", vec!["action"]);
        let facets = drill_down(&docs, &filter);
        // Only action docs remain: codec facet should show h264=1, hevc=1
        let codec = facets.get("codec").expect("codec facet");
        assert_eq!(codec.total(), 2);
    }
}
