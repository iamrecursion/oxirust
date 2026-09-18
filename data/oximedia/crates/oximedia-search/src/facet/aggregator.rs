//! Faceted search aggregation.
//!
//! Accumulates facet counts across a document corpus and provides filtering
//! utilities so that application code can narrow a result set by one or more
//! facet values.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A single value within a facet field, together with its document count and
/// its current selection state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetValue {
    /// The string value of the facet (e.g. `"h264"`, `"1920x1080"`).
    pub value: String,
    /// Number of documents that carry this value for the enclosing field.
    pub count: u64,
    /// `true` when this value has been selected (active filter).
    pub selected: bool,
}

impl FacetValue {
    /// Creates a new, unselected `FacetValue` with the given count.
    #[must_use]
    pub fn new(value: impl Into<String>, count: u64) -> Self {
        Self {
            value: value.into(),
            count,
            selected: false,
        }
    }
}

/// All values observed for one facet field.
#[derive(Debug, Clone)]
pub struct FacetField {
    /// Name of the field (e.g. `"codec"`, `"resolution"`).
    pub name: String,
    /// Values sorted by descending count.
    pub values: Vec<FacetValue>,
    /// Sum of all value counts (≥ the number of documents because a single
    /// document may carry multiple values for the same field).
    pub total_count: u64,
}

impl FacetField {
    /// Creates an empty `FacetField`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            total_count: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FacetAggregator
// ──────────────────────────────────────────────────────────────────────────────

/// Accumulates per-field value counts across many documents and produces
/// aggregated [`FacetField`] summaries.
///
/// # Example
///
/// ```
/// use oximedia_search::facet::aggregator::FacetAggregator;
/// use std::collections::HashMap;
///
/// let mut agg = FacetAggregator::new();
/// let mut facets = HashMap::new();
/// facets.insert("codec".to_string(), vec!["h264".to_string()]);
/// agg.add_document(1, facets);
/// let result = agg.aggregate();
/// assert!(result.contains_key("codec"));
/// ```
#[derive(Default)]
pub struct FacetAggregator {
    /// field → (value → count)
    counts: HashMap<String, HashMap<String, u64>>,
}

impl FacetAggregator {
    /// Creates a new, empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Records the facet values for a single document.
    ///
    /// `doc_id` is accepted for API symmetry and future use (e.g. deduplication)
    /// but is not stored.
    pub fn add_document(&mut self, _doc_id: u64, facets: HashMap<String, Vec<String>>) {
        for (field, values) in facets {
            let field_map = self.counts.entry(field).or_default();
            for value in values {
                *field_map.entry(value).or_insert(0) += 1;
            }
        }
    }

    /// Produces a map of field name → [`FacetField`] sorted by descending count.
    #[must_use]
    pub fn aggregate(&self) -> HashMap<String, FacetField> {
        self.counts
            .iter()
            .map(|(field_name, value_map)| {
                let mut values: Vec<FacetValue> = value_map
                    .iter()
                    .map(|(v, &c)| FacetValue::new(v.clone(), c))
                    .collect();
                values.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
                let total_count: u64 = values.iter().map(|v| v.count).sum();
                let ff = FacetField {
                    name: field_name.clone(),
                    values,
                    total_count,
                };
                (field_name.clone(), ff)
            })
            .collect()
    }

    /// Clears all accumulated counts.
    pub fn clear(&mut self) {
        self.counts.clear();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FacetFilter
// ──────────────────────────────────────────────────────────────────────────────

/// A filter that selects documents whose facets contain at least one of the
/// listed values for the named field.
#[derive(Debug, Clone)]
pub struct FacetFilter {
    /// Field to filter on.
    pub field: String,
    /// Accepted values (OR semantics: any match passes).
    pub values: Vec<String>,
}

impl FacetFilter {
    /// Creates a new filter.
    #[must_use]
    pub fn new(
        field: impl Into<String>,
        values: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            field: field.into(),
            values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns `true` when the document's facets satisfy this filter.
    ///
    /// If the filter's value list is empty, every document passes.
    #[must_use]
    pub fn matches(&self, doc_facets: &HashMap<String, Vec<String>>) -> bool {
        if self.values.is_empty() {
            return true;
        }
        match doc_facets.get(&self.field) {
            None => false,
            Some(doc_values) => doc_values.iter().any(|v| self.values.contains(v)),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// apply_filters
// ──────────────────────────────────────────────────────────────────────────────

/// Filters a list of document IDs against a set of facet filters.
///
/// `doc_facets` maps each document ID to its facet map.  A document passes
/// when it satisfies **all** of the supplied filters.
#[must_use]
pub fn apply_filters(
    filters: &[FacetFilter],
    results: &[u64],
    doc_facets: &HashMap<u64, HashMap<String, Vec<String>>>,
) -> Vec<u64> {
    if filters.is_empty() {
        return results.to_vec();
    }

    results
        .iter()
        .filter(|&&doc_id| {
            let empty = HashMap::new();
            let facets = doc_facets.get(&doc_id).unwrap_or(&empty);
            filters.iter().all(|f| f.matches(facets))
        })
        .copied()
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_facets(pairs: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(k, vs)| (k.to_string(), vs.iter().map(|v| v.to_string()).collect()))
            .collect()
    }

    // ── FacetValue ──

    #[test]
    fn test_facet_value_new() {
        let fv = FacetValue::new("h264", 42);
        assert_eq!(fv.value, "h264");
        assert_eq!(fv.count, 42);
        assert!(!fv.selected);
    }

    // ── FacetAggregator ──

    #[test]
    fn test_aggregator_single_document() {
        let mut agg = FacetAggregator::new();
        agg.add_document(1, make_facets(&[("codec", &["h264"])]));
        let result = agg.aggregate();
        let ff = result.get("codec").expect("codec field missing");
        assert_eq!(ff.values.len(), 1);
        assert_eq!(ff.values[0].value, "h264");
        assert_eq!(ff.values[0].count, 1);
        assert_eq!(ff.total_count, 1);
    }

    #[test]
    fn test_aggregator_multiple_documents() {
        let mut agg = FacetAggregator::new();
        for (id, codec) in [(1, "h264"), (2, "h264"), (3, "av1")] {
            agg.add_document(id, make_facets(&[("codec", &[codec])]));
        }
        let result = agg.aggregate();
        let ff = result.get("codec").expect("should succeed in test");
        // h264 should come first (count = 2).
        assert_eq!(ff.values[0].value, "h264");
        assert_eq!(ff.values[0].count, 2);
        assert_eq!(ff.total_count, 3);
    }

    #[test]
    fn test_aggregator_multiple_values_per_doc() {
        let mut agg = FacetAggregator::new();
        agg.add_document(1, make_facets(&[("tags", &["nature", "wildlife"])]));
        let result = agg.aggregate();
        let ff = result.get("tags").expect("should succeed in test");
        assert_eq!(ff.total_count, 2);
        assert_eq!(ff.values.len(), 2);
    }

    #[test]
    fn test_aggregator_multiple_fields() {
        let mut agg = FacetAggregator::new();
        agg.add_document(
            1,
            make_facets(&[("codec", &["h264"]), ("resolution", &["1920x1080"])]),
        );
        let result = agg.aggregate();
        assert!(result.contains_key("codec"));
        assert!(result.contains_key("resolution"));
    }

    #[test]
    fn test_aggregator_clear() {
        let mut agg = FacetAggregator::new();
        agg.add_document(1, make_facets(&[("codec", &["h264"])]));
        agg.clear();
        let result = agg.aggregate();
        assert!(result.is_empty());
    }

    // ── FacetFilter ──

    #[test]
    fn test_facet_filter_matches() {
        let f = FacetFilter::new("codec", ["h264", "av1"]);
        let doc = make_facets(&[("codec", &["h264"])]);
        assert!(f.matches(&doc));
    }

    #[test]
    fn test_facet_filter_no_match() {
        let f = FacetFilter::new("codec", ["vp9"]);
        let doc = make_facets(&[("codec", &["h264"])]);
        assert!(!f.matches(&doc));
    }

    #[test]
    fn test_facet_filter_missing_field() {
        let f = FacetFilter::new("codec", ["h264"]);
        let doc = make_facets(&[("resolution", &["1920x1080"])]);
        assert!(!f.matches(&doc));
    }

    #[test]
    fn test_facet_filter_empty_values_always_passes() {
        let f = FacetFilter::new("codec", Vec::<String>::new());
        let doc = make_facets(&[]);
        assert!(f.matches(&doc));
    }

    // ── apply_filters ──

    #[test]
    fn test_apply_filters_basic() {
        let mut doc_facets: HashMap<u64, HashMap<String, Vec<String>>> = HashMap::new();
        doc_facets.insert(1, make_facets(&[("codec", &["h264"])]));
        doc_facets.insert(2, make_facets(&[("codec", &["av1"])]));
        doc_facets.insert(3, make_facets(&[("codec", &["h264"])]));

        let filters = vec![FacetFilter::new("codec", ["h264"])];
        let results = apply_filters(&filters, &[1, 2, 3], &doc_facets);
        assert_eq!(results, vec![1, 3]);
    }

    #[test]
    fn test_apply_filters_no_filters() {
        let doc_facets: HashMap<u64, HashMap<String, Vec<String>>> = HashMap::new();
        let results = apply_filters(&[], &[1, 2, 3], &doc_facets);
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_apply_filters_and_semantics() {
        let mut doc_facets: HashMap<u64, HashMap<String, Vec<String>>> = HashMap::new();
        // Doc 1: h264 + 1080p
        doc_facets.insert(1, make_facets(&[("codec", &["h264"]), ("res", &["1080p"])]));
        // Doc 2: h264 + 720p
        doc_facets.insert(2, make_facets(&[("codec", &["h264"]), ("res", &["720p"])]));
        // Doc 3: av1 + 1080p
        doc_facets.insert(3, make_facets(&[("codec", &["av1"]), ("res", &["1080p"])]));

        let filters = vec![
            FacetFilter::new("codec", ["h264"]),
            FacetFilter::new("res", ["1080p"]),
        ];
        let results = apply_filters(&filters, &[1, 2, 3], &doc_facets);
        assert_eq!(results, vec![1]);
    }
}
