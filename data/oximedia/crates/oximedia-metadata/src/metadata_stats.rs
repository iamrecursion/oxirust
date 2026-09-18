#![allow(dead_code)]
//! Metadata statistics and analytics.
//!
//! Computes aggregate statistics across collections of metadata records,
//! such as field coverage, value distributions, and completeness scores.

use std::collections::{HashMap, HashSet};

/// Summary statistics for a single metadata field across a collection.
#[derive(Debug, Clone)]
pub struct FieldStats {
    /// The field name.
    pub name: String,
    /// Number of records that have this field populated.
    pub present_count: u64,
    /// Number of records that are missing this field.
    pub absent_count: u64,
    /// Minimum character length among non-empty values.
    pub min_length: Option<usize>,
    /// Maximum character length among non-empty values.
    pub max_length: Option<usize>,
    /// Sum of all value lengths (for computing average).
    total_length: u64,
    /// Distinct values seen (capped to avoid unbounded memory).
    distinct_values: HashSet<String>,
    /// Maximum number of distinct values to track.
    distinct_cap: usize,
}

impl FieldStats {
    /// Create a new field stats tracker.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            present_count: 0,
            absent_count: 0,
            min_length: None,
            max_length: None,
            total_length: 0,
            distinct_values: HashSet::new(),
            distinct_cap: 10_000,
        }
    }

    /// Record a present value for this field.
    pub fn record_value(&mut self, value: &str) {
        self.present_count += 1;
        let len = value.len();
        self.total_length += len as u64;

        self.min_length = Some(match self.min_length {
            Some(cur) => cur.min(len),
            None => len,
        });
        self.max_length = Some(match self.max_length {
            Some(cur) => cur.max(len),
            None => len,
        });

        if self.distinct_values.len() < self.distinct_cap {
            self.distinct_values.insert(value.to_string());
        }
    }

    /// Record an absent value for this field.
    pub fn record_absent(&mut self) {
        self.absent_count += 1;
    }

    /// Get the coverage ratio (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage(&self) -> f64 {
        let total = self.present_count + self.absent_count;
        if total == 0 {
            return 0.0;
        }
        self.present_count as f64 / total as f64
    }

    /// Get the average value length.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_length(&self) -> f64 {
        if self.present_count == 0 {
            return 0.0;
        }
        self.total_length as f64 / self.present_count as f64
    }

    /// Get the number of distinct values observed.
    pub fn distinct_count(&self) -> usize {
        self.distinct_values.len()
    }

    /// Get the total number of records examined.
    pub fn total_records(&self) -> u64 {
        self.present_count + self.absent_count
    }
}

/// A completeness score for a metadata record.
#[derive(Debug, Clone)]
pub struct CompletenessScore {
    /// The record identifier.
    pub record_id: String,
    /// Number of required fields that are present.
    pub required_present: u32,
    /// Total number of required fields.
    pub required_total: u32,
    /// Number of optional fields that are present.
    pub optional_present: u32,
    /// Total number of optional fields.
    pub optional_total: u32,
}

impl CompletenessScore {
    /// Create a new completeness score.
    pub fn new(record_id: &str) -> Self {
        Self {
            record_id: record_id.to_string(),
            required_present: 0,
            required_total: 0,
            optional_present: 0,
            optional_total: 0,
        }
    }

    /// Compute the overall completeness as a percentage (0.0 to 100.0).
    ///
    /// Required fields are weighted 2x compared to optional fields.
    #[allow(clippy::cast_precision_loss)]
    pub fn score(&self) -> f64 {
        let req_weight = 2.0_f64;
        let opt_weight = 1.0_f64;
        let weighted_total =
            self.required_total as f64 * req_weight + self.optional_total as f64 * opt_weight;
        if weighted_total == 0.0 {
            return 100.0;
        }
        let weighted_present =
            self.required_present as f64 * req_weight + self.optional_present as f64 * opt_weight;
        (weighted_present / weighted_total) * 100.0
    }

    /// Check whether all required fields are present.
    pub fn is_complete(&self) -> bool {
        self.required_present == self.required_total
    }
}

/// Frequency distribution for a metadata field's values.
#[derive(Debug, Clone)]
pub struct ValueDistribution {
    /// The field name.
    pub field: String,
    /// Value -> count mapping.
    pub counts: HashMap<String, u64>,
    /// Total number of values observed.
    pub total: u64,
}

impl ValueDistribution {
    /// Create a new distribution tracker for a field.
    pub fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            counts: HashMap::new(),
            total: 0,
        }
    }

    /// Record a value occurrence.
    pub fn record(&mut self, value: &str) {
        *self.counts.entry(value.to_string()).or_insert(0) += 1;
        self.total += 1;
    }

    /// Get the top-N most frequent values.
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut entries: Vec<(String, u64)> = self.counts.clone().into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }

    /// Get the frequency of a specific value as a ratio (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn frequency(&self, value: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let count = self.counts.get(value).copied().unwrap_or(0);
        count as f64 / self.total as f64
    }

    /// Get the number of unique values.
    pub fn unique_count(&self) -> usize {
        self.counts.len()
    }
}

/// Aggregated statistics across an entire metadata collection.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Per-field statistics.
    pub field_stats: HashMap<String, FieldStats>,
    /// Total records examined.
    pub total_records: u64,
    /// Names of fields considered required.
    required_fields: HashSet<String>,
    /// Names of fields considered optional.
    optional_fields: HashSet<String>,
}

impl CollectionStats {
    /// Create a new collection stats tracker.
    pub fn new(required: &[&str], optional: &[&str]) -> Self {
        Self {
            field_stats: HashMap::new(),
            total_records: 0,
            required_fields: required.iter().map(|s| (*s).to_string()).collect(),
            optional_fields: optional.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Record a metadata record (as field -> value pairs).
    pub fn record(&mut self, fields: &HashMap<String, String>) {
        self.total_records += 1;
        let all_tracked: HashSet<String> = self
            .required_fields
            .union(&self.optional_fields)
            .cloned()
            .collect();

        for field_name in &all_tracked {
            let stats = self
                .field_stats
                .entry(field_name.clone())
                .or_insert_with(|| FieldStats::new(field_name));
            if let Some(value) = fields.get(field_name) {
                stats.record_value(value);
            } else {
                stats.record_absent();
            }
        }
    }

    /// Compute a completeness score for a single record.
    pub fn completeness(
        &self,
        record_id: &str,
        fields: &HashMap<String, String>,
    ) -> CompletenessScore {
        let mut score = CompletenessScore::new(record_id);
        for req in &self.required_fields {
            score.required_total += 1;
            if fields.contains_key(req) {
                score.required_present += 1;
            }
        }
        for opt in &self.optional_fields {
            score.optional_total += 1;
            if fields.contains_key(opt) {
                score.optional_present += 1;
            }
        }
        score
    }

    /// Get the overall average coverage across all tracked fields.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_coverage(&self) -> f64 {
        if self.field_stats.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.field_stats.values().map(|s| s.coverage()).sum();
        sum / self.field_stats.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fields(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_field_stats_basic() {
        let mut fs = FieldStats::new("title");
        fs.record_value("Hello");
        fs.record_value("World!!");
        assert_eq!(fs.present_count, 2);
        assert_eq!(fs.min_length, Some(5));
        assert_eq!(fs.max_length, Some(7));
    }

    #[test]
    fn test_field_stats_coverage() {
        let mut fs = FieldStats::new("artist");
        fs.record_value("John");
        fs.record_absent();
        fs.record_absent();
        let cov = fs.coverage();
        assert!((cov - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_field_stats_avg_length() {
        let mut fs = FieldStats::new("genre");
        fs.record_value("Rock"); // 4
        fs.record_value("Pop"); // 3
        fs.record_value("Blues"); // 5
        let avg = fs.avg_length();
        assert!((avg - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_field_stats_empty_coverage() {
        let fs = FieldStats::new("empty");
        assert_eq!(fs.coverage(), 0.0);
        assert_eq!(fs.avg_length(), 0.0);
    }

    #[test]
    fn test_field_stats_distinct_count() {
        let mut fs = FieldStats::new("genre");
        fs.record_value("Rock");
        fs.record_value("Pop");
        fs.record_value("Rock");
        assert_eq!(fs.distinct_count(), 2);
    }

    #[test]
    fn test_completeness_score_all_present() {
        let mut score = CompletenessScore::new("rec1");
        score.required_total = 3;
        score.required_present = 3;
        score.optional_total = 2;
        score.optional_present = 2;
        assert_eq!(score.score(), 100.0);
        assert!(score.is_complete());
    }

    #[test]
    fn test_completeness_score_partial() {
        let mut score = CompletenessScore::new("rec2");
        score.required_total = 2;
        score.required_present = 1;
        score.optional_total = 2;
        score.optional_present = 0;
        // weighted: present = 1*2 + 0*1 = 2, total = 2*2 + 2*1 = 6
        let expected = (2.0 / 6.0) * 100.0;
        assert!((score.score() - expected).abs() < 1e-9);
        assert!(!score.is_complete());
    }

    #[test]
    fn test_value_distribution_top_n() {
        let mut dist = ValueDistribution::new("genre");
        dist.record("Rock");
        dist.record("Pop");
        dist.record("Rock");
        dist.record("Jazz");
        dist.record("Rock");
        let top = dist.top_n(2);
        assert_eq!(top[0].0, "Rock");
        assert_eq!(top[0].1, 3);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_value_distribution_frequency() {
        let mut dist = ValueDistribution::new("format");
        dist.record("MP3");
        dist.record("MP3");
        dist.record("FLAC");
        dist.record("WAV");
        let freq = dist.frequency("MP3");
        assert!((freq - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_value_distribution_empty() {
        let dist = ValueDistribution::new("empty");
        assert_eq!(dist.frequency("anything"), 0.0);
        assert_eq!(dist.unique_count(), 0);
    }

    #[test]
    fn test_collection_stats_record_and_coverage() {
        let mut stats = CollectionStats::new(&["title", "artist"], &["genre"]);
        stats.record(&make_fields(&[("title", "Song A"), ("artist", "Artist 1")]));
        stats.record(&make_fields(&[("title", "Song B")]));
        assert_eq!(stats.total_records, 2);
        let title_stats = stats
            .field_stats
            .get("title")
            .expect("should succeed in test");
        assert_eq!(title_stats.present_count, 2);
        let artist_stats = stats
            .field_stats
            .get("artist")
            .expect("should succeed in test");
        assert_eq!(artist_stats.present_count, 1);
        assert_eq!(artist_stats.absent_count, 1);
    }

    #[test]
    fn test_collection_completeness() {
        let stats = CollectionStats::new(&["title", "artist"], &["genre"]);
        let fields = make_fields(&[("title", "Song"), ("genre", "Rock")]);
        let cs = stats.completeness("r1", &fields);
        assert_eq!(cs.required_present, 1);
        assert_eq!(cs.required_total, 2);
        assert_eq!(cs.optional_present, 1);
        assert_eq!(cs.optional_total, 1);
        assert!(!cs.is_complete());
    }

    #[test]
    fn test_collection_avg_coverage() {
        let mut stats = CollectionStats::new(&["title"], &["artist"]);
        stats.record(&make_fields(&[("title", "A")]));
        stats.record(&make_fields(&[("title", "B"), ("artist", "C")]));
        // title coverage: 2/2 = 1.0, artist coverage: 1/2 = 0.5, avg = 0.75
        let avg = stats.avg_coverage();
        assert!((avg - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_field_stats_total_records() {
        let mut fs = FieldStats::new("test");
        fs.record_value("a");
        fs.record_absent();
        fs.record_value("b");
        assert_eq!(fs.total_records(), 3);
    }
}
