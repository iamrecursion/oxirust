//! Extended deduplication statistics: space savings, group statistics, action recommendations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use std::path::PathBuf;

/// Statistics for a single duplicate group.
#[derive(Debug, Clone)]
pub struct GroupStats {
    /// Number of files in this group.
    pub count: usize,
    /// Total size of all files in bytes.
    pub total_bytes: u64,
    /// Size of the representative (keep) file in bytes.
    pub representative_bytes: u64,
    /// Bytes that could be reclaimed by removing duplicates.
    pub reclaimable_bytes: u64,
    /// Average similarity score across all pairs.
    pub avg_similarity: f64,
}

impl GroupStats {
    /// Compute stats from a list of (path, size) pairs and the representative path.
    #[must_use]
    pub fn compute(
        members: &[(PathBuf, u64)],
        representative: &PathBuf,
        avg_similarity: f64,
    ) -> Self {
        let total_bytes: u64 = members.iter().map(|(_, s)| *s).sum();
        let representative_bytes = members
            .iter()
            .find(|(p, _)| p == representative)
            .map(|(_, s)| *s)
            .unwrap_or(0);
        let reclaimable_bytes = total_bytes.saturating_sub(representative_bytes);
        Self {
            count: members.len(),
            total_bytes,
            representative_bytes,
            reclaimable_bytes,
            avg_similarity,
        }
    }
}

/// Overall deduplication space savings report.
#[derive(Debug, Clone, Default)]
pub struct SpaceSavingsReport {
    /// Total number of files scanned.
    pub total_files: usize,
    /// Total bytes across all scanned files.
    pub total_bytes: u64,
    /// Number of duplicate groups found.
    pub duplicate_groups: usize,
    /// Total number of files that are duplicates (redundant copies).
    pub duplicate_files: usize,
    /// Total bytes reclaimable by removing duplicates.
    pub reclaimable_bytes: u64,
    /// Percentage of storage that could be freed.
    pub savings_percent: f64,
    /// Per-group stats.
    pub group_stats: Vec<GroupStats>,
}

impl SpaceSavingsReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a group's stats to the report.
    pub fn add_group(&mut self, stats: GroupStats) {
        self.duplicate_groups += 1;
        self.duplicate_files += stats.count.saturating_sub(1);
        self.reclaimable_bytes += stats.reclaimable_bytes;
        self.group_stats.push(stats);
    }

    /// Finalise the report by computing derived fields.
    pub fn finalise(&mut self, total_files: usize, total_bytes: u64) {
        self.total_files = total_files;
        self.total_bytes = total_bytes;
        self.savings_percent = if total_bytes > 0 {
            100.0 * self.reclaimable_bytes as f64 / total_bytes as f64
        } else {
            0.0
        };
    }

    /// Human-readable summary line.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} duplicate groups | {} redundant files | {} MB reclaimable ({:.1}%)",
            self.duplicate_groups,
            self.duplicate_files,
            self.reclaimable_bytes / (1024 * 1024),
            self.savings_percent,
        )
    }
}

/// Severity of a recommended action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionSeverity {
    /// Informational only.
    Info,
    /// Suggested optimisation.
    Suggestion,
    /// Strongly recommended.
    Warning,
    /// Critical (e.g. very large duplicate sets).
    Critical,
}

/// A recommended action for a duplicate group.
#[derive(Debug, Clone)]
pub struct ActionRecommendation {
    /// Severity of this recommendation.
    pub severity: ActionSeverity,
    /// Human-readable message.
    pub message: String,
    /// Optionally, a list of files recommended for deletion.
    pub files_to_remove: Vec<PathBuf>,
    /// The file to keep.
    pub keep: Option<PathBuf>,
}

impl ActionRecommendation {
    /// Build a recommendation from a group's stats.
    #[must_use]
    pub fn from_group(
        representative: Option<PathBuf>,
        members: &[PathBuf],
        reclaimable_bytes: u64,
    ) -> Self {
        let severity = if reclaimable_bytes > 1_000_000_000 {
            ActionSeverity::Critical
        } else if reclaimable_bytes > 100_000_000 {
            ActionSeverity::Warning
        } else if reclaimable_bytes > 0 {
            ActionSeverity::Suggestion
        } else {
            ActionSeverity::Info
        };

        let files_to_remove: Vec<PathBuf> = members
            .iter()
            .filter(|p| Some(*p) != representative.as_ref())
            .cloned()
            .collect();

        let message = format!(
            "Remove {} duplicate(s) to reclaim {} MB",
            files_to_remove.len(),
            reclaimable_bytes / (1024 * 1024),
        );

        Self {
            severity,
            message,
            files_to_remove,
            keep: representative,
        }
    }
}

/// Generate action recommendations for all groups.
#[must_use]
pub fn generate_recommendations(
    report: &SpaceSavingsReport,
    clusters: &[ClusterInfo],
) -> Vec<ActionRecommendation> {
    clusters
        .iter()
        .zip(report.group_stats.iter())
        .map(|(cluster, stats)| {
            ActionRecommendation::from_group(
                cluster.representative.clone(),
                &cluster.members,
                stats.reclaimable_bytes,
            )
        })
        .collect()
}

/// Lightweight cluster info for recommendation generation.
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    /// Members of this cluster.
    pub members: Vec<PathBuf>,
    /// The representative file to keep.
    pub representative: Option<PathBuf>,
}

impl ClusterInfo {
    /// Create a new cluster info.
    #[must_use]
    pub fn new(members: Vec<PathBuf>, representative: Option<PathBuf>) -> Self {
        Self {
            members,
            representative,
        }
    }
}

/// Histogram of similarity score distribution across all pairs.
#[derive(Debug, Clone)]
pub struct SimilarityHistogram {
    /// Bucket boundaries [0.0, 0.1, 0.2, ..., 1.0].
    pub buckets: Vec<u64>,
    /// Number of buckets.
    pub num_buckets: usize,
}

impl SimilarityHistogram {
    /// Create a new histogram with `num_buckets` equal-width buckets.
    #[must_use]
    pub fn new(num_buckets: usize) -> Self {
        Self {
            buckets: vec![0; num_buckets],
            num_buckets,
        }
    }

    /// Record a similarity score.
    pub fn record(&mut self, score: f64) {
        let score = score.clamp(0.0, 1.0);
        let idx = ((score * self.num_buckets as f64) as usize).min(self.num_buckets - 1);
        self.buckets[idx] += 1;
    }

    /// Total number of recorded samples.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.buckets.iter().sum()
    }

    /// Mode bucket index (the bucket with the most samples).
    #[must_use]
    pub fn mode_bucket(&self) -> usize {
        self.buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pb(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn test_group_stats_compute() {
        let members = vec![(pb("a.mp4"), 1000u64), (pb("b.mp4"), 2000u64)];
        let rep = pb("a.mp4");
        let stats = GroupStats::compute(&members, &rep, 0.95);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_bytes, 3000);
        assert_eq!(stats.representative_bytes, 1000);
        assert_eq!(stats.reclaimable_bytes, 2000);
        assert!((stats.avg_similarity - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_group_stats_missing_representative() {
        let members = vec![(pb("a.mp4"), 500u64)];
        let rep = pb("missing.mp4");
        let stats = GroupStats::compute(&members, &rep, 0.0);
        assert_eq!(stats.representative_bytes, 0);
        assert_eq!(stats.reclaimable_bytes, 500);
    }

    #[test]
    fn test_space_savings_report_add_and_finalise() {
        let mut report = SpaceSavingsReport::new();
        let stats = GroupStats {
            count: 3,
            total_bytes: 9_000_000,
            representative_bytes: 3_000_000,
            reclaimable_bytes: 6_000_000,
            avg_similarity: 0.98,
        };
        report.add_group(stats);
        report.finalise(10, 20_000_000);
        assert_eq!(report.duplicate_groups, 1);
        assert_eq!(report.duplicate_files, 2);
        assert_eq!(report.reclaimable_bytes, 6_000_000);
        assert!((report.savings_percent - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_space_savings_report_zero_total() {
        let mut report = SpaceSavingsReport::new();
        report.finalise(0, 0);
        assert_eq!(report.savings_percent, 0.0);
    }

    #[test]
    fn test_space_savings_report_summary() {
        let mut report = SpaceSavingsReport::new();
        let stats = GroupStats {
            count: 2,
            total_bytes: 2_097_152,
            representative_bytes: 1_048_576,
            reclaimable_bytes: 1_048_576,
            avg_similarity: 0.9,
        };
        report.add_group(stats);
        report.finalise(5, 10_485_760);
        let s = report.summary();
        assert!(s.contains("1 duplicate groups"));
        assert!(s.contains("1 redundant files"));
    }

    #[test]
    fn test_action_severity_ordering() {
        assert!(ActionSeverity::Info < ActionSeverity::Suggestion);
        assert!(ActionSeverity::Suggestion < ActionSeverity::Warning);
        assert!(ActionSeverity::Warning < ActionSeverity::Critical);
    }

    #[test]
    fn test_action_recommendation_critical() {
        let members = vec![pb("a.mp4"), pb("b.mp4")];
        let rec = ActionRecommendation::from_group(Some(pb("a.mp4")), &members, 2_000_000_000);
        assert_eq!(rec.severity, ActionSeverity::Critical);
        assert_eq!(rec.files_to_remove.len(), 1);
        assert_eq!(rec.keep, Some(pb("a.mp4")));
    }

    #[test]
    fn test_action_recommendation_suggestion() {
        let members = vec![pb("a.mp4"), pb("b.mp4")];
        let rec = ActionRecommendation::from_group(Some(pb("a.mp4")), &members, 50_000_000);
        assert_eq!(rec.severity, ActionSeverity::Suggestion);
    }

    #[test]
    fn test_action_recommendation_info() {
        let members = vec![pb("a.mp4")];
        let rec = ActionRecommendation::from_group(Some(pb("a.mp4")), &members, 0);
        assert_eq!(rec.severity, ActionSeverity::Info);
        assert!(rec.files_to_remove.is_empty());
    }

    #[test]
    fn test_generate_recommendations() {
        let mut report = SpaceSavingsReport::new();
        report.add_group(GroupStats {
            count: 2,
            total_bytes: 200,
            representative_bytes: 100,
            reclaimable_bytes: 100,
            avg_similarity: 0.9,
        });
        report.finalise(2, 200);
        let clusters = vec![ClusterInfo::new(
            vec![pb("a.mp4"), pb("b.mp4")],
            Some(pb("a.mp4")),
        )];
        let recs = generate_recommendations(&report, &clusters);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].files_to_remove.len(), 1);
    }

    #[test]
    fn test_similarity_histogram_record_and_total() {
        let mut h = SimilarityHistogram::new(10);
        h.record(0.0);
        h.record(0.5);
        h.record(0.5);
        h.record(1.0);
        assert_eq!(h.total(), 4);
    }

    #[test]
    fn test_similarity_histogram_clamp() {
        let mut h = SimilarityHistogram::new(10);
        h.record(-0.1); // clamps to 0.0
        h.record(1.5); // clamps to 1.0
        assert_eq!(h.total(), 2);
    }

    #[test]
    fn test_similarity_histogram_mode() {
        let mut h = SimilarityHistogram::new(10);
        h.record(0.95);
        h.record(0.96);
        h.record(0.97);
        h.record(0.1);
        // Bucket 9 (0.9-1.0) should be the mode.
        assert_eq!(h.mode_bucket(), 9);
    }
}
