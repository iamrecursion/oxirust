//! Deduplication reporting: statistics, summaries, and formatted reports.

#![allow(dead_code)]

use std::collections::HashMap;

// ── DedupStat ────────────────────────────────────────────────────────────────

/// A single deduplication statistic value.
#[derive(Debug, Clone, PartialEq)]
pub enum DedupStat {
    /// A count of items.
    Count(u64),
    /// A size in bytes.
    Bytes(u64),
    /// A ratio (0.0 – 1.0).
    Ratio(f64),
    /// A plain label.
    Label(String),
}

impl DedupStat {
    /// Returns a human-readable name describing this statistic's type.
    #[must_use]
    pub fn stat_name(&self) -> &'static str {
        match self {
            Self::Count(_) => "count",
            Self::Bytes(_) => "bytes",
            Self::Ratio(_) => "ratio",
            Self::Label(_) => "label",
        }
    }

    /// Returns the inner u64 for a `Count` or `Bytes` variant, or `None`.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Count(v) | Self::Bytes(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the inner f64 for a `Ratio` variant, or `None`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Ratio(v) => Some(*v),
            _ => None,
        }
    }
}

// ── DedupSummary ──────────────────────────────────────────────────────────────

/// High-level summary computed from deduplication statistics.
#[derive(Debug, Clone)]
pub struct DedupSummary {
    /// Total files examined.
    pub total_files: u64,
    /// Number of unique files.
    pub unique_files: u64,
    /// Number of duplicate files (total - unique).
    pub duplicate_files: u64,
    /// Total size of all files in bytes.
    pub total_bytes: u64,
    /// Bytes that can be reclaimed by removing duplicates.
    pub reclaimable_bytes: u64,
}

impl DedupSummary {
    /// Returns the number of bytes that would be saved by removing duplicates.
    #[must_use]
    pub fn space_saved_bytes(&self) -> u64 {
        self.reclaimable_bytes
    }

    /// Returns the fraction of storage consumed by duplicates (0.0 – 1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn duplication_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.reclaimable_bytes as f64 / self.total_bytes as f64
    }

    /// Returns the number of duplicate files.
    #[must_use]
    pub fn duplicate_count(&self) -> u64 {
        self.duplicate_files
    }

    /// Returns a human-readable description.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{} / {} files are duplicates ({:.1}% duplication); {} bytes reclaimable",
            self.duplicate_files,
            self.total_files,
            self.duplication_ratio() * 100.0,
            self.reclaimable_bytes,
        )
    }
}

impl Default for DedupSummary {
    fn default() -> Self {
        Self {
            total_files: 0,
            unique_files: 0,
            duplicate_files: 0,
            total_bytes: 0,
            reclaimable_bytes: 0,
        }
    }
}

// ── DedupReport ───────────────────────────────────────────────────────────────

/// A complete deduplication report aggregating named statistics.
#[derive(Debug, Default)]
pub struct DedupReport {
    stats: HashMap<String, DedupStat>,
}

impl DedupReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named statistic to the report (overwrites if name exists).
    pub fn add_stat(&mut self, name: impl Into<String>, stat: DedupStat) {
        self.stats.insert(name.into(), stat);
    }

    /// Retrieve a statistic by name.
    #[must_use]
    pub fn get_stat(&self, name: &str) -> Option<&DedupStat> {
        self.stats.get(name)
    }

    /// Returns the total number of statistics stored.
    #[must_use]
    pub fn stat_count(&self) -> usize {
        self.stats.len()
    }

    /// Generate a `DedupSummary` from the standard named statistics.
    ///
    /// Expects the following keys to be present (as `Count` or `Bytes`):
    /// - `"total_files"` – `Count`
    /// - `"unique_files"` – `Count`
    /// - `"total_bytes"` – `Bytes`
    /// - `"reclaimable_bytes"` – `Bytes`
    #[must_use]
    pub fn generate_summary(&self) -> DedupSummary {
        let total_files = self
            .stats
            .get("total_files")
            .and_then(DedupStat::as_u64)
            .unwrap_or(0);
        let unique_files = self
            .stats
            .get("unique_files")
            .and_then(DedupStat::as_u64)
            .unwrap_or(0);
        let total_bytes = self
            .stats
            .get("total_bytes")
            .and_then(DedupStat::as_u64)
            .unwrap_or(0);
        let reclaimable_bytes = self
            .stats
            .get("reclaimable_bytes")
            .and_then(DedupStat::as_u64)
            .unwrap_or(0);

        DedupSummary {
            total_files,
            unique_files,
            duplicate_files: total_files.saturating_sub(unique_files),
            total_bytes,
            reclaimable_bytes,
        }
    }

    /// Return all stat names, sorted alphabetically.
    #[must_use]
    pub fn stat_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.stats.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

// ── Per-group disk space savings estimation ──────────────────────────────────

/// Estimated disk-space savings for a single duplicate group.
#[derive(Debug, Clone)]
pub struct GroupSavings {
    /// Total bytes across all files in the group.
    pub total_bytes: u64,
    /// Size of the largest file (the one to keep).
    pub largest_file_bytes: u64,
    /// Bytes recoverable by removing all but the largest file.
    pub recoverable_bytes: u64,
    /// Number of files in the group.
    pub file_count: usize,
    /// Human-readable summary (e.g., "3 files, 15.2 MB recoverable").
    pub description: String,
}

/// Disk-space savings summary across all duplicate groups.
#[derive(Debug, Clone)]
pub struct SavingsSummary {
    /// Per-group savings details.
    pub groups: Vec<GroupSavings>,
    /// Grand total of recoverable bytes across all groups.
    pub total_recoverable_bytes: u64,
    /// Grand total bytes of all files involved in duplicate groups.
    pub total_bytes_in_duplicates: u64,
    /// Total number of duplicate files (files that would be removed).
    pub total_removable_files: usize,
}

impl GroupSavings {
    /// Potential disk space savings for this group, in bytes.
    ///
    /// Equivalent to `recoverable_bytes`: the total bytes across all files
    /// minus the size of the largest file (which is the one to keep).
    ///
    /// Provided as a named accessor to match the terminology used in the
    /// deduplication report API (`potential_savings_bytes`).
    #[must_use]
    pub fn potential_savings_bytes(&self) -> u64 {
        self.recoverable_bytes
    }
}

impl SavingsSummary {
    /// Human-readable summary string.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{} duplicate groups, {} removable files, {} recoverable out of {} total",
            self.groups.len(),
            self.total_removable_files,
            format_bytes_compact(self.total_recoverable_bytes),
            format_bytes_compact(self.total_bytes_in_duplicates),
        )
    }

    /// Fraction of total duplicate bytes that are recoverable (0.0 - 1.0).
    #[must_use]
    pub fn recovery_ratio(&self) -> f64 {
        if self.total_bytes_in_duplicates == 0 {
            return 0.0;
        }
        self.total_recoverable_bytes as f64 / self.total_bytes_in_duplicates as f64
    }

    /// Grand total of potential disk-space savings across all groups, in bytes.
    ///
    /// This is the sum of [`GroupSavings::potential_savings_bytes`] for every
    /// group — i.e., how many bytes would be freed if one copy of each
    /// duplicate group were kept and all others removed.
    ///
    /// Equivalent to `total_recoverable_bytes` but provided as an explicit
    /// method so callers can use a consistent vocabulary.
    #[must_use]
    pub fn total_potential_savings_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(GroupSavings::potential_savings_bytes)
            .fold(0u64, u64::saturating_add)
    }
}

/// Estimate disk-space savings for a list of duplicate groups.
///
/// Each group is a list of file paths. The function reads file sizes from
/// the filesystem. Files that cannot be stat'd are assumed to have 0 bytes.
///
/// # Arguments
/// * `groups` - Duplicate groups, each a slice of file path strings.
#[must_use]
pub fn estimate_savings(groups: &[Vec<String>]) -> SavingsSummary {
    let mut group_savings = Vec::with_capacity(groups.len());
    let mut grand_total_recoverable = 0u64;
    let mut grand_total_bytes = 0u64;
    let mut grand_removable = 0usize;

    for file_paths in groups {
        if file_paths.len() < 2 {
            continue;
        }

        let sizes: Vec<u64> = file_paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .collect();

        let total: u64 = sizes.iter().sum();
        let largest = sizes.iter().copied().max().unwrap_or(0);
        let recoverable = total.saturating_sub(largest);
        let removable = file_paths.len() - 1;

        let desc = format!(
            "{} files, {} recoverable",
            file_paths.len(),
            format_bytes_compact(recoverable),
        );

        group_savings.push(GroupSavings {
            total_bytes: total,
            largest_file_bytes: largest,
            recoverable_bytes: recoverable,
            file_count: file_paths.len(),
            description: desc,
        });

        grand_total_recoverable += recoverable;
        grand_total_bytes += total;
        grand_removable += removable;
    }

    SavingsSummary {
        groups: group_savings,
        total_recoverable_bytes: grand_total_recoverable,
        total_bytes_in_duplicates: grand_total_bytes,
        total_removable_files: grand_removable,
    }
}

/// Estimate savings from a `DedupSummary`.
///
/// Computes recovery ratio and total recoverable bytes from the summary stats.
#[must_use]
pub fn estimate_savings_from_summary(summary: &DedupSummary) -> GroupSavings {
    GroupSavings {
        total_bytes: summary.total_bytes,
        largest_file_bytes: summary
            .total_bytes
            .saturating_sub(summary.reclaimable_bytes),
        recoverable_bytes: summary.reclaimable_bytes,
        file_count: summary.total_files as usize,
        description: format!(
            "{} total files, {} recoverable",
            summary.total_files,
            format_bytes_compact(summary.reclaimable_bytes),
        ),
    }
}

/// Format bytes in a compact human-readable form.
fn format_bytes_compact(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_stat_name_count() {
        let s = DedupStat::Count(42);
        assert_eq!(s.stat_name(), "count");
    }

    #[test]
    fn test_dedup_stat_name_bytes() {
        let s = DedupStat::Bytes(1024);
        assert_eq!(s.stat_name(), "bytes");
    }

    #[test]
    fn test_dedup_stat_name_ratio() {
        let s = DedupStat::Ratio(0.5);
        assert_eq!(s.stat_name(), "ratio");
    }

    #[test]
    fn test_dedup_stat_name_label() {
        let s = DedupStat::Label("hello".to_string());
        assert_eq!(s.stat_name(), "label");
    }

    #[test]
    fn test_dedup_stat_as_u64() {
        assert_eq!(DedupStat::Count(99).as_u64(), Some(99));
        assert_eq!(DedupStat::Bytes(512).as_u64(), Some(512));
        assert_eq!(DedupStat::Ratio(0.5).as_u64(), None);
    }

    #[test]
    fn test_dedup_stat_as_f64() {
        assert_eq!(DedupStat::Ratio(0.75).as_f64(), Some(0.75));
        assert_eq!(DedupStat::Count(10).as_f64(), None);
    }

    #[test]
    fn test_summary_space_saved_bytes() {
        let s = DedupSummary {
            total_files: 100,
            unique_files: 60,
            duplicate_files: 40,
            total_bytes: 10_000,
            reclaimable_bytes: 4_000,
        };
        assert_eq!(s.space_saved_bytes(), 4_000);
    }

    #[test]
    fn test_summary_duplication_ratio() {
        let s = DedupSummary {
            total_files: 100,
            unique_files: 50,
            duplicate_files: 50,
            total_bytes: 1_000,
            reclaimable_bytes: 500,
        };
        assert!((s.duplication_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_summary_duplication_ratio_zero_bytes() {
        let s = DedupSummary::default();
        assert_eq!(s.duplication_ratio(), 0.0);
    }

    #[test]
    fn test_report_add_and_get_stat() {
        let mut r = DedupReport::new();
        r.add_stat("total_files", DedupStat::Count(200));
        assert_eq!(r.get_stat("total_files"), Some(&DedupStat::Count(200)));
        assert_eq!(r.get_stat("nonexistent"), None);
    }

    #[test]
    fn test_report_stat_count() {
        let mut r = DedupReport::new();
        r.add_stat("a", DedupStat::Count(1));
        r.add_stat("b", DedupStat::Bytes(2));
        assert_eq!(r.stat_count(), 2);
    }

    #[test]
    fn test_report_generate_summary() {
        let mut r = DedupReport::new();
        r.add_stat("total_files", DedupStat::Count(100));
        r.add_stat("unique_files", DedupStat::Count(70));
        r.add_stat("total_bytes", DedupStat::Bytes(10_000));
        r.add_stat("reclaimable_bytes", DedupStat::Bytes(3_000));
        let s = r.generate_summary();
        assert_eq!(s.total_files, 100);
        assert_eq!(s.unique_files, 70);
        assert_eq!(s.duplicate_files, 30);
        assert_eq!(s.space_saved_bytes(), 3_000);
    }

    #[test]
    fn test_report_stat_names_sorted() {
        let mut r = DedupReport::new();
        r.add_stat("zebra", DedupStat::Count(1));
        r.add_stat("apple", DedupStat::Count(2));
        r.add_stat("mango", DedupStat::Count(3));
        let names = r.stat_names();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn test_report_overwrite_stat() {
        let mut r = DedupReport::new();
        r.add_stat("x", DedupStat::Count(1));
        r.add_stat("x", DedupStat::Count(99));
        assert_eq!(r.get_stat("x"), Some(&DedupStat::Count(99)));
        assert_eq!(r.stat_count(), 1);
    }

    #[test]
    fn test_summary_description_contains_percentage() {
        let s = DedupSummary {
            total_files: 10,
            unique_files: 8,
            duplicate_files: 2,
            total_bytes: 1000,
            reclaimable_bytes: 200,
        };
        let desc = s.description();
        assert!(desc.contains('%'));
        assert!(desc.contains("200"));
    }

    // ---- GroupSavings / SavingsSummary tests ----

    #[test]
    fn test_estimate_savings_empty() {
        let summary = estimate_savings(&[]);
        assert!(summary.groups.is_empty());
        assert_eq!(summary.total_recoverable_bytes, 0);
        assert_eq!(summary.total_removable_files, 0);
    }

    #[test]
    fn test_estimate_savings_single_file_group_skipped() {
        let groups = vec![vec!["only_one.mp4".to_string()]];
        let summary = estimate_savings(&groups);
        assert!(summary.groups.is_empty());
    }

    #[test]
    fn test_estimate_savings_nonexistent_files() {
        let groups = vec![vec![
            "/nonexistent/a.mp4".to_string(),
            "/nonexistent/b.mp4".to_string(),
        ]];
        let summary = estimate_savings(&groups);
        assert_eq!(summary.groups.len(), 1);
        // All sizes are 0 since files don't exist
        assert_eq!(summary.groups[0].total_bytes, 0);
        assert_eq!(summary.groups[0].recoverable_bytes, 0);
    }

    #[test]
    fn test_estimate_savings_with_temp_files() {
        let dir = std::env::temp_dir().join("oximedia_dedup_savings_test");
        let _ = std::fs::create_dir_all(&dir);

        let file_a = dir.join("dup_a.bin");
        let file_b = dir.join("dup_b.bin");
        let file_c = dir.join("dup_c.bin");

        // Write files of known sizes
        std::fs::write(&file_a, &[0u8; 1000]).expect("write a");
        std::fs::write(&file_b, &[0u8; 2000]).expect("write b");
        std::fs::write(&file_c, &[0u8; 500]).expect("write c");

        let groups = vec![vec![
            file_a.to_string_lossy().to_string(),
            file_b.to_string_lossy().to_string(),
            file_c.to_string_lossy().to_string(),
        ]];

        let summary = estimate_savings(&groups);
        assert_eq!(summary.groups.len(), 1);
        assert_eq!(summary.groups[0].total_bytes, 3500);
        assert_eq!(summary.groups[0].largest_file_bytes, 2000);
        assert_eq!(summary.groups[0].recoverable_bytes, 1500);
        assert_eq!(summary.groups[0].file_count, 3);
        assert_eq!(summary.total_recoverable_bytes, 1500);
        assert_eq!(summary.total_removable_files, 2);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_savings_summary_recovery_ratio() {
        let summary = SavingsSummary {
            groups: Vec::new(),
            total_recoverable_bytes: 500,
            total_bytes_in_duplicates: 1000,
            total_removable_files: 3,
        };
        assert!((summary.recovery_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_savings_summary_recovery_ratio_zero() {
        let summary = SavingsSummary {
            groups: Vec::new(),
            total_recoverable_bytes: 0,
            total_bytes_in_duplicates: 0,
            total_removable_files: 0,
        };
        assert_eq!(summary.recovery_ratio(), 0.0);
    }

    #[test]
    fn test_savings_summary_description() {
        let summary = SavingsSummary {
            groups: vec![GroupSavings {
                total_bytes: 3000,
                largest_file_bytes: 2000,
                recoverable_bytes: 1000,
                file_count: 3,
                description: "3 files, 1000 B recoverable".to_string(),
            }],
            total_recoverable_bytes: 1000,
            total_bytes_in_duplicates: 3000,
            total_removable_files: 2,
        };
        let desc = summary.description();
        assert!(desc.contains("1 duplicate groups"));
        assert!(desc.contains("2 removable files"));
    }

    #[test]
    fn test_estimate_savings_from_summary() {
        let summary = DedupSummary {
            total_files: 100,
            unique_files: 60,
            duplicate_files: 40,
            total_bytes: 10_000,
            reclaimable_bytes: 4_000,
        };
        let gs = estimate_savings_from_summary(&summary);
        assert_eq!(gs.recoverable_bytes, 4_000);
        assert_eq!(gs.total_bytes, 10_000);
        assert_eq!(gs.file_count, 100);
    }

    #[test]
    fn test_format_bytes_compact() {
        assert_eq!(format_bytes_compact(500), "500 B");
        assert_eq!(format_bytes_compact(1024), "1.0 KB");
        assert_eq!(format_bytes_compact(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes_compact(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_multiple_groups_savings() {
        let dir = std::env::temp_dir().join("oximedia_dedup_multi_savings");
        let _ = std::fs::create_dir_all(&dir);

        let a1 = dir.join("group1_a.bin");
        let a2 = dir.join("group1_b.bin");
        let b1 = dir.join("group2_a.bin");
        let b2 = dir.join("group2_b.bin");

        std::fs::write(&a1, &[0u8; 100]).expect("write");
        std::fs::write(&a2, &[0u8; 200]).expect("write");
        std::fs::write(&b1, &[0u8; 500]).expect("write");
        std::fs::write(&b2, &[0u8; 300]).expect("write");

        let groups = vec![
            vec![
                a1.to_string_lossy().to_string(),
                a2.to_string_lossy().to_string(),
            ],
            vec![
                b1.to_string_lossy().to_string(),
                b2.to_string_lossy().to_string(),
            ],
        ];

        let summary = estimate_savings(&groups);
        assert_eq!(summary.groups.len(), 2);
        // Group 1: keep 200, save 100
        assert_eq!(summary.groups[0].recoverable_bytes, 100);
        // Group 2: keep 500, save 300
        assert_eq!(summary.groups[1].recoverable_bytes, 300);
        assert_eq!(summary.total_recoverable_bytes, 400);
        assert_eq!(summary.total_removable_files, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- potential_savings_bytes / total_potential_savings_bytes tests ----

    #[test]
    fn test_report_savings_three_files() {
        let dir = std::env::temp_dir().join("oximedia_dedup_potential_three");
        let _ = std::fs::create_dir_all(&dir);

        let f1 = dir.join("a.bin");
        let f2 = dir.join("b.bin");
        let f3 = dir.join("c.bin");
        std::fs::write(&f1, &[0u8; 1000]).expect("write f1");
        std::fs::write(&f2, &[0u8; 2000]).expect("write f2");
        std::fs::write(&f3, &[0u8; 500]).expect("write f3");

        let groups = vec![vec![
            f1.to_string_lossy().to_string(),
            f2.to_string_lossy().to_string(),
            f3.to_string_lossy().to_string(),
        ]];
        let summary = estimate_savings(&groups);

        assert_eq!(summary.groups.len(), 1);
        // potential_savings_bytes == total - max == 3500 - 2000 == 1500
        assert_eq!(summary.groups[0].potential_savings_bytes(), 1500);
        // total_potential_savings_bytes == sum over all groups
        assert_eq!(summary.total_potential_savings_bytes(), 1500);
        // Must equal total_recoverable_bytes for consistency
        assert_eq!(
            summary.total_potential_savings_bytes(),
            summary.total_recoverable_bytes
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_report_savings_single_file_group_yields_zero() {
        // A single-file "group" is skipped by estimate_savings, so there are
        // no GroupSavings entries; total_potential_savings_bytes should be 0.
        let groups: Vec<Vec<String>> = vec![vec!["solo.bin".to_string()]];
        let summary = estimate_savings(&groups);
        assert_eq!(summary.total_potential_savings_bytes(), 0);
    }

    #[test]
    fn test_group_savings_potential_bytes_accessor() {
        // Direct unit test of GroupSavings::potential_savings_bytes().
        let gs = GroupSavings {
            total_bytes: 5000,
            largest_file_bytes: 3000,
            recoverable_bytes: 2000,
            file_count: 3,
            description: "test".to_string(),
        };
        assert_eq!(gs.potential_savings_bytes(), 2000);
    }

    #[test]
    fn test_total_potential_savings_multiple_groups() {
        let gs1 = GroupSavings {
            total_bytes: 1000,
            largest_file_bytes: 600,
            recoverable_bytes: 400,
            file_count: 2,
            description: "g1".to_string(),
        };
        let gs2 = GroupSavings {
            total_bytes: 3000,
            largest_file_bytes: 2000,
            recoverable_bytes: 1000,
            file_count: 3,
            description: "g2".to_string(),
        };
        let summary = SavingsSummary {
            groups: vec![gs1, gs2],
            total_recoverable_bytes: 1400,
            total_bytes_in_duplicates: 4000,
            total_removable_files: 3,
        };
        assert_eq!(summary.total_potential_savings_bytes(), 1400);
    }
}
