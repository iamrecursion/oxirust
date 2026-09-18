#![allow(dead_code)]
//! Archive-level deduplication
//!
//! Provides content-addressable deduplication for archived media assets.
//! Uses rolling hashes and fingerprints to detect duplicate content at the
//! file level and chunk level, reducing storage costs for large archives.

use std::collections::HashMap;
use std::fmt;

/// A content fingerprint used for dedup comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentFingerprint {
    /// Hex-encoded digest string.
    digest: String,
    /// Size in bytes of the content that was fingerprinted.
    size_bytes: u64,
}

impl ContentFingerprint {
    /// Create a new content fingerprint.
    pub fn new(digest: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            digest: digest.into(),
            size_bytes,
        }
    }

    /// Return the digest string.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Return the size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

impl fmt::Display for ContentFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.digest, self.size_bytes)
    }
}

/// Deduplication level granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupLevel {
    /// Whole-file deduplication.
    File,
    /// Fixed-size chunk deduplication.
    FixedChunk,
    /// Variable-size (content-defined) chunk deduplication.
    VariableChunk,
}

impl fmt::Display for DedupLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::File => "file",
            Self::FixedChunk => "fixed-chunk",
            Self::VariableChunk => "variable-chunk",
        };
        write!(f, "{s}")
    }
}

/// Configuration for dedup operations.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Dedup granularity level.
    pub level: DedupLevel,
    /// Fixed chunk size in bytes (used when level is FixedChunk).
    pub chunk_size: usize,
    /// Minimum chunk size for variable chunking.
    pub min_chunk: usize,
    /// Maximum chunk size for variable chunking.
    pub max_chunk: usize,
    /// Whether to keep a reference count.
    pub track_refcount: bool,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            level: DedupLevel::File,
            chunk_size: 4 * 1024 * 1024, // 4 MiB
            min_chunk: 512 * 1024,       // 512 KiB
            max_chunk: 16 * 1024 * 1024, // 16 MiB
            track_refcount: true,
        }
    }
}

/// A record for a stored chunk or file entry.
#[derive(Debug, Clone)]
pub struct DedupEntry {
    /// The content fingerprint.
    pub fingerprint: ContentFingerprint,
    /// Path where the canonical copy is stored.
    pub canonical_path: String,
    /// Number of references to this entry.
    pub refcount: u32,
}

impl DedupEntry {
    /// Create a new dedup entry.
    pub fn new(fingerprint: ContentFingerprint, canonical_path: impl Into<String>) -> Self {
        Self {
            fingerprint,
            canonical_path: canonical_path.into(),
            refcount: 1,
        }
    }

    /// Increment the reference count.
    pub fn add_ref(&mut self) {
        self.refcount = self.refcount.saturating_add(1);
    }

    /// Decrement the reference count, returning the new count.
    pub fn release_ref(&mut self) -> u32 {
        self.refcount = self.refcount.saturating_sub(1);
        self.refcount
    }
}

/// Result of a dedup lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DedupResult {
    /// Content is new and was stored.
    Stored,
    /// Content was a duplicate; reference added.
    Duplicate,
    /// Content was skipped (error or policy).
    Skipped,
}

/// Simple rolling hash for content-defined chunking.
#[allow(clippy::cast_precision_loss)]
pub fn rolling_hash(data: &[u8], window: usize) -> Vec<u64> {
    if data.len() < window || window == 0 {
        return vec![];
    }
    let mut hashes = Vec::with_capacity(data.len() - window + 1);
    let mut h: u64 = 0;
    // Initial window
    for &b in &data[..window] {
        h = h.wrapping_mul(31).wrapping_add(u64::from(b));
    }
    hashes.push(h);
    // Rolling
    let base_pow = 31_u64.wrapping_pow(window as u32);
    for i in window..data.len() {
        h = h
            .wrapping_mul(31)
            .wrapping_add(u64::from(data[i]))
            .wrapping_sub(base_pow.wrapping_mul(u64::from(data[i - window])));
        hashes.push(h);
    }
    hashes
}

/// Compute a simple 64-bit fingerprint for a byte slice.
pub fn compute_fingerprint(data: &[u8]) -> ContentFingerprint {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    ContentFingerprint::new(format!("{h:016x}"), data.len() as u64)
}

/// In-memory dedup index for archive content.
#[derive(Debug)]
pub struct DedupIndex {
    /// Configuration.
    config: DedupConfig,
    /// Map from fingerprint digest to entry.
    entries: HashMap<String, DedupEntry>,
    /// Total bytes saved by dedup.
    bytes_saved: u64,
}

impl DedupIndex {
    /// Create a new dedup index with the given config.
    pub fn new(config: DedupConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            bytes_saved: 0,
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(DedupConfig::default())
    }

    /// Get the current config.
    pub fn config(&self) -> &DedupConfig {
        &self.config
    }

    /// Return number of unique entries.
    pub fn unique_count(&self) -> usize {
        self.entries.len()
    }

    /// Return total bytes saved.
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_saved
    }

    /// Ingest content: returns whether it was stored or a duplicate.
    pub fn ingest(&mut self, data: &[u8], path: &str) -> DedupResult {
        let fp = compute_fingerprint(data);
        let digest = fp.digest().to_string();
        if let Some(entry) = self.entries.get_mut(&digest) {
            entry.add_ref();
            self.bytes_saved += fp.size_bytes();
            DedupResult::Duplicate
        } else {
            self.entries.insert(digest, DedupEntry::new(fp, path));
            DedupResult::Stored
        }
    }

    /// Look up whether content already exists.
    pub fn contains(&self, data: &[u8]) -> bool {
        let fp = compute_fingerprint(data);
        self.entries.contains_key(fp.digest())
    }

    /// Release a reference; returns true if entry was removed.
    pub fn release(&mut self, data: &[u8]) -> bool {
        let fp = compute_fingerprint(data);
        let digest = fp.digest().to_string();
        if let Some(entry) = self.entries.get_mut(&digest) {
            if entry.release_ref() == 0 {
                self.entries.remove(&digest);
                return true;
            }
        }
        false
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> DedupStats {
        let total_refs: u64 = self.entries.values().map(|e| u64::from(e.refcount)).sum();
        DedupStats {
            unique_entries: self.entries.len(),
            total_references: total_refs,
            bytes_saved: self.bytes_saved,
        }
    }
}

/// Statistics about dedup index.
#[derive(Debug, Clone)]
pub struct DedupStats {
    /// Number of unique entries.
    pub unique_entries: usize,
    /// Total number of references.
    pub total_references: u64,
    /// Bytes saved through dedup.
    pub bytes_saved: u64,
}

// ---------------------------------------------------------------------------
// Deduplication reporting
// ---------------------------------------------------------------------------

/// Information about a single duplicate group (files sharing the same content).
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    /// The content fingerprint digest shared by all files in this group.
    pub fingerprint_digest: String,
    /// Size of each file in bytes.
    pub size_bytes: u64,
    /// The canonical (first-stored) path.
    pub canonical_path: String,
    /// All reference paths including the canonical one.
    pub all_paths: Vec<String>,
    /// Number of duplicate copies (total references - 1).
    pub duplicate_count: u32,
    /// Bytes wasted by duplicates (size * duplicate_count).
    pub wasted_bytes: u64,
}

/// Comprehensive deduplication report with space savings analysis.
#[derive(Debug, Clone)]
pub struct DedupReport {
    /// Total number of unique content items.
    pub unique_count: usize,
    /// Total number of references (unique + duplicates).
    pub total_references: u64,
    /// Total number of duplicate references (references - unique).
    pub duplicate_count: u64,
    /// Total bytes saved by deduplication.
    pub total_bytes_saved: u64,
    /// Total logical size (sum of all referenced content sizes).
    pub total_logical_size: u64,
    /// Total physical size (sum of unique content sizes only).
    pub total_physical_size: u64,
    /// Deduplication ratio (logical / physical). Higher = more savings.
    pub dedup_ratio: f64,
    /// Percentage of space saved (0.0 to 100.0).
    pub savings_percentage: f64,
    /// Groups of duplicate files, sorted by wasted bytes descending.
    pub duplicate_groups: Vec<DuplicateGroup>,
    /// Top N duplicate groups by wasted bytes.
    pub top_wasters: Vec<DuplicateGroup>,
    /// Distribution of reference counts (refcount -> number of entries with that refcount).
    pub refcount_distribution: Vec<(u32, usize)>,
}

impl DedupReport {
    /// Format the report as a human-readable string.
    #[must_use]
    pub fn to_summary_string(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Deduplication Report ===\n");
        out.push_str(&format!("Unique items:       {}\n", self.unique_count));
        out.push_str(&format!("Total references:   {}\n", self.total_references));
        out.push_str(&format!("Duplicate refs:     {}\n", self.duplicate_count));
        out.push_str(&format!(
            "Logical size:       {} bytes\n",
            self.total_logical_size
        ));
        out.push_str(&format!(
            "Physical size:      {} bytes\n",
            self.total_physical_size
        ));
        out.push_str(&format!(
            "Space saved:        {} bytes ({:.1}%)\n",
            self.total_bytes_saved, self.savings_percentage
        ));
        out.push_str(&format!("Dedup ratio:        {:.2}x\n", self.dedup_ratio));

        if !self.top_wasters.is_empty() {
            out.push_str("\nTop duplicate groups by wasted space:\n");
            for (i, group) in self.top_wasters.iter().enumerate() {
                out.push_str(&format!(
                    "  {}. {} — {} bytes x {} copies = {} bytes wasted\n",
                    i + 1,
                    group.canonical_path,
                    group.size_bytes,
                    group.duplicate_count,
                    group.wasted_bytes,
                ));
            }
        }

        if !self.refcount_distribution.is_empty() {
            out.push_str("\nReference count distribution:\n");
            for (refcount, count) in &self.refcount_distribution {
                out.push_str(&format!("  {refcount} refs: {count} entries\n"));
            }
        }

        out
    }
}

/// Enhanced dedup index with reporting capabilities.
impl DedupIndex {
    /// Generate a comprehensive deduplication report.
    #[must_use]
    pub fn generate_report(&self, top_n: usize) -> DedupReport {
        let mut duplicate_groups = Vec::new();
        let mut total_logical_size: u64 = 0;
        let mut total_physical_size: u64 = 0;
        let mut refcount_map: HashMap<u32, usize> = HashMap::new();

        for entry in self.entries.values() {
            let size = entry.fingerprint.size_bytes();
            let refs = entry.refcount;
            total_physical_size += size;
            total_logical_size += size * u64::from(refs);

            *refcount_map.entry(refs).or_insert(0) += 1;

            if refs > 1 {
                let duplicate_count = refs - 1;
                let wasted = size * u64::from(duplicate_count);

                duplicate_groups.push(DuplicateGroup {
                    fingerprint_digest: entry.fingerprint.digest().to_string(),
                    size_bytes: size,
                    canonical_path: entry.canonical_path.clone(),
                    all_paths: vec![entry.canonical_path.clone()],
                    duplicate_count,
                    wasted_bytes: wasted,
                });
            }
        }

        // Sort by wasted bytes descending
        duplicate_groups.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));

        let top_wasters: Vec<DuplicateGroup> =
            duplicate_groups.iter().take(top_n).cloned().collect();

        let total_refs: u64 = self.entries.values().map(|e| u64::from(e.refcount)).sum();
        let duplicate_count = total_refs.saturating_sub(self.entries.len() as u64);

        let dedup_ratio = if total_physical_size == 0 {
            1.0
        } else {
            total_logical_size as f64 / total_physical_size as f64
        };

        let savings_percentage = if total_logical_size == 0 {
            0.0
        } else {
            (self.bytes_saved as f64 / total_logical_size as f64) * 100.0
        };

        let mut refcount_distribution: Vec<(u32, usize)> = refcount_map.into_iter().collect();
        refcount_distribution.sort_by_key(|(rc, _)| *rc);

        DedupReport {
            unique_count: self.entries.len(),
            total_references: total_refs,
            duplicate_count,
            total_bytes_saved: self.bytes_saved,
            total_logical_size,
            total_physical_size,
            dedup_ratio,
            savings_percentage,
            duplicate_groups,
            top_wasters,
            refcount_distribution,
        }
    }

    /// Get a list of all duplicate file paths grouped by fingerprint.
    #[must_use]
    pub fn list_duplicates(&self) -> Vec<(&str, u32, u64)> {
        self.entries
            .values()
            .filter(|e| e.refcount > 1)
            .map(|e| {
                (
                    e.canonical_path.as_str(),
                    e.refcount,
                    e.fingerprint.size_bytes(),
                )
            })
            .collect()
    }

    /// Estimate total storage needed if dedup is applied.
    #[must_use]
    pub fn estimated_physical_storage(&self) -> u64 {
        self.entries
            .values()
            .map(|e| e.fingerprint.size_bytes())
            .sum()
    }

    /// Estimate total logical storage (without dedup).
    #[must_use]
    pub fn estimated_logical_storage(&self) -> u64 {
        self.entries
            .values()
            .map(|e| e.fingerprint.size_bytes() * u64::from(e.refcount))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let fp = ContentFingerprint::new("abc123", 1024);
        assert_eq!(fp.digest(), "abc123");
        assert_eq!(fp.size_bytes(), 1024);
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = ContentFingerprint::new("abc123", 1024);
        assert_eq!(fp.to_string(), "abc123:1024");
    }

    #[test]
    fn test_dedup_level_display() {
        assert_eq!(DedupLevel::File.to_string(), "file");
        assert_eq!(DedupLevel::FixedChunk.to_string(), "fixed-chunk");
        assert_eq!(DedupLevel::VariableChunk.to_string(), "variable-chunk");
    }

    #[test]
    fn test_default_config() {
        let cfg = DedupConfig::default();
        assert_eq!(cfg.level, DedupLevel::File);
        assert_eq!(cfg.chunk_size, 4 * 1024 * 1024);
        assert!(cfg.track_refcount);
    }

    #[test]
    fn test_dedup_entry_refcount() {
        let fp = ContentFingerprint::new("abc", 100);
        let mut entry = DedupEntry::new(fp, "/store/abc");
        assert_eq!(entry.refcount, 1);
        entry.add_ref();
        assert_eq!(entry.refcount, 2);
        entry.release_ref();
        assert_eq!(entry.refcount, 1);
        entry.release_ref();
        assert_eq!(entry.refcount, 0);
        // saturating: doesn't go below 0
        entry.release_ref();
        assert_eq!(entry.refcount, 0);
    }

    #[test]
    fn test_compute_fingerprint_deterministic() {
        let data = b"hello world archive";
        let fp1 = compute_fingerprint(data);
        let fp2 = compute_fingerprint(data);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_compute_fingerprint_different_data() {
        let fp1 = compute_fingerprint(b"data_a");
        let fp2 = compute_fingerprint(b"data_b");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_rolling_hash_basic() {
        let data = b"abcdefghij";
        let hashes = rolling_hash(data, 4);
        assert_eq!(hashes.len(), 7); // 10 - 4 + 1
    }

    #[test]
    fn test_rolling_hash_empty() {
        let hashes = rolling_hash(b"ab", 5);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_index_ingest_new() {
        let mut idx = DedupIndex::with_defaults();
        let result = idx.ingest(b"content_one", "/archive/a.mxf");
        assert_eq!(result, DedupResult::Stored);
        assert_eq!(idx.unique_count(), 1);
    }

    #[test]
    fn test_index_ingest_duplicate() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content_one", "/archive/a.mxf");
        let result = idx.ingest(b"content_one", "/archive/b.mxf");
        assert_eq!(result, DedupResult::Duplicate);
        assert_eq!(idx.unique_count(), 1);
        assert!(idx.bytes_saved() > 0);
    }

    #[test]
    fn test_index_contains() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content_one", "/archive/a.mxf");
        assert!(idx.contains(b"content_one"));
        assert!(!idx.contains(b"content_two"));
    }

    #[test]
    fn test_index_release() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content_one", "/archive/a.mxf");
        let removed = idx.release(b"content_one");
        assert!(removed);
        assert_eq!(idx.unique_count(), 0);
    }

    #[test]
    fn test_index_stats() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content_one", "/archive/a.mxf");
        idx.ingest(b"content_one", "/archive/b.mxf");
        idx.ingest(b"content_two", "/archive/c.mxf");
        let stats = idx.stats();
        assert_eq!(stats.unique_entries, 2);
        assert_eq!(stats.total_references, 3); // 2 refs for one, 1 for other
    }

    // --- Dedup reporting tests ---

    #[test]
    fn test_report_empty_index() {
        let idx = DedupIndex::with_defaults();
        let report = idx.generate_report(5);
        assert_eq!(report.unique_count, 0);
        assert_eq!(report.total_references, 0);
        assert_eq!(report.duplicate_count, 0);
        assert!((report.dedup_ratio - 1.0).abs() < f64::EPSILON);
        assert!(report.duplicate_groups.is_empty());
    }

    #[test]
    fn test_report_no_duplicates() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"unique_a", "/a.mxf");
        idx.ingest(b"unique_b", "/b.mxf");
        idx.ingest(b"unique_c", "/c.mxf");

        let report = idx.generate_report(5);
        assert_eq!(report.unique_count, 3);
        assert_eq!(report.duplicate_count, 0);
        assert!(report.duplicate_groups.is_empty());
        assert_eq!(report.total_bytes_saved, 0);
    }

    #[test]
    fn test_report_with_duplicates() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content_dup", "/archive/a.mxf");
        idx.ingest(b"content_dup", "/archive/b.mxf");
        idx.ingest(b"content_dup", "/archive/c.mxf");
        idx.ingest(b"content_unique", "/archive/d.mxf");

        let report = idx.generate_report(5);
        assert_eq!(report.unique_count, 2);
        assert_eq!(report.total_references, 4);
        assert_eq!(report.duplicate_count, 2);
        assert!(report.total_bytes_saved > 0);
        assert_eq!(report.duplicate_groups.len(), 1);
        assert_eq!(report.duplicate_groups[0].duplicate_count, 2);
    }

    #[test]
    fn test_report_top_wasters_limit() {
        let mut idx = DedupIndex::with_defaults();
        // Create several duplicate groups
        idx.ingest(b"small_dup", "/s1.mxf");
        idx.ingest(b"small_dup", "/s2.mxf");
        idx.ingest(b"medium_dup_data!", "/m1.mxf");
        idx.ingest(b"medium_dup_data!", "/m2.mxf");
        idx.ingest(b"large_duplicate_content_here", "/l1.mxf");
        idx.ingest(b"large_duplicate_content_here", "/l2.mxf");
        idx.ingest(b"large_duplicate_content_here", "/l3.mxf");

        let report = idx.generate_report(2);
        assert_eq!(report.top_wasters.len(), 2);
        // Top waster should have more wasted bytes than second
        assert!(report.top_wasters[0].wasted_bytes >= report.top_wasters[1].wasted_bytes);
    }

    #[test]
    fn test_report_dedup_ratio() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"aaaaaaaaaa", "/a.bin"); // 10 bytes
        idx.ingest(b"aaaaaaaaaa", "/b.bin"); // duplicate

        let report = idx.generate_report(5);
        // Logical = 10 * 2 = 20, Physical = 10, ratio = 2.0
        assert!((report.dedup_ratio - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_savings_percentage() {
        let mut idx = DedupIndex::with_defaults();
        let content = b"test_content";
        idx.ingest(content, "/a.bin");
        idx.ingest(content, "/b.bin");

        let report = idx.generate_report(5);
        assert!(report.savings_percentage > 0.0);
        assert!(report.savings_percentage <= 100.0);
    }

    #[test]
    fn test_report_refcount_distribution() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"once", "/a.bin");
        idx.ingest(b"twice", "/b.bin");
        idx.ingest(b"twice", "/c.bin");
        idx.ingest(b"thrice", "/d.bin");
        idx.ingest(b"thrice", "/e.bin");
        idx.ingest(b"thrice", "/f.bin");

        let report = idx.generate_report(5);
        // Should have entries for refcount 1, 2, and 3
        assert!(!report.refcount_distribution.is_empty());
        let rc_map: HashMap<u32, usize> = report.refcount_distribution.into_iter().collect();
        assert_eq!(rc_map.get(&1), Some(&1)); // 1 item with refcount 1
        assert_eq!(rc_map.get(&2), Some(&1)); // 1 item with refcount 2
        assert_eq!(rc_map.get(&3), Some(&1)); // 1 item with refcount 3
    }

    #[test]
    fn test_report_summary_string() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"content", "/a.mxf");
        idx.ingest(b"content", "/b.mxf");

        let report = idx.generate_report(5);
        let summary = report.to_summary_string();
        assert!(summary.contains("Deduplication Report"));
        assert!(summary.contains("Unique items"));
        assert!(summary.contains("Space saved"));
        assert!(summary.contains("Dedup ratio"));
    }

    #[test]
    fn test_list_duplicates() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"unique", "/a.bin");
        idx.ingest(b"dup_content", "/b.bin");
        idx.ingest(b"dup_content", "/c.bin");

        let dups = idx.list_duplicates();
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].1, 2); // refcount
    }

    #[test]
    fn test_estimated_storage() {
        let mut idx = DedupIndex::with_defaults();
        let content_a = b"aaaa"; // 4 bytes
        let content_b = b"bbbb"; // 4 bytes
        idx.ingest(content_a, "/a.bin");
        idx.ingest(content_a, "/a2.bin");
        idx.ingest(content_b, "/b.bin");

        // Physical = 4 + 4 = 8
        assert_eq!(idx.estimated_physical_storage(), 8);
        // Logical = 4*2 + 4*1 = 12
        assert_eq!(idx.estimated_logical_storage(), 12);
    }

    #[test]
    fn test_report_sorted_by_wasted() {
        let mut idx = DedupIndex::with_defaults();
        idx.ingest(b"aa", "/s.bin");
        idx.ingest(b"aa", "/s2.bin");
        idx.ingest(b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "/l.bin");
        idx.ingest(b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "/l2.bin");

        let report = idx.generate_report(10);
        if report.duplicate_groups.len() >= 2 {
            assert!(
                report.duplicate_groups[0].wasted_bytes >= report.duplicate_groups[1].wasted_bytes
            );
        }
    }
}
