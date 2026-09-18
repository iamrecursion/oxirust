//! Exact file and content matching for deduplication.
//!
//! Uses FNV-based hashing to build a 128-bit content identity for each piece
//! of data, then groups identical identities together to surface duplicates.

use std::collections::HashMap;

/// A 128-bit content identifier derived from raw bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentId {
    /// Raw 16-byte identifier
    pub bytes: [u8; 16],
}

impl ContentId {
    /// Return a lowercase hex string representation of this identifier.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.bytes
            .iter()
            .fold(String::with_capacity(32), |mut s, b| {
                s.push_str(&format!("{b:02x}"));
                s
            })
    }

    /// Derive a `ContentId` from raw data using FNV-1a over two 64-bit lanes.
    ///
    /// The two lanes use different starting offsets so that the pair forms a
    /// 128-bit value with better collision resistance than a single 64-bit hash.
    #[must_use]
    pub fn from_data(data: &[u8]) -> ContentId {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        const OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
        const OFFSET_B: u64 = 0xaf63_dc4c_8601_ec8c;

        let mut h0 = OFFSET_A;
        let mut h1 = OFFSET_B;

        for (i, &byte) in data.iter().enumerate() {
            if i % 2 == 0 {
                h0 ^= u64::from(byte);
                h0 = h0.wrapping_mul(FNV_PRIME);
            } else {
                h1 ^= u64::from(byte);
                h1 = h1.wrapping_mul(FNV_PRIME);
            }
        }

        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&h0.to_le_bytes());
        bytes[8..].copy_from_slice(&h1.to_le_bytes());
        ContentId { bytes }
    }
}

/// An index mapping content identifiers to lists of file paths.
#[derive(Debug, Default)]
pub struct ExactMatchIndex {
    /// Internal storage: hash → list of paths
    pub entries: HashMap<[u8; 16], Vec<String>>,
}

impl ExactMatchIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a file path under the given content identifier.
    pub fn insert(&mut self, id: ContentId, path: &str) {
        self.entries
            .entry(id.bytes)
            .or_default()
            .push(path.to_owned());
    }

    /// Return all file paths that share the same content identifier as `id`.
    ///
    /// Returns an empty slice when there is no entry for `id`.
    #[must_use]
    pub fn find_duplicates(&self, id: ContentId) -> &[String] {
        self.entries
            .get(&id.bytes)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Total number of individual path entries across all identifiers.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }

    /// Return all groups of paths that have more than one entry (actual duplicates).
    #[must_use]
    pub fn duplicate_sets(&self) -> Vec<Vec<String>> {
        self.entries
            .values()
            .filter(|v| v.len() > 1)
            .cloned()
            .collect()
    }
}

/// A size-based pre-filter to skip files outside a plausible range.
#[derive(Debug, Clone, Copy)]
pub struct FileSizeFilter {
    /// Minimum acceptable file size in bytes (inclusive)
    pub min_bytes: u64,
    /// Maximum acceptable file size in bytes (inclusive)
    pub max_bytes: u64,
}

impl FileSizeFilter {
    /// Create a new filter.
    #[must_use]
    pub fn new(min_bytes: u64, max_bytes: u64) -> Self {
        Self {
            min_bytes,
            max_bytes,
        }
    }

    /// Return `true` if `size` falls within the accepted range.
    #[must_use]
    pub fn accepts(&self, size: u64) -> bool {
        size >= self.min_bytes && size <= self.max_bytes
    }
}

/// Summary report produced after an exact-dedup scan.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactDedupReport {
    /// Number of files examined
    pub scanned: u64,
    /// Number of duplicate files found
    pub duplicates: u64,
    /// Total bytes consumed by duplicate copies
    pub wasted_bytes: u64,
}

impl ExactDedupReport {
    /// Create a new report.
    #[must_use]
    pub fn new(scanned: u64, duplicates: u64, wasted_bytes: u64) -> Self {
        Self {
            scanned,
            duplicates,
            wasted_bytes,
        }
    }

    /// Return the percentage of files that are duplicates (0.0–100.0).
    #[must_use]
    pub fn savings_pct(&self) -> f64 {
        if self.scanned == 0 {
            return 0.0;
        }
        (self.duplicates as f64 / self.scanned as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_id_from_data_deterministic() {
        let id1 = ContentId::from_data(b"hello world");
        let id2 = ContentId::from_data(b"hello world");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_content_id_different_data() {
        let id1 = ContentId::from_data(b"hello");
        let id2 = ContentId::from_data(b"world");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_content_id_empty_data() {
        let id = ContentId::from_data(b"");
        // Should not panic and produce a 16-byte value
        assert_eq!(id.bytes.len(), 16);
    }

    #[test]
    fn test_content_id_to_hex_length() {
        let id = ContentId::from_data(b"test");
        assert_eq!(id.to_hex().len(), 32);
    }

    #[test]
    fn test_content_id_to_hex_valid_chars() {
        let id = ContentId::from_data(b"abc");
        assert!(id.to_hex().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_exact_match_index_insert_and_find() {
        let mut index = ExactMatchIndex::new();
        let id = ContentId::from_data(b"duplicate content");
        index.insert(id, "/media/file_a.mp4");
        index.insert(id, "/media/file_b.mp4");
        let dups = index.find_duplicates(id);
        assert_eq!(dups.len(), 2);
        assert!(dups.contains(&"/media/file_a.mp4".to_string()));
        assert!(dups.contains(&"/media/file_b.mp4".to_string()));
    }

    #[test]
    fn test_exact_match_index_find_missing() {
        let index = ExactMatchIndex::new();
        let id = ContentId::from_data(b"not present");
        assert!(index.find_duplicates(id).is_empty());
    }

    #[test]
    fn test_exact_match_index_total_entries() {
        let mut index = ExactMatchIndex::new();
        let id_a = ContentId::from_data(b"aaa");
        let id_b = ContentId::from_data(b"bbb");
        index.insert(id_a, "/a/1.mp4");
        index.insert(id_a, "/a/2.mp4");
        index.insert(id_b, "/b/1.mp4");
        assert_eq!(index.total_entries(), 3);
    }

    #[test]
    fn test_exact_match_duplicate_sets() {
        let mut index = ExactMatchIndex::new();
        let id_a = ContentId::from_data(b"same_data");
        let id_b = ContentId::from_data(b"unique_data");
        index.insert(id_a, "/dup1.mp4");
        index.insert(id_a, "/dup2.mp4");
        index.insert(id_b, "/unique.mp4");
        let sets = index.duplicate_sets();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].len(), 2);
    }

    #[test]
    fn test_file_size_filter_accepts_within_range() {
        let filter = FileSizeFilter::new(1024, 1024 * 1024);
        assert!(filter.accepts(512 * 1024));
    }

    #[test]
    fn test_file_size_filter_rejects_below_min() {
        let filter = FileSizeFilter::new(1024, 1024 * 1024);
        assert!(!filter.accepts(100));
    }

    #[test]
    fn test_file_size_filter_rejects_above_max() {
        let filter = FileSizeFilter::new(1024, 1024 * 1024);
        assert!(!filter.accepts(2 * 1024 * 1024));
    }

    #[test]
    fn test_file_size_filter_boundary_values() {
        let filter = FileSizeFilter::new(100, 200);
        assert!(filter.accepts(100));
        assert!(filter.accepts(200));
        assert!(!filter.accepts(99));
        assert!(!filter.accepts(201));
    }

    #[test]
    fn test_exact_dedup_report_savings_pct_zero_scanned() {
        let report = ExactDedupReport::new(0, 0, 0);
        assert_eq!(report.savings_pct(), 0.0);
    }

    #[test]
    fn test_exact_dedup_report_savings_pct() {
        let report = ExactDedupReport::new(100, 25, 1_000_000);
        assert!((report.savings_pct() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_exact_dedup_report_full_duplicates() {
        let report = ExactDedupReport::new(50, 50, 5_000_000);
        assert!((report.savings_pct() - 100.0).abs() < 1e-9);
    }
}
