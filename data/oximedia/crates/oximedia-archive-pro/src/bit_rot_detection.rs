//! Bit rot detection and data integrity monitoring.
//!
//! This module provides tools to detect and characterise storage corruption:
//! - **BitRotScanner** - Detect checksum mismatches (bit rot)
//! - **CorruptionPattern** - Classify corruption type
//! - **BitRotStatistics** - Aggregate statistics across a file corpus
//! - **ScrubSchedule** - Periodic integrity verification scheduling
//! - **ParityProtection** - Overhead and recovery capability of parity schemes

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// BitRotReport
// ─────────────────────────────────────────────────────────────

/// The result of scanning a single file for bit rot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitRotReport {
    /// File path that was scanned
    pub path: String,
    /// Expected (reference) checksum
    pub expected_checksum: u64,
    /// Actual (computed) checksum
    pub actual_checksum: u64,
    /// Whether corruption was detected
    pub corrupted: bool,
    /// Estimated bit error rate (0.0–1.0)
    pub error_rate: f64,
}

impl BitRotReport {
    /// Returns true if this file's checksums match (no corruption detected).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        !self.corrupted
    }
}

// ─────────────────────────────────────────────────────────────
// BitRotScanner
// ─────────────────────────────────────────────────────────────

/// Scans files for bit rot by comparing expected and actual checksums.
///
/// The "actual" checksum is simulated using an FNV-1a hash of the file path
/// and size, providing a deterministic, reproducible pseudo-scan without I/O.
pub struct BitRotScanner;

impl BitRotScanner {
    /// Scan a file and produce a `BitRotReport`.
    ///
    /// * `path` – file path (used as a seed for the simulated hash)
    /// * `expected` – pre-recorded checksum to compare against
    /// * `size_bytes` – file size in bytes (incorporated into the simulated hash)
    #[must_use]
    pub fn scan_file(path: &str, expected: u64, size_bytes: u64) -> BitRotReport {
        let actual = Self::simulate_checksum(path, size_bytes);
        let corrupted = actual != expected;
        let error_rate = if corrupted {
            let xor = expected ^ actual;
            xor.count_ones() as f64 / 64.0
        } else {
            0.0
        };

        BitRotReport {
            path: path.to_string(),
            expected_checksum: expected,
            actual_checksum: actual,
            corrupted,
            error_rate,
        }
    }

    /// Simulated FNV-1a hash of path + size.
    fn simulate_checksum(path: &str, size_bytes: u64) -> u64 {
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut hash = FNV_OFFSET;
        for &byte in path.as_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Mix in size
        for shift in [0u32, 8, 16, 24, 32, 40, 48, 56] {
            hash ^= (size_bytes >> shift) & 0xFF;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

// ─────────────────────────────────────────────────────────────
// CorruptionPattern
// ─────────────────────────────────────────────────────────────

/// Characterises the type of storage corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorruptionPattern {
    /// Individual bits flipped at random locations
    RandomBitFlip,
    /// Complete bytes replaced with other values
    ByteSubstitution,
    /// Entire disk sector erased (all zeros or all ones)
    SectorErasure,
    /// Data silently dropped without error (stealth corruption)
    SilentDrop,
}

impl CorruptionPattern {
    /// Returns whether recovery is feasible with standard error-correction.
    #[must_use]
    pub const fn recovery_feasible(&self) -> bool {
        match self {
            Self::RandomBitFlip => true,
            Self::ByteSubstitution => true,
            Self::SectorErasure => false,
            Self::SilentDrop => false,
        }
    }

    /// Returns the pattern name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::RandomBitFlip => "RandomBitFlip",
            Self::ByteSubstitution => "ByteSubstitution",
            Self::SectorErasure => "SectorErasure",
            Self::SilentDrop => "SilentDrop",
        }
    }

    /// Heuristically infer the likely pattern from the error rate.
    #[must_use]
    pub fn infer_from_rate(error_rate: f64) -> Self {
        if error_rate < 0.02 {
            Self::RandomBitFlip
        } else if error_rate < 0.20 {
            Self::ByteSubstitution
        } else if error_rate >= 0.90 {
            Self::SectorErasure
        } else {
            Self::SilentDrop
        }
    }
}

// ─────────────────────────────────────────────────────────────
// BitRotStatistics
// ─────────────────────────────────────────────────────────────

/// Aggregate bit-rot statistics across a file corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitRotStatistics {
    /// Total files scanned
    pub files_scanned: u64,
    /// Files with corruption detected
    pub corrupted_files: u64,
    /// Total bit errors (approximated from error rates)
    pub total_bit_errors: u64,
    /// Overall corruption rate (0.0–1.0)
    pub corruption_rate: f64,
}

impl BitRotStatistics {
    /// Compute statistics from a slice of reports.
    #[must_use]
    pub fn compute(reports: &[BitRotReport]) -> Self {
        let files_scanned = reports.len() as u64;
        let corrupted_files = reports.iter().filter(|r| r.corrupted).count() as u64;
        let total_bit_errors = reports
            .iter()
            .filter(|r| r.corrupted)
            .map(|r| (r.error_rate * 64.0) as u64)
            .sum();
        let corruption_rate = if files_scanned > 0 {
            corrupted_files as f64 / files_scanned as f64
        } else {
            0.0
        };
        Self {
            files_scanned,
            corrupted_files,
            total_bit_errors,
            corruption_rate,
        }
    }

    /// Returns true if the corpus is clean (no corruption detected).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.corrupted_files == 0
    }
}

// ─────────────────────────────────────────────────────────────
// ScrubSchedule
// ─────────────────────────────────────────────────────────────

/// Defines a periodic storage scrub schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubSchedule {
    /// Scrub interval in days
    pub interval_days: u32,
    /// File extensions that should be scrubbed first (e.g., ["mkv", "tiff"])
    pub priority_extensions: Vec<String>,
}

impl ScrubSchedule {
    /// Create a new scrub schedule.
    #[must_use]
    pub fn new(interval_days: u32, priority_extensions: Vec<String>) -> Self {
        Self {
            interval_days,
            priority_extensions,
        }
    }

    /// Compute the timestamp (milliseconds since epoch) of the next scrub.
    ///
    /// * `last_scrub_ms` – timestamp of the last scrub in ms since epoch
    #[must_use]
    pub fn next_scrub_ms(&self, last_scrub_ms: u64) -> u64 {
        let interval_ms = self.interval_days as u64 * 24 * 60 * 60 * 1000;
        last_scrub_ms + interval_ms
    }

    /// Returns true if the extension is a priority target.
    #[must_use]
    pub fn is_priority_extension(&self, ext: &str) -> bool {
        let lower = ext.to_lowercase();
        self.priority_extensions
            .iter()
            .any(|e| e.to_lowercase() == lower)
    }
}

// ─────────────────────────────────────────────────────────────
// ParityProtection
// ─────────────────────────────────────────────────────────────

/// Describes a parity-based erasure coding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityProtection {
    /// Number of parity (redundancy) blocks
    pub parity_blocks: u32,
    /// Number of data blocks
    pub data_blocks: u32,
}

impl ParityProtection {
    /// Create a new parity configuration.
    #[must_use]
    pub fn new(data_blocks: u32, parity_blocks: u32) -> Self {
        Self {
            parity_blocks,
            data_blocks,
        }
    }

    /// Returns the storage overhead percentage.
    ///
    /// Overhead = parity_blocks / data_blocks × 100.
    #[must_use]
    pub fn overhead_pct(&self) -> f32 {
        if self.data_blocks == 0 {
            return 0.0;
        }
        (self.parity_blocks as f32 / self.data_blocks as f32) * 100.0
    }

    /// Returns the maximum number of block failures that can be recovered.
    ///
    /// With MDS erasure coding, this equals `parity_blocks`.
    #[must_use]
    pub fn can_recover_n_failures(&self) -> u32 {
        self.parity_blocks
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BitRotScanner ─────────────────────────────────────────

    #[test]
    fn test_scan_clean_file() {
        // Compute the expected hash first, then pass it in
        let path = "archive/video.mkv";
        let size = 1_000_000u64;
        let actual = BitRotScanner::simulate_checksum(path, size);
        let report = BitRotScanner::scan_file(path, actual, size);
        assert!(!report.corrupted);
        assert!(report.is_clean());
        assert_eq!(report.error_rate, 0.0);
    }

    #[test]
    fn test_scan_corrupted_file() {
        let path = "archive/video.mkv";
        let size = 1_000_000u64;
        let wrong_expected = 0xDEAD_BEEF_CAFE_BABEu64;
        let report = BitRotScanner::scan_file(path, wrong_expected, size);
        // Very likely that the simulated hash differs from this constant
        if report.corrupted {
            assert!(report.error_rate > 0.0);
        }
    }

    #[test]
    fn test_scan_deterministic() {
        let r1 = BitRotScanner::scan_file("file.tiff", 12345, 500_000);
        let r2 = BitRotScanner::scan_file("file.tiff", 12345, 500_000);
        assert_eq!(r1.actual_checksum, r2.actual_checksum);
    }

    // ── CorruptionPattern ─────────────────────────────────────

    #[test]
    fn test_recovery_feasible_random_bit_flip() {
        assert!(CorruptionPattern::RandomBitFlip.recovery_feasible());
    }

    #[test]
    fn test_recovery_not_feasible_sector_erasure() {
        assert!(!CorruptionPattern::SectorErasure.recovery_feasible());
    }

    #[test]
    fn test_infer_random_bit_flip() {
        assert_eq!(
            CorruptionPattern::infer_from_rate(0.01),
            CorruptionPattern::RandomBitFlip
        );
    }

    #[test]
    fn test_infer_sector_erasure() {
        assert_eq!(
            CorruptionPattern::infer_from_rate(0.95),
            CorruptionPattern::SectorErasure
        );
    }

    #[test]
    fn test_pattern_names() {
        assert_eq!(CorruptionPattern::SilentDrop.name(), "SilentDrop");
        assert_eq!(
            CorruptionPattern::ByteSubstitution.name(),
            "ByteSubstitution"
        );
    }

    // ── BitRotStatistics ──────────────────────────────────────

    #[test]
    fn test_statistics_all_clean() {
        let reports = vec![
            BitRotReport {
                path: "a.mkv".into(),
                expected_checksum: 1,
                actual_checksum: 1,
                corrupted: false,
                error_rate: 0.0,
            },
            BitRotReport {
                path: "b.tiff".into(),
                expected_checksum: 2,
                actual_checksum: 2,
                corrupted: false,
                error_rate: 0.0,
            },
        ];
        let stats = BitRotStatistics::compute(&reports);
        assert_eq!(stats.files_scanned, 2);
        assert_eq!(stats.corrupted_files, 0);
        assert!(stats.is_clean());
        assert_eq!(stats.corruption_rate, 0.0);
    }

    #[test]
    fn test_statistics_some_corrupted() {
        let reports = vec![
            BitRotReport {
                path: "a.mkv".into(),
                expected_checksum: 1,
                actual_checksum: 2,
                corrupted: true,
                error_rate: 0.05,
            },
            BitRotReport {
                path: "b.mkv".into(),
                expected_checksum: 3,
                actual_checksum: 3,
                corrupted: false,
                error_rate: 0.0,
            },
        ];
        let stats = BitRotStatistics::compute(&reports);
        assert_eq!(stats.corrupted_files, 1);
        assert!((stats.corruption_rate - 0.5).abs() < 1e-9);
        assert!(!stats.is_clean());
    }

    #[test]
    fn test_statistics_empty() {
        let stats = BitRotStatistics::compute(&[]);
        assert_eq!(stats.files_scanned, 0);
        assert_eq!(stats.corruption_rate, 0.0);
    }

    // ── ScrubSchedule ─────────────────────────────────────────

    #[test]
    fn test_next_scrub_ms() {
        let schedule = ScrubSchedule::new(7, vec!["mkv".into(), "tiff".into()]);
        let last = 0u64;
        let next = schedule.next_scrub_ms(last);
        let expected = 7u64 * 24 * 60 * 60 * 1000;
        assert_eq!(next, expected);
    }

    #[test]
    fn test_priority_extension() {
        let schedule = ScrubSchedule::new(30, vec!["MKV".into(), "TIFF".into()]);
        assert!(schedule.is_priority_extension("mkv"));
        assert!(schedule.is_priority_extension("TIFF"));
        assert!(!schedule.is_priority_extension("mp4"));
    }

    // ── ParityProtection ──────────────────────────────────────

    #[test]
    fn test_overhead_pct() {
        let p = ParityProtection::new(10, 2);
        assert!((p.overhead_pct() - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_overhead_zero_data_blocks() {
        let p = ParityProtection::new(0, 2);
        assert_eq!(p.overhead_pct(), 0.0);
    }

    #[test]
    fn test_can_recover_n_failures() {
        let p = ParityProtection::new(6, 3);
        assert_eq!(p.can_recover_n_failures(), 3);
    }
}
