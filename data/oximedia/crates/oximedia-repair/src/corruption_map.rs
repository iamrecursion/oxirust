//! Media corruption mapping.
//!
//! Provides types for recording and analysing the location, type, and severity
//! of corrupted regions within a media file.

/// The category of corruption present in a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionType {
    /// The file header is damaged or missing.
    HeaderDamage,
    /// Payload data is corrupted.
    DataCorruption,
    /// A frame is incomplete (partial data).
    IncompleteFrame,
    /// Synchronisation markers are missing or wrong.
    SyncLoss,
    /// The index / seek-table is corrupted.
    IndexCorruption,
    /// The file was truncated prematurely.
    Truncation,
}

/// A contiguous byte range within a file that is corrupted.
#[derive(Debug, Clone)]
pub struct CorruptedRegion {
    /// Byte offset where corruption begins (inclusive).
    pub start_byte: u64,
    /// Byte offset where corruption ends (exclusive).
    pub end_byte: u64,
    /// Classification of the corruption.
    pub corruption_type: CorruptionType,
    /// Severity of the corruption (0.0 = minor, 1.0 = catastrophic).
    pub severity: f64,
    /// Whether automated repair is possible for this region.
    pub repairable: bool,
}

impl CorruptedRegion {
    /// Return the number of corrupted bytes in this region.
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.end_byte.saturating_sub(self.start_byte)
    }

    /// Return `true` when this region is considered critical for playback.
    ///
    /// Header damage, sync loss at high severity, and truncation are all
    /// considered critical.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        match self.corruption_type {
            CorruptionType::HeaderDamage => true,
            CorruptionType::SyncLoss => self.severity >= 0.7,
            CorruptionType::Truncation => true,
            _ => self.severity >= 0.9,
        }
    }
}

/// A complete map of all corrupted regions within a single media file.
#[derive(Debug)]
pub struct CorruptionMap {
    /// Total size of the file in bytes.
    pub file_size: u64,
    /// All detected corrupted regions, in discovery order.
    regions: Vec<CorruptedRegion>,
}

impl CorruptionMap {
    /// Create a new, empty `CorruptionMap` for a file of `file_size` bytes.
    #[must_use]
    pub fn new(file_size: u64) -> Self {
        Self {
            file_size,
            regions: Vec::new(),
        }
    }

    /// Add a corrupted region to the map.
    pub fn add_region(&mut self, r: CorruptedRegion) {
        self.regions.push(r);
    }

    /// Return the total number of corrupted bytes (regions may overlap).
    #[must_use]
    pub fn total_corrupted_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.size_bytes()).sum()
    }

    /// Return the percentage of the file that is corrupted (0.0–100.0).
    ///
    /// Returns 0.0 when `file_size` is zero.
    #[must_use]
    pub fn corruption_pct(&self) -> f64 {
        if self.file_size == 0 {
            return 0.0;
        }
        (self.total_corrupted_bytes() as f64 / self.file_size as f64 * 100.0).min(100.0)
    }

    /// Return `true` when all regions are individually repairable.
    ///
    /// An empty map (no corruption) is considered repairable.
    #[must_use]
    pub fn is_repairable(&self) -> bool {
        self.regions.iter().all(|r| r.repairable)
    }

    /// Return references to all regions that are considered critical.
    #[must_use]
    pub fn critical_regions(&self) -> Vec<&CorruptedRegion> {
        self.regions.iter().filter(|r| r.is_critical()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(
        start: u64,
        end: u64,
        ctype: CorruptionType,
        severity: f64,
        repairable: bool,
    ) -> CorruptedRegion {
        CorruptedRegion {
            start_byte: start,
            end_byte: end,
            corruption_type: ctype,
            severity,
            repairable,
        }
    }

    #[test]
    fn test_region_size_bytes() {
        let r = make_region(100, 200, CorruptionType::DataCorruption, 0.5, true);
        assert_eq!(r.size_bytes(), 100);
    }

    #[test]
    fn test_region_size_bytes_zero_when_start_equals_end() {
        let r = make_region(50, 50, CorruptionType::DataCorruption, 0.5, true);
        assert_eq!(r.size_bytes(), 0);
    }

    #[test]
    fn test_region_is_critical_header_damage() {
        let r = make_region(0, 4, CorruptionType::HeaderDamage, 0.1, false);
        assert!(r.is_critical());
    }

    #[test]
    fn test_region_is_critical_truncation() {
        let r = make_region(1000, 1010, CorruptionType::Truncation, 0.3, false);
        assert!(r.is_critical());
    }

    #[test]
    fn test_region_is_critical_sync_loss_high_severity() {
        let r = make_region(500, 520, CorruptionType::SyncLoss, 0.8, true);
        assert!(r.is_critical());
    }

    #[test]
    fn test_region_is_not_critical_sync_loss_low_severity() {
        let r = make_region(500, 520, CorruptionType::SyncLoss, 0.3, true);
        assert!(!r.is_critical());
    }

    #[test]
    fn test_corruption_map_new() {
        let map = CorruptionMap::new(1_000_000);
        assert_eq!(map.file_size, 1_000_000);
        assert_eq!(map.total_corrupted_bytes(), 0);
    }

    #[test]
    fn test_corruption_pct_zero_on_empty() {
        let map = CorruptionMap::new(1_000_000);
        assert!((map.corruption_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_corruption_pct_calculation() {
        let mut map = CorruptionMap::new(1_000);
        map.add_region(make_region(
            0,
            100,
            CorruptionType::DataCorruption,
            0.5,
            true,
        ));
        assert!((map.corruption_pct() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_corruption_pct_zero_file_size() {
        let map = CorruptionMap::new(0);
        assert!((map.corruption_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_repairable_all_repairable() {
        let mut map = CorruptionMap::new(1_000);
        map.add_region(make_region(
            0,
            50,
            CorruptionType::DataCorruption,
            0.4,
            true,
        ));
        map.add_region(make_region(
            200,
            250,
            CorruptionType::IncompleteFrame,
            0.3,
            true,
        ));
        assert!(map.is_repairable());
    }

    #[test]
    fn test_is_repairable_one_unrepairable() {
        let mut map = CorruptionMap::new(1_000);
        map.add_region(make_region(
            0,
            50,
            CorruptionType::DataCorruption,
            0.4,
            true,
        ));
        map.add_region(make_region(
            200,
            250,
            CorruptionType::HeaderDamage,
            0.9,
            false,
        ));
        assert!(!map.is_repairable());
    }

    #[test]
    fn test_critical_regions_filter() {
        let mut map = CorruptionMap::new(10_000);
        map.add_region(make_region(0, 4, CorruptionType::HeaderDamage, 0.2, false));
        map.add_region(make_region(
            100,
            200,
            CorruptionType::DataCorruption,
            0.3,
            true,
        ));
        map.add_region(make_region(
            500,
            600,
            CorruptionType::Truncation,
            0.8,
            false,
        ));
        let critical = map.critical_regions();
        assert_eq!(critical.len(), 2); // Header + Truncation
    }

    #[test]
    fn test_total_corrupted_bytes_multiple_regions() {
        let mut map = CorruptionMap::new(10_000);
        map.add_region(make_region(
            0,
            100,
            CorruptionType::DataCorruption,
            0.5,
            true,
        ));
        map.add_region(make_region(
            200,
            350,
            CorruptionType::IncompleteFrame,
            0.4,
            true,
        ));
        assert_eq!(map.total_corrupted_bytes(), 250);
    }
}
