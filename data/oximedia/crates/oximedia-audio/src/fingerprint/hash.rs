//! Hash generation from peak pairs.

use super::constellation::{ConstellationMap, Peak};
use std::collections::HashSet;

/// Hash value for a peak pair.
///
/// The hash encodes the relationship between two peaks (anchor and target):
/// - Anchor frequency
/// - Target frequency
/// - Time difference between peaks
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Hash {
    value: u64,
}

impl Hash {
    /// Create a hash from peak pair.
    #[must_use]
    pub fn from_peaks(anchor: &Peak, target: &Peak) -> Self {
        // Quantize frequencies (use bins)
        let f1 = anchor.quantized_frequency();
        let f2 = target.quantized_frequency();

        // Quantize time difference (in milliseconds)
        let time_delta = target
            .quantized_time()
            .saturating_sub(anchor.quantized_time());

        // Combine into hash value
        // Format: [f1: 20 bits][f2: 20 bits][delta_t: 24 bits]
        let hash_value = ((u64::from(f1) & 0x0F_FFFF) << 44)
            | ((u64::from(f2) & 0x0F_FFFF) << 24)
            | (u64::from(time_delta) & 0x00FF_FFFF);

        Self { value: hash_value }
    }

    /// Create a hash from raw components.
    #[must_use]
    pub fn from_components(freq1: u32, freq2: u32, time_delta: u32) -> Self {
        let hash_value = ((u64::from(freq1) & 0x0F_FFFF) << 44)
            | ((u64::from(freq2) & 0x0F_FFFF) << 24)
            | (u64::from(time_delta) & 0x00FF_FFFF);

        Self { value: hash_value }
    }

    /// Get raw hash value.
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.value
    }

    /// Extract frequency 1 from hash.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn freq1(&self) -> u32 {
        ((self.value >> 44) & 0x0F_FFFF) as u32
    }

    /// Extract frequency 2 from hash.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn freq2(&self) -> u32 {
        ((self.value >> 24) & 0x0F_FFFF) as u32
    }

    /// Extract time delta from hash.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn time_delta(&self) -> u32 {
        (self.value & 0x00FF_FFFF) as u32
    }

    /// Convert to bytes for serialization.
    #[must_use]
    pub const fn to_bytes(&self) -> [u8; 8] {
        self.value.to_le_bytes()
    }

    /// Create from bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: u64::from_le_bytes(bytes),
        }
    }
}

impl From<u64> for Hash {
    fn from(value: u64) -> Self {
        Self { value }
    }
}

impl From<Hash> for u64 {
    fn from(hash: Hash) -> Self {
        hash.value
    }
}

/// Hash generator using combinatorial pairing.
pub struct HashGenerator {
    target_zone_size: usize,
    target_zone_offset: usize,
    num_targets_per_anchor: usize,
}

impl HashGenerator {
    /// Create a new hash generator.
    #[must_use]
    pub const fn new(
        target_zone_size: usize,
        target_zone_offset: usize,
        num_targets_per_anchor: usize,
    ) -> Self {
        Self {
            target_zone_size,
            target_zone_offset,
            num_targets_per_anchor,
        }
    }

    /// Generate hashes from constellation map.
    ///
    /// For each peak (anchor), finds nearby peaks (targets) and creates hashes
    /// encoding their relationship.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate(&self, constellation: &ConstellationMap) -> Vec<(Hash, f64)> {
        let mut hashes = Vec::new();
        let mut seen_hashes = HashSet::new();

        for anchor in &constellation.peaks {
            // Define target zone (time range after anchor)
            let zone_start = self.target_zone_offset as f64 * 0.01; // Convert to seconds
            let zone_end = (self.target_zone_offset + self.target_zone_size) as f64 * 0.01;

            // Find target peaks in the zone
            let targets = constellation.nearest_peaks(
                anchor,
                (zone_start, zone_end),
                self.num_targets_per_anchor,
            );

            // Create hash for each anchor-target pair
            for target in targets {
                let hash = Hash::from_peaks(anchor, target);

                // Avoid duplicate hashes
                if seen_hashes.insert(hash) {
                    hashes.push((hash, anchor.time));
                }
            }
        }

        hashes
    }

    /// Generate hashes with filtering.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_filtered(
        &self,
        constellation: &ConstellationMap,
        min_time_delta: f64,
        max_time_delta: f64,
    ) -> Vec<(Hash, f64)> {
        let mut hashes = Vec::new();
        let mut seen_hashes = HashSet::new();

        for anchor in &constellation.peaks {
            // Define target zone
            let zone_start = min_time_delta;
            let zone_end = max_time_delta;

            // Find target peaks
            let targets = constellation.nearest_peaks(
                anchor,
                (zone_start, zone_end),
                self.num_targets_per_anchor,
            );

            // Create hashes
            for target in targets {
                let time_diff = target.time - anchor.time;

                // Apply time delta constraints
                if time_diff >= min_time_delta && time_diff <= max_time_delta {
                    let hash = Hash::from_peaks(anchor, target);

                    if seen_hashes.insert(hash) {
                        hashes.push((hash, anchor.time));
                    }
                }
            }
        }

        hashes
    }

    /// Generate hashes in batches (for large constellations).
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_batched(
        &self,
        constellation: &ConstellationMap,
        batch_size: usize,
    ) -> Vec<Vec<(Hash, f64)>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut seen_hashes = HashSet::new();

        let zone_start = self.target_zone_offset as f64 * 0.01;
        let zone_end = (self.target_zone_offset + self.target_zone_size) as f64 * 0.01;

        for anchor in &constellation.peaks {
            let targets = constellation.nearest_peaks(
                anchor,
                (zone_start, zone_end),
                self.num_targets_per_anchor,
            );

            for target in targets {
                let hash = Hash::from_peaks(anchor, target);

                if seen_hashes.insert(hash) {
                    current_batch.push((hash, anchor.time));

                    if current_batch.len() >= batch_size {
                        batches.push(current_batch);
                        current_batch = Vec::new();
                    }
                }
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }

    /// Estimate number of hashes that will be generated.
    #[must_use]
    pub fn estimate_hash_count(&self, peak_count: usize) -> usize {
        // Each peak becomes an anchor, pairing with multiple targets
        peak_count * self.num_targets_per_anchor
    }

    /// Get hash statistics from a set of hashes.
    #[must_use]
    pub fn statistics(hashes: &[(Hash, f64)]) -> HashStatistics {
        if hashes.is_empty() {
            return HashStatistics::default();
        }

        // Count unique hashes
        let unique_hashes: HashSet<Hash> = hashes.iter().map(|(h, _)| *h).collect();

        // Time distribution
        let min_time = hashes
            .iter()
            .map(|(_, t)| *t)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_time = hashes
            .iter()
            .map(|(_, t)| *t)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Time delta distribution
        let time_deltas: Vec<u32> = hashes.iter().map(|(h, _)| h.time_delta()).collect();
        let avg_time_delta = if !time_deltas.is_empty() {
            time_deltas.iter().map(|&x| u64::from(x)).sum::<u64>() / time_deltas.len() as u64
        } else {
            0
        };

        HashStatistics {
            total_count: hashes.len(),
            unique_count: unique_hashes.len(),
            time_range: (min_time, max_time),
            avg_time_delta: avg_time_delta as u32,
            collision_rate: if hashes.len() > 0 {
                1.0 - (unique_hashes.len() as f64 / hashes.len() as f64)
            } else {
                0.0
            },
        }
    }
}

impl Default for HashGenerator {
    fn default() -> Self {
        Self::new(32, 1, 5)
    }
}

/// Hash statistics.
#[derive(Clone, Debug, Default)]
pub struct HashStatistics {
    /// Total number of hashes generated.
    pub total_count: usize,
    /// Number of unique hashes.
    pub unique_count: usize,
    /// Time range covered.
    pub time_range: (f64, f64),
    /// Average time delta between pairs.
    pub avg_time_delta: u32,
    /// Hash collision rate (0-1).
    pub collision_rate: f64,
}

impl HashStatistics {
    /// Check if statistics indicate good hash quality.
    #[must_use]
    pub fn is_good_quality(&self) -> bool {
        self.unique_count > 100 && self.collision_rate < 0.5
    }

    /// Get hash density (hashes per second).
    #[must_use]
    pub fn density(&self) -> f64 {
        let duration = self.time_range.1 - self.time_range.0;
        if duration > 0.0 {
            self.total_count as f64 / duration
        } else {
            0.0
        }
    }
}

/// Hash comparison utilities.
pub struct HashComparison;

impl HashComparison {
    /// Calculate Hamming distance between two hashes.
    #[must_use]
    pub fn hamming_distance(h1: Hash, h2: Hash) -> u32 {
        (h1.value() ^ h2.value()).count_ones()
    }

    /// Check if two hashes are similar (within tolerance).
    #[must_use]
    pub fn are_similar(h1: Hash, h2: Hash, max_distance: u32) -> bool {
        Self::hamming_distance(h1, h2) <= max_distance
    }

    /// Find similar hashes in a collection.
    #[must_use]
    pub fn find_similar(
        query: Hash,
        candidates: &[(Hash, f64)],
        max_distance: u32,
    ) -> Vec<&(Hash, f64)> {
        candidates
            .iter()
            .filter(|(h, _)| Self::are_similar(query, *h, max_distance))
            .collect()
    }

    /// Calculate similarity ratio between two hash sets.
    #[must_use]
    pub fn similarity_ratio(set1: &[(Hash, f64)], set2: &[(Hash, f64)]) -> f64 {
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let hashes1: HashSet<Hash> = set1.iter().map(|(h, _)| *h).collect();
        let hashes2: HashSet<Hash> = set2.iter().map(|(h, _)| *h).collect();

        let intersection = hashes1.intersection(&hashes2).count();
        let union = hashes1.union(&hashes2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
}
