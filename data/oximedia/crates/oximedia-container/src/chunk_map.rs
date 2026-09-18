//! ISOBMFF `stsc` (Sample-to-Chunk) table abstraction.
//!
//! The `stsc` box maps samples to chunks. Each [`ChunkEntry`] records
//! the first chunk index (1-based), the number of samples in that chunk,
//! and a sample description index.  [`ChunkMap`] wraps a list of entries and
//! provides fast lookups.

#![allow(dead_code)]

/// A single entry from the `stsc` (sample-to-chunk) box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkEntry {
    /// First chunk covered by this entry (1-based, as in the spec).
    pub first_chunk: u32,
    /// Number of samples per chunk for chunks covered by this entry.
    pub samples_per_chunk: u32,
    /// Sample description index (1-based) for these chunks.
    pub sample_description_index: u32,
}

impl ChunkEntry {
    /// Creates a new entry.
    #[must_use]
    pub fn new(first_chunk: u32, samples_per_chunk: u32, sample_description_index: u32) -> Self {
        Self {
            first_chunk,
            samples_per_chunk,
            sample_description_index,
        }
    }

    /// Returns the number of samples in each chunk covered by this entry.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.samples_per_chunk
    }

    /// Returns `true` if this entry covers a single chunk.
    #[must_use]
    pub fn is_single_chunk(&self) -> bool {
        self.samples_per_chunk == 1
    }
}

/// A map from chunks to samples, constructed from the `stsc` box entries.
///
/// Also tracks the total number of chunks (from the `stco`/`co64` box) to
/// allow computing the total sample count.
#[derive(Debug)]
pub struct ChunkMap {
    entries: Vec<ChunkEntry>,
    /// Total number of chunks in the track (from `stco`/`co64`).
    total_chunks: u32,
}

impl ChunkMap {
    /// Creates an empty [`ChunkMap`] with the given total chunk count.
    #[must_use]
    pub fn new(total_chunks: u32) -> Self {
        Self {
            entries: Vec::new(),
            total_chunks,
        }
    }

    /// Adds a [`ChunkEntry`] to the map.
    ///
    /// Entries should be appended in ascending `first_chunk` order.
    pub fn add_chunk(&mut self, entry: ChunkEntry) {
        self.entries.push(entry);
    }

    /// Returns the total number of samples across all chunks.
    ///
    /// Computed by summing samples-per-chunk × number-of-chunks for each entry.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        if self.entries.is_empty() || self.total_chunks == 0 {
            return 0;
        }
        let mut total: u64 = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            let next_first_chunk = self
                .entries
                .get(i + 1)
                .map_or(self.total_chunks + 1, |e| e.first_chunk);
            let chunk_count = u64::from(next_first_chunk.saturating_sub(entry.first_chunk));
            total += chunk_count * u64::from(entry.samples_per_chunk);
        }
        total
    }

    /// Finds the chunk number (1-based) that contains the given sample (0-based).
    ///
    /// Returns `None` if `sample_index` is out of range.
    #[must_use]
    pub fn find_chunk_for_sample(&self, sample_index: u64) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }
        let mut cumulative: u64 = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            let next_first_chunk = self
                .entries
                .get(i + 1)
                .map_or(self.total_chunks + 1, |e| e.first_chunk);
            let chunk_count = u64::from(next_first_chunk.saturating_sub(entry.first_chunk));
            let samples_in_range = chunk_count * u64::from(entry.samples_per_chunk);
            if sample_index < cumulative + samples_in_range {
                // Found the right entry.
                let offset_in_range = sample_index - cumulative;
                let chunk_offset = offset_in_range / u64::from(entry.samples_per_chunk);
                #[allow(clippy::cast_possible_truncation)]
                return Some(entry.first_chunk + chunk_offset as u32);
            }
            cumulative += samples_in_range;
        }
        None
    }

    /// Returns the number of samples in the given chunk (1-based chunk number).
    ///
    /// Returns `None` if the chunk number is 0 or exceeds `total_chunks`.
    #[must_use]
    pub fn samples_in_chunk(&self, chunk: u32) -> Option<u32> {
        if chunk == 0 || chunk > self.total_chunks {
            return None;
        }
        // Find the entry whose range covers `chunk`.
        let entry = self.entries.iter().rev().find(|e| e.first_chunk <= chunk)?;
        Some(entry.samples_per_chunk)
    }

    /// Returns the number of entries in this map.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the map has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over the entries.
    pub fn entries(&self) -> impl Iterator<Item = &ChunkEntry> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_map() -> ChunkMap {
        // 3 chunks, each with 10 samples → 30 total samples
        let mut map = ChunkMap::new(3);
        map.add_chunk(ChunkEntry::new(1, 10, 1));
        map
    }

    #[test]
    fn test_chunk_entry_sample_count() {
        let e = ChunkEntry::new(1, 5, 1);
        assert_eq!(e.sample_count(), 5);
    }

    #[test]
    fn test_chunk_entry_is_single_chunk() {
        let e = ChunkEntry::new(1, 1, 1);
        assert!(e.is_single_chunk());
    }

    #[test]
    fn test_chunk_entry_not_single_chunk() {
        let e = ChunkEntry::new(1, 4, 1);
        assert!(!e.is_single_chunk());
    }

    #[test]
    fn test_chunk_map_total_samples_uniform() {
        let map = simple_map();
        assert_eq!(map.total_samples(), 30);
    }

    #[test]
    fn test_chunk_map_total_samples_empty() {
        let map = ChunkMap::new(0);
        assert_eq!(map.total_samples(), 0);
    }

    #[test]
    fn test_chunk_map_total_samples_two_entries() {
        // Chunks 1-2: 5 samples each; chunks 3-4: 3 samples each → 2*5 + 2*3 = 16
        let mut map = ChunkMap::new(4);
        map.add_chunk(ChunkEntry::new(1, 5, 1));
        map.add_chunk(ChunkEntry::new(3, 3, 1));
        assert_eq!(map.total_samples(), 16);
    }

    #[test]
    fn test_find_chunk_for_sample_first_chunk() {
        let map = simple_map();
        assert_eq!(map.find_chunk_for_sample(0), Some(1));
        assert_eq!(map.find_chunk_for_sample(9), Some(1));
    }

    #[test]
    fn test_find_chunk_for_sample_second_chunk() {
        let map = simple_map();
        assert_eq!(map.find_chunk_for_sample(10), Some(2));
        assert_eq!(map.find_chunk_for_sample(19), Some(2));
    }

    #[test]
    fn test_find_chunk_for_sample_third_chunk() {
        let map = simple_map();
        assert_eq!(map.find_chunk_for_sample(20), Some(3));
        assert_eq!(map.find_chunk_for_sample(29), Some(3));
    }

    #[test]
    fn test_find_chunk_for_sample_out_of_range() {
        let map = simple_map();
        assert!(map.find_chunk_for_sample(30).is_none());
    }

    #[test]
    fn test_find_chunk_for_sample_empty_map() {
        let map = ChunkMap::new(5);
        assert!(map.find_chunk_for_sample(0).is_none());
    }

    #[test]
    fn test_samples_in_chunk_valid() {
        let map = simple_map();
        assert_eq!(map.samples_in_chunk(1), Some(10));
        assert_eq!(map.samples_in_chunk(3), Some(10));
    }

    #[test]
    fn test_samples_in_chunk_zero_returns_none() {
        let map = simple_map();
        assert!(map.samples_in_chunk(0).is_none());
    }

    #[test]
    fn test_samples_in_chunk_beyond_total_returns_none() {
        let map = simple_map();
        assert!(map.samples_in_chunk(4).is_none());
    }

    #[test]
    fn test_chunk_map_is_empty() {
        let map = ChunkMap::new(0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_chunk_map_entry_count() {
        let map = simple_map();
        assert_eq!(map.entry_count(), 1);
    }

    #[test]
    fn test_chunk_map_entries_iter() {
        let mut map = ChunkMap::new(4);
        map.add_chunk(ChunkEntry::new(1, 5, 1));
        map.add_chunk(ChunkEntry::new(3, 3, 1));
        let counts: Vec<u32> = map.entries().map(|e| e.samples_per_chunk).collect();
        assert_eq!(counts, vec![5, 3]);
    }
}
