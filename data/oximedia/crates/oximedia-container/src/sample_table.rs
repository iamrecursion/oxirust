#![allow(dead_code)]
//! ISO Base Media File Format sample table abstractions.
//!
//! Models the `stbl` box family (`stts`, `stsc`, `stsz`, `stco`, `stss`)
//! providing sample-to-chunk, sample-size, chunk-offset, and sync-sample
//! look-ups required for random access into MP4/MOV containers.

use std::collections::BTreeSet;

/// Time-to-sample entry (`stts` box row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeToSampleEntry {
    /// Number of consecutive samples with the same delta.
    pub sample_count: u32,
    /// Duration of each sample in timescale units.
    pub sample_delta: u32,
}

/// Sample-to-chunk entry (`stsc` box row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleToChunkEntry {
    /// First chunk number for this run (1-based).
    pub first_chunk: u32,
    /// Number of samples per chunk in this run.
    pub samples_per_chunk: u32,
    /// Sample description index (1-based).
    pub sample_description_index: u32,
}

/// How sample sizes are stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleSizeMode {
    /// All samples share the same size.
    Uniform(u32),
    /// Per-sample size table.
    Variable(Vec<u32>),
}

/// Complete sample table for one track.
#[derive(Debug, Clone)]
pub struct SampleTable {
    /// Time-to-sample entries from `stts`.
    pub time_to_sample: Vec<TimeToSampleEntry>,
    /// Sample-to-chunk entries from `stsc`.
    pub sample_to_chunk: Vec<SampleToChunkEntry>,
    /// Per-sample sizes from `stsz`.
    pub sample_sizes: SampleSizeMode,
    /// Chunk offsets from `stco` / `co64`.
    pub chunk_offsets: Vec<u64>,
    /// Set of sync (key-frame) sample numbers (1-based) from `stss`.
    /// If `None`, every sample is a sync sample.
    pub sync_samples: Option<BTreeSet<u32>>,
    /// Track timescale (ticks per second).
    pub timescale: u32,
}

impl SampleTable {
    /// Creates an empty sample table with the given timescale.
    #[must_use]
    pub fn new(timescale: u32) -> Self {
        Self {
            time_to_sample: Vec::new(),
            sample_to_chunk: Vec::new(),
            sample_sizes: SampleSizeMode::Uniform(0),
            chunk_offsets: Vec::new(),
            sync_samples: None,
            timescale,
        }
    }

    /// Returns the total number of samples described by the `stts` entries.
    #[must_use]
    pub fn sample_count_from_stts(&self) -> u64 {
        self.time_to_sample
            .iter()
            .map(|e| u64::from(e.sample_count))
            .sum()
    }

    /// Returns the total number of samples from the size table.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        match &self.sample_sizes {
            SampleSizeMode::Uniform(_) => self.sample_count_from_stts(),
            SampleSizeMode::Variable(sizes) => sizes.len() as u64,
        }
    }

    /// Returns the total media duration in timescale units.
    #[must_use]
    pub fn total_duration_ticks(&self) -> u64 {
        self.time_to_sample
            .iter()
            .map(|e| u64::from(e.sample_count) * u64::from(e.sample_delta))
            .sum()
    }

    /// Returns the total media duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        self.total_duration_ticks() as f64 / f64::from(self.timescale)
    }

    /// Returns the size of the given sample (1-based index).
    #[must_use]
    pub fn sample_size(&self, sample_number: u32) -> Option<u32> {
        if sample_number == 0 {
            return None;
        }
        match &self.sample_sizes {
            SampleSizeMode::Uniform(sz) => {
                if u64::from(sample_number) <= self.sample_count() {
                    Some(*sz)
                } else {
                    None
                }
            }
            SampleSizeMode::Variable(sizes) => sizes.get((sample_number - 1) as usize).copied(),
        }
    }

    /// Returns `true` if the given sample (1-based) is a sync sample.
    #[must_use]
    pub fn is_sync_sample(&self, sample_number: u32) -> bool {
        match &self.sync_samples {
            None => true, // all samples are sync
            Some(set) => set.contains(&sample_number),
        }
    }

    /// Finds the nearest sync sample at or before `sample_number` (1-based).
    #[must_use]
    pub fn nearest_sync_before(&self, sample_number: u32) -> Option<u32> {
        match &self.sync_samples {
            None => Some(sample_number),
            Some(set) => set.range(..=sample_number).next_back().copied(),
        }
    }

    /// Finds the nearest sync sample at or after `sample_number` (1-based).
    #[must_use]
    pub fn nearest_sync_after(&self, sample_number: u32) -> Option<u32> {
        match &self.sync_samples {
            None => Some(sample_number),
            Some(set) => set.range(sample_number..).next().copied(),
        }
    }

    /// Converts a decode timestamp (in timescale ticks) to a sample number
    /// (1-based). Returns `None` if the timestamp exceeds the track duration.
    #[must_use]
    pub fn sample_at_time(&self, ticks: u64) -> Option<u32> {
        let mut remaining = ticks;
        let mut sample_num: u64 = 1;

        for entry in &self.time_to_sample {
            let run_duration = u64::from(entry.sample_count) * u64::from(entry.sample_delta);
            if remaining < run_duration {
                if entry.sample_delta == 0 {
                    #[allow(clippy::cast_possible_truncation)]
                    return Some(sample_num as u32);
                }
                let offset = remaining / u64::from(entry.sample_delta);
                #[allow(clippy::cast_possible_truncation)]
                return Some((sample_num + offset) as u32);
            }
            remaining -= run_duration;
            sample_num += u64::from(entry.sample_count);
        }
        None
    }

    /// Returns the decode timestamp (in ticks) of a given sample (1-based).
    #[must_use]
    pub fn sample_time(&self, sample_number: u32) -> Option<u64> {
        if sample_number == 0 {
            return None;
        }
        let target = u64::from(sample_number);
        let mut current_sample: u64 = 1;
        let mut current_time: u64 = 0;

        for entry in &self.time_to_sample {
            let count = u64::from(entry.sample_count);
            if target < current_sample + count {
                let offset = target - current_sample;
                return Some(current_time + offset * u64::from(entry.sample_delta));
            }
            current_time += count * u64::from(entry.sample_delta);
            current_sample += count;
        }
        None
    }

    /// Total data size of all samples.
    #[must_use]
    pub fn total_data_size(&self) -> u64 {
        match &self.sample_sizes {
            SampleSizeMode::Uniform(sz) => u64::from(*sz) * self.sample_count(),
            SampleSizeMode::Variable(sizes) => sizes.iter().map(|&s| u64::from(s)).sum(),
        }
    }

    /// Returns the average sample size in bytes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_sample_size(&self) -> f64 {
        let count = self.sample_count();
        if count == 0 {
            return 0.0;
        }
        self.total_data_size() as f64 / count as f64
    }

    /// Number of chunk offsets.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunk_offsets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table_30fps() -> SampleTable {
        let mut st = SampleTable::new(30_000);
        // 300 samples at delta 1000 → 10 seconds at 30000 timescale
        st.time_to_sample.push(TimeToSampleEntry {
            sample_count: 300,
            sample_delta: 1000,
        });
        st.sample_sizes = SampleSizeMode::Uniform(4096);
        st.chunk_offsets = vec![0, 40960, 81920]; // 3 chunks
        st.sample_to_chunk.push(SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 100,
            sample_description_index: 1,
        });
        st
    }

    #[test]
    fn test_sample_count_from_stts() {
        let st = sample_table_30fps();
        assert_eq!(st.sample_count_from_stts(), 300);
    }

    #[test]
    fn test_sample_count_uniform() {
        let st = sample_table_30fps();
        assert_eq!(st.sample_count(), 300);
    }

    #[test]
    fn test_sample_count_variable() {
        let mut st = SampleTable::new(1000);
        st.sample_sizes = SampleSizeMode::Variable(vec![100, 200, 300]);
        assert_eq!(st.sample_count(), 3);
    }

    #[test]
    fn test_total_duration_ticks() {
        let st = sample_table_30fps();
        assert_eq!(st.total_duration_ticks(), 300_000);
    }

    #[test]
    fn test_duration_seconds() {
        let st = sample_table_30fps();
        assert!((st.duration_seconds() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_duration_seconds_zero_timescale() {
        let st = SampleTable::new(0);
        assert!((st.duration_seconds()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sample_size_uniform() {
        let st = sample_table_30fps();
        assert_eq!(st.sample_size(1), Some(4096));
        assert_eq!(st.sample_size(300), Some(4096));
        assert_eq!(st.sample_size(301), None);
        assert_eq!(st.sample_size(0), None);
    }

    #[test]
    fn test_sample_size_variable() {
        let mut st = SampleTable::new(1000);
        st.sample_sizes = SampleSizeMode::Variable(vec![100, 200, 300]);
        assert_eq!(st.sample_size(1), Some(100));
        assert_eq!(st.sample_size(3), Some(300));
        assert_eq!(st.sample_size(4), None);
    }

    #[test]
    fn test_is_sync_sample_all_sync() {
        let st = sample_table_30fps();
        assert!(st.is_sync_sample(1));
        assert!(st.is_sync_sample(150));
    }

    #[test]
    fn test_is_sync_sample_selective() {
        let mut st = sample_table_30fps();
        let mut syncs = BTreeSet::new();
        syncs.insert(1);
        syncs.insert(30);
        syncs.insert(60);
        st.sync_samples = Some(syncs);

        assert!(st.is_sync_sample(1));
        assert!(st.is_sync_sample(30));
        assert!(!st.is_sync_sample(2));
    }

    #[test]
    fn test_nearest_sync_before() {
        let mut st = sample_table_30fps();
        let mut syncs = BTreeSet::new();
        syncs.insert(1);
        syncs.insert(30);
        syncs.insert(60);
        st.sync_samples = Some(syncs);

        assert_eq!(st.nearest_sync_before(29), Some(1));
        assert_eq!(st.nearest_sync_before(30), Some(30));
        assert_eq!(st.nearest_sync_before(45), Some(30));
    }

    #[test]
    fn test_nearest_sync_after() {
        let mut st = sample_table_30fps();
        let mut syncs = BTreeSet::new();
        syncs.insert(1);
        syncs.insert(30);
        syncs.insert(60);
        st.sync_samples = Some(syncs);

        assert_eq!(st.nearest_sync_after(2), Some(30));
        assert_eq!(st.nearest_sync_after(30), Some(30));
        assert_eq!(st.nearest_sync_after(61), None);
    }

    #[test]
    fn test_sample_at_time() {
        let st = sample_table_30fps();
        // sample 1 is at tick 0, sample 2 at tick 1000, ...
        assert_eq!(st.sample_at_time(0), Some(1));
        assert_eq!(st.sample_at_time(999), Some(1));
        assert_eq!(st.sample_at_time(1000), Some(2));
        assert_eq!(st.sample_at_time(299_999), Some(300));
        assert_eq!(st.sample_at_time(300_000), None); // past end
    }

    #[test]
    fn test_sample_time() {
        let st = sample_table_30fps();
        assert_eq!(st.sample_time(1), Some(0));
        assert_eq!(st.sample_time(2), Some(1000));
        assert_eq!(st.sample_time(300), Some(299_000));
        assert_eq!(st.sample_time(0), None);
        assert_eq!(st.sample_time(301), None);
    }

    #[test]
    fn test_total_data_size() {
        let st = sample_table_30fps();
        assert_eq!(st.total_data_size(), 300 * 4096);
    }

    #[test]
    fn test_average_sample_size() {
        let mut st = SampleTable::new(1000);
        st.sample_sizes = SampleSizeMode::Variable(vec![100, 200, 300]);
        st.time_to_sample.push(TimeToSampleEntry {
            sample_count: 3,
            sample_delta: 1000,
        });
        assert!((st.average_sample_size() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chunk_count() {
        let st = sample_table_30fps();
        assert_eq!(st.chunk_count(), 3);
    }
}
