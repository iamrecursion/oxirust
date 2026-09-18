//! Constellation map representation of spectral peaks.

use std::collections::HashMap;

/// A peak in the time-frequency domain.
#[derive(Clone, Debug)]
pub struct Peak {
    /// Time position (seconds).
    pub time: f64,
    /// Frequency (Hz).
    pub frequency: f64,
    /// Magnitude (amplitude).
    pub magnitude: f64,
    /// Frequency bin index.
    pub bin: usize,
}

impl Peak {
    /// Create a new peak.
    #[must_use]
    pub const fn new(time: f64, frequency: f64, magnitude: f64, bin: usize) -> Self {
        Self {
            time,
            frequency,
            magnitude,
            bin,
        }
    }

    /// Get quantized frequency for hashing (in bins).
    #[must_use]
    pub const fn quantized_frequency(&self) -> u32 {
        self.bin as u32
    }

    /// Get quantized time for hashing (in milliseconds).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn quantized_time(&self) -> u32 {
        (self.time * 1000.0) as u32
    }

    /// Distance to another peak in time.
    #[must_use]
    pub fn time_distance(&self, other: &Self) -> f64 {
        (self.time - other.time).abs()
    }

    /// Distance to another peak in frequency.
    #[must_use]
    pub fn frequency_distance(&self, other: &Self) -> f64 {
        (self.frequency - other.frequency).abs()
    }

    /// Combined time-frequency distance.
    #[must_use]
    pub fn distance(&self, other: &Self, time_weight: f64, freq_weight: f64) -> f64 {
        let time_dist = self.time_distance(other);
        let freq_dist = self.frequency_distance(other);
        (time_weight * time_dist * time_dist + freq_weight * freq_dist * freq_dist).sqrt()
    }
}

/// Constellation map: a sparse representation of spectral peaks.
#[derive(Clone, Debug)]
pub struct ConstellationMap {
    /// All peaks in time-frequency space.
    pub peaks: Vec<Peak>,
    /// Sample rate of source audio.
    pub sample_rate: u32,
    /// Duration of source audio.
    pub duration: f64,
    /// Peaks indexed by time bins for fast lookup.
    time_index: HashMap<u32, Vec<usize>>,
}

impl ConstellationMap {
    /// Create a new constellation map.
    #[must_use]
    pub fn new(peaks: Vec<Peak>, sample_rate: u32, duration: f64) -> Self {
        let time_index = Self::build_time_index(&peaks);

        Self {
            peaks,
            sample_rate,
            duration,
            time_index,
        }
    }

    /// Build time index for fast lookup.
    fn build_time_index(peaks: &[Peak]) -> HashMap<u32, Vec<usize>> {
        let mut index = HashMap::new();

        for (i, peak) in peaks.iter().enumerate() {
            let time_bin = peak.quantized_time() / 100; // 100ms bins
            index.entry(time_bin).or_insert_with(Vec::new).push(i);
        }

        index
    }

    /// Get peaks in a time range.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn peaks_in_time_range(&self, start_time: f64, end_time: f64) -> Vec<&Peak> {
        let start_bin = ((start_time * 1000.0) as u32) / 100;
        let end_bin = ((end_time * 1000.0) as u32) / 100;

        let mut result = Vec::new();

        for bin in start_bin..=end_bin {
            if let Some(indices) = self.time_index.get(&bin) {
                for &idx in indices {
                    if let Some(peak) = self.peaks.get(idx) {
                        if peak.time >= start_time && peak.time <= end_time {
                            result.push(peak);
                        }
                    }
                }
            }
        }

        result
    }

    /// Get peaks in a frequency range.
    #[must_use]
    pub fn peaks_in_frequency_range(&self, min_freq: f64, max_freq: f64) -> Vec<&Peak> {
        self.peaks
            .iter()
            .filter(|p| p.frequency >= min_freq && p.frequency <= max_freq)
            .collect()
    }

    /// Get nearest peaks to a given peak within a region.
    #[must_use]
    pub fn nearest_peaks(
        &self,
        anchor: &Peak,
        time_range: (f64, f64),
        max_count: usize,
    ) -> Vec<&Peak> {
        let start_time = anchor.time + time_range.0;
        let end_time = anchor.time + time_range.1;

        let mut candidates = self.peaks_in_time_range(start_time, end_time);

        // Sort by time-frequency distance
        candidates.sort_by(|a, b| {
            let dist_a = anchor.distance(a, 1.0, 0.1);
            let dist_b = anchor.distance(b, 1.0, 0.1);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(max_count);
        candidates
    }

    /// Get peak count.
    #[must_use]
    pub fn peak_count(&self) -> usize {
        self.peaks.len()
    }

    /// Get peak density (peaks per second).
    #[must_use]
    pub fn peak_density(&self) -> f64 {
        if self.duration > 0.0 {
            self.peaks.len() as f64 / self.duration
        } else {
            0.0
        }
    }

    /// Get frequency range of peaks.
    #[must_use]
    pub fn frequency_range(&self) -> (f64, f64) {
        if self.peaks.is_empty() {
            return (0.0, 0.0);
        }

        let min_freq = self
            .peaks
            .iter()
            .map(|p| p.frequency)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_freq = self
            .peaks
            .iter()
            .map(|p| p.frequency)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        (min_freq, max_freq)
    }

    /// Filter peaks by magnitude threshold.
    #[must_use]
    pub fn filter_by_magnitude(&self, threshold: f64) -> Self {
        let filtered_peaks: Vec<Peak> = self
            .peaks
            .iter()
            .filter(|p| p.magnitude >= threshold)
            .cloned()
            .collect();

        Self::new(filtered_peaks, self.sample_rate, self.duration)
    }

    /// Filter peaks by frequency range.
    #[must_use]
    pub fn filter_by_frequency(&self, min_freq: f64, max_freq: f64) -> Self {
        let filtered_peaks: Vec<Peak> = self
            .peaks
            .iter()
            .filter(|p| p.frequency >= min_freq && p.frequency <= max_freq)
            .cloned()
            .collect();

        Self::new(filtered_peaks, self.sample_rate, self.duration)
    }

    /// Get statistics about the constellation.
    #[must_use]
    pub fn statistics(&self) -> ConstellationStatistics {
        if self.peaks.is_empty() {
            return ConstellationStatistics::default();
        }

        let total_magnitude: f64 = self.peaks.iter().map(|p| p.magnitude).sum();
        let avg_magnitude = total_magnitude / self.peaks.len() as f64;

        let max_magnitude = self
            .peaks
            .iter()
            .map(|p| p.magnitude)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let min_magnitude = self
            .peaks
            .iter()
            .map(|p| p.magnitude)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let (min_freq, max_freq) = self.frequency_range();

        ConstellationStatistics {
            peak_count: self.peaks.len(),
            density: self.peak_density(),
            avg_magnitude,
            max_magnitude,
            min_magnitude,
            frequency_range: (min_freq, max_freq),
        }
    }

    /// Subsample constellation (reduce peak count).
    #[must_use]
    pub fn subsample(&self, factor: usize) -> Self {
        if factor <= 1 {
            return self.clone();
        }

        let subsampled_peaks: Vec<Peak> = self.peaks.iter().step_by(factor).cloned().collect();

        Self::new(subsampled_peaks, self.sample_rate, self.duration)
    }

    /// Merge with another constellation map.
    #[must_use]
    pub fn merge(&self, other: &Self, time_offset: f64) -> Self {
        let mut merged_peaks = self.peaks.clone();

        for peak in &other.peaks {
            let mut shifted_peak = peak.clone();
            shifted_peak.time += time_offset;
            merged_peaks.push(shifted_peak);
        }

        let total_duration = self.duration.max(other.duration + time_offset);

        Self::new(merged_peaks, self.sample_rate, total_duration)
    }
}

/// Statistics about a constellation map.
#[derive(Clone, Debug, Default)]
pub struct ConstellationStatistics {
    /// Total number of peaks.
    pub peak_count: usize,
    /// Peak density (peaks per second).
    pub density: f64,
    /// Average peak magnitude.
    pub avg_magnitude: f64,
    /// Maximum peak magnitude.
    pub max_magnitude: f64,
    /// Minimum peak magnitude.
    pub min_magnitude: f64,
    /// Frequency range covered.
    pub frequency_range: (f64, f64),
}

impl ConstellationStatistics {
    /// Check if statistics indicate a valid constellation.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.peak_count > 0 && self.density > 0.0 && self.max_magnitude > 0.0
    }

    /// Get quality score (0-1, based on density and magnitude).
    #[must_use]
    pub fn quality_score(&self) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }

        // Ideal density is around 10-50 peaks per second
        let density_score = if self.density < 10.0 {
            self.density / 10.0
        } else if self.density > 50.0 {
            50.0 / self.density
        } else {
            1.0
        };

        // Magnitude dynamic range
        let dynamic_range = if self.min_magnitude > 0.0 {
            (self.max_magnitude / self.min_magnitude).ln() / 10.0
        } else {
            0.5
        };

        (density_score * 0.7 + dynamic_range.min(1.0) * 0.3).min(1.0)
    }
}
