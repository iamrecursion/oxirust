#![allow(dead_code)]
//! Noise profile matching and adaptive noise reduction.
//!
//! This module learns a noise fingerprint from a "noise-only" reference segment
//! and then applies spectral subtraction guided by that profile to suppress
//! matching noise across an entire recording. This is useful for removing
//! consistent background noise like HVAC hum, tape hiss, or room tone.

/// Number of frequency bins used in a compact noise profile.
const PROFILE_BINS: usize = 128;

/// A captured noise profile represented as average spectral magnitudes.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseProfile {
    /// Average magnitude per frequency bin.
    pub magnitudes: Vec<f64>,
    /// Sample rate the profile was captured at.
    pub sample_rate: u32,
    /// Number of frames that were averaged to produce this profile.
    pub frame_count: usize,
}

impl NoiseProfile {
    /// Create a new empty noise profile for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            magnitudes: vec![0.0; PROFILE_BINS],
            sample_rate,
            frame_count: 0,
        }
    }

    /// Create a noise profile from pre-computed magnitudes.
    pub fn from_magnitudes(magnitudes: Vec<f64>, sample_rate: u32) -> Self {
        Self {
            magnitudes,
            sample_rate,
            frame_count: 1,
        }
    }

    /// Return the number of bins in this profile.
    pub fn num_bins(&self) -> usize {
        self.magnitudes.len()
    }

    /// Return the peak magnitude across all bins.
    pub fn peak_magnitude(&self) -> f64 {
        self.magnitudes.iter().copied().fold(0.0_f64, f64::max)
    }

    /// Normalize the profile so the peak is 1.0.
    pub fn normalize(&mut self) {
        let peak = self.peak_magnitude();
        if peak > 1e-15 {
            for m in &mut self.magnitudes {
                *m /= peak;
            }
        }
    }

    /// Compute the spectral distance (Euclidean) between two noise profiles.
    pub fn distance(&self, other: &NoiseProfile) -> f64 {
        let len = self.magnitudes.len().min(other.magnitudes.len());
        let mut sum_sq = 0.0;
        for i in 0..len {
            let diff = self.magnitudes[i] - other.magnitudes[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Compute the cosine similarity between two noise profiles.
    pub fn cosine_similarity(&self, other: &NoiseProfile) -> f64 {
        let len = self.magnitudes.len().min(other.magnitudes.len());
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..len {
            dot += self.magnitudes[i] * other.magnitudes[i];
            norm_a += self.magnitudes[i] * self.magnitudes[i];
            norm_b += other.magnitudes[i] * other.magnitudes[i];
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            dot / denom
        }
    }
}

/// Builder that accumulates spectral frames to produce a noise profile.
#[derive(Debug)]
pub struct NoiseProfileBuilder {
    /// Accumulator for each bin.
    accumulator: Vec<f64>,
    /// Number of frames accumulated.
    frame_count: usize,
    /// Sample rate.
    sample_rate: u32,
}

impl NoiseProfileBuilder {
    /// Create a new builder for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            accumulator: vec![0.0; PROFILE_BINS],
            frame_count: 0,
            sample_rate,
        }
    }

    /// Add a spectral frame (magnitudes) to the running average.
    ///
    /// The frame will be resampled to `PROFILE_BINS` bins via linear interpolation
    /// if its length differs.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_frame(&mut self, magnitudes: &[f64]) {
        let resampled = resample_bins(magnitudes, PROFILE_BINS);
        for (acc, &val) in self.accumulator.iter_mut().zip(resampled.iter()) {
            *acc += val;
        }
        self.frame_count += 1;
    }

    /// Finalize and produce the averaged noise profile.
    #[allow(clippy::cast_precision_loss)]
    pub fn build(self) -> NoiseProfile {
        let mut mags = self.accumulator;
        if self.frame_count > 0 {
            let n = self.frame_count as f64;
            for m in &mut mags {
                *m /= n;
            }
        }
        NoiseProfile {
            magnitudes: mags,
            sample_rate: self.sample_rate,
            frame_count: self.frame_count,
        }
    }

    /// Return how many frames have been accumulated.
    pub fn frames_accumulated(&self) -> usize {
        self.frame_count
    }
}

/// Spectral subtraction parameters for profile-guided noise reduction.
#[derive(Debug, Clone)]
pub struct SpectralSubtractConfig {
    /// Over-subtraction factor (alpha). Values > 1.0 remove more noise.
    pub alpha: f64,
    /// Spectral floor to prevent musical noise artifacts.
    pub spectral_floor: f64,
    /// Smoothing factor for the noise estimate (0.0-1.0).
    pub smoothing: f64,
}

impl Default for SpectralSubtractConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            spectral_floor: 0.01,
            smoothing: 0.9,
        }
    }
}

/// Apply spectral subtraction to magnitude bins using a noise profile.
///
/// Returns the cleaned magnitude bins.
pub fn spectral_subtract(
    signal_mags: &[f64],
    noise_profile: &NoiseProfile,
    config: &SpectralSubtractConfig,
) -> Vec<f64> {
    let profile_resampled = resample_bins(&noise_profile.magnitudes, signal_mags.len());
    let mut output = Vec::with_capacity(signal_mags.len());
    for (i, &sig) in signal_mags.iter().enumerate() {
        let noise_est = profile_resampled[i] * config.alpha;
        let cleaned = sig - noise_est;
        let floored = cleaned.max(sig * config.spectral_floor);
        output.push(floored.max(0.0));
    }
    output
}

/// Resample a vector of bins to a target length using linear interpolation.
#[allow(clippy::cast_precision_loss)]
fn resample_bins(input: &[f64], target_len: usize) -> Vec<f64> {
    if input.is_empty() || target_len == 0 {
        return vec![0.0; target_len];
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    let mut output = Vec::with_capacity(target_len);
    let ratio = (input.len() - 1) as f64 / (target_len - 1).max(1) as f64;
    for i in 0..target_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = pos - idx as f64;
        let val = if idx + 1 < input.len() {
            input[idx] + frac * (input[idx + 1] - input[idx])
        } else {
            input[input.len() - 1]
        };
        output.push(val);
    }
    output
}

/// A library of named noise profiles for quick matching.
#[derive(Debug)]
pub struct NoiseProfileLibrary {
    /// Stored profiles with their labels.
    entries: Vec<(String, NoiseProfile)>,
}

impl NoiseProfileLibrary {
    /// Create a new empty library.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a named profile to the library.
    pub fn add(&mut self, name: impl Into<String>, profile: NoiseProfile) {
        self.entries.push((name.into(), profile));
    }

    /// Find the best matching profile for a given input profile.
    ///
    /// Returns the name and similarity score of the best match, or `None` if empty.
    pub fn find_best_match(&self, target: &NoiseProfile) -> Option<(&str, f64)> {
        self.entries
            .iter()
            .map(|(name, prof)| (name.as_str(), prof.cosine_similarity(target)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Return the number of profiles in the library.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for NoiseProfileLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_profile_new() {
        let np = NoiseProfile::new(44100);
        assert_eq!(np.num_bins(), PROFILE_BINS);
        assert_eq!(np.sample_rate, 44100);
        assert_eq!(np.frame_count, 0);
    }

    #[test]
    fn test_noise_profile_peak() {
        let mags = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let np = NoiseProfile::from_magnitudes(mags, 44100);
        assert!((np.peak_magnitude() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_noise_profile_normalize() {
        let mags = vec![0.2, 0.4, 0.8];
        let mut np = NoiseProfile::from_magnitudes(mags, 44100);
        np.normalize();
        assert!((np.peak_magnitude() - 1.0).abs() < 1e-10);
        assert!((np.magnitudes[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_noise_profile_distance_identical() {
        let mags = vec![1.0, 2.0, 3.0];
        let a = NoiseProfile::from_magnitudes(mags.clone(), 44100);
        let b = NoiseProfile::from_magnitudes(mags, 44100);
        assert!(a.distance(&b) < 1e-10);
    }

    #[test]
    fn test_noise_profile_distance_different() {
        let a = NoiseProfile::from_magnitudes(vec![1.0, 0.0], 44100);
        let b = NoiseProfile::from_magnitudes(vec![0.0, 1.0], 44100);
        let d = a.distance(&b);
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mags = vec![1.0, 2.0, 3.0];
        let a = NoiseProfile::from_magnitudes(mags.clone(), 44100);
        let b = NoiseProfile::from_magnitudes(mags, 44100);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = NoiseProfile::from_magnitudes(vec![1.0, 0.0], 44100);
        let b = NoiseProfile::from_magnitudes(vec![0.0, 1.0], 44100);
        assert!(a.cosine_similarity(&b).abs() < 1e-10);
    }

    #[test]
    fn test_builder_accumulation() {
        let mut builder = NoiseProfileBuilder::new(48000);
        builder.add_frame(&[1.0; 128]);
        builder.add_frame(&[3.0; 128]);
        assert_eq!(builder.frames_accumulated(), 2);
        let profile = builder.build();
        // Average should be 2.0
        for &m in &profile.magnitudes {
            assert!((m - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spectral_subtract_removes_noise() {
        let signal = vec![1.0, 0.8, 0.6, 0.5];
        let noise = NoiseProfile::from_magnitudes(vec![0.2, 0.2, 0.2, 0.2], 44100);
        let config = SpectralSubtractConfig {
            alpha: 1.0,
            spectral_floor: 0.0,
            smoothing: 0.9,
        };
        let cleaned = spectral_subtract(&signal, &noise, &config);
        assert_eq!(cleaned.len(), 4);
        assert!((cleaned[0] - 0.8).abs() < 0.01);
        assert!((cleaned[3] - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_spectral_floor_prevents_negatives() {
        let signal = vec![0.1];
        let noise = NoiseProfile::from_magnitudes(vec![1.0], 44100);
        let config = SpectralSubtractConfig {
            alpha: 1.0,
            spectral_floor: 0.01,
            smoothing: 0.9,
        };
        let cleaned = spectral_subtract(&signal, &noise, &config);
        assert!(cleaned[0] >= 0.0);
    }

    #[test]
    fn test_library_find_best_match() {
        let mut lib = NoiseProfileLibrary::new();
        lib.add(
            "hiss",
            NoiseProfile::from_magnitudes(vec![1.0, 1.0, 1.0], 44100),
        );
        lib.add(
            "hum",
            NoiseProfile::from_magnitudes(vec![1.0, 0.0, 0.0], 44100),
        );

        let target = NoiseProfile::from_magnitudes(vec![0.9, 0.1, 0.05], 44100);
        let (name, _score) = lib
            .find_best_match(&target)
            .expect("should succeed in test");
        assert_eq!(name, "hum");
    }

    #[test]
    fn test_library_empty() {
        let lib = NoiseProfileLibrary::new();
        assert!(lib.is_empty());
        let target = NoiseProfile::new(44100);
        assert!(lib.find_best_match(&target).is_none());
    }

    #[test]
    fn test_resample_bins_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = resample_bins(&input, 4);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_resample_bins_upscale() {
        let input = vec![0.0, 1.0];
        let output = resample_bins(&input, 3);
        assert_eq!(output.len(), 3);
        assert!((output[0] - 0.0).abs() < 1e-10);
        assert!((output[1] - 0.5).abs() < 1e-10);
        assert!((output[2] - 1.0).abs() < 1e-10);
    }
}
