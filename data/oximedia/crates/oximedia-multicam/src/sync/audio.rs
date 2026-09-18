//! Audio cross-correlation synchronization for multi-camera production.
//!
//! This module provides audio-based synchronization using cross-correlation.

use super::{SyncConfig, SyncMethod, SyncOffset, SyncResult, Synchronizer};
use crate::{AngleId, Result};

/// Audio synchronizer using cross-correlation
#[derive(Debug)]
pub struct AudioSync {
    /// Audio samples for each angle (mono, downmixed)
    audio_tracks: Vec<Vec<f32>>,
    /// Sample rate for each track
    sample_rates: Vec<u32>,
}

impl AudioSync {
    /// Create a new audio synchronizer
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            audio_tracks: vec![Vec::new(); angle_count],
            sample_rates: vec![0; angle_count],
        }
    }

    /// Add audio track for an angle
    pub fn add_audio(&mut self, angle: AngleId, samples: Vec<f32>, sample_rate: u32) {
        if angle < self.audio_tracks.len() {
            self.audio_tracks[angle] = samples;
            self.sample_rates[angle] = sample_rate;
        }
    }

    /// Find offset using cross-correlation
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails
    pub fn find_offset(
        &self,
        angle_a: AngleId,
        angle_b: AngleId,
        config: &SyncConfig,
    ) -> Result<SyncOffset> {
        if angle_a >= self.audio_tracks.len() || angle_b >= self.audio_tracks.len() {
            return Err(crate::MultiCamError::AngleNotFound(angle_a.max(angle_b)));
        }

        let audio_a = &self.audio_tracks[angle_a];
        let audio_b = &self.audio_tracks[angle_b];

        if audio_a.is_empty() || audio_b.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No audio data available".to_string(),
            ));
        }

        // Compute cross-correlation
        let (offset_samples, correlation) = self.cross_correlate(audio_a, audio_b, config)?;

        // Convert sample offset to frame offset
        let sample_rate = self.sample_rates[angle_a];
        let samples_per_frame = f64::from(sample_rate) / config.frame_rate;
        let frame_offset = (offset_samples as f64 / samples_per_frame).floor() as i64;
        let sub_frame = (offset_samples as f64 / samples_per_frame) - frame_offset as f64;

        // Confidence based on correlation strength
        let confidence = correlation.abs().min(1.0).max(0.0);

        Ok(SyncOffset::new(
            angle_b,
            frame_offset,
            sub_frame,
            confidence,
        ))
    }

    /// Compute cross-correlation between two audio signals
    fn cross_correlate(
        &self,
        signal_a: &[f32],
        signal_b: &[f32],
        config: &SyncConfig,
    ) -> Result<(i64, f64)> {
        let sample_rate = config.sample_rate;
        let max_offset_samples =
            (f64::from(config.max_offset) * f64::from(sample_rate) / config.frame_rate) as usize;

        // Downsample for faster processing
        let downsample_factor = 4;
        let signal_a_ds = Self::downsample(signal_a, downsample_factor);
        let signal_b_ds = Self::downsample(signal_b, downsample_factor);

        // Search for best correlation
        let max_offset_ds = max_offset_samples / downsample_factor;
        let (best_offset_ds, _best_corr) =
            self.find_best_correlation(&signal_a_ds, &signal_b_ds, max_offset_ds);

        // Refine around best offset at full resolution
        let search_range = downsample_factor * 2;
        let center_offset = best_offset_ds * downsample_factor as i64;
        let (refined_offset, refined_corr) =
            self.refine_correlation(signal_a, signal_b, center_offset, search_range);

        Ok((refined_offset, refined_corr))
    }

    /// Downsample audio signal
    fn downsample(signal: &[f32], factor: usize) -> Vec<f32> {
        signal.iter().step_by(factor).copied().collect()
    }

    /// Find best correlation in search range
    fn find_best_correlation(
        &self,
        signal_a: &[f32],
        signal_b: &[f32],
        max_offset: usize,
    ) -> (i64, f64) {
        let mut best_offset = 0i64;
        let mut best_correlation = f64::NEG_INFINITY;

        let len = signal_a.len().min(signal_b.len());
        let window_size = len / 4; // Use 25% of signal for correlation

        for offset in -(max_offset as i64)..=(max_offset as i64) {
            let corr = self.compute_correlation(signal_a, signal_b, offset as isize, window_size);
            if corr > best_correlation {
                best_correlation = corr;
                best_offset = offset;
            }
        }

        (best_offset, best_correlation)
    }

    /// Refine correlation around a center offset
    fn refine_correlation(
        &self,
        signal_a: &[f32],
        signal_b: &[f32],
        center: i64,
        range: usize,
    ) -> (i64, f64) {
        let mut best_offset = center;
        let mut best_correlation = f64::NEG_INFINITY;

        let len = signal_a.len().min(signal_b.len());
        let window_size = len / 2;

        for offset in (center - range as i64)..=(center + range as i64) {
            let corr = self.compute_correlation(signal_a, signal_b, offset as isize, window_size);
            if corr > best_correlation {
                best_correlation = corr;
                best_offset = offset;
            }
        }

        (best_offset, best_correlation)
    }

    /// Compute normalized cross-correlation at specific offset
    fn compute_correlation(
        &self,
        signal_a: &[f32],
        signal_b: &[f32],
        offset: isize,
        window_size: usize,
    ) -> f64 {
        let len_a = signal_a.len();
        let len_b = signal_b.len();

        if len_a == 0 || len_b == 0 {
            return 0.0;
        }

        let start_a = if offset >= 0 { 0 } else { -offset as usize };
        let start_b = if offset >= 0 { offset as usize } else { 0 };

        let overlap = len_a
            .saturating_sub(start_a)
            .min(len_b.saturating_sub(start_b));
        if overlap == 0 {
            return 0.0;
        }

        let count = overlap.min(window_size);
        if count == 0 {
            return 0.0;
        }

        // Compute normalized cross-correlation
        let mut sum_ab = 0.0f64;
        let mut sum_aa = 0.0f64;
        let mut sum_bb = 0.0f64;

        for i in 0..count {
            let a = f64::from(signal_a[start_a + i]);
            let b = f64::from(signal_b[start_b + i]);
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
        }

        let denominator = (sum_aa * sum_bb).sqrt();
        if denominator > 0.0 {
            sum_ab / denominator
        } else {
            0.0
        }
    }

    /// Normalize audio for better correlation
    pub fn normalize_audio(&mut self, angle: AngleId) {
        if angle >= self.audio_tracks.len() {
            return;
        }

        let audio = &mut self.audio_tracks[angle];
        if audio.is_empty() {
            return;
        }

        // Find peak
        let peak = audio.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
        if peak > 0.0 {
            let scale = 1.0 / peak;
            for sample in audio {
                *sample *= scale;
            }
        }
    }

    /// Apply high-pass filter to remove DC offset
    pub fn high_pass_filter(&mut self, angle: AngleId, cutoff_hz: f32) {
        if angle >= self.audio_tracks.len() {
            return;
        }

        let sample_rate = self.sample_rates[angle];
        let audio = &mut self.audio_tracks[angle];

        if audio.is_empty() || sample_rate == 0 {
            return;
        }

        // Simple first-order high-pass filter
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate as f32;
        let alpha = rc / (rc + dt);

        let mut prev_input = 0.0;
        let mut prev_output = 0.0;

        for sample in audio {
            let input = *sample;
            let output = alpha * (prev_output + input - prev_input);
            *sample = output;
            prev_input = input;
            prev_output = output;
        }
    }

    /// Compute audio energy for activity detection
    #[must_use]
    pub fn compute_energy(&self, angle: AngleId, start_sample: usize, end_sample: usize) -> f32 {
        if angle >= self.audio_tracks.len() {
            return 0.0;
        }

        let audio = &self.audio_tracks[angle];
        let start = start_sample.min(audio.len());
        let end = end_sample.min(audio.len());

        if start >= end {
            return 0.0;
        }

        let sum: f32 = audio[start..end].iter().map(|&s| s * s).sum();
        sum / (end - start) as f32
    }
}

impl Synchronizer for AudioSync {
    fn synchronize(&self, config: &SyncConfig) -> Result<SyncResult> {
        if self.audio_tracks.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No audio tracks available".to_string(),
            ));
        }

        let mut offsets = Vec::new();
        let reference_angle = 0;

        // Calculate offsets relative to first angle
        for angle in 1..self.audio_tracks.len() {
            let offset = self.find_offset(reference_angle, angle, config)?;
            offsets.push(offset);
        }

        // Add zero offset for reference angle
        offsets.insert(0, SyncOffset::new(reference_angle, 0, 0.0, 1.0));

        // Calculate average confidence
        let confidence = offsets.iter().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64;

        Ok(SyncResult {
            reference_angle,
            offsets,
            confidence,
            method: SyncMethod::Audio,
        })
    }

    fn method(&self) -> SyncMethod {
        SyncMethod::Audio
    }

    fn is_reliable(&self) -> bool {
        !self.audio_tracks.is_empty() && self.audio_tracks.iter().all(|track| !track.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_sync_creation() {
        let sync = AudioSync::new(3);
        assert_eq!(sync.audio_tracks.len(), 3);
        assert_eq!(sync.sample_rates.len(), 3);
    }

    #[test]
    fn test_add_audio() {
        let mut sync = AudioSync::new(2);
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        sync.add_audio(0, samples.clone(), 48000);
        assert_eq!(sync.audio_tracks[0], samples);
        assert_eq!(sync.sample_rates[0], 48000);
    }

    #[test]
    fn test_downsample() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let downsampled = AudioSync::downsample(&signal, 2);
        assert_eq!(downsampled, vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_compute_energy() {
        let mut sync = AudioSync::new(1);
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        sync.add_audio(0, samples, 48000);
        let energy = sync.compute_energy(0, 0, 5);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_normalize_audio() {
        let mut sync = AudioSync::new(1);
        let samples = vec![0.0, 0.5, 2.0, 1.0, 0.0];
        sync.add_audio(0, samples, 48000);
        sync.normalize_audio(0);
        let peak = sync.audio_tracks[0]
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.01);
    }
}
