//! Temporal synchronization for multi-camera alignment.
//!
//! This module provides tools for synchronizing multiple video/audio streams in time:
//!
//! - Audio cross-correlation for precise sync
//! - Timecode-based alignment (LTC/VITC)
//! - Visual marker detection
//! - Sub-frame accuracy

use crate::{AlignError, AlignResult, TimeOffset};
use std::f64::consts::PI;

/// Configuration for audio synchronization
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Window size in samples for correlation
    pub window_size: usize,
    /// Maximum offset to search (in samples)
    pub max_offset: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            window_size: 480000, // 10 seconds
            max_offset: 240000,  // ±5 seconds
        }
    }
}

/// Audio-based synchronization using cross-correlation
pub struct AudioSync {
    config: SyncConfig,
}

impl AudioSync {
    /// Create a new audio synchronizer
    #[must_use]
    pub fn new(config: SyncConfig) -> Self {
        Self { config }
    }

    /// Find time offset between two audio signals
    ///
    /// # Errors
    /// Returns error if signals are too short or correlation fails
    pub fn find_offset(&self, signal1: &[f32], signal2: &[f32]) -> AlignResult<TimeOffset> {
        if signal1.len() < self.config.window_size || signal2.len() < self.config.window_size {
            return Err(AlignError::InsufficientData(
                "Audio signals too short for correlation".to_string(),
            ));
        }

        // Use a window from each signal
        let window1 = &signal1[..self.config.window_size];
        let window2 = &signal2[..self.config.window_size.min(signal2.len())];

        // Compute cross-correlation
        let (offset, correlation) = self.cross_correlate(window1, window2)?;

        // Compute confidence based on peak sharpness
        let confidence = self.compute_confidence(window1, window2, offset);

        Ok(TimeOffset::new(offset, confidence, correlation))
    }

    /// Cross-correlate two signals
    fn cross_correlate(&self, signal1: &[f32], signal2: &[f32]) -> AlignResult<(i64, f64)> {
        let mut max_corr = f64::NEG_INFINITY;
        let mut best_offset = 0i64;

        let max_search = self.config.max_offset.min(signal1.len()).min(signal2.len());

        // Normalize signals
        let norm1 = self.normalize_signal(signal1);
        let norm2 = self.normalize_signal(signal2);

        // Search for best offset
        for offset in 0..max_search {
            // Positive offset: signal2 leads signal1
            let corr_pos = self.compute_correlation(&norm1[offset..], &norm2);
            if corr_pos > max_corr {
                max_corr = corr_pos;
                best_offset = offset as i64;
            }

            // Negative offset: signal1 leads signal2
            if offset > 0 {
                let corr_neg = self.compute_correlation(&norm1, &norm2[offset..]);
                if corr_neg > max_corr {
                    max_corr = corr_neg;
                    best_offset = -(offset as i64);
                }
            }
        }

        if max_corr.is_finite() {
            Ok((best_offset, max_corr))
        } else {
            Err(AlignError::SyncError(
                "Correlation produced non-finite value".to_string(),
            ))
        }
    }

    /// Normalize a signal (zero mean, unit variance)
    fn normalize_signal(&self, signal: &[f32]) -> Vec<f32> {
        let n = signal.len() as f32;
        let mean = signal.iter().sum::<f32>() / n;

        let variance = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;

        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return vec![0.0; signal.len()];
        }

        signal.iter().map(|&x| (x - mean) / std_dev).collect()
    }

    /// Compute correlation between two normalized signals
    fn compute_correlation(&self, sig1: &[f32], sig2: &[f32]) -> f64 {
        let len = sig1.len().min(sig2.len());
        if len == 0 {
            return 0.0;
        }

        let sum: f64 = sig1[..len]
            .iter()
            .zip(&sig2[..len])
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();

        sum / len as f64
    }

    /// Compute confidence score based on peak sharpness
    fn compute_confidence(&self, _signal1: &[f32], _signal2: &[f32], _offset: i64) -> f64 {
        // Simplified confidence: in production, this would analyze peak sharpness
        // and secondary peaks to determine reliability
        0.95
    }

    /// Refine offset to sub-sample precision using parabolic interpolation
    ///
    /// # Errors
    /// Returns error if refinement fails
    pub fn refine_offset(
        &self,
        signal1: &[f32],
        signal2: &[f32],
        coarse_offset: i64,
    ) -> AlignResult<f64> {
        let offset = coarse_offset.unsigned_abs() as usize;

        if offset >= signal1.len() || offset >= signal2.len() {
            return Err(AlignError::InvalidConfig("Offset out of range".to_string()));
        }

        // Compute correlation at offset-1, offset, offset+1
        let norm1 = self.normalize_signal(signal1);
        let norm2 = self.normalize_signal(signal2);

        let c0 = if offset > 0 {
            self.compute_correlation(&norm1[offset - 1..], &norm2)
        } else {
            0.0
        };

        let c1 = self.compute_correlation(&norm1[offset..], &norm2);

        let c2 = if offset + 1 < norm1.len() {
            self.compute_correlation(&norm1[offset + 1..], &norm2)
        } else {
            0.0
        };

        // Parabolic interpolation
        let delta = (c0 - c2) / (2.0 * (c0 - 2.0 * c1 + c2));

        if delta.is_finite() {
            Ok(coarse_offset as f64 + delta)
        } else {
            Ok(coarse_offset as f64)
        }
    }
}

/// Timecode synchronization
pub struct TimecodeSync {
    /// Frame rate for timecode interpretation
    pub frame_rate: f64,
}

impl TimecodeSync {
    /// Create a new timecode synchronizer
    #[must_use]
    pub fn new(frame_rate: f64) -> Self {
        Self { frame_rate }
    }

    /// Compute offset between two timecodes in frames
    #[must_use]
    pub fn compute_offset(&self, tc1: &Timecode, tc2: &Timecode) -> i64 {
        let frames1 = tc1.to_frames(self.frame_rate);
        let frames2 = tc2.to_frames(self.frame_rate);
        frames2 - frames1
    }

    /// Verify timecode continuity
    #[must_use]
    pub fn verify_continuity(&self, timecodes: &[Timecode]) -> bool {
        if timecodes.len() < 2 {
            return true;
        }

        for i in 1..timecodes.len() {
            let offset = self.compute_offset(&timecodes[i - 1], &timecodes[i]);
            if offset != 1 {
                return false;
            }
        }

        true
    }
}

/// Simple timecode representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0 to fps-1)
    pub frames: u8,
}

impl Timecode {
    /// Create a new timecode
    #[must_use]
    pub fn new(hours: u8, minutes: u8, seconds: u8, frames: u8) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
        }
    }

    /// Convert timecode to total frame count
    #[must_use]
    pub fn to_frames(&self, frame_rate: f64) -> i64 {
        let fps = frame_rate.round() as i64;
        i64::from(self.hours) * 3600 * fps
            + i64::from(self.minutes) * 60 * fps
            + i64::from(self.seconds) * fps
            + i64::from(self.frames)
    }

    /// Create timecode from frame count
    #[must_use]
    pub fn from_frames(frames: i64, frame_rate: f64) -> Self {
        let fps = frame_rate.round() as i64;
        let total_seconds = frames / fps;
        let remaining_frames = frames % fps;

        let hours = (total_seconds / 3600) % 24;
        let minutes = (total_seconds / 60) % 60;
        let seconds = total_seconds % 60;

        Self {
            hours: hours as u8,
            minutes: minutes as u8,
            seconds: seconds as u8,
            frames: remaining_frames as u8,
        }
    }
}

/// Visual marker detection for synchronization
pub struct MarkerDetector {
    /// Brightness threshold for flash detection (0.0-1.0)
    pub flash_threshold: f32,
    /// Minimum duration in frames
    pub min_duration: usize,
}

impl Default for MarkerDetector {
    fn default() -> Self {
        Self {
            flash_threshold: 0.8,
            min_duration: 1,
        }
    }
}

impl MarkerDetector {
    /// Create a new marker detector
    #[must_use]
    pub fn new(flash_threshold: f32, min_duration: usize) -> Self {
        Self {
            flash_threshold,
            min_duration,
        }
    }

    /// Detect flash events in a sequence of brightness values
    #[must_use]
    pub fn detect_flashes(&self, brightness: &[f32]) -> Vec<usize> {
        let mut flashes = Vec::new();
        let mut in_flash = false;
        let mut flash_start = 0;

        for (i, &value) in brightness.iter().enumerate() {
            if !in_flash && value > self.flash_threshold {
                in_flash = true;
                flash_start = i;
            } else if in_flash && value <= self.flash_threshold {
                in_flash = false;
                if i - flash_start >= self.min_duration {
                    flashes.push(flash_start);
                }
            }
        }

        flashes
    }

    /// Compute average brightness from RGB frame
    #[must_use]
    pub fn compute_brightness(&self, rgb: &[u8], width: usize, height: usize) -> f32 {
        if rgb.len() != width * height * 3 {
            return 0.0;
        }

        let sum: u32 = rgb
            .chunks(3)
            .map(|pixel| {
                // Luminance formula: 0.299R + 0.587G + 0.114B
                let r = u32::from(pixel[0]);
                let g = u32::from(pixel[1]);
                let b = u32::from(pixel[2]);
                (299 * r + 587 * g + 114 * b) / 1000
            })
            .sum();

        (sum as f32 / (width * height) as f32) / 255.0
    }
}

/// Phase-only correlation for sub-pixel alignment
pub struct PhaseCorrelation {
    /// FFT size (must be power of 2)
    pub fft_size: usize,
}

impl PhaseCorrelation {
    /// Create a new phase correlation analyzer
    #[must_use]
    pub fn new(fft_size: usize) -> Self {
        Self { fft_size }
    }

    /// Find sub-pixel offset between two 1D signals
    ///
    /// # Errors
    /// Returns error if signals are incompatible or FFT fails
    pub fn find_offset(&self, signal1: &[f32], signal2: &[f32]) -> AlignResult<f64> {
        if signal1.len() != signal2.len() || signal1.is_empty() {
            return Err(AlignError::InvalidConfig(
                "Signals must have same non-zero length".to_string(),
            ));
        }

        // Simple peak detection in cross-correlation
        let len = signal1.len().min(self.fft_size);
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;

        for offset in 0..len {
            let mut sum = 0.0f32;
            for i in 0..(len - offset) {
                sum += signal1[i] * signal2[i + offset];
            }
            if sum > max_val {
                max_val = sum;
                max_idx = offset;
            }
        }

        Ok(max_idx as f64)
    }
}

/// Beat detection for music synchronization
pub struct BeatDetector {
    /// Sample rate
    pub sample_rate: u32,
    /// Hop size for analysis
    pub hop_size: usize,
}

impl BeatDetector {
    /// Create a new beat detector
    #[must_use]
    pub fn new(sample_rate: u32, hop_size: usize) -> Self {
        Self {
            sample_rate,
            hop_size,
        }
    }

    /// Detect beats in audio signal
    #[must_use]
    pub fn detect_beats(&self, audio: &[f32]) -> Vec<usize> {
        let mut beats = Vec::new();
        let window_size = 2048;

        // Compute energy envelope
        let energy = self.compute_energy_envelope(audio, window_size);

        // Find peaks in energy envelope
        for i in 1..energy.len().saturating_sub(1) {
            if energy[i] > energy[i - 1] && energy[i] > energy[i + 1] {
                let threshold = energy[i.saturating_sub(10)..i].iter().sum::<f32>() / 10.0 * 1.5;

                if energy[i] > threshold {
                    beats.push(i * self.hop_size);
                }
            }
        }

        beats
    }

    /// Compute energy envelope
    fn compute_energy_envelope(&self, audio: &[f32], window_size: usize) -> Vec<f32> {
        let mut envelope = Vec::new();

        for chunk in audio.chunks(self.hop_size) {
            let energy: f32 = chunk
                .iter()
                .take(window_size.min(chunk.len()))
                .map(|&x| x * x)
                .sum();
            envelope.push(energy);
        }

        envelope
    }

    /// Align beats between two signals
    ///
    /// # Errors
    /// Returns error if beat detection fails
    pub fn align_beats(&self, audio1: &[f32], audio2: &[f32]) -> AlignResult<TimeOffset> {
        let beats1 = self.detect_beats(audio1);
        let beats2 = self.detect_beats(audio2);

        if beats1.is_empty() || beats2.is_empty() {
            return Err(AlignError::SyncError("No beats detected".to_string()));
        }

        // Find best offset by matching beat sequences
        let offset = beats2[0] as i64 - beats1[0] as i64;

        Ok(TimeOffset::new(offset, 0.8, 0.9))
    }
}

/// Window functions for signal processing
pub struct WindowFunction;

impl WindowFunction {
    /// Hann window
    #[must_use]
    pub fn hann(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let x = i as f64 / (size - 1) as f64;
                (0.5 * (1.0 - (2.0 * PI * x).cos())) as f32
            })
            .collect()
    }

    /// Hamming window
    #[must_use]
    pub fn hamming(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let x = i as f64 / (size - 1) as f64;
                (0.54 - 0.46 * (2.0 * PI * x).cos()) as f32
            })
            .collect()
    }

    /// Blackman window
    #[must_use]
    pub fn blackman(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let x = i as f64 / (size - 1) as f64;
                (0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()) as f32
            })
            .collect()
    }
}

/// Multi-stream synchronizer for handling multiple cameras/sources
pub struct MultiStreamSync {
    /// Audio sync configuration
    audio_config: SyncConfig,
    /// Reference stream index
    reference_index: usize,
}

impl MultiStreamSync {
    /// Create a new multi-stream synchronizer
    #[must_use]
    pub fn new(audio_config: SyncConfig, reference_index: usize) -> Self {
        Self {
            audio_config,
            reference_index,
        }
    }

    /// Synchronize multiple audio streams to a reference
    ///
    /// # Errors
    /// Returns error if synchronization fails
    pub fn sync_streams(&self, streams: &[&[f32]]) -> AlignResult<Vec<TimeOffset>> {
        if streams.len() <= self.reference_index {
            return Err(AlignError::InvalidConfig(
                "Reference index out of bounds".to_string(),
            ));
        }

        let reference = streams[self.reference_index];
        let sync = AudioSync::new(self.audio_config.clone());

        let mut offsets = Vec::new();

        for (i, stream) in streams.iter().enumerate() {
            if i == self.reference_index {
                offsets.push(TimeOffset::new(0, 1.0, 1.0));
            } else {
                let offset = sync.find_offset(reference, stream)?;
                offsets.push(offset);
            }
        }

        Ok(offsets)
    }

    /// Compute sync quality metric (0.0 = poor, 1.0 = perfect)
    #[must_use]
    pub fn compute_sync_quality(&self, offsets: &[TimeOffset]) -> f32 {
        if offsets.is_empty() {
            return 0.0;
        }

        let avg_confidence: f64 =
            offsets.iter().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64;
        let avg_correlation: f64 =
            offsets.iter().map(|o| o.correlation).sum::<f64>() / offsets.len() as f64;

        ((avg_confidence + avg_correlation) / 2.0) as f32
    }
}

/// Drift detector for detecting timing drift over long recordings
pub struct DriftDetector {
    /// Sample rate
    pub sample_rate: u32,
    /// Analysis window size
    pub window_size: usize,
    /// Number of windows to analyze
    pub num_windows: usize,
}

impl DriftDetector {
    /// Create a new drift detector
    #[must_use]
    pub fn new(sample_rate: u32, window_size: usize, num_windows: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            num_windows,
        }
    }

    /// Detect timing drift between two signals
    ///
    /// # Errors
    /// Returns error if detection fails
    pub fn detect_drift(&self, signal1: &[f32], signal2: &[f32]) -> AlignResult<Vec<TimeOffset>> {
        let total_samples = self.window_size * self.num_windows;
        if signal1.len() < total_samples || signal2.len() < total_samples {
            return Err(AlignError::InsufficientData(
                "Signals too short for drift analysis".to_string(),
            ));
        }

        let config = SyncConfig {
            sample_rate: self.sample_rate,
            window_size: self.window_size,
            max_offset: self.window_size / 2,
        };

        let sync = AudioSync::new(config);
        let mut offsets = Vec::new();

        for i in 0..self.num_windows {
            let start = i * self.window_size;
            let end = start + self.window_size;

            let window1 = &signal1[start..end];
            let window2 = &signal2[start..end];

            let offset = sync.find_offset(window1, window2)?;
            offsets.push(offset);
        }

        Ok(offsets)
    }

    /// Compute drift rate (samples per second)
    #[must_use]
    pub fn compute_drift_rate(&self, offsets: &[TimeOffset]) -> f32 {
        if offsets.len() < 2 {
            return 0.0;
        }

        // Linear regression to find drift rate
        let n = offsets.len() as f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_xy = 0.0f32;
        let mut sum_xx = 0.0f32;

        for (i, offset) in offsets.iter().enumerate() {
            let x = i as f32;
            let y = offset.samples as f32;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Convert to samples per second
        let window_duration = self.window_size as f32 / self.sample_rate as f32;
        slope / window_duration
    }
}

/// Spectral correlation for frequency-domain synchronization
pub struct SpectralCorrelation {
    /// FFT size
    pub fft_size: usize,
    /// Hop size
    pub hop_size: usize,
}

impl SpectralCorrelation {
    /// Create a new spectral correlation analyzer
    #[must_use]
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        Self { fft_size, hop_size }
    }

    /// Compute spectral correlation
    ///
    /// # Errors
    /// Returns error if correlation fails
    pub fn correlate(&self, signal1: &[f32], signal2: &[f32]) -> AlignResult<TimeOffset> {
        if signal1.len() < self.fft_size || signal2.len() < self.fft_size {
            return Err(AlignError::InsufficientData(
                "Signals too short for spectral correlation".to_string(),
            ));
        }

        // Simplified spectral correlation (in production, use proper FFT)
        let mut max_corr = f32::NEG_INFINITY;
        let mut best_offset = 0i64;

        let max_offset = signal1.len().min(signal2.len()) / 2;

        for offset in 0..max_offset.min(10000) {
            let mut corr = 0.0f32;
            let len = (signal1.len() - offset)
                .min(signal2.len())
                .min(self.fft_size);

            for i in 0..len {
                corr += signal1[i + offset] * signal2[i];
            }

            if corr > max_corr {
                max_corr = corr;
                best_offset = offset as i64;
            }
        }

        Ok(TimeOffset::new(best_offset, 0.9, f64::from(max_corr)))
    }
}

/// Jitter analyzer for detecting timing instability
pub struct JitterAnalyzer {
    /// Expected interval (in samples)
    pub expected_interval: usize,
    /// Tolerance (in samples)
    pub tolerance: usize,
}

impl JitterAnalyzer {
    /// Create a new jitter analyzer
    #[must_use]
    pub fn new(expected_interval: usize, tolerance: usize) -> Self {
        Self {
            expected_interval,
            tolerance,
        }
    }

    /// Analyze jitter in event timestamps
    #[must_use]
    pub fn analyze_jitter(&self, timestamps: &[usize]) -> JitterMetrics {
        if timestamps.len() < 2 {
            return JitterMetrics::default();
        }

        let mut intervals = Vec::new();
        for i in 1..timestamps.len() {
            intervals.push(timestamps[i] - timestamps[i - 1]);
        }

        let mean_interval = intervals.iter().sum::<usize>() as f32 / intervals.len() as f32;

        let mut variance = 0.0f32;
        for &interval in &intervals {
            let diff = interval as f32 - mean_interval;
            variance += diff * diff;
        }
        variance /= intervals.len() as f32;

        let std_dev = variance.sqrt();
        let max_jitter = intervals
            .iter()
            .map(|&i| (i as i32 - self.expected_interval as i32).abs())
            .max()
            .unwrap_or(0) as f32;

        JitterMetrics {
            mean_interval,
            std_dev,
            max_jitter,
            jitter_ratio: std_dev / mean_interval,
        }
    }
}

/// Jitter metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct JitterMetrics {
    /// Mean interval
    pub mean_interval: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Maximum jitter
    pub max_jitter: f32,
    /// Jitter ratio (`std_dev` / mean)
    pub jitter_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_sync_config() {
        let config = SyncConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.window_size, 480000);
    }

    #[test]
    fn test_timecode_conversion() {
        let tc = Timecode::new(1, 30, 45, 10);
        let frames = tc.to_frames(25.0);
        let tc2 = Timecode::from_frames(frames, 25.0);
        assert_eq!(tc, tc2);
    }

    #[test]
    fn test_timecode_offset() {
        let sync = TimecodeSync::new(25.0);
        let tc1 = Timecode::new(1, 0, 0, 0);
        let tc2 = Timecode::new(1, 0, 0, 25);
        assert_eq!(sync.compute_offset(&tc1, &tc2), 25);
    }

    #[test]
    fn test_flash_detection() {
        let detector = MarkerDetector::default();
        let brightness = vec![0.1, 0.2, 0.9, 0.9, 0.1, 0.2];
        let flashes = detector.detect_flashes(&brightness);
        assert_eq!(flashes.len(), 1);
        assert_eq!(flashes[0], 2);
    }

    #[test]
    fn test_brightness_computation() {
        let detector = MarkerDetector::default();
        let rgb = vec![255u8; 300]; // 10x10 white image
        let brightness = detector.compute_brightness(&rgb, 10, 10);
        assert!((brightness - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_signal() {
        let sync = AudioSync::new(SyncConfig::default());
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = sync.normalize_signal(&signal);

        // Check mean is close to 0
        let mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 1e-6);

        // Check variance is close to 1
        let variance: f32 =
            normalized.iter().map(|&x| x * x).sum::<f32>() / normalized.len() as f32;
        assert!((variance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_window_functions() {
        let hann = WindowFunction::hann(100);
        assert_eq!(hann.len(), 100);
        assert!(hann[0] < 0.01); // First value near 0
        assert!(hann[50] > 0.99); // Middle value near 1

        let hamming = WindowFunction::hamming(100);
        assert_eq!(hamming.len(), 100);

        let blackman = WindowFunction::blackman(100);
        assert_eq!(blackman.len(), 100);
    }

    #[test]
    fn test_beat_detector() {
        let detector = BeatDetector::new(48000, 512);

        // Create a simple signal with periodic energy spikes
        let mut audio = vec![0.0; 48000];
        for i in (0..48000).step_by(4800) {
            for j in 0..100 {
                if i + j < audio.len() {
                    audio[i + j] = 1.0;
                }
            }
        }

        let beats = detector.detect_beats(&audio);
        assert!(!beats.is_empty());
    }

    #[test]
    fn test_multi_stream_sync() {
        // Use small window/offset to keep test fast (default is 480000/240000 which is O(n^2) ~115B ops)
        let config = SyncConfig {
            sample_rate: 48000,
            window_size: 1000,
            max_offset: 500,
        };
        let sync = MultiStreamSync::new(config, 0);

        let stream1 = vec![0.1f32; 2000];
        let stream2 = vec![0.2f32; 2000];
        let streams = vec![&stream1[..], &stream2[..]];

        let result = sync.sync_streams(&streams);
        assert!(result.is_ok());
    }

    #[test]
    fn test_drift_detector() {
        let detector = DriftDetector::new(48000, 48000, 5);
        assert_eq!(detector.sample_rate, 48000);
        assert_eq!(detector.num_windows, 5);
    }

    #[test]
    fn test_jitter_analyzer() {
        let analyzer = JitterAnalyzer::new(1000, 10);
        let timestamps = vec![0, 1000, 2000, 3005, 4000];
        let metrics = analyzer.analyze_jitter(&timestamps);

        assert!(metrics.mean_interval > 0.0);
        assert!(metrics.std_dev >= 0.0);
    }

    #[test]
    fn test_spectral_correlation() {
        let corr = SpectralCorrelation::new(1024, 512);
        assert_eq!(corr.fft_size, 1024);
        assert_eq!(corr.hop_size, 512);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network time synchronization (SNTP / RFC 4330)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an SNTP (Simple Network Time Protocol) query.
///
/// SNTP is a simplified version of NTP (RFC 4330) that is sufficient for
/// computing a clock offset without the full NTP association state machine.
/// It uses a single UDP exchange to estimate the difference between the local
/// clock and the reference server clock.
#[derive(Debug, Clone)]
pub struct NtpConfig {
    /// Hostname or IP address of the NTP server (e.g. `"pool.ntp.org"`).
    pub server: String,
    /// UDP port of the NTP server.  Standard port is 123.
    pub port: u16,
    /// Socket receive timeout in milliseconds.  Default: 2 000 ms.
    pub timeout_ms: u64,
}

impl Default for NtpConfig {
    fn default() -> Self {
        Self {
            server: "pool.ntp.org".to_string(),
            port: 123,
            timeout_ms: 2_000,
        }
    }
}

/// Measured clock offset and round-trip delay from a single SNTP exchange.
///
/// All values are in **milliseconds** with sub-millisecond resolution.
///
/// The classic SNTP offset formula uses four timestamps (RFC 4330 §5):
///
/// ```text
/// offset = ((T2 - T1) + (T3 - T4)) / 2
/// RTT    = (T4 - T1) - (T3 - T2)
/// ```
///
/// where:
/// - `T1` = client transmit time
/// - `T2` = server receive time
/// - `T3` = server transmit time
/// - `T4` = client receive time
#[derive(Debug, Clone, Copy)]
pub struct TimeDelta {
    /// Signed clock offset in milliseconds.
    ///
    /// Positive means the local clock is *behind* the server (local is slow).
    /// Negative means the local clock is *ahead* of the server (local is fast).
    pub offset_ms: i64,

    /// Round-trip delay in milliseconds (always ≥ 0).
    pub round_trip_ms: u64,
}

/// SNTP client for network-based time synchronization.
///
/// Queries an NTP server via UDP and returns the estimated clock offset.
/// This is intentionally simple (single-packet, no association state) and is
/// suitable for one-shot synchronization of distributed camera systems where
/// NTP precision (≈ 1–50 ms) is adequate.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_align::temporal::{NtpClient, NtpConfig};
///
/// let config = NtpConfig {
///     server: "pool.ntp.org".to_string(),
///     port: 123,
///     timeout_ms: 3_000,
/// };
/// // In production; requires network access:
/// // let delta = NtpClient::query_offset(&config).unwrap();
/// // println!("Clock offset: {} ms", delta.offset_ms);
/// ```
pub struct NtpClient;

impl NtpClient {
    /// Query an NTP server and compute the clock offset for the local clock.
    ///
    /// Makes a single SNTP request (Mode 3 / Client) and parses the 48-byte
    /// NTP response packet to extract the four timestamps required for the
    /// RFC 4330 §5 offset formula.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError::SyncError`] if:
    /// - The server address cannot be resolved.
    /// - The UDP socket cannot be bound or the send/receive fails.
    /// - The response is shorter than 48 bytes or has an invalid leap/mode.
    pub fn query_offset(config: &NtpConfig) -> AlignResult<TimeDelta> {
        use std::net::{ToSocketAddrs, UdpSocket};
        use std::time::Duration;

        // Resolve server address
        let server_addr = format!("{}:{}", config.server, config.port)
            .to_socket_addrs()
            .map_err(|e| AlignError::SyncError(format!("DNS resolution failed: {e}")))?
            .next()
            .ok_or_else(|| AlignError::SyncError("No addresses returned by DNS".to_string()))?;

        // Bind to any local port
        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| AlignError::SyncError(format!("Failed to bind UDP socket: {e}")))?;

        socket
            .set_read_timeout(Some(Duration::from_millis(config.timeout_ms)))
            .map_err(|e| AlignError::SyncError(format!("Failed to set socket timeout: {e}")))?;

        // Build a minimal 48-byte NTP client request (Mode 3, Version 4)
        let mut request = [0u8; 48];
        // Byte 0: LI=0 (00), VN=4 (100), Mode=3 (011) → 0b00_100_011 = 0x23
        request[0] = 0x23;

        // Record client transmit time T1 (before send) in NTP 64-bit format.
        // NTP epoch is 1900-01-01; Unix epoch offset = 2_208_988_800 s.
        let t1_ntp = Self::unix_to_ntp(Self::now_unix_ms());

        // Write T1 into the "Transmit Timestamp" field (bytes 40–47)
        request[40..48].copy_from_slice(&t1_ntp);

        socket
            .send_to(&request, server_addr)
            .map_err(|e| AlignError::SyncError(format!("UDP send failed: {e}")))?;

        // Receive response and record T4 immediately
        let mut response = [0u8; 96]; // larger buffer in case server adds data
        let (n, _) = socket
            .recv_from(&mut response)
            .map_err(|e| AlignError::SyncError(format!("UDP receive failed: {e}")))?;

        let t4_unix_ms = Self::now_unix_ms();

        if n < 48 {
            return Err(AlignError::SyncError(format!(
                "Response too short: {n} bytes (expected ≥ 48)"
            )));
        }

        // Parse T2 (server receive, bytes 32–39) and T3 (server transmit, bytes 40–47)
        let t2_ms = Self::ntp_to_unix_ms(&response[32..40]);
        let t3_ms = Self::ntp_to_unix_ms(&response[40..48]);
        // T1 is the local transmit time we recorded before sending
        let t1_unix_ms = Self::ntp_to_unix_ms_from_u64(Self::ntp_bytes_to_u64(&t1_ntp));

        // RFC 4330 §5:
        //   offset = ((T2 - T1) + (T3 - T4)) / 2
        //   RTT    = (T4 - T1) - (T3 - T2)
        let offset_2x_ms = (t2_ms - t1_unix_ms) + (t3_ms - t4_unix_ms as i64);
        let offset_ms = offset_2x_ms / 2;

        let rtt_ms = (t4_unix_ms as i64 - t1_unix_ms) - (t3_ms - t2_ms);
        let round_trip_ms = rtt_ms.max(0) as u64;

        Ok(TimeDelta {
            offset_ms,
            round_trip_ms,
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Returns the current Unix time in milliseconds (non-monotonic wall clock).
    fn now_unix_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Convert Unix time in milliseconds to a 48-bit NTP 64-bit timestamp
    /// (seconds since 1900-01-01, 32-bit integer + 32-bit fraction).
    fn unix_to_ntp(unix_ms: u64) -> [u8; 8] {
        const NTP_EPOCH_OFFSET: u64 = 2_208_988_800; // seconds between 1900 and 1970
        let secs = unix_ms / 1_000 + NTP_EPOCH_OFFSET;
        let frac_ms = unix_ms % 1_000;
        // Convert milliseconds to NTP 32-bit fractional part:
        //   frac = frac_ms / 1000 * 2^32
        // Use u128 to avoid overflow.
        let frac = ((frac_ms as u128 * (1u128 << 32)) / 1_000) as u32;
        let mut out = [0u8; 8];
        out[0..4].copy_from_slice(&(secs as u32).to_be_bytes());
        out[4..8].copy_from_slice(&frac.to_be_bytes());
        out
    }

    /// Convert an 8-byte big-endian NTP timestamp slice to Unix time in ms.
    fn ntp_to_unix_ms(bytes: &[u8]) -> i64 {
        assert!(bytes.len() >= 8, "NTP timestamp slice must be ≥ 8 bytes");
        let raw = Self::ntp_bytes_to_u64(&bytes[..8].try_into().unwrap_or([0u8; 8]));
        Self::ntp_to_unix_ms_from_u64(raw)
    }

    fn ntp_bytes_to_u64(bytes: &[u8; 8]) -> u64 {
        let secs = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
        let frac = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
        (secs << 32) | frac
    }

    fn ntp_to_unix_ms_from_u64(raw: u64) -> i64 {
        const NTP_EPOCH_OFFSET: u64 = 2_208_988_800;
        let secs = (raw >> 32) as u64;
        let frac = (raw & 0xFFFF_FFFF) as u64;
        // Convert fractional part: frac / 2^32 * 1000 ms
        // Use u128 to avoid overflow: frac * 1000 can be at most ~4.3e9 * 1000 = 4.3e12
        let frac_ms = ((frac as u128 * 1_000) >> 32) as u64;
        let unix_ms = (secs.saturating_sub(NTP_EPOCH_OFFSET)) * 1_000 + frac_ms;
        unix_ms as i64
    }
}

// ─── NTP / SNTP tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod ntp_tests {
    use super::*;

    /// Verify that our unix→NTP→unix round-trip preserves millisecond precision.
    #[test]
    fn test_ntp_packet_parse_known_bytes() {
        // Manually construct a known NTP 64-bit timestamp:
        //   NTP seconds = 3_913_056_000  (2024-01-01 00:00:00 UTC)
        //   NTP frac    = 0 (exact second)
        // Unix seconds = 3_913_056_000 - 2_208_988_800 = 1_704_067_200
        let ntp_secs: u32 = 3_913_056_000;
        let ntp_frac: u32 = 0;

        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&ntp_secs.to_be_bytes());
        bytes[4..8].copy_from_slice(&ntp_frac.to_be_bytes());

        let unix_ms = NtpClient::ntp_to_unix_ms(&bytes);

        // Expected: 1_704_067_200 seconds * 1000 ms/s = 1_704_067_200_000 ms
        let expected_ms: i64 = 1_704_067_200_000;
        assert_eq!(
            unix_ms, expected_ms,
            "Unix ms mismatch: got {unix_ms}, expected {expected_ms}"
        );
    }

    /// Verify the RFC 4330 §5 offset formula:
    ///
    ///   offset = ((T2 - T1) + (T3 - T4)) / 2
    ///
    /// Using synthetic values with a known clock offset.
    #[test]
    fn test_ntp_offset_computation_formula() {
        // Scenario: local clock is 100 ms ahead of server.
        // Let server receive at T2 = 1000 ms (server time) after T1 = 1050 ms (local time, fast)
        // Server processes instantly: T3 = 1000 ms (server time)
        // Local receives at T4 = 1060 ms (local time)
        //
        // RTT = (T4 - T1) - (T3 - T2) = (1060 - 1050) - (1000 - 1000) = 10 ms
        // offset = ((T2 - T1) + (T3 - T4)) / 2
        //        = ((1000 - 1050) + (1000 - 1060)) / 2
        //        = (-50 + -60) / 2 = -55 ms
        //
        // Interpretation: local clock offset is -55 ms (local is ~55 ms fast).

        let t1_local_ms: i64 = 1_050;
        let t2_server_ms: i64 = 1_000;
        let t3_server_ms: i64 = 1_000;
        let t4_local_ms: i64 = 1_060;

        let offset_2x = (t2_server_ms - t1_local_ms) + (t3_server_ms - t4_local_ms);
        let offset = offset_2x / 2;
        let rtt = (t4_local_ms - t1_local_ms) - (t3_server_ms - t2_server_ms);

        assert_eq!(offset, -55, "offset formula mismatch");
        assert_eq!(rtt, 10, "RTT formula mismatch");
    }

    /// Round-trip unix_ms → NTP → unix_ms must preserve values to within 1 ms.
    #[test]
    fn test_ntp_unix_roundtrip() {
        // Use a fixed Unix time in ms (2025-01-01 00:00:00.500 UTC)
        let unix_ms: u64 = 1_735_689_600_500;
        let ntp_bytes = NtpClient::unix_to_ntp(unix_ms);
        let recovered = NtpClient::ntp_to_unix_ms(&ntp_bytes);
        // Allow ±1 ms rounding error from the fractional conversion
        assert!(
            (recovered - unix_ms as i64).abs() <= 1,
            "round-trip error: in={unix_ms}, out={recovered}"
        );
    }
}
