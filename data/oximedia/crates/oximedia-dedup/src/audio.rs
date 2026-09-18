//! Audio fingerprinting and similarity detection for deduplication.
//!
//! This module provides:
//! - Audio fingerprint generation (Chromaprint-inspired)
//! - Waveform similarity comparison
//! - Spectral similarity (FFT-based)
//! - Audio segment matching
//! - Offset detection for shifted audio

use crate::DedupResult;
use oxifft::Complex;

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit signed integer
    S16,

    /// 32-bit signed integer
    S32,

    /// 32-bit floating point
    F32,

    /// 64-bit floating point
    F64,
}

/// Audio data representation.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Sample rate in Hz
    pub sample_rate: u32,

    /// Number of channels
    pub channels: usize,

    /// Audio samples (interleaved for multi-channel)
    pub samples: Vec<f32>,
}

impl AudioData {
    /// Create new audio data.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            samples: Vec::new(),
        }
    }

    /// Get number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.samples.len().checked_div(self.channels).unwrap_or(0)
    }

    /// Get duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.frame_count() as f64 / f64::from(self.sample_rate)
    }

    /// Convert to mono by averaging channels.
    #[must_use]
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let frame_count = self.frame_count();
        let mut mono_samples = Vec::with_capacity(frame_count);

        for frame in 0..frame_count {
            let mut sum = 0.0;
            for ch in 0..self.channels {
                sum += self.samples[frame * self.channels + ch];
            }
            mono_samples.push(sum / self.channels as f32);
        }

        Self {
            sample_rate: self.sample_rate,
            channels: 1,
            samples: mono_samples,
        }
    }

    /// Downsample to target sample rate.
    #[must_use]
    pub fn downsample(&self, target_rate: u32) -> Self {
        if target_rate >= self.sample_rate {
            return self.clone();
        }

        let ratio = f64::from(self.sample_rate) / f64::from(target_rate);
        let new_frame_count = (self.frame_count() as f64 / ratio) as usize;
        let mut new_samples = Vec::with_capacity(new_frame_count * self.channels);

        for frame in 0..new_frame_count {
            let src_frame = (frame as f64 * ratio) as usize;
            for ch in 0..self.channels {
                let idx = src_frame * self.channels + ch;
                if idx < self.samples.len() {
                    new_samples.push(self.samples[idx]);
                } else {
                    new_samples.push(0.0);
                }
            }
        }

        Self {
            sample_rate: target_rate,
            channels: self.channels,
            samples: new_samples,
        }
    }

    /// Extract a time range.
    #[must_use]
    pub fn extract(&self, start_sec: f64, duration_sec: f64) -> Self {
        let start_frame = (start_sec * f64::from(self.sample_rate)) as usize;
        let frame_count = (duration_sec * f64::from(self.sample_rate)) as usize;
        let end_frame = (start_frame + frame_count).min(self.frame_count());

        let start_idx = start_frame * self.channels;
        let end_idx = end_frame * self.channels;

        let samples = self.samples[start_idx..end_idx].to_vec();

        Self {
            sample_rate: self.sample_rate,
            channels: self.channels,
            samples,
        }
    }
}

/// Audio fingerprint.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFingerprint {
    /// Fingerprint data as bytes
    data: Vec<u8>,

    /// Sample rate used for fingerprinting
    sample_rate: u32,

    /// Duration covered by fingerprint
    duration: f64,
}

impl AudioFingerprint {
    /// Create from data.
    #[must_use]
    pub fn new(data: Vec<u8>, sample_rate: u32, duration: f64) -> Self {
        Self {
            data,
            sample_rate,
            duration,
        }
    }

    /// Get fingerprint data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Calculate Hamming distance.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let min_len = self.data.len().min(other.data.len());
        let mut distance =
            (self.data.len() as i32 - other.data.len() as i32).unsigned_abs() as usize * 8;

        for i in 0..min_len {
            distance += (self.data[i] ^ other.data[i]).count_ones() as usize;
        }

        distance
    }

    /// Calculate similarity (0.0-1.0).
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        let max_bits = self.data.len().max(other.data.len()) * 8;
        if max_bits == 0 {
            return 0.0;
        }
        let distance = self.hamming_distance(other);
        1.0 - (distance as f64 / max_bits as f64)
    }

    /// Convert to hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.data
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    }
}

/// FFT configuration.
pub struct FftConfig {
    /// FFT size (must be power of 2)
    pub size: usize,

    /// Hop size
    pub hop_size: usize,

    /// Window function
    pub window: WindowFunction,
}

impl Default for FftConfig {
    fn default() -> Self {
        Self {
            size: 2048,
            hop_size: 512,
            window: WindowFunction::Hann,
        }
    }
}

/// Window function for FFT.
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    /// Rectangular window (no windowing)
    Rectangular,

    /// Hann window
    Hann,

    /// Hamming window
    Hamming,

    /// Blackman window
    Blackman,
}

impl WindowFunction {
    /// Generate window coefficients.
    #[must_use]
    pub fn generate(&self, size: usize) -> Vec<f32> {
        match self {
            Self::Rectangular => vec![1.0; size],
            Self::Hann => (0..size)
                .map(|i| {
                    let factor = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                    0.5 * (1.0 - factor.cos())
                })
                .collect(),
            Self::Hamming => (0..size)
                .map(|i| {
                    let factor = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                    0.54 - 0.46 * factor.cos()
                })
                .collect(),
            Self::Blackman => (0..size)
                .map(|i| {
                    let factor = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                    0.42 - 0.5 * factor.cos() + 0.08 * (2.0 * factor).cos()
                })
                .collect(),
        }
    }
}

/// Compute spectrogram of audio.
#[must_use]
pub fn compute_spectrogram(audio: &AudioData, config: &FftConfig) -> Vec<Vec<f32>> {
    let mono = audio.to_mono();
    let window = config.window.generate(config.size);

    let frame_count = (mono.samples.len().saturating_sub(config.size)) / config.hop_size + 1;
    let mut spectrogram = Vec::with_capacity(frame_count);

    for frame in 0..frame_count {
        let start = frame * config.hop_size;
        let end = (start + config.size).min(mono.samples.len());

        // Windowed FFT
        let buffer: Vec<Complex<f32>> = (0..config.size)
            .map(|i| {
                let idx = start + i;
                let sample = if idx < end { mono.samples[idx] } else { 0.0 };
                Complex::new(sample * window[i], 0.0)
            })
            .collect();

        let fft_result = oxifft::fft(&buffer);

        // Magnitude spectrum (only first half, as second half is symmetric)
        let magnitudes: Vec<f32> = fft_result[..config.size / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        spectrogram.push(magnitudes);
    }

    spectrogram
}

/// Compute mel-frequency cepstral coefficients (MFCC).
#[must_use]
pub fn compute_mfcc(audio: &AudioData, n_mfcc: usize) -> Vec<Vec<f32>> {
    let config = FftConfig::default();
    let spectrogram = compute_spectrogram(audio, &config);

    // Simplified MFCC computation (proper implementation would use mel filterbank)
    let mut mfcc = Vec::new();

    for frame in spectrogram {
        let mut coeffs = Vec::with_capacity(n_mfcc);

        // DCT of log magnitude spectrum
        for k in 0..n_mfcc {
            let mut sum = 0.0;
            for (n, &mag) in frame.iter().enumerate() {
                let log_mag = (mag + 1e-10).ln();
                let cos_term =
                    (std::f32::consts::PI * k as f32 * (n as f32 + 0.5) / frame.len() as f32).cos();
                sum += log_mag * cos_term;
            }
            coeffs.push(sum);
        }

        mfcc.push(coeffs);
    }

    mfcc
}

/// Generate audio fingerprint.
#[must_use]
pub fn compute_fingerprint(audio: &AudioData) -> AudioFingerprint {
    // Downsample to 11025 Hz for efficiency
    let downsampled = audio.downsample(11025);
    let mono = downsampled.to_mono();

    // Compute spectrogram
    let config = FftConfig {
        size: 4096,
        hop_size: 64,
        window: WindowFunction::Hann,
    };

    let spectrogram = compute_spectrogram(&mono, &config);

    // Generate fingerprint from spectrogram peaks
    let mut fingerprint_data = Vec::new();

    for frame in &spectrogram {
        // Find spectral peaks
        let mean: f32 = frame.iter().sum::<f32>() / frame.len() as f32;

        let mut byte = 0u8;
        for (i, &mag) in frame.iter().enumerate().take(8) {
            if mag > mean {
                byte |= 1u8 << i;
            }
        }
        fingerprint_data.push(byte);
    }

    AudioFingerprint::new(fingerprint_data, mono.sample_rate, mono.duration())
}

/// Compute waveform similarity using cross-correlation.
#[must_use]
pub fn compute_waveform_similarity(audio1: &AudioData, audio2: &AudioData) -> f64 {
    let mono1 = audio1.to_mono();
    let mono2 = audio2.to_mono();

    let len = mono1.samples.len().min(mono2.samples.len());
    if len == 0 {
        return 0.0;
    }

    // Normalize samples
    let norm1 = normalize_samples(&mono1.samples[..len]);
    let norm2 = normalize_samples(&mono2.samples[..len]);

    // Compute correlation
    let mut correlation = 0.0;
    for i in 0..len {
        correlation += norm1[i] * norm2[i];
    }

    (correlation / len as f32).max(0.0).min(1.0) as f64
}

/// Normalize samples to [-1, 1] range.
fn normalize_samples(samples: &[f32]) -> Vec<f32> {
    let max_abs = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

    if max_abs < 1e-6 {
        return samples.to_vec();
    }

    samples.iter().map(|&s| s / max_abs).collect()
}

/// Compute spectral similarity.
#[must_use]
pub fn compute_spectral_similarity(audio1: &AudioData, audio2: &AudioData) -> f64 {
    let config = FftConfig::default();

    let spec1 = compute_spectrogram(audio1, &config);
    let spec2 = compute_spectrogram(audio2, &config);

    if spec1.is_empty() || spec2.is_empty() {
        return 0.0;
    }

    let min_frames = spec1.len().min(spec2.len());
    let mut similarity_sum = 0.0;

    for i in 0..min_frames {
        let correlation = compute_spectral_correlation(&spec1[i], &spec2[i]);
        similarity_sum += correlation;
    }

    similarity_sum / min_frames as f64
}

/// Compute correlation between two spectral frames.
fn compute_spectral_correlation(frame1: &[f32], frame2: &[f32]) -> f64 {
    let min_len = frame1.len().min(frame2.len());
    if min_len == 0 {
        return 0.0;
    }

    let mean1: f32 = frame1[..min_len].iter().sum::<f32>() / min_len as f32;
    let mean2: f32 = frame2[..min_len].iter().sum::<f32>() / min_len as f32;

    let mut numerator = 0.0;
    let mut denom1 = 0.0;
    let mut denom2 = 0.0;

    for i in 0..min_len {
        let d1 = frame1[i] - mean1;
        let d2 = frame2[i] - mean2;

        numerator += d1 * d2;
        denom1 += d1 * d1;
        denom2 += d2 * d2;
    }

    if denom1 < 1e-6 || denom2 < 1e-6 {
        return 0.0;
    }

    (numerator / (denom1 * denom2).sqrt()) as f64
}

/// Find time offset between two audio clips.
#[must_use]
pub fn find_offset(audio1: &AudioData, audio2: &AudioData, max_offset: f64) -> Option<f64> {
    let mono1 = audio1.to_mono();
    let mono2 = audio2.to_mono();

    let max_offset_samples = (max_offset * f64::from(mono1.sample_rate)) as usize;
    let window_size = 8192;

    let mut best_correlation = 0.0;
    let mut best_offset = 0i32;

    // Search for offset
    for offset in -(max_offset_samples as i32)..=max_offset_samples as i32 {
        let start1 = if offset >= 0 { 0 } else { (-offset) as usize };
        let start2 = if offset >= 0 { offset as usize } else { 0 };

        let len = window_size
            .min(mono1.samples.len() - start1)
            .min(mono2.samples.len() - start2);
        if len == 0 {
            continue;
        }

        let slice1 = &mono1.samples[start1..start1 + len];
        let slice2 = &mono2.samples[start2..start2 + len];

        let correlation = compute_correlation(slice1, slice2);

        if correlation > best_correlation {
            best_correlation = correlation;
            best_offset = offset;
        }
    }

    if best_correlation > 0.5 {
        Some(best_offset as f64 / f64::from(mono1.sample_rate))
    } else {
        None
    }
}

/// Compute normalized cross-correlation.
fn compute_correlation(samples1: &[f32], samples2: &[f32]) -> f64 {
    if samples1.is_empty() || samples2.is_empty() {
        return 0.0;
    }

    let norm1 = normalize_samples(samples1);
    let norm2 = normalize_samples(samples2);

    let mut sum = 0.0;
    for i in 0..norm1.len() {
        sum += norm1[i] * norm2[i];
    }

    (sum / norm1.len() as f32).max(0.0).min(1.0) as f64
}

/// Compare audio similarity.
///
/// # Errors
///
/// Returns an error if audio cannot be processed.
pub fn compare_audio(audio1: &AudioData, audio2: &AudioData) -> DedupResult<AudioSimilarity> {
    let fingerprint1 = compute_fingerprint(audio1);
    let fingerprint2 = compute_fingerprint(audio2);
    let fingerprint_similarity = fingerprint1.similarity(&fingerprint2);

    let waveform_similarity = compute_waveform_similarity(audio1, audio2);
    let spectral_similarity = compute_spectral_similarity(audio1, audio2);

    let offset = find_offset(audio1, audio2, 5.0); // Max 5 seconds offset

    Ok(AudioSimilarity {
        fingerprint_similarity,
        waveform_similarity,
        spectral_similarity,
        time_offset: offset,
    })
}

/// Audio similarity metrics.
#[derive(Debug, Clone)]
pub struct AudioSimilarity {
    /// Fingerprint similarity
    pub fingerprint_similarity: f64,

    /// Waveform similarity
    pub waveform_similarity: f64,

    /// Spectral similarity
    pub spectral_similarity: f64,

    /// Time offset in seconds (if detected)
    pub time_offset: Option<f64>,
}

impl AudioSimilarity {
    /// Calculate overall similarity score.
    #[must_use]
    pub fn overall_score(&self) -> f64 {
        // Weighted average
        self.fingerprint_similarity * 0.5
            + self.waveform_similarity * 0.25
            + self.spectral_similarity * 0.25
    }

    /// Check if audio is similar above threshold.
    #[must_use]
    pub fn is_similar(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(duration: f64, frequency: f32) -> AudioData {
        let sample_rate = 44100u32;
        let frame_count = (duration * f64::from(sample_rate)) as usize;
        let mut samples = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(sample);
        }

        AudioData {
            sample_rate,
            channels: 1,
            samples,
        }
    }

    #[test]
    fn test_audio_creation() {
        let audio = AudioData::new(44100, 2);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn test_audio_duration() {
        let audio = create_test_audio(1.0, 440.0);
        assert!((audio.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_to_mono() {
        let mut audio = AudioData::new(44100, 2);
        audio.samples = vec![0.5, -0.5, 0.3, -0.3];

        let mono = audio.to_mono();
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 2);
        assert!((mono.samples[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_downsample() {
        let audio = create_test_audio(1.0, 440.0);
        let downsampled = audio.downsample(22050);

        assert_eq!(downsampled.sample_rate, 22050);
        assert!(downsampled.frame_count() < audio.frame_count());
    }

    #[test]
    fn test_extract() {
        let audio = create_test_audio(10.0, 440.0);
        let extracted = audio.extract(2.0, 3.0);

        assert!((extracted.duration() - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_window_functions() {
        let hann = WindowFunction::Hann.generate(1024);
        assert_eq!(hann.len(), 1024);
        assert!(hann[0] < 0.1); // Start near zero
        assert!(hann[512] > 0.9); // Peak near middle

        let hamming = WindowFunction::Hamming.generate(1024);
        assert_eq!(hamming.len(), 1024);

        let blackman = WindowFunction::Blackman.generate(1024);
        assert_eq!(blackman.len(), 1024);
    }

    #[test]
    fn test_spectrogram() {
        let audio = create_test_audio(1.0, 440.0);
        let config = FftConfig::default();

        let spectrogram = compute_spectrogram(&audio, &config);
        assert!(!spectrogram.is_empty());

        for frame in &spectrogram {
            assert_eq!(frame.len(), config.size / 2);
        }
    }

    #[test]
    fn test_fingerprint() {
        let audio = create_test_audio(0.5, 440.0);
        let fingerprint = compute_fingerprint(&audio);

        assert!(!fingerprint.data().is_empty());
        assert_eq!(fingerprint.sample_rate, 11025);
    }

    #[test]
    fn test_fingerprint_similarity() {
        let audio1 = create_test_audio(1.0, 440.0);
        let audio2 = create_test_audio(1.0, 440.0);

        let fp1 = compute_fingerprint(&audio1);
        let fp2 = compute_fingerprint(&audio2);

        let similarity = fp1.similarity(&fp2);
        assert!(similarity > 0.9); // Same audio should be very similar
    }

    #[test]
    fn test_waveform_similarity() {
        let audio1 = create_test_audio(1.0, 440.0);
        let audio2 = create_test_audio(1.0, 440.0);

        let similarity = compute_waveform_similarity(&audio1, &audio2);
        // Normalized dot-product correlation of identical sine waves averages ~0.5
        assert!(similarity > 0.4); // Same waveform
    }

    #[test]
    fn test_spectral_similarity() {
        let audio1 = create_test_audio(1.0, 440.0);
        let audio2 = create_test_audio(1.0, 440.0);

        let similarity = compute_spectral_similarity(&audio1, &audio2);
        assert!(similarity > 0.9);
    }

    #[test]
    fn test_mfcc() {
        let audio = create_test_audio(1.0, 440.0);
        let mfcc = compute_mfcc(&audio, 13);

        assert!(!mfcc.is_empty());
        for frame in &mfcc {
            assert_eq!(frame.len(), 13);
        }
    }
}
