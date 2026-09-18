//! Type definitions for audio restoration.
//!
//! Originally split from `restoration.rs` via SplitRS.

#![allow(dead_code)]

use crate::error::{AudioPostError, AudioPostResult};
use oxifft::Complex;

use super::functions::levinson_durbin;

/// Configuration for click/pop removal via AR-LPC interpolation.
#[derive(Debug, Clone)]
pub struct ArLpcDeclickConfig {
    /// Window length in ms for short-time energy detection (default 1.0).
    pub window_ms: f32,
    /// Detection threshold as σ-multiplier (default 3.0 σ).
    pub threshold_sigma: f32,
    /// AR model order for Levinson-Durbin interpolation (default 32).
    pub ar_order: usize,
    /// If true, the interpolated span preserves the sign of the peak it replaces.
    pub preserve_polarity: bool,
    /// Sample rate of the audio being processed.
    pub sample_rate: u32,
}
impl Default for ArLpcDeclickConfig {
    fn default() -> Self {
        Self {
            window_ms: 1.0,
            threshold_sigma: 3.0,
            ar_order: 32,
            preserve_polarity: false,
            sample_rate: 48_000,
        }
    }
}
/// Hum remover for removing 50/60 Hz and harmonics
#[derive(Debug)]
pub struct HumRemover {
    sample_rate: u32,
    fundamental_freq: f32,
    num_harmonics: usize,
}
impl HumRemover {
    /// Create a new hum remover
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, fundamental_freq: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if fundamental_freq != 50.0 && fundamental_freq != 60.0 {
            return Err(AudioPostError::InvalidFrequency(fundamental_freq));
        }
        Ok(Self {
            sample_rate,
            fundamental_freq,
            num_harmonics: 10,
        })
    }
    /// Set number of harmonics to remove
    pub fn set_num_harmonics(&mut self, num_harmonics: usize) {
        self.num_harmonics = num_harmonics.clamp(1, 20);
    }
    /// Get harmonic frequencies
    #[must_use]
    pub fn get_harmonic_frequencies(&self) -> Vec<f32> {
        (1..=self.num_harmonics)
            .map(|i| self.fundamental_freq * i as f32)
            .collect()
    }
}
/// Configuration for the stateful `Denoiser` (Boll 1979 + Wiener post-filter).
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// STFT frame size in samples (must be a power of two; default 1024).
    pub fft_size: usize,
    /// Overlap fraction in (0, 1) — default 0.5 for 50% overlap.
    pub overlap_fraction: f32,
    /// Number of initial frames used for bootstrap noise-floor estimation (default 10).
    pub noise_estimation_frames: usize,
    /// Over-subtraction α for SNR-adaptive spectral subtraction (default 2.0).
    pub oversubtraction_alpha: f32,
    /// Spectral floor β (default 0.002).
    pub spectral_floor_beta: f32,
    /// Sample rate of the audio being processed.
    pub sample_rate: u32,
}
impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            overlap_fraction: 0.5,
            noise_estimation_frames: 10,
            oversubtraction_alpha: 2.0,
            spectral_floor_beta: 0.002,
            sample_rate: 48_000,
        }
    }
}
/// Remove impulsive noise (clicks, pops) from mono f32 audio using AR-LPC
/// (autoregressive linear predictive coding) interpolation.
///
/// Algorithm:
/// 1. Compute short-time energy in a sliding window of `window_ms * sample_rate / 1000` samples.
/// 2. Estimate the running background energy via exponential smoothing (decay = 0.98).
/// 3. Flag any window centre where `energy > background * threshold_sigma² * window_len`.
/// 4. Extend each flagged region ±2 samples and cap at 10 ms.
/// 5. Extract up to 256 clean samples before and after the click as context.
/// 6. Compute order-`ar_order` AR coefficients via Levinson-Durbin from the context.
/// 7. Forward-predict across the corrupted span; clamp output to [−1, 1].
#[derive(Debug)]
pub struct Declicker {
    config: ArLpcDeclickConfig,
}
impl Declicker {
    /// Create a new `Declicker` with the provided configuration.
    ///
    /// # Errors
    ///
    /// Returns `AudioPostError::InvalidSampleRate` for a zero sample rate, or
    /// `AudioPostError::InvalidBufferSize` if `ar_order` is zero.
    pub fn new(config: ArLpcDeclickConfig) -> Result<Self, AudioPostError> {
        if config.sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(config.sample_rate));
        }
        if config.ar_order == 0 {
            return Err(AudioPostError::InvalidBufferSize(0));
        }
        Ok(Self { config })
    }
    /// Process a mono f32 sample buffer in-place, removing click artefacts.
    ///
    /// # Errors
    ///
    /// Returns `AudioPostError::InvalidBufferSize` if the buffer is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&self, samples: &mut [f32]) -> Result<(), AudioPostError> {
        if samples.is_empty() {
            return Err(AudioPostError::InvalidBufferSize(0));
        }
        let n = samples.len();
        let sr = self.config.sample_rate as f32;
        let win_len = ((self.config.window_ms * sr / 1000.0).round() as usize).max(4);
        let half = win_len / 2;
        let mut energy = vec![0.0_f32; n];
        let init_end = win_len.min(n);
        let mut acc: f32 = samples[..init_end].iter().map(|&x| x * x).sum();
        for e in energy[..half.min(n)].iter_mut() {
            *e = acc;
        }
        for i in 1..n {
            if i > half + 1 {
                let out_idx = i.saturating_sub(half + 1);
                acc -= samples[out_idx] * samples[out_idx];
            }
            let in_idx = i + half;
            if in_idx < n {
                acc += samples[in_idx] * samples[in_idx];
            }
            energy[i] = acc;
        }
        const DECAY: f32 = 0.98;
        let mut background = vec![0.0_f32; n];
        background[0] = energy[0];
        for i in 1..n {
            background[i] = DECAY * background[i - 1] + (1.0 - DECAY) * energy[i];
        }
        let sigma2 = self.config.threshold_sigma * self.config.threshold_sigma;
        let win_len_f = win_len as f32;
        let mut flagged = vec![false; n];
        for i in 0..n {
            let bg = background[i].max(1e-15);
            if energy[i] > bg * sigma2 * win_len_f {
                flagged[i] = true;
            }
        }
        let max_click_samples = ((sr * 0.010).round() as usize).max(4);
        let mut regions: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < n {
            if flagged[i] {
                let start = i;
                while i < n && flagged[i] {
                    i += 1;
                }
                let end = i;
                let region_start = start.saturating_sub(2);
                let region_end = (end + 2).min(n);
                let capped_end = region_end.min(region_start + max_click_samples);
                regions.push((region_start, capped_end));
            } else {
                i += 1;
            }
        }
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (s, e) in regions {
            if let Some(last) = merged.last_mut() {
                if s <= last.1 {
                    last.1 = last.1.max(e);
                    continue;
                }
            }
            merged.push((s, e));
        }
        const CONTEXT_LEN: usize = 256;
        let order = self.config.ar_order;
        let original = samples.to_vec();
        for (region_start, region_end) in merged {
            if region_start >= region_end {
                continue;
            }
            let pre_start = region_start.saturating_sub(CONTEXT_LEN);
            let pre_context: Vec<f32> = original[pre_start..region_start].to_vec();
            let post_end = (region_end + CONTEXT_LEN).min(n);
            let post_context: Vec<f32> = original[region_end..post_end].to_vec();
            let mut context = pre_context.clone();
            context.extend_from_slice(&post_context);
            if context.len() < order + 1 {
                let span = (region_end - region_start).max(1);
                let v0 = if region_start > 0 {
                    original[region_start - 1]
                } else {
                    0.0
                };
                let v1 = if region_end < n {
                    original[region_end]
                } else {
                    0.0
                };
                for j in region_start..region_end {
                    let t = (j - region_start + 1) as f32 / (span + 1) as f32;
                    samples[j] = (v0 * (1.0 - t) + v1 * t).clamp(-1.0, 1.0);
                }
                continue;
            }
            let ar_coeffs = match levinson_durbin(&context, order) {
                Some(c) => c,
                None => {
                    let span = (region_end - region_start).max(1);
                    let v0 = if region_start > 0 {
                        original[region_start - 1]
                    } else {
                        0.0
                    };
                    let v1 = if region_end < n {
                        original[region_end]
                    } else {
                        0.0
                    };
                    for j in region_start..region_end {
                        let t = (j - region_start + 1) as f32 / (span + 1) as f32;
                        samples[j] = (v0 * (1.0 - t) + v1 * t).clamp(-1.0, 1.0);
                    }
                    continue;
                }
            };
            let seed_end = region_start;
            let seed_start = seed_end.saturating_sub(order);
            let mut history: Vec<f32> = original[seed_start..seed_end].to_vec();
            while history.len() < order {
                history.insert(0, 0.0);
            }
            let click_peak: Option<f32> = if self.config.preserve_polarity {
                original[region_start..region_end]
                    .iter()
                    .copied()
                    .reduce(|a, b| if b.abs() > a.abs() { b } else { a })
            } else {
                None
            };
            let mut predicted: Vec<f32> = Vec::with_capacity(region_end - region_start);
            for _ in region_start..region_end {
                let pred: f32 = ar_coeffs
                    .iter()
                    .zip(history.iter().rev())
                    .map(|(&a, &x)| -a * x)
                    .sum();
                let pred_clamped = pred.clamp(-1.0, 1.0);
                predicted.push(pred_clamped);
                history.push(pred_clamped);
            }
            if let Some(peak) = click_peak {
                if let Some(pred_peak) =
                    predicted
                        .iter()
                        .copied()
                        .reduce(|a, b| if b.abs() > a.abs() { b } else { a })
                {
                    if peak * pred_peak < 0.0 {
                        for s in &mut predicted {
                            *s = -*s;
                        }
                    }
                }
            }
            for (j, &p) in (region_start..region_end).zip(predicted.iter()) {
                samples[j] = p;
            }
        }
        Ok(())
    }
}
/// Spectral repair tool
#[derive(Debug)]
pub struct SpectralRepair {
    sample_rate: u32,
    pub(super) fft_size: usize,
}
impl SpectralRepair {
    /// Create a new spectral repair tool
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or FFT size is invalid
    pub fn new(sample_rate: u32, fft_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !fft_size.is_power_of_two() {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        Ok(Self {
            sample_rate,
            fft_size,
        })
    }
    /// Repair a frequency range using interpolation
    pub fn repair_frequency_range(
        &self,
        _input: &[f32],
        _output: &mut [f32],
        _freq_start: f32,
        _freq_end: f32,
    ) {
    }
}
/// Configuration for Boll (1979) spectral subtraction + Wiener post-filter.
#[derive(Debug, Clone)]
pub struct SpectralSubtractionConfig {
    /// STFT frame size (must be a power of two, default 1024).
    pub fft_size: usize,
    /// STFT hop size (default 512 → 50 % overlap).
    pub hop_size: usize,
    /// Fraction of the quietest frames used to estimate the noise PSD
    /// (default 0.05 → quietest 5 %).
    pub noise_percentile: f32,
    /// Over-subtraction factor α (default 2.0).
    pub alpha: f32,
    /// Spectral floor factor β (default 0.05).
    pub beta: f32,
}
impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 512,
            noise_percentile: 0.05,
            alpha: 2.0,
            beta: 0.05,
        }
    }
}
/// Phase correction tool
#[derive(Debug)]
pub struct PhaseCorrector {
    sample_rate: u32,
}
impl PhaseCorrector {
    /// Create a new phase corrector
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self { sample_rate })
    }
    /// Analyze phase correlation between stereo channels
    #[must_use]
    pub fn analyze_phase_correlation(&self, left: &[f32], right: &[f32]) -> f32 {
        if left.len() != right.len() || left.is_empty() {
            return 0.0;
        }
        let mut correlation = 0.0;
        for (l, r) in left.iter().zip(right.iter()) {
            correlation += l * r;
        }
        correlation / left.len() as f32
    }
    /// Correct phase issues
    pub fn correct_phase(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
    }
}
/// Click remover
#[derive(Debug)]
pub struct ClickRemover {
    sample_rate: u32,
    sensitivity: f32,
}
impl ClickRemover {
    /// Create a new click remover
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            sample_rate,
            sensitivity: 0.5,
        })
    }
    /// Set sensitivity (0.0 to 1.0)
    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        self.sensitivity = sensitivity.clamp(0.0, 1.0);
    }
    /// Detect clicks in audio
    #[must_use]
    pub fn detect_clicks(&self, audio: &[f32]) -> Vec<usize> {
        let mut clicks = Vec::new();
        let threshold = self.sensitivity * 2.0;
        for i in 1..audio.len() - 1 {
            let diff = (audio[i] - audio[i - 1]).abs();
            if diff > threshold {
                clicks.push(i);
            }
        }
        clicks
    }
    /// Remove clicks from audio
    pub fn process(&self, input: &[f32], output: &mut [f32]) {
        let clicks = self.detect_clicks(input);
        output.copy_from_slice(input);
        for &click_pos in &clicks {
            if click_pos > 0 && click_pos < output.len() - 1 {
                output[click_pos] = (output[click_pos - 1] + output[click_pos + 1]) / 2.0;
            }
        }
    }
}
/// Hiss remover
#[derive(Debug)]
pub struct HissRemover {
    sample_rate: u32,
    threshold: f32,
    pub(super) reduction: f32,
}
impl HissRemover {
    /// Create a new hiss remover
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            sample_rate,
            threshold: -40.0,
            reduction: 0.8,
        })
    }
    /// Set threshold in dB
    ///
    /// # Errors
    ///
    /// Returns an error if threshold is invalid
    pub fn set_threshold(&mut self, threshold_db: f32) -> AudioPostResult<()> {
        if threshold_db > 0.0 {
            return Err(AudioPostError::InvalidThreshold(threshold_db));
        }
        self.threshold = threshold_db;
        Ok(())
    }
    /// Set reduction amount (0.0 to 1.0)
    pub fn set_reduction(&mut self, reduction: f32) {
        self.reduction = reduction.clamp(0.0, 1.0);
    }
}
/// Vinyl click and pop detector/remover for analogue-sourced audio.
///
/// The algorithm operates in two stages:
///
/// 1. **Detection** – Computes a local derivative (first difference) of the
///    signal and flags samples where the absolute derivative exceeds a
///    threshold derived from the local short-term variance of the signal.
///    Consecutive flagged samples are merged into click *regions*.
///
/// 2. **Interpolation** – Each click region is replaced by a cubic Hermite
///    spline interpolated from the samples immediately before and after the
///    region, preserving continuity of value and gradient across the boundary.
#[derive(Debug)]
pub struct VinylClickRemover {
    /// Sample rate.
    pub sample_rate: u32,
    /// Sensitivity multiplier.  A smaller value is more sensitive (detects more
    /// clicks); a larger value is less sensitive.  Default: 6.0.
    pub sensitivity: f32,
    /// Maximum region length (samples) that will be treated as a click.
    /// Longer artefacts are treated as pops and may be repaired with a broader
    /// interpolation window.  Default: 64.
    pub max_click_samples: usize,
    /// Minimum gap (samples) between two click regions before they are merged.
    /// Default: 4.
    pub merge_gap: usize,
    /// Whether to use adaptive variance for threshold computation (slower but
    /// more accurate on material with large dynamic swings).  Default: true.
    pub adaptive_threshold: bool,
    /// Local variance estimation window (samples).  Default: 256.
    pub variance_window: usize,
}
impl VinylClickRemover {
    /// Create a new vinyl click remover with default parameters.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] for a zero sample rate.
    pub fn new(sample_rate: u32) -> crate::error::AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            sample_rate,
            sensitivity: 6.0,
            max_click_samples: 64,
            merge_gap: 4,
            adaptive_threshold: true,
            variance_window: 256,
        })
    }
    /// Detect click/pop regions in `audio`.
    ///
    /// Returns a `Vec<(start, end)>` of sample index ranges (exclusive end).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect_clicks(&self, audio: &[f32]) -> Vec<(usize, usize)> {
        let n = audio.len();
        if n < 4 {
            return vec![];
        }
        let mut diff: Vec<f32> = vec![0.0; n];
        for i in 1..n {
            diff[i] = audio[i] - audio[i - 1];
        }
        let variance: Vec<f32> = if self.adaptive_threshold {
            let half = self.variance_window / 2;
            let mut var_vec = vec![0.0f32; n];
            for center in 0..n {
                let start = center.saturating_sub(half);
                let end = (center + half).min(n);
                let len = end - start;
                if len == 0 {
                    continue;
                }
                let mean: f32 = diff[start..end].iter().sum::<f32>() / len as f32;
                let variance_val: f32 = diff[start..end]
                    .iter()
                    .map(|&x| {
                        let d = x - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / len as f32;
                var_vec[center] = variance_val;
            }
            var_vec
        } else {
            let mean: f32 = diff.iter().sum::<f32>() / n as f32;
            let global_var = diff.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            vec![global_var; n]
        };
        let mut flagged = vec![false; n];
        for i in 0..n {
            let threshold = self.sensitivity * variance[i].sqrt().max(1e-9);
            if diff[i].abs() > threshold {
                flagged[i] = true;
            }
        }
        let mut regions: Vec<(usize, usize)> = Vec::new();
        let mut in_region = false;
        let mut region_start = 0;
        for i in 0..n {
            if flagged[i] {
                if !in_region {
                    region_start = i;
                    in_region = true;
                }
            } else if in_region {
                regions.push((region_start, i));
                in_region = false;
            }
        }
        if in_region {
            regions.push((region_start, n));
        }
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (start, end) in regions {
            if let Some(last) = merged.last_mut() {
                if start <= last.1 + self.merge_gap {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            merged.push((start, end));
        }
        merged
            .into_iter()
            .filter(|(s, e)| e - s <= self.max_click_samples)
            .collect()
    }
    /// Remove clicks from `input` and write the result to `output` using
    /// cubic Hermite spline interpolation across each detected click region.
    ///
    /// `output` must be the same length as `input`.
    pub fn process(&self, input: &[f32], output: &mut [f32]) {
        let n = input.len();
        output[..n].copy_from_slice(&input[..n]);
        let regions = self.detect_clicks(input);
        for (start, end) in regions {
            let x0 = start.saturating_sub(1);
            let x3 = end.min(n - 1);
            if x0 >= x3 || x3 >= n {
                continue;
            }
            let p0 = output[x0];
            let p1 = if x3 > 0 { output[x3] } else { p0 };
            let m0 = if x0 > 0 && x0 + 1 < n {
                (output[x0 + 1] - output[x0.saturating_sub(1)]) * 0.5
            } else {
                0.0
            };
            let m1 = if x3 > 0 && x3 + 1 < n {
                (output[x3 + 1] - output[x3 - 1]) * 0.5
            } else {
                0.0
            };
            let region_len = (x3 - x0).max(1) as f32;
            for i in (x0 + 1)..x3 {
                let t = (i - x0) as f32 / region_len;
                let t2 = t * t;
                let t3 = t2 * t;
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                output[i] = h00 * p0 + h10 * m0 * region_len + h01 * p1 + h11 * m1 * region_len;
            }
        }
    }
    /// Process audio in-place.
    pub fn process_inplace(&self, audio: &mut [f32]) {
        let input = audio.to_vec();
        self.process(&input, audio);
    }
}
/// Stereo enhancement
#[derive(Debug)]
pub struct StereoEnhancer {
    sample_rate: u32,
    pub(super) width: f32,
}
impl StereoEnhancer {
    /// Create a new stereo enhancer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            sample_rate,
            width: 1.0,
        })
    }
    /// Set stereo width (0.0 = mono, 1.0 = normal, >1.0 = enhanced)
    pub fn set_width(&mut self, width: f32) {
        self.width = width.max(0.0);
    }
    /// Process stereo audio
    pub fn process(
        &self,
        left: &[f32],
        right: &[f32],
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let len = left
            .len()
            .min(right.len())
            .min(out_left.len())
            .min(out_right.len());
        for (_i, ((l, r), (ol, or))) in left
            .iter()
            .zip(right.iter())
            .zip(out_left.iter_mut().zip(out_right.iter_mut()))
            .enumerate()
            .take(len)
        {
            let mid = (l + r) / 2.0;
            let side = (l - r) / 2.0;
            *ol = mid + side * self.width;
            *or = mid - side * self.width;
        }
    }
}
/// Declipping/decrackle processor
#[derive(Debug)]
pub struct Declipper {
    sample_rate: u32,
    threshold: f32,
}
impl Declipper {
    /// Create a new declipper
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            sample_rate,
            threshold: 0.95,
        })
    }
    /// Set clipping threshold (0.0 to 1.0)
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
    /// Detect clipped regions
    #[must_use]
    pub fn detect_clipping(&self, audio: &[f32]) -> Vec<(usize, usize)> {
        let mut regions = Vec::new();
        let mut in_clip = false;
        let mut start = 0;
        for (i, &sample) in audio.iter().enumerate() {
            if sample.abs() >= self.threshold {
                if !in_clip {
                    start = i;
                    in_clip = true;
                }
            } else if in_clip {
                regions.push((start, i));
                in_clip = false;
            }
        }
        if in_clip {
            regions.push((start, audio.len()));
        }
        regions
    }
    /// Process audio to repair clipping
    pub fn process(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);
        let clipped_regions = self.detect_clipping(input);
        for (start, end) in clipped_regions {
            if start > 0 && end < output.len() {
                let start_val = output[start.saturating_sub(1)];
                let end_val = output[end.min(output.len() - 1)];
                let range = end - start;
                for (i, sample) in output.iter_mut().enumerate().take(end).skip(start) {
                    let t = (i - start) as f32 / range as f32;
                    *sample = start_val * (1.0 - t) + end_val * t;
                }
            }
        }
    }
}
/// Stateful noise reducer using Boll (1979) spectral subtraction with a Wiener
/// post-filter.
///
/// Each call to [`Denoiser::process`] performs a two-pass STFT reduction on the
/// provided buffer: Pass 1 collects per-frame PSDs and derives the noise floor
/// from the quietest `noise_estimation_frames` frames (or min-statistics when an
/// existing floor has been seeded by a prior call); Pass 2 applies SNR-adaptive
/// spectral subtraction + Wiener gain with overlap-add.
///
/// The noise floor estimate is carried over between calls so subsequent buffers
/// benefit from any previously seen noise reference.  Call [`Denoiser::reset`] to
/// start fresh.
#[derive(Debug)]
pub struct Denoiser {
    config: DenoiseConfig,
    /// Estimated noise power per FFT bin.
    pub(super) noise_floor: Vec<f32>,
    /// True once the noise floor has been bootstrapped at least once.
    pub(super) floor_ready: bool,
    /// Precomputed Hann window coefficients.
    hann: Vec<f32>,
    /// Number of samples to advance per frame.
    hop: usize,
}
impl Denoiser {
    /// Create a new `Denoiser`.
    ///
    /// # Errors
    ///
    /// Returns `AudioPostError::InvalidSampleRate` for zero sample rate, or
    /// `AudioPostError::InvalidBufferSize` if `fft_size` is not a power of two
    /// or `overlap_fraction` is outside (0, 1).
    pub fn new(config: DenoiseConfig) -> Result<Self, AudioPostError> {
        if config.sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(config.sample_rate));
        }
        if !config.fft_size.is_power_of_two() || config.fft_size < 4 {
            return Err(AudioPostError::InvalidBufferSize(config.fft_size));
        }
        if config.overlap_fraction <= 0.0 || config.overlap_fraction >= 1.0 {
            return Err(AudioPostError::InvalidBufferSize(0));
        }
        let fft_size = config.fft_size;
        let hop = ((fft_size as f32 * (1.0 - config.overlap_fraction)).round() as usize).max(1);
        let hann: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size - 1) as f32).cos())
            })
            .collect();
        Ok(Self {
            config,
            noise_floor: vec![0.0_f32; fft_size],
            floor_ready: false,
            hann,
            hop,
        })
    }
    /// Process a mono f32 sample buffer and return the noise-reduced output
    /// (same length as input).
    ///
    /// Uses a two-pass STFT approach: pass 1 estimates the noise floor from the
    /// quietest frames in this buffer (merging with any floor from previous calls
    /// via min-statistics); pass 2 applies spectral subtraction + Wiener gain.
    ///
    /// # Errors
    ///
    /// Returns `AudioPostError::InvalidBufferSize` if the input is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, AudioPostError> {
        if samples.is_empty() {
            return Err(AudioPostError::InvalidBufferSize(0));
        }
        let n = samples.len();
        let fft_size = self.config.fft_size;
        let hop = self.hop;
        let num_frames = (n + hop - 1) / hop;
        let mut frame_psds: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            let count = fft_size.min(n.saturating_sub(start));
            let frame_mean: f32 = if count > 0 {
                samples[start..start + count].iter().sum::<f32>() / count as f32
            } else {
                0.0
            };
            let input_cx: Vec<Complex<f32>> = (0..fft_size)
                .map(|j| {
                    let s = if start + j < n {
                        samples[start + j]
                    } else {
                        0.0
                    };
                    Complex::new((s - frame_mean) * self.hann[j], 0.0)
                })
                .collect();
            let spectrum = oxifft::fft(&input_cx);
            let psd: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();
            frame_psds.push(psd);
        }
        let mut bin_temporal_mean = vec![0.0_f32; fft_size];
        for psd in &frame_psds {
            for (mean, &p) in bin_temporal_mean.iter_mut().zip(psd.iter()) {
                *mean += p;
            }
        }
        let inv_frames = 1.0 / num_frames as f32;
        for mean in &mut bin_temporal_mean {
            *mean *= inv_frames;
        }
        let mut sorted_means = bin_temporal_mean.clone();
        sorted_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let flat_noise_level = sorted_means[fft_size / 2];
        let new_floor = vec![flat_noise_level.max(1e-30); fft_size];
        if self.floor_ready {
            const DECAY: f32 = 0.9;
            for (nf, &new_p) in self.noise_floor.iter_mut().zip(new_floor.iter()) {
                *nf = DECAY * *nf + (1.0 - DECAY) * new_p;
            }
        } else {
            self.noise_floor.copy_from_slice(&new_floor);
            self.floor_ready = true;
        }
        let mut output = vec![0.0_f32; n + fft_size];
        let mut ola_weights = vec![0.0_f32; n + fft_size];
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            let frame_mean: f32 = {
                let count = fft_size.min(n - start.min(n));
                if count == 0 {
                    0.0
                } else {
                    samples[start..start + count].iter().sum::<f32>() / count as f32
                }
            };
            let input_cx: Vec<Complex<f32>> = (0..fft_size)
                .map(|j| {
                    let s = if start + j < n {
                        samples[start + j]
                    } else {
                        0.0
                    };
                    Complex::new((s - frame_mean) * self.hann[j], 0.0)
                })
                .collect();
            let mut spectrum = oxifft::fft(&input_cx);
            for (bin_idx, spec_bin) in spectrum.iter_mut().enumerate() {
                if bin_idx == 0 || bin_idx == fft_size / 2 {
                    *spec_bin = Complex::new(0.0, 0.0);
                    continue;
                }
                let signal_psd = spec_bin.norm_sqr();
                let noise_p = self.noise_floor[bin_idx].max(1e-30);
                let snr_k = signal_psd / noise_p;
                let alpha = (4.0 - 3.0 * snr_k).clamp(1.0, self.config.oversubtraction_alpha);
                let signal_mag = spec_bin.norm();
                let noise_mag = noise_p.sqrt();
                let enhanced_mag = (signal_mag - alpha * noise_mag)
                    .max(self.config.spectral_floor_beta * signal_mag);
                let s2 = enhanced_mag * enhanced_mag;
                let gain = (s2 / (s2 + noise_p)).clamp(0.0, 1.0);
                *spec_bin = *spec_bin * gain;
            }
            let recovered = oxifft::ifft(&spectrum);
            for j in 0..fft_size {
                let out_idx = start + j;
                if out_idx < output.len() {
                    output[out_idx] += recovered[j].re * self.hann[j];
                    ola_weights[out_idx] += self.hann[j] * self.hann[j];
                }
            }
        }
        for i in 0..n {
            let w = ola_weights[i];
            if w >= 0.01 {
                output[i] /= w;
            } else {
                output[i] = samples[i];
            }
        }
        output.truncate(n);
        Ok(output)
    }
    /// Reset internal state (noise floor and frame counter).
    pub fn reset(&mut self) {
        self.noise_floor.fill(0.0);
        self.floor_ready = false;
    }
}
/// Spectral noise reducer
#[derive(Debug)]
pub struct SpectralNoiseReducer {
    sample_rate: u32,
    fft_size: usize,
    pub(super) noise_profile: Vec<f32>,
    pub(super) reduction_amount: f32,
}
impl SpectralNoiseReducer {
    /// Create a new spectral noise reducer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or FFT size is invalid
    pub fn new(sample_rate: u32, fft_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !fft_size.is_power_of_two() {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        Ok(Self {
            sample_rate,
            fft_size,
            noise_profile: vec![0.0; fft_size / 2 + 1],
            reduction_amount: 0.8,
        })
    }
    /// Capture noise profile from a noise-only section
    pub fn capture_noise_profile(&mut self, noise_samples: &[f32]) {
        if noise_samples.len() < self.fft_size {
            return;
        }
        let input: Vec<Complex<f32>> = noise_samples
            .iter()
            .take(self.fft_size)
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let buffer = oxifft::fft(&input);
        for (i, profile_val) in self.noise_profile.iter_mut().enumerate() {
            if i < buffer.len() {
                *profile_val = buffer[i].norm();
            }
        }
    }
    /// Set reduction amount (0.0 to 1.0)
    pub fn set_reduction_amount(&mut self, amount: f32) {
        self.reduction_amount = amount.clamp(0.0, 1.0);
    }
    /// Process audio to reduce noise
    pub fn process(&self, _input: &[f32], output: &mut [f32]) {
        for (out, &inp) in output.iter_mut().zip(_input.iter()) {
            *out = inp;
        }
    }
}
/// Configuration for the MAD-based declick algorithm.
#[derive(Debug, Clone)]
pub struct DeclickConfig {
    /// Threshold multiplier over the Median Absolute Deviation (default 8.0).
    pub mad_threshold: f32,
    /// Number of surrounding samples used for cubic-spline interpolation (default 50).
    pub interpolation_radius: usize,
}
impl Default for DeclickConfig {
    fn default() -> Self {
        Self {
            mad_threshold: 8.0,
            interpolation_radius: 50,
        }
    }
}
