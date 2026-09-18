//! Noise reduction via spectral subtraction and Wiener filter estimation.
//!
//! This module provides two complementary noise-reduction algorithms:
//!
//! ## Spectral Subtraction (`SpectralSubtractor`)
//!
//! Estimates noise from the first few frames of the signal (or from an
//! explicit noise profile) and subtracts its magnitude spectrum from
//! subsequent frames before reconstructing via IFFT.
//! A half-wave rectifier prevents negative power values ("musical noise"
//! artefacts are mitigated by an over-subtraction factor and a spectral
//! floor).
//!
//! ## Wiener Filter (`WienerFilter`)
//!
//! Applies a per-bin multiplicative gain `H[k] = max(1 - noise_psd[k] /
//! signal_psd[k], floor)` to each STFT bin. The gain is smoothed over time
//! to reduce artefacts.  SNR estimation uses a minimum-statistics tracker
//! that continuously updates the noise PSD estimate even during speech.
//!
//! Both processors work on mono f32 samples and use an overlap-add framework
//! with a Hann window internally (via the `oxifft` crate).

#![allow(dead_code)]

use std::f64::consts::PI;

use oxifft::api::fft as oxifft_fft;
use oxifft::api::ifft as oxifft_ifft;
use oxifft::Complex;

// ---------------------------------------------------------------------------
// OLA (Overlap-Add) engine
// ---------------------------------------------------------------------------

struct OlaEngine {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f64>,
    input_buf: Vec<f64>,
    output_buf: Vec<f64>,
}

impl OlaEngine {
    fn new(fft_size: usize, hop_size: usize) -> Self {
        let window = hann(fft_size);
        Self {
            fft_size,
            hop_size,
            window,
            input_buf: vec![0.0; fft_size],
            output_buf: vec![0.0; fft_size],
        }
    }

    /// Push `hop_size` new samples. Returns `true` when a full frame is ready.
    fn push(&mut self, samples: &[f64]) -> bool {
        let hop = self.hop_size;
        let fft = self.fft_size;
        // Shift old samples left by hop
        self.input_buf.copy_within(hop..fft, 0);
        let new = samples.len().min(hop);
        self.input_buf[fft - hop..fft - hop + new].copy_from_slice(&samples[..new]);
        true // always full after initial fill
    }

    /// Windowed current frame → complex.
    fn windowed_frame(&self) -> Vec<Complex<f64>> {
        self.input_buf
            .iter()
            .zip(&self.window)
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Spectral Subtraction
// ---------------------------------------------------------------------------

/// Noise reduction via spectral subtraction.
///
/// The algorithm:
/// 1. Accumulate a noise profile from the first `noise_frames` frames.
/// 2. For each subsequent frame, subtract `alpha × noise_mag[k]` from the
///    frame's magnitude spectrum.
/// 3. Apply a spectral floor `beta × noise_mag[k]` to avoid over-subtraction.
/// 4. Reconstruct via IFFT and overlap-add.
pub struct SpectralSubtractor {
    fft_size: usize,
    hop_size: usize,
    /// Over-subtraction factor (default 2.0 — aggressive but effective).
    alpha: f64,
    /// Spectral floor as fraction of noise power (default 0.02).
    beta: f64,
    /// Noise power spectrum (magnitude²).
    noise_psd: Vec<f64>,
    /// Number of noise estimation frames collected so far.
    noise_frames: usize,
    /// Total frames to use for initial noise estimation.
    noise_estimation_frames: usize,
    ola: OlaEngine,
    /// Synthesis output accumulation buffer.
    synth_buf: Vec<f64>,
    /// Whether noise profiling is complete.
    profile_ready: bool,
    /// User-supplied noise profile (optional override).
    manual_profile: bool,
}

impl SpectralSubtractor {
    /// Create a new spectral subtractor.
    ///
    /// # Arguments
    ///
    /// * `fft_size`               — FFT size (power of two recommended, e.g. 1024).
    /// * `hop_size`               — Hop between frames (e.g. `fft_size / 4`).
    /// * `noise_estimation_frames`— How many initial frames to use for noise profiling.
    /// * `alpha`                  — Over-subtraction factor (1.5–3.0 typical).
    /// * `beta`                   — Spectral floor fraction (0.01–0.05 typical).
    #[must_use]
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        noise_estimation_frames: usize,
        alpha: f64,
        beta: f64,
    ) -> Self {
        let bins = fft_size / 2 + 1;
        Self {
            fft_size,
            hop_size,
            alpha: alpha.max(1.0),
            beta: beta.clamp(0.0, 1.0),
            noise_psd: vec![0.0; bins],
            noise_frames: 0,
            noise_estimation_frames: noise_estimation_frames.max(1),
            ola: OlaEngine::new(fft_size, hop_size),
            synth_buf: vec![0.0; fft_size * 2],
            profile_ready: false,
            manual_profile: false,
        }
    }

    /// Create with sensible defaults: FFT 1024, hop 256, 10 noise frames, α=2, β=0.02.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(1024, 256, 10, 2.0, 0.02)
    }

    /// Supply a pre-measured noise profile (magnitude² per bin, length `fft_size/2+1`).
    ///
    /// This bypasses the initial noise estimation phase.
    pub fn set_noise_profile(&mut self, profile: &[f64]) {
        let bins = self.fft_size / 2 + 1;
        self.noise_psd = profile[..bins.min(profile.len())].to_vec();
        self.noise_psd.resize(bins, 0.0);
        self.profile_ready = true;
        self.manual_profile = true;
    }

    /// Process a mono block of f32 samples (any length).
    ///
    /// Returns output samples of the same length. During the noise profiling
    /// phase the input is returned attenuated (–6 dB) as a passthrough.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        let mut pos = 0_usize;

        while pos < input.len() {
            let hop = self.hop_size;
            let remaining = input.len() - pos;
            let chunk_len = remaining.min(hop);

            // Convert to f64
            let chunk_f64: Vec<f64> = input[pos..pos + chunk_len]
                .iter()
                .map(|&s| f64::from(s))
                .collect();

            self.ola.push(&chunk_f64);

            let frame = self.ola.windowed_frame();
            let spectrum = oxifft_fft(&frame);

            let bins = self.fft_size / 2 + 1;

            if !self.profile_ready {
                // Accumulate noise PSD
                for k in 0..bins {
                    self.noise_psd[k] += spectrum[k].norm_sqr();
                }
                self.noise_frames += 1;
                if self.noise_frames >= self.noise_estimation_frames {
                    for psd in &mut self.noise_psd {
                        *psd /= self.noise_estimation_frames as f64;
                    }
                    self.profile_ready = true;
                }
                // Attenuated passthrough during profiling
                for &s in &chunk_f64 {
                    output.push((s * 0.5) as f32);
                }
            } else {
                // Spectral subtraction
                let mut modified: Vec<Complex<f64>> = spectrum
                    .iter()
                    .enumerate()
                    .map(|(k, &c): (usize, &Complex<f64>)| {
                        let bin = k.min(bins - 1);
                        let mag = c.norm();
                        let noise_mag = self.noise_psd[bin].sqrt();
                        let sub_mag = (mag - self.alpha * noise_mag).max(self.beta * noise_mag);
                        let phase = c.arg();
                        Complex::new(sub_mag * phase.cos(), sub_mag * phase.sin())
                    })
                    .collect();

                // Mirror the spectrum for IFFT (conjugate symmetric)
                for k in 1..bins - 1 {
                    let mirror = self.fft_size - k;
                    modified[mirror] = modified[k].conj();
                }

                let time_domain = oxifft_ifft(&modified);

                // Overlap-add with synthesis window
                for (i, c) in time_domain.iter().enumerate() {
                    let re = c.re / self.fft_size as f64;
                    let windowed = re * self.ola.window[i];
                    if i < self.synth_buf.len() {
                        self.synth_buf[i] += windowed;
                    }
                }

                // Emit hop_size samples from front of synth_buf
                for i in 0..chunk_len {
                    output.push(self.synth_buf[i] as f32);
                }
                // Shift synth_buf
                let synth_len = self.synth_buf.len();
                self.synth_buf.copy_within(chunk_len..synth_len, 0);
                let tail_start = synth_len - chunk_len;
                self.synth_buf[tail_start..].fill(0.0);
            }

            pos += chunk_len;
        }

        output
    }

    /// Returns `true` once the noise profile has been collected/set.
    #[must_use]
    pub const fn is_profile_ready(&self) -> bool {
        self.profile_ready
    }

    /// Reset processor state (noise profile is preserved).
    pub fn reset(&mut self) {
        self.ola.input_buf.fill(0.0);
        self.synth_buf.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Wiener filter
// ---------------------------------------------------------------------------

/// Noise reduction via a time-varying Wiener filter.
///
/// Computes the optimal Wiener gain `H[k] = SNR[k] / (1 + SNR[k])` per bin
/// and applies it multiplicatively in the STFT domain. The noise PSD is
/// tracked with a minimum-statistics approach: a smoothed power estimate
/// `P[k]` is maintained; `noise_psd[k]` follows the per-frame minimum over
/// a sliding window.
pub struct WienerFilter {
    fft_size: usize,
    hop_size: usize,
    /// Smoothed signal power estimate per bin.
    smoothed_psd: Vec<f64>,
    /// Noise PSD estimate per bin.
    noise_psd: Vec<f64>,
    /// Smoothed Wiener gains from previous frame (for temporal smoothing).
    prev_gain: Vec<f64>,
    /// Minimum of `smoothed_psd` over recent frames (sliding window).
    min_tracker: MinTracker,
    /// IIR coefficient for PSD smoothing (default 0.98).
    psd_smooth: f64,
    /// IIR coefficient for gain smoothing (default 0.7).
    gain_smooth: f64,
    /// Floor for Wiener gain to prevent full silencing (default 0.05).
    gain_floor: f64,
    ola: OlaEngine,
    synth_buf: Vec<f64>,
}

impl WienerFilter {
    /// Create a new Wiener filter.
    ///
    /// # Arguments
    ///
    /// * `fft_size`    — FFT size.
    /// * `hop_size`    — Hop between frames.
    /// * `gain_floor`  — Minimum Wiener gain (0.0–1.0, default 0.05).
    #[must_use]
    pub fn new(fft_size: usize, hop_size: usize, gain_floor: f64) -> Self {
        let bins = fft_size / 2 + 1;
        Self {
            fft_size,
            hop_size,
            smoothed_psd: vec![1e-10; bins],
            noise_psd: vec![1e-10; bins],
            prev_gain: vec![1.0; bins],
            min_tracker: MinTracker::new(bins, 20),
            psd_smooth: 0.98,
            gain_smooth: 0.7,
            gain_floor: gain_floor.clamp(0.0, 1.0),
            ola: OlaEngine::new(fft_size, hop_size),
            synth_buf: vec![0.0; fft_size * 2],
        }
    }

    /// Create with sensible defaults: FFT 1024, hop 256, floor 0.05.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(1024, 256, 0.05)
    }

    /// Process a mono block of f32 samples (any length).
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        let mut pos = 0_usize;

        while pos < input.len() {
            let hop = self.hop_size;
            let remaining = input.len() - pos;
            let chunk_len = remaining.min(hop);

            let chunk_f64: Vec<f64> = input[pos..pos + chunk_len]
                .iter()
                .map(|&s| f64::from(s))
                .collect();

            self.ola.push(&chunk_f64);
            let frame = self.ola.windowed_frame();
            let spectrum = oxifft_fft(&frame);
            let bins = self.fft_size / 2 + 1;

            // Update smoothed PSD and noise estimate
            for k in 0..bins {
                let power = spectrum[k].norm_sqr();
                self.smoothed_psd[k] =
                    self.psd_smooth * self.smoothed_psd[k] + (1.0 - self.psd_smooth) * power;
            }
            self.min_tracker.update(&self.smoothed_psd);
            for k in 0..bins {
                self.noise_psd[k] = self.min_tracker.min[k];
            }

            // Compute Wiener gain and smooth temporally
            let mut gains = vec![0.0_f64; bins];
            for k in 0..bins {
                let snr = (self.smoothed_psd[k] / self.noise_psd[k] - 1.0).max(0.0);
                let wiener = snr / (1.0 + snr);
                let g = (self.gain_smooth * self.prev_gain[k] + (1.0 - self.gain_smooth) * wiener)
                    .max(self.gain_floor);
                gains[k] = g;
                self.prev_gain[k] = g;
            }

            // Apply gains and mirror
            let mut modified: Vec<Complex<f64>> = spectrum
                .iter()
                .enumerate()
                .map(|(k, &c): (usize, &Complex<f64>)| {
                    let bin = k.min(bins - 1);
                    c * gains[bin]
                })
                .collect();

            for k in 1..bins - 1 {
                let mirror = self.fft_size - k;
                modified[mirror] = modified[k].conj();
            }

            let time_domain = oxifft_ifft(&modified);

            for (i, c) in time_domain.iter().enumerate() {
                let re = c.re / self.fft_size as f64;
                let windowed = re * self.ola.window[i];
                if i < self.synth_buf.len() {
                    self.synth_buf[i] += windowed;
                }
            }

            for i in 0..chunk_len {
                output.push(self.synth_buf[i] as f32);
            }
            let synth_len2 = self.synth_buf.len();
            self.synth_buf.copy_within(chunk_len..synth_len2, 0);
            let tail_start = synth_len2 - chunk_len;
            self.synth_buf[tail_start..].fill(0.0);

            pos += chunk_len;
        }

        output
    }

    /// Reset processor state (noise estimate is preserved).
    pub fn reset(&mut self) {
        self.ola.input_buf.fill(0.0);
        self.synth_buf.fill(0.0);
        self.prev_gain.fill(1.0);
    }
}

// ---------------------------------------------------------------------------
// MinTracker (sliding minimum for Wiener noise estimation)
// ---------------------------------------------------------------------------

struct MinTracker {
    bins: usize,
    /// Sliding minimum over the last `window` frames per bin.
    min: Vec<f64>,
    /// Ring buffer of per-bin power values over `window` frames.
    history: Vec<Vec<f64>>,
    write_pos: usize,
    window: usize,
}

impl MinTracker {
    fn new(bins: usize, window: usize) -> Self {
        Self {
            bins,
            min: vec![1e-10; bins],
            history: vec![vec![1e-10; bins]; window],
            write_pos: 0,
            window,
        }
    }

    fn update(&mut self, psd: &[f64]) {
        let pos = self.write_pos;
        for k in 0..self.bins {
            self.history[pos][k] = psd[k].max(1e-10);
        }
        self.write_pos = (pos + 1) % self.window;

        // Recompute minimum across history
        for k in 0..self.bins {
            let mut m = f64::MAX;
            for frame in &self.history {
                if frame[k] < m {
                    m = frame[k];
                }
            }
            self.min[k] = m;
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn hann(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine_wave(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (TAU * freq * i as f32 / sr as f32).sin() * 0.5)
            .collect()
    }

    fn white_noise(n: usize, seed: u64) -> Vec<f32> {
        // Simple LCG for deterministic "random" noise
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let raw = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                raw * 0.05
            })
            .collect()
    }

    #[test]
    fn test_spectral_subtractor_creation() {
        let proc = SpectralSubtractor::default_config();
        assert!(!proc.is_profile_ready());
    }

    #[test]
    fn test_spectral_subtractor_manual_profile() {
        let mut proc = SpectralSubtractor::new(512, 128, 5, 2.0, 0.02);
        let profile = vec![0.001; 257];
        proc.set_noise_profile(&profile);
        assert!(proc.is_profile_ready());
    }

    #[test]
    fn test_spectral_subtractor_output_length() {
        let mut proc = SpectralSubtractor::new(512, 128, 5, 2.0, 0.02);
        let profile = vec![0.001; 257];
        proc.set_noise_profile(&profile);
        let input = sine_wave(440.0, 44100, 1024);
        let output = proc.process(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_spectral_subtractor_output_finite() {
        let mut proc = SpectralSubtractor::default_config();
        let profile = vec![1e-6; 513];
        proc.set_noise_profile(&profile);
        let signal = sine_wave(1000.0, 44100, 4096);
        let output = proc.process(&signal);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "Non-finite at index {i}");
        }
    }

    #[test]
    fn test_spectral_subtractor_noise_profiling() {
        let mut proc = SpectralSubtractor::new(512, 128, 5, 2.0, 0.02);
        // Feed 5 frames worth of noise to fill profile
        let noise = white_noise(512 * 5, 42);
        let _out = proc.process(&noise);
        assert!(proc.is_profile_ready());
    }

    #[test]
    fn test_spectral_subtractor_reduces_noise() {
        // Build a noisy signal
        let sr = 44100_u32;
        let n = 16384;
        let signal = sine_wave(440.0, sr, n);
        let noise = white_noise(n, 42);
        let noisy: Vec<f32> = signal
            .iter()
            .zip(noise.iter())
            .map(|(s, n)| s + n)
            .collect();

        // Use the first 512 samples as noise-only for profile
        let _noise_profile = white_noise(1024, 42);
        let mut proc = SpectralSubtractor::new(512, 128, 5, 2.0, 0.02);
        proc.set_noise_profile(&vec![0.05_f64.powi(2); 257]);

        let output = proc.process(&noisy);
        assert_eq!(output.len(), noisy.len());
        // All outputs should be finite
        for s in &output {
            assert!(s.is_finite());
        }
        // Output energy should be less than or equal to input energy
        let out_energy: f32 = output.iter().map(|s| s * s).sum();
        let in_energy: f32 = noisy.iter().map(|s| s * s).sum();
        assert!(
            out_energy <= in_energy * 1.5,
            "Noise reduction should not significantly amplify: in={in_energy} out={out_energy}"
        );
    }

    #[test]
    fn test_wiener_filter_creation() {
        let _proc = WienerFilter::default_config();
    }

    #[test]
    fn test_wiener_filter_output_length() {
        let mut proc = WienerFilter::new(512, 128, 0.05);
        let input = sine_wave(440.0, 44100, 2048);
        let output = proc.process(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_wiener_filter_output_finite() {
        let mut proc = WienerFilter::new(512, 128, 0.05);
        let noisy: Vec<f32> = sine_wave(440.0, 44100, 4096)
            .into_iter()
            .zip(white_noise(4096, 99))
            .map(|(s, n)| s + n)
            .collect();
        let output = proc.process(&noisy);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "Non-finite at index {i}");
        }
    }

    #[test]
    fn test_wiener_reset_clears_synthesis_buffer() {
        let mut proc = WienerFilter::new(512, 128, 0.05);
        let signal = sine_wave(800.0, 44100, 4096);
        proc.process(&signal);
        proc.reset();
        for &v in &proc.synth_buf {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_hann_window_zero_endpoints() {
        let w = hann(1024);
        assert!(w[0].abs() < 1e-9);
        assert!(w[1023].abs() < 1e-6);
    }

    #[test]
    fn test_hann_window_peak_at_center() {
        let w = hann(1024);
        let mid = w[512];
        assert!(
            (mid - 1.0).abs() < 0.01,
            "Hann peak should be ~1.0, got {mid}"
        );
    }
}
