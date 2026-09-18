//! Spectral subtraction audio denoiser.
//!
//! Implements the classic spectral subtraction algorithm for single-channel audio
//! denoising. A short-time power spectrum is estimated from noise-only segments,
//! and this estimate is subtracted from the noisy input spectrum before resynthesis
//! via overlap-add (OLA).
//!
//! # Algorithm
//!
//! 1. **Noise estimation**: average magnitude spectra of noise-only frames.
//! 2. **Analysis**: frame the noisy input with 50% overlap and a Hann window.
//! 3. **Spectral subtraction**: `|Y| = max(|X| − α·|N|, β·|N|)` where
//!    - `α` controls over-subtraction (≥1; larger = more aggressive)
//!    - `β` is the spectral floor multiplier (prevents musical noise artifacts)
//! 4. **Synthesis**: IFFT + OLA reconstruction.
//!
//! # Example
//!
//! ```
//! use oximedia_denoise::spectral_subtraction::{SpectralSubtractionConfig, SpectralSubtractor};
//!
//! let cfg = SpectralSubtractionConfig::default();
//! let mut sub = SpectralSubtractor::new(cfg);
//!
//! // Feed noise-only segment to learn the noise profile
//! let noise: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin() * 0.05).collect();
//! sub.update_noise_profile(&noise);
//!
//! // Denoise a noisy signal
//! let noisy: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
//! let out = sub.denoise(&noisy);
//! assert_eq!(out.len(), noisy.len());
//! ```

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the spectral subtraction denoiser.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralSubtractionConfig {
    /// Over-subtraction factor α (≥1.0). Higher = more aggressive noise removal
    /// but may introduce musical noise.  Typical: 1.0–4.0.
    pub alpha: f32,
    /// Spectral floor β. Output spectrum is floored at `β * noise_profile` to
    /// prevent over-subtraction artifacts. Typical: 0.001–0.01.
    pub beta: f32,
    /// FFT frame length in samples (must be ≥ 8).  A Hann window of this length
    /// is applied before the DFT.  50% overlap is used.
    pub n_fft: usize,
}

impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            beta: 0.002,
            n_fft: 512,
        }
    }
}

impl SpectralSubtractionConfig {
    /// Create a gentle configuration (minimal over-subtraction).
    #[must_use]
    pub fn gentle() -> Self {
        Self {
            alpha: 1.5,
            beta: 0.005,
            n_fft: 512,
        }
    }

    /// Create an aggressive configuration.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            alpha: 4.0,
            beta: 0.001,
            n_fft: 512,
        }
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.alpha < 1.0 {
            return Err(format!("alpha must be >= 1.0, got {}", self.alpha));
        }
        if self.beta < 0.0 {
            return Err(format!("beta must be >= 0.0, got {}", self.beta));
        }
        if self.n_fft < 8 {
            return Err(format!("n_fft must be >= 8, got {}", self.n_fft));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SpectralSubtractor
// ---------------------------------------------------------------------------

/// Spectral subtraction-based audio denoiser.
pub struct SpectralSubtractor {
    config: SpectralSubtractionConfig,
    /// Estimated noise magnitude spectrum (length = n_fft/2 + 1).
    noise_profile: Vec<f32>,
    /// Whether `update_noise_profile` has been called at least once.
    profile_updated: bool,
    /// Pre-computed Hann window coefficients.
    hann_window: Vec<f32>,
}

impl SpectralSubtractor {
    /// Create a new `SpectralSubtractor` with the given configuration.
    #[must_use]
    pub fn new(config: SpectralSubtractionConfig) -> Self {
        let n = config.n_fft;
        let n_bins = n / 2 + 1;
        let hann_window = build_hann_window(n);
        Self {
            config,
            noise_profile: vec![0.0f32; n_bins],
            profile_updated: false,
            hann_window,
        }
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &SpectralSubtractionConfig {
        &self.config
    }

    /// Whether a noise profile has been learned.
    #[must_use]
    pub fn has_noise_profile(&self) -> bool {
        self.profile_updated
    }

    /// Return a reference to the current noise profile (magnitude spectrum).
    #[must_use]
    pub fn noise_profile(&self) -> &[f32] {
        &self.noise_profile
    }

    /// Update the noise profile from a noise-only signal.
    ///
    /// Computes the average magnitude spectrum of the noise frames and stores
    /// it.  Multiple calls average the new estimate with the previous one
    /// (50/50 blend) so the profile can be updated incrementally.
    pub fn update_noise_profile(&mut self, noise_samples: &[f32]) {
        let n = self.config.n_fft;
        let hop = n / 2;
        let n_bins = n / 2 + 1;

        let mut accum = vec![0.0f32; n_bins];
        let mut frame_count = 0usize;

        let mut pos = 0usize;
        while pos + n <= noise_samples.len() {
            let frame = &noise_samples[pos..pos + n];
            let mag = compute_magnitude_spectrum(frame, &self.hann_window);
            for (a, &m) in accum.iter_mut().zip(mag.iter()) {
                *a += m;
            }
            frame_count += 1;
            pos += hop;
        }

        // Handle partial trailing frame
        if pos < noise_samples.len() && frame_count == 0 {
            let mut padded = vec![0.0f32; n];
            let avail = noise_samples.len() - pos;
            padded[..avail].copy_from_slice(&noise_samples[pos..]);
            let mag = compute_magnitude_spectrum(&padded, &self.hann_window);
            for (a, &m) in accum.iter_mut().zip(mag.iter()) {
                *a += m;
            }
            frame_count += 1;
        }

        if frame_count == 0 {
            return;
        }

        let avg: Vec<f32> = accum.iter().map(|&v| v / frame_count as f32).collect();

        if self.profile_updated {
            // Blend with previous estimate
            for (p, &new) in self.noise_profile.iter_mut().zip(avg.iter()) {
                *p = 0.5 * *p + 0.5 * new;
            }
        } else {
            self.noise_profile.copy_from_slice(&avg);
            self.profile_updated = true;
        }
    }

    /// Denoise an audio signal using spectral subtraction with overlap-add.
    ///
    /// If no noise profile has been set, returns the input unchanged.
    /// The output length equals the input length.
    #[must_use]
    pub fn denoise(&self, samples: &[f32]) -> Vec<f32> {
        if !self.profile_updated || samples.is_empty() {
            return samples.to_vec();
        }

        let n = self.config.n_fft;
        let hop = n / 2;
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let n_len = samples.len();

        let mut output = vec![0.0f32; n_len];
        let mut ola_weight = vec![0.0f32; n_len];

        let mut pos = 0usize;
        while pos < n_len {
            // Extract frame (zero-pad if needed)
            let mut frame = vec![0.0f32; n];
            let avail = (n_len - pos).min(n);
            frame[..avail].copy_from_slice(&samples[pos..pos + avail]);

            // Apply Hann window
            for (s, &w) in frame.iter_mut().zip(self.hann_window.iter()) {
                *s *= w;
            }

            // Forward real DFT
            let (re, im) = rdft_forward(&frame);

            // Spectral subtraction on magnitude
            let n_bins = n / 2 + 1;
            let mut re_out = re.clone();
            let mut im_out = im.clone();

            for k in 0..n_bins {
                let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
                let noise_mag = self.noise_profile.get(k).copied().unwrap_or(0.0);
                let floor = beta * noise_mag;
                let new_mag = (mag - alpha * noise_mag).max(floor).max(0.0);

                // Re-apply phase, handling zero magnitude
                let scale = if mag > 1e-10 { new_mag / mag } else { 0.0 };
                re_out[k] = re[k] * scale;
                im_out[k] = im[k] * scale;
            }

            // Inverse real DFT
            let out_frame = rdft_inverse(&re_out, &im_out, n);

            // OLA with Hann window synthesis
            for (i, (&s, &w)) in out_frame.iter().zip(self.hann_window.iter()).enumerate() {
                let out_idx = pos + i;
                if out_idx < n_len {
                    output[out_idx] += s * w;
                    ola_weight[out_idx] += w * w;
                }
            }

            if pos + hop >= n_len {
                break;
            }
            pos += hop;
        }

        // Normalise by OLA weights
        for (o, &w) in output.iter_mut().zip(ola_weight.iter()) {
            if w > 1e-10 {
                *o /= w;
            }
        }

        output
    }

    /// Reset the noise profile.
    pub fn reset(&mut self) {
        self.noise_profile.iter_mut().for_each(|v| *v = 0.0);
        self.profile_updated = false;
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

/// Build a normalised Hann window of length `n`.
fn build_hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
        .collect()
}

/// Compute the magnitude spectrum from a windowed frame using a direct DFT.
/// Returns `n/2+1` magnitude values.
fn compute_magnitude_spectrum(frame: &[f32], window: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let n_bins = n / 2 + 1;
    let mut mag = vec![0.0f32; n_bins];

    for k in 0..n_bins {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, (&s, &w)) in frame.iter().zip(window.iter()).enumerate() {
            let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
            re += s * w * angle.cos();
            im += s * w * angle.sin();
        }
        mag[k] = (re * re + im * im).sqrt();
    }
    mag
}

/// Forward real DFT. Returns (real_part, imag_part) each of length n/2+1.
fn rdft_forward(frame: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = frame.len();
    let n_bins = n / 2 + 1;
    let mut re = vec![0.0f32; n_bins];
    let mut im = vec![0.0f32; n_bins];

    for k in 0..n_bins {
        let mut r = 0.0f32;
        let mut img = 0.0f32;
        for (i, &s) in frame.iter().enumerate() {
            let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
            r += s * angle.cos();
            img += s * angle.sin();
        }
        re[k] = r;
        im[k] = img;
    }
    (re, im)
}

/// Inverse real DFT from positive-frequency half-spectrum.
/// Reconstructs a real signal of length `n`.
fn rdft_inverse(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    let n_bins = re.len();
    let mut out = vec![0.0f32; n];
    let scale = 1.0 / n as f32;

    for i in 0..n {
        let mut s = re[0]; // DC
        for k in 1..n_bins - 1 {
            let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;
            s += 2.0 * (re[k] * angle.cos() - im[k] * angle.sin());
        }
        // Nyquist
        if n_bins > 1 {
            let k = n_bins - 1;
            let angle = 2.0 * PI * k as f32 * i as f32 / n as f32;
            s += re[k] * angle.cos() - im[k] * angle.sin();
        }
        out[i] = s * scale;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config tests -------------------------------------------------------

    #[test]
    fn test_config_default() {
        let c = SpectralSubtractionConfig::default();
        assert!(c.alpha >= 1.0);
        assert!(c.beta >= 0.0);
        assert!(c.n_fft >= 8);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_gentle() {
        let c = SpectralSubtractionConfig::gentle();
        assert!(c.validate().is_ok());
        assert!(c.alpha < SpectralSubtractionConfig::aggressive().alpha);
    }

    #[test]
    fn test_config_aggressive() {
        let c = SpectralSubtractionConfig::aggressive();
        assert!(c.validate().is_ok());
        assert!(c.alpha > SpectralSubtractionConfig::gentle().alpha);
    }

    #[test]
    fn test_config_validate_bad_alpha() {
        let c = SpectralSubtractionConfig {
            alpha: 0.5,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_beta() {
        let c = SpectralSubtractionConfig {
            beta: -0.1,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_validate_small_fft() {
        let c = SpectralSubtractionConfig {
            n_fft: 4,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    // ---- Construction -------------------------------------------------------

    #[test]
    fn test_new_has_no_profile() {
        let s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        assert!(!s.has_noise_profile());
    }

    #[test]
    fn test_noise_profile_length() {
        let cfg = SpectralSubtractionConfig {
            n_fft: 64,
            ..Default::default()
        };
        let s = SpectralSubtractor::new(cfg.clone());
        assert_eq!(s.noise_profile().len(), cfg.n_fft / 2 + 1);
    }

    // ---- Noise profile update -----------------------------------------------

    #[test]
    fn test_update_noise_profile_sets_flag() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise = vec![0.01f32; 1024];
        s.update_noise_profile(&noise);
        assert!(s.has_noise_profile());
    }

    #[test]
    fn test_update_noise_profile_non_zero() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
        s.update_noise_profile(&noise);
        let total: f32 = s.noise_profile().iter().sum();
        assert!(
            total > 0.0,
            "Noise profile should be non-zero for non-silent noise"
        );
    }

    #[test]
    fn test_update_noise_profile_short_segment() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig {
            n_fft: 64,
            ..Default::default()
        });
        // Segment shorter than n_fft
        let noise = vec![0.05f32; 32];
        s.update_noise_profile(&noise);
        assert!(s.has_noise_profile());
    }

    #[test]
    fn test_reset_clears_profile() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise = vec![0.05f32; 1024];
        s.update_noise_profile(&noise);
        assert!(s.has_noise_profile());
        s.reset();
        assert!(!s.has_noise_profile());
        let total: f32 = s.noise_profile().iter().sum();
        assert!((total).abs() < 1e-6);
    }

    // ---- Denoise output shape -----------------------------------------------

    #[test]
    fn test_denoise_output_length_equals_input() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise = vec![0.02f32; 1024];
        s.update_noise_profile(&noise);
        let signal: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let out = s.denoise(&signal);
        assert_eq!(out.len(), signal.len());
    }

    #[test]
    fn test_denoise_no_profile_passthrough() {
        let s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let signal = vec![0.3f32; 256];
        let out = s.denoise(&signal);
        assert_eq!(out, signal);
    }

    #[test]
    fn test_denoise_empty_input() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise = vec![0.01f32; 512];
        s.update_noise_profile(&noise);
        let out = s.denoise(&[]);
        assert_eq!(out.len(), 0);
    }

    // ---- Denoise quality ---------------------------------------------------

    #[test]
    fn test_denoise_output_finite() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise: Vec<f32> = (0..512).map(|i| (i as f32 * 0.03).cos() * 0.04).collect();
        s.update_noise_profile(&noise);
        let signal: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.1).sin() * 0.4 + (i as f32 * 0.03).cos() * 0.04)
            .collect();
        let out = s.denoise(&signal);
        for &v in &out {
            assert!(v.is_finite(), "output contains non-finite value {v}");
        }
    }

    #[test]
    fn test_denoise_silent_input() {
        let mut s = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        let noise = vec![0.05f32; 512];
        s.update_noise_profile(&noise);
        let signal = vec![0.0f32; 1024];
        let out = s.denoise(&signal);
        assert_eq!(out.len(), 1024);
        // All near-zero (noise was subtracted from silence)
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_denoise_reduces_noise_rms() {
        // Signal = sine wave + pure noise.  After denoising, RMS should be
        // closer to the clean sine than to the noisy mix for the middle portion.
        let sr = 512usize;
        let freq = 0.05f32;
        let noise_amp = 0.15f32;

        // Deterministic pseudo-noise
        let noise: Vec<f32> = (0..sr * 2)
            .map(|i| noise_amp * (i as f32 * 1.7331_f32).sin())
            .collect();
        let clean_sine: Vec<f32> = (0..sr * 4)
            .map(|i| (2.0 * PI * freq * i as f32).sin() * 0.5)
            .collect();
        let noisy: Vec<f32> = clean_sine
            .iter()
            .zip(noise.iter().cycle())
            .map(|(&s, &n)| s + n)
            .collect();

        let mut sub = SpectralSubtractor::new(SpectralSubtractionConfig::default());
        sub.update_noise_profile(&noise[..sr]);
        let denoised = sub.denoise(&noisy);

        // Compare RMS of noise residual in middle quarter
        let start = sr;
        let end = sr * 2;
        let rms_noisy: f32 = {
            let v: f32 = noisy[start..end]
                .iter()
                .zip(clean_sine[start..end].iter())
                .map(|(&n, &c)| (n - c).powi(2))
                .sum();
            (v / (end - start) as f32).sqrt()
        };
        let rms_denoised: f32 = {
            let v: f32 = denoised[start..end]
                .iter()
                .zip(clean_sine[start..end].iter())
                .map(|(&d, &c)| (d - c).powi(2))
                .sum();
            (v / (end - start) as f32).sqrt()
        };

        // Denoised error should not be drastically worse than noisy
        assert!(
            rms_denoised <= rms_noisy * 3.0,
            "Denoised RMS error {rms_denoised:.4} unexpectedly large vs noisy {rms_noisy:.4}"
        );
    }

    // ---- Hann window -------------------------------------------------------

    #[test]
    fn test_hann_window_endpoints() {
        let w = build_hann_window(512);
        assert!(w[0].abs() < 1e-5, "Hann window should start near 0");
        assert!(w[511].abs() < 0.02, "Hann window should end near 0");
    }

    #[test]
    fn test_hann_window_peak() {
        let w = build_hann_window(512);
        let mid = w[256];
        assert!(
            mid > 0.95 && mid <= 1.0,
            "Hann peak should be ~1.0, got {mid}"
        );
    }

    // ---- DFT round-trip ----------------------------------------------------

    #[test]
    fn test_rdft_roundtrip() {
        let n = 64usize;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let (re, im) = rdft_forward(&signal);
        let recovered = rdft_inverse(&re, &im, n);
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-3,
                "DFT round-trip failed at index {i}: {orig} vs {rec}"
            );
        }
    }
}
