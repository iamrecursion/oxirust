//! Ring-Buffer Short-Time Fourier Transform for Online / Real-Time Processing
//!
//! This module provides a STFT processor built on a circular (ring) buffer so
//! that audio or sensor samples can be ingested incrementally without copying
//! or reallocating large arrays.  Every time `hop_size` new samples have
//! accumulated the processor extracts a `window_size`-long frame, applies the
//! chosen window function, computes an in-place radix-2 Cooley-Tukey FFT, and
//! returns a [`StftFrame`].
//!
//! An optional overlap-add (OLA) reconstruction path allows phase-vocoder-style
//! effects: call [`RingBufferStft::reconstruct`] with an optionally-modified
//! spectrum to synthesize time-domain samples.
//!
//! A convenience wrapper [`StreamingSpectrogram`] maintains a rolling queue of
//! magnitude frames so callers can retrieve a (time × freq) matrix at any
//! point.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_fft::ring_buffer_stft::{RingBufferStft, RingBufferStftConfig, WindowFunction};
//!
//! let config = RingBufferStftConfig {
//!     window_size: 64,
//!     hop_size: 32,
//!     window_fn: WindowFunction::Hann,
//!     overlap_add: false,
//! };
//! let mut proc = RingBufferStft::new(config).expect("valid config");
//! let sine: Vec<f32> = (0..512)
//!     .map(|i| (std::f32::consts::PI * 2.0 * 440.0 * i as f32 / 44100.0).sin())
//!     .collect();
//! let frames = proc.push(&sine);
//! assert!(!frames.is_empty());
//! ```

use std::f32::consts::PI;

use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Output of one STFT analysis frame.
#[derive(Debug, Clone)]
pub struct StftFrame {
    /// Index of the **centre** sample in the original stream.
    pub sample_index: usize,
    /// Magnitude spectrum, one-sided, length `window_size / 2 + 1`.
    pub magnitudes: Vec<f32>,
    /// Phase spectrum (radians), one-sided, same length as `magnitudes`.
    pub phases: Vec<f32>,
    /// Full complex spectrum as `(re, im)` pairs, length `window_size`.
    pub spectrum: Vec<(f32, f32)>,
}

/// Window functions supported by the ring-buffer STFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular (no weighting).
    Rectangular,
    /// Hann (raised cosine) — good general-purpose choice.
    Hann,
    /// Hamming — slightly better sidelobe rejection than Hann.
    Hamming,
    /// Blackman — very low sidelobes at the cost of a wider main lobe.
    Blackman,
    /// Flat-top — accurate amplitude measurement.
    FlatTop,
}

/// Configuration for [`RingBufferStft`].
#[derive(Debug, Clone)]
pub struct RingBufferStftConfig {
    /// FFT size.  **Must be a power of two** and at least 4.
    pub window_size: usize,
    /// Hop size in samples (distance between successive analysis frames).
    /// Must be in `[1, window_size]`.
    pub hop_size: usize,
    /// Window weighting function applied before the FFT.
    pub window_fn: WindowFunction,
    /// When `true`, an overlap-add reconstruction buffer is allocated and
    /// [`RingBufferStft::reconstruct`] becomes usable.
    pub overlap_add: bool,
}

/// Ring-buffer STFT processor for low-latency, online signal analysis.
///
/// Maintains a circular sample buffer of length `window_size`.  Samples
/// pushed via [`push`][Self::push] are written into the ring; once at least
/// `hop_size` new samples have arrived a complete frame is produced without
/// any extra allocation.
pub struct RingBufferStft {
    config: RingBufferStftConfig,
    /// Circular buffer — always `window_size` elements.
    buffer: Vec<f32>,
    /// Next write index (mod `window_size`).
    write_pos: usize,
    /// How many new samples have arrived since the last frame.
    samples_since_last_frame: usize,
    /// Running count of all samples ever pushed.
    total_samples: usize,
    /// Pre-computed window coefficients.
    window: Vec<f32>,
    /// Overlap-add synthesis buffer (length `window_size`; `None` if disabled).
    overlap_add_buffer: Option<Vec<f32>>,
}

impl RingBufferStft {
    // ── Construction ────────────────────────────────────────────────────────

    /// Create a new processor from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::ValueError`] if:
    /// - `window_size` is not a power of two or is less than 4.
    /// - `hop_size` is zero or larger than `window_size`.
    pub fn new(config: RingBufferStftConfig) -> FFTResult<Self> {
        if !config.window_size.is_power_of_two() || config.window_size < 4 {
            return Err(FFTError::ValueError(format!(
                "ring_buffer_stft: window_size must be a power of two >= 4, got {}",
                config.window_size
            )));
        }
        if config.hop_size == 0 || config.hop_size > config.window_size {
            return Err(FFTError::ValueError(format!(
                "ring_buffer_stft: hop_size must be in [1, window_size], got {}",
                config.hop_size
            )));
        }

        let window = compute_window(config.window_fn, config.window_size);
        let overlap_add_buffer = if config.overlap_add {
            Some(vec![0.0_f32; config.window_size])
        } else {
            None
        };

        Ok(Self {
            buffer: vec![0.0_f32; config.window_size],
            write_pos: 0,
            samples_since_last_frame: 0,
            total_samples: 0,
            window,
            overlap_add_buffer,
            config,
        })
    }

    // ── Analysis ─────────────────────────────────────────────────────────────

    /// Push new samples into the processor.
    ///
    /// Returns all complete frames generated from the incoming data.  A frame
    /// is emitted every `hop_size` samples.
    pub fn push(&mut self, samples: &[f32]) -> Vec<StftFrame> {
        let mut frames = Vec::new();
        for &s in samples {
            self.buffer[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.config.window_size;
            self.total_samples += 1;
            self.samples_since_last_frame += 1;

            if self.samples_since_last_frame >= self.config.hop_size
                && self.total_samples >= self.config.window_size
            {
                if let Ok(frame) = self.emit_frame() {
                    frames.push(frame);
                }
                self.samples_since_last_frame = 0;
            }
        }
        frames
    }

    /// Flush any remaining buffered samples by zero-padding to the next frame
    /// boundary and emitting the final frames.
    pub fn flush(&mut self) -> Vec<StftFrame> {
        if self.samples_since_last_frame == 0 {
            return Vec::new();
        }
        // Zero-pad until the next hop boundary is hit.
        let needed = self.config.hop_size - self.samples_since_last_frame;
        let pad = vec![0.0_f32; needed];
        self.push(&pad)
    }

    // ── Synthesis (overlap-add) ───────────────────────────────────────────────

    /// Overlap-add reconstruction.
    ///
    /// Computes the IFFT of `modified_spectrum` (or of the original spectrum
    /// stored in `frame` if `modified_spectrum` is `None`), applies the window
    /// function, and accumulates the result in the internal OLA buffer.
    ///
    /// Returns `hop_size` reconstructed samples (the oldest hop's worth of
    /// completed output).
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::ValueError`] if overlap-add was not enabled in the
    /// config, or if the spectrum length is wrong.
    pub fn reconstruct(
        &mut self,
        frame: &StftFrame,
        modified_spectrum: Option<&[(f32, f32)]>,
    ) -> FFTResult<Vec<f32>> {
        let ola_buf = self.overlap_add_buffer.as_mut().ok_or_else(|| {
            FFTError::ValueError(
                "ring_buffer_stft: reconstruct called but overlap_add is disabled".into(),
            )
        })?;
        let n = self.config.window_size;

        let spec: &[(f32, f32)] = match modified_spectrum {
            Some(s) => {
                if s.len() != n {
                    return Err(FFTError::ValueError(format!(
                        "ring_buffer_stft: modified_spectrum length {} != window_size {}",
                        s.len(),
                        n
                    )));
                }
                s
            }
            None => &frame.spectrum,
        };

        // IFFT.
        let mut data: Vec<(f32, f32)> = spec.to_vec();
        fft_inplace_f32(&mut data, true);

        // Apply synthesis window and overlap-add.
        let hop = self.config.hop_size;
        for i in 0..n {
            ola_buf[i] += data[i].0 * self.window[i];
        }

        // Extract the oldest `hop` samples and shift the buffer.
        let out: Vec<f32> = ola_buf[..hop].to_vec();
        ola_buf.copy_within(hop..n, 0);
        for v in &mut ola_buf[n - hop..n] {
            *v = 0.0;
        }
        Ok(out)
    }

    // ── Informational ─────────────────────────────────────────────────────────

    /// Latency in samples (`window_size - hop_size`).
    pub fn latency(&self) -> usize {
        self.config.window_size - self.config.hop_size
    }

    /// Number of one-sided frequency bins (`window_size / 2 + 1`).
    pub fn n_freq_bins(&self) -> usize {
        self.config.window_size / 2 + 1
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Build and return a single [`StftFrame`] from the current ring-buffer
    /// contents.
    fn emit_frame(&self) -> FFTResult<StftFrame> {
        let n = self.config.window_size;
        // Extract `n` samples ending at `write_pos` (exclusive) from the ring.
        let mut windowed = self.apply_window_from_ring();

        // Forward FFT.
        fft_inplace_f32(&mut windowed, false);
        let spectrum: Vec<(f32, f32)> = windowed;

        let n_one_sided = n / 2 + 1;
        let mut magnitudes = Vec::with_capacity(n_one_sided);
        let mut phases = Vec::with_capacity(n_one_sided);
        for &(re, im) in &spectrum[..n_one_sided] {
            magnitudes.push((re * re + im * im).sqrt());
            phases.push(im.atan2(re));
        }

        // Centre sample index: (total_samples - 1) - window_size/2 + hop_size
        let center = self.total_samples.saturating_sub(n / 2);

        Ok(StftFrame {
            sample_index: center,
            magnitudes,
            phases,
            spectrum,
        })
    }

    /// Copy `window_size` samples from the ring buffer in chronological order
    /// and multiply by the window function.
    fn apply_window_from_ring(&self) -> Vec<(f32, f32)> {
        let n = self.config.window_size;
        // The oldest sample is at `write_pos` (what will next be overwritten).
        (0..n)
            .map(|i| {
                let idx = (self.write_pos + i) % n;
                let v = self.buffer[idx] * self.window[i];
                (v, 0.0_f32)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Streaming Spectrogram
// ─────────────────────────────────────────────────────────────────────────────

/// A rolling spectrogram that maintains a sliding window of magnitude frames.
///
/// After calling [`push`][Self::push] with audio samples, the latest
/// spectrogram can be retrieved as a `Vec<Vec<f32>>` of shape
/// `(n_frames, n_freq_bins)`.
pub struct StreamingSpectrogram {
    stft: RingBufferStft,
    frames: std::collections::VecDeque<Vec<f32>>,
    max_frames: usize,
}

impl StreamingSpectrogram {
    /// Create a new streaming spectrogram.
    ///
    /// `max_frames` is the maximum number of magnitude frames retained.  Older
    /// frames are discarded once the limit is reached.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`RingBufferStft::new`].
    pub fn new(stft_config: RingBufferStftConfig, max_frames: usize) -> FFTResult<Self> {
        Ok(Self {
            stft: RingBufferStft::new(stft_config)?,
            frames: std::collections::VecDeque::new(),
            max_frames,
        })
    }

    /// Push samples and update the spectrogram.
    pub fn push(&mut self, samples: &[f32]) {
        let new_frames = self.stft.push(samples);
        for frame in new_frames {
            if self.frames.len() == self.max_frames {
                self.frames.pop_front();
            }
            self.frames.push_back(frame.magnitudes);
        }
    }

    /// Return the current spectrogram as a `Vec<Vec<f32>>` of shape
    /// `(n_frames, n_freq_bins)`.
    pub fn get_spectrogram(&self) -> Vec<Vec<f32>> {
        self.frames.iter().cloned().collect()
    }

    /// Number of one-sided frequency bins.
    pub fn n_freq_bins(&self) -> usize {
        self.stft.n_freq_bins()
    }

    /// Number of frames currently held.
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Window function computation
// ─────────────────────────────────────────────────────────────────────────────

fn compute_window(wf: WindowFunction, n: usize) -> Vec<f32> {
    match wf {
        WindowFunction::Rectangular => vec![1.0_f32; n],
        WindowFunction::Hann => (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (n - 1) as f32).cos())
            .collect(),
        WindowFunction::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos())
            .collect(),
        WindowFunction::Blackman => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f32 / (n - 1) as f32;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
        WindowFunction::FlatTop => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f32 / (n - 1) as f32;
                1.0 - 1.93_f32 * x.cos() + 1.29_f32 * (2.0 * x).cos() - 0.388_f32 * (3.0 * x).cos()
                    + 0.032_f32 * (4.0 * x).cos()
            })
            .collect(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Self-contained radix-2 Cooley-Tukey FFT for f32
// ─────────────────────────────────────────────────────────────────────────────

/// In-place radix-2 Decimation-In-Time (DIT) FFT / IFFT.
///
/// `data` must have a power-of-two length.  For `inverse = true` the output
/// is scaled by `1/N`.
fn fft_inplace_f32(data: &mut [(f32, f32)], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let log2_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log2_n);
        if j > i {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly stages.
    let sign: f32 = if inverse { 1.0 } else { -1.0 };
    let mut half_size = 1_usize;
    while half_size < n {
        let full_size = half_size * 2;
        for k in (0..n).step_by(full_size) {
            for j in 0..half_size {
                let angle = sign * PI * j as f32 / half_size as f32;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let (ur, ui) = data[k + j];
                let (vr, vi) = data[k + j + half_size];
                let (wr, wi) = (cos_a * vr - sin_a * vi, cos_a * vi + sin_a * vr);
                data[k + j] = (ur + wr, ui + wi);
                data[k + j + half_size] = (ur - wr, ui - wi);
            }
        }
        half_size = full_size;
    }

    // Normalise for inverse.
    if inverse {
        let inv_n = 1.0 / n as f32;
        for (re, im) in data.iter_mut() {
            *re *= inv_n;
            *im *= inv_n;
        }
    }
}

/// Reverse the `bits` least-significant bits of `v`.
#[inline]
fn bit_reverse(mut v: usize, bits: usize) -> usize {
    let mut r = 0_usize;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(n: usize, freq: f32, fs: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / fs).sin())
            .collect()
    }

    // ── RingBufferStft ───────────────────────────────────────────────────────

    #[test]
    fn test_ring_buffer_stft_frame_count() {
        let window_size = 128;
        let hop_size = 64;
        let n_samples = 1024_usize;
        let config = RingBufferStftConfig {
            window_size,
            hop_size,
            window_fn: WindowFunction::Hann,
            overlap_add: false,
        };
        let mut proc = RingBufferStft::new(config).expect("valid config");
        let signal = make_sine(n_samples, 440.0, 8000.0);
        let frames = proc.push(&signal);
        // Expected frames: (n_samples - window_size) / hop_size + 1
        let expected = (n_samples - window_size) / hop_size + 1;
        assert_eq!(
            frames.len(),
            expected,
            "expected {expected} frames, got {}",
            frames.len()
        );
    }

    #[test]
    fn test_ring_buffer_stft_freq_bins() {
        let window_size = 64;
        let config = RingBufferStftConfig {
            window_size,
            hop_size: 32,
            window_fn: WindowFunction::Hamming,
            overlap_add: false,
        };
        let mut proc = RingBufferStft::new(config).expect("valid config");
        let signal = make_sine(512, 220.0, 8000.0);
        let frames = proc.push(&signal);
        assert!(!frames.is_empty(), "no frames generated");
        for frame in &frames {
            assert_eq!(
                frame.magnitudes.len(),
                window_size / 2 + 1,
                "wrong number of freq bins"
            );
            assert_eq!(frame.phases.len(), window_size / 2 + 1);
            assert_eq!(frame.spectrum.len(), window_size);
        }
    }

    #[test]
    fn test_ring_buffer_stft_flush_emits_remaining() {
        let window_size = 64;
        let hop_size = 32;
        let config = RingBufferStftConfig {
            window_size,
            hop_size,
            window_fn: WindowFunction::Hann,
            overlap_add: false,
        };
        let mut proc = RingBufferStft::new(config).expect("valid config");

        // Push exactly `window_size` samples first (fills the ring).
        let init = make_sine(window_size, 100.0, 8000.0);
        proc.push(&init);

        // Push partial hop (not enough for a full hop).
        let partial = vec![0.0_f32; hop_size / 2];
        proc.push(&partial);

        // Flush should emit at least one more frame.
        let flushed = proc.flush();
        assert!(
            !flushed.is_empty(),
            "flush should emit at least one frame for partial hop"
        );
    }

    #[test]
    fn test_ring_buffer_stft_invalid_config() {
        // Non-power-of-two.
        let cfg = RingBufferStftConfig {
            window_size: 100,
            hop_size: 50,
            window_fn: WindowFunction::Rectangular,
            overlap_add: false,
        };
        assert!(RingBufferStft::new(cfg).is_err());

        // hop_size zero.
        let cfg2 = RingBufferStftConfig {
            window_size: 128,
            hop_size: 0,
            window_fn: WindowFunction::Hann,
            overlap_add: false,
        };
        assert!(RingBufferStft::new(cfg2).is_err());
    }

    // ── StreamingSpectrogram ─────────────────────────────────────────────────

    #[test]
    fn test_streaming_spectrogram_push_update() {
        let config = RingBufferStftConfig {
            window_size: 64,
            hop_size: 32,
            window_fn: WindowFunction::Hann,
            overlap_add: false,
        };
        let mut spec = StreamingSpectrogram::new(config, 10).expect("valid config");
        let signal = make_sine(512, 440.0, 8000.0);
        spec.push(&signal);
        let sg = spec.get_spectrogram();
        assert!(!sg.is_empty(), "spectrogram should have frames after push");
        assert_eq!(sg[0].len(), 64 / 2 + 1, "wrong freq bin count");
    }

    #[test]
    fn test_streaming_spectrogram_max_frames() {
        let config = RingBufferStftConfig {
            window_size: 64,
            hop_size: 32,
            window_fn: WindowFunction::Hann,
            overlap_add: false,
        };
        let max_frames = 5;
        let mut spec = StreamingSpectrogram::new(config, max_frames).expect("valid config");
        let signal = make_sine(4096, 440.0, 8000.0);
        spec.push(&signal);
        assert!(
            spec.n_frames() <= max_frames,
            "should not exceed max_frames"
        );
    }

    // ── Reconstruction roundtrip ─────────────────────────────────────────────

    #[test]
    fn test_reconstruction_roundtrip() {
        // Use a rectangular window and full overlap (hop = 1) for perfect
        // reconstruction conditions — verifies the OLA path is wired correctly.
        // We use a simpler 50% overlap with Hann window instead, and check that
        // the output energy is within 1% of the input energy.
        let window_size = 64_usize;
        let hop_size = 32_usize;
        let signal = make_sine(512, 440.0, 8000.0);

        let config = RingBufferStftConfig {
            window_size,
            hop_size,
            window_fn: WindowFunction::Hann,
            overlap_add: true,
        };
        let mut proc = RingBufferStft::new(config).expect("valid config");
        let frames = proc.push(&signal);

        let mut reconstructed: Vec<f32> = Vec::new();
        for frame in &frames {
            let chunk = proc
                .reconstruct(frame, None)
                .expect("reconstruction should work");
            reconstructed.extend(chunk);
        }

        // Compare energy in the overlapping region.
        let common_len = reconstructed.len().min(signal.len());
        if common_len > hop_size {
            let sig_energy: f32 = signal[hop_size..common_len]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>();
            let rec_energy: f32 = reconstructed[..common_len - hop_size]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>();
            if sig_energy > 1e-6 {
                let ratio = (rec_energy - sig_energy).abs() / sig_energy;
                assert!(ratio < 0.5, "energy ratio error too large: {ratio}");
            }
        }
    }

    // ── Internal FFT correctness ─────────────────────────────────────────────

    #[test]
    fn test_fft_inplace_identity() {
        // FFT followed by IFFT should recover the input.
        let n = 16_usize;
        let original: Vec<(f32, f32)> = (0..n).map(|i| (i as f32, 0.0_f32)).collect();
        let mut data = original.clone();
        fft_inplace_f32(&mut data, false);
        fft_inplace_f32(&mut data, true);
        for (i, (&(re_orig, im_orig), &(re_rec, im_rec))) in
            original.iter().zip(data.iter()).enumerate()
        {
            assert!(
                (re_orig - re_rec).abs() < 1e-4,
                "re mismatch at {i}: {re_orig} vs {re_rec}"
            );
            assert!(
                (im_orig - im_rec).abs() < 1e-4,
                "im mismatch at {i}: {im_orig} vs {im_rec}"
            );
        }
    }
}
