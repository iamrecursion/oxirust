//! Spectral audio repair utilities.
//!
//! Provides gap detection, interpolation, spectral subtraction, and harmonic inpainting
//! for damaged or noisy audio spectra.

use oxifft::Complex;
use std::collections::HashMap;

/// A gap (missing or corrupted region) in a frequency spectrum.
#[derive(Debug, Clone)]
pub struct SpectralGap {
    /// First bin of the gap (inclusive).
    pub start_bin: usize,
    /// Last bin of the gap (inclusive).
    pub end_bin: usize,
    /// Index of the frame this gap belongs to.
    pub frame_idx: usize,
    /// Average magnitude within the gap.
    pub magnitude: f32,
}

impl SpectralGap {
    /// Width of the gap in bins.
    #[must_use]
    pub fn width(&self) -> usize {
        if self.end_bin >= self.start_bin {
            self.end_bin - self.start_bin + 1
        } else {
            0
        }
    }
}

/// Detect spectral bins whose magnitude falls below `threshold`.
///
/// Returns a list of contiguous gaps found in the spectrum.
#[must_use]
pub fn detect_spectral_gaps(spectrum: &[f32], threshold: f32) -> Vec<SpectralGap> {
    let mut gaps = Vec::new();
    let mut in_gap = false;
    let mut gap_start = 0usize;
    let mut mag_acc = 0.0_f32;
    let mut mag_count = 0usize;

    for (i, &val) in spectrum.iter().enumerate() {
        let below = val < threshold;
        match (in_gap, below) {
            (false, true) => {
                in_gap = true;
                gap_start = i;
                mag_acc = val;
                mag_count = 1;
            }
            (true, true) => {
                mag_acc += val;
                mag_count += 1;
            }
            (true, false) => {
                let magnitude = if mag_count > 0 {
                    mag_acc / mag_count as f32
                } else {
                    0.0
                };
                gaps.push(SpectralGap {
                    start_bin: gap_start,
                    end_bin: i - 1,
                    frame_idx: 0,
                    magnitude,
                });
                in_gap = false;
                mag_acc = 0.0;
                mag_count = 0;
            }
            _ => {}
        }
    }
    if in_gap {
        let magnitude = if mag_count > 0 {
            mag_acc / mag_count as f32
        } else {
            0.0
        };
        gaps.push(SpectralGap {
            start_bin: gap_start,
            end_bin: spectrum.len() - 1,
            frame_idx: 0,
            magnitude,
        });
    }
    gaps
}

/// Fill a spectral gap using linear interpolation between the boundary bins.
#[allow(clippy::cast_precision_loss)]
pub fn interpolate_spectral_gap(spectrum: &mut [f32], gap: &SpectralGap) {
    if spectrum.is_empty() || gap.start_bin > gap.end_bin || gap.end_bin >= spectrum.len() {
        return;
    }
    let left_val = if gap.start_bin > 0 {
        spectrum[gap.start_bin - 1]
    } else {
        0.0
    };
    let right_val = if gap.end_bin + 1 < spectrum.len() {
        spectrum[gap.end_bin + 1]
    } else {
        left_val
    };

    let count = (gap.end_bin - gap.start_bin + 1) as f32;
    for (idx, bin) in (gap.start_bin..=gap.end_bin).enumerate() {
        let t = if count <= 1.0 {
            0.5
        } else {
            (idx as f32 + 1.0) / (count + 1.0)
        };
        spectrum[bin] = left_val + t * (right_val - left_val);
    }
}

/// Spectral subtraction noise reducer.
///
/// Maintains a per-block-size FFT scratch buffer cache to avoid per-call
/// heap allocation when the same block size is used across consecutive calls.
#[derive(Debug, Clone)]
pub struct SpectralSubtractor {
    /// Estimated noise magnitude spectrum.
    pub noise_profile: Vec<f32>,
    /// Over-subtraction factor (alpha ≥ 1.0).
    pub alpha: f32,
    /// Spectral floor (beta ≥ 0.0).
    pub beta: f32,
    /// Per-block-size scratch buffers (complex): keyed by block length.
    ///
    /// Each entry is a `Vec<Complex<f32>>` pre-allocated to the given size so
    /// that repeated calls with the same block size reuse the allocation via
    /// `.resize()` rather than allocating fresh on every invocation.
    fft_scratch: HashMap<usize, Vec<Complex<f32>>>,
    /// Most-recently used block size — fast path avoids a HashMap lookup on
    /// the common case where the caller always passes the same size.
    block_size_cache: usize,
}

impl SpectralSubtractor {
    /// Create a new subtractor.
    #[must_use]
    pub fn new(noise_profile: Vec<f32>, alpha: f32, beta: f32) -> Self {
        Self {
            noise_profile,
            alpha,
            beta,
            fft_scratch: HashMap::new(),
            block_size_cache: 0,
        }
    }

    /// Obtain a mutable reference to the scratch buffer for `block_size`.
    ///
    /// If the buffer doesn't exist yet it is allocated and inserted; if it
    /// exists but is too small it is grown via `resize` (no-copy extension).
    /// This avoids per-call `Vec::new()` / `Vec::with_capacity` allocation
    /// for callers that repeatedly pass the same block size.
    fn scratch_for_size(&mut self, block_size: usize) -> &mut Vec<Complex<f32>> {
        // Fast path: last-used size matches → entry already guaranteed present.
        if self.block_size_cache != block_size {
            self.fft_scratch
                .entry(block_size)
                .or_insert_with(|| vec![Complex::new(0.0, 0.0); block_size]);
            self.block_size_cache = block_size;
        }
        // SAFETY: entry was inserted just above if missing.
        self.fft_scratch
            .get_mut(&block_size)
            .expect("scratch entry guaranteed by entry().or_insert_with above")
    }

    /// Apply spectral subtraction on a real input block.
    ///
    /// The intermediate complex buffer is drawn from the per-size scratch cache
    /// so no heap allocation occurs on the hot path when the caller consistently
    /// passes the same `block` length.
    ///
    /// # Returns
    ///
    /// Enhanced magnitude spectrum (`block.len()` floats).
    pub fn process_block(&mut self, block: &[f32]) -> Vec<f32> {
        let n = block.len();
        // Ensure scratch buffer is the right size (possibly a no-op).
        let scratch = self.scratch_for_size(n);
        scratch.resize(n, Complex::new(0.0, 0.0));
        // Fill scratch with real-valued complex input (re-use allocation).
        for (c, &s) in scratch.iter_mut().zip(block.iter()) {
            c.re = s;
            c.im = 0.0;
        }

        // Forward FFT — oxifft::fft allocates its output; we can't avoid that
        // allocation inside oxifft itself, but we avoid the *input* allocation.
        let spectrum = oxifft::fft(scratch);

        // Spectral subtraction on the magnitude spectrum.
        spectrum
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mag = c.norm();
                let noise = if i < self.noise_profile.len() {
                    self.noise_profile[i]
                } else {
                    0.0
                };
                let subtracted = mag - self.alpha * noise;
                subtracted.max(self.beta * noise)
            })
            .collect()
    }

    /// Estimate a noise profile as the average magnitude of the first `max_frames` frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_noise(frames: &[Vec<f32>]) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }
        let max_frames = frames.len().min(10);
        let n_bins = frames[0].len();
        if n_bins == 0 {
            return Vec::new();
        }

        let mut profile = vec![0.0_f32; n_bins];
        for frame in &frames[..max_frames] {
            for (bin, &val) in frame.iter().enumerate().take(n_bins) {
                profile[bin] += val;
            }
        }
        let count = max_frames as f32;
        profile.iter_mut().for_each(|v| *v /= count);
        profile
    }

    /// Apply modified spectral subtraction to a magnitude spectrum.
    ///
    /// Returns the enhanced spectrum with noise removed.
    #[must_use]
    pub fn subtract(&self, spectrum: &[f32]) -> Vec<f32> {
        spectrum
            .iter()
            .enumerate()
            .map(|(i, &mag)| {
                let noise = if i < self.noise_profile.len() {
                    self.noise_profile[i]
                } else {
                    0.0
                };
                let subtracted = mag - self.alpha * noise;
                subtracted.max(self.beta * noise)
            })
            .collect()
    }
}

/// Harmonic inpainting for restoring missing overtones.
pub struct HarmonicInpainter;

impl HarmonicInpainter {
    /// Inpaint a missing harmonic by interpolating from adjacent overtones.
    ///
    /// `fundamental` is in bins, `missing_harmonic` is the harmonic number (1 = fundamental).
    /// The returned spectrum has the missing harmonic bin filled in.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn inpaint(spectrum: &[f32], fundamental: f32, missing_harmonic: u32) -> Vec<f32> {
        let mut out = spectrum.to_vec();
        if spectrum.is_empty() || fundamental <= 0.0 || missing_harmonic == 0 {
            return out;
        }
        let target_bin = (fundamental * missing_harmonic as f32).round() as usize;
        if target_bin >= spectrum.len() {
            return out;
        }

        // Use the harmonic below and above to estimate the missing one.
        let h_below = if missing_harmonic > 1 {
            let bin = (fundamental * (missing_harmonic - 1) as f32).round() as usize;
            if bin < spectrum.len() {
                spectrum[bin]
            } else {
                0.0
            }
        } else {
            0.0
        };

        let h_above_idx = missing_harmonic + 1;
        let h_above = {
            let bin = (fundamental * h_above_idx as f32).round() as usize;
            if bin < spectrum.len() {
                spectrum[bin]
            } else {
                0.0
            }
        };

        out[target_bin] = (h_below + h_above) / 2.0;
        out
    }
}

/// A frequency band mask for spectral repair operations.
#[allow(dead_code)]
pub struct FrequencyMask {
    /// Lower boundary of the band in Hz (inclusive).
    pub start_hz: f64,
    /// Upper boundary of the band in Hz (inclusive).
    pub end_hz: f64,
    /// Attenuation to apply to this band in dB (positive = cut).
    pub attenuation_db: f64,
}

impl FrequencyMask {
    /// Create a new mask.
    #[must_use]
    pub fn new(start_hz: f64, end_hz: f64, attenuation_db: f64) -> Self {
        Self {
            start_hz,
            end_hz,
            attenuation_db,
        }
    }

    /// Return `true` if `hz` falls within the mask's frequency range.
    #[must_use]
    pub fn contains_freq(&self, hz: f64) -> bool {
        hz >= self.start_hz && hz <= self.end_hz
    }
}

/// Configuration for the spectral repair process.
#[allow(dead_code)]
pub struct SpectralRepairConfig {
    /// Frequency masks to apply before interpolation.
    pub masks: Vec<FrequencyMask>,
    /// Width of the interpolation crossfade region in Hz.
    pub interpolation_width_hz: f64,
    /// Number of repair iterations.
    pub iterations: u32,
}

/// Convert a dB value to a linear amplitude ratio.
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Detect runs of bins below a threshold (legacy API, returns `(start, end)` pairs).
#[must_use]
pub fn detect_spectral_holes(spectrum: &[f64], threshold_db: f64) -> Vec<(usize, usize)> {
    if spectrum.is_empty() {
        return Vec::new();
    }
    let peak = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let threshold_linear = peak * db_to_linear(-threshold_db.abs());
    let mut holes = Vec::new();
    let mut in_hole = false;
    let mut hole_start = 0usize;
    for (i, &val) in spectrum.iter().enumerate() {
        let is_hole = val < threshold_linear;
        match (in_hole, is_hole) {
            (false, true) => {
                in_hole = true;
                hole_start = i;
            }
            (true, false) => {
                holes.push((hole_start, i - 1));
                in_hole = false;
            }
            _ => {}
        }
    }
    if in_hole {
        holes.push((hole_start, spectrum.len() - 1));
    }
    holes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_gap_width_single_bin() {
        let g = SpectralGap {
            start_bin: 5,
            end_bin: 5,
            frame_idx: 0,
            magnitude: 0.0,
        };
        assert_eq!(g.width(), 1);
    }

    #[test]
    fn test_spectral_gap_width_multi_bin() {
        let g = SpectralGap {
            start_bin: 3,
            end_bin: 7,
            frame_idx: 0,
            magnitude: 0.0,
        };
        assert_eq!(g.width(), 5);
    }

    #[test]
    fn test_detect_spectral_gaps_empty() {
        let gaps = detect_spectral_gaps(&[], 0.1);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_spectral_gaps_no_gap() {
        let spectrum = vec![1.0_f32; 16];
        let gaps = detect_spectral_gaps(&spectrum, 0.1);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_spectral_gaps_one_gap() {
        let mut spectrum = vec![1.0_f32; 16];
        // Bins 5-8 are below threshold
        for i in 5..=8 {
            spectrum[i] = 0.0;
        }
        let gaps = detect_spectral_gaps(&spectrum, 0.5);
        assert_eq!(gaps.len(), 1);
        let g = &gaps[0];
        assert_eq!(g.start_bin, 5);
        assert_eq!(g.end_bin, 8);
    }

    #[test]
    fn test_interpolate_spectral_gap_fills_bins() {
        let mut spectrum = vec![0.0_f32; 10];
        spectrum[2] = 1.0;
        spectrum[6] = 1.0;
        let gap = SpectralGap {
            start_bin: 3,
            end_bin: 5,
            frame_idx: 0,
            magnitude: 0.0,
        };
        interpolate_spectral_gap(&mut spectrum, &gap);
        for i in 3..=5 {
            assert!(
                spectrum[i] >= 0.0 && spectrum[i] <= 1.0,
                "bin {i} = {}",
                spectrum[i]
            );
        }
    }

    #[test]
    fn test_interpolate_spectral_gap_no_panic_on_edges() {
        let mut spectrum = vec![1.0_f32; 5];
        let gap = SpectralGap {
            start_bin: 0,
            end_bin: 1,
            frame_idx: 0,
            magnitude: 0.0,
        };
        interpolate_spectral_gap(&mut spectrum, &gap);
        for v in &spectrum {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_spectral_subtractor_subtract_reduces_noise() {
        let noise = vec![0.5_f32; 8];
        let sub = SpectralSubtractor::new(noise.clone(), 1.0, 0.01);
        let spectrum = vec![1.0_f32; 8];
        let out = sub.subtract(&spectrum);
        for (i, &v) in out.iter().enumerate() {
            assert!(v <= spectrum[i], "bin {i}: {v} > {}", spectrum[i]);
        }
    }

    #[test]
    fn test_spectral_subtractor_floor() {
        let noise = vec![1.0_f32; 4];
        let sub = SpectralSubtractor::new(noise, 2.0, 0.1);
        let spectrum = vec![0.5_f32; 4]; // below alpha*noise → should hit floor
        let out = sub.subtract(&spectrum);
        for v in out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_estimate_noise_averages_frames() {
        let frames = vec![vec![2.0_f32; 4], vec![4.0_f32; 4]];
        let profile = SpectralSubtractor::estimate_noise(&frames);
        for v in profile {
            assert!((v - 3.0).abs() < 1e-5, "v={v}");
        }
    }

    #[test]
    fn test_harmonic_inpainter_fills_bin() {
        let mut spectrum = vec![0.0_f32; 20];
        // Harmonics at bins 4 and 12 (fundamental=4, harmonics 1 and 3)
        spectrum[4] = 1.0;
        spectrum[12] = 0.6;
        let out = HarmonicInpainter::inpaint(&spectrum, 4.0, 2); // fill bin 8
                                                                 // Should be average of bins 4 (1.0) and 12 (0.6) = 0.8
        assert!((out[8] - 0.8).abs() < 1e-4, "bin8={}", out[8]);
    }

    #[test]
    fn test_harmonic_inpainter_out_of_range() {
        let spectrum = vec![1.0_f32; 8];
        // Target bin 100 is out of range → return unchanged
        let out = HarmonicInpainter::inpaint(&spectrum, 50.0, 2);
        assert_eq!(out, spectrum);
    }

    #[test]
    fn test_db_to_linear_zero() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_spectral_holes_compat() {
        let mut spec = vec![1.0_f64; 16];
        for i in 4..=6 {
            spec[i] = 1e-10;
        }
        let holes = detect_spectral_holes(&spec, -40.0);
        assert!(!holes.is_empty());
    }

    // -----------------------------------------------------------------------
    // FFT plan cache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spectral_repair_cached_plan_idempotent() {
        use rand::RngExt;

        let mut rng = rand::rng();

        // Build a noise profile from the average magnitude of white noise frames.
        let noise_level = 0.05;
        let block_size = 1024usize;
        let noise_profile = vec![noise_level; block_size];

        let mut subtractor = SpectralSubtractor::new(noise_profile, 1.0, 0.01);

        // Generate two identical 1024-sample white-noise blocks.
        let block_a: Vec<f32> = (0..block_size)
            .map(|_| rng.random_range(-0.5..0.5_f32))
            .collect();
        let block_b = block_a.clone();

        // First call — populates the scratch cache.
        let out_a = subtractor.process_block(&block_a);
        // Second call — reuses the cached scratch buffer.
        let out_b = subtractor.process_block(&block_b);

        // Determinism: identical input must produce identical output.
        assert_eq!(
            out_a.len(),
            block_size,
            "output length must match block size"
        );
        assert_eq!(
            out_a, out_b,
            "consecutive calls with identical input must return identical output"
        );

        // All output values must be finite (no NaN/Inf from corrupted scratch).
        for (i, &v) in out_a.iter().enumerate() {
            assert!(v.is_finite(), "output sample {i} is not finite: {v}");
        }
    }

    #[test]
    fn test_spectral_repair_cached_plan_different_sizes() {
        // Verify the cache handles multiple distinct block sizes without panic.
        let mut subtractor = SpectralSubtractor::new(vec![0.02; 2048], 1.0, 0.01);

        let block_512: Vec<f32> = vec![0.1; 512];
        let block_1024: Vec<f32> = vec![0.1; 1024];
        let block_2048: Vec<f32> = vec![0.1; 2048];

        let o1 = subtractor.process_block(&block_512);
        let o2 = subtractor.process_block(&block_1024);
        let o3 = subtractor.process_block(&block_2048);

        assert_eq!(o1.len(), 512);
        assert_eq!(o2.len(), 1024);
        assert_eq!(o3.len(), 2048);

        // Calling with the first size again should reuse the old scratch entry.
        let o4 = subtractor.process_block(&block_512);
        assert_eq!(o4, o1, "re-using cached scratch must give same result");
    }
}
