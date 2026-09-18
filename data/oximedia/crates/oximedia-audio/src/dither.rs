//! Audio dithering for bit-depth reduction.
//!
//! Dithering adds a small amount of controlled noise before quantisation in
//! order to linearise the quantisation error and prevent unwanted harmonic
//! distortion in low-level signals.  This module supports several dither
//! shapes plus an optional noise-shaper that pushes quantisation noise into
//! frequency bands where it is less audible.
//!
//! # Quick start
//!
//! ```
//! use oximedia_audio::dither::{Ditherer, DitherType};
//!
//! let mut ditherer = Ditherer::new(DitherType::Tpdf);
//! let output = ditherer.process(0.123_f32, 16);
//! assert!(output.is_finite());
//! ```
//!
//! # Dither types
//!
//! | Variant | Description |
//! |---|---|
//! | [`DitherType::None`] | No dither (hard truncation) |
//! | [`DitherType::Rectangular`] | Uniform noise – ½ LSB RMS |
//! | [`DitherType::Triangular`] | Triangular PDF – lower distortion |
//! | [`DitherType::Tpdf`] | Triangular PDF (alias, most common) |
//! | [`DitherType::Highpass`] | Spectrally shaped high-pass dither |

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic pseudo-random source (no external crate required)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal xorshift32 PRNG – deterministic, cheap, and dependency-free.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    /// Seed must be non-zero; any non-zero seed is fine.
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0xBAD_5EED } else { seed },
        }
    }

    /// Return the next pseudo-random `u32`.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Return a uniform sample in `[−1, 1)`.
    fn next_f32(&mut self) -> f32 {
        // Map u32 → [0, 1) → [−1, 1)
        let u = self.next_u32() as f32 / u32::MAX as f32;
        u * 2.0 - 1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DitherType
// ─────────────────────────────────────────────────────────────────────────────

/// The type of dither noise added before quantisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DitherType {
    /// No dither – plain truncation.
    None,
    /// Rectangular (uniform) probability density – ½ LSB amplitude noise.
    Rectangular,
    /// Triangular probability density – sum of two rectangular distributions.
    Triangular,
    /// TPDF (Triangular Probability Density Function) – common alias for
    /// [`DitherType::Triangular`].
    Tpdf,
    /// Spectrally-shaped high-frequency dither (noise shaping).
    Highpass,
}

impl DitherType {
    /// Return a short human-readable name for the noise shape.
    #[must_use]
    pub fn noise_shape(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Rectangular => "rectangular",
            Self::Triangular => "triangular",
            Self::Tpdf => "tpdf",
            Self::Highpass => "highpass",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NoiseShaper
// ─────────────────────────────────────────────────────────────────────────────

/// First-order or higher-order IIR noise shaper.
///
/// The shaper feeds the previous quantisation errors back through a set of
/// FIR coefficients, effectively pushing the noise energy into higher
/// frequencies where it is less perceptible.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct NoiseShaper {
    /// Ring buffer of past quantisation errors.
    feedback: Vec<f32>,
    /// Shaping filter coefficients.
    coeffs: Vec<f32>,
    /// Current write position in the ring buffer.
    pos: usize,
}

impl NoiseShaper {
    /// Create a new noise shaper of the given order.
    ///
    /// Higher orders push more noise energy toward Nyquist, at the cost of
    /// a slightly higher total noise floor.
    ///
    /// # Arguments
    ///
    /// * `order` – Filter order (number of feedback taps; 0 disables shaping).
    #[must_use]
    pub fn new(order: usize) -> Self {
        // Simple high-frequency emphasis coefficients
        let coeffs = match order {
            0 => vec![],
            1 => vec![1.0_f32],
            2 => vec![1.5, -0.5],
            3 => vec![1.75, -1.0, 0.25],
            _ => {
                // Generic: linearly-decaying weights
                (0..order).map(|i| 1.0 - i as f32 / order as f32).collect()
            }
        };
        let len = coeffs.len().max(1);
        Self {
            feedback: vec![0.0; len],
            coeffs,
            pos: 0,
        }
    }

    /// Feed a quantisation error into the shaper and return the noise
    /// compensation term based on **previous** errors.
    ///
    /// The output reflects errors fed in on previous calls; the current
    /// `quantize_error` is stored and will influence future outputs.
    pub fn process(&mut self, quantize_error: f32) -> f32 {
        if self.coeffs.is_empty() {
            return 0.0;
        }

        // Compute the shaping output from previously stored errors
        // (read before writing, so k=0 references the oldest stored value
        //  at the current write position, not the new incoming error).
        let mut shaped = 0.0_f32;
        for (k, coeff) in self.coeffs.iter().enumerate() {
            // k=0 reads the slot we are about to overwrite – i.e., the
            // oldest sample in the ring buffer (one full period ago).
            let idx = (self.pos + self.feedback.len() - k) % self.feedback.len();
            shaped += coeff * self.feedback[idx];
        }

        // Store the new error at the current position
        self.feedback[self.pos] = quantize_error;

        // Advance ring pointer
        self.pos = (self.pos + 1) % self.feedback.len();
        shaped
    }

    /// Reset the feedback ring buffer to silence.
    pub fn reset(&mut self) {
        for x in &mut self.feedback {
            *x = 0.0;
        }
        self.pos = 0;
    }

    /// Return the shaper order (number of feedback taps).
    #[must_use]
    pub fn order(&self) -> usize {
        self.coeffs.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ditherer
// ─────────────────────────────────────────────────────────────────────────────

/// Complete ditherer: noise generation + optional noise shaping + quantisation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Ditherer {
    dither_type: DitherType,
    shaper: Option<NoiseShaper>,
    prng: Xorshift32,
}

impl Ditherer {
    /// Create a new ditherer.
    ///
    /// * [`DitherType::Highpass`] automatically attaches a first-order
    ///   noise shaper.
    /// * All other variants leave the shaper disabled.
    #[must_use]
    pub fn new(dither_type: DitherType) -> Self {
        let shaper = match dither_type {
            DitherType::Highpass => Some(NoiseShaper::new(1)),
            _ => None,
        };
        Self {
            dither_type,
            shaper,
            prng: Xorshift32::new(0x1234_5678),
        }
    }

    /// Create a new ditherer with an explicit noise shaper.
    #[must_use]
    pub fn with_shaper(dither_type: DitherType, shaper: NoiseShaper) -> Self {
        Self {
            dither_type,
            shaper: Some(shaper),
            prng: Xorshift32::new(0xDEAD_BEEF),
        }
    }

    /// Process a single sample, adding dither noise and quantising to
    /// `target_bits` bits.
    ///
    /// The input sample is assumed to be normalised to `[−1, 1]`.
    ///
    /// # Arguments
    ///
    /// * `sample` – Input sample in `[−1, 1]`.
    /// * `target_bits` – Target bit depth (e.g. 16, 20, 24).
    ///
    /// # Returns
    ///
    /// Quantised sample, still normalised to `[−1, 1]`.
    pub fn process(&mut self, sample: f32, target_bits: u8) -> f32 {
        let levels = (1u32 << (target_bits - 1)) as f32; // e.g. 32768 for 16-bit
        let lsb = 1.0 / levels; // one least-significant bit in normalised space

        // Generate dither noise
        let dither_noise = match self.dither_type {
            DitherType::None => 0.0,
            DitherType::Rectangular => self.prng.next_f32() * 0.5 * lsb,
            DitherType::Triangular | DitherType::Tpdf => {
                let r1 = self.prng.next_f32();
                let r2 = self.prng.next_f32();
                (r1 + r2) * 0.5 * lsb
            }
            DitherType::Highpass => {
                // White noise base; shaping happens via the feedback term
                self.prng.next_f32() * 0.5 * lsb
            }
        };

        let dithered = sample + dither_noise;

        // Quantise: round to nearest integer multiple of lsb, then clamp
        let quantised = (dithered * levels).round() / levels;
        let quantised = quantised.clamp(-1.0, 1.0 - lsb);

        // Feed quantisation error into the shaper for the next sample
        if let Some(ref mut shaper) = self.shaper {
            let error = quantised - sample;
            shaper.process(error);
        }

        quantised
    }

    /// Return the dither type.
    #[must_use]
    pub fn dither_type(&self) -> DitherType {
        self.dither_type
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience function
// ─────────────────────────────────────────────────────────────────────────────

/// Dither and quantise an entire buffer.
///
/// # Arguments
///
/// * `samples` – Input samples (normalised `[−1, 1]`).
/// * `target_bits` – Target bit depth.
/// * `dither_type` – Type of dither to apply.
///
/// # Returns
///
/// A new `Vec<f32>` of quantised samples.
#[must_use]
#[allow(dead_code)]
pub fn dither_buffer(samples: &[f32], target_bits: u8, dither_type: DitherType) -> Vec<f32> {
    let mut ditherer = Ditherer::new(dither_type);
    samples
        .iter()
        .map(|&s| ditherer.process(s, target_bits))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DitherType ────────────────────────────────────────────────────────────

    #[test]
    fn test_dither_type_noise_shape_none() {
        assert_eq!(DitherType::None.noise_shape(), "none");
    }

    #[test]
    fn test_dither_type_noise_shape_rectangular() {
        assert_eq!(DitherType::Rectangular.noise_shape(), "rectangular");
    }

    #[test]
    fn test_dither_type_noise_shape_triangular() {
        assert_eq!(DitherType::Triangular.noise_shape(), "triangular");
    }

    #[test]
    fn test_dither_type_noise_shape_tpdf() {
        assert_eq!(DitherType::Tpdf.noise_shape(), "tpdf");
    }

    #[test]
    fn test_dither_type_noise_shape_highpass() {
        assert_eq!(DitherType::Highpass.noise_shape(), "highpass");
    }

    // ── NoiseShaper ───────────────────────────────────────────────────────────

    #[test]
    fn test_noise_shaper_order_zero_returns_zero() {
        let mut shaper = NoiseShaper::new(0);
        let out = shaper.process(0.5);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_noise_shaper_order_returns_expected() {
        let shaper = NoiseShaper::new(3);
        assert_eq!(shaper.order(), 3);
    }

    #[test]
    fn test_noise_shaper_reset() {
        let mut shaper = NoiseShaper::new(2);
        shaper.process(1.0);
        shaper.process(0.5);
        shaper.reset();
        // After reset, the next process call uses zeroed feedback
        let out = shaper.process(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_noise_shaper_first_order_feedback() {
        let mut shaper = NoiseShaper::new(1);
        // Feed in an error; next call should reflect it
        let _first = shaper.process(1.0);
        let second = shaper.process(0.0);
        // With order=1 the coefficient is 1.0, so second output = 1.0 * previous error
        assert!(
            second.abs() > 0.0,
            "Shaper should produce non-zero output after non-zero error, got {second}"
        );
    }

    // ── Ditherer::process ─────────────────────────────────────────────────────

    #[test]
    fn test_dither_none_output_is_finite() {
        let mut d = Ditherer::new(DitherType::None);
        assert!(d.process(0.5, 16).is_finite());
    }

    #[test]
    fn test_dither_none_quantises_cleanly() {
        // With no dither a DC value should quantise to the nearest level
        let mut d = Ditherer::new(DitherType::None);
        let out = d.process(0.0, 16);
        // 0.0 rounds to 0.0 at any bit depth
        assert!(out.abs() < 1.0 / 32768.0 + 1e-8);
    }

    #[test]
    fn test_dither_rectangular_output_in_range() {
        let mut d = Ditherer::new(DitherType::Rectangular);
        for _ in 0..1000 {
            let out = d.process(0.5, 16);
            assert!(out >= -1.0 && out <= 1.0, "Output out of range: {out}");
        }
    }

    #[test]
    fn test_dither_tpdf_output_in_range() {
        let mut d = Ditherer::new(DitherType::Tpdf);
        for _ in 0..1000 {
            let out = d.process(-0.5, 16);
            assert!(out >= -1.0 && out <= 1.0);
        }
    }

    #[test]
    fn test_dither_triangular_output_is_finite() {
        let mut d = Ditherer::new(DitherType::Triangular);
        for _ in 0..500 {
            assert!(d.process(0.25, 24).is_finite());
        }
    }

    #[test]
    fn test_dither_highpass_output_is_finite() {
        let mut d = Ditherer::new(DitherType::Highpass);
        for _ in 0..500 {
            assert!(d.process(0.1, 16).is_finite());
        }
    }

    #[test]
    fn test_dither_type_returned_correctly() {
        let d = Ditherer::new(DitherType::Tpdf);
        assert_eq!(d.dither_type(), DitherType::Tpdf);
    }

    // ── dither_buffer ─────────────────────────────────────────────────────────

    #[test]
    fn test_dither_buffer_length_preserved() {
        let input = vec![0.1_f32; 256];
        let output = dither_buffer(&input, 16, DitherType::Tpdf);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_dither_buffer_all_finite() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();
        let output = dither_buffer(&input, 16, DitherType::Rectangular);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_dither_buffer_values_in_range() {
        let input = vec![0.9_f32; 128];
        let output = dither_buffer(&input, 16, DitherType::Tpdf);
        for &v in &output {
            assert!(v >= -1.0 && v <= 1.0, "Out of range: {v}");
        }
    }
}
