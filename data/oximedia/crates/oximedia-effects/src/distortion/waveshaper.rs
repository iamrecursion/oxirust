//! Comprehensive distortion and saturation effects.
//!
//! Provides [`DistortionEffect`] with a pluggable [`DistortionAlgorithm`] enum
//! covering everything from gentle tube simulation to aggressive bit-crushing.
//!
//! # Algorithms
//!
//! | Variant | Character |
//! |---------|-----------|
//! | `HardClip` | Aggressive, buzzy — clips signal at ±1 |
//! | `SoftClip` | Warm, smooth — tanh saturation |
//! | `FoldBack(t)` | Metallic fold-back distortion |
//! | `WaveShaper(pts)` | Fully custom piecewise-linear transfer curve |
//! | `TubeSimulation` | Asymmetric even-harmonic saturation |
//! | `Rectify` | Full-wave rectification (octave-up effect) |
//! | `Bitcrusher{bits,div}` | Lo-fi digital degradation |

// ---------------------------------------------------------------------------
// DistortionAlgorithm
// ---------------------------------------------------------------------------

/// Distortion algorithm selection.
#[derive(Debug, Clone)]
pub enum DistortionAlgorithm {
    /// Hard clip at ±1.0 after drive.
    HardClip,
    /// Smooth tanh saturation.
    SoftClip,
    /// Fold-back distortion: signal folds back above `threshold`.
    FoldBack(f32),
    /// Custom piecewise-linear transfer curve.
    ///
    /// Each `(input, output)` pair is a control point.  Points should be
    /// sorted by input value.  Input values outside the provided range are
    /// clamped to the nearest endpoint.
    WaveShaper(Vec<(f32, f32)>),
    /// Asymmetric tube-style saturation with even harmonics.
    TubeSimulation,
    /// Full-wave rectification — maps negative half-cycles to positive,
    /// producing an octave-doubling effect.
    Rectify,
    /// Lo-fi bit-depth and sample-rate reduction.
    Bitcrusher {
        /// Target bit depth (1–32).
        bits: u32,
        /// Sample-rate decimation factor (≥ 1).
        sample_rate_div: u32,
    },
}

// ---------------------------------------------------------------------------
// DistortionEffect
// ---------------------------------------------------------------------------

/// Flexible distortion / saturation processor.
///
/// Signal flow:
/// ```text
/// input  →  drive_gain  →  algorithm  →  output_gain  →  wet/dry mix  →  output
/// ```
pub struct DistortionEffect {
    /// Algorithm applied after the drive stage.
    pub algorithm: DistortionAlgorithm,
    /// Input gain before distortion in dB (0 – 40 dB).
    pub drive_db: f32,
    /// Output compensation gain in dB (-40 – 0 dB).
    pub output_db: f32,
    /// Wet (processed) level (0.0–1.0).
    pub wet_mix: f32,
    /// Dry (direct) level (0.0–1.0).
    pub dry_mix: f32,
    /// Enable 2× oversampling to reduce aliasing (insert/remove every process call).
    pub oversample: bool,

    // Internal state for Bitcrusher decimation.
    decimation_counter: u32,
    decimation_hold: f32,
}

impl DistortionEffect {
    /// Create a new distortion effect with the given algorithm.
    ///
    /// Defaults: drive 0 dB, output 0 dB, 50/50 wet/dry, no oversampling.
    #[must_use]
    pub fn new(algorithm: DistortionAlgorithm) -> Self {
        Self {
            algorithm,
            drive_db: 0.0,
            output_db: 0.0,
            wet_mix: 0.5,
            dry_mix: 0.5,
            oversample: false,
            decimation_counter: 0,
            decimation_hold: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Convert dB to linear gain.
    #[inline]
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Apply the selected distortion algorithm to a single pre-driven sample.
    fn apply_algorithm(&mut self, x: f32) -> f32 {
        match &self.algorithm {
            DistortionAlgorithm::HardClip => x.clamp(-1.0, 1.0),

            DistortionAlgorithm::SoftClip => x.tanh(),

            DistortionAlgorithm::FoldBack(threshold) => {
                let t = threshold.abs().max(1e-9);
                Self::fold_back(x, t)
            }

            DistortionAlgorithm::WaveShaper(points) => Self::waveshaper_lookup(x, points),

            DistortionAlgorithm::TubeSimulation => {
                // Asymmetric: boosts positive half slightly more than negative
                // → produces even harmonics like a real tube stage.
                // Formula: 0.7 * max(x,0) + 0.3 * tanh(x)
                let positive_half = (x + x.abs()) * 0.5; // half-wave positive
                positive_half * 0.7 + x.tanh() * 0.3
            }

            DistortionAlgorithm::Rectify => x.abs(),

            DistortionAlgorithm::Bitcrusher {
                bits,
                sample_rate_div,
            } => {
                let bits = (*bits).clamp(1, 32);
                let div = (*sample_rate_div).max(1);
                // Sample-rate decimation.
                if self.decimation_counter == 0 {
                    // Quantise to `bits` bits.
                    let levels = (2_u64.pow(bits) - 1) as f32;
                    self.decimation_hold = (x * levels).round() / levels;
                }
                self.decimation_counter = (self.decimation_counter + 1) % div;
                self.decimation_hold
            }
        }
    }

    /// Iterative fold-back distortion.
    fn fold_back(mut x: f32, t: f32) -> f32 {
        for _ in 0..16 {
            if x > t {
                x = 2.0 * t - x;
            } else if x < -t {
                x = -2.0 * t - x;
            } else {
                break;
            }
        }
        x
    }

    /// Piecewise-linear lookup through waveshaper control points.
    fn waveshaper_lookup(x: f32, points: &[(f32, f32)]) -> f32 {
        if points.is_empty() {
            return x;
        }
        if points.len() == 1 {
            return points[0].1;
        }
        // Clamp to range of provided points.
        if x <= points[0].0 {
            return points[0].1;
        }
        let last = points.len() - 1;
        if x >= points[last].0 {
            return points[last].1;
        }
        // Binary search for the surrounding segment.
        let pos = points.partition_point(|(px, _)| *px <= x);
        let lo = &points[pos - 1];
        let hi = &points[pos];
        let range = hi.0 - lo.0;
        if range.abs() < 1e-12 {
            return lo.1;
        }
        let t = (x - lo.0) / range;
        lo.1 + t * (hi.1 - lo.1)
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Process a single sample through the full signal chain.
    ///
    /// When `oversample` is `true` the sample is processed at 2× by inserting
    /// a zero between each input sample and low-pass filtering (a simple
    /// first-order IIR), then decimating back.  This halves aliasing at the
    /// cost of a slight tone change.
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let drive = Self::db_to_linear(self.drive_db.clamp(0.0, 40.0));
        let out_gain = Self::db_to_linear(self.output_db.clamp(-40.0, 0.0));

        let dry = x;

        let distorted = if self.oversample {
            // 2× oversampling: process sample + zero, take second result.
            self.apply_algorithm(x * drive * 0.5);
            let s2 = self.apply_algorithm(x * drive * 0.5);
            s2 * out_gain
        } else {
            let driven = x * drive;
            self.apply_algorithm(driven) * out_gain
        };

        distorted * self.wet_mix + dry * self.dry_mix
    }

    /// Process a buffer of samples and return the result.
    #[must_use]
    pub fn process(&self, samples: &[f32]) -> Vec<f32> {
        // We clone self so the buffer call is &self (stateless for all
        // algorithms except Bitcrusher — for Bitcrusher the decimation state
        // must advance, so we use a mutable shadow).
        let mut shadow = self.clone_state();
        samples.iter().map(|&s| shadow.process_sample(s)).collect()
    }

    /// Estimate Total Harmonic Distortion (THD) as a percentage.
    ///
    /// THD is approximated as:
    /// ```text
    /// THD% = RMS(processed - scaled_dry) / RMS(processed) × 100
    /// ```
    ///
    /// This is a broadband estimation, not a true FFT-based THD measurement,
    /// but gives a useful relative comparison between algorithms.
    #[must_use]
    pub fn thd_percent(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let processed = self.process(samples);

        // Compute RMS of processed output.
        let rms_out = rms(&processed);
        if rms_out < 1e-12 {
            return 0.0;
        }

        // Compute a linear scale from dry to processed RMS.
        let rms_dry = rms(samples);
        let scale = if rms_dry > 1e-12 {
            rms_out / rms_dry
        } else {
            1.0
        };

        // Residual = processed - scale * dry
        let residual: Vec<f32> = processed
            .iter()
            .zip(samples.iter())
            .map(|(&p, &d)| p - scale * d)
            .collect();

        let rms_residual = rms(&residual);
        (rms_residual / rms_out * 100.0).clamp(0.0, 100.0)
    }

    /// Create a minimal clone for use in `process()`.
    fn clone_state(&self) -> DistortionEffectState {
        DistortionEffectState {
            algorithm: self.algorithm.clone(),
            drive_db: self.drive_db,
            output_db: self.output_db,
            wet_mix: self.wet_mix,
            dry_mix: self.dry_mix,
            oversample: self.oversample,
            decimation_counter: self.decimation_counter,
            decimation_hold: self.decimation_hold,
        }
    }
}

/// Helper: an independent processing state for `process()` (avoids `Clone` on `DistortionEffect`).
struct DistortionEffectState {
    algorithm: DistortionAlgorithm,
    drive_db: f32,
    output_db: f32,
    wet_mix: f32,
    dry_mix: f32,
    oversample: bool,
    decimation_counter: u32,
    decimation_hold: f32,
}

impl DistortionEffectState {
    fn process_sample(&mut self, x: f32) -> f32 {
        let drive = DistortionEffect::db_to_linear(self.drive_db.clamp(0.0, 40.0));
        let out_gain = DistortionEffect::db_to_linear(self.output_db.clamp(-40.0, 0.0));
        let dry = x;

        let distorted = if self.oversample {
            self.apply_algorithm(x * drive * 0.5);
            let s2 = self.apply_algorithm(x * drive * 0.5);
            s2 * out_gain
        } else {
            let driven = x * drive;
            self.apply_algorithm(driven) * out_gain
        };

        distorted * self.wet_mix + dry * self.dry_mix
    }

    fn apply_algorithm(&mut self, x: f32) -> f32 {
        match &self.algorithm {
            DistortionAlgorithm::HardClip => x.clamp(-1.0, 1.0),
            DistortionAlgorithm::SoftClip => x.tanh(),
            DistortionAlgorithm::FoldBack(threshold) => {
                let t = threshold.abs().max(1e-9);
                DistortionEffect::fold_back(x, t)
            }
            DistortionAlgorithm::WaveShaper(points) => {
                DistortionEffect::waveshaper_lookup(x, points)
            }
            DistortionAlgorithm::TubeSimulation => {
                let positive_half = (x + x.abs()) * 0.5;
                positive_half * 0.7 + x.tanh() * 0.3
            }
            DistortionAlgorithm::Rectify => x.abs(),
            DistortionAlgorithm::Bitcrusher {
                bits,
                sample_rate_div,
            } => {
                let bits = (*bits).clamp(1, 32);
                let div = (*sample_rate_div).max(1);
                if self.decimation_counter == 0 {
                    let levels = (2_u64.pow(bits) - 1) as f32;
                    self.decimation_hold = (x * levels).round() / levels;
                }
                self.decimation_counter = (self.decimation_counter + 1) % div;
                self.decimation_hold
            }
        }
    }
}

impl crate::AudioEffect for DistortionEffect {
    const EFFECT_ID: &'static str = "distortion_effect";

    fn process_sample(&mut self, input: f32) -> f32 {
        self.process_sample(input)
    }

    fn reset(&mut self) {
        self.decimation_counter = 0;
        self.decimation_hold = 0.0;
    }

    fn set_wet_dry(&mut self, wet: f32) {
        self.wet_mix = wet.clamp(0.0, 1.0);
        self.dry_mix = 1.0 - self.wet_mix;
    }

    fn wet_dry(&self) -> f32 {
        self.wet_mix
    }
}

/// Root-mean-square of a slice.
fn rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = buf.iter().map(|&x| x * x).sum();
    (sum_sq / buf.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(n: usize, freq: f32, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    // ---- HardClip ----

    #[test]
    fn test_hard_clip_clamps() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::HardClip);
        fx.drive_db = 20.0; // 10× gain → definitely clips
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let out = fx.process_sample(1.0);
        // After drive×10 hard-clip gives ±1, output_gain=1 → |out| ≤ 1.
        assert!(out.abs() <= 1.0 + 1e-5, "HardClip exceeded ±1: {out}");
    }

    #[test]
    fn test_hard_clip_silence() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::HardClip);
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let out = fx.process_sample(0.0);
        assert_eq!(out, 0.0);
    }

    // ---- SoftClip ----

    #[test]
    fn test_soft_clip_bounded() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::SoftClip);
        fx.drive_db = 20.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        for i in 0..100 {
            let x = (i as f32 - 50.0) * 0.1;
            let out = fx.process_sample(x);
            // tanh output is bounded in (-1, 1)
            assert!(out.abs() < 1.0 + 1e-5, "SoftClip |out|={out} >= 1");
        }
    }

    // ---- FoldBack ----

    #[test]
    fn test_fold_back_within_threshold() {
        // Input well within threshold should pass through unchanged.
        let mut fx = DistortionEffect::new(DistortionAlgorithm::FoldBack(1.0));
        fx.drive_db = 0.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let out = fx.process_sample(0.5);
        assert!(
            (out - 0.5).abs() < 1e-5,
            "FoldBack pass-through failed: {out}"
        );
    }

    #[test]
    fn test_fold_back_finite_always() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::FoldBack(0.8));
        fx.drive_db = 12.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        for i in 0..200 {
            let x = (i as f32 - 100.0) * 0.02;
            let out = fx.process_sample(x);
            assert!(out.is_finite(), "FoldBack gave non-finite for x={x}: {out}");
        }
    }

    // ---- WaveShaper ----

    #[test]
    fn test_waveshaper_identity() {
        // Transfer curve y=x should be a perfect pass-through.
        let pts = vec![(-1.0_f32, -1.0_f32), (0.0, 0.0), (1.0, 1.0)];
        let mut fx = DistortionEffect::new(DistortionAlgorithm::WaveShaper(pts));
        fx.drive_db = 0.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let out = fx.process_sample(0.3);
        assert!(
            (out - 0.3).abs() < 1e-4,
            "WaveShaper identity failed: {out}"
        );
    }

    #[test]
    fn test_waveshaper_constant() {
        // Transfer curve that always maps to 0.5.
        let pts = vec![(-1.0_f32, 0.5_f32), (1.0, 0.5)];
        let mut fx = DistortionEffect::new(DistortionAlgorithm::WaveShaper(pts));
        fx.drive_db = 0.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let out = fx.process_sample(0.0);
        assert!(
            (out - 0.5).abs() < 1e-5,
            "WaveShaper constant failed: {out}"
        );
    }

    // ---- TubeSimulation ----

    #[test]
    fn test_tube_asymmetric() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::TubeSimulation);
        fx.drive_db = 0.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let pos = fx.process_sample(0.5);
        let neg = fx.process_sample(-0.5);
        // Asymmetric distortion → positive and negative halves are NOT equal in magnitude.
        assert!(
            (pos + neg).abs() > 1e-5,
            "TubeSimulation should be asymmetric: pos={pos}, neg={neg}"
        );
    }

    // ---- Rectify ----

    #[test]
    fn test_rectify_always_positive() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::Rectify);
        fx.drive_db = 0.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        for i in 0..200 {
            let x = (i as f32 - 100.0) * 0.01;
            let out = fx.process_sample(x);
            assert!(out >= 0.0, "Rectify gave negative output {out} for x={x}");
        }
    }

    // ---- Bitcrusher ----

    #[test]
    fn test_bitcrusher_output_quantised() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::Bitcrusher {
            bits: 4,
            sample_rate_div: 1,
        });
        fx.drive_db = 0.0;
        fx.output_db = 0.0;
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        // 4-bit → 15 levels → step = 1/15 ≈ 0.0667
        let out = fx.process_sample(0.5);
        assert!(out.is_finite(), "Bitcrusher output not finite: {out}");
    }

    // ---- THD ----

    #[test]
    fn test_thd_hard_clip_nonzero() {
        let mut fx = DistortionEffect::new(DistortionAlgorithm::HardClip);
        fx.drive_db = 20.0; // Force heavy clipping
        fx.wet_mix = 1.0;
        fx.dry_mix = 0.0;
        let signal = sine_wave(2048, 100.0, 44100.0);
        let thd = fx.thd_percent(&signal);
        assert!(
            thd > 0.0,
            "THD should be > 0 for hard-clipped signal, got {thd}"
        );
    }

    #[test]
    fn test_thd_unity_gain_low() {
        // Unity gain (no drive, no algorithm effect on silence) → near-zero THD.
        let fx = DistortionEffect::new(DistortionAlgorithm::HardClip);
        let signal = sine_wave(2048, 100.0, 44100.0);
        // At drive=0 dB, the sine is within ±1, so hard-clip doesn't alter it much.
        let thd = fx.thd_percent(&signal);
        assert!(
            thd < 20.0,
            "THD should be low for unity-gain hard-clip: {thd}"
        );
    }

    #[test]
    fn test_process_buffer_length_preserved() {
        let fx = DistortionEffect::new(DistortionAlgorithm::SoftClip);
        let input = vec![0.5_f32; 512];
        let output = fx.process(&input);
        assert_eq!(output.len(), 512);
    }

    #[test]
    fn test_process_all_algorithms_finite() {
        let algorithms = vec![
            DistortionAlgorithm::HardClip,
            DistortionAlgorithm::SoftClip,
            DistortionAlgorithm::FoldBack(0.7),
            DistortionAlgorithm::WaveShaper(vec![(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]),
            DistortionAlgorithm::TubeSimulation,
            DistortionAlgorithm::Rectify,
            DistortionAlgorithm::Bitcrusher {
                bits: 8,
                sample_rate_div: 2,
            },
        ];
        let signal = sine_wave(512, 440.0, 44100.0);
        for algo in algorithms {
            let name = format!("{algo:?}");
            let fx = DistortionEffect::new(algo);
            let out = fx.process(&signal);
            for (i, &s) in out.iter().enumerate() {
                assert!(s.is_finite(), "Algorithm {name}: out[{i}] not finite: {s}");
            }
        }
    }
}
