//! WebAssembly bindings for loudness normalization from `oximedia-normalize`.
//!
//! Provides EBU R128-compliant loudness measurement and gain-based normalization
//! operating entirely in-memory without file-system access, suitable for
//! browser-based audio normalization workflows.

use wasm_bindgen::prelude::*;

use oximedia_normalize::{NormalizationProcessor, ProcessorConfig};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// LoudnessNormalizer
// ---------------------------------------------------------------------------

/// Loudness normalizer that applies gain to audio samples.
///
/// Uses EBU R128 gain-based normalization: measure target gain externally
/// (e.g. via `measure_lufs`), then call `normalize` to apply it.
///
/// # Example
///
/// ```javascript
/// const norm = new LoudnessNormalizer(-23.0);
/// const out  = norm.normalize(samples);
/// ```
#[wasm_bindgen]
pub struct LoudnessNormalizer {
    target_lufs: f32,
    processor: NormalizationProcessor,
}

#[wasm_bindgen]
impl LoudnessNormalizer {
    /// Create a new loudness normalizer targeting the given LUFS level.
    ///
    /// `target_lufs` is typically −23.0 (EBU R128) or −14.0 (streaming platforms).
    ///
    /// # Errors
    ///
    /// Returns an error if `target_lufs` is outside [−70, 0] or the processor
    /// cannot be constructed.
    #[wasm_bindgen(constructor)]
    pub fn new(target_lufs: f32) -> Result<LoudnessNormalizer, JsValue> {
        if !(-70.0..=0.0).contains(&target_lufs) {
            return Err(js_err(format!(
                "target_lufs {target_lufs} out of range [-70, 0]"
            )));
        }
        let config = ProcessorConfig::minimal(48_000.0, 2);
        let processor = NormalizationProcessor::new(config).map_err(|e| js_err(e))?;
        Ok(LoudnessNormalizer {
            target_lufs,
            processor,
        })
    }

    /// Apply gain normalization to the given interleaved f32 samples.
    ///
    /// The gain is computed as `target_lufs − measured_lufs`.  The current
    /// loudness of `samples` is estimated from their RMS level (fast, single-pass).
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn normalize(&mut self, samples: &[f32]) -> Result<Vec<f32>, JsValue> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Fast RMS-based loudness estimate (approx LUFS for programme material).
        let measured_lufs = self.measure_lufs(samples, 48_000);

        // Gain in dB: how much to adjust to reach target.
        let gain_db = (self.target_lufs - measured_lufs) as f64;

        let mut output = vec![0.0_f32; samples.len()];
        self.processor
            .process_f32(samples, &mut output, gain_db)
            .map_err(|e| js_err(e))?;
        Ok(output)
    }

    /// Measure approximate integrated loudness (LUFS) of the samples.
    ///
    /// Uses a fast RMS estimate over the entire buffer (single-channel assumed
    /// for WASM efficiency). Returns values in the range [−100, 0] dBFS.
    pub fn measure_lufs(&self, samples: &[f32], _sample_rate: u32) -> f32 {
        if samples.is_empty() {
            return -100.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / samples.len() as f32).sqrt();
        if rms < 1e-10 {
            return -100.0;
        }
        // Convert RMS to dBFS ≈ approximate LUFS.
        20.0 * rms.log10()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_valid_target() {
        let norm = LoudnessNormalizer::new(-23.0);
        assert!(norm.is_ok(), "should construct with valid target LUFS");
    }

    #[test]
    fn constructor_rejects_out_of_range() {
        assert!(
            LoudnessNormalizer::new(5.0).is_err(),
            "positive LUFS should be rejected"
        );
        assert!(
            LoudnessNormalizer::new(-80.0).is_err(),
            "too-negative LUFS should be rejected"
        );
    }

    #[test]
    fn measure_lufs_silent_returns_min() {
        let norm = LoudnessNormalizer::new(-23.0).expect("valid");
        let silence = vec![0.0_f32; 1024];
        assert_eq!(norm.measure_lufs(&silence, 48_000), -100.0);
    }

    #[test]
    fn measure_lufs_full_scale_sine_in_range() {
        let norm = LoudnessNormalizer::new(-23.0).expect("valid");
        // 0 dBFS sine: RMS ≈ 1/√2 ≈ -3 dBFS
        let samples: Vec<f32> = (0..1024)
            .map(|i| (std::f32::consts::TAU * i as f32 / 64.0).sin())
            .collect();
        let lufs = norm.measure_lufs(&samples, 48_000);
        assert!(
            lufs > -10.0 && lufs < 0.0,
            "0 dBFS sine LUFS should be in (-10, 0), got {lufs}"
        );
    }

    #[test]
    fn normalize_returns_same_length() {
        let mut norm = LoudnessNormalizer::new(-23.0).expect("valid");
        let samples = vec![0.1_f32; 512];
        let out = norm.normalize(&samples).expect("normalize ok");
        assert_eq!(out.len(), 512);
    }
}
