//! Audio processing pipeline with composable stages.
//!
//! This module provides a flexible, chain-based audio pipeline that allows
//! multiple processing stages to be composed in sequence. Each stage implements
//! the [`PipelineStage`] trait, enabling gain control, metering, filtering, and
//! custom DSP to be wired together without allocation in the hot path.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::audio_pipeline::{AudioPipeline, GainStage, MeterStage};
//!
//! let mut pipeline = AudioPipeline::new(48_000.0);
//! pipeline.add_stage(Box::new(GainStage::new(0.5)));
//! pipeline.add_stage(Box::new(MeterStage::new()));
//!
//! let mut buf = vec![1.0_f32; 256];
//! pipeline.process(&mut buf);
//! ```

#![allow(dead_code)]

/// A single stage in an [`AudioPipeline`].
pub trait PipelineStage: Send {
    /// Process a mono buffer in-place.
    fn process(&mut self, buffer: &mut [f32], sample_rate: f32);

    /// Human-readable name for this stage.
    fn name(&self) -> &str;

    /// Reset internal state (e.g. clear filter history).
    fn reset(&mut self);

    /// Returns the latency this stage introduces, in samples.
    fn latency_samples(&self) -> usize {
        0
    }
}

/// Logical stage type tag for introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// Gain / attenuation stage.
    Gain,
    /// Level metering stage (non-destructive).
    Meter,
    /// Biquad IIR filter stage.
    Filter,
    /// Custom / user-defined stage.
    Custom,
}

/// A simple gain (amplitude scaling) stage.
///
/// Multiplies every sample by `gain_linear`. Values above `1.0` amplify;
/// values below attenuate.
pub struct GainStage {
    /// Linear gain factor.
    pub gain_linear: f32,
}

impl GainStage {
    /// Create a new gain stage with the given linear gain.
    #[must_use]
    pub fn new(gain_linear: f32) -> Self {
        Self { gain_linear }
    }

    /// Create a gain stage from a dB value.
    #[must_use]
    pub fn from_db(db: f32) -> Self {
        let linear = 10.0_f32.powf(db / 20.0);
        Self {
            gain_linear: linear,
        }
    }
}

impl PipelineStage for GainStage {
    fn process(&mut self, buffer: &mut [f32], _sample_rate: f32) {
        for s in buffer.iter_mut() {
            *s *= self.gain_linear;
        }
    }

    fn name(&self) -> &str {
        "GainStage"
    }

    fn reset(&mut self) {
        // stateless — nothing to reset
    }
}

/// A metering (level monitoring) stage that measures peak and RMS without
/// modifying the signal.
pub struct MeterStage {
    /// Most recently measured peak amplitude.
    pub peak: f32,
    /// Most recently measured RMS level.
    pub rms: f32,
}

impl MeterStage {
    /// Create a new meter stage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peak: 0.0,
            rms: 0.0,
        }
    }

    /// Return the last measured peak in dBFS.
    #[must_use]
    pub fn peak_db(&self) -> f32 {
        if self.peak <= 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * self.peak.log10()
        }
    }

    /// Return the last measured RMS in dBFS.
    #[must_use]
    pub fn rms_db(&self) -> f32 {
        if self.rms <= 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * self.rms.log10()
        }
    }
}

impl Default for MeterStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for MeterStage {
    fn process(&mut self, buffer: &mut [f32], _sample_rate: f32) {
        if buffer.is_empty() {
            return;
        }
        let mut peak = 0.0_f32;
        let mut sum_sq = 0.0_f32;
        for &s in buffer.iter() {
            let abs = s.abs();
            if abs > peak {
                peak = abs;
            }
            sum_sq += s * s;
        }
        self.peak = peak;
        #[allow(clippy::cast_precision_loss)]
        let n = buffer.len() as f32;
        self.rms = (sum_sq / n).sqrt();
    }

    fn name(&self) -> &str {
        "MeterStage"
    }

    fn reset(&mut self) {
        self.peak = 0.0;
        self.rms = 0.0;
    }
}

/// A one-pole low-pass smoothing stage for de-clicking gain ramps.
///
/// Uses a first-order IIR: `y[n] = alpha * x[n] + (1 - alpha) * y[n-1]`
pub struct SmoothingStage {
    /// Filter coefficient (0 < alpha <= 1). Higher = faster response.
    pub alpha: f32,
    prev: f32,
}

impl SmoothingStage {
    /// Create a new smoothing stage.
    ///
    /// `alpha` controls bandwidth: `1.0` = bypass, `0.001` = very slow.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(1e-6, 1.0),
            prev: 0.0,
        }
    }

    /// Create a stage with a given time constant in milliseconds.
    #[must_use]
    pub fn from_time_constant_ms(tc_ms: f32, sample_rate: f32) -> Self {
        let tc_samples = tc_ms * 0.001 * sample_rate;
        let alpha = 1.0 - (-1.0_f32 / tc_samples).exp();
        Self::new(alpha)
    }
}

impl PipelineStage for SmoothingStage {
    fn process(&mut self, buffer: &mut [f32], _sample_rate: f32) {
        for s in buffer.iter_mut() {
            self.prev = self.alpha * (*s) + (1.0 - self.alpha) * self.prev;
            *s = self.prev;
        }
    }

    fn name(&self) -> &str {
        "SmoothingStage"
    }

    fn reset(&mut self) {
        self.prev = 0.0;
    }
}

/// A composable audio processing pipeline.
///
/// Stages are applied in insertion order. The pipeline tracks total latency and
/// can be reset atomically.
pub struct AudioPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    sample_rate: f32,
}

impl AudioPipeline {
    /// Create a new empty pipeline at the given sample rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            stages: Vec::new(),
            sample_rate,
        }
    }

    /// Append a stage to the end of the pipeline.
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }

    /// Remove all stages from the pipeline.
    pub fn clear(&mut self) {
        self.stages.clear();
    }

    /// Return the number of stages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Return `true` if no stages have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Total latency introduced by all stages, in samples.
    #[must_use]
    pub fn total_latency_samples(&self) -> usize {
        self.stages.iter().map(|s| s.latency_samples()).sum()
    }

    /// Process a mono buffer through all stages in order.
    pub fn process(&mut self, buffer: &mut [f32]) {
        let sr = self.sample_rate;
        for stage in self.stages.iter_mut() {
            stage.process(buffer, sr);
        }
    }

    /// Reset all stage states.
    pub fn reset(&mut self) {
        for stage in self.stages.iter_mut() {
            stage.reset();
        }
    }

    /// Update the sample rate for all future `process` calls.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_stage_unity() {
        let mut g = GainStage::new(1.0);
        let mut buf = vec![0.5_f32, -0.5, 1.0, -1.0];
        g.process(&mut buf, 48_000.0);
        assert!((buf[0] - 0.5).abs() < 1e-6);
        assert!((buf[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_gain_stage_attenuation() {
        let mut g = GainStage::new(0.5);
        let mut buf = vec![1.0_f32];
        g.process(&mut buf, 48_000.0);
        assert!((buf[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gain_stage_from_db() {
        let mut g = GainStage::from_db(0.0);
        let mut buf = vec![1.0_f32];
        g.process(&mut buf, 48_000.0);
        assert!((buf[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gain_stage_minus6db() {
        let mut g = GainStage::from_db(-6.0);
        let mut buf = vec![1.0_f32];
        g.process(&mut buf, 48_000.0);
        // -6 dB ≈ 0.501
        assert!(buf[0] > 0.49 && buf[0] < 0.52);
    }

    #[test]
    fn test_gain_stage_name() {
        let g = GainStage::new(1.0);
        assert_eq!(g.name(), "GainStage");
    }

    #[test]
    fn test_gain_stage_reset_is_noop() {
        let mut g = GainStage::new(0.5);
        g.reset();
        assert!((g.gain_linear - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_meter_stage_peak() {
        let mut m = MeterStage::new();
        let mut buf = vec![0.1_f32, 0.3, -0.8, 0.2];
        m.process(&mut buf, 48_000.0);
        assert!((m.peak - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_meter_stage_rms() {
        let mut m = MeterStage::new();
        let mut buf = vec![1.0_f32, 1.0, 1.0, 1.0];
        m.process(&mut buf, 48_000.0);
        assert!((m.rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_meter_stage_non_destructive() {
        let mut m = MeterStage::new();
        let original = vec![0.5_f32, -0.5, 0.25];
        let mut buf = original.clone();
        m.process(&mut buf, 48_000.0);
        assert_eq!(buf, original);
    }

    #[test]
    fn test_meter_stage_peak_db() {
        let mut m = MeterStage::new();
        let mut buf = vec![1.0_f32];
        m.process(&mut buf, 48_000.0);
        assert!((m.peak_db() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_meter_stage_reset() {
        let mut m = MeterStage::new();
        let mut buf = vec![0.9_f32];
        m.process(&mut buf, 48_000.0);
        m.reset();
        assert_eq!(m.peak, 0.0);
        assert_eq!(m.rms, 0.0);
    }

    #[test]
    fn test_smoothing_stage_clamps_alpha() {
        let s = SmoothingStage::new(2.0);
        assert!((s.alpha - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_smoothing_stage_tracks_dc() {
        let mut s = SmoothingStage::new(1.0); // alpha=1 => bypass
        let mut buf = vec![0.5_f32; 8];
        s.process(&mut buf, 48_000.0);
        for &v in &buf {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pipeline_empty() {
        let mut p = AudioPipeline::new(44_100.0);
        let mut buf = vec![0.7_f32; 16];
        p.process(&mut buf);
        assert!(buf.iter().all(|&v| (v - 0.7).abs() < 1e-6));
    }

    #[test]
    fn test_pipeline_gain_then_meter() {
        let mut p = AudioPipeline::new(48_000.0);
        p.add_stage(Box::new(GainStage::new(0.5)));
        let meter = MeterStage::new();
        // We need the meter inside the pipeline; add separately
        p.add_stage(Box::new(meter));

        let mut buf = vec![1.0_f32; 4];
        p.process(&mut buf);
        // After gain=0.5, all samples should be 0.5
        for &v in &buf {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_pipeline_len_and_clear() {
        let mut p = AudioPipeline::new(48_000.0);
        assert!(p.is_empty());
        p.add_stage(Box::new(GainStage::new(1.0)));
        p.add_stage(Box::new(MeterStage::new()));
        assert_eq!(p.len(), 2);
        p.clear();
        assert!(p.is_empty());
    }

    #[test]
    fn test_pipeline_reset() {
        let mut p = AudioPipeline::new(48_000.0);
        p.add_stage(Box::new(SmoothingStage::new(0.1)));
        p.reset();
        // After reset, output of silence should stay at zero
        let mut buf = vec![0.0_f32; 4];
        p.process(&mut buf);
        for &v in &buf {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_pipeline_total_latency() {
        let mut p = AudioPipeline::new(48_000.0);
        p.add_stage(Box::new(GainStage::new(1.0)));
        p.add_stage(Box::new(MeterStage::new()));
        assert_eq!(p.total_latency_samples(), 0);
    }

    #[test]
    fn test_smoothing_from_time_constant() {
        let s = SmoothingStage::from_time_constant_ms(10.0, 48_000.0);
        assert!(s.alpha > 0.0 && s.alpha < 1.0);
    }
}
