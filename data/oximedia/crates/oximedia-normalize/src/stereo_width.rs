//! Stereo width adjustment via mid-side (M/S) processing.
//!
//! Encodes a stereo signal into mid (mono sum) and side (mono difference) channels,
//! scales the side channel by a width coefficient, then decodes back to L/R.
//!
//! Width = 0.0  → pure mono (all side removed)
//! Width = 1.0  → unchanged
//! Width > 1.0  → widened (side boosted)

#![allow(dead_code)]

/// Encode a stereo pair (L, R) into mid/side components.
///
/// mid  = (L + R) / 2
/// side = (L - R) / 2
#[inline]
pub fn encode_ms(left: f64, right: f64) -> (f64, f64) {
    ((left + right) * 0.5, (left - right) * 0.5)
}

/// Decode mid/side components back to a stereo pair (L, R).
///
/// L = mid + side
/// R = mid - side
#[inline]
pub fn decode_ms(mid: f64, side: f64) -> (f64, f64) {
    (mid + side, mid - side)
}

/// Apply a width coefficient to a single mid/side encoded pair.
///
/// `width` of 1.0 leaves the signal unchanged; 0.0 collapses to mono; 2.0 doubles side.
#[inline]
pub fn apply_width(mid: f64, side: f64, width: f64) -> (f64, f64) {
    (mid, side * width)
}

/// Process a block of interleaved stereo samples (LRLRLR…) with the given width.
///
/// Modifies `samples` in-place.  The slice length must be even.
pub fn process_stereo_width(samples: &mut [f64], width: f64) {
    assert!(
        samples.len() % 2 == 0,
        "interleaved stereo requires even length"
    );
    for frame in samples.chunks_exact_mut(2) {
        let (mid, side) = encode_ms(frame[0], frame[1]);
        let (mid2, side2) = apply_width(mid, side, width);
        let (l, r) = decode_ms(mid2, side2);
        frame[0] = l;
        frame[1] = r;
    }
}

/// Calculate the mono compatibility score of an interleaved stereo signal.
///
/// Returns a value in [0.0, 1.0] where 1.0 is perfectly mono-compatible and
/// 0.0 would completely cancel in mono summing.
///
/// Formula: `1.0 - (side_rms / (mid_rms + side_rms + ε))`
#[allow(clippy::cast_precision_loss)]
pub fn mono_compatibility(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 1.0;
    }
    let mut mid_sq_sum = 0.0_f64;
    let mut side_sq_sum = 0.0_f64;
    let frames = samples.len() / 2;
    for frame in samples.chunks_exact(2) {
        let (mid, side) = encode_ms(frame[0], frame[1]);
        mid_sq_sum += mid * mid;
        side_sq_sum += side * side;
    }
    let mid_rms = (mid_sq_sum / frames as f64).sqrt();
    let side_rms = (side_sq_sum / frames as f64).sqrt();
    let denom = mid_rms + side_rms + 1e-12;
    1.0 - (side_rms / denom)
}

/// Configuration for the stereo width processor.
#[derive(Debug, Clone)]
pub struct StereoWidthConfig {
    /// Width coefficient (0.0 = mono, 1.0 = original, 2.0 = widened).
    pub width: f64,
    /// Smoothing coefficient for width changes (0 = instant, 1 = frozen).
    pub smoothing: f64,
}

impl Default for StereoWidthConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            smoothing: 0.99,
        }
    }
}

/// Stateful stereo width processor with smoothed width changes.
#[derive(Debug)]
pub struct StereoWidthProcessor {
    config: StereoWidthConfig,
    current_width: f64,
}

impl StereoWidthProcessor {
    /// Create a new processor from a config.
    pub fn new(config: StereoWidthConfig) -> Self {
        let current_width = config.width;
        Self {
            config,
            current_width,
        }
    }

    /// Set the target width (will be approached smoothly).
    pub fn set_width(&mut self, width: f64) {
        self.config.width = width.max(0.0);
    }

    /// Process a block of interleaved stereo samples in-place.
    pub fn process(&mut self, samples: &mut [f64]) {
        assert!(
            samples.len() % 2 == 0,
            "interleaved stereo requires even length"
        );
        let target = self.config.width;
        let smooth = self.config.smoothing;
        for frame in samples.chunks_exact_mut(2) {
            // Smooth the width
            self.current_width = self.current_width * smooth + target * (1.0 - smooth);
            let (mid, side) = encode_ms(frame[0], frame[1]);
            let (mid2, side2) = apply_width(mid, side, self.current_width);
            let (l, r) = decode_ms(mid2, side2);
            frame[0] = l;
            frame[1] = r;
        }
    }

    /// Return the current (smoothed) width value.
    pub fn current_width(&self) -> f64 {
        self.current_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_ms_sum_and_difference() {
        let (mid, side) = encode_ms(1.0, 0.0);
        assert!((mid - 0.5).abs() < 1e-12);
        assert!((side - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_decode_ms_inverse_of_encode() {
        let l = 0.7;
        let r = -0.3;
        let (mid, side) = encode_ms(l, r);
        let (l2, r2) = decode_ms(mid, side);
        assert!((l2 - l).abs() < 1e-12);
        assert!((r2 - r).abs() < 1e-12);
    }

    #[test]
    fn test_apply_width_unity() {
        let (mid_out, side_out) = apply_width(0.5, 0.3, 1.0);
        assert!((mid_out - 0.5).abs() < 1e-12);
        assert!((side_out - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_apply_width_mono() {
        let (mid_out, side_out) = apply_width(0.5, 0.3, 0.0);
        assert!((mid_out - 0.5).abs() < 1e-12);
        assert!((side_out - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_width_double() {
        let (_mid_out, side_out) = apply_width(0.5, 0.3, 2.0);
        assert!((side_out - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_process_stereo_width_unity_unchanged() {
        let original: Vec<f64> = vec![0.4, -0.2, 0.6, 0.1];
        let mut samples = original.clone();
        process_stereo_width(&mut samples, 1.0);
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "unity width should not change signal"
            );
        }
    }

    #[test]
    fn test_process_stereo_width_mono() {
        let mut samples = vec![0.8, 0.4, -0.6, 0.2];
        process_stereo_width(&mut samples, 0.0);
        // L and R should be equal (mono)
        assert!((samples[0] - samples[1]).abs() < 1e-12);
        assert!((samples[2] - samples[3]).abs() < 1e-12);
    }

    #[test]
    fn test_mono_compatibility_pure_mono() {
        // Pure mono: L == R, side = 0
        let samples: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.5 } else { 0.5 })
            .collect();
        let score = mono_compatibility(&samples);
        assert!(score > 0.99, "score = {score}");
    }

    #[test]
    fn test_mono_compatibility_out_of_phase() {
        // Out of phase: L = -R, side is large
        let samples: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let score = mono_compatibility(&samples);
        assert!(score < 0.1, "score = {score}");
    }

    #[test]
    fn test_stereo_width_processor_default_unity() {
        let proc = StereoWidthProcessor::new(StereoWidthConfig::default());
        assert!((proc.current_width() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stereo_width_processor_set_width() {
        let mut proc = StereoWidthProcessor::new(StereoWidthConfig::default());
        proc.set_width(2.0);
        assert!((proc.config.width - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_stereo_width_processor_negative_width_clamped() {
        let mut proc = StereoWidthProcessor::new(StereoWidthConfig::default());
        proc.set_width(-1.0);
        assert!(proc.config.width >= 0.0);
    }

    #[test]
    fn test_stereo_width_processor_process_block() {
        let cfg = StereoWidthConfig {
            width: 0.0,
            smoothing: 0.0,
        };
        let mut proc = StereoWidthProcessor::new(cfg);
        let mut samples = vec![0.8f64, -0.4, 0.6, 0.2];
        proc.process(&mut samples);
        // With width=0.0 and no smoothing, L and R should be equal after processing
        assert!((samples[0] - samples[1]).abs() < 1e-10);
        assert!((samples[2] - samples[3]).abs() < 1e-10);
    }
}
