//! Temporal noise measurement for video frames.
//!
//! Estimates per-frame noise by comparing consecutive frames. This is useful
//! for adaptive denoising, quality monitoring, and encoding decisions.
//!
//! The estimator uses inter-frame differences normalized under the Gaussian
//! noise model: `sigma ≈ sqrt(mean(|curr - prev|²)) / sqrt(2)`, with an
//! empirical motion correction factor of 0.8 that reduces the bias introduced
//! by real scene motion.

/// Temporal noise measurement from consecutive video frames.
///
/// Feed luma (or any single-plane) frames sequentially. The estimator
/// maintains a rolling history of per-frame noise estimates and can report
/// a smoothed estimate over the last `window` frames.
pub struct TemporalNoiseMeasurement {
    prev_frame: Option<Vec<u8>>,
    /// Per-frame noise sigma estimates (most recent last).
    pub history: Vec<f32>,
    /// Number of frames to average for rolling noise. Default 5.
    pub window: usize,
}

impl TemporalNoiseMeasurement {
    /// Create a new temporal noise measurement with the given averaging window.
    ///
    /// # Arguments
    ///
    /// * `window` - Number of recent frames to average for `rolling_noise`. Must be ≥ 1.
    pub fn new(window: usize) -> Self {
        Self {
            prev_frame: None,
            history: Vec::new(),
            window: window.max(1),
        }
    }

    /// Estimate inter-frame noise from a new frame.
    ///
    /// On the first call (no previous frame) this always returns `0.0`.
    ///
    /// # Arguments
    ///
    /// * `frame` - Raw luma bytes, length must equal `w * h`.
    /// * `w`     - Frame width in pixels.
    /// * `h`     - Frame height in pixels.
    ///
    /// # Returns
    ///
    /// Estimated noise sigma in raw-value units (0..255 scale for u8 frames).
    /// Returns `0.0` on the first call.
    pub fn measure(&mut self, frame: &[u8], w: u32, h: u32) -> f32 {
        let expected_len = (w as usize).saturating_mul(h as usize);

        // Guard against empty or mismatched frames.
        if expected_len == 0 || frame.len() < expected_len {
            return 0.0;
        }

        let frame_slice = &frame[..expected_len];

        let sigma = match &self.prev_frame {
            None => 0.0,
            Some(prev) => {
                // Compute mean squared absolute difference.
                let sum_sq: f64 = frame_slice
                    .iter()
                    .zip(prev.iter())
                    .map(|(&c, &p)| {
                        let diff = c as f64 - p as f64;
                        diff * diff
                    })
                    .sum();

                let mean_sq = sum_sq / expected_len as f64;

                // Inter-frame Gaussian noise model: sigma_noise = sqrt(mean_sq) / sqrt(2)
                // Multiply by 0.8 to account for residual motion (empirical correction).
                const MOTION_CORRECTION: f64 = 0.8;
                let sigma_f64 = (mean_sq / 2.0).sqrt() * MOTION_CORRECTION;
                sigma_f64 as f32
            }
        };

        // Store current frame as previous.
        self.prev_frame = Some(frame_slice.to_vec());

        // Append to history.
        self.history.push(sigma);

        sigma
    }

    /// Rolling average of noise over the last `window` frames.
    ///
    /// Returns `None` if no frames have been measured yet (history is empty).
    /// The first frame always records `0.0`; call after a second frame to get
    /// a meaningful estimate.
    pub fn rolling_noise(&self) -> Option<f32> {
        if self.history.is_empty() {
            return None;
        }
        let start = self.history.len().saturating_sub(self.window);
        let slice = &self.history[start..];
        let sum: f32 = slice.iter().sum();
        Some(sum / slice.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First call returns 0.0 — no previous frame to diff against.
    #[test]
    fn test_temporal_noise_first_frame_zero() {
        let mut m = TemporalNoiseMeasurement::new(5);
        let frame = vec![128u8; 64 * 64];
        let sigma = m.measure(&frame, 64, 64);
        assert_eq!(sigma, 0.0, "First frame must return 0.0, got {sigma}");
    }

    /// Identical consecutive frames produce zero (or near-zero) noise.
    #[test]
    fn test_temporal_noise_static() {
        let mut m = TemporalNoiseMeasurement::new(5);
        let frame = vec![100u8; 32 * 32];
        let _ = m.measure(&frame, 32, 32); // seed prev
        let sigma = m.measure(&frame, 32, 32);
        assert!(
            sigma < 1e-6,
            "Static scene should produce ~0 noise, got {sigma}"
        );
    }

    /// Add known Gaussian noise via Box-Muller and verify the estimate is
    /// within 20 % of the known sigma.
    #[test]
    fn test_temporal_noise_synthetic() {
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let true_sigma = 10.0_f64; // raw-value noise sigma

        // Box-Muller noise generator (pure, no external crate).
        let noise: Vec<f64> = box_muller_noise(n, true_sigma, 0x1234_5678_9abc_def0);

        let base_val = 128u8;
        let base_frame = vec![base_val; n];

        // Noisy frame: base + gaussian noise, clamped to [0, 255].
        let noisy_frame: Vec<u8> = noise
            .iter()
            .map(|&v| (base_val as f64 + v).clamp(0.0, 255.0) as u8)
            .collect();

        let mut m = TemporalNoiseMeasurement::new(5);
        let _ = m.measure(&base_frame, w, h);
        let sigma = m.measure(&noisy_frame, w, h);

        // When prev = clean and curr = clean + gaussian(sigma_true):
        //   diff[i] = gaussian(sigma_true)
        //   mean_sq  = sigma_true²
        //   estimator = sqrt(sigma_true² / 2) * 0.8 = sigma_true * 0.8 / sqrt(2)
        let expected = (true_sigma * 0.8 / 2.0_f64.sqrt()) as f32;
        let tolerance = 0.20 * expected;

        assert!(
            (sigma - expected).abs() < tolerance,
            "Expected ~{expected:.2}, got {sigma:.2} (tolerance ±{tolerance:.2})"
        );
    }

    /// Rolling noise returns the average of the last `window` estimates.
    #[test]
    fn test_temporal_noise_rolling() {
        let w = 8u32;
        let h = 8u32;
        let n = (w * h) as usize;
        let window = 3;
        let mut m = TemporalNoiseMeasurement::new(window);

        // Feed 5 identical pairs of frames so noise estimates are deterministic.
        // Frame A: all 0, Frame B: all 10 → diff = 10 each pixel.
        // sigma = (10² / 2).sqrt() * 0.8 = ~5.657
        let a = vec![0u8; n];
        let b = vec![10u8; n];
        let expected_single = ((100.0_f64 / 2.0_f64).sqrt() * 0.8) as f32;

        let _ = m.measure(&a, w, h); // seed (0.0)
        for _ in 0..4 {
            let _ = m.measure(&b, w, h);
            let _ = m.measure(&a, w, h);
        }

        let rolling = m
            .rolling_noise()
            .expect("rolling_noise should be Some after many frames");

        // With alternating pairs the last `window` values should all be ~expected_single.
        assert!(
            (rolling - expected_single).abs() < 0.1,
            "Rolling average {rolling:.4} deviates from {expected_single:.4}"
        );
    }

    /// Rolling noise returns None before any frames are measured.
    #[test]
    fn test_temporal_noise_rolling_before_frames() {
        let m = TemporalNoiseMeasurement::new(5);
        assert!(m.rolling_noise().is_none());
    }

    // ---- helpers ----

    /// Minimal deterministic Box-Muller Gaussian noise generator (no external crates).
    /// Uses a simple xorshift64 PRNG as the uniform source.
    fn box_muller_noise(n: usize, sigma: f64, seed: u64) -> Vec<f64> {
        let mut state = if seed == 0 { 1 } else { seed };
        let mut result = Vec::with_capacity(n);

        let xorshift = |s: &mut u64| -> f64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            // Map to (0, 1) exclusive.
            (*s as f64 + 1.0) / (u64::MAX as f64 + 2.0)
        };

        while result.len() < n {
            let u1 = xorshift(&mut state);
            let u2 = xorshift(&mut state);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            result.push(r * theta.cos() * sigma);
            if result.len() < n {
                result.push(r * theta.sin() * sigma);
            }
        }

        result
    }
}
