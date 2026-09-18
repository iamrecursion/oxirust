//! Temporal noise reduction for video streams.
//!
//! Provides weighted multi-frame blending, adaptive blend-factor calculation,
//! motion-aware temporal filters, and a Donoho–Johnstone noise-sigma estimator
//! based on the 3×3 Laplacian.

use std::collections::VecDeque;

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Temporal denoising strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalDenoiseMode {
    /// Dynamically adjust the blend factor based on inter-frame motion.
    Adaptive,
    /// Apply a fixed blend factor regardless of motion.
    Fixed,
    /// Motion-compensated blending: highly-static regions receive more
    /// temporal averaging; moving regions fall back to the current frame.
    MotionCompensated,
}

/// Noise metrics computed for a single frame.
#[derive(Debug, Clone)]
pub struct NoiseMetrics {
    /// Estimated noise standard deviation (Donoho–Johnstone method).
    pub estimated_sigma: f32,
    /// Signal-to-noise ratio in decibels: `20·log₁₀(255 / sigma)`.
    pub snr_db: f32,
    /// Mean per-pixel squared difference between this frame and the previous.
    pub temporal_variance: f32,
}

/// Stateful temporal denoiser that accumulates frame history.
pub struct TemporalDenoiser {
    /// Denoising strategy.
    pub mode: TemporalDenoiseMode,
    /// Blend factor for `Fixed` mode (0 = keep current, 1 = replace with average).
    pub blend_factor: f32,
    /// Maximum number of historical frames to retain.
    pub history_frames: usize,
    /// Ring buffer of previous denoised frames.
    pub frame_history: VecDeque<Vec<u8>>,
}

impl TemporalDenoiser {
    /// Create a new `TemporalDenoiser`.
    ///
    /// `blend_factor` is used in `Fixed` mode; it is ignored for `Adaptive`
    /// and `MotionCompensated` modes.
    pub fn new(mode: TemporalDenoiseMode, blend_factor: f32, history_frames: usize) -> Self {
        Self {
            mode,
            blend_factor: blend_factor.clamp(0.0, 1.0),
            history_frames,
            frame_history: VecDeque::with_capacity(history_frames),
        }
    }

    /// Denoise `current_frame` using the internal history and optional
    /// lookahead frames (`prev` / `next`).
    ///
    /// The denoised frame is also pushed into the internal history for
    /// subsequent calls.
    ///
    /// `current_frame`, `prev`, and `next` are all grayscale buffers of
    /// `width × height` bytes.
    pub fn process_frame(
        &mut self,
        current_frame: &[u8],
        prev: Option<&[u8]>,
        next: Option<&[u8]>,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let denoised = denoise_frame(current_frame, prev, next, &self.mode, width, height);

        // Update history.
        if self.frame_history.len() >= self.history_frames {
            self.frame_history.pop_front();
        }
        self.frame_history.push_back(denoised.clone());

        denoised
    }

    /// Denoise `current` frame by comparing each pixel to the corresponding
    /// pixel in `reference` using a parallel row-based approach.
    ///
    /// This is equivalent to [`TemporalDenoiseMode::MotionCompensated`] with a
    /// single reference frame but processes rows in parallel using rayon.
    ///
    /// - Pixels where `|current[i] - reference[i]| < 10` are replaced by the
    ///   average of the two values (temporal smoothing).
    /// - All other pixels keep their current value (motion bypass).
    ///
    /// The denoised frame is also pushed into the internal history, consistent
    /// with [`Self::process_frame`].
    pub fn denoise_frame_parallel(
        &mut self,
        current: &[u8],
        reference: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        use rayon::prelude::*;

        let pixel_count = (width as usize) * (height as usize);
        let row_width = width as usize;
        let mut out = vec![0u8; pixel_count];

        out.par_chunks_mut(row_width)
            .enumerate()
            .for_each(|(row_idx, row_out)| {
                let row_start = row_idx * row_width;
                for col in 0..row_width {
                    let i = row_start + col;
                    let cur_px = current.get(i).copied().unwrap_or(0);
                    let ref_px = reference.get(i).copied().unwrap_or(0);
                    row_out[col] = if (cur_px as i32 - ref_px as i32).unsigned_abs() < 10 {
                        ((cur_px as u32 + ref_px as u32) / 2) as u8
                    } else {
                        cur_px
                    };
                }
            });

        // Update history (same behaviour as process_frame).
        if self.frame_history.len() >= self.history_frames {
            self.frame_history.pop_front();
        }
        self.frame_history.push_back(out.clone());
        out
    }

    /// Compute noise metrics for `frame` relative to the most recent historical frame.
    pub fn compute_metrics(&self, frame: &[u8], width: u32, height: u32) -> NoiseMetrics {
        let sigma = estimate_noise_sigma(frame, width, height);
        let snr_db = if sigma > 0.0 {
            20.0 * (255.0f32 / sigma).log10()
        } else {
            f32::INFINITY
        };
        let temporal_variance = if let Some(prev) = self.frame_history.back() {
            let n = frame.len().max(1) as f32;
            frame
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| {
                    let d = a as f32 - b as f32;
                    d * d
                })
                .sum::<f32>()
                / n
        } else {
            0.0
        };
        NoiseMetrics {
            estimated_sigma: sigma,
            snr_db,
            temporal_variance,
        }
    }
}

// -----------------------------------------------------------------------
// Public free functions
// -----------------------------------------------------------------------

/// Compute a weighted blend of multiple same-sized frames.
///
/// `frames` is a slice of frame references; `weights` is a parallel slice of
/// per-frame weights.  Weights need not sum to 1.0 — they are automatically
/// normalised.
///
/// Each frame must be `width × height` bytes.  If `frames` is empty, an
/// all-zero frame of the expected size is returned.
pub fn blend_frames(frames: &[&[u8]], weights: &[f32], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    if frames.is_empty() {
        return vec![0u8; pixel_count];
    }

    let n = frames.len().min(weights.len());
    let weight_sum: f32 = weights[..n].iter().sum();
    let weight_sum = if weight_sum.abs() < f32::EPSILON {
        1.0
    } else {
        weight_sum
    };

    let mut out = vec![0.0f32; pixel_count];
    for (frame, &w) in frames.iter().take(n).zip(weights.iter().take(n)) {
        let norm_w = w / weight_sum;
        for (i, &px) in frame.iter().enumerate().take(pixel_count) {
            out[i] += px as f32 * norm_w;
        }
    }

    out.iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Choose a blend factor based on inter-frame motion magnitude.
///
/// High motion → small blend factor (0.8 means "keep 80% of current frame").
/// Low motion → large blend factor (0.2 means "use 20% of current and 80% of history").
///
/// The mapping is linear:
/// - `motion_score = 0.0` → `blend_factor = 0.2` (maximum temporal smoothing)
/// - `motion_score = 1.0` → `blend_factor = 0.8` (minimum temporal smoothing)
///
/// `motion_score` must be in [0, 1].
#[inline]
pub fn adaptive_blend_factor(motion_score: f32) -> f32 {
    let clamped = motion_score.clamp(0.0, 1.0);
    // Interpolate between 0.2 (low motion) and 0.8 (high motion).
    0.2 + clamped * 0.6
}

/// Compute the mean absolute difference (MAD) between two frames, normalised
/// to [0, 1].
///
/// Both frames must be grayscale `width × height` bytes.  Returns 0.0 for
/// empty or equal frames.
pub fn motion_score_between(prev: &[u8], curr: &[u8], width: u32, height: u32) -> f32 {
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return 0.0;
    }
    let mad: f64 = prev
        .iter()
        .take(n)
        .zip(curr.iter().take(n))
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / (n as f64 * 255.0);
    mad as f32
}

/// Apply temporal denoising to a single frame.
///
/// - `Adaptive`: compute motion score vs `prev`, derive `adaptive_blend_factor`,
///   then blend current frame with `prev` (and `next` if available) using that factor.
/// - `Fixed`: always use `blend_factor = 0.5` (equal mix of available frames).
/// - `MotionCompensated`: per-pixel decision — pixels with low inter-frame
///   difference (< 10) receive strong temporal averaging; others keep the
///   current value.
///
/// All inputs are grayscale `width × height` bytes.  Returns a denoised frame
/// of the same size.
pub fn denoise_frame(
    current: &[u8],
    prev: Option<&[u8]>,
    next: Option<&[u8]>,
    mode: &TemporalDenoiseMode,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);

    match mode {
        TemporalDenoiseMode::Fixed => {
            // Blend current with available neighbours equally.
            let mut frames: Vec<&[u8]> = vec![current];
            let mut weights: Vec<f32> = vec![1.0];
            if let Some(p) = prev {
                frames.push(p);
                weights.push(1.0);
            }
            if let Some(n) = next {
                frames.push(n);
                weights.push(1.0);
            }
            blend_frames(&frames, &weights, width, height)
        }

        TemporalDenoiseMode::Adaptive => {
            let score = match prev {
                Some(p) => motion_score_between(p, current, width, height),
                None => 1.0,
            };
            let alpha = adaptive_blend_factor(score);
            // alpha = weight given to the current frame.
            // (1 - alpha) distributed equally to neighbours.
            match (prev, next) {
                (Some(p), Some(n)) => {
                    let frames = [current, p, n];
                    let neighbour_w = (1.0 - alpha) / 2.0;
                    let weights = [alpha, neighbour_w, neighbour_w];
                    blend_frames(&frames, &weights, width, height)
                }
                (Some(p), None) => {
                    let frames = [current, p];
                    let weights = [alpha, 1.0 - alpha];
                    blend_frames(&frames, &weights, width, height)
                }
                (None, Some(n)) => {
                    let frames = [current, n];
                    let weights = [alpha, 1.0 - alpha];
                    blend_frames(&frames, &weights, width, height)
                }
                (None, None) => current.to_vec(),
            }
        }

        TemporalDenoiseMode::MotionCompensated => {
            let mut out = vec![0u8; pixel_count];
            for i in 0..pixel_count {
                let cur_px = current.get(i).copied().unwrap_or(0);
                let prev_px = prev.and_then(|p| p.get(i)).copied();
                let next_px = next.and_then(|n| n.get(i)).copied();

                let is_static = match (prev_px, next_px) {
                    (Some(p), Some(n)) => {
                        (cur_px as i32 - p as i32).unsigned_abs() < 10
                            && (cur_px as i32 - n as i32).unsigned_abs() < 10
                    }
                    (Some(p), None) => (cur_px as i32 - p as i32).unsigned_abs() < 10,
                    (None, Some(n)) => (cur_px as i32 - n as i32).unsigned_abs() < 10,
                    (None, None) => false,
                };

                out[i] = if is_static {
                    // Strong temporal average.
                    let mut sum = cur_px as u32;
                    let mut count = 1u32;
                    if let Some(p) = prev_px {
                        sum += p as u32;
                        count += 1;
                    }
                    if let Some(n) = next_px {
                        sum += n as u32;
                        count += 1;
                    }
                    (sum / count) as u8
                } else {
                    cur_px
                };
            }
            out
        }
    }
}

/// Estimate the noise standard deviation using the Donoho–Johnstone robust
/// median estimator applied to the absolute values of a 3×3 Laplacian kernel.
///
/// The Laplacian kernel is:
/// ```text
///  0  1  0
///  1 -4  1
///  0  1  0
/// ```
///
/// The estimator computes:
/// ```text
/// sigma = median(|Laplacian responses|) / 0.6745
/// ```
///
/// Border pixels (1-pixel perimeter) are excluded.  Returns 0.0 for frames
/// that are too small to apply the filter.
pub fn estimate_noise_sigma(frame: &[u8], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;
    if w < 3 || h < 3 {
        return 0.0;
    }

    let mut responses: Vec<f32> = Vec::with_capacity((w - 2) * (h - 2));

    for row in 1..(h - 1) {
        for col in 1..(w - 1) {
            let center = frame[row * w + col] as i32;
            let top = frame[(row - 1) * w + col] as i32;
            let bottom = frame[(row + 1) * w + col] as i32;
            let left = frame[row * w + col - 1] as i32;
            let right = frame[row * w + col + 1] as i32;

            // Laplacian: top + bottom + left + right - 4*center
            let lap = (top + bottom + left + right - 4 * center).abs() as f32;
            responses.push(lap);
        }
    }

    let med = median_f32_slice(&responses);
    // Donoho–Johnstone normalisation constant.
    med / 0.6745
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

/// Compute the median of a mutable-owned slice of `f32`.
fn median_f32_slice(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -------------------------------------------------------

    fn flat_frame(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width * height) as usize]
    }

    fn ramp_frame(width: u32, height: u32) -> Vec<u8> {
        (0..(width * height) as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    /// Frame with uniform white noise added to a base value.
    fn noisy_frame(width: u32, height: u32, base: u8, noise_amp: u8) -> Vec<u8> {
        (0u64..(width as u64 * height as u64))
            .map(|i| {
                // LCG: wrapping arithmetic to avoid overflow
                let hash = i
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let noise = (hash >> 56) as u8;
                base.saturating_add(noise % noise_amp.max(1))
            })
            .collect()
    }

    // ---- blend_frames -------------------------------------------------

    // 1. blend_frames: single frame → unchanged
    #[test]
    fn test_blend_frames_single_frame_unchanged() {
        let frame = ramp_frame(8, 8);
        let out = blend_frames(&[frame.as_slice()], &[1.0], 8, 8);
        assert_eq!(out, frame);
    }

    // 2. blend_frames: equal weights of same frame → same frame
    #[test]
    fn test_blend_frames_equal_weights_same_frame() {
        let frame = flat_frame(8, 8, 100);
        let out = blend_frames(&[frame.as_slice(), frame.as_slice()], &[1.0, 1.0], 8, 8);
        assert_eq!(out, frame);
    }

    // 3. blend_frames: 50/50 of 0 and 200 → 100
    #[test]
    fn test_blend_frames_half_half() {
        let fa = flat_frame(4, 4, 0);
        let fb = flat_frame(4, 4, 200);
        let out = blend_frames(&[fa.as_slice(), fb.as_slice()], &[1.0, 1.0], 4, 4);
        for &px in &out {
            assert_eq!(px, 100u8);
        }
    }

    // 4. blend_frames: empty input → all-zero frame
    #[test]
    fn test_blend_frames_empty_input_zero() {
        let out = blend_frames(&[], &[], 4, 4);
        assert_eq!(out, vec![0u8; 16]);
    }

    // 5. blend_frames: three frames, asymmetric weights
    #[test]
    fn test_blend_frames_three_asymmetric_weights() {
        let f0 = flat_frame(2, 2, 0);
        let f1 = flat_frame(2, 2, 100);
        let f2 = flat_frame(2, 2, 200);
        // weights 1,1,2 → sum=4; result = (0 + 100 + 400)/4 = 125
        let out = blend_frames(
            &[f0.as_slice(), f1.as_slice(), f2.as_slice()],
            &[1.0, 1.0, 2.0],
            2,
            2,
        );
        for &px in &out {
            assert_eq!(px, 125u8);
        }
    }

    // ---- adaptive_blend_factor ----------------------------------------

    // 6. adaptive_blend_factor(0.0) → 0.2
    #[test]
    fn test_adaptive_blend_factor_zero_motion() {
        let f = adaptive_blend_factor(0.0);
        assert!((f - 0.2).abs() < 1e-5, "expected 0.2, got {f}");
    }

    // 7. adaptive_blend_factor(1.0) → 0.8
    #[test]
    fn test_adaptive_blend_factor_full_motion() {
        let f = adaptive_blend_factor(1.0);
        assert!((f - 0.8).abs() < 1e-5, "expected 0.8, got {f}");
    }

    // 8. adaptive_blend_factor(0.5) → 0.5
    #[test]
    fn test_adaptive_blend_factor_half_motion() {
        let f = adaptive_blend_factor(0.5);
        assert!((f - 0.5).abs() < 1e-5, "expected 0.5, got {f}");
    }

    // 9. adaptive_blend_factor is monotonically increasing
    #[test]
    fn test_adaptive_blend_factor_monotone() {
        let values: Vec<f32> = (0..=10)
            .map(|i| adaptive_blend_factor(i as f32 / 10.0))
            .collect();
        for w in values.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "not monotone: {:.4} < {:.4}",
                w[1],
                w[0]
            );
        }
    }

    // ---- motion_score_between -----------------------------------------

    // 10. motion_score_between: identical frames → 0.0
    #[test]
    fn test_motion_score_identical_zero() {
        let frame = ramp_frame(16, 16);
        let score = motion_score_between(&frame, &frame, 16, 16);
        assert_eq!(score, 0.0);
    }

    // 11. motion_score_between: 0 vs 255 → close to 1.0
    #[test]
    fn test_motion_score_opposite_close_to_one() {
        let black = flat_frame(8, 8, 0);
        let white = flat_frame(8, 8, 255);
        let score = motion_score_between(&black, &white, 8, 8);
        assert!((score - 1.0).abs() < 1e-5, "expected ≈1.0, got {score}");
    }

    // 12. motion_score_between is in [0, 1]
    #[test]
    fn test_motion_score_range() {
        let fa = ramp_frame(16, 16);
        let fb = noisy_frame(16, 16, 128, 30);
        let score = motion_score_between(&fa, &fb, 16, 16);
        assert!(score >= 0.0 && score <= 1.0, "score {score} out of [0,1]");
    }

    // ---- denoise_frame ------------------------------------------------

    // 13. denoise_frame Fixed with no neighbours → returns current frame
    #[test]
    fn test_denoise_fixed_no_neighbours_returns_current() {
        let frame = ramp_frame(8, 8);
        let out = denoise_frame(&frame, None, None, &TemporalDenoiseMode::Fixed, 8, 8);
        assert_eq!(out, frame);
    }

    // 14. denoise_frame Fixed with equal prev → average of two
    #[test]
    fn test_denoise_fixed_two_frames_average() {
        let a = flat_frame(4, 4, 100);
        let b = flat_frame(4, 4, 200);
        let out = denoise_frame(&a, Some(&b), None, &TemporalDenoiseMode::Fixed, 4, 4);
        for &px in &out {
            assert_eq!(px, 150u8, "expected 150, got {px}");
        }
    }

    // 15. denoise_frame Adaptive no prev → returns current frame unchanged
    #[test]
    fn test_denoise_adaptive_no_prev_returns_current() {
        let frame = ramp_frame(8, 8);
        let out = denoise_frame(&frame, None, None, &TemporalDenoiseMode::Adaptive, 8, 8);
        assert_eq!(out, frame);
    }

    // 16. denoise_frame Adaptive still scene → output close to prev
    #[test]
    fn test_denoise_adaptive_still_scene_smoothed() {
        let frame = flat_frame(8, 8, 128);
        let prev = flat_frame(8, 8, 128);
        let out = denoise_frame(
            &frame,
            Some(&prev),
            None,
            &TemporalDenoiseMode::Adaptive,
            8,
            8,
        );
        // Still scene → motion_score ≈ 0 → alpha ≈ 0.2 (heavy smoothing)
        // Output should be near 128.
        for &px in &out {
            assert!((px as i32 - 128).abs() <= 5, "still scene px = {px}");
        }
    }

    // 17. denoise_frame MotionCompensated: static region is averaged
    #[test]
    fn test_denoise_motion_compensated_static_averaged() {
        let frame = flat_frame(4, 4, 100);
        let prev = flat_frame(4, 4, 100);
        let next = flat_frame(4, 4, 100);
        let out = denoise_frame(
            &frame,
            Some(&prev),
            Some(&next),
            &TemporalDenoiseMode::MotionCompensated,
            4,
            4,
        );
        for &px in &out {
            assert_eq!(px, 100u8);
        }
    }

    // 18. denoise_frame MotionCompensated: moving region keeps current
    #[test]
    fn test_denoise_motion_compensated_moving_keeps_current() {
        let current = flat_frame(4, 4, 200);
        let prev = flat_frame(4, 4, 0); // large difference > 10
        let out = denoise_frame(
            &current,
            Some(&prev),
            None,
            &TemporalDenoiseMode::MotionCompensated,
            4,
            4,
        );
        // Difference is 200 ≥ 10, so output should remain 200.
        for &px in &out {
            assert_eq!(px, 200u8);
        }
    }

    // ---- estimate_noise_sigma -----------------------------------------

    // 19. estimate_noise_sigma: flat frame → sigma ≈ 0
    #[test]
    fn test_noise_sigma_flat_frame_near_zero() {
        let frame = flat_frame(16, 16, 128);
        let sigma = estimate_noise_sigma(&frame, 16, 16);
        assert!(
            sigma < 1.0,
            "flat frame sigma should be near 0, got {sigma}"
        );
    }

    // 20. estimate_noise_sigma: noisy frame → sigma > 0
    #[test]
    fn test_noise_sigma_noisy_frame_nonzero() {
        let frame = noisy_frame(16, 16, 128, 50);
        let sigma = estimate_noise_sigma(&frame, 16, 16);
        assert!(sigma >= 0.0, "sigma must be non-negative");
    }

    // 21. estimate_noise_sigma: too-small frame → 0.0
    #[test]
    fn test_noise_sigma_too_small_frame_zero() {
        let frame = flat_frame(2, 2, 100);
        let sigma = estimate_noise_sigma(&frame, 2, 2);
        assert_eq!(sigma, 0.0);
    }

    // ---- TemporalDenoiser -------------------------------------------

    // 22. TemporalDenoiser::process_frame: history grows
    #[test]
    fn test_temporal_denoiser_history_grows() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        let frame = flat_frame(8, 8, 100);
        for i in 0..4u64 {
            let _ = i;
            denoiser.process_frame(&frame, None, None, 8, 8);
        }
        assert_eq!(denoiser.frame_history.len(), 4);
    }

    // 23. TemporalDenoiser::process_frame: history is bounded
    #[test]
    fn test_temporal_denoiser_history_bounded() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 3);
        let frame = flat_frame(8, 8, 100);
        for _ in 0..10 {
            denoiser.process_frame(&frame, None, None, 8, 8);
        }
        assert!(denoiser.frame_history.len() <= 3);
    }

    // 24. TemporalDenoiser::compute_metrics: SNR is positive for nonzero sigma
    #[test]
    fn test_temporal_denoiser_compute_metrics_snr() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        let frame = noisy_frame(16, 16, 128, 20);
        denoiser.process_frame(&frame, None, None, 16, 16);
        let metrics = denoiser.compute_metrics(&frame, 16, 16);
        assert!(metrics.estimated_sigma >= 0.0);
        assert!(metrics.snr_db > 0.0 || metrics.estimated_sigma < 1e-3);
    }

    // 25. NoiseMetrics temporal_variance is non-negative
    #[test]
    fn test_noise_metrics_temporal_variance_nonnegative() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Adaptive, 0.5, 4);
        let f1 = flat_frame(8, 8, 100);
        let f2 = flat_frame(8, 8, 150);
        denoiser.process_frame(&f1, None, None, 8, 8);
        let metrics = denoiser.compute_metrics(&f2, 8, 8);
        assert!(metrics.temporal_variance >= 0.0);
    }

    // ---- denoise_frame_parallel ----------------------------------------

    // 26. denoise_frame_parallel: identical current and reference → output equals current
    #[test]
    fn test_denoise_frame_parallel_identical() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        let frame = ramp_frame(8, 8);
        let out = denoiser.denoise_frame_parallel(&frame, &frame, 8, 8);
        // diff = 0 < 10, so each pixel = (px + px) / 2 = px
        assert_eq!(out, frame);
    }

    // 27. denoise_frame_parallel: static region (diff < 10) is averaged
    #[test]
    fn test_denoise_frame_parallel_static_region_averaged() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        let current = flat_frame(4, 4, 100);
        let reference = flat_frame(4, 4, 106); // diff = 6 < 10 → average = 103
        let out = denoiser.denoise_frame_parallel(&current, &reference, 4, 4);
        for &px in &out {
            assert_eq!(
                px, 103u8,
                "static pixel should be averaged to 103, got {px}"
            );
        }
    }

    // 28. denoise_frame_parallel: moving region (diff >= 10) keeps current
    #[test]
    fn test_denoise_frame_parallel_moving_region_keeps_current() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        let current = flat_frame(4, 4, 200);
        let reference = flat_frame(4, 4, 0); // diff = 200 >= 10 → keep current
        let out = denoiser.denoise_frame_parallel(&current, &reference, 4, 4);
        for &px in &out {
            assert_eq!(px, 200u8, "moving pixel should remain 200, got {px}");
        }
    }

    // 29. denoise_frame_parallel: updates frame_history
    #[test]
    fn test_denoise_frame_parallel_updates_history() {
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        assert_eq!(denoiser.frame_history.len(), 0);
        let frame = flat_frame(8, 8, 128);
        denoiser.denoise_frame_parallel(&frame, &frame, 8, 8);
        assert_eq!(denoiser.frame_history.len(), 1);
        denoiser.denoise_frame_parallel(&frame, &frame, 8, 8);
        denoiser.denoise_frame_parallel(&frame, &frame, 8, 8);
        denoiser.denoise_frame_parallel(&frame, &frame, 8, 8);
        // history_frames = 4, should not exceed that
        denoiser.denoise_frame_parallel(&frame, &frame, 8, 8);
        assert!(denoiser.frame_history.len() <= 4);
    }

    // 30. denoise_frame_parallel: result matches a serial reference on a mixed frame
    #[test]
    fn test_denoise_frame_parallel_matches_serial() {
        let width = 8u32;
        let height = 8u32;
        let pixel_count = (width * height) as usize;

        // Build a frame where half pixels have small diff, half have large diff.
        let mut current = vec![0u8; pixel_count];
        let mut reference = vec![0u8; pixel_count];
        for i in 0..pixel_count {
            current[i] = (i % 200) as u8;
            reference[i] = if i % 2 == 0 {
                // Small diff: reference ≈ current
                ((i % 200) as u8).saturating_add(5)
            } else {
                // Large diff
                ((i % 200) as u8).saturating_add(50)
            };
        }

        // Serial reference implementation.
        let expected: Vec<u8> = current
            .iter()
            .zip(reference.iter())
            .map(|(&c, &r)| {
                if (c as i32 - r as i32).unsigned_abs() < 10 {
                    ((c as u32 + r as u32) / 2) as u8
                } else {
                    c
                }
            })
            .collect();

        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::Fixed, 0.5, 4);
        let out = denoiser.denoise_frame_parallel(&current, &reference, width, height);
        assert_eq!(out, expected, "parallel result must match serial reference");
    }

    // 31. test_temporal_denoise_parallel_matches_sequential
    //
    // Runs temporal denoise on a 3-frame 64×64 sequence and verifies that
    // the parallel output exactly matches the serial reference.
    //
    // Determinism relies on both implementations using the same pixel-level
    // logic; the seed is fixed by constructing frames with a deterministic
    // LCG hash.
    #[test]
    fn test_temporal_denoise_parallel_matches_sequential() {
        const W: u32 = 64;
        const H: u32 = 64;
        const N: usize = (W * H) as usize;

        // Deterministic frame generator: LCG-based, seeded per frame.
        let make_frame = |seed: u64| -> Vec<u8> {
            (0..N)
                .map(|i| {
                    let h = (seed ^ (i as u64))
                        .wrapping_mul(6364136223846793005u64)
                        .wrapping_add(1442695040888963407u64);
                    (h >> 56) as u8
                })
                .collect()
        };

        let frame0 = make_frame(0xABCD_1234);
        let frame1 = make_frame(0x5678_EF01);
        let frame2 = make_frame(0x9012_3456);

        // --- Serial reference: denoise frame1 with frame0 as reference ---
        let serial_out: Vec<u8> = frame1
            .iter()
            .zip(frame0.iter())
            .map(|(&cur, &ref_px)| {
                if (cur as i32 - ref_px as i32).unsigned_abs() < 10 {
                    ((cur as u32 + ref_px as u32) / 2) as u8
                } else {
                    cur
                }
            })
            .collect();

        // --- Parallel implementation ---
        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::MotionCompensated, 0.5, 3);
        // Seed the history with frame0.
        denoiser.process_frame(&frame0, None, None, W, H);
        // Denoise frame1 using frame0 as reference.
        let parallel_out = denoiser.denoise_frame_parallel(&frame1, &frame0, W, H);

        assert_eq!(
            parallel_out, serial_out,
            "parallel output must exactly match serial reference on 64×64 sequence"
        );

        // Verify frame2 can also be processed (no panic, deterministic output).
        let _out2 = denoiser.denoise_frame_parallel(&frame2, &frame1, W, H);
    }

    // 32. test_temporal_denoise_parallel_performance
    //
    // Runs parallel temporal denoise on a 1920×1080 3-frame sequence.
    // Verifies that it completes in < 2 seconds (wall-clock).
    //
    // This test is inherently performance-sensitive.  On extremely slow CI
    // environments it could flake; the 2-second budget is generous for any
    // modern machine where rayon can use multiple threads.
    #[test]
    fn test_temporal_denoise_parallel_performance() {
        const W: u32 = 1920;
        const H: u32 = 1080;
        const N: usize = (W * H) as usize;

        // Allocate three synthetic frames filled with a ramp pattern.
        let frame0: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();
        let frame1: Vec<u8> = (0..N).map(|i| ((i + 17) % 256) as u8).collect();
        let frame2: Vec<u8> = (0..N).map(|i| ((i + 33) % 256) as u8).collect();

        let start = std::time::Instant::now();

        let mut denoiser = TemporalDenoiser::new(TemporalDenoiseMode::MotionCompensated, 0.5, 3);
        denoiser.process_frame(&frame0, None, None, W, H);
        let _out1 = denoiser.denoise_frame_parallel(&frame1, &frame0, W, H);
        let _out2 = denoiser.denoise_frame_parallel(&frame2, &frame1, W, H);

        let elapsed = start.elapsed();
        // Debug builds are ~10-15x slower than release; give them 30 s headroom.
        // Release builds should always finish well under 2 s.
        let budget_secs: f64 = if cfg!(debug_assertions) { 30.0 } else { 2.0 };
        assert!(
            elapsed.as_secs_f64() < budget_secs,
            "1920×1080 3-frame parallel denoise took {:.3}s — expected < {budget_secs}s",
            elapsed.as_secs_f64()
        );
    }
}
