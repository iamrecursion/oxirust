//! Frame rate conversion and intermediate frame generation.
//!
//! Supports blend-based, motion-compensated, duplicate, and drop methods
//! for converting between arbitrary frame rates.

use crate::motion_compensation::{compensate_frame, MeAlgorithm, MotionEstimator};

/// Method used to synthesize intermediate or replacement frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameInterpolationMethod {
    /// Linear blend of adjacent frames.
    Blend,
    /// Motion-compensated interpolation.
    MotionBased,
    /// Repeat previous frame (no synthesis).
    Duplicate,
    /// Drop frames for rate reduction.
    Drop,
}

/// Frame rate converter.
pub struct FrameInterpolator {
    /// Source frame rate in frames per second.
    pub src_fps: f32,
    /// Destination frame rate in frames per second.
    pub dst_fps: f32,
    /// Interpolation method.
    pub method: FrameInterpolationMethod,
}

/// Result of a frame interpolation pass.
pub struct InterpResult {
    /// RGBA output frames (each is `width * height * 4` bytes).
    pub frames: Vec<Vec<u8>>,
    /// Presentation timestamps in milliseconds for each output frame.
    pub timestamps_ms: Vec<u64>,
}

impl FrameInterpolator {
    /// Create a new `FrameInterpolator`.
    pub fn new(src_fps: f32, dst_fps: f32, method: FrameInterpolationMethod) -> Self {
        Self {
            src_fps,
            dst_fps,
            method,
        }
    }

    /// Returns `dst_fps / src_fps`.
    pub fn ratio(&self) -> f32 {
        self.dst_fps / self.src_fps
    }

    /// Predict how many output frames will be produced for `input_count` source frames.
    pub fn output_frame_count(&self, input_count: usize) -> usize {
        let ratio = self.ratio();
        if ratio > 1.0 {
            let count = (input_count as f32 * ratio).round() as usize;
            count.max(input_count)
        } else {
            let count = (input_count as f32 * ratio).round() as usize;
            count.max(1)
        }
    }

    /// Interpolate a single pair of RGBA frames at fractional position `alpha` ∈ \[0,1\].
    ///
    /// - `alpha = 0.0` → `frame_a`
    /// - `alpha = 1.0` → `frame_b`
    pub fn interpolate_pair(
        &self,
        frame_a: &[u8],
        frame_b: &[u8],
        width: u32,
        height: u32,
        alpha: f32,
    ) -> Vec<u8> {
        match self.method {
            FrameInterpolationMethod::Blend
            | FrameInterpolationMethod::Duplicate
            | FrameInterpolationMethod::Drop => blend_rgba(frame_a, frame_b, alpha),
            FrameInterpolationMethod::MotionBased => {
                self.motion_based_interp(frame_a, frame_b, width, height, alpha)
            }
        }
    }

    /// Process a sequence of RGBA frames, converting from `src_fps` to `dst_fps`.
    pub fn process(&self, frames: &[Vec<u8>], width: u32, height: u32) -> InterpResult {
        if frames.is_empty() {
            return InterpResult {
                frames: Vec::new(),
                timestamps_ms: Vec::new(),
            };
        }

        let ratio = self.ratio();

        if (ratio - 1.0f32).abs() < 1e-4 {
            // Same rate — pass through with recalculated timestamps.
            let out_frames: Vec<Vec<u8>> = frames.to_vec();
            let timestamps: Vec<u64> = (0..out_frames.len())
                .map(|i| (i as f64 * 1000.0 / self.dst_fps as f64).round() as u64)
                .collect();
            return InterpResult {
                frames: out_frames,
                timestamps_ms: timestamps,
            };
        }

        if ratio > 1.0 {
            self.upsample(frames, width, height)
        } else {
            self.downsample(frames, width, height)
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn upsample(&self, frames: &[Vec<u8>], width: u32, height: u32) -> InterpResult {
        // Number of output frames to generate between consecutive source frames.
        // For ratio R, we need R-1 intermediate frames per source interval.
        let ratio = self.ratio();
        let mut out_frames: Vec<Vec<u8>> = Vec::new();
        let mut timestamps: Vec<u64> = Vec::new();

        let src_duration_ms = 1000.0f64 / self.src_fps as f64;
        let dst_duration_ms = 1000.0f64 / self.dst_fps as f64;

        // How many output frames fit within one source interval?
        let steps_per_src = (ratio).round() as usize;
        let steps_per_src = steps_per_src.max(1);

        let mut out_idx = 0usize;

        for (i, frame_a) in frames.iter().enumerate() {
            // Emit the source frame itself.
            let src_ts_ms = i as f64 * src_duration_ms;

            // Push source frame if its timestamp aligns with an output slot.
            let out_slot = (src_ts_ms / dst_duration_ms).round() as usize;
            // Ensure output index stays in sync.
            while out_idx < out_slot {
                out_idx += 1;
            }

            out_frames.push(frame_a.clone());
            timestamps.push((out_idx as f64 * dst_duration_ms).round() as u64);
            out_idx += 1;

            // Generate intermediate frames between frame_a and frame_b.
            if let Some(frame_b) = frames.get(i + 1) {
                for step in 1..steps_per_src {
                    let alpha = step as f32 / steps_per_src as f32;
                    let interp = self.interpolate_pair(frame_a, frame_b, width, height, alpha);
                    out_frames.push(interp);
                    timestamps.push((out_idx as f64 * dst_duration_ms).round() as u64);
                    out_idx += 1;
                }
            }
        }

        InterpResult {
            frames: out_frames,
            timestamps_ms: timestamps,
        }
    }

    fn downsample(&self, frames: &[Vec<u8>], _width: u32, _height: u32) -> InterpResult {
        let dst_duration_ms = 1000.0f64 / self.dst_fps as f64;
        let src_duration_ms = 1000.0f64 / self.src_fps as f64;

        let total_src_duration_ms = frames.len() as f64 * src_duration_ms;
        let total_out = (total_src_duration_ms / dst_duration_ms).round() as usize;
        let total_out = total_out.max(1);

        let mut out_frames: Vec<Vec<u8>> = Vec::with_capacity(total_out);
        let mut timestamps: Vec<u64> = Vec::with_capacity(total_out);

        for out_i in 0..total_out {
            let target_ms = out_i as f64 * dst_duration_ms;
            // Find nearest source frame.
            let src_idx = (target_ms / src_duration_ms).round() as usize;
            let src_idx = src_idx.min(frames.len() - 1);
            out_frames.push(frames[src_idx].clone());
            timestamps.push((target_ms).round() as u64);
        }

        InterpResult {
            frames: out_frames,
            timestamps_ms: timestamps,
        }
    }

    fn motion_based_interp(
        &self,
        frame_a: &[u8],
        frame_b: &[u8],
        width: u32,
        height: u32,
        alpha: f32,
    ) -> Vec<u8> {
        let pixels = (width * height) as usize;

        // Extract luma (R channel as proxy, stride = width*4 → luma stride = width).
        let luma_a: Vec<u8> = frame_a.chunks_exact(4).map(|p| p[0]).collect();
        let luma_b: Vec<u8> = frame_b.chunks_exact(4).map(|p| p[0]).collect();

        let block_size = 8u32;
        let search_range = 8i32;
        let estimator = MotionEstimator::new(block_size, search_range, MeAlgorithm::FullSearch);

        // Forward motion: ref=A, cur=B → vectors describe A→B motion.
        let fwd_vectors = estimator.estimate_frame(&luma_a, &luma_b, width, height);

        // Warp frame_a forward by alpha × motion.
        let scaled_fwd: Vec<_> = fwd_vectors
            .iter()
            .map(|mv| crate::motion_compensation::MotionVector {
                dx: ((mv.dx as f32 * alpha).round() as i16),
                dy: ((mv.dy as f32 * alpha).round() as i16),
                sad: mv.sad,
                block_x: mv.block_x,
                block_y: mv.block_y,
            })
            .collect();

        let pred_a_luma = compensate_frame(&luma_a, &scaled_fwd, width, height, block_size);

        // Backward motion: ref=B, cur=A → vectors describe B→A motion.
        let bwd_vectors = estimator.estimate_frame(&luma_b, &luma_a, width, height);
        let _scaled_bwd: Vec<_> = bwd_vectors
            .iter()
            .map(|mv| crate::motion_compensation::MotionVector {
                dx: ((mv.dx as f32 * (1.0 - alpha)).round() as i16),
                dy: ((mv.dy as f32 * (1.0 - alpha)).round() as i16),
                sad: mv.sad,
                block_x: mv.block_x,
                block_y: mv.block_y,
            })
            .collect();

        let pred_b_luma = compensate_frame(&luma_b, &_scaled_bwd, width, height, block_size);

        // Blend predictions and reconstruct RGBA.
        let mut out = vec![0u8; pixels * 4];
        for i in 0..pixels {
            let luma = ((pred_a_luma[i] as u16 + pred_b_luma[i] as u16) / 2) as u8;
            // For chroma channels, fall back to a simple blend of original RGBA.
            let base = i * 4;
            out[base] = luma;
            for ch in 1..4 {
                let a = frame_a.get(base + ch).copied().unwrap_or(0);
                let b = frame_b.get(base + ch).copied().unwrap_or(0);
                out[base + ch] = blend_channel(a, b, alpha);
            }
        }
        out
    }
}

// -----------------------------------------------------------------------
// Free functions
// -----------------------------------------------------------------------

/// Per-channel linear blend: `a*(1-alpha) + b*alpha`.
#[inline]
fn blend_channel(a: u8, b: u8, alpha: f32) -> u8 {
    let v = a as f32 * (1.0 - alpha) + b as f32 * alpha;
    v.round().clamp(0.0, 255.0) as u8
}

/// Element-wise RGBA blend.
fn blend_rgba(frame_a: &[u8], frame_b: &[u8], alpha: f32) -> Vec<u8> {
    frame_a
        .iter()
        .zip(frame_b.iter())
        .map(|(&a, &b)| blend_channel(a, b, alpha))
        .collect()
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(width * height * 4);
        for _ in 0..(width * height) {
            v.push(r);
            v.push(g);
            v.push(b);
            v.push(a);
        }
        v
    }

    // 1. FrameInterpolator::new sets fields correctly
    #[test]
    fn test_new_sets_fields() {
        let fi = FrameInterpolator::new(24.0, 60.0, FrameInterpolationMethod::Blend);
        assert!((fi.src_fps - 24.0).abs() < 1e-4);
        assert!((fi.dst_fps - 60.0).abs() < 1e-4);
        assert_eq!(fi.method, FrameInterpolationMethod::Blend);
    }

    // 2. ratio() for upconvert 24→60
    #[test]
    fn test_ratio_upconvert() {
        let fi = FrameInterpolator::new(24.0, 60.0, FrameInterpolationMethod::Blend);
        assert!((fi.ratio() - 2.5).abs() < 1e-3);
    }

    // 3. ratio() for downconvert 60→24
    #[test]
    fn test_ratio_downconvert() {
        let fi = FrameInterpolator::new(60.0, 24.0, FrameInterpolationMethod::Blend);
        assert!((fi.ratio() - 0.4).abs() < 1e-3);
    }

    // 4. output_frame_count for upconvert
    #[test]
    fn test_output_frame_count_upconvert() {
        let fi = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
        // ratio = 2.0 → 10 input → 20 output
        assert_eq!(fi.output_frame_count(10), 20);
    }

    // 5. output_frame_count for downconvert
    #[test]
    fn test_output_frame_count_downconvert() {
        let fi = FrameInterpolator::new(60.0, 30.0, FrameInterpolationMethod::Blend);
        // ratio = 0.5 → 10 input → 5 output
        assert_eq!(fi.output_frame_count(10), 5);
    }

    // 6. interpolate_pair Blend alpha=0 returns frame_a
    #[test]
    fn test_blend_alpha_zero_returns_a() {
        let fa = make_rgba(4, 4, 100, 150, 200, 255);
        let fb = make_rgba(4, 4, 0, 0, 0, 0);
        let fi = FrameInterpolator::new(24.0, 60.0, FrameInterpolationMethod::Blend);
        let out = fi.interpolate_pair(&fa, &fb, 4, 4, 0.0);
        assert_eq!(out, fa);
    }

    // 7. interpolate_pair Blend alpha=1.0 returns frame_b
    #[test]
    fn test_blend_alpha_one_returns_b() {
        let fa = make_rgba(4, 4, 0, 0, 0, 0);
        let fb = make_rgba(4, 4, 200, 100, 50, 255);
        let fi = FrameInterpolator::new(24.0, 60.0, FrameInterpolationMethod::Blend);
        let out = fi.interpolate_pair(&fa, &fb, 4, 4, 1.0);
        assert_eq!(out, fb);
    }

    // 8. interpolate_pair Blend alpha=0.5 returns midpoint
    #[test]
    fn test_blend_alpha_half_midpoint() {
        let fa = make_rgba(2, 2, 100, 100, 100, 100);
        let fb = make_rgba(2, 2, 200, 200, 200, 200);
        let fi = FrameInterpolator::new(24.0, 60.0, FrameInterpolationMethod::Blend);
        let out = fi.interpolate_pair(&fa, &fb, 2, 2, 0.5);
        // Each channel: 100*0.5 + 200*0.5 = 150
        for &px in &out {
            assert_eq!(px, 150u8);
        }
    }

    // 9. interpolate_pair MotionBased identical frames returns same frame
    #[test]
    fn test_motion_based_identical_frames() {
        let fa = make_rgba(16, 16, 80, 120, 160, 255);
        let fi = FrameInterpolator::new(24.0, 60.0, FrameInterpolationMethod::MotionBased);
        let out = fi.interpolate_pair(&fa, &fa, 16, 16, 0.5);
        // R channel (used as luma) should remain 80 after identity motion
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], 80u8);
        }
    }

    // 10. process upconvert produces more frames than input
    #[test]
    fn test_process_upconvert_more_frames() {
        let fa = make_rgba(8, 8, 50, 50, 50, 255);
        let fb = make_rgba(8, 8, 100, 100, 100, 255);
        let frames = vec![fa, fb];
        let fi = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
        let result = fi.process(&frames, 8, 8);
        assert!(result.frames.len() > frames.len());
    }

    // 11. process downconvert produces fewer frames than input
    #[test]
    fn test_process_downconvert_fewer_frames() {
        let frames: Vec<Vec<u8>> = (0..10)
            .map(|i| make_rgba(4, 4, i * 25, i * 25, i * 25, 255))
            .collect();
        let fi = FrameInterpolator::new(60.0, 24.0, FrameInterpolationMethod::Drop);
        let result = fi.process(&frames, 4, 4);
        assert!(result.frames.len() < frames.len());
    }

    // 12. InterpResult timestamps are monotonically increasing
    #[test]
    fn test_process_timestamps_monotonic() {
        let frames: Vec<Vec<u8>> = (0..5)
            .map(|_| make_rgba(4, 4, 128, 128, 128, 255))
            .collect();
        let fi = FrameInterpolator::new(24.0, 48.0, FrameInterpolationMethod::Blend);
        let result = fi.process(&frames, 4, 4);
        for w in result.timestamps_ms.windows(2) {
            assert!(
                w[1] >= w[0],
                "timestamps not monotonic: {} >= {}",
                w[1],
                w[0]
            );
        }
    }
}
