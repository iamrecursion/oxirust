//! Slow-motion frame interpolation.
//!
//! Provides blend-based, motion-compensated, and flow-based methods for
//! generating intermediate frames to achieve slow-motion playback at
//! higher target frame rates.
//!
//! Features:
//! - Optical flow-based frame synthesis (bilinear warping with forward/backward flow)
//! - Variable speed ramp curves (ease-in, ease-out, ease-in-out, custom Bezier)
//! - Frame blending with adaptive weight (motion-aware blending)
//! - Speed ramp keyframe interpolation

use crate::motion_compensation::{compensate_frame, MeAlgorithm, MotionEstimator, MotionVector};

/// Method used to generate intermediate frames for slow-motion effects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Simple per-pixel linear blend of adjacent frames.
    Blend,
    /// SAD block-matching motion compensation (block=8, search=8).
    MotionCompensated,
    /// Fine-grained optical-flow-like estimation (block=4, search=4).
    FlowBased,
}

/// Configuration for slow-motion processing.
#[derive(Debug, Clone, PartialEq)]
pub struct SlowMotionConfig {
    /// Original capture frame rate (e.g., 240 fps for high-speed camera).
    pub source_fps: f32,
    /// Desired playback frame rate (e.g., 24 fps for slow-motion output).
    pub target_fps: f32,
    /// Interpolation method to use.
    pub method: InterpolationMethod,
}

/// Processor that generates interpolated frames for slow-motion playback.
pub struct SlowMotionProcessor {
    /// Configuration used by this processor.
    pub config: SlowMotionConfig,
}

impl SlowMotionProcessor {
    /// Create a new `SlowMotionProcessor` with the given configuration.
    pub fn new(config: SlowMotionConfig) -> Self {
        Self { config }
    }

    /// Returns `target_fps / source_fps`.
    ///
    /// Values less than 1.0 indicate genuine slow-motion (target is slower
    /// than source); values greater than 1.0 indicate speed-up.
    pub fn slowdown_factor(&self) -> f32 {
        self.config.target_fps / self.config.source_fps
    }

    /// Interpolate an intermediate RGBA frame at temporal position `t` in [0, 1].
    ///
    /// - `t = 0.0` -> returns a copy of `frame_a`
    /// - `t = 1.0` -> returns a copy of `frame_b`
    ///
    /// `frame_a` and `frame_b` must each be `width * height * 4` bytes (RGBA).
    pub fn interpolate(
        &self,
        frame_a: &[u8],
        frame_b: &[u8],
        t: f32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let t_clamped = t.clamp(0.0, 1.0);
        match self.config.method {
            InterpolationMethod::Blend => blend_rgba(frame_a, frame_b, t_clamped),
            InterpolationMethod::MotionCompensated => {
                motion_compensated_interp(frame_a, frame_b, t_clamped, width, height, 8, 8)
            }
            InterpolationMethod::FlowBased => {
                motion_compensated_interp(frame_a, frame_b, t_clamped, width, height, 4, 4)
            }
        }
    }

    /// Interpolate using optical-flow-based bilinear warping.
    ///
    /// This performs bidirectional flow estimation, then bilinearly warps both
    /// frames and blends them proportionally to `t`.
    pub fn interpolate_optical_flow(
        &self,
        frame_a: &[u8],
        frame_b: &[u8],
        t: f32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let t_clamped = t.clamp(0.0, 1.0);
        optical_flow_interp(frame_a, frame_b, t_clamped, width, height)
    }

    /// Interpolate with motion-adaptive blending weights.
    ///
    /// Areas with high motion get more weight from the motion-compensated
    /// prediction, while static areas use simple blending to avoid artifacts.
    pub fn interpolate_adaptive(
        &self,
        frame_a: &[u8],
        frame_b: &[u8],
        t: f32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let t_clamped = t.clamp(0.0, 1.0);
        adaptive_blend_interp(frame_a, frame_b, t_clamped, width, height)
    }
}

// ---------------------------------------------------------------------------
// Speed ramp curves
// ---------------------------------------------------------------------------

/// Types of speed ramp easing curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedRampCurve {
    /// Linear (no easing).
    Linear,
    /// Ease-in: starts slow, accelerates. `t^2`
    EaseIn,
    /// Ease-out: starts fast, decelerates. `1-(1-t)^2`
    EaseOut,
    /// Ease-in-out: slow start and end. Smooth hermite `3t^2 - 2t^3`
    EaseInOut,
    /// Custom cubic Bezier defined by two control points (x1,y1), (x2,y2).
    CubicBezier {
        /// First control point x.
        x1: f32,
        /// First control point y.
        y1: f32,
        /// Second control point x.
        x2: f32,
        /// Second control point y.
        y2: f32,
    },
}

/// Evaluate a speed ramp curve at normalized time `t` in [0, 1].
///
/// Returns the remapped time in [0, 1].
pub fn evaluate_speed_ramp(curve: SpeedRampCurve, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        SpeedRampCurve::Linear => t,
        SpeedRampCurve::EaseIn => t * t,
        SpeedRampCurve::EaseOut => {
            let inv = 1.0 - t;
            1.0 - inv * inv
        }
        SpeedRampCurve::EaseInOut => {
            // Smoothstep: 3t^2 - 2t^3
            t * t * (3.0 - 2.0 * t)
        }
        SpeedRampCurve::CubicBezier { x1, y1, x2, y2 } => evaluate_cubic_bezier(t, x1, y1, x2, y2),
    }
}

/// Evaluate a cubic Bezier curve (approximation via iterative Newton method).
///
/// The curve goes from (0,0) to (1,1) with control points (x1,y1) and (x2,y2).
/// We solve for the parameter `s` such that `bezier_x(s) = t`, then return `bezier_y(s)`.
fn evaluate_cubic_bezier(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    // Find s such that x(s) = t using Newton's method
    let mut s = t; // initial guess
    for _ in 0..8 {
        let x = cubic_bezier_component(s, x1, x2);
        let dx = cubic_bezier_derivative(s, x1, x2);
        if dx.abs() < 1e-9 {
            break;
        }
        s -= (x - t) / dx;
        s = s.clamp(0.0, 1.0);
    }
    cubic_bezier_component(s, y1, y2)
}

/// Evaluate one component of a cubic Bezier: `3(1-s)^2*s*p1 + 3(1-s)*s^2*p2 + s^3`
#[inline]
fn cubic_bezier_component(s: f32, p1: f32, p2: f32) -> f32 {
    let inv_s = 1.0 - s;
    3.0 * inv_s * inv_s * s * p1 + 3.0 * inv_s * s * s * p2 + s * s * s
}

/// Derivative of one component of a cubic Bezier.
#[inline]
fn cubic_bezier_derivative(s: f32, p1: f32, p2: f32) -> f32 {
    let inv_s = 1.0 - s;
    3.0 * inv_s * inv_s * p1 + 6.0 * inv_s * s * (p2 - p1) + 3.0 * s * s * (1.0 - p2)
}

// ---------------------------------------------------------------------------
// Speed ramp keyframe interpolation
// ---------------------------------------------------------------------------

/// A keyframe defining speed at a specific time.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeedKeyframe {
    /// Time in seconds.
    pub time: f32,
    /// Speed multiplier at this keyframe (1.0 = normal, 0.5 = half speed, 2.0 = double).
    pub speed: f32,
    /// Easing curve to use when interpolating toward the next keyframe.
    pub curve: SpeedRampCurve,
}

/// Interpolator for speed ramp keyframes.
pub struct SpeedRampInterpolator {
    /// Sorted keyframes.
    keyframes: Vec<SpeedKeyframe>,
}

impl SpeedRampInterpolator {
    /// Create a new interpolator from a set of keyframes.
    ///
    /// Keyframes are sorted by time. Returns `None` if empty.
    pub fn new(mut keyframes: Vec<SpeedKeyframe>) -> Option<Self> {
        if keyframes.is_empty() {
            return None;
        }
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(Self { keyframes })
    }

    /// Get the interpolated speed at a given time.
    pub fn speed_at(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 1.0;
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return self.keyframes[0].speed;
        }

        // After last keyframe
        if let Some(last) = self.keyframes.last() {
            if time >= last.time {
                return last.speed;
            }
        }

        // Find the bracketing keyframes
        for i in 0..self.keyframes.len() - 1 {
            let kf0 = &self.keyframes[i];
            let kf1 = &self.keyframes[i + 1];
            if time >= kf0.time && time <= kf1.time {
                let duration = kf1.time - kf0.time;
                if duration < 1e-9 {
                    return kf0.speed;
                }
                let local_t = (time - kf0.time) / duration;
                let eased_t = evaluate_speed_ramp(kf0.curve, local_t);
                return kf0.speed + (kf1.speed - kf0.speed) * eased_t;
            }
        }

        // Fallback
        self.keyframes.last().map_or(1.0, |kf| kf.speed)
    }

    /// Compute the accumulated time (in source timeline) for output time,
    /// integrating the variable speed curve.
    ///
    /// Uses trapezoidal integration with the given number of steps.
    pub fn accumulated_source_time(&self, output_time: f32, steps: u32) -> f32 {
        if steps == 0 {
            return 0.0;
        }
        let dt = output_time / steps as f32;
        let mut acc = 0.0f64;
        let mut prev_speed = self.speed_at(0.0) as f64;

        for i in 1..=steps {
            let t = dt * i as f32;
            let curr_speed = self.speed_at(t) as f64;
            acc += (prev_speed + curr_speed) * 0.5 * dt as f64;
            prev_speed = curr_speed;
        }

        acc as f32
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Per-channel linear blend: `a*(1-t) + b*t`, rounded and clamped.
#[inline]
fn blend_channel(a: u8, b: u8, t: f32) -> u8 {
    let v = a as f32 * (1.0 - t) + b as f32 * t;
    v.round().clamp(0.0, 255.0) as u8
}

/// Element-wise RGBA blend of two frames.
fn blend_rgba(frame_a: &[u8], frame_b: &[u8], t: f32) -> Vec<u8> {
    let len = frame_a.len().min(frame_b.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(blend_channel(
            *frame_a.get(i).unwrap_or(&0),
            *frame_b.get(i).unwrap_or(&0),
            t,
        ));
    }
    out
}

/// Extract luma from RGBA by treating the R channel as a luminance proxy.
#[inline]
fn extract_luma(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).map(|p| p[0]).collect()
}

/// Warp `frame` using scaled motion vectors, returning a new RGBA frame.
fn warp_frame(
    frame: &[u8],
    mvs: &[MotionVector],
    width: u32,
    height: u32,
    block_size: u32,
    scale: f32,
) -> Vec<u8> {
    let blocks_x = (width + block_size - 1) / block_size;
    let pixels = (width * height) as usize;
    let mut out = vec![0u8; pixels * 4];

    for mv in mvs {
        let scaled_dx = (mv.dx as f32 * scale).round() as i32;
        let scaled_dy = (mv.dy as f32 * scale).round() as i32;

        let bx = mv.block_x;
        let by = mv.block_y;
        let bw = block_size.min(width.saturating_sub(bx));
        let bh = block_size.min(height.saturating_sub(by));

        for row in 0..bh {
            for col in 0..bw {
                let dst_x = bx + col;
                let dst_y = by + row;

                // Source pixel = destination - motion_vector (inverse warp)
                let src_x = (dst_x as i32 - scaled_dx).clamp(0, width as i32 - 1) as u32;
                let src_y = (dst_y as i32 - scaled_dy).clamp(0, height as i32 - 1) as u32;

                let src_idx = (src_y * width + src_x) as usize * 4;
                let dst_idx = (dst_y * width + dst_x) as usize * 4;

                for ch in 0..4 {
                    out[dst_idx + ch] = *frame.get(src_idx + ch).unwrap_or(&0);
                }
            }
        }
    }

    let _ = blocks_x; // suppress unused warning
    out
}

/// Bilinear warp using per-pixel flow vectors (sub-pixel accuracy).
fn bilinear_warp(
    frame: &[u8],
    flow_x: &[f32],
    flow_y: &[f32],
    width: u32,
    height: u32,
    scale: f32,
) -> Vec<u8> {
    let pixels = (width * height) as usize;
    let mut out = vec![0u8; pixels * 4];
    let w_max = (width.saturating_sub(1)) as f32;
    let h_max = (height.saturating_sub(1)) as f32;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let fx = flow_x.get(idx).copied().unwrap_or(0.0) * scale;
            let fy = flow_y.get(idx).copied().unwrap_or(0.0) * scale;

            let sx = (x as f32 - fx).clamp(0.0, w_max);
            let sy = (y as f32 - fy).clamp(0.0, h_max);

            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(width.saturating_sub(1));
            let y1 = (y0 + 1).min(height.saturating_sub(1));
            let frac_x = sx - x0 as f32;
            let frac_y = sy - y0 as f32;

            let dst_idx = idx * 4;
            for ch in 0..4 {
                let i00 = (y0 * width + x0) as usize * 4 + ch;
                let i10 = (y0 * width + x1) as usize * 4 + ch;
                let i01 = (y1 * width + x0) as usize * 4 + ch;
                let i11 = (y1 * width + x1) as usize * 4 + ch;

                let v00 = *frame.get(i00).unwrap_or(&0) as f32;
                let v10 = *frame.get(i10).unwrap_or(&0) as f32;
                let v01 = *frame.get(i01).unwrap_or(&0) as f32;
                let v11 = *frame.get(i11).unwrap_or(&0) as f32;

                let top = v00 * (1.0 - frac_x) + v10 * frac_x;
                let bot = v01 * (1.0 - frac_x) + v11 * frac_x;
                let val = top * (1.0 - frac_y) + bot * frac_y;
                out[dst_idx + ch] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

/// Estimate per-pixel dense optical flow from block-based MVs.
///
/// Spreads block MVs to pixel level with bilinear interpolation between blocks.
fn estimate_dense_flow(
    mvs: &[MotionVector],
    width: u32,
    height: u32,
    block_size: u32,
) -> (Vec<f32>, Vec<f32>) {
    let pixels = (width * height) as usize;
    let mut flow_x = vec![0.0f32; pixels];
    let mut flow_y = vec![0.0f32; pixels];

    let blocks_x = (width + block_size - 1) / block_size;
    let blocks_y = (height + block_size - 1) / block_size;

    // Create block-level flow map
    let n_blocks = (blocks_x * blocks_y) as usize;
    let mut block_dx = vec![0.0f32; n_blocks];
    let mut block_dy = vec![0.0f32; n_blocks];

    for mv in mvs {
        let bx_idx = mv.block_x / block_size;
        let by_idx = mv.block_y / block_size;
        if bx_idx < blocks_x && by_idx < blocks_y {
            let bi = (by_idx * blocks_x + bx_idx) as usize;
            if bi < n_blocks {
                block_dx[bi] = mv.dx as f32;
                block_dy[bi] = mv.dy as f32;
            }
        }
    }

    // Spread to pixel level with simple nearest-block assignment
    let half_block = block_size as f32 / 2.0;
    for y in 0..height {
        for x in 0..width {
            let bx_f = (x as f32 + half_block) / block_size as f32 - 0.5;
            let by_f = (y as f32 + half_block) / block_size as f32 - 0.5;

            let bx0 = (bx_f.floor() as i32).clamp(0, blocks_x as i32 - 1) as u32;
            let by0 = (by_f.floor() as i32).clamp(0, blocks_y as i32 - 1) as u32;
            let bx1 = (bx0 + 1).min(blocks_x.saturating_sub(1));
            let by1 = (by0 + 1).min(blocks_y.saturating_sub(1));

            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);
            let fy = (by_f - by0 as f32).clamp(0.0, 1.0);

            let i00 = (by0 * blocks_x + bx0) as usize;
            let i10 = (by0 * blocks_x + bx1) as usize;
            let i01 = (by1 * blocks_x + bx0) as usize;
            let i11 = (by1 * blocks_x + bx1) as usize;

            let dx = block_dx.get(i00).copied().unwrap_or(0.0) * (1.0 - fx) * (1.0 - fy)
                + block_dx.get(i10).copied().unwrap_or(0.0) * fx * (1.0 - fy)
                + block_dx.get(i01).copied().unwrap_or(0.0) * (1.0 - fx) * fy
                + block_dx.get(i11).copied().unwrap_or(0.0) * fx * fy;
            let dy = block_dy.get(i00).copied().unwrap_or(0.0) * (1.0 - fx) * (1.0 - fy)
                + block_dy.get(i10).copied().unwrap_or(0.0) * fx * (1.0 - fy)
                + block_dy.get(i01).copied().unwrap_or(0.0) * (1.0 - fx) * fy
                + block_dy.get(i11).copied().unwrap_or(0.0) * fx * fy;

            let pi = (y * width + x) as usize;
            flow_x[pi] = dx;
            flow_y[pi] = dy;
        }
    }

    (flow_x, flow_y)
}

/// Optical-flow-based interpolation using bilinear warping.
fn optical_flow_interp(frame_a: &[u8], frame_b: &[u8], t: f32, width: u32, height: u32) -> Vec<u8> {
    let block_size = 4u32;
    let search_range = 4i32;
    let luma_a = extract_luma(frame_a);
    let luma_b = extract_luma(frame_b);

    let estimator = MotionEstimator::new(block_size, search_range, MeAlgorithm::FullSearch);
    let fwd_mvs = estimator.estimate_frame(&luma_a, &luma_b, width, height);
    let bwd_mvs = estimator.estimate_frame(&luma_b, &luma_a, width, height);

    // Dense flow
    let (fwd_fx, fwd_fy) = estimate_dense_flow(&fwd_mvs, width, height, block_size);
    let (bwd_fx, bwd_fy) = estimate_dense_flow(&bwd_mvs, width, height, block_size);

    // Bilinear warp both directions
    let warped_a = bilinear_warp(frame_a, &fwd_fx, &fwd_fy, width, height, t);
    let warped_b = bilinear_warp(frame_b, &bwd_fx, &bwd_fy, width, height, 1.0 - t);

    // Blend
    let pixels = (width * height) as usize;
    let mut out = vec![0u8; pixels * 4];
    for i in 0..pixels * 4 {
        let a = *warped_a.get(i).unwrap_or(&0);
        let b = *warped_b.get(i).unwrap_or(&0);
        out[i] = blend_channel(a, b, t);
    }
    out
}

/// Adaptive blend interpolation: motion-aware blending weights.
///
/// High-motion regions use motion-compensated prediction; low-motion regions
/// use simple blending to reduce artifacts.
fn adaptive_blend_interp(
    frame_a: &[u8],
    frame_b: &[u8],
    t: f32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let pixels = (width * height) as usize;

    // Simple blend
    let simple = blend_rgba(frame_a, frame_b, t);

    // Motion-compensated blend
    let mc = motion_compensated_interp(frame_a, frame_b, t, width, height, 8, 8);

    // Per-pixel motion magnitude estimation (difference between frames)
    let mut out = vec![0u8; pixels * 4];
    for p in 0..pixels {
        let base = p * 4;

        // Estimate local motion from frame difference (R channel as proxy)
        let diff = (frame_a.get(base).copied().unwrap_or(0) as f32
            - frame_b.get(base).copied().unwrap_or(0) as f32)
            .abs();

        // Motion weight: higher difference → prefer MC, lower → prefer simple
        // Sigmoid-like: weight = diff / (diff + threshold)
        let threshold = 30.0f32;
        let mc_weight = diff / (diff + threshold);

        for ch in 0..4 {
            let s_val = *simple.get(base + ch).unwrap_or(&0) as f32;
            let m_val = *mc.get(base + ch).unwrap_or(&0) as f32;
            let blended = s_val * (1.0 - mc_weight) + m_val * mc_weight;
            out[base + ch] = blended.round().clamp(0.0, 255.0) as u8;
        }
    }

    out
}

/// Core motion-compensated interpolation used by both MotionCompensated and FlowBased.
fn motion_compensated_interp(
    frame_a: &[u8],
    frame_b: &[u8],
    t: f32,
    width: u32,
    height: u32,
    block_size: u32,
    search_range: i32,
) -> Vec<u8> {
    let pixels = (width * height) as usize;

    let luma_a = extract_luma(frame_a);
    let luma_b = extract_luma(frame_b);

    let estimator = MotionEstimator::new(block_size, search_range, MeAlgorithm::FullSearch);

    // Forward vectors: A->B (where did each block in A move to in B?)
    let fwd_vecs = estimator.estimate_frame(&luma_a, &luma_b, width, height);
    // Backward vectors: B->A
    let bwd_vecs = estimator.estimate_frame(&luma_b, &luma_a, width, height);

    // Build scaled forward and backward MVs
    let scaled_fwd: Vec<MotionVector> = fwd_vecs
        .iter()
        .map(|mv| MotionVector {
            dx: (mv.dx as f32 * t).round() as i16,
            dy: (mv.dy as f32 * t).round() as i16,
            sad: mv.sad,
            block_x: mv.block_x,
            block_y: mv.block_y,
        })
        .collect();

    let scaled_bwd: Vec<MotionVector> = bwd_vecs
        .iter()
        .map(|mv| MotionVector {
            dx: (mv.dx as f32 * (1.0 - t)).round() as i16,
            dy: (mv.dy as f32 * (1.0 - t)).round() as i16,
            sad: mv.sad,
            block_x: mv.block_x,
            block_y: mv.block_y,
        })
        .collect();

    // Warp luma planes (used for quality reference but we reconstruct in RGBA)
    let _pred_a_luma = compensate_frame(&luma_a, &scaled_fwd, width, height, block_size);
    let _pred_b_luma = compensate_frame(&luma_b, &scaled_bwd, width, height, block_size);

    // Warp full RGBA frames
    let warped_a = warp_frame(frame_a, &scaled_fwd, width, height, block_size, 1.0);
    let warped_b = warp_frame(frame_b, &scaled_bwd, width, height, block_size, 1.0);

    // Blend the two warped frames
    let mut out = vec![0u8; pixels * 4];
    for i in 0..pixels * 4 {
        let a = *warped_a.get(i).unwrap_or(&0);
        let b = *warped_b.get(i).unwrap_or(&0);
        out[i] = blend_channel(a, b, t);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(width * height * 4);
        for _ in 0..(width * height) {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    fn make_config(method: InterpolationMethod) -> SlowMotionConfig {
        SlowMotionConfig {
            source_fps: 240.0,
            target_fps: 24.0,
            method,
        }
    }

    // 1. Constructor sets config correctly
    #[test]
    fn test_new_sets_config() {
        let cfg = make_config(InterpolationMethod::Blend);
        let proc = SlowMotionProcessor::new(cfg.clone());
        assert_eq!(proc.config, cfg);
    }

    // 2. slowdown_factor for 24->96
    #[test]
    fn test_slowdown_factor() {
        let cfg = SlowMotionConfig {
            source_fps: 24.0,
            target_fps: 96.0,
            method: InterpolationMethod::Blend,
        };
        let proc = SlowMotionProcessor::new(cfg);
        assert!((proc.slowdown_factor() - 4.0).abs() < 1e-4);
    }

    // 3. slowdown_factor for 60->30
    #[test]
    fn test_slowdown_factor_half() {
        let cfg = SlowMotionConfig {
            source_fps: 60.0,
            target_fps: 30.0,
            method: InterpolationMethod::Blend,
        };
        let proc = SlowMotionProcessor::new(cfg);
        assert!((proc.slowdown_factor() - 0.5).abs() < 1e-4);
    }

    // 4. Blend at t=0 returns frame_a
    #[test]
    fn test_blend_t_zero_returns_a() {
        let fa = make_rgba(4, 4, 100, 150, 200, 255);
        let fb = make_rgba(4, 4, 0, 0, 0, 0);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&fa, &fb, 0.0, 4, 4);
        assert_eq!(out, fa);
    }

    // 5. Blend at t=1 returns frame_b
    #[test]
    fn test_blend_t_one_returns_b() {
        let fa = make_rgba(4, 4, 0, 0, 0, 0);
        let fb = make_rgba(4, 4, 200, 100, 50, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&fa, &fb, 1.0, 4, 4);
        assert_eq!(out, fb);
    }

    // 6. Blend at t=0.5 gives midpoint for uniform frames
    #[test]
    fn test_blend_t_half_midpoint() {
        let fa = make_rgba(2, 2, 100, 100, 100, 100);
        let fb = make_rgba(2, 2, 200, 200, 200, 200);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&fa, &fb, 0.5, 2, 2);
        for &px in &out {
            assert_eq!(px, 150u8);
        }
    }

    // 7. Output size is correct for Blend
    #[test]
    fn test_blend_output_size() {
        let fa = make_rgba(8, 6, 50, 50, 50, 255);
        let fb = make_rgba(8, 6, 100, 100, 100, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&fa, &fb, 0.3, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 8. All 4 RGBA channels are blended
    #[test]
    fn test_blend_all_channels_blended() {
        let fa = vec![0u8, 0, 0, 0];
        let fb = vec![200u8, 100, 50, 255];
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&fa, &fb, 0.5, 1, 1);
        assert_eq!(out.len(), 4);
        // Each channel: 0*0.5 + value*0.5
        assert_eq!(out[0], 100); // R: 200/2
        assert_eq!(out[1], 50); // G: 100/2
        assert_eq!(out[2], 25); // B: 50/2
        assert_eq!(out[3], 128); // A: 255/2 = 128
    }

    // 9. MotionCompensated on identical frames preserves R-channel values
    #[test]
    fn test_motion_compensated_identical_frames() {
        let fa = make_rgba(16, 16, 80, 120, 160, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::MotionCompensated));
        let out = proc.interpolate(&fa, &fa, 0.5, 16, 16);
        // With zero motion on identical frames, output should be close to original
        for chunk in out.chunks_exact(4) {
            let diff = (chunk[0] as i32 - 80i32).abs();
            assert!(diff <= 5, "R channel diff too large: {}", diff);
        }
    }

    // 10. MotionCompensated output size is correct
    #[test]
    fn test_motion_compensated_output_size() {
        let fa = make_rgba(16, 16, 50, 50, 50, 255);
        let fb = make_rgba(16, 16, 100, 100, 100, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::MotionCompensated));
        let out = proc.interpolate(&fa, &fb, 0.5, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 4);
    }

    // 11. MotionCompensated at t=0 result is close to frame_a
    #[test]
    fn test_motion_compensated_t_zero() {
        let fa = make_rgba(16, 16, 200, 100, 50, 255);
        let fb = make_rgba(16, 16, 50, 200, 150, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::MotionCompensated));
        let out = proc.interpolate(&fa, &fb, 0.0, 16, 16);
        // At t=0, all scaling is 0 so we warp by (0,0) which should give frame_a
        for (i, chunk) in out.chunks_exact(4).enumerate() {
            let fa_chunk = &fa[i * 4..i * 4 + 4];
            // R channel should match frame_a (luma proxy)
            let diff = (chunk[0] as i32 - fa_chunk[0] as i32).abs();
            assert!(diff <= 5, "pixel {} R diff too large: {}", i, diff);
        }
    }

    // 12. FlowBased on identical frames preserves values
    #[test]
    fn test_flow_based_identical_frames() {
        let fa = make_rgba(16, 16, 60, 90, 120, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::FlowBased));
        let out = proc.interpolate(&fa, &fa, 0.5, 16, 16);
        for chunk in out.chunks_exact(4) {
            let diff = (chunk[0] as i32 - 60i32).abs();
            assert!(diff <= 5, "R channel diff: {}", diff);
        }
    }

    // 13. FlowBased output size is correct
    #[test]
    fn test_flow_based_output_size() {
        let fa = make_rgba(16, 16, 50, 50, 50, 255);
        let fb = make_rgba(16, 16, 100, 100, 100, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::FlowBased));
        let out = proc.interpolate(&fa, &fb, 0.5, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 4);
    }

    // 14. No channel value exceeds 255 or below 0 (ensured by u8)
    #[test]
    fn test_interpolate_clamps_to_u8() {
        let fa = make_rgba(8, 8, 255, 255, 255, 255);
        let fb = make_rgba(8, 8, 0, 0, 0, 0);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&fa, &fb, 0.7, 8, 8);
        // Since it's u8, all values are already in [0,255]; just verify no overflow
        assert_eq!(out.len(), 8 * 8 * 4);
        for &v in &out {
            // v is u8 so always <= 255; this is a compile-time guarantee
            let _ = v;
        }
    }

    // 15. Blend uniform white->black at t=0.5 -> ~128
    #[test]
    fn test_blend_uniform_white_to_black_half() {
        let white = make_rgba(4, 4, 255, 255, 255, 255);
        let black = make_rgba(4, 4, 0, 0, 0, 0);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate(&white, &black, 0.5, 4, 4);
        for &px in &out {
            assert!((px as i32 - 128).abs() <= 1, "expected ~128, got {}", px);
        }
    }

    // 16. InterpolationMethod derives Debug
    #[test]
    fn test_interpolation_method_debug() {
        let m = InterpolationMethod::MotionCompensated;
        let s = format!("{:?}", m);
        assert!(s.contains("MotionCompensated"));
    }

    // 17. SlowMotionConfig can be cloned
    #[test]
    fn test_slow_motion_config_clone() {
        let cfg = SlowMotionConfig {
            source_fps: 120.0,
            target_fps: 30.0,
            method: InterpolationMethod::FlowBased,
        };
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    // === NEW TESTS ===

    // 18. Speed ramp: Linear identity
    #[test]
    fn test_speed_ramp_linear() {
        assert!((evaluate_speed_ramp(SpeedRampCurve::Linear, 0.0)).abs() < 1e-6);
        assert!((evaluate_speed_ramp(SpeedRampCurve::Linear, 0.5) - 0.5).abs() < 1e-6);
        assert!((evaluate_speed_ramp(SpeedRampCurve::Linear, 1.0) - 1.0).abs() < 1e-6);
    }

    // 19. Speed ramp: EaseIn starts slow
    #[test]
    fn test_speed_ramp_ease_in() {
        let mid = evaluate_speed_ramp(SpeedRampCurve::EaseIn, 0.5);
        assert!(mid < 0.5, "EaseIn(0.5) = {} should be < 0.5", mid);
        assert!((evaluate_speed_ramp(SpeedRampCurve::EaseIn, 0.0)).abs() < 1e-6);
        assert!((evaluate_speed_ramp(SpeedRampCurve::EaseIn, 1.0) - 1.0).abs() < 1e-6);
    }

    // 20. Speed ramp: EaseOut starts fast
    #[test]
    fn test_speed_ramp_ease_out() {
        let mid = evaluate_speed_ramp(SpeedRampCurve::EaseOut, 0.5);
        assert!(mid > 0.5, "EaseOut(0.5) = {} should be > 0.5", mid);
        assert!((evaluate_speed_ramp(SpeedRampCurve::EaseOut, 0.0)).abs() < 1e-6);
        assert!((evaluate_speed_ramp(SpeedRampCurve::EaseOut, 1.0) - 1.0).abs() < 1e-6);
    }

    // 21. Speed ramp: EaseInOut is symmetric
    #[test]
    fn test_speed_ramp_ease_in_out() {
        let mid = evaluate_speed_ramp(SpeedRampCurve::EaseInOut, 0.5);
        assert!(
            (mid - 0.5).abs() < 1e-5,
            "EaseInOut(0.5) = {} should be 0.5",
            mid
        );
        let quarter = evaluate_speed_ramp(SpeedRampCurve::EaseInOut, 0.25);
        let three_quarter = evaluate_speed_ramp(SpeedRampCurve::EaseInOut, 0.75);
        assert!(
            (quarter + three_quarter - 1.0).abs() < 1e-4,
            "should be symmetric: {} + {} = {}",
            quarter,
            three_quarter,
            quarter + three_quarter
        );
    }

    // 22. Speed ramp: CubicBezier linear equivalent
    #[test]
    fn test_speed_ramp_cubic_bezier_linear() {
        // Bezier with (0.33, 0.33, 0.67, 0.67) should be roughly linear
        let curve = SpeedRampCurve::CubicBezier {
            x1: 0.33,
            y1: 0.33,
            x2: 0.67,
            y2: 0.67,
        };
        let mid = evaluate_speed_ramp(curve, 0.5);
        assert!(
            (mid - 0.5).abs() < 0.05,
            "near-linear bezier at 0.5 = {}",
            mid
        );
    }

    // 23. Speed ramp: CubicBezier boundary conditions
    #[test]
    fn test_speed_ramp_cubic_bezier_boundaries() {
        let curve = SpeedRampCurve::CubicBezier {
            x1: 0.25,
            y1: 0.1,
            x2: 0.75,
            y2: 0.9,
        };
        let at_zero = evaluate_speed_ramp(curve, 0.0);
        let at_one = evaluate_speed_ramp(curve, 1.0);
        assert!(at_zero.abs() < 0.05, "bezier(0) = {}", at_zero);
        assert!((at_one - 1.0).abs() < 0.05, "bezier(1) = {}", at_one);
    }

    // 24. SpeedRampInterpolator: constant speed
    #[test]
    fn test_speed_ramp_interpolator_constant() {
        let kfs = vec![
            SpeedKeyframe {
                time: 0.0,
                speed: 2.0,
                curve: SpeedRampCurve::Linear,
            },
            SpeedKeyframe {
                time: 10.0,
                speed: 2.0,
                curve: SpeedRampCurve::Linear,
            },
        ];
        let interp = SpeedRampInterpolator::new(kfs).expect("should create");
        assert!((interp.speed_at(5.0) - 2.0).abs() < 1e-5);
    }

    // 25. SpeedRampInterpolator: linear ramp
    #[test]
    fn test_speed_ramp_interpolator_linear_ramp() {
        let kfs = vec![
            SpeedKeyframe {
                time: 0.0,
                speed: 1.0,
                curve: SpeedRampCurve::Linear,
            },
            SpeedKeyframe {
                time: 10.0,
                speed: 3.0,
                curve: SpeedRampCurve::Linear,
            },
        ];
        let interp = SpeedRampInterpolator::new(kfs).expect("should create");
        let mid = interp.speed_at(5.0);
        assert!(
            (mid - 2.0).abs() < 0.1,
            "expected ~2.0 at midpoint, got {}",
            mid
        );
    }

    // 26. SpeedRampInterpolator: before/after keyframes
    #[test]
    fn test_speed_ramp_interpolator_extrapolation() {
        let kfs = vec![
            SpeedKeyframe {
                time: 1.0,
                speed: 0.5,
                curve: SpeedRampCurve::Linear,
            },
            SpeedKeyframe {
                time: 5.0,
                speed: 2.0,
                curve: SpeedRampCurve::Linear,
            },
        ];
        let interp = SpeedRampInterpolator::new(kfs).expect("should create");
        assert!((interp.speed_at(0.0) - 0.5).abs() < 1e-5, "before first KF");
        assert!((interp.speed_at(10.0) - 2.0).abs() < 1e-5, "after last KF");
    }

    // 27. SpeedRampInterpolator: empty returns None
    #[test]
    fn test_speed_ramp_interpolator_empty() {
        let result = SpeedRampInterpolator::new(vec![]);
        assert!(result.is_none());
    }

    // 28. SpeedRampInterpolator: accumulated source time with constant speed
    #[test]
    fn test_accumulated_source_time_constant() {
        let kfs = vec![
            SpeedKeyframe {
                time: 0.0,
                speed: 2.0,
                curve: SpeedRampCurve::Linear,
            },
            SpeedKeyframe {
                time: 100.0,
                speed: 2.0,
                curve: SpeedRampCurve::Linear,
            },
        ];
        let interp = SpeedRampInterpolator::new(kfs).expect("should create");
        // At constant speed 2x, after 5 seconds of output, 10 seconds of source have passed
        let src_time = interp.accumulated_source_time(5.0, 1000);
        assert!(
            (src_time - 10.0).abs() < 0.1,
            "expected ~10.0, got {}",
            src_time
        );
    }

    // 29. Optical flow interpolation output size
    #[test]
    fn test_optical_flow_interp_output_size() {
        let fa = make_rgba(16, 16, 80, 80, 80, 255);
        let fb = make_rgba(16, 16, 120, 120, 120, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::FlowBased));
        let out = proc.interpolate_optical_flow(&fa, &fb, 0.5, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 4);
    }

    // 30. Optical flow on identical frames preserves values
    #[test]
    fn test_optical_flow_identical_frames() {
        let fa = make_rgba(16, 16, 100, 100, 100, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::FlowBased));
        let out = proc.interpolate_optical_flow(&fa, &fa, 0.5, 16, 16);
        for chunk in out.chunks_exact(4) {
            let diff = (chunk[0] as i32 - 100).abs();
            assert!(diff <= 5, "optical flow identical diff too large: {}", diff);
        }
    }

    // 31. Adaptive blend output size
    #[test]
    fn test_adaptive_blend_output_size() {
        let fa = make_rgba(16, 16, 50, 50, 50, 255);
        let fb = make_rgba(16, 16, 100, 100, 100, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate_adaptive(&fa, &fb, 0.5, 16, 16);
        assert_eq!(out.len(), 16 * 16 * 4);
    }

    // 32. Adaptive blend on identical frames
    #[test]
    fn test_adaptive_blend_identical() {
        let fa = make_rgba(16, 16, 80, 80, 80, 255);
        let proc = SlowMotionProcessor::new(make_config(InterpolationMethod::Blend));
        let out = proc.interpolate_adaptive(&fa, &fa, 0.5, 16, 16);
        for chunk in out.chunks_exact(4) {
            let diff = (chunk[0] as i32 - 80).abs();
            assert!(diff <= 5, "adaptive identical diff: {}", diff);
        }
    }

    // 33. Speed ramp: EaseIn/EaseOut are complementary
    #[test]
    fn test_speed_ramp_ease_complementary() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let ein = evaluate_speed_ramp(SpeedRampCurve::EaseIn, t);
            let eout = evaluate_speed_ramp(SpeedRampCurve::EaseOut, 1.0 - t);
            assert!(
                (ein + eout - 1.0).abs() < 1e-4,
                "EaseIn({}) + EaseOut({}) = {} != 1.0",
                t,
                1.0 - t,
                ein + eout
            );
        }
    }

    // 34. Speed ramp values are monotonically increasing
    #[test]
    fn test_speed_ramp_monotonic() {
        for curve in &[
            SpeedRampCurve::Linear,
            SpeedRampCurve::EaseIn,
            SpeedRampCurve::EaseOut,
            SpeedRampCurve::EaseInOut,
        ] {
            let mut prev = 0.0f32;
            for i in 0..=100 {
                let t = i as f32 / 100.0;
                let val = evaluate_speed_ramp(*curve, t);
                assert!(
                    val >= prev - 1e-6,
                    "{:?} not monotonic at t={}: {} < {}",
                    curve,
                    t,
                    val,
                    prev
                );
                prev = val;
            }
        }
    }
}
