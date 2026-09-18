//! Real-time single-pass streaming video stabilization with bounded look-ahead.
//!
//! # Algorithm
//!
//! This module implements a **causal** stabilization algorithm suitable for live
//! streaming and low-latency pipelines.  Classic multi-pass stabilizers must
//! buffer the entire clip before producing output; the streaming variant
//! operates with a bounded delay of at most `lookahead_frames` frames.
//!
//! ## Mathematical foundation
//!
//! Let `M[t]` be the inter-frame motion transform (3×3 homography or affine
//! matrix) estimated externally and supplied to [`StreamingStabilizer::process_frame`].
//!
//! **Cumulative path** (what the camera actually did):
//! ```text
//! P[0] = I  (identity)
//! P[t] = P[t-1] ⊗ M[t]   (matrix multiplication, right-to-left composition)
//! ```
//!
//! **Causal moving average** over the last `smooth_window` entries of `P`:
//! ```text
//! S[t] = (1 / min(t+1, K)) * Σ_{i=max(0,t-K+1)}^{t} P[i]
//! ```
//! where `K = smooth_window`.  Because we average element-wise over the
//! 3×3 matrices this is a linear approximation; it works well when the
//! per-frame motion is small (which is the common case for shaky camera
//! footage).
//!
//! **Stabilizing correction** for frame `t`:
//! ```text
//! C[t] = S[t] ⊗ P[t]^{-1}
//! ```
//!
//! The correction `C[t]` is then used to warp the raw pixel buffer for frame `t`.
//!
//! **Look-ahead** — when `lookahead_frames = L > 0` the pipeline delays output
//! by `L` frames so that `S[t]` can incorporate motions `M[t+1]…M[t+L]`.  The
//! `L` extra path entries are included in the moving average before the
//! correction for frame `t-L` is emitted.  This reduces stabilisation lag for
//! content with predictable motion (e.g. smooth camera pan) at the cost of a
//! bounded latency of `L/fps` seconds.

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────
//  Type alias
// ─────────────────────────────────────────────────────────────────

/// A 3×3 transform matrix in row-major order.
pub type Mat3 = [[f32; 3]; 3];

// ─────────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────────

/// Configuration for [`StreamingStabilizer`].
#[derive(Debug, Clone)]
pub struct StreamingStabConfig {
    /// Number of look-ahead frames buffered before output begins.
    ///
    /// `0` = purely causal (zero additional latency beyond the processing time
    /// of a single frame).  Typical values: 0–15.
    pub lookahead_frames: usize,

    /// Width of the causal moving-average window used to smooth the camera
    /// path.  Larger values produce smoother output but allow slower reaction
    /// to intentional camera moves.  Default: 30 (≈ 1 second at 30 fps).
    pub smooth_window: usize,

    /// Fraction of each output dimension that is guaranteed not to show black
    /// border artefacts.  Range 0.0–1.0.  A value of 0.95 means the output is
    /// cropped by 5 % on each side, hiding any border introduced by the warp.
    ///
    /// Set to `1.0` to disable cropping (borders may be visible on large
    /// motions).
    pub crop_ratio: f32,
}

impl Default for StreamingStabConfig {
    fn default() -> Self {
        Self {
            lookahead_frames: 0,
            smooth_window: 30,
            crop_ratio: 0.95,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Mat3 helpers (pure arithmetic, no external deps)
// ─────────────────────────────────────────────────────────────────

/// Identity 3×3 matrix.
#[inline]
const fn mat3_identity() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Element-wise addition of two 3×3 matrices.
#[inline]
fn mat3_add(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

/// Scalar multiplication of a 3×3 matrix.
#[inline]
fn mat3_scale(m: &Mat3, s: f32) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = m[r][c] * s;
        }
    }
    out
}

/// 3×3 matrix multiplication: `a ⊗ b`.
#[inline]
fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

/// Compute the inverse of a 3×3 matrix using the adjugate / cofactor method.
///
/// Returns `None` if the matrix is singular (determinant < ε).
fn mat3_inv(m: &Mat3) -> Option<Mat3> {
    // Cofactors
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;

    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    // Adjugate is transpose of cofactor matrix
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

// ─────────────────────────────────────────────────────────────────
//  Frame warping helpers
// ─────────────────────────────────────────────────────────────────

/// Apply a 3×3 homography `H` to warp `src` (RGB interleaved, row-major) into
/// a new buffer of the same dimensions, with bilinear interpolation and
/// constant (0) boundary padding.
///
/// The warp uses the *inverse mapping* approach: for each destination pixel
/// `(dx, dy)` it solves `H ⊗ src_h = dst_h` for the source homogeneous
/// coordinate, then samples `src` at that position.
fn warp_frame_rgb(src: &[u8], width: u32, height: u32, h: &Mat3) -> Vec<u8> {
    let w = width as usize;
    let h_img = height as usize;
    let mut dst = vec![0u8; w * h_img * 3];

    // Pre-compute inverse so we can back-project destination → source.
    let inv = match mat3_inv(h) {
        Some(m) => m,
        None => {
            // Singular: return zeros (shouldn't happen in practice)
            return dst;
        }
    };

    for dy in 0..h_img {
        for dx in 0..w {
            // Homogeneous destination coordinate
            let xh = dx as f32;
            let yh = dy as f32;

            // Back-project: [sx, sy, sw] = inv ⊗ [xh, yh, 1]
            let sw = inv[2][0] * xh + inv[2][1] * yh + inv[2][2];
            if sw.abs() < 1e-10 {
                continue;
            }
            let sx = (inv[0][0] * xh + inv[0][1] * yh + inv[0][2]) / sw;
            let sy = (inv[1][0] * xh + inv[1][1] * yh + inv[1][2]) / sw;

            // Bilinear sample — skip only pixels that fall outside [0, w) × [0, h).
            // The inner bilinear clamps x1/y1 to (w-1)/(h-1) already, so we
            // only need to reject truly out-of-bounds coordinates.
            if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h_img as f32 {
                continue; // outside source image → black border pixel
            }

            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h_img - 1);

            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let dst_base = (dy * w + dx) * 3;

            for ch in 0..3 {
                let p00 = f32::from(src[(y0 * w + x0) * 3 + ch]);
                let p10 = f32::from(src[(y0 * w + x1) * 3 + ch]);
                let p01 = f32::from(src[(y1 * w + x0) * 3 + ch]);
                let p11 = f32::from(src[(y1 * w + x1) * 3 + ch]);

                let val = (1.0 - fx) * (1.0 - fy) * p00
                    + fx * (1.0 - fy) * p10
                    + (1.0 - fx) * fy * p01
                    + fx * fy * p11;

                dst[dst_base + ch] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }

    dst
}

/// Build a crop-and-zoom homography that scales the output by `crop_ratio`
/// around the image centre, effectively hiding the black borders introduced
/// by a corrective warp.
///
/// `crop_ratio = 0.95` → output is scaled up by `1/0.95 ≈ 1.053` so that
/// the 5 % border margin is cut off on each side.
fn crop_homography(width: u32, height: u32, crop_ratio: f32) -> Mat3 {
    if crop_ratio >= 1.0 || crop_ratio <= 0.0 {
        return mat3_identity();
    }
    let cx = (width as f32) / 2.0;
    let cy = (height as f32) / 2.0;
    let s = 1.0 / crop_ratio; // scale > 1 — zoom in to hide borders

    // Translate centre to origin → scale → translate back
    // H = T(cx,cy) ⊗ S(s) ⊗ T(-cx,-cy)
    [
        [s, 0.0, cx * (1.0 - s)],
        [0.0, s, cy * (1.0 - s)],
        [0.0, 0.0, 1.0],
    ]
}

// ─────────────────────────────────────────────────────────────────
//  StreamingStabilizer
// ─────────────────────────────────────────────────────────────────

/// Real-time single-pass video stabilizer.
///
/// Feed frames one at a time via [`process_frame`][StreamingStabilizer::process_frame].
/// Call [`flush`][StreamingStabilizer::flush] at end-of-stream to drain any
/// remaining buffered frames.
///
/// # Example
///
/// ```rust
/// use oximedia_stabilize::streaming::{StreamingStabilizer, StreamingStabConfig};
///
/// let config = StreamingStabConfig { lookahead_frames: 5, ..Default::default() };
/// let mut stab = StreamingStabilizer::new(320, 240, config);
///
/// let identity = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let frame = vec![128u8; 320 * 240 * 3];
///
/// // First L-1 frames return None (still filling look-ahead buffer)
/// for _ in 0..5 {
///     assert!(stab.process_frame(frame.clone(), identity).is_none());
/// }
/// // L-th frame returns the first stabilised output
/// assert!(stab.process_frame(frame.clone(), identity).is_some());
///
/// // Flush remaining frames
/// let remaining = stab.flush();
/// assert_eq!(remaining.len(), 5); // the L frames still in the buffer
/// ```
#[derive(Debug)]
pub struct StreamingStabilizer {
    lookahead_frames: usize,
    smooth_window: usize,
    crop_ratio: f32,

    /// Ring buffer of *cumulative path* matrices `P[t]`.
    ///
    /// The front is the oldest still needed for the moving average;
    /// the back is the most recently pushed.
    path_history: VecDeque<Mat3>,

    /// Pending raw frames awaiting output.  Index 0 is the frame whose
    /// stabilised version will be emitted next time the buffer is full.
    pending_frames: VecDeque<Vec<u8>>,

    /// The current cumulative path `P[t]` (updated on each `process_frame`).
    cumulative_path: Mat3,

    /// Frame dimensions (RGB interleaved).
    width: u32,
    height: u32,
}

impl StreamingStabilizer {
    /// Create a new streaming stabilizer.
    ///
    /// # Arguments
    ///
    /// * `width`  — frame width in pixels
    /// * `height` — frame height in pixels
    /// * `config` — stabilisation parameters
    #[must_use]
    pub fn new(width: u32, height: u32, config: StreamingStabConfig) -> Self {
        Self {
            lookahead_frames: config.lookahead_frames,
            smooth_window: config.smooth_window.max(1),
            crop_ratio: config.crop_ratio.clamp(0.0, 1.0),
            path_history: VecDeque::new(),
            pending_frames: VecDeque::new(),
            cumulative_path: mat3_identity(),
            width,
            height,
        }
    }

    /// Feed one frame together with its inter-frame motion transform `motion`.
    ///
    /// `motion` is the 3×3 matrix that maps coordinates in frame `t-1` to
    /// frame `t` (i.e. the camera motion relative to the previous frame).
    /// For the very first frame supply the identity matrix.
    ///
    /// Returns `Some(stabilised_frame)` once the look-ahead buffer is full, or
    /// `None` while accumulating the first `lookahead_frames` frames.
    ///
    /// The returned frame is an RGB interleaved `Vec<u8>` of length
    /// `width * height * 3`.
    #[must_use]
    pub fn process_frame(&mut self, frame: Vec<u8>, motion: Mat3) -> Option<Vec<u8>> {
        // ── 1. Update cumulative path ────────────────────────────────────────
        self.cumulative_path = mat3_mul(&self.cumulative_path, &motion);

        // ── 2. Append to history and frame queue ─────────────────────────────
        self.path_history.push_back(self.cumulative_path);
        self.pending_frames.push_back(frame);

        // Keep path_history no longer than smooth_window + lookahead_frames
        // so we hold every path value that may still contribute to a moving
        // average for an un-emitted frame.
        let max_history = self.smooth_window + self.lookahead_frames;
        while self.path_history.len() > max_history {
            self.path_history.pop_front();
        }

        // ── 3. Decide whether to emit a frame ────────────────────────────────
        // We emit when we have strictly more than `lookahead_frames` frames
        // buffered (i.e., the buffer is full and the oldest frame can be
        // stabilised using the full look-ahead window).
        if self.pending_frames.len() <= self.lookahead_frames {
            return None;
        }

        self.emit_oldest_frame()
    }

    /// Flush all remaining buffered frames.
    ///
    /// Call this at the end of the stream.  The returned `Vec` may contain
    /// fewer elements than `lookahead_frames` if the stream was shorter than
    /// the look-ahead window.
    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(self.pending_frames.len());
        while !self.pending_frames.is_empty() {
            if let Some(f) = self.emit_oldest_frame() {
                out.push(f);
            }
        }
        out
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Compute and return the stabilised version of the oldest pending frame.
    fn emit_oldest_frame(&mut self) -> Option<Vec<u8>> {
        let raw_frame = self.pending_frames.pop_front()?;

        // Index of the frame we are about to emit within the *current*
        // path_history.  The path for the emitted frame is at index 0 of
        // path_history (it was the first one appended after pop_front pruning,
        // but since we cap history to smooth_window+lookahead we may need to
        // compute the smooth path from what is available).
        //
        // Strategy: the emitted frame corresponds to path_history index 0
        // (oldest entry).  We compute the smooth path S as the mean of all
        // entries in path_history (causal + look-ahead).
        let smooth_path = self.causal_smooth_path();

        // Retrieve the *actual* path for the frame being emitted.
        // It is the first (oldest) entry in path_history.
        let actual_path = match self.path_history.front() {
            Some(p) => *p,
            None => return Some(raw_frame), // no history → identity (no-op)
        };

        // Remove the oldest path entry since this frame is being emitted.
        self.path_history.pop_front();

        // Compute correction C = S ⊗ P^{-1}
        let correction = match mat3_inv(&actual_path) {
            Some(inv_p) => mat3_mul(&smooth_path, &inv_p),
            None => mat3_identity(),
        };

        // Compose with crop homography to hide borders
        let crop = crop_homography(self.width, self.height, self.crop_ratio);
        let full_warp = mat3_mul(&crop, &correction);

        Some(warp_frame_rgb(
            &raw_frame,
            self.width,
            self.height,
            &full_warp,
        ))
    }

    /// Compute the causal (+ look-ahead) moving average of the path.
    ///
    /// This is a simple element-wise mean over *all* entries currently in
    /// `path_history`.  The window is naturally bounded to `smooth_window`
    /// by the pruning in `process_frame`.
    fn causal_smooth_path(&self) -> Mat3 {
        let n = self.path_history.len();
        if n == 0 {
            return mat3_identity();
        }

        // Use only the last `smooth_window` entries from path_history for the
        // moving-average window so old motion doesn't pollute the estimate.
        let window = self.smooth_window.min(n);
        let start = n - window;

        let mut sum = [[0.0f32; 3]; 3];
        for entry in self.path_history.iter().skip(start) {
            sum = mat3_add(&sum, entry);
        }
        mat3_scale(&sum, 1.0 / window as f32)
    }
}

// ─────────────────────────────────────────────────────────────────
//  PSNR helper for tests
// ─────────────────────────────────────────────────────────────────

/// Compute PSNR (dB) between two equal-length u8 buffers.
/// Returns `f64::INFINITY` for identical inputs.
#[must_use]
pub fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse < 1e-10 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

// ─────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 64;
    const H: u32 = 64;

    fn identity() -> Mat3 {
        mat3_identity()
    }

    fn uniform_frame(v: u8) -> Vec<u8> {
        vec![v; (W * H * 3) as usize]
    }

    /// Feed L-1 frames → all None; L-th frame → Some.
    #[test]
    fn test_streaming_stab_accumulates_lookahead() {
        let config = StreamingStabConfig {
            lookahead_frames: 5,
            smooth_window: 10,
            crop_ratio: 0.9,
        };
        let mut stab = StreamingStabilizer::new(W, H, config);

        for i in 0..5 {
            let result = stab.process_frame(uniform_frame(100), identity());
            assert!(
                result.is_none(),
                "frame {i}: expected None while filling look-ahead"
            );
        }
        // 5th frame (index 5) should produce first output
        let result = stab.process_frame(uniform_frame(100), identity());
        assert!(
            result.is_some(),
            "6th frame should emit first stabilised output"
        );
    }

    /// Identity motion on a uniform frame → output ≈ input (PSNR > 40 dB).
    #[test]
    fn test_streaming_stab_static_scene() {
        let config = StreamingStabConfig {
            lookahead_frames: 0,
            smooth_window: 10,
            crop_ratio: 1.0, // no crop so identity warp preserves pixel values exactly
        };
        let mut stab = StreamingStabilizer::new(W, H, config);
        let frame = uniform_frame(180);

        let mut psnr_values = Vec::new();
        for _ in 0..60 {
            if let Some(out) = stab.process_frame(frame.clone(), identity()) {
                let psnr = compute_psnr(&frame, &out);
                psnr_values.push(psnr);
            }
        }
        // Flush remaining
        for out in stab.flush() {
            let psnr = compute_psnr(&frame, &out);
            psnr_values.push(psnr);
        }

        assert!(
            !psnr_values.is_empty(),
            "should have produced output frames"
        );
        for (i, &psnr) in psnr_values.iter().enumerate() {
            assert!(
                psnr > 40.0,
                "frame {i}: PSNR {psnr:.1} dB < 40 dB for static identity motion"
            );
        }
    }

    /// Feed 20 frames with 5-frame look-ahead; verify 20 total outputs.
    #[test]
    fn test_streaming_stab_flush() {
        let lookahead = 5;
        let config = StreamingStabConfig {
            lookahead_frames: lookahead,
            smooth_window: 10,
            crop_ratio: 0.9,
        };
        let mut stab = StreamingStabilizer::new(W, H, config);

        let mut inline_count = 0usize;
        for _ in 0..20 {
            if stab.process_frame(uniform_frame(64), identity()).is_some() {
                inline_count += 1;
            }
        }

        let flushed = stab.flush();
        let total = inline_count + flushed.len();
        assert_eq!(
            total, 20,
            "total output frames should equal total input frames (got {total})"
        );
    }

    #[test]
    fn test_mat3_mul_identity() {
        let a = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let result = mat3_mul(&a, &mat3_identity());
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (result[r][c] - a[r][c]).abs() < 1e-6,
                    "element [{r}][{c}] should be unchanged"
                );
            }
        }
    }

    #[test]
    fn test_mat3_inv_identity() {
        let inv = mat3_inv(&mat3_identity()).expect("identity is invertible");
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0f32 } else { 0.0 };
                assert!(
                    (inv[r][c] - expected).abs() < 1e-6,
                    "identity inverse element [{r}][{c}]"
                );
            }
        }
    }

    #[test]
    fn test_mat3_inv_singular() {
        let singular = [[0.0f32; 3]; 3];
        assert!(
            mat3_inv(&singular).is_none(),
            "singular matrix has no inverse"
        );
    }

    #[test]
    fn test_compute_psnr_identical() {
        let a = vec![128u8; 100];
        let b = vec![128u8; 100];
        assert!(
            compute_psnr(&a, &b).is_infinite(),
            "identical buffers → infinite PSNR"
        );
    }

    #[test]
    fn test_compute_psnr_different() {
        let a = vec![0u8; 100];
        let b = vec![255u8; 100];
        let psnr = compute_psnr(&a, &b);
        assert!(psnr < 1.0, "maximally different → very low PSNR");
    }

    #[test]
    fn test_crop_homography_no_crop() {
        let h = crop_homography(100, 100, 1.0);
        // crop_ratio=1.0 → identity
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0f32 } else { 0.0 };
                assert!((h[r][c] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_streaming_stab_zero_lookahead() {
        // With lookahead=0 every frame should emit immediately
        let config = StreamingStabConfig {
            lookahead_frames: 0,
            smooth_window: 5,
            crop_ratio: 0.9,
        };
        let mut stab = StreamingStabilizer::new(W, H, config);
        let frame = uniform_frame(200);
        for i in 0..10 {
            let result = stab.process_frame(frame.clone(), identity());
            assert!(
                result.is_some(),
                "frame {i}: lookahead=0 should emit immediately"
            );
        }
        let flushed = stab.flush();
        assert_eq!(flushed.len(), 0, "nothing pending after zero-lookahead run");
    }
}
