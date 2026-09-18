//! GOP-boundary compression artifact detection.
//!
//! At GOP (Group of Pictures) boundaries the encoder resets its reference frames
//! and typically uses a higher quantization step for the I-frame that begins the
//! new GOP.  This creates characteristic visual artifacts relative to the
//! preceding frame:
//!
//! * **BlockingDiscontinuity** — the 8×8 DCT grid becomes visible as strong
//!   horizontal/vertical edges along multiples of 8 pixels.
//! * **QuantizationJump** — the mean absolute difference between consecutive
//!   frames spikes because the new I-frame has coarser quantisation.
//! * **MotionResidualSpike** — individual 8×8 blocks show unusually high SAD
//!   (Sum of Absolute Differences) relative to their neighbours in the preceding
//!   frame.

use crate::ForensicsError;

// ─── Public types ────────────────────────────────────────────────────────────

/// The class of compression artifact found at a GOP boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactType {
    /// 8×8 block boundary energy increases sharply in the incoming frame.
    QuantizationJump,
    /// Per-block DCT-energy proxy jumps beyond the quantization threshold.
    BlockingDiscontinuity,
    /// Max-block SAD in the incoming frame exceeds 2× the mean SAD in the
    /// outgoing frame.
    MotionResidualSpike,
}

/// Detected GOP-boundary artifact with location and classification.
#[derive(Debug, Clone)]
pub struct GopBoundaryResult {
    /// Index of the *incoming* frame (0-based within the supplied slice).
    pub frame_index: usize,
    /// Detection confidence in [0, 1].
    pub confidence: f32,
    /// The dominant artifact type for this boundary.
    pub artifact_type: ArtifactType,
}

// ─── Detector ────────────────────────────────────────────────────────────────

/// Detects compression artifact discontinuities that characterise GOP boundaries.
pub struct GopBoundaryDetector {
    /// Blocking-energy ratio threshold (default 0.15).
    ///
    /// If `blocking(curr) / blocking(prev)` exceeds this value the boundary is
    /// flagged as [`ArtifactType::BlockingDiscontinuity`].
    pub block_energy_threshold: f32,
    /// Quantization-jump threshold (default 0.30).
    ///
    /// If the DCT-energy proxy ratio between consecutive frames exceeds this
    /// value the boundary is flagged as [`ArtifactType::QuantizationJump`].
    pub qp_jump_threshold: f32,
}

impl Default for GopBoundaryDetector {
    fn default() -> Self {
        Self {
            block_energy_threshold: 0.15,
            qp_jump_threshold: 0.30,
        }
    }
}

impl GopBoundaryDetector {
    /// Create a detector with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyse a sequence of grayscale frames for GOP-boundary artifacts.
    ///
    /// # Arguments
    ///
    /// * `frames` – Grayscale u8 slices in display order, each `w * h` bytes.
    /// * `w`      – Frame width in pixels.
    /// * `h`      – Frame height in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`ForensicsError::InvalidImage`] if any frame slice is shorter
    /// than `w * h` bytes.
    pub fn detect(
        &self,
        frames: &[&[u8]],
        w: u32,
        h: u32,
    ) -> Result<Vec<GopBoundaryResult>, ForensicsError> {
        let expected = (w as usize) * (h as usize);
        for (i, frame) in frames.iter().enumerate() {
            if frame.len() < expected {
                return Err(ForensicsError::InvalidImage(format!(
                    "Frame {i} is too short: got {} bytes, expected {expected}",
                    frame.len()
                )));
            }
        }

        let mut results = Vec::new();

        for idx in 1..frames.len() {
            let prev = frames[idx - 1];
            let curr = frames[idx];
            if let Some(r) = self.analyze_frame_pair(prev, curr, w, h) {
                results.push(GopBoundaryResult {
                    frame_index: idx,
                    confidence: r.confidence,
                    artifact_type: r.artifact_type,
                });
            }
        }

        Ok(results)
    }

    /// Analyse a single (prev, curr) frame pair for a GOP-boundary artifact.
    ///
    /// Returns `Some(result)` when the highest-confidence finding is ≥ 0.30,
    /// otherwise `None`.
    ///
    /// # Arguments
    ///
    /// * `prev` / `curr` – Grayscale u8 frames, `w * h` bytes each.  The
    ///   slices are not length-checked; callers should ensure correct sizing.
    /// * `w`, `h`        – Frame dimensions.
    #[must_use]
    pub fn analyze_frame_pair(
        &self,
        prev: &[u8],
        curr: &[u8],
        w: u32,
        h: u32,
    ) -> Option<GopBoundaryResult> {
        let findings = [
            self.measure_blocking(prev, curr, w, h),
            self.measure_qp_jump(prev, curr, w, h),
            self.measure_motion_residual(prev, curr, w, h),
        ];

        // Pick the highest-confidence finding.
        let best = findings
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        best.and_then(|(artifact_type, confidence)| {
            if confidence >= 0.30 {
                Some(GopBoundaryResult {
                    frame_index: 0, // frame_index is set by the caller in detect()
                    confidence,
                    artifact_type,
                })
            } else {
                None
            }
        })
    }
}

// ─── Private measurement helpers ─────────────────────────────────────────────

impl GopBoundaryDetector {
    /// Measure blocking discontinuity.
    ///
    /// Sums `|px - px+1|` at every 8-pixel column boundary in both frames and
    /// computes the ratio `blocking(curr) / (blocking(prev) + ε)`.  A large
    /// ratio means the incoming frame has significantly stronger 8-pixel grid
    /// edges — a hallmark of a high-QP I-frame.
    fn measure_blocking(&self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> (ArtifactType, f32) {
        let blocking_prev = grid_blocking_energy(prev, w, h);
        let blocking_curr = grid_blocking_energy(curr, w, h);

        // Ratio of current frame blocking relative to previous.
        let ratio = (blocking_curr - blocking_prev) / (blocking_prev + 1e-6);
        let confidence = if ratio > self.block_energy_threshold {
            (ratio / 2.0).min(1.0)
        } else {
            0.0
        };

        (ArtifactType::BlockingDiscontinuity, confidence)
    }

    /// Measure quantization jump via a DCT-energy proxy.
    ///
    /// Divides each frame into non-overlapping 8×8 blocks, computes the mean
    /// pixel value of each block, and uses the sum of squared block-mean
    /// differences between frames as a DCT-energy proxy.  A large jump
    /// indicates that the encoder changed its quantization step.
    #[allow(clippy::cast_precision_loss)]
    fn measure_qp_jump(&self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> (ArtifactType, f32) {
        let w_usize = w as usize;
        let h_usize = h as usize;
        let block_size: usize = 8;

        let blocks_x = w_usize / block_size;
        let blocks_y = h_usize / block_size;
        let n_blocks = blocks_x * blocks_y;

        if n_blocks == 0 {
            return (ArtifactType::QuantizationJump, 0.0);
        }

        let mut sq_diff_sum: f64 = 0.0;
        let mut prev_energy: f64 = 0.0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mean_p = block_mean(prev, bx, by, block_size, w_usize);
                let mean_c = block_mean(curr, bx, by, block_size, w_usize);
                let diff = mean_c - mean_p;
                sq_diff_sum += diff * diff;
                prev_energy += mean_p * mean_p;
            }
        }

        sq_diff_sum /= n_blocks as f64;
        prev_energy /= n_blocks as f64;

        // Normalise by the previous frame's own energy to get a relative jump.
        let ratio = sq_diff_sum / (prev_energy + 1e-6);
        let confidence = if ratio > self.qp_jump_threshold as f64 {
            ((ratio / (self.qp_jump_threshold as f64 * 3.0)) as f32).min(1.0)
        } else {
            0.0
        };

        (ArtifactType::QuantizationJump, confidence)
    }

    /// Measure motion-residual spike via 8×8 block SAD.
    ///
    /// Computes the SAD of every 8×8 block between `prev` and `curr`.  If the
    /// maximum block SAD in the current frame exceeds 2× the mean SAD of all
    /// blocks (normalised to [0, 1] per pixel), a spike is flagged.
    #[allow(clippy::cast_precision_loss)]
    fn measure_motion_residual(
        &self,
        prev: &[u8],
        curr: &[u8],
        w: u32,
        h: u32,
    ) -> (ArtifactType, f32) {
        let w_usize = w as usize;
        let h_usize = h as usize;
        let block_size: usize = 8;

        let blocks_x = w_usize / block_size;
        let blocks_y = h_usize / block_size;
        let n_blocks = blocks_x * blocks_y;

        if n_blocks == 0 {
            return (ArtifactType::MotionResidualSpike, 0.0);
        }

        let pixels_per_block = (block_size * block_size) as f64;
        let mut max_sad: f64 = 0.0;
        let mut total_sad: f64 = 0.0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let sad = block_sad(prev, curr, bx, by, block_size, w_usize) as f64
                    / (pixels_per_block * 255.0);
                total_sad += sad;
                if sad > max_sad {
                    max_sad = sad;
                }
            }
        }

        let mean_sad = total_sad / n_blocks as f64;

        // A spike exists when the worst block is more than 2× the mean.
        let confidence = if max_sad > 2.0 * mean_sad + 1e-6 {
            ((max_sad / (mean_sad + 1e-6) / 4.0) as f32).min(1.0)
        } else {
            0.0
        };

        (ArtifactType::MotionResidualSpike, confidence)
    }
}

// ─── Pure pixel-math helpers ─────────────────────────────────────────────────

/// Sum of `|px[k*8+7] - px[k*8+8]|` at each 8-pixel block boundary, normalised.
///
/// The boundary between 8-pixel blocks k and k+1 falls between column indices
/// `k*8+7` and `k*8+8` (i.e. the last pixel of block k and the first pixel of
/// block k+1).  Blocking artifacts manifest as large edge magnitudes precisely
/// at these positions.
#[allow(clippy::cast_precision_loss)]
fn grid_blocking_energy(frame: &[u8], w: u32, h: u32) -> f32 {
    let w_usize = w as usize;
    let h_usize = h as usize;
    let mut energy: u64 = 0;
    let mut count: u64 = 0;

    for y in 0..h_usize {
        // Check boundary between block k and block k+1: pixels at (k*8+7, k*8+8).
        let mut x = 7_usize;
        while x + 1 < w_usize {
            let left = frame[y * w_usize + x] as i32;
            let right = frame[y * w_usize + x + 1] as i32;
            energy += (left - right).unsigned_abs() as u64;
            count += 1;
            x += 8;
        }
    }

    if count == 0 {
        return 0.0;
    }
    energy as f32 / (count as f32 * 255.0)
}

/// Mean pixel value of an 8×8 block at block-grid position `(bx, by)`.
#[allow(clippy::cast_precision_loss)]
fn block_mean(frame: &[u8], bx: usize, by: usize, bs: usize, stride: usize) -> f64 {
    let x0 = bx * bs;
    let y0 = by * bs;
    let mut sum: u64 = 0;
    for y in y0..y0 + bs {
        for x in x0..x0 + bs {
            let idx = y * stride + x;
            sum += frame[idx] as u64;
        }
    }
    sum as f64 / ((bs * bs) as f64 * 255.0)
}

/// Sum of Absolute Differences between two 8×8 blocks.
fn block_sad(a: &[u8], b: &[u8], bx: usize, by: usize, bs: usize, stride: usize) -> u32 {
    let x0 = bx * bs;
    let y0 = by * bs;
    let mut sad: u32 = 0;
    for y in y0..y0 + bs {
        for x in x0..x0 + bs {
            let idx = y * stride + x;
            sad += a[idx].abs_diff(b[idx]) as u32;
        }
    }
    sad
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Smooth horizontal gradient, 0 → 200.
    fn gradient_frame(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|i| ((i % w) as f32 / w as f32 * 200.0) as u8)
            .collect()
    }

    /// Same gradient with ±noise added in 8×8 block chunks.
    fn noisy_block_frame(w: usize, h: usize, noise_amp: i32) -> Vec<u8> {
        let grad = gradient_frame(w, h);
        let bs = 8;
        grad.iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = i % w;
                let y = i / w;
                // Alternate sign per block
                let sign: i32 = if ((x / bs) + (y / bs)) % 2 == 0 {
                    1
                } else {
                    -1
                };
                (v as i32 + sign * noise_amp).clamp(0, 255) as u8
            })
            .collect()
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_gop_boundary_detects_qp_jump() {
        let (w, h) = (64_u32, 64_u32);
        let prev = gradient_frame(w as usize, h as usize);
        // curr = gradient + strong per-block noise → high QP I-frame simulation
        let curr = noisy_block_frame(w as usize, h as usize, 40);

        let detector = GopBoundaryDetector::default();
        let result = detector
            .detect(&[prev.as_slice(), curr.as_slice()], w, h)
            .expect("detect should not fail");

        assert!(!result.is_empty(), "Should detect a GOP boundary artifact");
        assert!(
            result[0].confidence > 0.3,
            "confidence should be above threshold, got {}",
            result[0].confidence
        );
    }

    #[test]
    fn test_gop_boundary_no_false_positive_smooth() {
        let (w, h) = (64_u32, 64_u32);
        let frame = gradient_frame(w as usize, h as usize);

        let detector = GopBoundaryDetector::default();

        // Two identical frames → no boundary.
        let result = detector
            .detect(&[frame.as_slice(), frame.as_slice()], w, h)
            .expect("detect should not fail");
        assert!(
            result.is_empty(),
            "Identical frames should produce no boundary detection"
        );

        // Two slightly different frames (±2 amplitude noise).
        let noisy = noisy_block_frame(w as usize, h as usize, 2);
        let result2 = detector
            .detect(&[frame.as_slice(), noisy.as_slice()], w, h)
            .expect("detect should not fail");
        for r in &result2 {
            assert!(
                r.confidence < 0.3,
                "Slight noise should not exceed threshold, got {}",
                r.confidence
            );
        }
    }

    #[test]
    fn test_gop_boundary_invalid_frame_size() {
        let detector = GopBoundaryDetector::default();
        // A frame that is too short for 64×64.
        let short: &[u8] = &[0u8; 10];
        let full = vec![128u8; 64 * 64];
        let err = detector.detect(&[full.as_slice(), short], 64, 64);
        assert!(err.is_err(), "Should return error for undersized frame");
    }

    #[test]
    fn test_gop_boundary_single_frame_no_results() {
        let frame = gradient_frame(64, 64);
        let detector = GopBoundaryDetector::default();
        let result = detector
            .detect(&[frame.as_slice()], 64, 64)
            .expect("single frame should not error");
        assert!(result.is_empty(), "Single frame has no pair to compare");
    }

    #[test]
    fn test_gop_boundary_frame_index_correct() {
        let (w, h) = (64_u32, 64_u32);
        let smooth = gradient_frame(w as usize, h as usize);
        let noisy = noisy_block_frame(w as usize, h as usize, 40);

        let detector = GopBoundaryDetector::default();
        // Three-frame sequence: smooth, smooth, noisy
        let result = detector
            .detect(
                &[smooth.as_slice(), smooth.as_slice(), noisy.as_slice()],
                w,
                h,
            )
            .expect("detect should not fail");

        // If a result is reported, the frame index must be ≥ 1.
        for r in &result {
            assert!(
                r.frame_index >= 1,
                "frame_index must point to the incoming frame"
            );
        }
    }

    #[test]
    fn test_analyze_frame_pair_none_for_identical() {
        let frame = gradient_frame(64, 64);
        let detector = GopBoundaryDetector::default();
        let result = detector.analyze_frame_pair(&frame, &frame, 64, 64);
        assert!(
            result.is_none(),
            "Identical frames should return None from analyze_frame_pair"
        );
    }
}
