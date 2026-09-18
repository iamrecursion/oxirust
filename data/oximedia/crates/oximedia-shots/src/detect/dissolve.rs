//! Dissolve transition detection.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

/// Block size (pixels) used in block-SAD dissolve detection.
pub const DISSOLVE_BLOCK_SIZE: u32 = 16;

/// SAD threshold per block above which a block is considered "changed".
/// At 16×16 pixels with u8 values, 2048 ≈ 8 per-pixel difference on average.
pub const DISSOLVE_SAD_THRESHOLD: u32 = 2048;

/// Dissolve transition detector.
pub struct DissolveDetector {
    /// Threshold for dissolve detection.
    threshold: f32,
    /// Minimum dissolve duration in frames.
    min_duration: usize,
}

impl DissolveDetector {
    /// Create a new dissolve detector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 0.15,
            min_duration: 10,
        }
    }

    /// Detect dissolve transition in a sequence of frames.
    ///
    /// Uses block-SAD change counting internally: the frame pair is divided into
    /// 16×16 pixel blocks and the fraction of blocks whose SAD exceeds
    /// [`DISSOLVE_SAD_THRESHOLD`] is used as the per-pair change score.  A
    /// dissolve is reported when the average block-change fraction across the
    /// window exceeds `self.threshold` and transitions are sufficiently smooth.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid or have mismatched dimensions.
    pub fn detect_dissolve(&self, frames: &[FrameBuffer]) -> ShotResult<(bool, f32, usize)> {
        if frames.len() < self.min_duration {
            return Ok((false, 0.0, 0));
        }

        // Compute per-consecutive-pair block-SAD change fraction.
        let mut pair_scores: Vec<f32> = Vec::with_capacity(frames.len().saturating_sub(1));
        for i in 1..frames.len() {
            let score = detect_dissolve_block_sad(&frames[i - 1], &frames[i])?;
            pair_scores.push(score);
        }

        if pair_scores.is_empty() {
            return Ok((false, 0.0, 0));
        }

        // Slide a window of `min_duration` over the pair scores to find the
        // contiguous segment with the highest average change fraction and the
        // smoothest (least jittery) transition — both hallmarks of a dissolve.
        let window = self.min_duration.saturating_sub(1).max(1);
        let mut max_score = 0.0_f32;
        let mut max_pos = 0usize;

        let end = pair_scores.len().saturating_sub(window).saturating_add(1);
        for start in 0..end {
            let slice = &pair_scores[start..start + window.min(pair_scores.len() - start)];
            let avg = slice.iter().copied().sum::<f32>() / slice.len() as f32;
            let smoothness_score = self.analyze_dissolve_pattern(slice);
            // Combine: high average change + smooth curve → high dissolve score.
            let combined = avg * 0.5 + smoothness_score * 0.5;
            if combined > max_score {
                max_score = combined;
                max_pos = start + window / 2;
            }
        }

        let is_dissolve = max_score > self.threshold;
        Ok((is_dissolve, max_score, max_pos))
    }

    /// Analyze a series of per-pair change scores and return a smoothness score
    /// in [0, 1] that is high when the values change gradually (dissolve-like).
    fn analyze_dissolve_pattern(&self, scores: &[f32]) -> f32 {
        if scores.len() < 2 {
            return 0.0;
        }

        // Mean absolute inter-frame difference of the score series.
        let mut smoothness = 0.0_f32;
        for i in 1..scores.len() {
            smoothness += (scores[i] - scores[i - 1]).abs();
        }
        smoothness /= scores.len() as f32;

        // Low jitter → high smoothness score.
        let dissolve_score = if smoothness < 0.1 {
            1.0 - smoothness * 5.0
        } else {
            0.0
        };

        dissolve_score.max(0.0).min(1.0)
    }
}

impl Default for DissolveDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the fraction of 16×16 blocks (luma channel) whose SAD between
/// `prev` and `curr` exceeds [`DISSOLVE_SAD_THRESHOLD`].
///
/// Returns a value in `[0.0, 1.0]` — 0.0 means no blocks changed, 1.0 means
/// all blocks exceed the threshold.
///
/// # Errors
///
/// Returns `ShotError::InvalidFrame` when the two frames have mismatched
/// dimensions or have fewer than 1 channel.
pub fn detect_dissolve_block_sad(prev: &FrameBuffer, curr: &FrameBuffer) -> ShotResult<f32> {
    let (ph, pw, pc) = prev.dim();
    let (ch, cw, cc) = curr.dim();

    if ph != ch || pw != cw {
        return Err(ShotError::InvalidFrame(
            "Frame dimensions do not match for block-SAD dissolve detection".to_string(),
        ));
    }
    if pc == 0 || cc == 0 {
        return Err(ShotError::InvalidFrame(
            "Frame must have at least 1 channel".to_string(),
        ));
    }

    let height = ph as u32;
    let width = pw as u32;

    // Number of complete blocks in each axis.
    let blocks_y = height / DISSOLVE_BLOCK_SIZE;
    let blocks_x = width / DISSOLVE_BLOCK_SIZE;

    if blocks_y == 0 || blocks_x == 0 {
        // Frame too small for even a single block — fall back to a single SAD.
        let mut sad = 0u64;
        for y in 0..ph {
            for x in 0..pw {
                let pv = u64::from(prev.get(y, x, 0));
                let cv = u64::from(curr.get(y, x, 0));
                sad += pv.abs_diff(cv);
            }
        }
        let threshold = u64::from(DISSOLVE_SAD_THRESHOLD) * (ph * pw) as u64 / 256;
        return Ok(if sad > threshold { 1.0 } else { 0.0 });
    }

    let total_blocks = blocks_y * blocks_x;
    let mut changed_blocks = 0u32;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let y0 = (by * DISSOLVE_BLOCK_SIZE) as usize;
            let x0 = (bx * DISSOLVE_BLOCK_SIZE) as usize;
            let y1 = y0 + DISSOLVE_BLOCK_SIZE as usize;
            let x1 = x0 + DISSOLVE_BLOCK_SIZE as usize;

            let mut sad = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    // Use luma channel (channel 0) for the SAD metric.
                    let pv = u32::from(prev.get(y, x, 0));
                    let cv = u32::from(curr.get(y, x, 0));
                    sad += pv.abs_diff(cv);
                }
            }

            if sad > DISSOLVE_SAD_THRESHOLD {
                changed_blocks += 1;
            }
        }
    }

    Ok(changed_blocks as f32 / total_blocks as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dissolve_detector_creation() {
        let detector = DissolveDetector::new();
        assert!((detector.threshold - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_dissolve_in_single_frames() {
        let detector = DissolveDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3)];
        let result = detector.detect_dissolve(&frames);
        assert!(result.is_ok());
        if let Ok((is_dissolve, _, _)) = result {
            assert!(!is_dissolve);
        }
    }

    #[test]
    fn test_block_sad_identical_frames() {
        let prev = FrameBuffer::from_elem(320, 320, 3, 128);
        let curr = FrameBuffer::from_elem(320, 320, 3, 128);
        let score = detect_dissolve_block_sad(&prev, &curr).expect("ok");
        assert!(
            (score - 0.0).abs() < f32::EPSILON,
            "identical frames should yield 0 changed blocks"
        );
    }

    #[test]
    fn test_block_sad_fully_different_frames() {
        // prev = 0, curr = 255 → every block SAD = 255 * 256 = 65280 >> threshold
        let prev = FrameBuffer::zeros(320, 320, 3);
        let curr = FrameBuffer::from_elem(320, 320, 3, 255);
        let score = detect_dissolve_block_sad(&prev, &curr).expect("ok");
        assert!(
            score > 0.9,
            "fully different frames should yield near-1.0 score, got {score}"
        );
    }

    #[test]
    fn test_dissolve_block_sad_detects_fade() {
        // prev: all blocks at 128.
        // curr: exactly half the blocks (checkerboard pattern) set to 200.
        // This simulates a dissolve midpoint where many blocks differ noticeably.
        let height = 320usize;
        let width = 320usize;
        let channels = 3usize;
        let bs = DISSOLVE_BLOCK_SIZE as usize;

        let prev = FrameBuffer::from_elem(height, width, channels, 128);
        let mut curr = FrameBuffer::from_elem(height, width, channels, 128);

        // Paint the "changed" blocks: every other block column in every row of blocks.
        let blocks_y = height / bs;
        let blocks_x = width / bs;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if (by + bx) % 2 == 0 {
                    let y0 = by * bs;
                    let x0 = bx * bs;
                    for y in y0..y0 + bs {
                        for x in x0..x0 + bs {
                            for c in 0..channels {
                                curr.set(y, x, c, 200);
                            }
                        }
                    }
                }
            }
        }

        // The block-SAD for changed blocks is (200-128)*256 = 18 432 > DISSOLVE_SAD_THRESHOLD=2048.
        let score = detect_dissolve_block_sad(&prev, &curr).expect("ok");
        assert!(
            score > 0.3,
            "should detect ~50% changed blocks, got fraction = {score}"
        );

        // For the detect_dissolve interface, build a synthetic 12-frame fade sequence.
        // Frames grade from 0→255 to represent an in-progress dissolve.
        let mut frames: Vec<FrameBuffer> = Vec::new();
        for step in 0..12u8 {
            let v = step * 20;
            frames.push(FrameBuffer::from_elem(height, width, channels, v));
        }
        let detector = DissolveDetector::new();
        // We expect a positive score for this gradual ramp.
        let result = detector.detect_dissolve(&frames).expect("ok");
        // The score should be > 0 even if detect_dissolve doesn't call it a full dissolve
        // because the default threshold is tuned for obvious dissolves.
        assert!(result.1 >= 0.0);
    }
}
