//! Image segmentation — foreground/background and semantic region detection.
//!
//! The [`Segmenter`] struct provides a unified entry point that dispatches work
//! in parallel via rayon when the image is large enough for parallelism to pay
//! off.

pub mod foreground;
pub mod semantic;

pub use foreground::{ForegroundSegmenter, SegmentMask};
pub use semantic::{SemanticRegion, SemanticSegmenter};

use crate::error::{SceneError, SceneResult};
use rayon::prelude::*;

/// Minimum image size (pixels) before parallelism is applied.
const PARALLEL_THRESHOLD: usize = 128 * 128;

/// Combined segmenter that runs foreground and semantic segmentation,
/// using rayon parallel processing for large images.
pub struct Segmenter {
    foreground: ForegroundSegmenter,
    semantic: SemanticSegmenter,
}

/// Combined segmentation result.
pub struct SegmentResult {
    /// Foreground/background mask.
    pub foreground_mask: SegmentMask,
    /// Semantic regions.
    pub semantic_regions: Vec<SemanticRegion>,
}

impl Segmenter {
    /// Create a new segmenter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            foreground: ForegroundSegmenter::new(),
            semantic: SemanticSegmenter::new(),
        }
    }

    /// Segment an image. For large images (> `PARALLEL_THRESHOLD` pixels)
    /// the per-pixel edge computation runs in parallel via rayon.
    ///
    /// # Errors
    ///
    /// Returns error if the input dimensions are inconsistent.
    pub fn segment(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<SegmentResult> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let pixel_count = width * height;

        let foreground_mask = if pixel_count >= PARALLEL_THRESHOLD {
            self.segment_foreground_parallel(rgb_data, width, height)?
        } else {
            self.foreground.segment(rgb_data, width, height)?
        };

        let semantic_regions = self.semantic.segment(rgb_data, width, height)?;

        Ok(SegmentResult {
            foreground_mask,
            semantic_regions,
        })
    }

    /// Parallel foreground segmentation using rayon.
    fn segment_foreground_parallel(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<SegmentMask> {
        // Compute mask rows in parallel. Each row is independent once we
        // have read-only access to the full image.
        let mask: Vec<u8> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                let mut row = vec![0u8; width];
                if y == 0 || y == height - 1 {
                    return row;
                }
                for x in 1..width - 1 {
                    let idx = (y * width + x) * 3;
                    let mut edge_strength = 0.0_f32;
                    for c in 0..3 {
                        let center = rgb_data[idx + c] as f32;
                        let left = rgb_data[idx - 3 + c] as f32;
                        let right = rgb_data[idx + 3 + c] as f32;
                        let top = rgb_data[idx - width * 3 + c] as f32;
                        let bottom = rgb_data[idx + width * 3 + c] as f32;
                        edge_strength += ((center - left).abs()
                            + (center - right).abs()
                            + (center - top).abs()
                            + (center - bottom).abs())
                            / 4.0;
                    }
                    if edge_strength > 30.0 {
                        row[x] = 255;
                    }
                }
                row
            })
            .collect();

        Ok(SegmentMask {
            data: mask,
            width,
            height,
        })
    }
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmenter_small_image() {
        let segmenter = Segmenter::new();
        let w = 50;
        let h = 50;
        let rgb_data = vec![128u8; w * h * 3];
        let result = segmenter.segment(&rgb_data, w, h);
        assert!(result.is_ok());
        let seg = result.expect("ok");
        assert_eq!(seg.foreground_mask.data.len(), w * h);
    }

    #[test]
    fn test_segmenter_large_image_parallel() {
        // Image above PARALLEL_THRESHOLD triggers rayon path
        let segmenter = Segmenter::new();
        let w = 256;
        let h = 256; // 65536 > PARALLEL_THRESHOLD (16384)
        let rgb_data = vec![100u8; w * h * 3];
        let result = segmenter.segment(&rgb_data, w, h);
        assert!(result.is_ok());
        let seg = result.expect("ok");
        assert_eq!(seg.foreground_mask.data.len(), w * h);
    }

    #[test]
    fn test_segmenter_parallel_same_as_sequential() {
        // For an image just below the threshold, both paths should produce
        // the same result when the threshold is lowered conceptually.
        // We compare directly by running both manually.
        let fg = ForegroundSegmenter::new();
        let segmenter = Segmenter::new();
        let w = 200;
        let h = 200;
        // Build a non-uniform image so there are actual edges
        let mut rgb_data = vec![50u8; w * h * 3];
        for y in 50..150 {
            for x in 50..150 {
                let idx = (y * w + x) * 3;
                rgb_data[idx] = 200;
                rgb_data[idx + 1] = 100;
                rgb_data[idx + 2] = 50;
            }
        }
        let seq = fg.segment(&rgb_data, w, h).expect("ok");
        // Force the parallel path by calling directly
        let par = segmenter
            .segment_foreground_parallel(&rgb_data, w, h)
            .expect("ok");
        assert_eq!(
            seq.data, par.data,
            "parallel and sequential results must match"
        );
    }

    #[test]
    fn test_segmenter_invalid_dimensions() {
        let segmenter = Segmenter::new();
        let result = segmenter.segment(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    /// Canonical test name from the task specification.
    /// Verifies that the rayon parallel path and the sequential edge-detection
    /// path produce bit-identical masks for a non-uniform image.
    #[test]
    fn test_segmenter_parallel_matches_sequential() {
        let fg = ForegroundSegmenter::new();
        let segmenter = Segmenter::new();
        let w = 256;
        let h = 256;
        let mut rgb_data = vec![30u8; w * h * 3];
        // A bright rectangle to create edges
        for y in 64..192 {
            for x in 64..192 {
                let idx = (y * w + x) * 3;
                rgb_data[idx] = 220;
                rgb_data[idx + 1] = 180;
                rgb_data[idx + 2] = 90;
            }
        }
        let seq = fg.segment(&rgb_data, w, h).expect("sequential ok");
        let par = segmenter
            .segment_foreground_parallel(&rgb_data, w, h)
            .expect("parallel ok");
        assert_eq!(
            seq.data, par.data,
            "parallel and sequential segmentation must produce identical masks"
        );
    }
}
