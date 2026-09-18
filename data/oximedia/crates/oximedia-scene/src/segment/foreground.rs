//! Foreground/background separation.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Segmentation mask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMask {
    /// Mask data (0 = background, 255 = foreground).
    pub data: Vec<u8>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
}

/// Foreground segmenter.
pub struct ForegroundSegmenter;

impl ForegroundSegmenter {
    /// Create a new foreground segmenter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Segment foreground from background.
    ///
    /// # Errors
    ///
    /// Returns error if segmentation fails.
    pub fn segment(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<SegmentMask> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Simple edge-based foreground detection
        let mut mask = vec![0u8; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                let mut edge_strength = 0.0;

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
                    mask[y * width + x] = 255;
                }
            }
        }

        Ok(SegmentMask {
            data: mask,
            width,
            height,
        })
    }
}

impl Default for ForegroundSegmenter {
    fn default() -> Self {
        Self::new()
    }
}
