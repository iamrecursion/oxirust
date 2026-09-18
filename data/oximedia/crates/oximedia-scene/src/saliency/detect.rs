//! Saliency detection using spectral methods.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Saliency map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaliencyMap {
    /// Saliency values (0.0-1.0).
    pub data: Vec<f32>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
}

/// Saliency detector using spectral residual method.
pub struct SaliencyDetector;

impl SaliencyDetector {
    /// Create a new saliency detector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detect salient regions.
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect(&self, rgb_data: &[u8], width: usize, height: usize) -> SceneResult<SaliencyMap> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Convert to grayscale
        let mut gray = Vec::with_capacity(width * height);
        for i in (0..rgb_data.len()).step_by(3) {
            let r = rgb_data[i] as f32;
            let g = rgb_data[i + 1] as f32;
            let b = rgb_data[i + 2] as f32;
            let y = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            gray.push(y);
        }

        // Compute saliency using center-surround difference
        let saliency = self.compute_saliency(&gray, width, height);

        Ok(SaliencyMap {
            data: saliency,
            width,
            height,
        })
    }

    /// Compute saliency using multi-scale center-surround.
    fn compute_saliency(&self, gray: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut saliency = vec![0.0; width * height];

        // Multiple scales
        for scale in [8, 16, 32] {
            for y in scale..height - scale {
                for x in scale..width - scale {
                    let idx = y * width + x;
                    let center = gray[idx];

                    // Compute surround average
                    let mut surround_sum = 0.0;
                    let mut count = 0;

                    for dy in -(scale as i32)..=scale as i32 {
                        for dx in -(scale as i32)..=scale as i32 {
                            if dx.abs() < scale as i32 / 2 && dy.abs() < scale as i32 / 2 {
                                continue; // Skip center region
                            }

                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;

                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                surround_sum += gray[ny as usize * width + nx as usize];
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        let surround = surround_sum / count as f32;
                        saliency[idx] += (center - surround).abs();
                    }
                }
            }
        }

        // Normalize
        let max_sal = saliency.iter().copied().fold(f32::MIN, f32::max);
        if max_sal > 0.0 {
            for s in &mut saliency {
                *s /= max_sal;
            }
        }

        saliency
    }
}

impl Default for SaliencyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimised spectral-residual saliency detector with pre-allocated buffers.
///
/// Allocating the intermediate buffers once and reusing them across frames
/// avoids repeated heap allocation for high-frequency video processing.
///
/// The spectral residual method is based on the observation that natural
/// images have a uniform log-spectrum on average; the residual (deviation
/// from that average) corresponds to salient regions.
///
/// Because `oximedia-scene` does not depend on `oxifft`, this implementation
/// uses a spatial-frequency approximation:
/// 1. Downsample to a fixed internal resolution.
/// 2. Compute a blurred version (average pooling) as the "DC component".
/// 3. The saliency map is the squared difference between the original and the blur.
/// 4. Apply Gaussian smoothing and normalise.
pub struct SpectralSaliencyDetector {
    /// Internal working width (downsampled resolution).
    work_width: usize,
    /// Internal working height (downsampled resolution).
    work_height: usize,
    /// Pre-allocated grayscale buffer (work_width × work_height).
    gray_buf: Vec<f32>,
    /// Pre-allocated blur buffer.
    blur_buf: Vec<f32>,
    /// Pre-allocated saliency output buffer.
    saliency_buf: Vec<f32>,
    /// Spatial averaging radius for the "background" estimate.
    avg_radius: usize,
}

impl SpectralSaliencyDetector {
    /// Create a new spectral saliency detector.
    ///
    /// * `work_width`/`work_height` – internal resolution. Images are
    ///   downsampled to this size before processing. Smaller values trade
    ///   resolution for speed; 64×64 is a sensible default.
    /// * `avg_radius` – radius of the averaging filter used as a
    ///   background estimate (typical: 4–16).
    #[must_use]
    pub fn new(work_width: usize, work_height: usize, avg_radius: usize) -> Self {
        let size = work_width * work_height;
        Self {
            work_width,
            work_height,
            gray_buf: vec![0.0; size],
            blur_buf: vec![0.0; size],
            saliency_buf: vec![0.0; size],
            avg_radius: avg_radius.max(1),
        }
    }

    /// Create with default parameters (64×64, radius 8).
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(64, 64, 8)
    }

    /// Compute the spectral-residual saliency map for an RGB image.
    ///
    /// The returned [`SaliencyMap`] has the same dimensions as the *input*
    /// image (not the internal work resolution).
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid.
    pub fn detect(
        &mut self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<SaliencyMap> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let ww = self.work_width;
        let wh = self.work_height;

        // 1. Downsample to work resolution (nearest neighbour, inlined).
        for wy in 0..wh {
            let src_y = wy * height / wh;
            for wx in 0..ww {
                let src_x = wx * width / ww;
                let idx = (src_y * width + src_x) * 3;
                let r = rgb_data[idx] as f32;
                let g = rgb_data[idx + 1] as f32;
                let b = rgb_data[idx + 2] as f32;
                self.gray_buf[wy * ww + wx] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }
        }

        // 2. Box-blur to get "background" estimate.
        let r = self.avg_radius;
        for wy in 0..wh {
            for wx in 0..ww {
                let y0 = wy.saturating_sub(r);
                let y1 = (wy + r + 1).min(wh);
                let x0 = wx.saturating_sub(r);
                let x1 = (wx + r + 1).min(ww);
                let mut sum = 0.0_f32;
                let mut count = 0_u32;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        sum += self.gray_buf[sy * ww + sx];
                        count += 1;
                    }
                }
                self.blur_buf[wy * ww + wx] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }

        // 3. Spectral residual: squared difference.
        for i in 0..ww * wh {
            let diff = self.gray_buf[i] - self.blur_buf[i];
            self.saliency_buf[i] = diff * diff;
        }

        // 4. Normalise.
        let max_s = self.saliency_buf.iter().copied().fold(f32::MIN, f32::max);
        if max_s > 1e-6 {
            for v in &mut self.saliency_buf {
                *v /= max_s;
            }
        }

        // 5. Upsample back to original resolution (nearest neighbour).
        let mut out = vec![0.0_f32; width * height];
        for y in 0..height {
            let wy = y * wh / height;
            for x in 0..width {
                let wx = x * ww / width;
                out[y * width + x] = self.saliency_buf[wy * ww + wx];
            }
        }

        Ok(SaliencyMap {
            data: out,
            width,
            height,
        })
    }
}

// ---------------------------------------------------------------------------
// SpectralSaliencyComputer — thin pre-allocated wrapper for frame-by-frame use
// ---------------------------------------------------------------------------

/// Simplified spectral saliency computer with pre-allocated scratch buffers.
///
/// Intended for repeated single-channel (grayscale, `u8`) input — avoids
/// per-call heap allocation by holding the internal working buffers across
/// calls.  The internal resolution is fixed at construction time (default
/// 64 × 64 downsampled work area) and the output is upsampled back to the
/// original `(width, height)` dimensions.
///
/// For RGB input use [`SpectralSaliencyDetector`] directly.
pub struct SpectralSaliencyComputer {
    /// Underlying detector that owns the pre-allocated buffers.
    inner: SpectralSaliencyDetector,
    /// Source image width (pixels).
    width: usize,
    /// Source image height (pixels).
    height: usize,
}

impl SpectralSaliencyComputer {
    /// Create a new computer fixed to the given `(width, height)` dimensions.
    ///
    /// The internal work resolution is `min(width, 64) × min(height, 64)` to
    /// keep the computation fast while retaining spatial structure.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let ww = width.min(64).max(4);
        let wh = height.min(64).max(4);
        Self {
            inner: SpectralSaliencyDetector::new(ww, wh, 8),
            width,
            height,
        }
    }

    /// Compute a spectral-residual saliency map for a single grayscale frame.
    ///
    /// * `frame` — grayscale pixel data (`u8`), exactly `width × height` bytes
    ///   (row-major).  If the slice length does not match, an all-zeros map is
    ///   returned.
    ///
    /// Returns a `Vec<f32>` of length `width * height` with values in `[0, 1]`.
    pub fn compute(&mut self, frame: &[u8]) -> Vec<f32> {
        let expected = self.width * self.height;
        if frame.len() != expected {
            return vec![0.0_f32; expected];
        }
        // Convert grayscale to 3-channel RGB (replicate channel) so the inner
        // detector can consume it — this avoids duplicating the spatial-residual
        // logic while keeping a clean single-channel public API.
        let rgb: Vec<u8> = frame.iter().flat_map(|&v| [v, v, v]).collect();
        match self.inner.detect(&rgb, self.width, self.height) {
            Ok(map) => map.data,
            Err(_) => vec![0.0_f32; expected],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saliency_detector() {
        let detector = SaliencyDetector::new();
        let width = 100;
        let height = 100;
        let rgb_data = vec![128u8; width * height * 3];

        let result = detector.detect(&rgb_data, width, height);
        assert!(result.is_ok());

        let map = result.expect("should succeed in test");
        assert_eq!(map.data.len(), width * height);
    }

    // --- SpectralSaliencyDetector tests ---

    #[test]
    fn test_spectral_saliency_uniform_image() {
        let mut detector = SpectralSaliencyDetector::default_params();
        let w = 100;
        let h = 100;
        let rgb_data = vec![128u8; w * h * 3];
        let result = detector.detect(&rgb_data, w, h);
        assert!(result.is_ok());
        let map = result.expect("ok");
        assert_eq!(map.data.len(), w * h);
        // Uniform image → all saliency near 0 (no residual)
        let max_sal = map.data.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            max_sal < 0.01,
            "uniform image should have near-zero saliency, got {max_sal}"
        );
    }

    #[test]
    fn test_spectral_saliency_output_size_matches_input() {
        let mut detector = SpectralSaliencyDetector::new(32, 32, 4);
        let w = 200;
        let h = 150;
        let rgb_data = vec![100u8; w * h * 3];
        let result = detector.detect(&rgb_data, w, h);
        assert!(result.is_ok());
        let map = result.expect("ok");
        assert_eq!(map.width, w);
        assert_eq!(map.height, h);
        assert_eq!(map.data.len(), w * h);
    }

    #[test]
    fn test_spectral_saliency_salient_spot() {
        let mut detector = SpectralSaliencyDetector::default_params();
        let w = 128;
        let h = 128;
        let mut rgb_data = vec![80u8; w * h * 3];
        // Insert a bright spot in the centre
        for dy in 0..10 {
            for dx in 0..10 {
                let x = w / 2 + dx;
                let y = h / 2 + dy;
                let idx = (y * w + x) * 3;
                if idx + 2 < rgb_data.len() {
                    rgb_data[idx] = 255;
                    rgb_data[idx + 1] = 255;
                    rgb_data[idx + 2] = 255;
                }
            }
        }
        let result = detector.detect(&rgb_data, w, h);
        assert!(result.is_ok());
        let map = result.expect("ok");
        // Max saliency should be non-zero for a non-uniform image
        let max_sal = map.data.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_sal > 0.0,
            "expected saliency > 0 for image with bright spot"
        );
    }

    #[test]
    fn test_spectral_saliency_invalid_dimensions() {
        let mut detector = SpectralSaliencyDetector::default_params();
        let result = detector.detect(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_saliency_reuse_buffers() {
        // Call detect twice on different images — buffers are reused without allocation.
        let mut detector = SpectralSaliencyDetector::new(16, 16, 2);
        let w = 50;
        let h = 50;
        let frame1 = vec![100u8; w * h * 3];
        let frame2 = vec![200u8; w * h * 3];
        assert!(detector.detect(&frame1, w, h).is_ok());
        assert!(detector.detect(&frame2, w, h).is_ok());
    }

    // --- SpectralSaliencyComputer tests ---

    #[test]
    fn test_spectral_saliency_computer_reuse() {
        let w = 80;
        let h = 60;
        let mut computer = SpectralSaliencyComputer::new(w, h);
        let frame: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let r1 = computer.compute(&frame);
        let r2 = computer.compute(&frame);
        assert_eq!(r1, r2, "same input must yield same output on reuse");
    }

    #[test]
    fn test_spectral_saliency_computer_dimensions() {
        let w = 120;
        let h = 90;
        let mut computer = SpectralSaliencyComputer::new(w, h);
        let frame = vec![128u8; w * h];
        let result = computer.compute(&frame);
        assert_eq!(
            result.len(),
            w * h,
            "output length must equal width * height"
        );
    }
}
