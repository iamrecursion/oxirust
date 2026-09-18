//! Motion blur effect via directional convolution.
//!
//! Implements directional (linear) motion blur by building a 1D convolution kernel
//! oriented at the specified angle and applying it to the image. The kernel length
//! controls the blur distance. A box kernel gives a film-exposure-style blur.
//!
//! ## Frame accumulation cache
//!
//! [`MotionBlurCache`] is a ring-buffer that accumulates the last N frames and
//! computes a running-average blend, suitable for temporal motion blur from a
//! sequence of discrete video frames.  Adding a new frame is O(frame_size),
//! not O(N × frame_size), because a running per-channel sum is maintained.

use std::collections::VecDeque;

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

/// Configuration for motion blur.
#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    /// Blur direction in degrees (0 = horizontal right, 90 = downward).
    pub angle_degrees: f32,
    /// Number of samples / kernel length in pixels.
    /// More samples = more blur.
    pub samples: usize,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            angle_degrees: 0.0,
            samples: 15,
        }
    }
}

impl MotionBlurConfig {
    /// Horizontal blur (panning right).
    #[must_use]
    pub const fn horizontal(samples: usize) -> Self {
        Self {
            angle_degrees: 0.0,
            samples,
        }
    }

    /// Vertical blur (vertical pan or drop).
    #[must_use]
    pub const fn vertical(samples: usize) -> Self {
        Self {
            angle_degrees: 90.0,
            samples,
        }
    }

    /// Diagonal blur.
    #[must_use]
    pub const fn diagonal(samples: usize) -> Self {
        Self {
            angle_degrees: 45.0,
            samples,
        }
    }
}

/// Motion blur effect using directional box-kernel convolution.
pub struct MotionBlur {
    config: MotionBlurConfig,
}

impl MotionBlur {
    /// Create a new motion blur effect.
    #[must_use]
    pub fn new(config: MotionBlurConfig) -> Self {
        Self { config }
    }

    /// Apply motion blur to a pixel buffer in-place.
    ///
    /// The algorithm builds a set of `samples` offset vectors along the blur direction
    /// and averages the pixel values at those positions (box filter). Bilinear sampling
    /// is used for sub-pixel precision.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer size is incorrect.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn apply(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> VideoResult<()> {
        validate_buffer(data, width, height, format)?;
        if self.config.samples <= 1 {
            return Ok(()); // Nothing to do
        }

        let bpp = format.bytes_per_pixel();
        let angle_rad = self.config.angle_degrees.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let n = self.config.samples as f32;

        let source = data.to_vec();

        for py in 0..height {
            for px in 0..width {
                let mut acc_r = 0.0f32;
                let mut acc_g = 0.0f32;
                let mut acc_b = 0.0f32;
                let mut acc_a = 0.0f32;

                for s in 0..self.config.samples {
                    // Distribute samples symmetrically around the current pixel
                    let t = s as f32 - (n - 1.0) * 0.5;
                    let sx = px as f32 + t * cos_a;
                    let sy = py as f32 + t * sin_a;

                    let pixel = super::sample_bilinear(&source, width, height, bpp, sx, sy);
                    acc_r += pixel[0];
                    acc_g += pixel[1];
                    acc_b += pixel[2];
                    acc_a += pixel[3];
                }

                let inv_n = 1.0 / n;
                let idx = (py * width + px) * bpp;
                data[idx] = clamp_u8(acc_r * inv_n);
                data[idx + 1] = clamp_u8(acc_g * inv_n);
                data[idx + 2] = clamp_u8(acc_b * inv_n);
                if bpp == 4 {
                    data[idx + 3] = clamp_u8(acc_a * inv_n);
                }
            }
        }

        Ok(())
    }
}

// ── MotionBlurCache ──────────────────────────────────────────────────────────

/// Ring-buffer accumulation cache for temporal motion blur.
///
/// Stores the last `max_cache_size` frames and produces their pixel-wise
/// average via [`MotionBlurCache::accumulated_blend`].  A running per-pixel sum is maintained
/// so each `push_frame` call is O(frame_size) regardless of `max_cache_size`.
///
/// All frames must have the same pixel count (bytes length).  If a pushed
/// frame has a different length from the current running sum, the cache is
/// reset before adding the new frame.
///
/// # Example
///
/// ```ignore
/// use oximedia_effects::video::motion_blur::MotionBlurCache;
///
/// let mut cache = MotionBlurCache::new(3);
/// let frame = vec![128u8; 64 * 64 * 3];
/// for _ in 0..3 {
///     cache.push_frame(frame.clone());
/// }
/// let blend = cache.accumulated_blend();
/// assert_eq!(blend[0], 128);
/// ```
pub struct MotionBlurCache {
    /// Ring buffer of the last `max_cache_size` frames.
    cached_frames: VecDeque<Vec<u8>>,
    /// Maximum number of frames retained.
    max_cache_size: usize,
    /// Running sum of all cached frames (u32 per pixel-byte to avoid overflow).
    running_sum: Vec<u32>,
}

impl MotionBlurCache {
    /// Create a new `MotionBlurCache` that retains up to `max_samples` frames.
    ///
    /// `max_samples` is clamped to at least 1.
    #[must_use]
    pub fn new(max_samples: usize) -> Self {
        Self {
            cached_frames: VecDeque::new(),
            max_cache_size: max_samples.max(1),
            running_sum: Vec::new(),
        }
    }

    /// Return the number of frames currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cached_frames.len()
    }

    /// Return `true` if the cache contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cached_frames.is_empty()
    }

    /// Push a new frame into the cache.
    ///
    /// If the cache is full the oldest frame is evicted and its pixel values are
    /// subtracted from the running sum before adding the new frame.
    ///
    /// If `frame` has a different byte length from the current running sum, the
    /// cache is fully reset first (new frame replaces all state).
    pub fn push_frame(&mut self, frame: Vec<u8>) {
        // Reset on dimension change.
        if !self.running_sum.is_empty() && frame.len() != self.running_sum.len() {
            self.cached_frames.clear();
            self.running_sum.clear();
        }

        // Initialise running sum on first push.
        if self.running_sum.is_empty() {
            self.running_sum = vec![0u32; frame.len()];
        }

        // Evict oldest frame if at capacity.
        if self.cached_frames.len() == self.max_cache_size {
            if let Some(oldest) = self.cached_frames.pop_front() {
                for (sum, &old) in self.running_sum.iter_mut().zip(oldest.iter()) {
                    *sum = sum.saturating_sub(old as u32);
                }
            }
        }

        // Add new frame to running sum.
        for (sum, &new_val) in self.running_sum.iter_mut().zip(frame.iter()) {
            *sum += new_val as u32;
        }

        self.cached_frames.push_back(frame);
    }

    /// Return the pixel-wise average of all cached frames as a `Vec<u8>`.
    ///
    /// Returns an empty `Vec` if the cache is empty.
    #[must_use]
    pub fn accumulated_blend(&self) -> Vec<u8> {
        let n = self.cached_frames.len();
        if n == 0 {
            return Vec::new();
        }
        self.running_sum
            .iter()
            .map(|&s| (s / n as u32).min(255) as u8)
            .collect()
    }

    /// Clear all cached frames and reset the running sum.
    pub fn clear(&mut self) {
        self.cached_frames.clear();
        self.running_sum.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let val = if (px + py) % 2 == 0 { 255u8 } else { 0u8 };
                let idx = (py * w + px) * 3;
                buf[idx] = val;
                buf[idx + 1] = val;
                buf[idx + 2] = val;
            }
        }
        buf
    }

    #[test]
    fn test_motion_blur_horizontal() {
        let orig = checkerboard(64, 64);
        let mut buf = orig.clone();
        let mb = MotionBlur::new(MotionBlurConfig::horizontal(9));
        mb.apply(&mut buf, 64, 64, PixelFormat::Rgb)
            .expect("apply should succeed");
        // Blur should reduce sharp transitions; pixel values should be more uniform
        let diff: u32 = buf
            .iter()
            .zip(orig.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();
        assert!(diff > 0, "Blur should change the image");
    }

    #[test]
    fn test_motion_blur_samples_1_no_change() {
        let orig = checkerboard(32, 32);
        let mut buf = orig.clone();
        let mb = MotionBlur::new(MotionBlurConfig {
            samples: 1,
            ..Default::default()
        });
        mb.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "1 sample should not change image");
    }

    #[test]
    fn test_motion_blur_vertical() {
        let mut buf = checkerboard(32, 32);
        let mb = MotionBlur::new(MotionBlurConfig::vertical(7));
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_motion_blur_diagonal() {
        let mut buf = checkerboard(32, 32);
        let mb = MotionBlur::new(MotionBlurConfig::diagonal(7));
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_motion_blur_rgba() {
        let mut buf = vec![200u8; 32 * 32 * 4];
        let mb = MotionBlur::new(MotionBlurConfig::default());
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_motion_blur_wrong_size_err() {
        let mut buf = vec![0u8; 5];
        let mb = MotionBlur::new(MotionBlurConfig::default());
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_motion_blur_constant_image_unchanged() {
        // Blurring a constant-color image gives the same color
        let orig = vec![128u8; 32 * 32 * 3];
        let mut buf = orig.clone();
        let mb = MotionBlur::new(MotionBlurConfig::horizontal(11));
        mb.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "Constant image should survive blur");
    }

    // ── MotionBlurCache tests ─────────────────────────────────────────────────

    #[test]
    fn test_motion_blur_cache_correct() {
        // Cache of size 3 filled with identical frames → blend == that frame.
        let frame = vec![100u8; 16 * 16 * 3];
        let mut cache = MotionBlurCache::new(3);
        cache.push_frame(frame.clone());
        cache.push_frame(frame.clone());
        cache.push_frame(frame.clone());

        assert_eq!(cache.len(), 3);
        let blend = cache.accumulated_blend();
        assert_eq!(blend.len(), frame.len());
        for (&b, &f) in blend.iter().zip(frame.iter()) {
            assert_eq!(
                b, f,
                "blend of 3 identical frames should equal the frame: blend={b}, frame={f}"
            );
        }
    }

    #[test]
    fn test_motion_blur_cache_rolling() {
        // Push 5 frames into a cache of size 3.
        // Only the last 3 frames should contribute to the blend.
        // Frames: values 10, 20, 30, 40, 50 → after 5 pushes, last 3 = 30, 40, 50 → avg 40.
        let mut cache = MotionBlurCache::new(3);
        for value in [10u8, 20, 30, 40, 50] {
            let frame = vec![value; 4]; // 4-byte "frame" for simplicity
            cache.push_frame(frame);
        }
        assert_eq!(cache.len(), 3, "cache should hold exactly 3 frames");

        let blend = cache.accumulated_blend();
        assert_eq!(blend.len(), 4);
        // Average of 30, 40, 50 = 40.
        for &b in &blend {
            assert_eq!(
                b, 40,
                "rolling cache average of last 3 frames (30,40,50) should be 40, got {b}"
            );
        }
    }
}
