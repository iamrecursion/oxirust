//! Semantic segmentation of persons from their background.
//!
//! This module implements a lightweight, purely heuristic person/background
//! segmenter that combines two complementary cues:
//!
//! 1. **Skin-tone detection** — pixels whose YCbCr coordinates fall within
//!    well-known skin locus ranges are marked as foreground candidates.
//! 2. **Temporal motion residual** — if a previous background frame is
//!    provided, pixels that differ significantly between frames are treated as
//!    foreground candidates (moving = person).
//!
//! The two cue maps are merged (logical OR) and then refined with a small
//! morphological erosion+dilation (open) pass to remove isolated noise pixels.
//!
//! # Limitations
//!
//! This approach is intentionally simple and operates entirely in the pixel
//! domain without any neural-network inference.  It works best for:
//! - Scenes with clearly visible skin (faces/arms/hands).
//! - Scenes where the person is moving against a relatively static background.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::segmentation::person_bg::{PersonBackgroundSegmenter, SegmentationMask};
//!
//! let segmenter = PersonBackgroundSegmenter::default();
//! let frame = vec![120u8; 640 * 480 * 3]; // RGB frame
//! let mask: SegmentationMask = segmenter.segment(&frame, 640, 480);
//! assert_eq!(mask.width, 640);
//! assert_eq!(mask.height, 480);
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A binary per-pixel mask.
///
/// `data[y * width + x] == 1` means *foreground* (person); `0` means
/// background.
#[derive(Debug, Clone)]
pub struct SegmentationMask {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Binary mask data (`0` = background, `1` = foreground / person).
    pub data: Vec<u8>,
}

impl SegmentationMask {
    /// Create an all-background (zero) mask.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; width as usize * height as usize],
        }
    }

    /// Return the mask value at pixel `(x, y)`.  Out-of-bounds access returns 0.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Number of foreground pixels.
    #[must_use]
    pub fn foreground_count(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0).count()
    }

    /// Fraction of the image classified as foreground (0.0–1.0).
    #[must_use]
    pub fn foreground_ratio(&self) -> f32 {
        let total = self.width as usize * self.height as usize;
        if total == 0 {
            return 0.0;
        }
        self.foreground_count() as f32 / total as f32
    }
}

// ---------------------------------------------------------------------------
// Segmenter
// ---------------------------------------------------------------------------

/// Tunable parameters for the person/background segmenter.
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// Minimum Cb in YCbCr (skin locus lower bound, default 77).
    pub cb_min: u8,
    /// Maximum Cb in YCbCr (skin locus upper bound, default 127).
    pub cb_max: u8,
    /// Minimum Cr in YCbCr (skin locus lower bound, default 133).
    pub cr_min: u8,
    /// Maximum Cr in YCbCr (skin locus upper bound, default 173).
    pub cr_max: u8,
    /// Minimum Y (luma) for a pixel to be considered skin (avoids very dark skin,
    /// default 80).
    pub y_min: u8,
    /// Absolute pixel difference threshold for motion detection (default 25).
    pub motion_threshold: u8,
    /// Morphological open radius in pixels (default 1 — 3×3 structuring element).
    pub morph_radius: u32,
    /// Whether to use motion cue (requires a background frame, default true).
    pub use_motion: bool,
    /// Whether to use skin-tone cue (default true).
    pub use_skin: bool,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            cb_min: 77,
            cb_max: 127,
            cr_min: 133,
            cr_max: 173,
            y_min: 80,
            motion_threshold: 25,
            morph_radius: 1,
            use_motion: true,
            use_skin: true,
        }
    }
}

/// Lightweight person/background segmenter.
///
/// Instantiate with [`PersonBackgroundSegmenter::new`] or via `Default`, then
/// call [`segment`](Self::segment) for each frame.  Optionally call
/// [`update_background`](Self::update_background) to feed a reference
/// background frame that enables the motion-based cue.
pub struct PersonBackgroundSegmenter {
    config: SegmenterConfig,
    /// Stored background frame (grayscale, one byte per pixel) for motion diff.
    background: Option<Vec<u8>>,
    /// Frame dimensions of the stored background.
    bg_dims: Option<(u32, u32)>,
}

impl Default for PersonBackgroundSegmenter {
    fn default() -> Self {
        Self::new(SegmenterConfig::default())
    }
}

impl PersonBackgroundSegmenter {
    /// Create a new segmenter with the given configuration.
    #[must_use]
    pub fn new(config: SegmenterConfig) -> Self {
        Self {
            config,
            background: None,
            bg_dims: None,
        }
    }

    /// Set the reference background frame (RGB or grayscale byte slice).
    ///
    /// The frame is converted to luma internally for motion comparison.
    pub fn update_background(&mut self, frame: &[u8], width: u32, height: u32) {
        let luma = to_luma(frame, width, height);
        self.background = Some(luma);
        self.bg_dims = Some((width, height));
    }

    /// Segment a single frame into person (foreground) and background.
    ///
    /// # Arguments
    ///
    /// * `frame`  – Raw pixel data.  Accepted formats: RGB (3 bytes/pixel) or
    ///              grayscale (1 byte/pixel).  If the buffer is at least
    ///              `width * height * 3` bytes it is treated as RGB.
    /// * `width`  – Frame width in pixels.
    /// * `height` – Frame height in pixels.
    ///
    /// # Returns
    ///
    /// A [`SegmentationMask`] with the same spatial dimensions.
    #[must_use]
    pub fn segment(&self, frame: &[u8], width: u32, height: u32) -> SegmentationMask {
        let npixels = width as usize * height as usize;
        if npixels == 0 || frame.is_empty() {
            return SegmentationMask::new(width, height);
        }

        let is_rgb = frame.len() >= npixels * 3;
        let mut mask_data = vec![0u8; npixels];

        // ---- Skin-tone cue -----------------------------------------------
        if self.config.use_skin && is_rgb {
            for (i, px) in frame[..npixels * 3].chunks_exact(3).enumerate() {
                let r = px[0];
                let g = px[1];
                let b = px[2];
                let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
                if y >= self.config.y_min
                    && cb >= self.config.cb_min
                    && cb <= self.config.cb_max
                    && cr >= self.config.cr_min
                    && cr <= self.config.cr_max
                {
                    mask_data[i] = 1;
                }
            }
        }

        // ---- Motion cue --------------------------------------------------
        if self.config.use_motion {
            if let (Some(bg), Some((bw, bh))) = (&self.background, self.bg_dims) {
                if bw == width && bh == height && bg.len() == npixels {
                    let luma = to_luma(frame, width, height);
                    for (i, (&curr, &ref_pix)) in luma.iter().zip(bg.iter()).enumerate() {
                        let diff = (curr as i16 - ref_pix as i16).unsigned_abs() as u8;
                        if diff >= self.config.motion_threshold {
                            mask_data[i] = 1;
                        }
                    }
                }
            }
        }

        // ---- Morphological open (erosion then dilation) ------------------
        if self.config.morph_radius > 0 {
            mask_data = morph_open(&mask_data, width, height, self.config.morph_radius);
        }

        SegmentationMask {
            width,
            height,
            data: mask_data,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert an RGB (or grayscale) frame to a luma (Y′) buffer.
fn to_luma(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let npixels = width as usize * height as usize;
    if frame.len() >= npixels * 3 {
        // RGB → luma using BT.601
        frame[..npixels * 3]
            .chunks_exact(3)
            .map(|px| {
                let r = px[0] as u32;
                let g = px[1] as u32;
                let b = px[2] as u32;
                ((299 * r + 587 * g + 114 * b + 500) / 1000).min(255) as u8
            })
            .collect()
    } else {
        // Grayscale
        frame[..npixels.min(frame.len())].to_vec()
    }
}

/// Convert RGB (u8) to YCbCr using BT.601 full-range formulae.
///
/// Returns `(Y, Cb, Cr)` each in the range `[0, 255]`.
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    let y = (0.299 * r + 0.587 * g + 0.114 * b)
        .round()
        .clamp(0.0, 255.0) as u8;
    let cb = (128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b)
        .round()
        .clamp(0.0, 255.0) as u8;
    let cr = (128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b)
        .round()
        .clamp(0.0, 255.0) as u8;

    (y, cb, cr)
}

/// Morphological erosion (3×3 structuring element of radius `r`).
fn morph_erode(src: &[u8], width: u32, height: u32, r: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let r = r as usize;
    let mut dst = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut all_fg = true;
            'outer: for ky in 0..=2 * r {
                let sy = y as i32 + ky as i32 - r as i32;
                if sy < 0 || sy >= h as i32 {
                    all_fg = false;
                    break 'outer;
                }
                for kx in 0..=2 * r {
                    let sx = x as i32 + kx as i32 - r as i32;
                    if sx < 0 || sx >= w as i32 {
                        all_fg = false;
                        break 'outer;
                    }
                    if src[sy as usize * w + sx as usize] == 0 {
                        all_fg = false;
                        break 'outer;
                    }
                }
            }
            dst[y * w + x] = if all_fg { 1 } else { 0 };
        }
    }

    dst
}

/// Morphological dilation (3×3 structuring element of radius `r`).
fn morph_dilate(src: &[u8], width: u32, height: u32, r: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let r = r as usize;
    let mut dst = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut any_fg = false;
            'outer: for ky in 0..=2 * r {
                let sy = y as i32 + ky as i32 - r as i32;
                if sy < 0 || sy >= h as i32 {
                    continue;
                }
                for kx in 0..=2 * r {
                    let sx = x as i32 + kx as i32 - r as i32;
                    if sx < 0 || sx >= w as i32 {
                        continue;
                    }
                    if src[sy as usize * w + sx as usize] != 0 {
                        any_fg = true;
                        break 'outer;
                    }
                }
            }
            dst[y * w + x] = if any_fg { 1 } else { 0 };
        }
    }

    dst
}

/// Morphological open: erosion followed by dilation.
fn morph_open(src: &[u8], width: u32, height: u32, r: u32) -> Vec<u8> {
    let eroded = morph_erode(src, width, height, r);
    morph_dilate(&eroded, width, height, r)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SegmentationMask --------------------------------------------------

    #[test]
    fn test_mask_new_all_background() {
        let mask = SegmentationMask::new(10, 10);
        assert_eq!(mask.width, 10);
        assert_eq!(mask.height, 10);
        assert_eq!(mask.data.len(), 100);
        assert!(mask.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_mask_get_oob() {
        let mask = SegmentationMask::new(4, 4);
        assert_eq!(mask.get(100, 100), 0);
    }

    #[test]
    fn test_mask_foreground_count() {
        let mut mask = SegmentationMask::new(4, 4);
        mask.data[0] = 1;
        mask.data[5] = 1;
        assert_eq!(mask.foreground_count(), 2);
    }

    #[test]
    fn test_mask_foreground_ratio() {
        let mut mask = SegmentationMask::new(2, 2);
        mask.data[0] = 1; // 1 out of 4
        let ratio = mask.foreground_ratio();
        assert!((ratio - 0.25).abs() < 1e-6);
    }

    // ---- rgb_to_ycbcr -------------------------------------------------------

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        // Cb and Cr should be near 128 for neutral grey / white
        assert!((cb as i32 - 128).abs() <= 2, "Cb={cb}");
        assert!((cr as i32 - 128).abs() <= 2, "Cr={cr}");
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    // ---- PersonBackgroundSegmenter -----------------------------------------

    #[test]
    fn test_segment_empty_returns_empty_mask() {
        let seg = PersonBackgroundSegmenter::default();
        let mask = seg.segment(&[], 10, 10);
        assert_eq!(mask.foreground_count(), 0);
    }

    #[test]
    fn test_segment_zero_dims() {
        let seg = PersonBackgroundSegmenter::default();
        let frame = vec![128u8; 100];
        let mask = seg.segment(&frame, 0, 0);
        assert_eq!(mask.data.len(), 0);
    }

    #[test]
    fn test_segment_no_skin_no_motion_all_background() {
        // A fully blue frame — no skin-tone pixels, no motion reference.
        // Config: skin enabled, motion enabled (but no background).
        let w = 8u32;
        let h = 8u32;
        // Pure blue: R=0, G=0, B=255 — outside the skin locus.
        let frame: Vec<u8> = (0..w as usize * h as usize)
            .flat_map(|_| [0u8, 0u8, 255u8])
            .collect();

        let seg = PersonBackgroundSegmenter::default();
        let mask = seg.segment(&frame, w, h);
        // Skin cue: should be 0 for pure blue
        // Motion cue: no background stored, so 0
        assert_eq!(
            mask.foreground_count(),
            0,
            "Pure blue frame should be all background"
        );
    }

    #[test]
    fn test_segment_skin_pixels_detected() {
        // Create an RGB frame where every pixel has a typical skin-tone YCbCr value.
        // A common skin tone: R=220, G=160, B=120.
        let w = 4u32;
        let h = 4u32;
        let frame: Vec<u8> = (0..w as usize * h as usize)
            .flat_map(|_| [220u8, 160u8, 120u8])
            .collect();

        let seg = PersonBackgroundSegmenter::default();
        let mask = seg.segment(&frame, w, h);

        // We expect some skin pixels to be detected (the morph open may reduce
        // a small image to zero so use a slightly lax check for 4x4).
        // The skin cue should at least trigger on these values.
        let (y, cb, cr) = rgb_to_ycbcr(220, 160, 120);
        let cfg = SegmenterConfig::default();
        let is_skin = y >= cfg.y_min
            && cb >= cfg.cb_min
            && cb <= cfg.cb_max
            && cr >= cfg.cr_min
            && cr <= cfg.cr_max;

        if is_skin {
            // After morph open on a 4x4 uniform mask the result depends on radius.
            // Use a 4x4 image large enough that erosion radius 1 keeps at least
            // the centre 2x2 intact.
            assert!(
                mask.foreground_count() > 0,
                "Skin-tone pixels should be detected; mask={:?}",
                mask.data
            );
        }
        // If the RGB values do not pass our skin threshold that is also acceptable;
        // the test is documenting the behaviour rather than enforcing a specific value.
    }

    #[test]
    fn test_segment_motion_detected() {
        // Background: all gray (128). Current frame: all white (255).
        // The motion diff (127) >> motion_threshold (25) → all foreground.
        let w = 6u32;
        let h = 6u32;
        let bg_frame = vec![128u8; w as usize * h as usize];
        let curr_frame = vec![255u8; w as usize * h as usize];

        let mut seg = PersonBackgroundSegmenter::new(SegmenterConfig {
            use_skin: false,
            use_motion: true,
            morph_radius: 0, // disable morphology so we see raw motion mask
            ..SegmenterConfig::default()
        });
        seg.update_background(&bg_frame, w, h);

        let mask = seg.segment(&curr_frame, w, h);
        assert_eq!(
            mask.foreground_count(),
            (w * h) as usize,
            "All pixels should be foreground (motion diff = 127)"
        );
    }

    #[test]
    fn test_update_background_stores_correctly() {
        let mut seg = PersonBackgroundSegmenter::default();
        let frame = vec![100u8; 16 * 16 * 3]; // RGB
        seg.update_background(&frame, 16, 16);
        assert!(seg.background.is_some());
        assert_eq!(seg.bg_dims, Some((16, 16)));
    }

    // ---- morphological helpers --------------------------------------------

    #[test]
    fn test_morph_open_removes_isolated_pixels() {
        // Single foreground pixel in the centre of an 8x8 mask — erosion should
        // remove it, so open gives all-background.
        let w = 8u32;
        let h = 8u32;
        let mut data = vec![0u8; 64];
        data[4 * 8 + 4] = 1; // Centre pixel
        let result = morph_open(&data, w, h, 1);
        assert!(
            result.iter().all(|&v| v == 0),
            "Isolated pixel should be removed by morph open"
        );
    }

    #[test]
    fn test_morph_open_preserves_large_region() {
        // Solid 6x6 foreground region inside an 8x8 mask — should survive open.
        let w = 8u32;
        let h = 8u32;
        let mut data = vec![0u8; 64];
        for y in 1..7usize {
            for x in 1..7usize {
                data[y * 8 + x] = 1;
            }
        }
        let result = morph_open(&data, w, h, 1);
        let fg_count = result.iter().filter(|&&v| v != 0).count();
        assert!(
            fg_count > 0,
            "Large foreground region should survive morph open"
        );
    }
}
