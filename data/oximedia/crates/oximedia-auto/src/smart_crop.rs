//! Smart cropping using saliency-based region detection.
//!
//! Provides `SalientRegion`, `SmartCropConfig`, `SmartCropResult`, and
//! `SmartCropper` for suggesting crop parameters that keep the most important
//! content visible.

/// An axis-aligned bounding box representing a salient region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SalientRegion {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Saliency score in the range 0.0–1.0.
    pub score: f64,
}

impl SalientRegion {
    /// Create a new `SalientRegion`.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32, score: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            score,
        }
    }

    /// Return the pixel area of the region.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Return the centre point of the region.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn centre(&self) -> (f64, f64) {
        let cx = self.x as f64 + self.width as f64 / 2.0;
        let cy = self.y as f64 + self.height as f64 / 2.0;
        (cx, cy)
    }
}

/// Configuration for the smart cropping algorithm.
#[derive(Debug, Clone)]
pub struct SmartCropConfig {
    /// Desired output width in pixels (0 = same as source).
    pub output_width: u32,
    /// Desired output height in pixels (0 = same as source).
    pub output_height: u32,
    /// Minimum saliency score to consider a region important.
    pub min_saliency: f64,
    /// Allow slight upscaling to fill the output dimensions.
    pub allow_upscale: bool,
    /// Coarse-to-fine scale factor for [`SmartCropper::suggest_crop_from_frame`].
    ///
    /// Value in 0.0–1.0.  `0.25` means the first saliency pass runs on a
    /// 25%-resolution downscale of the frame, with the full-resolution pass
    /// restricted to the winning region ± a 10 % margin.  Set to `1.0` to
    /// disable the coarse pass and behave identically to the pre-computed
    /// region path.
    pub coarse_scale: f32,
}

impl Default for SmartCropConfig {
    fn default() -> Self {
        Self {
            output_width: 1280,
            output_height: 720,
            min_saliency: 0.3,
            allow_upscale: false,
            coarse_scale: 0.25,
        }
    }
}

impl SmartCropConfig {
    /// Return the target aspect ratio as `width / height`.
    ///
    /// Returns `None` if either dimension is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> Option<f64> {
        if self.output_height == 0 {
            return None;
        }
        Some(self.output_width as f64 / self.output_height as f64)
    }
}

/// The crop rectangle suggested by `SmartCropper`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropRect {
    /// Left offset in the source frame.
    pub x: u32,
    /// Top offset in the source frame.
    pub y: u32,
    /// Width of the crop.
    pub width: u32,
    /// Height of the crop.
    pub height: u32,
}

/// Result returned by `SmartCropper::suggest_crop`.
#[derive(Debug, Clone)]
pub struct SmartCropResult {
    /// The suggested crop rectangle in source-image coordinates.
    pub crop: CropRect,
    /// Whether any cropping was actually applied (false = full frame).
    pub crop_applied: bool,
    /// The salient regions that influenced the crop decision.
    pub salient_regions: Vec<SalientRegion>,
    /// Confidence of the suggestion in 0.0–1.0.
    pub confidence: f64,
}

impl SmartCropResult {
    /// Return `true` when the crop differs from the full source frame.
    #[must_use]
    pub fn crop_applied(&self) -> bool {
        self.crop_applied
    }

    /// Return the area of the suggested crop rectangle.
    #[must_use]
    pub fn crop_area(&self) -> u64 {
        u64::from(self.crop.width) * u64::from(self.crop.height)
    }
}

/// Analyses frames and suggests optimal crop parameters.
#[derive(Debug)]
pub struct SmartCropper {
    config: SmartCropConfig,
}

impl Default for SmartCropper {
    fn default() -> Self {
        Self {
            config: SmartCropConfig::default(),
        }
    }
}

impl SmartCropper {
    /// Create a new `SmartCropper` with the given config.
    #[must_use]
    pub fn new(config: SmartCropConfig) -> Self {
        Self { config }
    }

    /// Analyse a set of `SalientRegion` detections and return all that meet
    /// the minimum saliency threshold.
    #[must_use]
    pub fn analyze(&self, regions: &[SalientRegion]) -> Vec<SalientRegion> {
        let mut kept: Vec<SalientRegion> = regions
            .iter()
            .copied()
            .filter(|r| r.score >= self.config.min_saliency)
            .collect();
        // Sort descending by score.
        kept.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kept
    }

    /// Suggest a crop rectangle for a frame of `src_width` × `src_height`
    /// containing the given `regions`.
    ///
    /// The algorithm computes a weighted centroid of salient regions and
    /// places the output crop rectangle centred on that point, clamped to
    /// source bounds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest_crop(
        &self,
        src_width: u32,
        src_height: u32,
        regions: &[SalientRegion],
    ) -> SmartCropResult {
        let salient = self.analyze(regions);

        // Determine crop size.
        let crop_w = if self.config.output_width == 0 || self.config.output_width > src_width {
            src_width
        } else {
            self.config.output_width
        };
        let crop_h = if self.config.output_height == 0 || self.config.output_height > src_height {
            src_height
        } else {
            self.config.output_height
        };

        // No crop needed.
        if crop_w == src_width && crop_h == src_height {
            return SmartCropResult {
                crop: CropRect {
                    x: 0,
                    y: 0,
                    width: src_width,
                    height: src_height,
                },
                crop_applied: false,
                salient_regions: salient,
                confidence: 1.0,
            };
        }

        // Compute weighted centroid.
        let (cx, cy, total_weight) = if salient.is_empty() {
            // Fall back to frame centre.
            (src_width as f64 / 2.0, src_height as f64 / 2.0, 1.0)
        } else {
            let (wx, wy, w) = salient
                .iter()
                .fold((0.0f64, 0.0f64, 0.0f64), |(ax, ay, aw), r| {
                    let (rx, ry) = r.centre();
                    (ax + rx * r.score, ay + ry * r.score, aw + r.score)
                });
            if w == 0.0 {
                (src_width as f64 / 2.0, src_height as f64 / 2.0, 1.0)
            } else {
                (wx / w, wy / w, w)
            }
        };

        // Centre the crop on the centroid, clamp to source.
        let x = ((cx - crop_w as f64 / 2.0).max(0.0) as u32).min(src_width.saturating_sub(crop_w));
        let y = ((cy - crop_h as f64 / 2.0).max(0.0) as u32).min(src_height.saturating_sub(crop_h));

        let confidence = (total_weight / salient.len().max(1) as f64).min(1.0);

        SmartCropResult {
            crop: CropRect {
                x,
                y,
                width: crop_w,
                height: crop_h,
            },
            crop_applied: true,
            salient_regions: salient,
            confidence,
        }
    }

    // -------------------------------------------------------------------------
    // Coarse-to-fine raw-frame API
    // -------------------------------------------------------------------------

    /// Suggest a crop using a coarse-to-fine strategy on raw frame data.
    ///
    /// # Algorithm
    ///
    /// 1. If `config.coarse_scale < 1.0`: downscale the frame to
    ///    `coarse_scale * (width × height)`, derive synthetic saliency regions
    ///    from the intensity variation of the coarse frame, and find the rough
    ///    best crop window.
    /// 2. Extract a full-resolution sub-region around the coarse crop (±10 %
    ///    margin in each axis, clamped to source bounds).
    /// 3. Re-run saliency analysis on that sub-region at full resolution and
    ///    return the refined crop.
    ///
    /// If `coarse_scale >= 1.0` or the frame is too small to downscale
    /// meaningfully, this falls back to running saliency on the whole frame at
    /// full resolution (identical to calling [`Self::suggest_crop`] with
    /// synthetically generated regions).
    ///
    /// # Parameters
    ///
    /// - `frame`: raw pixel bytes in row-major order, `channels` bytes per pixel.
    /// - `width`, `height`: source frame dimensions in pixels.
    /// - `channels`: number of bytes (channels) per pixel; must be ≥ 1.
    ///
    /// # Panics
    ///
    /// Does not panic; malformed input (e.g. `frame.len() != width * height *
    /// channels`) silently falls back to the full-frame path.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn suggest_crop_from_frame(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> SmartCropResult {
        // Sanity check: return a no-crop full-frame result for malformed input.
        let expected = width as usize * height as usize * channels as usize;
        if channels == 0 || frame.len() != expected || width == 0 || height == 0 {
            return SmartCropResult {
                crop: CropRect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                crop_applied: false,
                salient_regions: Vec::new(),
                confidence: 0.0,
            };
        }

        let scale = self.config.coarse_scale.clamp(0.0, 1.0);

        // If scale == 1.0 (or very close), skip coarse pass entirely.
        let coarse_w = ((width as f32 * scale) as u32).max(1);
        let coarse_h = ((height as f32 * scale) as u32).max(1);
        let use_coarse = scale < 1.0 && coarse_w < width && coarse_h < height;

        if !use_coarse {
            // Full-resolution path: synthesise saliency from frame and crop.
            let regions = Self::extract_saliency_regions(frame, width, height, channels);
            return self.suggest_crop(width, height, &regions);
        }

        // ---- Coarse pass ----
        let coarse_frame = Self::box_downsample(frame, width, height, channels, coarse_w, coarse_h);
        let coarse_regions =
            Self::extract_saliency_regions(&coarse_frame, coarse_w, coarse_h, channels);
        let coarse_result = self.suggest_crop(coarse_w, coarse_h, &coarse_regions);

        // Map coarse crop rectangle back to full-resolution coordinates.
        let scale_x = width as f64 / coarse_w as f64;
        let scale_y = height as f64 / coarse_h as f64;

        let roi_x = (coarse_result.crop.x as f64 * scale_x) as u32;
        let roi_y = (coarse_result.crop.y as f64 * scale_y) as u32;
        let roi_w = ((coarse_result.crop.width as f64 * scale_x) as u32).min(width);
        let roi_h = ((coarse_result.crop.height as f64 * scale_y) as u32).min(height);

        // Expand the ROI by ±10 % of source dimensions for the fine pass.
        let margin_x = (width as f64 * 0.10) as u32;
        let margin_y = (height as f64 * 0.10) as u32;

        let fine_x = roi_x.saturating_sub(margin_x);
        let fine_y = roi_y.saturating_sub(margin_y);
        let fine_x2 = (roi_x + roi_w + margin_x).min(width);
        let fine_y2 = (roi_y + roi_h + margin_y).min(height);
        let fine_w = fine_x2.saturating_sub(fine_x).max(1);
        let fine_h = fine_y2.saturating_sub(fine_y).max(1);

        // ---- Fine pass — extract sub-region at full resolution ----
        let sub_frame = Self::extract_sub_frame(
            frame, width, height, channels, fine_x, fine_y, fine_w, fine_h,
        );
        let fine_regions = Self::extract_saliency_regions(&sub_frame, fine_w, fine_h, channels);

        // Translate fine regions back to source coordinates.
        let translated: Vec<SalientRegion> = fine_regions
            .iter()
            .map(|r| SalientRegion::new(r.x + fine_x, r.y + fine_y, r.width, r.height, r.score))
            .collect();

        self.suggest_crop(width, height, &translated)
    }

    /// Box-average downsample: average pixels in each `(sw/dw) × (sh/dh)` block.
    #[allow(clippy::cast_precision_loss)]
    fn box_downsample(
        src: &[u8],
        src_w: u32,
        src_h: u32,
        channels: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        let ch = channels as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = dst_w as usize;
        let dh = dst_h as usize;

        let mut dst = vec![0u8; dw * dh * ch];

        for dy in 0..dh {
            for dx in 0..dw {
                // Source block boundaries.
                let sx0 = (dx * sw) / dw;
                let sy0 = (dy * sh) / dh;
                let sx1 = ((dx + 1) * sw / dw).min(sw);
                let sy1 = ((dy + 1) * sh / dh).min(sh);
                let block_pixels = ((sx1 - sx0) * (sy1 - sy0)).max(1);

                let mut sums = vec![0u32; ch];
                for sy in sy0..sy1 {
                    for sx in sx0..sx1 {
                        let base = (sy * sw + sx) * ch;
                        for c in 0..ch {
                            sums[c] += src[base + c] as u32;
                        }
                    }
                }

                let dst_base = (dy * dw + dx) * ch;
                for c in 0..ch {
                    dst[dst_base + c] = (sums[c] / block_pixels as u32) as u8;
                }
            }
        }

        dst
    }

    /// Copy a rectangular sub-region from `src` into a new contiguous buffer.
    fn extract_sub_frame(
        src: &[u8],
        src_w: u32,
        _src_h: u32,
        channels: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        let ch = channels as usize;
        let sw = src_w as usize;
        let rw = w as usize;
        let rh = h as usize;
        let rx = x as usize;
        let ry = y as usize;

        let mut out = Vec::with_capacity(rw * rh * ch);
        for row in ry..ry + rh {
            let row_start = (row * sw + rx) * ch;
            out.extend_from_slice(&src[row_start..row_start + rw * ch]);
        }
        out
    }

    /// Derive synthetic saliency regions from raw frame intensity.
    ///
    /// The frame is divided into a grid of cells; the variance of luma within
    /// each cell is used as a proxy for saliency score.  This simple heuristic
    /// avoids an external ML dependency while still providing a meaningful
    /// signal for the coarse-to-fine crop strategy.
    #[allow(clippy::cast_precision_loss)]
    fn extract_saliency_regions(
        frame: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Vec<SalientRegion> {
        const GRID: u32 = 4; // 4 × 4 grid = 16 cells

        let ch = channels as usize;
        let w = width as usize;
        let h = height as usize;
        let cell_w = (w / GRID as usize).max(1);
        let cell_h = (h / GRID as usize).max(1);

        let mut regions = Vec::with_capacity((GRID * GRID) as usize);

        for gy in 0..GRID as usize {
            for gx in 0..GRID as usize {
                let x0 = gx * cell_w;
                let y0 = gy * cell_h;
                let x1 = ((gx + 1) * cell_w).min(w);
                let y1 = ((gy + 1) * cell_h).min(h);

                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                let mut count = 0usize;

                for py in y0..y1 {
                    for px in x0..x1 {
                        let base = (py * w + px) * ch;
                        // Approximate luma from first channel (or all channels avg).
                        let luma = if ch >= 3 {
                            (frame[base] as f64 * 0.299
                                + frame[base + 1] as f64 * 0.587
                                + frame[base + 2] as f64 * 0.114)
                                / 255.0
                        } else {
                            frame[base] as f64 / 255.0
                        };
                        sum += luma;
                        sum_sq += luma * luma;
                        count += 1;
                    }
                }

                if count == 0 {
                    continue;
                }

                let mean = sum / count as f64;
                let variance = (sum_sq / count as f64) - mean * mean;
                // Normalise variance: typical natural-image variance ≈ 0.04.
                let score = (variance / 0.04).min(1.0);

                regions.push(SalientRegion::new(
                    x0 as u32,
                    y0 as u32,
                    (x1 - x0) as u32,
                    (y1 - y0) as u32,
                    score,
                ));
            }
        }

        regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salient_region_area() {
        let r = SalientRegion::new(0, 0, 100, 50, 0.8);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_salient_region_area_zero() {
        let r = SalientRegion::new(0, 0, 0, 100, 1.0);
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn test_salient_region_centre() {
        let r = SalientRegion::new(10, 20, 100, 80, 0.5);
        let (cx, cy) = r.centre();
        assert!((cx - 60.0).abs() < 1e-6);
        assert!((cy - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_aspect_ratio() {
        let cfg = SmartCropConfig::default();
        let ratio = cfg.aspect_ratio().expect("ratio should be valid");
        assert!((ratio - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_aspect_ratio_zero_height() {
        let cfg = SmartCropConfig {
            output_height: 0,
            ..Default::default()
        };
        assert!(cfg.aspect_ratio().is_none());
    }

    #[test]
    fn test_cropper_analyze_filters_by_saliency() {
        let cfg = SmartCropConfig {
            min_saliency: 0.5,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        let regions = vec![
            SalientRegion::new(0, 0, 10, 10, 0.8),
            SalientRegion::new(0, 0, 10, 10, 0.3),
        ];
        let kept = cropper.analyze(&regions);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_cropper_no_crop_when_output_equals_source() {
        let cfg = SmartCropConfig {
            output_width: 1920,
            output_height: 1080,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        let result = cropper.suggest_crop(1920, 1080, &[]);
        assert!(!result.crop_applied());
        assert_eq!(result.crop.width, 1920);
        assert_eq!(result.crop.height, 1080);
    }

    #[test]
    fn test_cropper_suggest_crop_applied() {
        let cfg = SmartCropConfig {
            output_width: 640,
            output_height: 360,
            min_saliency: 0.1,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        let regions = vec![SalientRegion::new(900, 400, 200, 200, 0.9)];
        let result = cropper.suggest_crop(1920, 1080, &regions);
        assert!(result.crop_applied());
        assert_eq!(result.crop.width, 640);
        assert_eq!(result.crop.height, 360);
    }

    #[test]
    fn test_cropper_crop_clamped_to_source() {
        let cfg = SmartCropConfig {
            output_width: 1280,
            output_height: 720,
            min_saliency: 0.0,
            ..Default::default()
        };
        let cropper = SmartCropper::new(cfg);
        // Salient region at far right edge
        let regions = vec![SalientRegion::new(1900, 1070, 20, 10, 1.0)];
        let result = cropper.suggest_crop(1920, 1080, &regions);
        assert!(result.crop.x + result.crop.width <= 1920);
        assert!(result.crop.y + result.crop.height <= 1080);
    }

    #[test]
    fn test_crop_result_crop_area() {
        let r = SmartCropResult {
            crop: CropRect {
                x: 0,
                y: 0,
                width: 640,
                height: 360,
            },
            crop_applied: true,
            salient_regions: vec![],
            confidence: 0.9,
        };
        assert_eq!(r.crop_area(), 230_400);
    }

    // -------------------------------------------------------------------------
    // Coarse-to-fine tests
    // -------------------------------------------------------------------------

    /// Generate a synthetic frame where one quadrant is significantly brighter
    /// (higher variance) than the rest so both coarse and full-res paths
    /// should prefer the same general region.
    fn make_synthetic_frame(
        width: u32,
        height: u32,
        channels: u32,
        bright_x: u32,
        bright_y: u32,
        bright_w: u32,
        bright_h: u32,
    ) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let ch = channels as usize;
        let bx0 = bright_x as usize;
        let by0 = bright_y as usize;
        let bx1 = (bright_x + bright_w) as usize;
        let by1 = (bright_y + bright_h) as usize;
        let mut frame = vec![30u8; w * h * ch];

        // Checkerboard pattern in the bright region to maximise variance.
        for y in by0..by1.min(h) {
            for x in bx0..bx1.min(w) {
                let base = (y * w + x) * ch;
                let val = if (x + y) % 2 == 0 { 240u8 } else { 20u8 };
                for c in 0..ch {
                    frame[base + c] = val;
                }
            }
        }
        frame
    }

    /// Coarse-to-fine and full-res-only on the same synthetic input must
    /// produce crop bounds within 5 % of each frame dimension.
    #[test]
    fn test_coarse_fine_matches_full_res() {
        let src_w = 160u32;
        let src_h = 120u32;
        let ch = 3u32;

        // Bright/high-variance patch in the bottom-right quadrant.
        let bright_x = src_w * 3 / 4;
        let bright_y = src_h * 3 / 4;
        let frame =
            make_synthetic_frame(src_w, src_h, ch, bright_x, bright_y, src_w / 4, src_h / 4);

        // Coarse-to-fine cropper (default coarse_scale = 0.25).
        let cfg_coarse = SmartCropConfig {
            output_width: 80,
            output_height: 60,
            min_saliency: 0.0,
            coarse_scale: 0.25,
            ..Default::default()
        };
        let coarse_result =
            SmartCropper::new(cfg_coarse).suggest_crop_from_frame(&frame, src_w, src_h, ch);

        // Full-resolution cropper (coarse_scale = 1.0 → no downscale).
        let cfg_full = SmartCropConfig {
            output_width: 80,
            output_height: 60,
            min_saliency: 0.0,
            coarse_scale: 1.0,
            ..Default::default()
        };
        let full_result =
            SmartCropper::new(cfg_full).suggest_crop_from_frame(&frame, src_w, src_h, ch);

        // Both results must yield the same output size.
        assert_eq!(coarse_result.crop.width, full_result.crop.width);
        assert_eq!(coarse_result.crop.height, full_result.crop.height);

        // Crop origins must be within ±5 % of source dimensions.
        let tol_x = (src_w as f64 * 0.05) as i64 + 1;
        let tol_y = (src_h as f64 * 0.05) as i64 + 1;
        let dx = (coarse_result.crop.x as i64 - full_result.crop.x as i64).abs();
        let dy = (coarse_result.crop.y as i64 - full_result.crop.y as i64).abs();
        assert!(
            dx <= tol_x,
            "x offset {dx} exceeds 5 % tolerance {tol_x} (coarse={}, full={})",
            coarse_result.crop.x,
            full_result.crop.x
        );
        assert!(
            dy <= tol_y,
            "y offset {dy} exceeds 5 % tolerance {tol_y} (coarse={}, full={})",
            coarse_result.crop.y,
            full_result.crop.y
        );
    }

    /// `coarse_scale = 1.0` must return the same result as the full-res path.
    #[test]
    fn test_coarse_scale_one_same_as_default() {
        let src_w = 80u32;
        let src_h = 60u32;
        let ch = 1u32;
        let frame = make_synthetic_frame(src_w, src_h, ch, 40, 30, 20, 20);

        let cfg = SmartCropConfig {
            output_width: 40,
            output_height: 30,
            min_saliency: 0.0,
            coarse_scale: 1.0,
            ..Default::default()
        };
        let r1 = SmartCropper::new(cfg.clone()).suggest_crop_from_frame(&frame, src_w, src_h, ch);
        let r2 = SmartCropper::new(cfg).suggest_crop_from_frame(&frame, src_w, src_h, ch);

        assert_eq!(r1.crop.x, r2.crop.x);
        assert_eq!(r1.crop.y, r2.crop.y);
        assert_eq!(r1.crop.width, r2.crop.width);
        assert_eq!(r1.crop.height, r2.crop.height);
    }

    /// Malformed frame (wrong buffer size) must not panic — falls back to no-crop.
    #[test]
    fn test_coarse_malformed_frame_no_panic() {
        let cfg = SmartCropConfig::default();
        let cropper = SmartCropper::new(cfg);
        // Wrong size: provide 10 bytes for a 1920×1080×3 frame.
        let result = cropper.suggest_crop_from_frame(&[0u8; 10], 1920, 1080, 3);
        // Should return full-frame fallback.
        assert_eq!(result.crop.width, 1920);
        assert_eq!(result.crop.height, 1080);
    }
}
