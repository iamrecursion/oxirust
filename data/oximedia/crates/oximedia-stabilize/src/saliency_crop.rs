//! Content-aware saliency-based cropping.
//!
//! Provides saliency detection and content-aware crop region computation.
//! Instead of blindly cropping from the center (which may cut important subjects),
//! this module detects salient (visually important) regions using a frequency-domain
//! residual saliency approach and biases the crop window to protect them.

use crate::adaptive_crop::{AdaptiveCropResult, CropRect, FrameTransform};
use crate::error::{StabilizeError, StabilizeResult};
use crate::Frame;
use scirs2_core::ndarray::Array2;

/// Configuration for saliency-based cropping.
#[derive(Debug, Clone)]
pub struct SaliencyCropConfig {
    /// Saliency weight (0.0 = ignore saliency, 1.0 = strong saliency influence).
    pub saliency_weight: f64,
    /// Gaussian blur sigma for saliency map smoothing.
    pub blur_sigma: f64,
    /// Minimum preservation ratio (how much of the frame to keep).
    pub min_preservation: f64,
    /// Temporal smoothing for crop position (0.0-1.0).
    pub temporal_smooth: f64,
    /// Number of top salient regions to protect.
    pub num_protect_regions: usize,
    /// Size of the saliency analysis grid (downsample factor).
    pub analysis_scale: usize,
}

impl Default for SaliencyCropConfig {
    fn default() -> Self {
        Self {
            saliency_weight: 0.7,
            blur_sigma: 3.0,
            min_preservation: 0.8,
            temporal_smooth: 0.8,
            num_protect_regions: 3,
            analysis_scale: 4,
        }
    }
}

/// A salient region detected in a frame.
#[derive(Debug, Clone, Copy)]
pub struct SalientRegion {
    /// Center X coordinate.
    pub cx: f64,
    /// Center Y coordinate.
    pub cy: f64,
    /// Approximate radius (extent).
    pub radius: f64,
    /// Saliency score (0.0-1.0).
    pub score: f64,
}

/// Saliency map computed from a frame.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Per-pixel saliency values (0.0 = non-salient, 1.0 = highly salient).
    pub data: Array2<f64>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

impl SaliencyMap {
    /// Get the saliency-weighted centroid (center of mass of saliency).
    #[must_use]
    pub fn centroid(&self) -> (f64, f64) {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut total = 0.0;

        for y in 0..self.height {
            for x in 0..self.width {
                let s = self.data[[y, x]];
                sum_x += x as f64 * s;
                sum_y += y as f64 * s;
                total += s;
            }
        }

        if total > 1e-10 {
            (sum_x / total, sum_y / total)
        } else {
            (self.width as f64 * 0.5, self.height as f64 * 0.5)
        }
    }

    /// Extract the top N salient regions by flood-filling peaks.
    #[must_use]
    pub fn top_regions(&self, n: usize) -> Vec<SalientRegion> {
        if n == 0 || self.width == 0 || self.height == 0 {
            return Vec::new();
        }

        // Find local maxima on a coarse grid
        let step = 8.max(self.width / 16).max(self.height / 16);
        let mut candidates: Vec<(usize, usize, f64)> = Vec::new();

        for y in (0..self.height).step_by(step) {
            for x in (0..self.width).step_by(step) {
                let val = self.data[[y, x]];
                if val > 0.1 {
                    candidates.push((x, y, val));
                }
            }
        }

        // Sort descending by saliency
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N, suppressing nearby duplicates
        let suppress_dist = (step * 2) as f64;
        let mut regions = Vec::new();

        for (cx, cy, score) in &candidates {
            let too_close = regions.iter().any(|r: &SalientRegion| {
                let dx = r.cx - *cx as f64;
                let dy = r.cy - *cy as f64;
                (dx * dx + dy * dy).sqrt() < suppress_dist
            });
            if too_close {
                continue;
            }
            regions.push(SalientRegion {
                cx: *cx as f64,
                cy: *cy as f64,
                radius: step as f64,
                score: *score,
            });
            if regions.len() >= n {
                break;
            }
        }

        regions
    }

    /// Maximum saliency value.
    #[must_use]
    pub fn max_saliency(&self) -> f64 {
        self.data.iter().copied().fold(0.0_f64, f64::max)
    }
}

/// Saliency detector using spectral residual approach.
#[derive(Debug)]
pub struct SaliencyDetector {
    config: SaliencyCropConfig,
}

impl SaliencyDetector {
    /// Create with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SaliencyCropConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: SaliencyCropConfig) -> Self {
        Self { config }
    }

    /// Compute a saliency map for a frame.
    ///
    /// Uses a simplified spectral residual saliency approach:
    /// 1. Downsample the image
    /// 2. Compute local contrast (Itti-Koch inspired center-surround)
    /// 3. Normalize and smooth
    #[must_use]
    pub fn compute_saliency(&self, frame: &Frame) -> SaliencyMap {
        let scale = self.config.analysis_scale.max(1);
        let sw = (frame.width / scale).max(1);
        let sh = (frame.height / scale).max(1);

        // Downsample
        let small = downsample_frame(&frame.data, sw, sh);

        // Compute center-surround contrast at multiple scales
        let saliency = self.center_surround_saliency(&small);

        // Smooth
        let smoothed = self.gaussian_smooth(&saliency, self.config.blur_sigma);

        // Normalize to [0, 1]
        let max_val = smoothed.iter().copied().fold(0.0_f64, f64::max);
        let normalized = if max_val > 1e-10 {
            smoothed.mapv(|v| v / max_val)
        } else {
            smoothed
        };

        // Upsample back to original resolution
        let full = upsample_map(&normalized, frame.width, frame.height);

        SaliencyMap {
            data: full,
            width: frame.width,
            height: frame.height,
        }
    }

    /// Center-surround saliency computation (Itti-Koch inspired).
    fn center_surround_saliency(&self, image: &Array2<f64>) -> Array2<f64> {
        let (h, w) = image.dim();
        let mut saliency = Array2::zeros((h, w));

        // Compute at two scales: small (3x3) and large (7x7) windows
        let radii = [2, 5, 9];

        for &radius in &radii {
            let half = radius / 2;
            for y in half..(h.saturating_sub(half)) {
                for x in half..(w.saturating_sub(half)) {
                    let center = image[[y, x]];

                    // Surround mean
                    let mut surround_sum = 0.0;
                    let mut surround_count = 0;
                    for dy in 0..radius {
                        for dx in 0..radius {
                            let sy = y + dy - half;
                            let sx = x + dx - half;
                            if sy < h && sx < w {
                                surround_sum += image[[sy, sx]];
                                surround_count += 1;
                            }
                        }
                    }

                    let surround_mean = if surround_count > 0 {
                        surround_sum / surround_count as f64
                    } else {
                        center
                    };

                    // Center-surround difference
                    let contrast = (center - surround_mean).abs();
                    saliency[[y, x]] += contrast;
                }
            }
        }

        saliency
    }

    /// Simple Gaussian smoothing.
    fn gaussian_smooth(&self, data: &Array2<f64>, sigma: f64) -> Array2<f64> {
        let (h, w) = data.dim();
        let kernel_size = ((sigma * 3.0).ceil() as usize) * 2 + 1;
        let half = kernel_size / 2;

        // Create Gaussian kernel
        let mut kernel = Vec::with_capacity(kernel_size);
        let mut ksum = 0.0;
        for i in 0..kernel_size {
            let x = (i as f64 - half as f64) / sigma;
            let v = (-0.5 * x * x).exp();
            kernel.push(v);
            ksum += v;
        }
        for v in &mut kernel {
            *v /= ksum;
        }

        // Separable: horizontal pass
        let mut temp = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut wsum = 0.0;
                for (k, &kv) in kernel.iter().enumerate() {
                    let sx = x as i32 + k as i32 - half as i32;
                    if sx >= 0 && sx < w as i32 {
                        sum += data[[y, sx as usize]] * kv;
                        wsum += kv;
                    }
                }
                temp[[y, x]] = if wsum > 0.0 { sum / wsum } else { data[[y, x]] };
            }
        }

        // Vertical pass
        let mut result = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut wsum = 0.0;
                for (k, &kv) in kernel.iter().enumerate() {
                    let sy = y as i32 + k as i32 - half as i32;
                    if sy >= 0 && sy < h as i32 {
                        sum += temp[[sy as usize, x]] * kv;
                        wsum += kv;
                    }
                }
                result[[y, x]] = if wsum > 0.0 { sum / wsum } else { temp[[y, x]] };
            }
        }

        result
    }
}

impl Default for SaliencyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Content-aware cropper that uses saliency detection.
#[derive(Debug)]
pub struct SaliencyCropper {
    config: SaliencyCropConfig,
    detector: SaliencyDetector,
    frame_width: f64,
    frame_height: f64,
}

impl SaliencyCropper {
    /// Create a new saliency-aware cropper.
    #[must_use]
    pub fn new(frame_width: f64, frame_height: f64) -> Self {
        let config = SaliencyCropConfig::default();
        let detector = SaliencyDetector::with_config(config.clone());
        Self {
            config,
            detector,
            frame_width,
            frame_height,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(frame_width: f64, frame_height: f64, config: SaliencyCropConfig) -> Self {
        let detector = SaliencyDetector::with_config(config.clone());
        Self {
            config,
            detector,
            frame_width,
            frame_height,
        }
    }

    /// Compute content-aware crop regions for a sequence of frames and transforms.
    ///
    /// # Errors
    ///
    /// Returns an error if frames and transforms have different lengths.
    pub fn compute(
        &self,
        frames: &[Frame],
        transforms: &[FrameTransform],
    ) -> StabilizeResult<AdaptiveCropResult> {
        if frames.len() != transforms.len() {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", frames.len()),
                format!("{}", transforms.len()),
            ));
        }

        if frames.is_empty() {
            return Ok(AdaptiveCropResult {
                crops: Vec::new(),
                global_crop: CropRect::new(0.0, 0.0, self.frame_width, self.frame_height),
                avg_preservation: 1.0,
                min_preservation: 1.0,
            });
        }

        // Compute saliency maps and base crops
        let mut raw_crops = Vec::with_capacity(frames.len());

        for (i, frame) in frames.iter().enumerate() {
            let saliency = self.detector.compute_saliency(frame);
            let base_crop = self.base_crop_for_transform(&transforms[i]);
            let adjusted = self.adjust_crop_for_saliency(&base_crop, &saliency);
            raw_crops.push(adjusted);
        }

        // Temporal smoothing
        let smoothed = self.smooth_crops(&raw_crops);

        let total_area = self.frame_width * self.frame_height;
        let preservations: Vec<f64> = smoothed.iter().map(|c| c.area() / total_area).collect();
        let avg_pres = preservations.iter().sum::<f64>() / preservations.len() as f64;
        let min_pres = preservations.iter().copied().fold(f64::MAX, f64::min);

        let global_crop = self.compute_global(&smoothed);

        Ok(AdaptiveCropResult {
            crops: smoothed,
            global_crop,
            avg_preservation: avg_pres,
            min_preservation: min_pres,
        })
    }

    /// Compute base crop region from a stabilization transform.
    fn base_crop_for_transform(&self, t: &FrameTransform) -> CropRect {
        let margin_x = t.tx.abs() + self.frame_width * (1.0 - t.scale).max(0.0) * 0.5;
        let margin_y = t.ty.abs() + self.frame_height * (1.0 - t.scale).max(0.0) * 0.5;
        let w = (self.frame_width - 2.0 * margin_x).max(1.0);
        let h = (self.frame_height - 2.0 * margin_y).max(1.0);
        CropRect::new(margin_x, margin_y, w, h)
    }

    /// Adjust a crop rectangle to protect salient regions.
    fn adjust_crop_for_saliency(&self, base: &CropRect, saliency: &SaliencyMap) -> CropRect {
        let centroid = saliency.centroid();
        let regions = saliency.top_regions(self.config.num_protect_regions);

        // Compute saliency-weighted target center
        let frame_cx = self.frame_width * 0.5;
        let frame_cy = self.frame_height * 0.5;

        let sal_weight = self.config.saliency_weight;

        // Blend between frame center and saliency centroid
        let target_cx = frame_cx * (1.0 - sal_weight) + centroid.0 * sal_weight;
        let target_cy = frame_cy * (1.0 - sal_weight) + centroid.1 * sal_weight;

        // Move crop center toward saliency centroid
        let (base_cx, base_cy) = base.center();
        let bias_x = (target_cx - base_cx) * sal_weight * 0.3;
        let bias_y = (target_cy - base_cy) * sal_weight * 0.3;

        let mut new_left = (base.left + bias_x).max(0.0);
        let mut new_top = (base.top + bias_y).max(0.0);

        // Ensure crop stays within frame bounds
        if new_left + base.width > self.frame_width {
            new_left = (self.frame_width - base.width).max(0.0);
        }
        if new_top + base.height > self.frame_height {
            new_top = (self.frame_height - base.height).max(0.0);
        }

        // Expand crop if it would exclude an important salient region
        let mut final_width = base.width;
        let mut final_height = base.height;

        for region in &regions {
            if region.score > 0.5 {
                // If region center is outside the crop, try to expand
                if region.cx < new_left || region.cx > new_left + final_width {
                    let expansion = ((region.cx - (new_left + final_width * 0.5)).abs()
                        - final_width * 0.5)
                        .max(0.0);
                    let max_expansion =
                        self.frame_width * (1.0 - self.config.min_preservation) * 0.5;
                    final_width += expansion.min(max_expansion);
                    new_left = (new_left - expansion.min(max_expansion) * 0.5).max(0.0);
                }
                if region.cy < new_top || region.cy > new_top + final_height {
                    let expansion = ((region.cy - (new_top + final_height * 0.5)).abs()
                        - final_height * 0.5)
                        .max(0.0);
                    let max_expansion =
                        self.frame_height * (1.0 - self.config.min_preservation) * 0.5;
                    final_height += expansion.min(max_expansion);
                    new_top = (new_top - expansion.min(max_expansion) * 0.5).max(0.0);
                }
            }
        }

        // Clamp final size
        final_width = final_width.min(self.frame_width - new_left);
        final_height = final_height.min(self.frame_height - new_top);

        CropRect::new(new_left, new_top, final_width, final_height)
    }

    /// Temporally smooth crop regions.
    fn smooth_crops(&self, crops: &[CropRect]) -> Vec<CropRect> {
        if crops.is_empty() {
            return Vec::new();
        }
        let alpha = 1.0 - self.config.temporal_smooth;
        let mut result = Vec::with_capacity(crops.len());
        let mut prev = crops[0];
        result.push(prev);

        for crop in &crops[1..] {
            let smoothed = CropRect::new(
                alpha * crop.left + (1.0 - alpha) * prev.left,
                alpha * crop.top + (1.0 - alpha) * prev.top,
                alpha * crop.width + (1.0 - alpha) * prev.width,
                alpha * crop.height + (1.0 - alpha) * prev.height,
            );
            result.push(smoothed);
            prev = smoothed;
        }
        result
    }

    /// Compute global bounding crop.
    fn compute_global(&self, crops: &[CropRect]) -> CropRect {
        if crops.is_empty() {
            return CropRect::new(0.0, 0.0, self.frame_width, self.frame_height);
        }
        let max_left = crops.iter().map(|c| c.left).fold(0.0_f64, f64::max);
        let max_top = crops.iter().map(|c| c.top).fold(0.0_f64, f64::max);
        let min_right = crops.iter().map(|c| c.right()).fold(f64::MAX, f64::min);
        let min_bottom = crops.iter().map(|c| c.bottom()).fold(f64::MAX, f64::min);
        CropRect::new(
            max_left,
            max_top,
            (min_right - max_left).max(1.0),
            (min_bottom - max_top).max(1.0),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn downsample_frame(img: &Array2<u8>, new_w: usize, new_h: usize) -> Array2<f64> {
    let (h, w) = img.dim();
    let mut out = Array2::zeros((new_h, new_w));
    let sx = w as f64 / new_w as f64;
    let sy = h as f64 / new_h as f64;
    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = ((x as f64 * sx) as usize).min(w.saturating_sub(1));
            let src_y = ((y as f64 * sy) as usize).min(h.saturating_sub(1));
            out[[y, x]] = img[[src_y, src_x]] as f64;
        }
    }
    out
}

fn upsample_map(data: &Array2<f64>, new_w: usize, new_h: usize) -> Array2<f64> {
    let (h, w) = data.dim();
    let mut out = Array2::zeros((new_h, new_w));
    let sx = w as f64 / new_w as f64;
    let sy = h as f64 / new_h as f64;
    for y in 0..new_h {
        for x in 0..new_w {
            let src_xf = x as f64 * sx;
            let src_yf = y as f64 * sy;
            let x0 = (src_xf as usize).min(w.saturating_sub(2));
            let y0 = (src_yf as usize).min(h.saturating_sub(2));
            let fx = src_xf - x0 as f64;
            let fy = src_yf - y0 as f64;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            out[[y, x]] = data[[y0, x0]] * (1.0 - fx) * (1.0 - fy)
                + data[[y0, x1]] * fx * (1.0 - fy)
                + data[[y1, x0]] * (1.0 - fx) * fy
                + data[[y1, x1]] * fx * fy;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_frame(w: usize, h: usize) -> Frame {
        let mut data = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                // Create a bright spot in the upper-left corner (salient region)
                let dist = ((x as f64) * (x as f64) + (y as f64) * (y as f64)).sqrt();
                let val = (255.0 * (-dist * 0.1).exp()) as u8;
                data[[y, x]] = val;
            }
        }
        Frame::new(w, h, 0.0, data)
    }

    fn uniform_frame(w: usize, h: usize, val: u8) -> Frame {
        Frame::new(w, h, 0.0, Array2::from_elem((h, w), val))
    }

    #[test]
    fn test_saliency_config_default() {
        let cfg = SaliencyCropConfig::default();
        assert!(cfg.saliency_weight > 0.0);
        assert!(cfg.saliency_weight <= 1.0);
    }

    #[test]
    fn test_saliency_detector_creation() {
        let d = SaliencyDetector::new();
        assert!(d.config.blur_sigma > 0.0);
    }

    #[test]
    fn test_saliency_map_uniform() {
        let detector = SaliencyDetector::new();
        let frame = uniform_frame(32, 32, 128);
        let map = detector.compute_saliency(&frame);
        assert_eq!(map.width, 32);
        assert_eq!(map.height, 32);
        // Uniform image has no contrast => low saliency
        assert!(map.max_saliency() < 0.5);
    }

    #[test]
    fn test_saliency_map_gradient() {
        let detector = SaliencyDetector::new();
        let frame = gradient_frame(64, 64);
        let map = detector.compute_saliency(&frame);
        assert_eq!(map.width, 64);
        assert_eq!(map.height, 64);
        // Gradient image has contrast => nonzero saliency
        assert!(map.max_saliency() > 0.0);
    }

    #[test]
    fn test_saliency_centroid() {
        let mut map = SaliencyMap {
            data: Array2::zeros((4, 4)),
            width: 4,
            height: 4,
        };
        // Put all saliency in top-left corner
        map.data[[0, 0]] = 1.0;
        let (cx, cy) = map.centroid();
        assert!((cx - 0.0).abs() < 1e-10);
        assert!((cy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_saliency_centroid_uniform() {
        let map = SaliencyMap {
            data: Array2::from_elem((4, 4), 1.0),
            width: 4,
            height: 4,
        };
        let (cx, cy) = map.centroid();
        assert!((cx - 1.5).abs() < 1e-10);
        assert!((cy - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_saliency_centroid_empty() {
        let map = SaliencyMap {
            data: Array2::zeros((4, 4)),
            width: 4,
            height: 4,
        };
        let (cx, cy) = map.centroid();
        assert!((cx - 2.0).abs() < 1e-10);
        assert!((cy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_regions() {
        let mut data = Array2::zeros((32, 32));
        // Fill a broader area so the grid-based sampling hits it
        for y in 0..8 {
            for x in 0..8 {
                data[[y, x]] = 1.0;
            }
        }
        for y in 16..24 {
            for x in 16..24 {
                data[[y, x]] = 0.8;
            }
        }
        let map = SaliencyMap {
            data,
            width: 32,
            height: 32,
        };
        let regions = map.top_regions(5);
        // Should find at least some regions (grid samples at step >= 2 should hit)
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_top_regions_empty() {
        let map = SaliencyMap {
            data: Array2::zeros((4, 4)),
            width: 4,
            height: 4,
        };
        let regions = map.top_regions(0);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_saliency_cropper_creation() {
        let cropper = SaliencyCropper::new(1920.0, 1080.0);
        assert!((cropper.frame_width - 1920.0).abs() < 1e-10);
    }

    #[test]
    fn test_saliency_cropper_empty() {
        let cropper = SaliencyCropper::new(1920.0, 1080.0);
        let result = cropper.compute(&[], &[]);
        assert!(result.is_ok());
        let r = result.expect("should succeed in test");
        assert!(r.crops.is_empty());
    }

    #[test]
    fn test_saliency_cropper_mismatch() {
        let cropper = SaliencyCropper::new(1920.0, 1080.0);
        let frames = vec![uniform_frame(32, 32, 128)];
        let transforms = vec![FrameTransform::identity(), FrameTransform::identity()];
        let result = cropper.compute(&frames, &transforms);
        assert!(result.is_err());
    }

    #[test]
    fn test_saliency_cropper_basic() {
        let cropper = SaliencyCropper::new(64.0, 64.0);
        let frames = vec![
            gradient_frame(64, 64),
            gradient_frame(64, 64),
            gradient_frame(64, 64),
        ];
        let transforms = vec![
            FrameTransform::new(2.0, 1.0, 0.0, 1.0),
            FrameTransform::new(-1.0, 2.0, 0.0, 0.98),
            FrameTransform::new(3.0, -1.0, 0.0, 1.0),
        ];
        let result = cropper.compute(&frames, &transforms);
        assert!(result.is_ok());
        let r = result.expect("should succeed in test");
        assert_eq!(r.crops.len(), 3);
        assert!(r.avg_preservation > 0.0);
    }

    #[test]
    fn test_saliency_cropper_identity_transforms() {
        let cropper = SaliencyCropper::new(64.0, 64.0);
        let frames = vec![uniform_frame(64, 64, 100); 5];
        let transforms = vec![FrameTransform::identity(); 5];
        let result = cropper.compute(&frames, &transforms);
        assert!(result.is_ok());
        let r = result.expect("should succeed in test");
        // Identity transforms => near-full preservation
        assert!(r.avg_preservation > 0.9);
    }

    #[test]
    fn test_saliency_crop_protects_subject() {
        // Frame with bright spot in upper-left
        let frame = gradient_frame(64, 64);
        // Transform that shifts right (would crop left side)
        let transform = FrameTransform::new(10.0, 0.0, 0.0, 1.0);

        let cropper = SaliencyCropper::with_config(
            64.0,
            64.0,
            SaliencyCropConfig {
                saliency_weight: 0.9,
                ..SaliencyCropConfig::default()
            },
        );

        let result = cropper.compute(&[frame], &[transform]);
        assert!(result.is_ok());
        let r = result.expect("should succeed in test");
        let crop = &r.crops[0];

        // The saliency is in the upper-left, so the crop should try
        // to keep the left side. The crop left edge should be < center.
        assert!(crop.left < 32.0);
    }

    #[test]
    fn test_downsample_frame() {
        let img = Array2::from_elem((16, 16), 200u8);
        let down = downsample_frame(&img, 4, 4);
        assert_eq!(down.dim(), (4, 4));
        assert!((down[[0, 0]] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_upsample_map() {
        let data = Array2::from_elem((4, 4), 0.5);
        let up = upsample_map(&data, 8, 8);
        assert_eq!(up.dim(), (8, 8));
        assert!((up[[0, 0]] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_salient_region_fields() {
        let r = SalientRegion {
            cx: 10.0,
            cy: 20.0,
            radius: 5.0,
            score: 0.9,
        };
        assert!((r.cx - 10.0).abs() < 1e-10);
        assert!(r.score > 0.8);
    }
}
