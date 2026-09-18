//! Temporal saliency detection for video using motion-weighted attention maps.
//!
//! Combines spatial saliency with inter-frame motion to produce attention maps
//! that highlight regions of dynamic visual interest (e.g., moving objects).

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Temporal saliency map that blends spatial saliency with motion information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSaliencyMap {
    /// Combined saliency values (0.0-1.0).
    pub data: Vec<f32>,
    /// Motion magnitude map (0.0-1.0).
    pub motion_map: Vec<f32>,
    /// Spatial saliency component (0.0-1.0).
    pub spatial_map: Vec<f32>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
}

/// Configuration for temporal saliency.
#[derive(Debug, Clone)]
pub struct TemporalSaliencyConfig {
    /// Weight of spatial saliency in final blend (0.0-1.0).
    pub spatial_weight: f32,
    /// Weight of motion saliency in final blend (0.0-1.0).
    pub motion_weight: f32,
    /// Temporal decay factor for exponential moving average (0.0-1.0).
    pub temporal_decay: f32,
    /// Block size for motion estimation.
    pub motion_block_size: usize,
    /// Search range for block matching.
    pub search_range: usize,
}

impl Default for TemporalSaliencyConfig {
    fn default() -> Self {
        Self {
            spatial_weight: 0.4,
            motion_weight: 0.6,
            temporal_decay: 0.7,
            motion_block_size: 8,
            search_range: 4,
        }
    }
}

/// Temporal saliency detector for video sequences.
pub struct TemporalSaliencyDetector {
    config: TemporalSaliencyConfig,
    /// Previous grayscale frame for motion computation.
    prev_gray: Option<Vec<f32>>,
    prev_width: usize,
    prev_height: usize,
    /// Accumulated temporal saliency (EMA).
    accumulated: Option<Vec<f32>>,
}

impl TemporalSaliencyDetector {
    /// Create a new temporal saliency detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TemporalSaliencyConfig::default(),
            prev_gray: None,
            prev_width: 0,
            prev_height: 0,
            accumulated: None,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: TemporalSaliencyConfig) -> Self {
        Self {
            config,
            prev_gray: None,
            prev_width: 0,
            prev_height: 0,
            accumulated: None,
        }
    }

    /// Process a frame and return a temporal saliency map.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid.
    pub fn process_frame(
        &mut self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<TemporalSaliencyMap> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let gray = rgb_to_gray(rgb_data);
        let spatial = compute_spatial_saliency(&gray, width, height);
        let motion = self.compute_motion_saliency(&gray, width, height);

        // Blend spatial and motion
        let sw = self.config.spatial_weight;
        let mw = self.config.motion_weight;
        let total_weight = sw + mw;
        let mut combined: Vec<f32> = spatial
            .iter()
            .zip(motion.iter())
            .map(|(s, m)| (s * sw + m * mw) / total_weight)
            .collect();

        // Apply temporal accumulation (EMA)
        if let Some(ref acc) = self.accumulated {
            if acc.len() == combined.len() {
                let decay = self.config.temporal_decay;
                for (c, a) in combined.iter_mut().zip(acc.iter()) {
                    *c = *c * (1.0 - decay) + *a * decay;
                }
            }
        }

        // Normalize to [0, 1]
        let max_val = combined.iter().copied().fold(f32::MIN, f32::max);
        if max_val > 0.0 {
            for v in &mut combined {
                *v /= max_val;
            }
        }

        self.accumulated = Some(combined.clone());
        self.prev_gray = Some(gray);
        self.prev_width = width;
        self.prev_height = height;

        Ok(TemporalSaliencyMap {
            data: combined,
            motion_map: motion,
            spatial_map: spatial,
            width,
            height,
        })
    }

    /// Compute motion saliency from inter-frame difference.
    fn compute_motion_saliency(&self, gray: &[f32], width: usize, height: usize) -> Vec<f32> {
        let Some(ref prev) = self.prev_gray else {
            return vec![0.0; width * height];
        };

        if prev.len() != width * height || self.prev_width != width || self.prev_height != height {
            return vec![0.0; width * height];
        }

        let block = self.config.motion_block_size;
        let mut motion = vec![0.0_f32; width * height];

        // Block-level motion estimation
        let blocks_x = width / block;
        let blocks_y = height / block;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let ox = bx * block;
                let oy = by * block;

                // Compute block-level absolute difference (SAD)
                let mut sad = 0.0_f32;
                let mut count = 0;
                for dy in 0..block {
                    for dx in 0..block {
                        let px = ox + dx;
                        let py = oy + dy;
                        if px < width && py < height {
                            let idx = py * width + px;
                            sad += (gray[idx] - prev[idx]).abs();
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let avg_diff = sad / count as f32;
                    // Spread block motion to pixels
                    for dy in 0..block {
                        for dx in 0..block {
                            let px = ox + dx;
                            let py = oy + dy;
                            if px < width && py < height {
                                motion[py * width + px] = avg_diff;
                            }
                        }
                    }
                }
            }
        }

        // Normalize
        let max_motion = motion.iter().copied().fold(0.0_f32, f32::max);
        if max_motion > 0.0 {
            for m in &mut motion {
                *m /= max_motion;
            }
        }

        motion
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.prev_gray = None;
        self.accumulated = None;
    }

    /// Check if the detector has a previous frame for motion computation.
    #[must_use]
    pub fn has_previous_frame(&self) -> bool {
        self.prev_gray.is_some()
    }
}

impl Default for TemporalSaliencyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert RGB to grayscale.
fn rgb_to_gray(rgb: &[u8]) -> Vec<f32> {
    let mut gray = Vec::with_capacity(rgb.len() / 3);
    for chunk in rgb.chunks_exact(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        gray.push((0.299 * r + 0.587 * g + 0.114 * b) / 255.0);
    }
    gray
}

/// Compute spatial saliency using center-surround at a single scale.
fn compute_spatial_saliency(gray: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut saliency = vec![0.0; width * height];
    let scale = 8;

    if width <= scale * 2 || height <= scale * 2 {
        return saliency;
    }

    for y in scale..height - scale {
        for x in scale..width - scale {
            let idx = y * width + x;
            let center = gray[idx];

            let mut surround_sum = 0.0;
            let mut count = 0;
            for dy in -(scale as i32)..=scale as i32 {
                for dx in -(scale as i32)..=scale as i32 {
                    if dx.abs() < (scale as i32) / 2 && dy.abs() < (scale as i32) / 2 {
                        continue;
                    }
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    surround_sum += gray[ny * width + nx];
                    count += 1;
                }
            }

            if count > 0 {
                saliency[idx] = (center - surround_sum / count as f32).abs();
            }
        }
    }

    // Normalize
    let max_s = saliency.iter().copied().fold(f32::MIN, f32::max);
    if max_s > 0.0 {
        for s in &mut saliency {
            *s /= max_s;
        }
    }

    saliency
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_saliency_single_frame() {
        let mut detector = TemporalSaliencyDetector::new();
        let width = 100;
        let height = 100;
        let rgb_data = vec![128u8; width * height * 3];

        let result = detector.process_frame(&rgb_data, width, height);
        assert!(result.is_ok());
        let map = result.expect("should succeed");
        assert_eq!(map.data.len(), width * height);
        assert_eq!(map.width, width);
        assert_eq!(map.height, height);
        // First frame: motion should be zero
        assert!(map.motion_map.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn test_temporal_saliency_two_frames() {
        let mut detector = TemporalSaliencyDetector::new();
        let width = 100;
        let height = 100;
        let frame1 = vec![100u8; width * height * 3];
        let frame2 = vec![200u8; width * height * 3];

        let _ = detector.process_frame(&frame1, width, height);
        assert!(detector.has_previous_frame());

        let result = detector.process_frame(&frame2, width, height);
        assert!(result.is_ok());
        let map = result.expect("should succeed");
        // With different frames, there should be some motion
        let max_motion = map.motion_map.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_motion > 0.0);
    }

    #[test]
    fn test_temporal_saliency_identical_frames() {
        let mut detector = TemporalSaliencyDetector::new();
        let width = 100;
        let height = 100;
        let frame = vec![128u8; width * height * 3];

        let _ = detector.process_frame(&frame, width, height);
        let result = detector.process_frame(&frame, width, height);
        assert!(result.is_ok());
        let map = result.expect("should succeed");
        // No motion for identical frames
        assert!(map.motion_map.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn test_temporal_saliency_reset() {
        let mut detector = TemporalSaliencyDetector::new();
        let frame = vec![128u8; 100 * 100 * 3];
        let _ = detector.process_frame(&frame, 100, 100);
        assert!(detector.has_previous_frame());
        detector.reset();
        assert!(!detector.has_previous_frame());
    }

    #[test]
    fn test_temporal_saliency_invalid_size() {
        let mut detector = TemporalSaliencyDetector::new();
        let result = detector.process_frame(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_saliency_accumulation() {
        let mut detector = TemporalSaliencyDetector::new();
        let width = 50;
        let height = 50;
        let frame1 = vec![100u8; width * height * 3];
        let mut frame2 = vec![100u8; width * height * 3];
        // Add motion in one region
        for y in 10..20 {
            for x in 10..20 {
                let idx = (y * width + x) * 3;
                frame2[idx] = 255;
                frame2[idx + 1] = 255;
                frame2[idx + 2] = 255;
            }
        }

        let _ = detector.process_frame(&frame1, width, height);
        let result = detector.process_frame(&frame2, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_saliency_custom_config() {
        let config = TemporalSaliencyConfig {
            spatial_weight: 0.5,
            motion_weight: 0.5,
            temporal_decay: 0.5,
            motion_block_size: 4,
            search_range: 2,
        };
        let mut detector = TemporalSaliencyDetector::with_config(config);
        let frame = vec![128u8; 100 * 100 * 3];
        let result = detector.process_frame(&frame, 100, 100);
        assert!(result.is_ok());
    }
}

// ---------------------------------------------------------------------------
// Lightweight free-function API — pixel-level diff + Gaussian smoothing
// ---------------------------------------------------------------------------

/// Configuration for the lightweight `temporal_saliency` free function.
///
/// Uses pixel-level absolute difference (not block matching) and a
/// separable Gaussian kernel for motion smoothing.
#[derive(Debug, Clone)]
pub struct TemporalSaliencyFnConfig {
    /// Blend factor: `saliency = (1 - motion_weight) * spatial + motion_weight * motion`.
    /// Default 0.5.
    pub motion_weight: f32,
    /// Sigma for Gaussian smoothing of the motion map. Default 2.0.
    pub smooth_sigma: f32,
}

impl Default for TemporalSaliencyFnConfig {
    fn default() -> Self {
        Self {
            motion_weight: 0.5,
            smooth_sigma: 2.0,
        }
    }
}

/// Compute a temporally-weighted saliency map from two consecutive grayscale frames.
///
/// * `curr` / `prev` — grayscale frames as `u8` values (row-major, `w × h` elements).
/// * `w`, `h` — image dimensions.
/// * `cfg` — blend and smoothing parameters.
///
/// Returns a `Vec<f32>` of length `w * h` with values in `[0, 1]`.
pub fn temporal_saliency(
    curr: &[u8],
    prev: &[u8],
    w: u32,
    h: u32,
    cfg: &TemporalSaliencyFnConfig,
) -> Vec<f32> {
    let n = (w * h) as usize;

    // --- motion component: normalised absolute difference ---
    let mut motion: Vec<f32> = curr
        .iter()
        .zip(prev.iter())
        .map(|(&c, &p)| (c as f32 - p as f32).abs() / 255.0)
        .collect();

    // Gaussian smooth the motion map (separable 1-D passes)
    gaussian_smooth_inplace(&mut motion, w as usize, h as usize, cfg.smooth_sigma);

    // --- spatial component: single-scale centre-surround ---
    // Convert u8 gray → f32 in [0,1]
    let gray: Vec<f32> = curr.iter().map(|&v| v as f32 / 255.0).collect();
    let spatial = spatial_saliency_gray(&gray, w as usize, h as usize);

    // --- blend ---
    let mw = cfg.motion_weight.clamp(0.0, 1.0);
    let sw = 1.0 - mw;
    let mut out: Vec<f32> = (0..n).map(|i| sw * spatial[i] + mw * motion[i]).collect();

    // Normalise to [0, 1]
    let max_val = out.iter().copied().fold(0.0_f32, f32::max);
    if max_val > 1e-8 {
        for v in &mut out {
            *v /= max_val;
        }
    }

    out
}

/// Rolling temporal saliency accumulator for video sequences.
///
/// Holds the previous frame so successive calls to [`TemporalSaliencyAccumulator::push`] can compute
/// the inter-frame motion component.  The first call returns `None`
/// (no previous frame available).
pub struct TemporalSaliencyAccumulator {
    config: TemporalSaliencyFnConfig,
    prev_frame: Option<Vec<u8>>,
    width: u32,
    height: u32,
}

impl TemporalSaliencyAccumulator {
    /// Create a new accumulator with the given configuration and dimensions.
    #[must_use]
    pub fn new(config: TemporalSaliencyFnConfig, width: u32, height: u32) -> Self {
        Self {
            config,
            prev_frame: None,
            width,
            height,
        }
    }

    /// Push a new grayscale frame (u8, `width × height` bytes).
    ///
    /// Returns `None` on the first call (no previous frame).
    /// Subsequent calls return `Some(Vec<f32>)` — the temporally-weighted
    /// saliency map.
    pub fn push(&mut self, frame: &[u8]) -> Option<Vec<f32>> {
        let prev = self.prev_frame.take();
        // Store a copy of the current frame for the next call
        self.prev_frame = Some(frame.to_vec());

        let prev = prev?;
        Some(temporal_saliency(
            frame,
            &prev,
            self.width,
            self.height,
            &self.config,
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Separable 1-D Gaussian smoothing in-place.
fn gaussian_smooth_inplace(buf: &mut Vec<f32>, w: usize, h: usize, sigma: f32) {
    if sigma < 1e-3 || w == 0 || h == 0 {
        return;
    }
    let kernel = gaussian_kernel_1d(sigma);
    let half = kernel.len() / 2;

    // Horizontal pass
    let mut tmp = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            let mut weight = 0.0_f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = x as i32 + k as i32 - half as i32;
                if sx >= 0 && sx < w as i32 {
                    acc += buf[y * w + sx as usize] * kv;
                    weight += kv;
                }
            }
            tmp[y * w + x] = if weight > 0.0 {
                acc / weight
            } else {
                buf[y * w + x]
            };
        }
    }

    // Vertical pass into buf
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            let mut weight = 0.0_f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sy = y as i32 + k as i32 - half as i32;
                if sy >= 0 && sy < h as i32 {
                    acc += tmp[sy as usize * w + x] * kv;
                    weight += kv;
                }
            }
            buf[y * w + x] = if weight > 0.0 {
                acc / weight
            } else {
                tmp[y * w + x]
            };
        }
    }
}

/// Build a 1-D Gaussian kernel (radius = ceil(3σ), odd length).
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let len = 2 * radius + 1;
    let mut k = Vec::with_capacity(len);
    let s2 = 2.0 * sigma * sigma;
    for i in 0..len {
        let x = i as f32 - radius as f32;
        k.push((-x * x / s2).exp());
    }
    // Normalise so sum == 1
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Single-scale centre-surround spatial saliency on a [0,1] grayscale buffer.
fn spatial_saliency_gray(gray: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut sal = vec![0.0_f32; w * h];
    let scale: usize = 8;
    if w <= scale * 2 || h <= scale * 2 {
        return sal;
    }
    for y in scale..h - scale {
        for x in scale..w - scale {
            let idx = y * w + x;
            let center = gray[idx];
            let mut surround = 0.0_f32;
            let mut cnt = 0u32;
            for dy in -(scale as i32)..=scale as i32 {
                for dx in -(scale as i32)..=scale as i32 {
                    if dx.abs() < (scale as i32) / 2 && dy.abs() < (scale as i32) / 2 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        surround += gray[ny as usize * w + nx as usize];
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                sal[idx] = (center - surround / cnt as f32).abs();
            }
        }
    }
    // Normalise
    let max_s = sal.iter().copied().fold(0.0_f32, f32::max);
    if max_s > 1e-8 {
        for v in &mut sal {
            *v /= max_s;
        }
    }
    sal
}

// ---------------------------------------------------------------------------
// Tests for the free-function API
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fn_api_tests {
    use super::*;

    fn gray_frame(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h) as usize]
    }

    #[test]
    fn test_temporal_saliency_static_scene() {
        // Identical frames → motion component = 0 everywhere.
        // Saliency should be purely spatial (centre-surround).
        let w = 64_u32;
        let h = 64_u32;
        let frame = gray_frame(w, h, 128);
        let cfg = TemporalSaliencyFnConfig::default();
        let sal = temporal_saliency(&frame, &frame, w, h, &cfg);
        assert_eq!(sal.len() as u32, w * h);
        // Motion component is zero; output equals rescaled spatial (also near-zero
        // for a uniform image), so all values should be ≤ the spatial maximum.
        // The key invariant: max ∈ [0, 1].
        let max = sal.iter().copied().fold(0.0_f32, f32::max);
        assert!(max <= 1.0 + 1e-5, "max saliency should be ≤ 1.0, got {max}");
        // For a uniform image the spatial component is also 0 → all zeros.
        let all_zero = sal.iter().all(|&v| v < 1e-5);
        assert!(all_zero, "uniform identical frames → all saliency ≈ 0");
    }

    #[test]
    fn test_temporal_saliency_moving_region() {
        // A region that changes between frames should attract high saliency.
        let w = 64_u32;
        let h = 64_u32;
        let mut curr = gray_frame(w, h, 50);
        let prev = gray_frame(w, h, 50);
        // Bright patch in the top-left of the current frame
        for y in 4..20_u32 {
            for x in 4..20_u32 {
                curr[(y * w + x) as usize] = 240;
            }
        }
        let cfg = TemporalSaliencyFnConfig::default();
        let sal = temporal_saliency(&curr, &prev, w, h, &cfg);
        // The changed patch should have notably higher saliency than a static corner
        let patch_max = (4..20_u32)
            .flat_map(|y| (4..20_u32).map(move |x| (y, x)))
            .map(|(y, x)| sal[(y * w + x) as usize])
            .fold(0.0_f32, f32::max);
        let static_corner = (40..60_u32)
            .flat_map(|y| (40..60_u32).map(move |x| (y, x)))
            .map(|(y, x)| sal[(y * w + x) as usize])
            .fold(f32::MAX, f32::min);
        assert!(
            patch_max > static_corner,
            "moving region (max={patch_max}) should have higher saliency than static corner (min={static_corner})"
        );
    }

    #[test]
    fn test_accumulator_first_frame_none() {
        let cfg = TemporalSaliencyFnConfig::default();
        let mut acc = TemporalSaliencyAccumulator::new(cfg, 32, 32);
        let frame = gray_frame(32, 32, 100);
        let result = acc.push(&frame);
        assert!(result.is_none(), "first push must return None");
    }

    #[test]
    fn test_accumulator_second_frame_some() {
        let cfg = TemporalSaliencyFnConfig::default();
        let mut acc = TemporalSaliencyAccumulator::new(cfg, 32, 32);
        let frame1 = gray_frame(32, 32, 100);
        let frame2 = gray_frame(32, 32, 200);
        let _ = acc.push(&frame1);
        let result = acc.push(&frame2);
        assert!(result.is_some(), "second push must return Some");
        let sal = result.expect("just checked");
        assert_eq!(sal.len(), 32 * 32);
        let max = sal.iter().copied().fold(0.0_f32, f32::max);
        assert!(max <= 1.0 + 1e-5);
    }
}
