//! Advanced text detection using edge density, stroke width analysis, and text classification.
//!
//! This module provides enhanced text detection capabilities beyond basic connected-component
//! analysis, including:
//!
//! - **Edge density analysis**: Measures local edge density to identify text-like regions
//! - **Stroke width transform (SWT)**: Estimates stroke width consistency for text validation
//! - **Text orientation detection**: Classifies text as horizontal, vertical, or rotated
//! - **Scene text vs overlay classification**: Distinguishes natural scene text from overlays
//! - **Confidence scoring**: Multi-factor confidence for each detected region

use crate::common::{Confidence, Rect};
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Orientation of detected text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectedTextOrientation {
    /// Horizontal (left-to-right or right-to-left).
    Horizontal,
    /// Vertical (top-to-bottom or bottom-to-top).
    Vertical,
    /// Rotated by an estimated angle (degrees stored separately).
    Rotated,
    /// Cannot determine orientation.
    Unknown,
}

/// Whether the text is part of the scene or an overlay graphic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextType {
    /// Text burned into the scene (signs, billboards, book covers).
    SceneText,
    /// Overlay / lower-third / subtitle / graphic.
    OverlayText,
    /// Cannot determine.
    Unknown,
}

/// A single detected text region with full metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    /// Bounding box in pixel coordinates.
    pub bbox: Rect,
    /// Detection confidence (0..1).
    pub confidence: Confidence,
    /// Detected orientation.
    pub orientation: DetectedTextOrientation,
    /// Estimated rotation angle in degrees (0 for horizontal/vertical).
    pub rotation_angle_deg: f32,
    /// Scene text vs overlay classification.
    pub text_type: TextType,
    /// Edge density within the bounding box (0..1).
    pub edge_density: f32,
    /// Stroke width consistency score (0..1, higher = more uniform strokes).
    pub stroke_consistency: f32,
    /// Contrast ratio of the region (0..1).
    pub contrast: f32,
}

/// Configuration for the advanced text detector.
#[derive(Debug, Clone)]
pub struct AdvancedTextDetectorConfig {
    /// Minimum confidence to keep a region.
    pub min_confidence: f32,
    /// Sobel edge threshold (0..255).
    pub edge_threshold: u8,
    /// Minimum region area in pixels.
    pub min_area: usize,
    /// Maximum region area as fraction of image area.
    pub max_area_fraction: f32,
    /// Merge nearby regions within this pixel distance.
    pub merge_distance: f32,
    /// Sliding window step (pixels).
    pub window_step: usize,
    /// Minimum edge density for a window to be considered.
    pub min_edge_density: f32,
    /// Maximum edge density (very dense = not text).
    pub max_edge_density: f32,
}

impl Default for AdvancedTextDetectorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.35,
            edge_threshold: 40,
            min_area: 200,
            max_area_fraction: 0.4,
            merge_distance: 15.0,
            window_step: 16,
            min_edge_density: 0.08,
            max_edge_density: 0.70,
        }
    }
}

/// Advanced text detector with stroke width analysis and text classification.
pub struct AdvancedTextDetector {
    config: AdvancedTextDetectorConfig,
}

impl AdvancedTextDetector {
    /// Create a detector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AdvancedTextDetectorConfig::default(),
        }
    }

    /// Create a detector with the given configuration.
    #[must_use]
    pub fn with_config(config: AdvancedTextDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect text regions in an RGB image.
    ///
    /// # Errors
    ///
    /// Returns `SceneError::InvalidDimensions` when `rgb_data.len() != width * height * 3`.
    pub fn detect(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<TextRegion>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(format!(
                "expected {} bytes, got {}",
                width * height * 3,
                rgb_data.len()
            )));
        }
        if width < 4 || height < 4 {
            return Err(SceneError::InvalidDimensions(
                "image must be at least 4x4".into(),
            ));
        }

        let gray = rgb_to_gray(rgb_data);
        let edges = sobel_edges(&gray, width, height);
        let grad_dirs = gradient_directions(&gray, width, height);

        // Sliding-window edge-density scan
        let candidate_rects = self.scan_windows(&edges, width, height);

        // NMS on candidate windows: keep best non-overlapping windows
        let nms_rects = nms_by_density(&candidate_rects, &edges, width, self.config.edge_threshold);

        // For each surviving rect compute properties and filter
        let image_area = (width * height) as f32;
        let mut regions = Vec::new();

        for bbox in &nms_rects {
            let area = bbox.area();
            if area < self.config.min_area as f32 {
                continue;
            }
            if area > image_area * self.config.max_area_fraction {
                continue;
            }

            let edge_density = region_edge_density(&edges, width, bbox, self.config.edge_threshold);
            let stroke_consistency =
                stroke_width_consistency(&edges, &grad_dirs, width, height, bbox);
            let contrast = region_contrast(&gray, width, bbox);
            let (orientation, angle) = detect_orientation(&edges, width, height, bbox);
            let text_type = classify_text_type(&gray, width, height, bbox, contrast);

            let confidence = compute_confidence(edge_density, stroke_consistency, contrast);

            if confidence.value() >= self.config.min_confidence {
                regions.push(TextRegion {
                    bbox: *bbox,
                    confidence,
                    orientation,
                    rotation_angle_deg: angle,
                    text_type,
                    edge_density,
                    stroke_consistency,
                    contrast,
                });
            }
        }

        // Sort descending by confidence
        regions.sort_by(|a, b| {
            b.confidence
                .value()
                .partial_cmp(&a.confidence.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(regions)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Scan the image with sliding windows and collect those with suitable edge density.
    fn scan_windows(&self, edges: &[u8], width: usize, height: usize) -> Vec<Rect> {
        let step = self.config.window_step.max(1);
        let window_sizes: &[(usize, usize)] = &[
            (64, 24),
            (128, 32),
            (96, 48),
            (48, 96),
            (32, 128),
            (160, 40),
            (200, 50),
        ];

        let mut candidates = Vec::new();

        for &(ww, wh) in window_sizes {
            if ww > width || wh > height {
                continue;
            }
            let mut y = 0;
            while y + wh <= height {
                let mut x = 0;
                while x + ww <= width {
                    let density =
                        window_edge_density(edges, width, x, y, ww, wh, self.config.edge_threshold);
                    if density >= self.config.min_edge_density
                        && density <= self.config.max_edge_density
                    {
                        candidates.push(Rect::new(x as f32, y as f32, ww as f32, wh as f32));
                    }
                    x += step;
                }
                y += step;
            }
        }

        candidates
    }
}

impl Default for AdvancedTextDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Convert RGB bytes to grayscale (BT.601 luma).
fn rgb_to_gray(rgb: &[u8]) -> Vec<u8> {
    let pixel_count = rgb.len() / 3;
    let mut gray = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        let off = i * 3;
        let r = rgb[off] as f32;
        let g = rgb[off + 1] as f32;
        let b = rgb[off + 2] as f32;
        gray.push((0.299 * r + 0.587 * g + 0.114 * b) as u8);
    }
    gray
}

/// Compute Sobel edge magnitudes.
fn sobel_edges(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut edges = vec![0u8; width * height];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let tl = gray[idx - width - 1] as i32;
            let tc = gray[idx - width] as i32;
            let tr = gray[idx - width + 1] as i32;
            let ml = gray[idx - 1] as i32;
            let mr = gray[idx + 1] as i32;
            let bl = gray[idx + width - 1] as i32;
            let bc = gray[idx + width] as i32;
            let br = gray[idx + width + 1] as i32;

            let gx = (tr + 2 * mr + br - tl - 2 * ml - bl).abs();
            let gy = (bl + 2 * bc + br - tl - 2 * tc - tr).abs();
            let mag = ((gx * gx + gy * gy) as f32).sqrt() as i32;
            edges[idx] = mag.min(255) as u8;
        }
    }
    edges
}

/// Compute gradient direction at each pixel (radians, 0..PI mapped to 0..255 for compact storage).
fn gradient_directions(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut dirs = vec![0u8; width * height];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let tl = gray[idx - width - 1] as f32;
            let tc = gray[idx - width] as f32;
            let tr = gray[idx - width + 1] as f32;
            let ml = gray[idx - 1] as f32;
            let mr = gray[idx + 1] as f32;
            let bl = gray[idx + width - 1] as f32;
            let bc = gray[idx + width] as f32;
            let br = gray[idx + width + 1] as f32;

            let gx = tr + 2.0 * mr + br - tl - 2.0 * ml - bl;
            let gy = bl + 2.0 * bc + br - tl - 2.0 * tc - tr;
            // atan2 returns -PI..PI; shift to 0..PI then scale to 0..255
            let angle = gy.atan2(gx); // -PI..PI
            let normalized = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI); // 0..1
            dirs[idx] = (normalized * 255.0) as u8;
        }
    }
    dirs
}

/// Edge density within a window.
fn window_edge_density(
    edges: &[u8],
    img_width: usize,
    x0: usize,
    y0: usize,
    ww: usize,
    wh: usize,
    threshold: u8,
) -> f32 {
    let mut count = 0u32;
    let total = (ww * wh) as f32;
    for dy in 0..wh {
        let row_start = (y0 + dy) * img_width + x0;
        for dx in 0..ww {
            if edges[row_start + dx] >= threshold {
                count += 1;
            }
        }
    }
    count as f32 / total
}

/// Edge density within a `Rect`.
fn region_edge_density(edges: &[u8], img_width: usize, bbox: &Rect, threshold: u8) -> f32 {
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let x1 = (bbox.x + bbox.width) as usize;
    let y1 = (bbox.y + bbox.height) as usize;
    let mut count = 0u32;
    let mut total = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = y * img_width + x;
            if idx < edges.len() {
                total += 1;
                if edges[idx] >= threshold {
                    count += 1;
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}

/// Approximate stroke-width consistency via gradient direction variance along edge pixels.
///
/// For genuine text, stroke widths are locally uniform, so the gradient directions along
/// edges should cluster into a few modes. We measure the circular variance of gradient
/// directions for edge pixels in the region.  Lower variance => higher consistency score.
fn stroke_width_consistency(
    edges: &[u8],
    grad_dirs: &[u8],
    img_width: usize,
    _img_height: usize,
    bbox: &Rect,
) -> f32 {
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let x1 = (bbox.x + bbox.width) as usize;
    let y1 = (bbox.y + bbox.height) as usize;

    let mut sin_sum = 0.0f64;
    let mut cos_sum = 0.0f64;
    let mut n = 0u32;

    for y in y0..y1 {
        for x in x0..x1 {
            let idx = y * img_width + x;
            if idx < edges.len() && edges[idx] > 30 {
                let angle = (grad_dirs[idx] as f64 / 255.0) * 2.0 * std::f64::consts::PI;
                // Double the angle so opposite gradients (text stroke) map together
                let doubled = 2.0 * angle;
                sin_sum += doubled.sin();
                cos_sum += doubled.cos();
                n += 1;
            }
        }
    }

    if n < 4 {
        return 0.0;
    }

    let n_f = n as f64;
    let mean_resultant = ((sin_sum / n_f).powi(2) + (cos_sum / n_f).powi(2)).sqrt();
    // mean_resultant is 0..1; high = low variance = consistent strokes
    mean_resultant.min(1.0) as f32
}

/// Contrast within a bounding box (max - min) / 255.
fn region_contrast(gray: &[u8], img_width: usize, bbox: &Rect) -> f32 {
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let x1 = (bbox.x + bbox.width) as usize;
    let y1 = (bbox.y + bbox.height) as usize;

    let mut lo = 255u8;
    let mut hi = 0u8;

    for y in y0..y1 {
        for x in x0..x1 {
            let idx = y * img_width + x;
            if idx < gray.len() {
                lo = lo.min(gray[idx]);
                hi = hi.max(gray[idx]);
            }
        }
    }
    (hi.saturating_sub(lo)) as f32 / 255.0
}

/// Detect text orientation from horizontal vs vertical edge projection profiles.
fn detect_orientation(
    edges: &[u8],
    img_width: usize,
    _img_height: usize,
    bbox: &Rect,
) -> (DetectedTextOrientation, f32) {
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let w = bbox.width as usize;
    let h = bbox.height as usize;

    if w == 0 || h == 0 {
        return (DetectedTextOrientation::Unknown, 0.0);
    }

    // Horizontal projection: sum edge values per row
    let mut h_proj = vec![0u32; h];
    // Vertical projection: sum edge values per column
    let mut v_proj = vec![0u32; w];

    for dy in 0..h {
        for dx in 0..w {
            let idx = (y0 + dy) * img_width + (x0 + dx);
            if idx < edges.len() {
                let val = edges[idx] as u32;
                h_proj[dy] += val;
                v_proj[dx] += val;
            }
        }
    }

    // Compute variance of projections
    let h_var = projection_variance(&h_proj);
    let v_var = projection_variance(&v_proj);

    // For horizontal text, horizontal projection has high variance (text lines)
    // For vertical text, vertical projection has high variance
    let ratio = if v_var > 1e-6 { h_var / v_var } else { 10.0 };

    if ratio > 2.0 {
        (DetectedTextOrientation::Horizontal, 0.0)
    } else if ratio < 0.5 {
        (DetectedTextOrientation::Vertical, 90.0)
    } else {
        // Estimate rotation angle from dominant gradient direction in the region
        let angle = estimate_rotation_angle(edges, img_width, bbox);
        if angle.abs() > 5.0 {
            (DetectedTextOrientation::Rotated, angle)
        } else {
            (DetectedTextOrientation::Horizontal, angle)
        }
    }
}

/// Variance of a 1-D projection.
fn projection_variance(proj: &[u32]) -> f32 {
    if proj.is_empty() {
        return 0.0;
    }
    let n = proj.len() as f32;
    let mean = proj.iter().map(|&v| v as f32).sum::<f32>() / n;
    proj.iter()
        .map(|&v| {
            let d = v as f32 - mean;
            d * d
        })
        .sum::<f32>()
        / n
}

/// Rough rotation angle estimate from edge pixel centroid skew.
fn estimate_rotation_angle(edges: &[u8], img_width: usize, bbox: &Rect) -> f32 {
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let w = bbox.width as usize;
    let h = bbox.height as usize;

    // Split the bbox into left and right halves, measure centroid y for each
    let half_w = w / 2;
    if half_w == 0 || h == 0 {
        return 0.0;
    }

    let mut left_sum_y = 0.0f64;
    let mut left_count = 0u32;
    let mut right_sum_y = 0.0f64;
    let mut right_count = 0u32;

    for dy in 0..h {
        for dx in 0..w {
            let idx = (y0 + dy) * img_width + (x0 + dx);
            if idx < edges.len() && edges[idx] > 40 {
                if dx < half_w {
                    left_sum_y += dy as f64;
                    left_count += 1;
                } else {
                    right_sum_y += dy as f64;
                    right_count += 1;
                }
            }
        }
    }

    if left_count == 0 || right_count == 0 {
        return 0.0;
    }

    let left_centroid = left_sum_y / left_count as f64;
    let right_centroid = right_sum_y / right_count as f64;
    let dy = right_centroid - left_centroid;
    let dx = half_w as f64;
    (dy.atan2(dx) * 180.0 / std::f64::consts::PI) as f32
}

/// Classify text as scene text or overlay text.
///
/// Heuristics:
/// - Overlay text tends to be near the bottom or top of the frame
/// - Overlay text typically has high, uniform contrast (solid bg)
/// - Scene text has variable contrast and arbitrary position
fn classify_text_type(
    gray: &[u8],
    img_width: usize,
    img_height: usize,
    bbox: &Rect,
    contrast: f32,
) -> TextType {
    // Vertical position factor: overlay text is often in bottom 25% or top 15%
    let center_y = bbox.y + bbox.height / 2.0;
    let relative_y = center_y / img_height as f32;
    let near_edge = relative_y < 0.15 || relative_y > 0.75;

    // Width factor: overlays tend to span a large fraction of the frame width
    let width_fraction = bbox.width / img_width as f32;
    let wide = width_fraction > 0.3;

    // Contrast uniformity: measure std-dev of pixel values in the region
    let uniformity = region_intensity_uniformity(gray, img_width, bbox);

    // Score overlay likelihood
    let mut overlay_score = 0.0f32;
    if near_edge {
        overlay_score += 0.35;
    }
    if wide {
        overlay_score += 0.25;
    }
    if contrast > 0.6 {
        overlay_score += 0.2;
    }
    if uniformity > 0.7 {
        overlay_score += 0.2;
    }

    if overlay_score > 0.55 {
        TextType::OverlayText
    } else if overlay_score < 0.3 {
        TextType::SceneText
    } else {
        TextType::Unknown
    }
}

/// Intensity uniformity: 1 - (stddev / 128).  Higher = more uniform.
fn region_intensity_uniformity(gray: &[u8], img_width: usize, bbox: &Rect) -> f32 {
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let x1 = (bbox.x + bbox.width) as usize;
    let y1 = (bbox.y + bbox.height) as usize;

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n = 0u32;

    for y in y0..y1 {
        for x in x0..x1 {
            let idx = y * img_width + x;
            if idx < gray.len() {
                let v = gray[idx] as f64;
                sum += v;
                sum_sq += v * v;
                n += 1;
            }
        }
    }

    if n < 2 {
        return 0.0;
    }

    let mean = sum / n as f64;
    let variance = (sum_sq / n as f64) - mean * mean;
    let stddev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    (1.0 - (stddev / 128.0)).max(0.0).min(1.0) as f32
}

/// Non-maximum suppression by edge density: sort windows by density descending,
/// suppress any window that overlaps a better-scoring window (IoU > 0.5).
fn nms_by_density(rects: &[Rect], edges: &[u8], img_width: usize, threshold: u8) -> Vec<Rect> {
    if rects.is_empty() {
        return Vec::new();
    }

    // Score each rect by its edge density
    let mut scored: Vec<(f32, usize)> = rects
        .iter()
        .enumerate()
        .map(|(i, r)| (region_edge_density(edges, img_width, r, threshold), i))
        .collect();

    // Sort descending by density
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept = Vec::new();
    let mut suppressed = vec![false; rects.len()];

    for &(_, idx) in &scored {
        if suppressed[idx] {
            continue;
        }
        kept.push(rects[idx]);
        // Suppress all rects that significantly overlap with this one
        for &(_, jdx) in &scored {
            if !suppressed[jdx] && jdx != idx && rects[idx].iou(&rects[jdx]) > 0.3 {
                suppressed[jdx] = true;
            }
        }
        suppressed[idx] = true;
    }

    kept
}

/// Merge a set of rectangles that actually overlap (IoU > 0 or contained).
fn merge_overlapping_rects(rects: &[Rect]) -> Vec<Rect> {
    if rects.is_empty() {
        return Vec::new();
    }
    let mut used = vec![false; rects.len()];
    let mut merged = Vec::new();

    for i in 0..rects.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut cur = rects[i];

        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..rects.len() {
                if used[j] {
                    continue;
                }
                // Check actual overlap (not just proximity)
                if rects_overlap(&cur, &rects[j]) {
                    cur = union_rect(&cur, &rects[j]);
                    used[j] = true;
                    changed = true;
                }
            }
        }
        merged.push(cur);
    }
    merged
}

/// Check if two rects overlap (intersection area > 0).
fn rects_overlap(a: &Rect, b: &Rect) -> bool {
    a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y
}

/// Merge a set of rectangles that are within `distance` of each other.
fn merge_rects(rects: &[Rect], distance: f32) -> Vec<Rect> {
    if rects.is_empty() {
        return Vec::new();
    }

    let mut used = vec![false; rects.len()];
    let mut merged = Vec::new();

    for i in 0..rects.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut cur = rects[i];

        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..rects.len() {
                if used[j] {
                    continue;
                }
                if rects_close(&cur, &rects[j], distance) {
                    cur = union_rect(&cur, &rects[j]);
                    used[j] = true;
                    changed = true;
                }
            }
        }
        merged.push(cur);
    }
    merged
}

/// Check if two rects are within `d` pixels of each other.
fn rects_close(a: &Rect, b: &Rect, d: f32) -> bool {
    let dx = if a.x + a.width < b.x {
        b.x - (a.x + a.width)
    } else if b.x + b.width < a.x {
        a.x - (b.x + b.width)
    } else {
        0.0
    };
    let dy = if a.y + a.height < b.y {
        b.y - (a.y + a.height)
    } else if b.y + b.height < a.y {
        a.y - (b.y + b.height)
    } else {
        0.0
    };
    dx <= d && dy <= d
}

/// Union of two rectangles.
fn union_rect(a: &Rect, b: &Rect) -> Rect {
    let min_x = a.x.min(b.x);
    let min_y = a.y.min(b.y);
    let max_x = (a.x + a.width).max(b.x + b.width);
    let max_y = (a.y + a.height).max(b.y + b.height);
    Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
}

/// Compute overall confidence from edge density, stroke consistency, and contrast.
fn compute_confidence(edge_density: f32, stroke_consistency: f32, contrast: f32) -> Confidence {
    // Weighted combination
    let score =
        0.30 * edge_density_score(edge_density) + 0.35 * stroke_consistency + 0.35 * contrast;
    Confidence::new(score)
}

/// Map edge density to a 0..1 quality score.
///
/// Text typically has edge density in the range 0.10 -- 0.70.
/// We use a plateau with soft rolloff at the edges.
fn edge_density_score(density: f32) -> f32 {
    if density < 0.05 {
        // Very low density -- unlikely text
        density / 0.05
    } else if density <= 0.70 {
        // Good text range
        1.0
    } else {
        // Too dense -- diminish but don't zero out
        let excess = density - 0.70;
        (1.0 - excess * 2.0).max(0.1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a solid-color image.
    fn solid_image(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..(width * height) {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    /// Helper: draw a high-contrast horizontal bar (simulating text overlay).
    fn draw_text_bar(
        data: &mut [u8],
        width: usize,
        x0: usize,
        y0: usize,
        bar_w: usize,
        bar_h: usize,
    ) {
        for dy in 0..bar_h {
            for dx in 0..bar_w {
                let px = x0 + dx;
                let py = y0 + dy;
                let idx = (py * width + px) * 3;
                // Alternating black/white vertical stripes to simulate text edges
                if dx % 4 < 2 {
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                }
            }
        }
    }

    #[test]
    fn test_create_default() {
        let det = AdvancedTextDetector::new();
        assert!((det.config.min_confidence - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn test_create_with_config() {
        let cfg = AdvancedTextDetectorConfig {
            min_confidence: 0.5,
            ..Default::default()
        };
        let det = AdvancedTextDetector::with_config(cfg);
        assert!((det.config.min_confidence - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_invalid_dimensions() {
        let det = AdvancedTextDetector::new();
        let result = det.detect(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_small_image() {
        let det = AdvancedTextDetector::new();
        let result = det.detect(&[0u8; 9], 3, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_solid_image_no_text() {
        let det = AdvancedTextDetector::new();
        let data = solid_image(100, 100, 128, 128, 128);
        let regions = det.detect(&data, 100, 100).expect("should succeed");
        assert!(
            regions.is_empty(),
            "solid image should yield no text regions"
        );
    }

    #[test]
    fn test_detect_text_bar() {
        let width = 320;
        let height = 240;
        let mut data = solid_image(width, height, 64, 64, 64);
        // Draw text-like patterns: alternating 3px black/white vertical stripes
        for dy in 0..30 {
            for dx in 0..280 {
                let px = 20 + dx;
                let py = 100 + dy;
                let idx = (py * width + px) * 3;
                if (dx / 3) % 2 == 0 {
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                } else {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                }
            }
        }

        // Verify that edges are actually created in the stripe region
        let gray = rgb_to_gray(&data);
        let edges = sobel_edges(&gray, width, height);
        // Check edge values in the stripe region
        let sample_edge = edges[105 * width + 25];
        assert!(
            sample_edge > 0,
            "edges should be non-zero in the stripe region (got {sample_edge})"
        );

        // Count edge pixels in the text bar area
        let mut edge_count = 0;
        let total = 280 * 30;
        for dy in 0..30 {
            for dx in 0..280 {
                let idx = (100 + dy) * width + (20 + dx);
                if edges[idx] > 15 {
                    edge_count += 1;
                }
            }
        }
        let actual_density = edge_count as f32 / total as f32;

        let cfg = AdvancedTextDetectorConfig {
            min_confidence: 0.01,
            edge_threshold: 15,
            min_edge_density: 0.10,
            max_edge_density: 0.95,
            min_area: 50,
            max_area_fraction: 0.8,
            window_step: 32,
            ..Default::default()
        };
        let det = AdvancedTextDetector::with_config(cfg);
        let regions = det.detect(&data, width, height).expect("should succeed");
        assert!(
            !regions.is_empty(),
            "should detect text regions (edge density in bar: {actual_density:.3}, edge_count: {edge_count})"
        );
    }

    #[test]
    fn test_rgb_to_gray_length() {
        let rgb = vec![100u8; 30 * 3];
        let gray = rgb_to_gray(&rgb);
        assert_eq!(gray.len(), 30);
    }

    #[test]
    fn test_rgb_to_gray_white() {
        let rgb = vec![255u8; 3];
        let gray = rgb_to_gray(&rgb);
        // 0.299*255 + 0.587*255 + 0.114*255 = 254.745 => 254
        assert!(gray[0] >= 254);
    }

    #[test]
    fn test_sobel_edges_solid() {
        let gray = vec![128u8; 50 * 50];
        let edges = sobel_edges(&gray, 50, 50);
        assert_eq!(edges.len(), 2500);
        // All interior pixels should have zero edge
        assert_eq!(edges[51 + 50], 0);
    }

    #[test]
    fn test_sobel_edges_gradient() {
        let mut gray = vec![0u8; 20 * 20];
        for y in 0..20 {
            for x in 0..20 {
                gray[y * 20 + x] = (x * 12).min(255) as u8;
            }
        }
        let edges = sobel_edges(&gray, 20, 20);
        // Interior pixels should have non-zero horizontal gradient
        assert!(edges[5 * 20 + 5] > 0);
    }

    #[test]
    fn test_gradient_directions_length() {
        let gray = vec![100u8; 30 * 30];
        let dirs = gradient_directions(&gray, 30, 30);
        assert_eq!(dirs.len(), 900);
    }

    #[test]
    fn test_window_edge_density_empty() {
        let edges = vec![0u8; 100];
        let density = window_edge_density(&edges, 10, 0, 0, 10, 10, 40);
        assert!((density - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_window_edge_density_full() {
        let edges = vec![255u8; 100];
        let density = window_edge_density(&edges, 10, 0, 0, 10, 10, 40);
        assert!((density - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_region_contrast_uniform() {
        let gray = vec![128u8; 100];
        let bbox = Rect::new(0.0, 0.0, 10.0, 10.0);
        let c = region_contrast(&gray, 10, &bbox);
        assert!((c - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_region_contrast_full_range() {
        let mut gray = vec![0u8; 100];
        gray[0] = 0;
        gray[1] = 255;
        let bbox = Rect::new(0.0, 0.0, 10.0, 1.0);
        let c = region_contrast(&gray, 10, &bbox);
        assert!((c - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stroke_consistency_empty() {
        let edges = vec![0u8; 100];
        let dirs = vec![0u8; 100];
        let bbox = Rect::new(0.0, 0.0, 10.0, 10.0);
        let sc = stroke_width_consistency(&edges, &dirs, 10, 10, &bbox);
        assert!((sc - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_orientation_horizontal() {
        // Create edges with strong horizontal lines
        let w = 100;
        let h = 50;
        let mut edges = vec![0u8; w * h];
        for y in [10, 20, 30] {
            for x in 5..95 {
                edges[y * w + x] = 200;
            }
        }
        let bbox = Rect::new(0.0, 0.0, w as f32, h as f32);
        let (orient, _angle) = detect_orientation(&edges, w, h, &bbox);
        assert_eq!(orient, DetectedTextOrientation::Horizontal);
    }

    #[test]
    fn test_detect_orientation_vertical() {
        let w = 50;
        let h = 100;
        let mut edges = vec![0u8; w * h];
        for x in [10, 20, 30] {
            for y in 5..95 {
                edges[y * w + x] = 200;
            }
        }
        let bbox = Rect::new(0.0, 0.0, w as f32, h as f32);
        let (orient, _angle) = detect_orientation(&edges, w, h, &bbox);
        assert_eq!(orient, DetectedTextOrientation::Vertical);
    }

    #[test]
    fn test_classify_overlay_bottom() {
        let w = 320;
        let h = 240;
        let gray = vec![128u8; w * h];
        // Wide bar near bottom
        let bbox = Rect::new(10.0, 200.0, 300.0, 30.0);
        let tt = classify_text_type(&gray, w, h, &bbox, 0.8);
        assert_eq!(tt, TextType::OverlayText);
    }

    #[test]
    fn test_classify_scene_text_center() {
        let w = 640;
        let h = 480;
        let gray = vec![128u8; w * h];
        // Small region in the center
        let bbox = Rect::new(280.0, 220.0, 40.0, 20.0);
        let tt = classify_text_type(&gray, w, h, &bbox, 0.3);
        assert_eq!(tt, TextType::SceneText);
    }

    #[test]
    fn test_merge_rects_disjoint() {
        let rects = vec![
            Rect::new(0.0, 0.0, 10.0, 10.0),
            Rect::new(100.0, 100.0, 10.0, 10.0),
        ];
        let merged = merge_rects(&rects, 5.0);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_rects_close() {
        let rects = vec![
            Rect::new(0.0, 0.0, 10.0, 10.0),
            Rect::new(12.0, 0.0, 10.0, 10.0),
        ];
        let merged = merge_rects(&rects, 5.0);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].width - 22.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_merge_rects_empty() {
        let merged = merge_rects(&[], 5.0);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_edge_density_score_good_range() {
        // Any density in 0.10..0.70 should score 1.0
        assert!((edge_density_score(0.20) - 1.0).abs() < 0.01);
        assert!((edge_density_score(0.50) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_edge_density_score_low() {
        let score = edge_density_score(0.02);
        assert!(score < 0.5);
    }

    #[test]
    fn test_compute_confidence_range() {
        let c = compute_confidence(0.2, 0.8, 0.9);
        assert!(c.value() >= 0.0 && c.value() <= 1.0);
    }

    #[test]
    fn test_region_intensity_uniformity_solid() {
        let gray = vec![128u8; 100];
        let bbox = Rect::new(0.0, 0.0, 10.0, 10.0);
        let u = region_intensity_uniformity(&gray, 10, &bbox);
        assert!((u - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_projection_variance_constant() {
        let proj = vec![10u32; 20];
        let v = projection_variance(&proj);
        assert!((v - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_projection_variance_varying() {
        let proj = vec![0, 100, 0, 100, 0];
        let v = projection_variance(&proj);
        assert!(v > 0.0);
    }

    #[test]
    fn test_rects_close_true() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(13.0, 0.0, 10.0, 10.0);
        assert!(rects_close(&a, &b, 5.0));
    }

    #[test]
    fn test_rects_close_false() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(100.0, 0.0, 10.0, 10.0);
        assert!(!rects_close(&a, &b, 5.0));
    }

    #[test]
    fn test_union_rect() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        let u = union_rect(&a, &b);
        assert!((u.x - 0.0).abs() < f32::EPSILON);
        assert!((u.y - 0.0).abs() < f32::EPSILON);
        assert!((u.width - 15.0).abs() < f32::EPSILON);
        assert!((u.height - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_sorted_descending() {
        let width = 320;
        let height = 240;
        let mut data = solid_image(width, height, 64, 64, 64);
        draw_text_bar(&mut data, width, 20, 100, 200, 15);
        draw_text_bar(&mut data, width, 20, 200, 280, 25);

        let cfg = AdvancedTextDetectorConfig {
            min_confidence: 0.05,
            ..Default::default()
        };
        let det = AdvancedTextDetector::with_config(cfg);
        let regions = det.detect(&data, width, height).expect("ok");
        for w in regions.windows(2) {
            assert!(w[0].confidence.value() >= w[1].confidence.value());
        }
    }

    #[test]
    fn test_text_region_fields() {
        let region = TextRegion {
            bbox: Rect::new(10.0, 20.0, 100.0, 30.0),
            confidence: Confidence::new(0.8),
            orientation: DetectedTextOrientation::Horizontal,
            rotation_angle_deg: 0.0,
            text_type: TextType::OverlayText,
            edge_density: 0.15,
            stroke_consistency: 0.7,
            contrast: 0.9,
        };
        assert_eq!(region.orientation, DetectedTextOrientation::Horizontal);
        assert_eq!(region.text_type, TextType::OverlayText);
        assert!((region.rotation_angle_deg - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimate_rotation_angle_flat() {
        let w = 100;
        let h = 20;
        let mut edges = vec![0u8; w * h];
        // Horizontal line
        for x in 0..w {
            edges[10 * w + x] = 200;
        }
        let bbox = Rect::new(0.0, 0.0, w as f32, h as f32);
        let angle = estimate_rotation_angle(&edges, w, &bbox);
        assert!(angle.abs() < 5.0, "horizontal line should give ~0 angle");
    }
}
