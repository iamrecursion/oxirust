#![allow(dead_code)]
//! Motion vector visualization scope for video analysis.
//!
//! Displays estimated motion vectors between consecutive video frames as an
//! overlay or a separate visualization. Useful for analyzing camera motion,
//! object movement, compression efficiency, and visual flow direction.
//! Uses block-matching to estimate per-block motion vectors.

/// Default block size for motion estimation.
const DEFAULT_BLOCK_SIZE: u32 = 16;
/// Default search range for block matching.
const DEFAULT_SEARCH_RANGE: u32 = 16;

/// A 2D motion vector.
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Horizontal displacement in pixels.
    pub dx: f32,
    /// Vertical displacement in pixels.
    pub dy: f32,
    /// Block position x (top-left of the block).
    pub block_x: u32,
    /// Block position y (top-left of the block).
    pub block_y: u32,
    /// Matching cost (lower = better match).
    pub cost: f64,
}

impl MotionVector {
    /// Computes the magnitude of the motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Computes the direction angle in radians.
    #[must_use]
    pub fn angle(&self) -> f32 {
        self.dy.atan2(self.dx)
    }

    /// Returns true if this is a zero (no motion) vector.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.dx.abs() < f32::EPSILON && self.dy.abs() < f32::EPSILON
    }
}

/// Configuration for motion vector analysis.
#[derive(Debug, Clone)]
pub struct MotionVectorConfig {
    /// Block size for motion estimation (typically 8 or 16).
    pub block_size: u32,
    /// Search range in pixels.
    pub search_range: u32,
    /// Minimum vector magnitude to display.
    pub min_magnitude: f32,
    /// Whether to show vector arrows.
    pub show_arrows: bool,
    /// Whether to color-code vectors by direction.
    pub color_by_direction: bool,
    /// Whether to show a global motion summary.
    pub show_summary: bool,
    /// Subsampling factor (1 = every block, 2 = every other, etc.).
    pub subsample: u32,
}

impl Default for MotionVectorConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            search_range: DEFAULT_SEARCH_RANGE,
            min_magnitude: 0.5,
            show_arrows: true,
            color_by_direction: true,
            show_summary: true,
            subsample: 1,
        }
    }
}

/// Results from a motion vector analysis.
#[derive(Debug, Clone)]
pub struct MotionVectorResult {
    /// All computed motion vectors.
    pub vectors: Vec<MotionVector>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Average motion magnitude.
    pub avg_magnitude: f32,
    /// Maximum motion magnitude.
    pub max_magnitude: f32,
    /// Dominant motion direction in radians.
    pub dominant_direction: f32,
    /// Percentage of blocks with significant motion.
    pub motion_coverage: f32,
    /// Global motion estimate (camera motion).
    pub global_motion: (f32, f32),
}

/// Extracts the luma (Y) channel from an RGB24 frame.
#[allow(clippy::cast_precision_loss)]
fn rgb_to_luma(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut luma = Vec::with_capacity((width * height) as usize);
    for i in 0..(width * height) as usize {
        let idx = i * 3;
        if idx + 2 < frame.len() {
            let r = frame[idx] as f32;
            let g = frame[idx + 1] as f32;
            let b = frame[idx + 2] as f32;
            let y = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
            luma.push(y);
        }
    }
    luma
}

/// Computes the sum of absolute differences (SAD) between two blocks.
#[allow(clippy::cast_precision_loss)]
fn compute_sad(
    ref_frame: &[u8],
    cur_frame: &[u8],
    width: u32,
    ref_x: u32,
    ref_y: u32,
    cur_x: u32,
    cur_y: u32,
    block_size: u32,
) -> f64 {
    let mut sad = 0_u64;
    for by in 0..block_size {
        for bx in 0..block_size {
            let r_idx = ((ref_y + by) * width + (ref_x + bx)) as usize;
            let c_idx = ((cur_y + by) * width + (cur_x + bx)) as usize;
            if r_idx < ref_frame.len() && c_idx < cur_frame.len() {
                let diff = (ref_frame[r_idx] as i32 - cur_frame[c_idx] as i32).unsigned_abs();
                sad += u64::from(diff);
            }
        }
    }
    sad as f64
}

/// Performs block matching to find the motion vector for a single block.
#[allow(clippy::cast_precision_loss)]
fn find_block_motion(
    ref_luma: &[u8],
    cur_luma: &[u8],
    width: u32,
    height: u32,
    block_x: u32,
    block_y: u32,
    block_size: u32,
    search_range: u32,
) -> MotionVector {
    let mut best_dx: i32 = 0;
    let mut best_dy: i32 = 0;
    let mut best_cost = f64::MAX;

    let sr = search_range as i32;

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            let ref_x = block_x as i32 + dx;
            let ref_y = block_y as i32 + dy;

            if ref_x < 0
                || ref_y < 0
                || (ref_x as u32 + block_size) > width
                || (ref_y as u32 + block_size) > height
            {
                continue;
            }

            let sad = compute_sad(
                ref_luma,
                cur_luma,
                width,
                ref_x as u32,
                ref_y as u32,
                block_x,
                block_y,
                block_size,
            );

            if sad < best_cost {
                best_cost = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    MotionVector {
        dx: best_dx as f32,
        dy: best_dy as f32,
        block_x,
        block_y,
        cost: best_cost,
    }
}

/// Analyzes motion vectors between two consecutive RGB24 frames.
///
/// # Arguments
/// * `ref_frame` - Reference (previous) frame in RGB24
/// * `cur_frame` - Current frame in RGB24
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Motion vector analysis configuration
///
/// # Returns
/// Motion vector analysis result.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_motion_vectors(
    ref_frame: &[u8],
    cur_frame: &[u8],
    width: u32,
    height: u32,
    config: &MotionVectorConfig,
) -> MotionVectorResult {
    let expected_len = (width * height * 3) as usize;
    if ref_frame.len() < expected_len || cur_frame.len() < expected_len || width == 0 || height == 0
    {
        return MotionVectorResult {
            vectors: Vec::new(),
            width,
            height,
            avg_magnitude: 0.0,
            max_magnitude: 0.0,
            dominant_direction: 0.0,
            motion_coverage: 0.0,
            global_motion: (0.0, 0.0),
        };
    }

    let ref_luma = rgb_to_luma(ref_frame, width, height);
    let cur_luma = rgb_to_luma(cur_frame, width, height);

    let block_size = config.block_size.max(4);
    let subsample = config.subsample.max(1);
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    let mut vectors = Vec::new();

    for by in (0..blocks_y).step_by(subsample as usize) {
        for bx in (0..blocks_x).step_by(subsample as usize) {
            let px = bx * block_size;
            let py = by * block_size;

            let mv = find_block_motion(
                &ref_luma,
                &cur_luma,
                width,
                height,
                px,
                py,
                block_size,
                config.search_range,
            );

            vectors.push(mv);
        }
    }

    // Compute statistics
    let mut sum_mag = 0.0_f32;
    let mut max_mag = 0.0_f32;
    let mut sum_dx = 0.0_f32;
    let mut sum_dy = 0.0_f32;
    let mut motion_count = 0u32;

    for mv in &vectors {
        let mag = mv.magnitude();
        sum_mag += mag;
        if mag > max_mag {
            max_mag = mag;
        }
        if mag >= config.min_magnitude {
            motion_count += 1;
        }
        sum_dx += mv.dx;
        sum_dy += mv.dy;
    }

    let total = vectors.len().max(1) as f32;
    let avg_magnitude = sum_mag / total;
    let motion_coverage = motion_count as f32 / total;
    let global_dx = sum_dx / total;
    let global_dy = sum_dy / total;
    let dominant_direction = global_dy.atan2(global_dx);

    MotionVectorResult {
        vectors,
        width,
        height,
        avg_magnitude,
        max_magnitude: max_mag,
        dominant_direction,
        motion_coverage,
        global_motion: (global_dx, global_dy),
    }
}

/// Maps a motion vector direction to an RGB color for visualization.
///
/// Uses HSV color wheel: 0 degrees (right) = red, 90 (down) = green,
/// 180 (left) = cyan, 270 (up) = magenta.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn direction_to_color(angle_rad: f32, magnitude: f32, max_magnitude: f32) -> (u8, u8, u8) {
    let hue = (angle_rad + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let saturation = 1.0_f32;
    let value = if max_magnitude > 0.0 {
        (magnitude / max_magnitude).clamp(0.1, 1.0)
    } else {
        0.5
    };

    let h = (hue * 6.0) % 6.0;
    let f = h - h.floor();
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * f);
    let t = value * (1.0 - saturation * (1.0 - f));

    let (r, g, b) = match h as u32 {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };

    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Describes the dominant motion pattern.
#[must_use]
pub fn describe_motion(result: &MotionVectorResult) -> &'static str {
    if result.avg_magnitude < 0.5 {
        "Static (no significant motion)"
    } else if result.motion_coverage < 0.3 {
        "Local motion (partial frame)"
    } else {
        let (gx, gy) = result.global_motion;
        let global_mag = (gx * gx + gy * gy).sqrt();
        if global_mag > result.avg_magnitude * 0.5 {
            "Camera motion (global)"
        } else {
            "Complex motion (mixed)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector {
            dx: 3.0,
            dy: 4.0,
            block_x: 0,
            block_y: 0,
            cost: 0.0,
        };
        assert!((mv.magnitude() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_motion_vector_angle() {
        let mv = MotionVector {
            dx: 1.0,
            dy: 0.0,
            block_x: 0,
            block_y: 0,
            cost: 0.0,
        };
        assert!(mv.angle().abs() < 0.01); // 0 radians = right
    }

    #[test]
    fn test_motion_vector_is_zero() {
        let zero = MotionVector {
            dx: 0.0,
            dy: 0.0,
            block_x: 0,
            block_y: 0,
            cost: 0.0,
        };
        assert!(zero.is_zero());

        let non_zero = MotionVector {
            dx: 1.0,
            dy: 0.0,
            block_x: 0,
            block_y: 0,
            cost: 0.0,
        };
        assert!(!non_zero.is_zero());
    }

    #[test]
    fn test_rgb_to_luma() {
        // White pixel
        let frame = vec![255u8, 255, 255];
        let luma = rgb_to_luma(&frame, 1, 1);
        assert_eq!(luma.len(), 1);
        assert!(luma[0] > 250);
    }

    #[test]
    fn test_rgb_to_luma_black() {
        let frame = vec![0u8, 0, 0];
        let luma = rgb_to_luma(&frame, 1, 1);
        assert_eq!(luma[0], 0);
    }

    #[test]
    fn test_compute_sad_identical() {
        let block = vec![128u8; 256]; // 16x16 block
        let sad = compute_sad(&block, &block, 16, 0, 0, 0, 0, 16);
        assert!(sad.abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_sad_different() {
        let ref_block = vec![100u8; 256];
        let cur_block = vec![200u8; 256];
        let sad = compute_sad(&ref_block, &cur_block, 16, 0, 0, 0, 0, 16);
        // Each pixel differs by 100, 16*16=256 pixels
        assert!((sad - 25600.0).abs() < 1.0);
    }

    #[test]
    fn test_analyze_empty_frames() {
        let config = MotionVectorConfig::default();
        let result = analyze_motion_vectors(&[], &[], 0, 0, &config);
        assert!(result.vectors.is_empty());
        assert!(result.avg_magnitude.abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyze_identical_frames() {
        let width = 64_u32;
        let height = 64_u32;
        // Use a varied frame (gradient) so that block matching is meaningful
        let mut frame = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                frame[idx] = (x * 4) as u8; // R gradient
                frame[idx + 1] = (y * 4) as u8; // G gradient
                frame[idx + 2] = 128; // B constant
            }
        }
        let config = MotionVectorConfig {
            block_size: 16,
            search_range: 4,
            ..Default::default()
        };
        let result = analyze_motion_vectors(&frame, &frame, width, height, &config);
        // Identical frames should produce zero vectors
        assert!(result.avg_magnitude < 0.01);
        for mv in &result.vectors {
            assert!(mv.is_zero());
        }
    }

    #[test]
    fn test_direction_to_color() {
        // Right direction (0 radians)
        let (r, g, b) = direction_to_color(0.0, 1.0, 1.0);
        assert!(r > 0 || g > 0 || b > 0); // Some color
    }

    #[test]
    fn test_describe_motion_static() {
        let result = MotionVectorResult {
            vectors: Vec::new(),
            width: 64,
            height: 64,
            avg_magnitude: 0.1,
            max_magnitude: 0.2,
            dominant_direction: 0.0,
            motion_coverage: 0.0,
            global_motion: (0.0, 0.0),
        };
        assert_eq!(describe_motion(&result), "Static (no significant motion)");
    }

    #[test]
    fn test_describe_motion_local() {
        let result = MotionVectorResult {
            vectors: Vec::new(),
            width: 64,
            height: 64,
            avg_magnitude: 5.0,
            max_magnitude: 10.0,
            dominant_direction: 0.0,
            motion_coverage: 0.1,
            global_motion: (0.0, 0.0),
        };
        assert_eq!(describe_motion(&result), "Local motion (partial frame)");
    }

    #[test]
    fn test_config_default() {
        let config = MotionVectorConfig::default();
        assert_eq!(config.block_size, 16);
        assert_eq!(config.search_range, 16);
        assert!(config.show_arrows);
        assert!(config.color_by_direction);
    }

    #[test]
    fn test_analyze_with_subsample() {
        let width = 64_u32;
        let height = 64_u32;
        let frame = vec![128u8; (width * height * 3) as usize];
        let config1 = MotionVectorConfig {
            block_size: 16,
            search_range: 2,
            subsample: 1,
            ..Default::default()
        };
        let config2 = MotionVectorConfig {
            block_size: 16,
            search_range: 2,
            subsample: 2,
            ..Default::default()
        };
        let r1 = analyze_motion_vectors(&frame, &frame, width, height, &config1);
        let r2 = analyze_motion_vectors(&frame, &frame, width, height, &config2);
        // Subsampled should have fewer vectors
        assert!(r2.vectors.len() < r1.vectors.len());
    }
}
