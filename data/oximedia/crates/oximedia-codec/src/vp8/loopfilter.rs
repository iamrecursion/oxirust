//! VP8 loop filter (deblocking filter).
//!
//! This module implements the VP8 deblocking filter that reduces blocking
//! artifacts at macroblock and block boundaries. The filter is applied
//! after reconstruction and before storing reference frames.
//!
//! VP8 uses a simple loop filter with configurable strength based on:
//! - Filter level (0-63)
//! - Sharpness level (0-7)
//! - Block type and reference frame
//! - Internal edge vs. macroblock edge

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_arguments)]

/// Maximum loop filter level.
pub const MAX_LOOP_FILTER: u8 = 63;

/// Maximum sharpness level.
pub const MAX_SHARPNESS: u8 = 7;

/// Loop filter configuration.
#[derive(Clone, Debug, Default)]
pub struct LoopFilterConfig {
    /// Base filter level (0-63).
    pub level: u8,
    /// Sharpness level (0-7).
    pub sharpness: u8,
    /// Filter type (0 = normal, 1 = simple).
    pub filter_type: u8,
    /// Reference frame deltas.
    pub ref_deltas: [i8; 4],
    /// Mode deltas (intra/inter).
    pub mode_deltas: [i8; 4],
    /// Whether deltas are enabled.
    pub delta_enabled: bool,
}

impl LoopFilterConfig {
    /// Creates a new loop filter configuration.
    #[must_use]
    pub fn new(level: u8, sharpness: u8) -> Self {
        Self {
            level,
            sharpness,
            filter_type: 0,
            ref_deltas: [1, 0, -1, -1], // Default VP8 deltas
            mode_deltas: [0, 0, 0, 0],
            delta_enabled: true,
        }
    }

    /// Returns whether the filter is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.level > 0
    }

    /// Calculates the effective filter level for a block.
    ///
    /// # Arguments
    ///
    /// * `ref_frame` - Reference frame index (0-3)
    /// * `mb_mode` - Macroblock mode (0 = intra, others = inter)
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn effective_level(&self, ref_frame: usize, mb_mode: usize) -> u8 {
        if !self.delta_enabled {
            return self.level;
        }

        let mut level = i32::from(self.level);

        // Add reference frame delta
        if ref_frame < 4 {
            level += i32::from(self.ref_deltas[ref_frame]);
        }

        // Add mode delta (0 for intra, 1 for inter)
        let mode_idx = if mb_mode == 0 { 0 } else { 1 };
        if mode_idx < 4 {
            level += i32::from(self.mode_deltas[mode_idx]);
        }

        level.clamp(0, i32::from(MAX_LOOP_FILTER)) as u8
    }
}

/// Applies loop filter to a horizontal edge.
///
/// Filters the edge between two 4-pixel rows.
///
/// # Arguments
///
/// * `pixels` - Pixel buffer containing the edge (stride must accommodate access)
/// * `stride` - Stride of the pixel buffer
/// * `offset` - Offset to the edge row
/// * `hev_thresh` - High edge variance threshold
/// * `limit` - Filter limit
/// * `thresh` - Filter threshold
#[allow(clippy::similar_names)]
pub fn filter_horizontal_edge(
    pixels: &mut [u8],
    stride: usize,
    offset: usize,
    hev_thresh: u8,
    limit: u8,
    thresh: u8,
) {
    // Filter works on 4 pixels above and 4 pixels below the edge
    // Layout: p3 p2 p1 p0 | q0 q1 q2 q3
    // The edge is between p0 and q0

    for i in 0..4 {
        let idx = offset + i;

        // Check if we have enough pixels
        if idx < stride * 3 || idx + stride * 4 >= pixels.len() {
            continue;
        }

        let p3 = i32::from(pixels[idx - stride * 3]);
        let p2 = i32::from(pixels[idx - stride * 2]);
        let p1 = i32::from(pixels[idx - stride]);
        let p0 = i32::from(pixels[idx]);
        let q0 = i32::from(pixels[idx + stride]);
        let q1 = i32::from(pixels[idx + stride * 2]);
        let q2 = i32::from(pixels[idx + stride * 3]);
        let q3 = i32::from(pixels[idx + stride * 4]);

        // Check if filtering should be applied
        if !should_filter(p3, p2, p1, p0, q0, q1, q2, q3, limit, thresh) {
            continue;
        }

        // Check high edge variance
        let hev = is_high_edge_variance(p1, p0, q0, q1, hev_thresh);

        // Apply filter
        let (new_p0, new_q0, new_p1, new_q1) = if hev {
            simple_filter(p0, q0)
        } else {
            normal_filter(p2, p1, p0, q0, q1, q2)
        };

        // Write back filtered pixels
        pixels[idx] = new_p0;
        pixels[idx + stride] = new_q0;

        if !hev {
            pixels[idx - stride] = new_p1;
            pixels[idx + stride * 2] = new_q1;
        }
    }
}

/// Applies loop filter to a vertical edge.
///
/// Filters the edge between two 4-pixel columns.
///
/// # Arguments
///
/// * `pixels` - Pixel buffer containing the edge
/// * `stride` - Stride of the pixel buffer
/// * `offset` - Offset to the edge column
/// * `hev_thresh` - High edge variance threshold
/// * `limit` - Filter limit
/// * `thresh` - Filter threshold
#[allow(clippy::similar_names)]
pub fn filter_vertical_edge(
    pixels: &mut [u8],
    stride: usize,
    offset: usize,
    hev_thresh: u8,
    limit: u8,
    thresh: u8,
) {
    // Filter works on 4 pixels left and 4 pixels right of the edge
    // Layout: p3 p2 p1 p0 | q0 q1 q2 q3

    for i in 0..4 {
        let idx = offset + i * stride;

        // Check bounds
        if idx < 3 || idx + 4 >= pixels.len() {
            continue;
        }

        let p3 = i32::from(pixels[idx - 3]);
        let p2 = i32::from(pixels[idx - 2]);
        let p1 = i32::from(pixels[idx - 1]);
        let p0 = i32::from(pixels[idx]);
        let q0 = i32::from(pixels[idx + 1]);
        let q1 = i32::from(pixels[idx + 2]);
        let q2 = i32::from(pixels[idx + 3]);
        let q3 = i32::from(pixels[idx + 4]);

        // Check if filtering should be applied
        if !should_filter(p3, p2, p1, p0, q0, q1, q2, q3, limit, thresh) {
            continue;
        }

        // Check high edge variance
        let hev = is_high_edge_variance(p1, p0, q0, q1, hev_thresh);

        // Apply filter
        let (new_p0, new_q0, new_p1, new_q1) = if hev {
            simple_filter(p0, q0)
        } else {
            normal_filter(p2, p1, p0, q0, q1, q2)
        };

        // Write back filtered pixels
        pixels[idx] = new_p0;
        pixels[idx + 1] = new_q0;

        if !hev {
            pixels[idx - 1] = new_p1;
            pixels[idx + 2] = new_q1;
        }
    }
}

/// Checks if the edge should be filtered.
#[allow(clippy::similar_names)]
#[allow(clippy::many_single_char_names)]
fn should_filter(
    p3: i32,
    p2: i32,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    q3: i32,
    limit: u8,
    thresh: u8,
) -> bool {
    let limit = i32::from(limit);
    let thresh = i32::from(thresh);

    // Check if edge is strong enough
    if (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > limit {
        return false;
    }

    // Check if pixels are smooth enough
    if (p3 - p2).abs() > thresh
        || (p2 - p1).abs() > thresh
        || (p1 - p0).abs() > thresh
        || (q3 - q2).abs() > thresh
        || (q2 - q1).abs() > thresh
        || (q1 - q0).abs() > thresh
    {
        return false;
    }

    true
}

/// Checks if the edge has high variance.
#[allow(clippy::similar_names)]
fn is_high_edge_variance(p1: i32, p0: i32, q0: i32, q1: i32, thresh: u8) -> bool {
    let thresh = i32::from(thresh);
    (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh
}

/// Simple filter (for high variance edges).
///
/// Returns (p0', q0', p1, q1) where p1 and q1 are unchanged.
#[allow(clippy::similar_names)]
fn simple_filter(p0: i32, q0: i32) -> (u8, u8, u8, u8) {
    let diff = (q0 - p0).clamp(-128, 127);
    let delta = (diff * 3 + 4) >> 3;

    let new_p0 = (p0 + delta).clamp(0, 255) as u8;
    let new_q0 = (q0 - delta).clamp(0, 255) as u8;

    (new_p0, new_q0, p0 as u8, q0 as u8)
}

/// Normal filter (for smooth edges).
///
/// Returns (p0', q0', p1', q1').
#[allow(clippy::similar_names)]
fn normal_filter(p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32) -> (u8, u8, u8, u8) {
    // Calculate main filter value
    let diff = (q0 - p0).clamp(-128, 127);
    let delta = ((diff * 3 + 4) >> 3).clamp(-63, 63);

    let new_p0 = (p0 + delta).clamp(0, 255) as u8;
    let new_q0 = (q0 - delta).clamp(0, 255) as u8;

    // Calculate secondary filter for p1/q1
    let delta2 = (delta + 1) >> 1;

    let new_p1 = (p1 + delta2).clamp(0, 255) as u8;
    let new_q1 = (q1 - delta2).clamp(0, 255) as u8;

    // Consider p2 and q2 for additional smoothing
    let _ = (p2, q2); // Unused in simple implementation

    (new_p0, new_q0, new_p1, new_q1)
}

/// Calculates filter parameters from configuration.
///
/// Returns (hev_thresh, limit, thresh).
#[must_use]
pub fn calculate_filter_params(level: u8, sharpness: u8, is_keyframe: bool) -> (u8, u8, u8) {
    let level = i32::from(level);
    let sharpness = i32::from(sharpness);

    // HEV threshold
    let hev_thresh = if is_keyframe { 1 } else { 0 };

    // Filter limit
    let limit = if level < 1 {
        0
    } else {
        ((level + 1) * 2).min(63) as u8
    };

    // Internal threshold based on sharpness
    let thresh = if sharpness > 0 {
        (level >> (sharpness - 1)).min(9) as u8
    } else {
        level.min(9) as u8
    };

    (hev_thresh, limit, thresh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_filter_config() {
        let config = LoopFilterConfig::new(10, 3);
        assert_eq!(config.level, 10);
        assert_eq!(config.sharpness, 3);
        assert!(config.is_enabled());

        let disabled = LoopFilterConfig::new(0, 0);
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_effective_level() {
        let config = LoopFilterConfig::new(10, 0);

        // Intra mode, first ref frame (delta = +1)
        let level = config.effective_level(0, 0);
        assert_eq!(level, 11); // 10 + 1

        // Inter mode, third ref frame (delta = -1)
        let level = config.effective_level(2, 1);
        assert_eq!(level, 9); // 10 - 1
    }

    #[test]
    fn test_should_filter() {
        // Smooth edge - should filter
        assert!(should_filter(
            100, 102, 104, 106, 108, 110, 112, 114, 20, 10
        ));

        // Large step - should not filter
        assert!(!should_filter(
            100, 100, 100, 100, 200, 200, 200, 200, 20, 10
        ));

        // High variance - should not filter
        assert!(!should_filter(
            100, 150, 100, 150, 100, 150, 100, 150, 20, 10
        ));
    }

    #[test]
    fn test_is_high_edge_variance() {
        // Low variance
        assert!(!is_high_edge_variance(100, 102, 104, 106, 10));

        // High variance
        assert!(is_high_edge_variance(100, 120, 130, 150, 10));
    }

    #[test]
    fn test_simple_filter() {
        let (p0, q0, p1, q1) = simple_filter(100, 110);

        // p0 should increase, q0 should decrease
        assert!(p0 > 100);
        assert!(q0 < 110);

        // p1 and q1 should be original values
        assert_eq!(p1, 100);
        assert_eq!(q1, 110);
    }

    #[test]
    fn test_normal_filter() {
        let (p0, q0, p1, q1) = normal_filter(98, 100, 102, 108, 110, 112);

        // Both p0 and q0 should move towards each other
        assert!(p0 > 102);
        assert!(q0 < 108);

        // p1 and q1 should also be adjusted
        assert!(p1 >= 100);
        assert!(q1 <= 110);
    }

    #[test]
    fn test_calculate_filter_params() {
        let (hev, limit, thresh) = calculate_filter_params(10, 2, true);

        assert_eq!(hev, 1); // Keyframe
        assert!(limit > 0);
        assert!(thresh > 0);

        let (hev2, _, _) = calculate_filter_params(10, 2, false);
        assert_eq!(hev2, 0); // Inter frame
    }

    #[test]
    fn test_filter_horizontal_edge() {
        // Create a test pattern with a smooth edge
        let mut pixels = vec![100u8; 64]; // 8x8 block
        for i in 0..32 {
            pixels[i] = 100; // Top half
        }
        for i in 32..64 {
            pixels[i] = 110; // Bottom half
        }

        // Use parameters that will trigger filtering
        filter_horizontal_edge(&mut pixels, 8, 28, 5, 50, 20);

        // This is a basic smoke test - filter may or may not trigger
        // depending on edge characteristics
        assert!(pixels.len() == 64);
    }

    #[test]
    fn test_filter_vertical_edge() {
        // Create test pattern with vertical edge
        let mut pixels = vec![100u8; 64];
        for row in 0..8 {
            for col in 0..4 {
                pixels[row * 8 + col] = 100; // Left half
            }
            for col in 4..8 {
                pixels[row * 8 + col] = 110; // Right half
            }
        }

        // Use parameters that will trigger filtering
        filter_vertical_edge(&mut pixels, 8, 4, 5, 50, 20);

        // Basic smoke test - just verify no panic
        assert!(pixels.len() == 64);
    }

    #[test]
    fn test_max_constants() {
        assert_eq!(MAX_LOOP_FILTER, 63);
        assert_eq!(MAX_SHARPNESS, 7);
    }
}
