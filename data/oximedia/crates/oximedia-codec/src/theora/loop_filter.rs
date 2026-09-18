// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Loop filtering for Theora.
//!
//! Implements deblocking filter to reduce block artifacts at block boundaries.
//! The loop filter is applied after reconstruction to smooth discontinuities
//! caused by block-based encoding.

use crate::error::CodecResult;
use crate::theora::tables::LOOP_FILTER_LIMITS;

/// Loop filter configuration.
#[derive(Debug, Clone, Copy)]
pub struct LoopFilterConfig {
    /// Filter strength (0-63).
    pub strength: u8,
    /// Enable filtering.
    pub enabled: bool,
    /// Sharpness level (0-7).
    pub sharpness: u8,
}

impl Default for LoopFilterConfig {
    fn default() -> Self {
        Self {
            strength: 30,
            enabled: true,
            sharpness: 0,
        }
    }
}

impl LoopFilterConfig {
    /// Create a new loop filter configuration.
    #[must_use]
    pub const fn new(strength: u8) -> Self {
        Self {
            strength,
            enabled: true,
            sharpness: 0,
        }
    }

    /// Get the filter limit based on quality.
    #[must_use]
    pub fn limit(&self) -> u8 {
        let quality = self.strength.min(63);
        LOOP_FILTER_LIMITS[quality as usize]
    }

    /// Get interior filter limit (stronger).
    #[must_use]
    pub fn interior_limit(&self) -> u8 {
        let base = self.limit();
        if self.sharpness > 0 {
            base.saturating_sub(self.sharpness * 2)
        } else {
            base
        }
    }

    /// Get edge filter limit (weaker).
    #[must_use]
    pub fn edge_limit(&self) -> u8 {
        (self.limit() + 4) >> 1
    }
}

/// Loop filter context for a frame.
pub struct LoopFilter {
    /// Configuration.
    config: LoopFilterConfig,
    /// Frame width in pixels.
    width: usize,
    /// Frame height in pixels.
    height: usize,
}

impl LoopFilter {
    /// Create a new loop filter.
    #[must_use]
    pub const fn new(config: LoopFilterConfig, width: usize, height: usize) -> Self {
        Self {
            config,
            width,
            height,
        }
    }

    /// Apply loop filter to a frame.
    ///
    /// # Arguments
    ///
    /// * `y_plane` - Y plane data
    /// * `u_plane` - U plane data
    /// * `v_plane` - V plane data
    /// * `y_stride` - Y plane stride
    /// * `uv_stride` - UV plane stride
    pub fn filter_frame(
        &self,
        y_plane: &mut [u8],
        u_plane: &mut [u8],
        v_plane: &mut [u8],
        y_stride: usize,
        uv_stride: usize,
    ) -> CodecResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Filter Y plane
        self.filter_plane(y_plane, y_stride, self.width, self.height);

        // Filter U and V planes
        let uv_width = self.width / 2;
        let uv_height = self.height / 2;
        self.filter_plane(u_plane, uv_stride, uv_width, uv_height);
        self.filter_plane(v_plane, uv_stride, uv_width, uv_height);

        Ok(())
    }

    /// Filter a single plane.
    fn filter_plane(&self, plane: &mut [u8], stride: usize, width: usize, height: usize) {
        // Vertical edges
        for y in 0..height {
            for x in (8..width).step_by(8) {
                self.filter_vertical_edge(plane, stride, x, y);
            }
        }

        // Horizontal edges
        for y in (8..height).step_by(8) {
            for x in 0..width {
                self.filter_horizontal_edge(plane, stride, x, y);
            }
        }
    }

    /// Filter a vertical edge.
    fn filter_vertical_edge(&self, plane: &mut [u8], stride: usize, x: usize, y: usize) {
        if x < 4 || x >= self.width - 4 {
            return;
        }

        let offset = y * stride + x;
        if offset < 4 || offset + 4 >= plane.len() {
            return;
        }

        // Get pixels around the edge
        let p3 = i16::from(plane[offset - 4]);
        let p2 = i16::from(plane[offset - 3]);
        let p1 = i16::from(plane[offset - 2]);
        let p0 = i16::from(plane[offset - 1]);
        let q0 = i16::from(plane[offset]);
        let q1 = i16::from(plane[offset + 1]);
        let q2 = i16::from(plane[offset + 2]);
        let q3 = i16::from(plane[offset + 3]);

        // Calculate filter values
        let limit = i16::from(self.config.interior_limit());

        // Check if filtering is needed
        if (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > limit {
            return;
        }

        // Apply filter
        let filter_value = ((q0 - p0) * 3 + (p1 - q1)) / 8;
        let filter_value = filter_value.clamp(-limit, limit);

        let new_p0 = (p0 + filter_value).clamp(0, 255) as u8;
        let new_q0 = (q0 - filter_value).clamp(0, 255) as u8;

        plane[offset - 1] = new_p0;
        plane[offset] = new_q0;

        // Stronger filtering for smooth regions
        if (p2 - p0).abs() < (limit >> 1) && (q2 - q0).abs() < (limit >> 1) {
            let p1_filter = ((p2 + p0 + q0) / 3 - p1) / 2;
            let q1_filter = ((q2 + q0 + p0) / 3 - q1) / 2;

            let new_p1 = (p1 + p1_filter).clamp(0, 255) as u8;
            let new_q1 = (q1 + q1_filter).clamp(0, 255) as u8;

            plane[offset - 2] = new_p1;
            plane[offset + 1] = new_q1;
        }
    }

    /// Filter a horizontal edge.
    fn filter_horizontal_edge(&self, plane: &mut [u8], stride: usize, x: usize, y: usize) {
        if y < 4 || y >= self.height - 4 {
            return;
        }

        let offset = y * stride + x;
        if offset < stride * 4 || offset + stride * 4 >= plane.len() {
            return;
        }

        // Get pixels around the edge
        let p3 = i16::from(plane[offset - stride * 4]);
        let p2 = i16::from(plane[offset - stride * 3]);
        let p1 = i16::from(plane[offset - stride * 2]);
        let p0 = i16::from(plane[offset - stride]);
        let q0 = i16::from(plane[offset]);
        let q1 = i16::from(plane[offset + stride]);
        let q2 = i16::from(plane[offset + stride * 2]);
        let q3 = i16::from(plane[offset + stride * 3]);

        // Calculate filter values
        let limit = i16::from(self.config.interior_limit());

        // Check if filtering is needed
        if (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > limit {
            return;
        }

        // Apply filter
        let filter_value = ((q0 - p0) * 3 + (p1 - q1)) / 8;
        let filter_value = filter_value.clamp(-limit, limit);

        let new_p0 = (p0 + filter_value).clamp(0, 255) as u8;
        let new_q0 = (q0 - filter_value).clamp(0, 255) as u8;

        plane[offset - stride] = new_p0;
        plane[offset] = new_q0;

        // Stronger filtering for smooth regions
        if (p2 - p0).abs() < (limit >> 1) && (q2 - q0).abs() < (limit >> 1) {
            let p1_filter = ((p2 + p0 + q0) / 3 - p1) / 2;
            let q1_filter = ((q2 + q0 + p0) / 3 - q1) / 2;

            let new_p1 = (p1 + p1_filter).clamp(0, 255) as u8;
            let new_q1 = (q1 + q1_filter).clamp(0, 255) as u8;

            plane[offset - stride * 2] = new_p1;
            plane[offset + stride] = new_q1;
        }
    }
}

/// Simple loop filter for block boundaries.
///
/// This is a lighter version of the loop filter for use in fast encoding modes.
pub fn simple_loop_filter(plane: &mut [u8], stride: usize, x: usize, y: usize, limit: u8) {
    let offset = y * stride + x;
    if offset < 1 || offset >= plane.len() - 1 {
        return;
    }

    let p1 = i16::from(plane[offset - 1]);
    let p0 = i16::from(plane[offset]);
    let q0 = i16::from(plane[offset + 1]);

    let limit = i16::from(limit);

    if (p0 - q0).abs() > limit {
        return;
    }

    let filter = ((q0 - p0) * 3) / 8;
    let filter = filter.clamp(-limit, limit);

    plane[offset - 1] = (p1 + filter).clamp(0, 255) as u8;
    plane[offset + 1] = (q0 - filter).clamp(0, 255) as u8;
}

/// Adaptive loop filter strength calculation.
///
/// Adjusts filter strength based on local image characteristics.
pub fn adaptive_filter_strength(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    base_strength: u8,
) -> u8 {
    let offset = y * stride + x;
    if offset + stride + 1 >= plane.len() {
        return base_strength;
    }

    // Calculate local variance
    let mut variance = 0u32;
    let center = u32::from(plane[offset]);

    for dy in 0..2 {
        for dx in 0..2 {
            let pixel = u32::from(plane[offset + dy * stride + dx]);
            let diff = (pixel as i32 - center as i32).abs() as u32;
            variance += diff * diff;
        }
    }

    variance /= 4;

    // Adjust strength based on variance
    if variance < 100 {
        // Smooth region: stronger filter
        base_strength.saturating_add(10)
    } else if variance > 1000 {
        // Textured region: weaker filter
        base_strength.saturating_sub(10)
    } else {
        base_strength
    }
}

/// Directional loop filter.
///
/// Filters along edges with directional bias to preserve edges better.
pub struct DirectionalFilter {
    /// Horizontal strength.
    h_strength: u8,
    /// Vertical strength.
    v_strength: u8,
}

impl DirectionalFilter {
    /// Create a new directional filter.
    #[must_use]
    pub const fn new(h_strength: u8, v_strength: u8) -> Self {
        Self {
            h_strength,
            v_strength,
        }
    }

    /// Analyze edge direction at a position.
    pub fn analyze_direction(&self, plane: &[u8], stride: usize, x: usize, y: usize) -> Direction {
        if x < 1 || y < 1 || x >= stride - 1 {
            return Direction::None;
        }

        let offset = y * stride + x;
        if offset + stride + 1 >= plane.len() {
            return Direction::None;
        }

        // Calculate gradients
        let gx = (i16::from(plane[offset + 1]) - i16::from(plane[offset - 1])).abs();
        let gy = (i16::from(plane[offset + stride]) - i16::from(plane[offset - stride])).abs();

        // No significant gradient in either direction
        if gx == 0 && gy == 0 {
            Direction::None
        } else if gx > gy * 2 {
            Direction::Vertical
        } else if gy > gx * 2 {
            Direction::Horizontal
        } else {
            Direction::Both
        }
    }

    /// Apply directional filter.
    pub fn apply(&self, plane: &mut [u8], stride: usize, x: usize, y: usize, direction: Direction) {
        match direction {
            Direction::Horizontal => {
                simple_loop_filter(plane, stride, x, y, self.h_strength);
            }
            Direction::Vertical => {
                // Filter vertical edge by treating columns as rows
                if y > 0 && x < stride - 1 {
                    let offset = y * stride + x;
                    if offset + stride < plane.len() {
                        simple_loop_filter(plane, 1, offset, 0, self.v_strength);
                    }
                }
            }
            Direction::Both => {
                simple_loop_filter(plane, stride, x, y, (self.h_strength + self.v_strength) / 2);
            }
            Direction::None => {}
        }
    }
}

/// Edge direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// No strong direction.
    None,
    /// Horizontal edge.
    Horizontal,
    /// Vertical edge.
    Vertical,
    /// Both directions.
    Both,
}

/// Chroma loop filter with reduced strength.
///
/// Chroma planes typically need less aggressive filtering.
pub fn chroma_loop_filter(
    plane: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    strength: u8,
) {
    let reduced_strength = strength / 2;

    // Filter vertical edges
    for y in 0..height {
        for x in (4..width).step_by(4) {
            simple_loop_filter(plane, stride, x, y, reduced_strength);
        }
    }

    // Filter horizontal edges
    for y in (4..height).step_by(4) {
        for x in 0..width {
            if y * stride + x < plane.len() {
                let offset = y * stride + x;
                if offset >= stride && offset < plane.len() - stride {
                    simple_loop_filter(plane, stride, x, y, reduced_strength);
                }
            }
        }
    }
}

/// Calculate optimal filter strength for a macroblock.
///
/// Analyzes the macroblock content to determine the best filter strength.
pub fn calculate_mb_filter_strength(
    plane: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    base_strength: u8,
) -> u8 {
    let x = mb_x * 16;
    let y = mb_y * 16;

    if y * stride + x + 16 >= plane.len() || x + 16 > stride {
        return base_strength;
    }

    // Calculate macroblock variance
    let mut sum = 0u32;
    let mut sum_sq = 0u32;
    let mut count = 0u32;

    for dy in 0..16 {
        for dx in 0..16 {
            let offset = (y + dy) * stride + x + dx;
            if offset < plane.len() {
                let pixel = u32::from(plane[offset]);
                sum += pixel;
                sum_sq += pixel * pixel;
                count += 1;
            }
        }
    }

    if count == 0 {
        return base_strength;
    }

    let mean = sum / count;
    let variance = (sum_sq / count).saturating_sub(mean * mean);

    // Adjust strength based on variance
    match variance {
        0..=50 => base_strength.saturating_add(15), // Very smooth
        51..=200 => base_strength.saturating_add(10), // Smooth
        201..=500 => base_strength.saturating_add(5), // Moderate
        501..=1000 => base_strength,                // Normal
        1001..=2000 => base_strength.saturating_sub(5), // Textured
        _ => base_strength.saturating_sub(10),      // Very textured
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_filter_config() {
        let config = LoopFilterConfig::new(30);
        assert_eq!(config.strength, 30);
        assert!(config.enabled);
        assert!(config.limit() > 0);
    }

    #[test]
    fn test_simple_loop_filter() {
        // Use a smaller edge difference that falls within the filter limit
        let mut plane = vec![100u8, 100, 100, 108, 108, 108];
        simple_loop_filter(&mut plane, 6, 2, 0, 10);

        // Check that the edge was smoothed (p0-q0 = -8 which is within limit 10)
        assert!(plane[1] != 100 || plane[3] != 108);
    }

    #[test]
    fn test_adaptive_strength() {
        let plane = vec![128u8; 100];
        let strength = adaptive_filter_strength(&plane, 10, 5, 5, 30);
        assert!(strength > 30); // Smooth region should increase strength
    }

    #[test]
    fn test_directional_filter() {
        let filter = DirectionalFilter::new(20, 25);
        let plane = vec![128u8; 100];
        let direction = filter.analyze_direction(&plane, 10, 5, 5);
        assert_eq!(direction, Direction::None); // Uniform region has no direction
    }
}
