//! Focus peaking implementation.

use crate::{MonitorError, MonitorResult};
use serde::{Deserialize, Serialize};

/// Focus peaking color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeakingColor {
    /// Red peaking.
    Red,

    /// Green peaking.
    Green,

    /// Blue peaking.
    Blue,

    /// Yellow peaking.
    Yellow,

    /// Magenta peaking.
    Magenta,

    /// Cyan peaking.
    Cyan,

    /// White peaking.
    White,
}

/// Focus peaking metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FocusMetrics {
    /// Edge strength (0.0-1.0).
    pub edge_strength: f32,

    /// Focus score (0.0-1.0).
    pub focus_score: f32,

    /// Number of edge pixels.
    pub edge_pixel_count: u64,
}

/// Focus assist/peaking.
pub struct FocusAssist {
    threshold: f32,
    color: PeakingColor,
    metrics: FocusMetrics,
}

impl FocusAssist {
    /// Create a new focus assist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: 0.3,
            color: PeakingColor::Red,
            metrics: FocusMetrics::default(),
        }
    }

    /// Set edge detection threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set peaking color.
    pub fn set_color(&mut self, color: PeakingColor) {
        self.color = color;
    }

    /// Apply focus peaking to a frame.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn apply(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
        output: &mut [u8],
    ) -> MonitorResult<()> {
        if frame.len() != output.len() || frame.len() < (width * height * 3) as usize {
            return Err(MonitorError::ProcessingError(
                "Invalid frame dimensions".to_string(),
            ));
        }

        // Copy original frame
        output.copy_from_slice(frame);

        // Calculate edges using Sobel operator
        let edges = self.detect_edges(frame, width, height);

        // Apply peaking color
        let (peak_r, peak_g, peak_b) = self.get_peaking_rgb();

        let mut edge_count = 0u64;
        let mut total_edge_strength = 0.0f32;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = (y * width + x) as usize;
                let edge_strength = edges[idx];

                if edge_strength > self.threshold {
                    let out_idx = ((y * width + x) * 3) as usize;
                    if out_idx + 2 < output.len() {
                        let blend = edge_strength.min(1.0);
                        output[out_idx] = Self::blend_color(output[out_idx], peak_r, blend);
                        output[out_idx + 1] = Self::blend_color(output[out_idx + 1], peak_g, blend);
                        output[out_idx + 2] = Self::blend_color(output[out_idx + 2], peak_b, blend);

                        edge_count += 1;
                        total_edge_strength += edge_strength;
                    }
                }
            }
        }

        let pixel_count = (width * height) as f32;
        self.metrics.edge_pixel_count = edge_count;
        self.metrics.edge_strength = if edge_count > 0 {
            total_edge_strength / edge_count as f32
        } else {
            0.0
        };
        self.metrics.focus_score = (edge_count as f32) / pixel_count;

        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &FocusMetrics {
        &self.metrics
    }

    fn detect_edges(&self, frame: &[u8], width: u32, height: u32) -> Vec<f32> {
        let mut edges = vec![0.0f32; (width * height) as usize];

        // Sobel kernels
        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;

                // Apply Sobel operator
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x as i32 + kx - 1;
                        let py = y as i32 + ky - 1;

                        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                            let idx = ((py as u32 * width + px as u32) * 3) as usize;
                            if idx + 2 < frame.len() {
                                let r = f32::from(frame[idx]);
                                let g = f32::from(frame[idx + 1]);
                                let b = f32::from(frame[idx + 2]);

                                // Convert to grayscale
                                let gray = 0.299 * r + 0.587 * g + 0.114 * b;

                                let k_idx = (ky * 3 + kx) as usize;
                                gx += gray * sobel_x[k_idx];
                                gy += gray * sobel_y[k_idx];
                            }
                        }
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt() / (255.0 * 8.0); // Normalize
                edges[(y * width + x) as usize] = magnitude;
            }
        }

        edges
    }

    fn get_peaking_rgb(&self) -> (u8, u8, u8) {
        match self.color {
            PeakingColor::Red => (255, 0, 0),
            PeakingColor::Green => (0, 255, 0),
            PeakingColor::Blue => (0, 0, 255),
            PeakingColor::Yellow => (255, 255, 0),
            PeakingColor::Magenta => (255, 0, 255),
            PeakingColor::Cyan => (0, 255, 255),
            PeakingColor::White => (255, 255, 255),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn blend_color(original: u8, peak: u8, blend: f32) -> u8 {
        let orig_f = f32::from(original);
        let peak_f = f32::from(peak);
        let result = orig_f * (1.0 - blend) + peak_f * blend;
        result.clamp(0.0, 255.0) as u8
    }
}

impl Default for FocusAssist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_assist() {
        let mut focus = FocusAssist::new();
        focus.set_threshold(0.5);
        focus.set_color(PeakingColor::Green);

        let frame = vec![128u8; 100 * 100 * 3];
        let mut output = vec![0u8; 100 * 100 * 3];

        assert!(focus.apply(&frame, 100, 100, &mut output).is_ok());

        let metrics = focus.metrics();
        assert!(metrics.focus_score >= 0.0);
    }

    #[test]
    fn test_peaking_colors() {
        let mut focus = FocusAssist::new();

        focus.set_color(PeakingColor::Red);
        assert_eq!(focus.color, PeakingColor::Red);

        focus.set_color(PeakingColor::Cyan);
        assert_eq!(focus.color, PeakingColor::Cyan);
    }
}
