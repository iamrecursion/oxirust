//! Comb detection for interlaced video.
//!
//! This module implements various algorithms for detecting combing artifacts
//! that appear in interlaced video content. Combing occurs when the two fields
//! of an interlaced frame contain motion, creating a "comb teeth" pattern along
//! moving edges.

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

use super::metrics::InterlaceMetrics;

/// Configuration for comb detection.
#[derive(Debug, Clone)]
pub struct CombDetectorConfig {
    /// Minimum difference threshold for detecting combing (0-255).
    pub threshold: u8,
    /// Spatial search radius for comb pattern detection.
    pub spatial_radius: usize,
    /// Minimum comb length (consecutive pixels) to consider valid.
    pub min_comb_length: usize,
    /// Edge detection threshold for focused analysis.
    pub edge_threshold: u8,
    /// Enable FFT-based frequency analysis.
    pub enable_fft: bool,
}

impl CombDetectorConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 10,
            spatial_radius: 3,
            min_comb_length: 4,
            edge_threshold: 30,
            enable_fft: false,
        }
    }

    /// Creates a configuration optimized for sensitivity (detects more combing).
    #[must_use]
    pub const fn sensitive() -> Self {
        Self {
            threshold: 5,
            spatial_radius: 4,
            min_comb_length: 3,
            edge_threshold: 20,
            enable_fft: true,
        }
    }

    /// Creates a configuration optimized for specificity (fewer false positives).
    #[must_use]
    pub const fn specific() -> Self {
        Self {
            threshold: 15,
            spatial_radius: 2,
            min_comb_length: 5,
            edge_threshold: 40,
            enable_fft: false,
        }
    }
}

impl Default for CombDetectorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Comb detector for interlaced video analysis.
pub struct CombDetector {
    config: CombDetectorConfig,
}

impl CombDetector {
    /// Creates a new comb detector with the given configuration.
    #[must_use]
    pub const fn new(config: CombDetectorConfig) -> Self {
        Self { config }
    }

    /// Creates a new comb detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CombDetectorConfig::default())
    }

    /// Detects combing artifacts in a single frame.
    ///
    /// Returns metrics quantifying the amount of combing detected.
    pub fn detect(&self, frame: &VideoFrame) -> CvResult<InterlaceMetrics> {
        if frame.width == 0 || frame.height == 0 {
            return Err(CvError::invalid_dimensions(frame.width, frame.height));
        }

        if frame.planes.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        // Analyze luma plane (most visible combing)
        let luma = &frame.planes[0];
        let width = frame.width as usize;
        let height = frame.height as usize;

        // Calculate various comb metrics
        let comb_score = self.calculate_comb_score(luma, width, height)?;
        let field_diff = self.calculate_field_difference(luma, width, height)?;
        let spatial_comb = self.calculate_spatial_comb(luma, width, height)?;
        let temporal_comb = 0.0; // Requires multiple frames
        let edge_comb = self.calculate_edge_comb(luma, width, height)?;

        Ok(InterlaceMetrics::from_components(
            comb_score,
            field_diff,
            spatial_comb,
            temporal_comb,
            edge_comb,
        ))
    }

    /// Detects combing using temporal information from multiple frames.
    ///
    /// This provides more accurate detection by comparing consecutive frames.
    pub fn detect_temporal(&self, frames: &[VideoFrame]) -> CvResult<InterlaceMetrics> {
        if frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        if frames.len() < 2 {
            // Fall back to single-frame detection
            return self.detect(&frames[0]);
        }

        // Analyze the most recent frame
        let current = &frames[frames.len() - 1];
        let mut metrics = self.detect(current)?;

        // Calculate temporal combing by comparing with previous frame
        let previous = &frames[frames.len() - 2];
        metrics.temporal_comb = self.calculate_temporal_comb(previous, current)?;

        Ok(metrics)
    }

    /// Calculates the basic comb score by detecting line-to-line differences.
    fn calculate_comb_score(
        &self,
        plane: &oximedia_codec::Plane,
        width: usize,
        height: usize,
    ) -> CvResult<f64> {
        if height < 3 {
            return Ok(0.0);
        }

        let mut comb_pixels = 0;
        let mut total_pixels = 0;

        for y in 1..height - 1 {
            let curr_row = plane.row(y);
            let prev_row = plane.row(y - 1);
            let next_row = plane.row(y + 1);

            if curr_row.len() < width || prev_row.len() < width || next_row.len() < width {
                continue;
            }

            for x in 0..width {
                let curr = i32::from(curr_row[x]);
                let prev = i32::from(prev_row[x]);
                let next = i32::from(next_row[x]);

                // Check for comb pattern: current line differs from neighbors
                let diff_prev = (curr - prev).abs();
                let diff_next = (curr - next).abs();

                // Both differences should be significant for combing
                if diff_prev > i32::from(self.config.threshold)
                    && diff_next > i32::from(self.config.threshold)
                {
                    // Check if neighbors are similar (typical of combing)
                    if (prev - next).abs() < i32::from(self.config.threshold) {
                        comb_pixels += 1;
                    }
                }

                total_pixels += 1;
            }
        }

        if total_pixels == 0 {
            return Ok(0.0);
        }

        Ok(comb_pixels as f64 / total_pixels as f64)
    }

    /// Calculates field difference metric.
    ///
    /// Separates even and odd lines (fields) and measures their difference.
    fn calculate_field_difference(
        &self,
        plane: &oximedia_codec::Plane,
        width: usize,
        height: usize,
    ) -> CvResult<f64> {
        if height < 4 {
            return Ok(0.0);
        }

        let mut diff_sum = 0i64;
        let mut count = 0;

        // Compare odd field lines with interpolated even field
        for y in (2..height - 2).step_by(2) {
            let curr_row = plane.row(y);
            let prev_row = plane.row(y - 1);
            let next_row = plane.row(y + 1);

            if curr_row.len() < width || prev_row.len() < width || next_row.len() < width {
                continue;
            }

            for x in 0..width {
                let actual = i32::from(curr_row[x]);
                // Interpolate from neighboring field
                let interpolated = (i32::from(prev_row[x]) + i32::from(next_row[x])) / 2;
                let diff = (actual - interpolated).abs();

                diff_sum += i64::from(diff);
                count += 1;
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        // Normalize to 0.0-1.0 range
        let avg_diff = diff_sum as f64 / count as f64;
        Ok((avg_diff / 255.0).clamp(0.0, 1.0))
    }

    /// Calculates spatial comb metric using local pattern analysis.
    fn calculate_spatial_comb(
        &self,
        plane: &oximedia_codec::Plane,
        width: usize,
        height: usize,
    ) -> CvResult<f64> {
        let radius = self.config.spatial_radius;
        if height < radius * 2 + 1 || width < radius * 2 + 1 {
            return Ok(0.0);
        }

        let mut comb_strength = 0.0;
        let mut count = 0;

        for y in radius..height - radius {
            let row = plane.row(y);
            if row.len() < width {
                continue;
            }

            for x in radius..width - radius {
                let center = f64::from(row[x]);

                // Calculate local variance in vertical direction
                let mut vertical_variance = 0.0;
                for dy in 1..=radius {
                    let up_row = plane.row(y.saturating_sub(dy));
                    let down_row = plane.row((y + dy).min(height - 1));

                    if up_row.len() > x && down_row.len() > x {
                        let up = f64::from(up_row[x]);
                        let down = f64::from(down_row[x]);
                        vertical_variance += (center - up).abs() + (center - down).abs();
                    }
                }

                // Calculate local variance in horizontal direction
                let mut horizontal_variance = 0.0;
                for dx in 1..=radius {
                    let left = f64::from(row[x.saturating_sub(dx)]);
                    let right = f64::from(row[(x + dx).min(width - 1)]);
                    horizontal_variance += (center - left).abs() + (center - right).abs();
                }

                // Combing shows high vertical variance compared to horizontal
                if horizontal_variance > 0.0 {
                    let ratio = vertical_variance / horizontal_variance;
                    if ratio > 1.5 {
                        comb_strength += ratio;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            return Ok(0.0);
        }

        // Normalize the result
        let avg_strength = comb_strength / count as f64;
        Ok((avg_strength / 10.0).clamp(0.0, 1.0))
    }

    /// Calculates edge-focused comb metric.
    ///
    /// Focuses analysis on edges where combing is most visible.
    fn calculate_edge_comb(
        &self,
        plane: &oximedia_codec::Plane,
        width: usize,
        height: usize,
    ) -> CvResult<f64> {
        if height < 3 || width < 3 {
            return Ok(0.0);
        }

        let mut edge_comb_pixels = 0;
        let mut edge_pixels = 0;

        for y in 1..height - 1 {
            let curr_row = plane.row(y);
            let prev_row = plane.row(y - 1);
            let next_row = plane.row(y + 1);

            if curr_row.len() < width || prev_row.len() < width || next_row.len() < width {
                continue;
            }

            for x in 1..width - 1 {
                // Simple edge detection using Sobel-like operator
                let center = i32::from(curr_row[x]);
                let left = i32::from(curr_row[x - 1]);
                let right = i32::from(curr_row[x + 1]);
                let up = i32::from(prev_row[x]);
                let down = i32::from(next_row[x]);

                let edge_strength = ((right - left).abs() + (down - up).abs()) / 2;

                if edge_strength > i32::from(self.config.edge_threshold) {
                    edge_pixels += 1;

                    // Check for combing at this edge
                    let vertical_diff = (center - up).abs() + (center - down).abs();
                    if vertical_diff > i32::from(self.config.threshold) * 2 {
                        edge_comb_pixels += 1;
                    }
                }
            }
        }

        if edge_pixels == 0 {
            return Ok(0.0);
        }

        Ok(edge_comb_pixels as f64 / edge_pixels as f64)
    }

    /// Calculates temporal comb metric by comparing consecutive frames.
    fn calculate_temporal_comb(
        &self,
        prev_frame: &VideoFrame,
        curr_frame: &VideoFrame,
    ) -> CvResult<f64> {
        if prev_frame.width != curr_frame.width || prev_frame.height != curr_frame.height {
            return Ok(0.0);
        }

        if prev_frame.planes.is_empty() || curr_frame.planes.is_empty() {
            return Ok(0.0);
        }

        let width = curr_frame.width as usize;
        let height = curr_frame.height as usize;

        if height < 4 {
            return Ok(0.0);
        }

        let prev_plane = &prev_frame.planes[0];
        let curr_plane = &curr_frame.planes[0];

        let mut comb_pixels = 0;
        let mut motion_pixels = 0;

        // Check odd and even fields separately
        for field in 0..2 {
            for y in (field + 1..height - 1).step_by(2) {
                let curr_row = curr_plane.row(y);
                let prev_row = prev_plane.row(y);

                if curr_row.len() < width || prev_row.len() < width {
                    continue;
                }

                for x in 0..width {
                    let diff = (i32::from(curr_row[x]) - i32::from(prev_row[x])).abs();

                    // Motion detected
                    if diff > i32::from(self.config.threshold) {
                        motion_pixels += 1;

                        // Check if neighboring line (other field) also changed
                        if y > 0 && y < height - 1 {
                            let curr_neighbor = curr_plane.row(y - 1);
                            let prev_neighbor = prev_plane.row(y - 1);

                            if curr_neighbor.len() > x && prev_neighbor.len() > x {
                                let neighbor_diff = (i32::from(curr_neighbor[x])
                                    - i32::from(prev_neighbor[x]))
                                .abs();

                                // If motion is different between fields, likely combing
                                if (diff - neighbor_diff).abs() > i32::from(self.config.threshold) {
                                    comb_pixels += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        if motion_pixels == 0 {
            return Ok(0.0);
        }

        Ok(comb_pixels as f64 / motion_pixels as f64)
    }

    /// Generates a comb map showing where combing was detected.
    ///
    /// Returns a binary map where 255 indicates combing and 0 indicates no combing.
    pub fn generate_comb_map(&self, frame: &VideoFrame) -> CvResult<Vec<u8>> {
        if frame.planes.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        let luma = &frame.planes[0];
        let width = frame.width as usize;
        let height = frame.height as usize;

        let mut comb_map = vec![0u8; width * height];

        if height < 3 {
            return Ok(comb_map);
        }

        for y in 1..height - 1 {
            let curr_row = luma.row(y);
            let prev_row = luma.row(y - 1);
            let next_row = luma.row(y + 1);

            if curr_row.len() < width || prev_row.len() < width || next_row.len() < width {
                continue;
            }

            for x in 0..width {
                let curr = i32::from(curr_row[x]);
                let prev = i32::from(prev_row[x]);
                let next = i32::from(next_row[x]);

                let diff_prev = (curr - prev).abs();
                let diff_next = (curr - next).abs();

                if diff_prev > i32::from(self.config.threshold)
                    && diff_next > i32::from(self.config.threshold)
                    && (prev - next).abs() < i32::from(self.config.threshold)
                {
                    comb_map[y * width + x] = 255;
                }
            }
        }

        Ok(comb_map)
    }

    /// Detects comb patterns with length filtering.
    ///
    /// Only reports combing that extends for at least `min_comb_length` consecutive pixels.
    pub fn detect_comb_patterns(&self, frame: &VideoFrame) -> CvResult<Vec<CombPattern>> {
        let comb_map = self.generate_comb_map(frame)?;
        let width = frame.width as usize;
        let height = frame.height as usize;

        let mut patterns = Vec::new();

        for y in 0..height {
            let mut run_start = None;

            for x in 0..width {
                let is_comb = comb_map[y * width + x] > 0;

                match (is_comb, run_start) {
                    (true, None) => {
                        run_start = Some(x);
                    }
                    (false, Some(start)) => {
                        let length = x - start;
                        if length >= self.config.min_comb_length {
                            patterns.push(CombPattern {
                                y,
                                x_start: start,
                                x_end: x,
                                length,
                            });
                        }
                        run_start = None;
                    }
                    _ => {}
                }
            }

            // Handle pattern at end of row
            if let Some(start) = run_start {
                let length = width - start;
                if length >= self.config.min_comb_length {
                    patterns.push(CombPattern {
                        y,
                        x_start: start,
                        x_end: width,
                        length,
                    });
                }
            }
        }

        Ok(patterns)
    }
}

/// Represents a detected comb pattern in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CombPattern {
    /// Y coordinate (row) of the pattern.
    pub y: usize,
    /// Starting X coordinate.
    pub x_start: usize,
    /// Ending X coordinate.
    pub x_end: usize,
    /// Length of the comb pattern in pixels.
    pub length: usize,
}

impl CombPattern {
    /// Returns the center point of the comb pattern.
    #[must_use]
    pub const fn center(&self) -> (usize, usize) {
        ((self.x_start + self.x_end) / 2, self.y)
    }
}
