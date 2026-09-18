//! Edge-based scene detection.
//!
//! This module provides scene detection based on edge pattern changes.
//! It detects scene boundaries by analyzing how edge structures change
//! between consecutive frames.

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

use super::{ChangeType, SceneChange, SceneConfig, SceneMetadata};

/// Configuration for edge-based detection.
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    /// Threshold for edge detection (0-255).
    pub edge_threshold: u8,
    /// Use Sobel (true) or simple gradient (false).
    pub use_sobel: bool,
    /// Dilation size for edge enhancement.
    pub dilation_size: usize,
    /// Compare edge count (true) or edge patterns (false).
    pub compare_count: bool,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_threshold: 50,
            use_sobel: true,
            dilation_size: 0,
            compare_count: false,
        }
    }
}

/// Edge map representation.
#[derive(Debug, Clone)]
pub struct EdgeMap {
    /// Edge magnitude at each pixel.
    pub magnitude: Vec<u8>,
    /// Width of the edge map.
    pub width: u32,
    /// Height of the edge map.
    pub height: u32,
    /// Total number of edge pixels.
    pub edge_count: usize,
}

impl EdgeMap {
    /// Create a new edge map.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            magnitude: vec![0; size],
            width,
            height,
            edge_count: 0,
        }
    }

    /// Compute edge map from grayscale image using Sobel operator.
    pub fn from_sobel(data: &[u8], width: u32, height: u32, threshold: u8) -> CvResult<Self> {
        if width < 3 || height < 3 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height) as usize;
        if data.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, data.len()));
        }

        let mut edge_map = Self::new(width, height);
        let w = width as i32;
        let h = height as i32;

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = (y * w + x) as usize;

                // Sobel kernels
                let gx = (get_pixel(data, x + 1, y - 1, w) as i32
                    + 2 * get_pixel(data, x + 1, y, w) as i32
                    + get_pixel(data, x + 1, y + 1, w) as i32)
                    - (get_pixel(data, x - 1, y - 1, w) as i32
                        + 2 * get_pixel(data, x - 1, y, w) as i32
                        + get_pixel(data, x - 1, y + 1, w) as i32);

                let gy = (get_pixel(data, x - 1, y + 1, w) as i32
                    + 2 * get_pixel(data, x, y + 1, w) as i32
                    + get_pixel(data, x + 1, y + 1, w) as i32)
                    - (get_pixel(data, x - 1, y - 1, w) as i32
                        + 2 * get_pixel(data, x, y - 1, w) as i32
                        + get_pixel(data, x + 1, y - 1, w) as i32);

                let magnitude = ((gx * gx + gy * gy) as f64).sqrt() as u32;
                let magnitude = magnitude.min(255) as u8;

                edge_map.magnitude[idx] = magnitude;

                if magnitude >= threshold {
                    edge_map.edge_count += 1;
                }
            }
        }

        Ok(edge_map)
    }

    /// Compute edge map using simple gradient.
    pub fn from_gradient(data: &[u8], width: u32, height: u32, threshold: u8) -> CvResult<Self> {
        if width < 2 || height < 2 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height) as usize;
        if data.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, data.len()));
        }

        let mut edge_map = Self::new(width, height);
        let w = width as i32;
        let h = height as i32;

        for y in 0..(h - 1) {
            for x in 0..(w - 1) {
                let idx = (y * w + x) as usize;

                let curr = data[idx] as i32;
                let right = data[idx + 1] as i32;
                let down = data[(idx as i32 + w) as usize] as i32;

                let gx = (right - curr).abs();
                let gy = (down - curr).abs();
                let magnitude = (gx + gy).min(255) as u8;

                edge_map.magnitude[idx] = magnitude;

                if magnitude >= threshold {
                    edge_map.edge_count += 1;
                }
            }
        }

        Ok(edge_map)
    }

    /// Apply dilation to enhance edges.
    pub fn dilate(&mut self, size: usize) {
        if size == 0 {
            return;
        }

        let width = self.width as i32;
        let height = self.height as i32;
        let size = size as i32;
        let mut dilated = self.magnitude.clone();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let mut max_val = self.magnitude[idx];

                for dy in -size..=size {
                    for dx in -size..=size {
                        let nx = x + dx;
                        let ny = y + dy;

                        if nx >= 0 && nx < width && ny >= 0 && ny < height {
                            let nidx = (ny * width + nx) as usize;
                            max_val = max_val.max(self.magnitude[nidx]);
                        }
                    }
                }

                dilated[idx] = max_val;
            }
        }

        self.magnitude = dilated;
        self.edge_count = self.magnitude.iter().filter(|&&v| v > 0).count();
    }

    /// Compute edge change ratio with another edge map.
    #[must_use]
    pub fn change_ratio(&self, other: &Self) -> f64 {
        if self.width != other.width || self.height != other.height {
            return 1.0; // Maximum difference if dimensions don't match
        }

        let total_pixels = (self.width * self.height) as f64;
        let mut diff_count = 0;

        for (v1, v2) in self.magnitude.iter().zip(other.magnitude.iter()) {
            let is_edge1 = *v1 > 0;
            let is_edge2 = *v2 > 0;

            if is_edge1 != is_edge2 {
                diff_count += 1;
            }
        }

        diff_count as f64 / total_pixels
    }

    /// Compute edge pattern difference (pixel-wise).
    #[must_use]
    pub fn pattern_difference(&self, other: &Self) -> f64 {
        if self.width != other.width || self.height != other.height {
            return 1.0;
        }

        let mut sum_diff = 0u64;
        let mut sum_total = 0u64;

        for (v1, v2) in self.magnitude.iter().zip(other.magnitude.iter()) {
            let diff = (*v1 as i32 - *v2 as i32).unsigned_abs() as u64;
            sum_diff += diff;
            sum_total += (*v1 as u64).max(*v2 as u64);
        }

        if sum_total > 0 {
            sum_diff as f64 / sum_total as f64
        } else {
            0.0
        }
    }

    /// Compute edge count ratio.
    #[must_use]
    pub fn count_ratio(&self, other: &Self) -> f64 {
        let max_count = self.edge_count.max(other.edge_count) as f64;

        if max_count < 1.0 {
            return 0.0;
        }

        let diff = (self.edge_count as i64 - other.edge_count as i64).abs() as f64;
        diff / max_count
    }
}

/// Get pixel value safely.
fn get_pixel(data: &[u8], x: i32, y: i32, width: i32) -> u8 {
    let idx = (y * width + x) as usize;
    if idx < data.len() {
        data[idx]
    } else {
        0
    }
}

/// Extract grayscale data from a video frame.
fn extract_grayscale(frame: &VideoFrame) -> CvResult<Vec<u8>> {
    match frame.format {
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
            if frame.planes.is_empty() {
                return Err(CvError::insufficient_data(1, 0));
            }
            // Y plane is already grayscale
            Ok(frame.planes[0].data.clone())
        }
        PixelFormat::Rgb24 => {
            if frame.planes.is_empty() {
                return Err(CvError::insufficient_data(1, 0));
            }

            let data = &frame.planes[0].data;
            let size = (frame.width * frame.height) as usize;
            let mut gray = Vec::with_capacity(size);

            for chunk in data.chunks_exact(3) {
                // Rec. 601 luma
                let luma = (chunk[0] as f64 * 0.299
                    + chunk[1] as f64 * 0.587
                    + chunk[2] as f64 * 0.114) as u8;
                gray.push(luma);
            }

            Ok(gray)
        }
        _ => Err(CvError::unsupported_format(format!("{:?}", frame.format))),
    }
}

/// Compute edge similarity between two frames.
pub fn compute_edge_similarity(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    config: &EdgeConfig,
) -> CvResult<f64> {
    if frame1.width != frame2.width || frame1.height != frame2.height {
        return Err(CvError::invalid_parameter(
            "frames",
            "dimensions must match",
        ));
    }

    let gray1 = extract_grayscale(frame1)?;
    let gray2 = extract_grayscale(frame2)?;

    let mut edge1 = if config.use_sobel {
        EdgeMap::from_sobel(&gray1, frame1.width, frame1.height, config.edge_threshold)?
    } else {
        EdgeMap::from_gradient(&gray1, frame1.width, frame1.height, config.edge_threshold)?
    };

    let mut edge2 = if config.use_sobel {
        EdgeMap::from_sobel(&gray2, frame2.width, frame2.height, config.edge_threshold)?
    } else {
        EdgeMap::from_gradient(&gray2, frame2.width, frame2.height, config.edge_threshold)?
    };

    if config.dilation_size > 0 {
        edge1.dilate(config.dilation_size);
        edge2.dilate(config.dilation_size);
    }

    let difference = if config.compare_count {
        edge1.count_ratio(&edge2)
    } else {
        edge1.change_ratio(&edge2) * 0.7 + edge1.pattern_difference(&edge2) * 0.3
    };

    Ok(1.0 - difference)
}

/// Detect edge-based scene changes.
pub fn detect_edge_changes(
    frames: &[VideoFrame],
    config: &SceneConfig,
) -> CvResult<Vec<SceneChange>> {
    let mut changes = Vec::new();

    for i in 1..frames.len() {
        let similarity = compute_edge_similarity(&frames[i - 1], &frames[i], &config.edge_config)?;
        let diff = 1.0 - similarity;

        if diff > config.threshold {
            changes.push(SceneChange {
                frame_number: i,
                timestamp: frames[i].timestamp,
                confidence: diff,
                change_type: ChangeType::Cut,
                metadata: SceneMetadata {
                    edge_change_ratio: Some(diff),
                    ..Default::default()
                },
            });
        }
    }

    Ok(changes)
}

/// Compute edge density of a frame.
pub fn compute_edge_density(frame: &VideoFrame, config: &EdgeConfig) -> CvResult<f64> {
    let gray = extract_grayscale(frame)?;

    let edge_map = if config.use_sobel {
        EdgeMap::from_sobel(&gray, frame.width, frame.height, config.edge_threshold)?
    } else {
        EdgeMap::from_gradient(&gray, frame.width, frame.height, config.edge_threshold)?
    };

    let total_pixels = (frame.width * frame.height) as f64;
    Ok(edge_map.edge_count as f64 / total_pixels)
}

/// Compute edge histogram for a frame.
pub fn compute_edge_histogram(frame: &VideoFrame, bins: usize) -> CvResult<Vec<u32>> {
    let gray = extract_grayscale(frame)?;
    let config = EdgeConfig::default();

    let edge_map = EdgeMap::from_sobel(&gray, frame.width, frame.height, 0)?;

    let mut histogram = vec![0u32; bins];
    let bin_scale = bins as f64 / 256.0;

    for &magnitude in &edge_map.magnitude {
        let bin = ((magnitude as f64 * bin_scale) as usize).min(bins - 1);
        histogram[bin] += 1;
    }

    Ok(histogram)
}

/// Compare edge histograms between two frames.
pub fn compare_edge_histograms(hist1: &[u32], hist2: &[u32]) -> f64 {
    if hist1.len() != hist2.len() {
        return 1.0;
    }

    let total1: u32 = hist1.iter().sum();
    let total2: u32 = hist2.iter().sum();

    if total1 == 0 || total2 == 0 {
        return 0.0;
    }

    // Compute chi-squared distance
    let mut chi_sq = 0.0;

    for (h1, h2) in hist1.iter().zip(hist2.iter()) {
        let n1 = *h1 as f64 / total1 as f64;
        let n2 = *h2 as f64 / total2 as f64;
        let sum = n1 + n2;

        if sum > f64::EPSILON {
            let diff = n1 - n2;
            chi_sq += diff * diff / sum;
        }
    }

    (chi_sq / 2.0).min(1.0)
}

/// Detect scene changes using edge histogram comparison.
pub fn detect_edge_histogram_changes(
    frames: &[VideoFrame],
    config: &SceneConfig,
    bins: usize,
) -> CvResult<Vec<SceneChange>> {
    let mut changes = Vec::new();

    for i in 1..frames.len() {
        let hist1 = compute_edge_histogram(&frames[i - 1], bins)?;
        let hist2 = compute_edge_histogram(&frames[i], bins)?;

        let distance = compare_edge_histograms(&hist1, &hist2);

        if distance > config.threshold {
            changes.push(SceneChange {
                frame_number: i,
                timestamp: frames[i].timestamp,
                confidence: distance,
                change_type: ChangeType::Cut,
                metadata: SceneMetadata {
                    edge_change_ratio: Some(distance),
                    ..Default::default()
                },
            });
        }
    }

    Ok(changes)
}
