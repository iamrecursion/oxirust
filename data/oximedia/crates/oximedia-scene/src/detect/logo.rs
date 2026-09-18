//! Logo and graphic detection using template matching and features.

use crate::common::{Confidence, Rect};
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Detected logo or graphic element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogoDetection {
    /// Bounding box of the logo.
    pub bbox: Rect,
    /// Detection confidence.
    pub confidence: Confidence,
    /// Logo properties.
    pub properties: LogoProperties,
}

/// Properties of detected logo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogoProperties {
    /// Logo type (corner overlay, watermark, banner).
    pub logo_type: LogoType,
    /// Position category.
    pub position: LogoPosition,
    /// Opacity estimate (0.0-1.0).
    pub opacity: f32,
    /// Is animated or static.
    pub is_static: bool,
}

/// Type of logo detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogoType {
    /// Corner overlay (network logo).
    CornerOverlay,
    /// Watermark (semi-transparent).
    Watermark,
    /// Banner (lower third).
    Banner,
    /// Bug (small persistent graphic).
    Bug,
    /// Unknown type.
    Unknown,
}

/// Position of logo in frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogoPosition {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Top center.
    TopCenter,
    /// Bottom center.
    BottomCenter,
    /// Center.
    Center,
}

/// Configuration for logo detection.
#[derive(Debug, Clone)]
pub struct LogoDetectorConfig {
    /// Minimum confidence threshold.
    pub confidence_threshold: f32,
    /// Minimum logo size (pixels).
    pub min_size: usize,
    /// Maximum logo size (relative to image).
    pub max_size_ratio: f32,
    /// Detect corner overlays.
    pub detect_corners: bool,
    /// Detect watermarks.
    pub detect_watermarks: bool,
    /// Detect banners.
    pub detect_banners: bool,
}

impl Default for LogoDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_size: 20,
            max_size_ratio: 0.25,
            detect_corners: true,
            detect_watermarks: true,
            detect_banners: true,
        }
    }
}

/// Logo detector.
pub struct LogoDetector {
    config: LogoDetectorConfig,
}

impl LogoDetector {
    /// Create a new logo detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: LogoDetectorConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: LogoDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect logos in an RGB image.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - RGB image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<LogoDetection>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let mut detections = Vec::new();

        // Detect corner overlays
        if self.config.detect_corners {
            detections.extend(self.detect_corner_overlays(rgb_data, width, height)?);
        }

        // Detect watermarks
        if self.config.detect_watermarks {
            detections.extend(self.detect_watermarks(rgb_data, width, height)?);
        }

        // Detect banners
        if self.config.detect_banners {
            detections.extend(self.detect_banners(rgb_data, width, height)?);
        }

        Ok(detections)
    }

    /// Detect logo in temporal sequence (for persistence detection).
    ///
    /// # Errors
    ///
    /// Returns error if insufficient frames.
    pub fn detect_temporal(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<LogoDetection>> {
        if frames.is_empty() {
            return Err(SceneError::InsufficientData(
                "Need at least one frame".to_string(),
            ));
        }

        // Detect in each frame
        let mut frame_detections = Vec::new();
        for frame in frames {
            let detections = self.detect(frame, width, height)?;
            frame_detections.push(detections);
        }

        // Find persistent logos
        let persistent = self.find_persistent_logos(&frame_detections);

        Ok(persistent)
    }

    /// Detect corner overlay logos.
    fn detect_corner_overlays(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<LogoDetection>> {
        let mut detections = Vec::new();
        let corner_size = width.min(height) / 5;

        // Define corner regions
        let corners = [
            (0, 0, LogoPosition::TopLeft),
            (width - corner_size, 0, LogoPosition::TopRight),
            (0, height - corner_size, LogoPosition::BottomLeft),
            (
                width - corner_size,
                height - corner_size,
                LogoPosition::BottomRight,
            ),
        ];

        for (cx, cy, position) in corners {
            if let Some(logo) =
                self.analyze_region(rgb_data, width, height, cx, cy, corner_size, position)
            {
                detections.push(logo);
            }
        }

        Ok(detections)
    }

    /// Detect watermark logos (semi-transparent overlays).
    fn detect_watermarks(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<LogoDetection>> {
        let mut detections = Vec::new();

        // Analyze center region for watermarks
        let wm_size = width.min(height) / 3;
        let cx = (width - wm_size) / 2;
        let cy = (height - wm_size) / 2;

        // Look for low-contrast persistent patterns
        if let Some(watermark_score) =
            self.detect_watermark_pattern(rgb_data, width, height, cx, cy, wm_size)
        {
            if watermark_score > self.config.confidence_threshold {
                detections.push(LogoDetection {
                    bbox: Rect::new(cx as f32, cy as f32, wm_size as f32, wm_size as f32),
                    confidence: Confidence::new(watermark_score),
                    properties: LogoProperties {
                        logo_type: LogoType::Watermark,
                        position: LogoPosition::Center,
                        opacity: 0.3,
                        is_static: true,
                    },
                });
            }
        }

        Ok(detections)
    }

    /// Detect banner logos (lower thirds, etc.).
    fn detect_banners(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<LogoDetection>> {
        let mut detections = Vec::new();
        let banner_height = height / 6;

        // Check bottom third
        let bottom_y = height - banner_height;
        if let Some(banner_score) =
            self.detect_banner_pattern(rgb_data, width, height, 0, bottom_y, width, banner_height)
        {
            if banner_score > self.config.confidence_threshold {
                detections.push(LogoDetection {
                    bbox: Rect::new(0.0, bottom_y as f32, width as f32, banner_height as f32),
                    confidence: Confidence::new(banner_score),
                    properties: LogoProperties {
                        logo_type: LogoType::Banner,
                        position: LogoPosition::BottomCenter,
                        opacity: 1.0,
                        is_static: true,
                    },
                });
            }
        }

        Ok(detections)
    }

    /// Analyze a region for logo presence.
    fn analyze_region(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        rx: usize,
        ry: usize,
        size: usize,
        position: LogoPosition,
    ) -> Option<LogoDetection> {
        // Calculate region statistics
        let mut avg_brightness = 0.0;
        let mut edge_strength = 0.0;
        let mut pixel_count = 0;

        for y in ry..(ry + size).min(height) {
            for x in rx..(rx + size).min(width) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    let brightness = (rgb_data[idx] as f32
                        + rgb_data[idx + 1] as f32
                        + rgb_data[idx + 2] as f32)
                        / 3.0;
                    avg_brightness += brightness;
                    pixel_count += 1;

                    // Simple edge detection
                    if x > rx && y > ry {
                        let prev_idx = ((y - 1) * width + (x - 1)) * 3;
                        let diff = ((rgb_data[idx] as i32 - rgb_data[prev_idx] as i32).abs()
                            + (rgb_data[idx + 1] as i32 - rgb_data[prev_idx + 1] as i32).abs()
                            + (rgb_data[idx + 2] as i32 - rgb_data[prev_idx + 2] as i32).abs())
                            as f32;
                        edge_strength += diff;
                    }
                }
            }
        }

        if pixel_count == 0 {
            return None;
        }

        avg_brightness /= pixel_count as f32;
        edge_strength /= pixel_count as f32;

        // Heuristic: logos typically have strong edges and distinct appearance
        let has_strong_edges = edge_strength > 10.0;
        let has_distinct_appearance = avg_brightness < 200.0 || avg_brightness > 50.0;

        if has_strong_edges && has_distinct_appearance {
            let confidence = ((edge_strength / 50.0).min(1.0) * 0.7).clamp(0.0, 1.0);

            Some(LogoDetection {
                bbox: Rect::new(rx as f32, ry as f32, size as f32, size as f32),
                confidence: Confidence::new(confidence),
                properties: LogoProperties {
                    logo_type: LogoType::CornerOverlay,
                    position,
                    opacity: 1.0,
                    is_static: true,
                },
            })
        } else {
            None
        }
    }

    /// Detect watermark pattern.
    fn detect_watermark_pattern(
        &self,
        rgb_data: &[u8],
        width: usize,
        _height: usize,
        rx: usize,
        ry: usize,
        size: usize,
    ) -> Option<f32> {
        // Look for low-contrast repetitive patterns
        let mut variance = 0.0;
        let mut mean = 0.0;
        let mut count = 0;

        for y in ry..(ry + size) {
            for x in rx..(rx + size) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    let brightness = (rgb_data[idx] as f32
                        + rgb_data[idx + 1] as f32
                        + rgb_data[idx + 2] as f32)
                        / 3.0;
                    mean += brightness;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return None;
        }

        mean /= count as f32;

        for y in ry..(ry + size) {
            for x in rx..(rx + size) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    let brightness = (rgb_data[idx] as f32
                        + rgb_data[idx + 1] as f32
                        + rgb_data[idx + 2] as f32)
                        / 3.0;
                    variance += (brightness - mean).powi(2);
                }
            }
        }

        variance /= count as f32;

        // Watermarks typically have low variance (subtle)
        let score = if variance < 500.0 && variance > 10.0 {
            (1.0 - variance / 500.0) * 0.6
        } else {
            0.0
        };

        Some(score)
    }

    /// Detect banner pattern.
    fn detect_banner_pattern(
        &self,
        rgb_data: &[u8],
        width: usize,
        _height: usize,
        rx: usize,
        ry: usize,
        rwidth: usize,
        rheight: usize,
    ) -> Option<f32> {
        // Banners typically have uniform color and sharp edges
        let mut avg_color = [0.0f32; 3];
        let mut count = 0;

        for y in ry..(ry + rheight) {
            for x in rx..(rx + rwidth) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    avg_color[0] += rgb_data[idx] as f32;
                    avg_color[1] += rgb_data[idx + 1] as f32;
                    avg_color[2] += rgb_data[idx + 2] as f32;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return None;
        }

        for c in &mut avg_color {
            *c /= count as f32;
        }

        // Check uniformity
        let mut variance = 0.0;
        for y in ry..(ry + rheight) {
            for x in rx..(rx + rwidth) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    for c in 0..3 {
                        let diff = rgb_data[idx + c] as f32 - avg_color[c];
                        variance += diff * diff;
                    }
                }
            }
        }

        variance /= count as f32;

        // Low variance indicates uniform banner
        let score = if variance < 1000.0 {
            (1.0 - variance / 1000.0) * 0.7
        } else {
            0.0
        };

        Some(score)
    }

    /// Find logos that persist across frames.
    fn find_persistent_logos(&self, frame_detections: &[Vec<LogoDetection>]) -> Vec<LogoDetection> {
        if frame_detections.is_empty() {
            return Vec::new();
        }

        let mut persistent = Vec::new();

        // Take detections from first frame as candidates
        for detection in &frame_detections[0] {
            let mut appear_count = 1;

            // Check if this detection appears in other frames
            for frame in &frame_detections[1..] {
                for other in frame {
                    let iou = detection.bbox.iou(&other.bbox);
                    if iou > 0.5 {
                        appear_count += 1;
                        break;
                    }
                }
            }

            // If appears in most frames, it's persistent
            if appear_count as f32 / frame_detections.len() as f32 > 0.7 {
                let mut detection = detection.clone();
                detection.properties.is_static = true;
                persistent.push(detection);
            }
        }

        persistent
    }
}

impl Default for LogoDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logo_detector() {
        let detector = LogoDetector::new();
        let width = 320;
        let height = 240;
        let rgb_data = vec![128u8; width * height * 3];

        let result = detector.detect(&rgb_data, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_detection() {
        let detector = LogoDetector::new();
        let width = 320;
        let height = 240;
        let frame = vec![128u8; width * height * 3];
        let frames = vec![&frame[..], &frame[..], &frame[..]];

        let result = detector.detect_temporal(&frames, width, height);
        assert!(result.is_ok());
    }
}
