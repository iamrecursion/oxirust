//! Visual marker-based synchronization for multi-camera production.
//!
//! This module provides synchronization using visual markers like flashes and clappers.

use super::{SyncConfig, SyncMethod, SyncOffset, SyncResult, Synchronizer};
use crate::{AngleId, FrameNumber, Result};

/// Visual synchronizer
#[derive(Debug)]
pub struct VisualSync {
    /// Flash/marker detections for each angle
    markers: Vec<Vec<VisualMarker>>,
}

/// Visual synchronization marker
#[derive(Debug, Clone, Copy)]
pub struct VisualMarker {
    /// Frame number where marker was detected
    pub frame: FrameNumber,
    /// Marker type
    pub marker_type: MarkerType,
    /// Detection confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Marker intensity/brightness
    pub intensity: f32,
}

/// Type of visual marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerType {
    /// Bright flash (camera flash, lightning)
    Flash,
    /// Clapper/slate closing
    Clapper,
    /// LED marker pattern
    LedPattern,
    /// Color card
    ColorCard,
}

impl VisualSync {
    /// Create a new visual synchronizer
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            markers: vec![Vec::new(); angle_count],
        }
    }

    /// Add visual markers for an angle
    pub fn add_markers(&mut self, angle: AngleId, markers: Vec<VisualMarker>) {
        if angle < self.markers.len() {
            self.markers[angle] = markers;
        }
    }

    /// Detect flash in frame
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails
    pub fn detect_flash(
        &self,
        frame_data: &[u8],
        width: usize,
        height: usize,
        threshold: f32,
    ) -> Result<Option<VisualMarker>> {
        if frame_data.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No frame data".to_string(),
            ));
        }

        // Calculate average brightness
        let brightness = self.calculate_brightness(frame_data, width, height);

        // Detect flash if brightness exceeds threshold
        if brightness > threshold {
            let marker = VisualMarker {
                frame: 0, // Will be set by caller
                marker_type: MarkerType::Flash,
                confidence: f64::from(((brightness - threshold) / (1.0 - threshold)).min(1.0)),
                intensity: brightness,
            };
            Ok(Some(marker))
        } else {
            Ok(None)
        }
    }

    /// Calculate average brightness of frame
    fn calculate_brightness(&self, frame_data: &[u8], width: usize, height: usize) -> f32 {
        if frame_data.is_empty() || width == 0 || height == 0 {
            return 0.0;
        }

        let total_pixels = width * height;
        let bytes_per_pixel = frame_data.len().checked_div(total_pixels).unwrap_or(1);
        let mut sum = 0u64;
        let mut count = 0u64;

        for chunk in frame_data.chunks(bytes_per_pixel) {
            if chunk.is_empty() {
                continue;
            }
            // Simple luminance calculation
            let luma = if bytes_per_pixel >= 3 {
                // RGB or RGBA
                let r = u64::from(chunk[0]);
                let g = u64::from(chunk[1]);
                let b = u64::from(chunk[2]);
                (r * 299 + g * 587 + b * 114) / 1000
            } else {
                // Grayscale
                u64::from(chunk[0])
            };
            sum += luma;
            count += 1;
        }

        sum.checked_div(count).unwrap_or(0) as f32 / 255.0
    }

    /// Detect clapper closure
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails
    pub fn detect_clapper(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> Result<Option<FrameNumber>> {
        if frames.len() < 3 {
            return Err(crate::MultiCamError::InsufficientData(
                "Need at least 3 frames for clapper detection".to_string(),
            ));
        }

        // Look for sudden change in top portion of frame (where clapper closes)
        let roi_height = height / 4; // Top quarter
        let mut prev_brightness = 0.0f32;

        for (i, frame) in frames.iter().enumerate() {
            let brightness = self.calculate_roi_brightness(frame, width, roi_height, 0, roi_height);

            if i > 0 {
                let diff = (brightness - prev_brightness).abs();
                if diff > 0.2 {
                    // Significant change detected
                    return Ok(Some(i as FrameNumber));
                }
            }
            prev_brightness = brightness;
        }

        Ok(None)
    }

    /// Calculate brightness of region of interest
    fn calculate_roi_brightness(
        &self,
        frame_data: &[u8],
        width: usize,
        height: usize,
        y_start: usize,
        y_end: usize,
    ) -> f32 {
        if frame_data.is_empty() || width == 0 || height == 0 {
            return 0.0;
        }

        let total_pixels = width * height;
        let bytes_per_pixel = frame_data.len().checked_div(total_pixels).unwrap_or(1);
        let mut sum = 0u64;
        let mut count = 0u64;

        let y_start = y_start.min(height);
        let y_end = y_end.min(height);

        for y in y_start..y_end {
            for x in 0..width {
                let offset = (y * width + x) * bytes_per_pixel;
                if offset + bytes_per_pixel <= frame_data.len() {
                    let chunk = &frame_data[offset..offset + bytes_per_pixel];
                    let luma = if bytes_per_pixel >= 3 {
                        let r = u64::from(chunk[0]);
                        let g = u64::from(chunk[1]);
                        let b = u64::from(chunk[2]);
                        (r * 299 + g * 587 + b * 114) / 1000
                    } else {
                        u64::from(chunk[0])
                    };
                    sum += luma;
                    count += 1;
                }
            }
        }

        sum.checked_div(count).unwrap_or(0) as f32 / 255.0
    }

    /// Find offset using visual markers
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails
    pub fn find_offset(&self, angle_a: AngleId, angle_b: AngleId) -> Result<SyncOffset> {
        if angle_a >= self.markers.len() || angle_b >= self.markers.len() {
            return Err(crate::MultiCamError::AngleNotFound(angle_a.max(angle_b)));
        }

        let markers_a = &self.markers[angle_a];
        let markers_b = &self.markers[angle_b];

        if markers_a.is_empty() || markers_b.is_empty() {
            return Err(crate::MultiCamError::NoSyncMarkers);
        }

        // Find closest matching markers
        let (offset, confidence) = self.match_markers(markers_a, markers_b)?;

        Ok(SyncOffset::new(angle_b, offset, 0.0, confidence))
    }

    /// Match visual markers between two angles
    fn match_markers(
        &self,
        markers_a: &[VisualMarker],
        markers_b: &[VisualMarker],
    ) -> Result<(i64, f64)> {
        let mut best_offset = 0i64;
        let mut best_confidence = 0.0f64;

        // Try matching each marker in A with markers in B
        for marker_a in markers_a {
            for marker_b in markers_b {
                if marker_a.marker_type == marker_b.marker_type {
                    let offset = marker_b.frame as i64 - marker_a.frame as i64;
                    let confidence = (marker_a.confidence * marker_b.confidence).sqrt();

                    if confidence > best_confidence {
                        best_confidence = confidence;
                        best_offset = offset;
                    }
                }
            }
        }

        if best_confidence > 0.0 {
            Ok((best_offset, best_confidence))
        } else {
            Err(crate::MultiCamError::NoSyncMarkers)
        }
    }

    /// Detect LED pattern markers
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails
    pub fn detect_led_pattern(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
        pattern: &[bool],
    ) -> Result<Option<FrameNumber>> {
        if frames.len() < pattern.len() {
            return Err(crate::MultiCamError::InsufficientData(
                "Not enough frames to match pattern".to_string(),
            ));
        }

        let threshold = 0.5;
        for start in 0..=(frames.len() - pattern.len()) {
            let mut matches = true;
            for (i, &expected_on) in pattern.iter().enumerate() {
                let brightness = self.calculate_brightness(frames[start + i], width, height);
                let detected_on = brightness > threshold;
                if detected_on != expected_on {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Ok(Some(start as FrameNumber));
            }
        }

        Ok(None)
    }

    /// Filter markers by confidence threshold
    pub fn filter_markers(&mut self, angle: AngleId, min_confidence: f64) {
        if angle >= self.markers.len() {
            return;
        }

        self.markers[angle].retain(|m| m.confidence >= min_confidence);
    }

    /// Get markers of specific type
    #[must_use]
    pub fn get_markers_by_type(
        &self,
        angle: AngleId,
        marker_type: MarkerType,
    ) -> Vec<VisualMarker> {
        if angle >= self.markers.len() {
            return Vec::new();
        }

        self.markers[angle]
            .iter()
            .filter(|m| m.marker_type == marker_type)
            .copied()
            .collect()
    }

    /// Find nearest marker to frame
    #[must_use]
    pub fn find_nearest_marker(&self, angle: AngleId, frame: FrameNumber) -> Option<VisualMarker> {
        if angle >= self.markers.len() {
            return None;
        }

        self.markers[angle]
            .iter()
            .min_by_key(|m| (m.frame as i64 - frame as i64).abs())
            .copied()
    }
}

impl Synchronizer for VisualSync {
    fn synchronize(&self, _config: &SyncConfig) -> Result<SyncResult> {
        if self.markers.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No visual markers available".to_string(),
            ));
        }

        let mut offsets = Vec::new();
        let reference_angle = 0;

        // Calculate offsets relative to first angle
        for angle in 1..self.markers.len() {
            let offset = self.find_offset(reference_angle, angle)?;
            offsets.push(offset);
        }

        // Add zero offset for reference angle
        offsets.insert(0, SyncOffset::new(reference_angle, 0, 0.0, 1.0));

        // Calculate average confidence
        #[allow(clippy::manual_checked_ops)]
        let confidence = offsets.iter().map(|o| o.confidence).sum::<f64>() / offsets.len() as f64;

        Ok(SyncResult {
            reference_angle,
            offsets,
            confidence,
            method: SyncMethod::Visual,
        })
    }

    fn method(&self) -> SyncMethod {
        SyncMethod::Visual
    }

    fn is_reliable(&self) -> bool {
        !self.markers.is_empty() && self.markers.iter().all(|m| !m.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_sync_creation() {
        let sync = VisualSync::new(3);
        assert_eq!(sync.markers.len(), 3);
    }

    #[test]
    fn test_visual_marker() {
        let marker = VisualMarker {
            frame: 100,
            marker_type: MarkerType::Flash,
            confidence: 0.95,
            intensity: 0.9,
        };
        assert_eq!(marker.frame, 100);
        assert_eq!(marker.marker_type, MarkerType::Flash);
    }

    #[test]
    fn test_calculate_brightness() {
        let sync = VisualSync::new(1);
        let frame = vec![255u8; 1920 * 1080 * 3]; // White frame
        let brightness = sync.calculate_brightness(&frame, 1920, 1080);
        assert!((brightness - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_markers() {
        let mut sync = VisualSync::new(1);
        let markers = vec![
            VisualMarker {
                frame: 0,
                marker_type: MarkerType::Flash,
                confidence: 0.9,
                intensity: 0.9,
            },
            VisualMarker {
                frame: 10,
                marker_type: MarkerType::Flash,
                confidence: 0.5,
                intensity: 0.6,
            },
            VisualMarker {
                frame: 20,
                marker_type: MarkerType::Flash,
                confidence: 0.3,
                intensity: 0.4,
            },
        ];
        sync.add_markers(0, markers);
        sync.filter_markers(0, 0.7);
        assert_eq!(sync.markers[0].len(), 1);
    }

    #[test]
    fn test_get_markers_by_type() {
        let mut sync = VisualSync::new(1);
        let markers = vec![
            VisualMarker {
                frame: 0,
                marker_type: MarkerType::Flash,
                confidence: 0.9,
                intensity: 0.9,
            },
            VisualMarker {
                frame: 10,
                marker_type: MarkerType::Clapper,
                confidence: 0.8,
                intensity: 0.7,
            },
            VisualMarker {
                frame: 20,
                marker_type: MarkerType::Flash,
                confidence: 0.9,
                intensity: 0.9,
            },
        ];
        sync.add_markers(0, markers);
        let flashes = sync.get_markers_by_type(0, MarkerType::Flash);
        assert_eq!(flashes.len(), 2);
    }
}
