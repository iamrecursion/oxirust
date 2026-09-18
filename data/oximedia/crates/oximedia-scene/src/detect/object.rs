//! Lightweight object detection using HOG (Histogram of Oriented Gradients).

use crate::common::{Confidence, Rect};
use crate::error::{SceneError, SceneResult};
use crate::features::extract::HogFeatures;
use serde::{Deserialize, Serialize};

/// Type of detected object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectType {
    /// Person/human.
    Person,
    /// Vehicle (car, truck, etc.).
    Vehicle,
    /// Animal.
    Animal,
    /// Building or structure.
    Building,
    /// Plant or tree.
    Plant,
    /// Furniture.
    Furniture,
    /// Sports equipment.
    SportsEquipment,
    /// Unknown object.
    Unknown,
}

impl ObjectType {
    /// Get all object types.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Person,
            Self::Vehicle,
            Self::Animal,
            Self::Building,
            Self::Plant,
            Self::Furniture,
            Self::SportsEquipment,
            Self::Unknown,
        ]
    }

    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Person => "Person",
            Self::Vehicle => "Vehicle",
            Self::Animal => "Animal",
            Self::Building => "Building",
            Self::Plant => "Plant",
            Self::Furniture => "Furniture",
            Self::SportsEquipment => "Sports Equipment",
            Self::Unknown => "Unknown",
        }
    }
}

/// Detected object with location and type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetection {
    /// Object type.
    pub object_type: ObjectType,
    /// Bounding box.
    pub bbox: Rect,
    /// Detection confidence.
    pub confidence: Confidence,
    /// Additional properties.
    pub properties: ObjectProperties,
}

/// Properties of detected object.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObjectProperties {
    /// Aspect ratio (width/height).
    pub aspect_ratio: f32,
    /// Size relative to image (0.0-1.0).
    pub relative_size: f32,
    /// Position in image (center).
    pub position: (f32, f32),
}

/// Configuration for object detection.
#[derive(Debug, Clone)]
pub struct ObjectDetectorConfig {
    /// Minimum confidence threshold.
    pub confidence_threshold: f32,
    /// Minimum object size (pixels).
    pub min_size: usize,
    /// Maximum object size (pixels).
    pub max_size: usize,
    /// Enable multi-scale detection.
    pub multi_scale: bool,
    /// Non-maximum suppression threshold.
    pub nms_threshold: f32,
}

impl Default for ObjectDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_size: 20,
            max_size: 1000,
            multi_scale: true,
            nms_threshold: 0.3,
        }
    }
}

/// Object detector using HOG features.
pub struct ObjectDetector {
    config: ObjectDetectorConfig,
    hog: HogFeatures,
}

impl ObjectDetector {
    /// Create a new object detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ObjectDetectorConfig::default(),
            hog: HogFeatures::new(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: ObjectDetectorConfig) -> Self {
        Self {
            config,
            hog: HogFeatures::new(),
        }
    }

    /// Detect objects in an RGB image.
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
    ) -> SceneResult<Vec<ObjectDetection>> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Convert to grayscale
        let gray = self.rgb_to_gray(rgb_data, width, height);

        // Detect using sliding window
        let mut detections = Vec::new();

        if self.config.multi_scale {
            // Multi-scale detection
            for scale in [1.0, 1.2, 1.5, 2.0] {
                let scaled_detections = self.detect_at_scale(&gray, width, height, scale)?;
                detections.extend(scaled_detections);
            }
        } else {
            // Single scale
            let single_scale = self.detect_at_scale(&gray, width, height, 1.0)?;
            detections.extend(single_scale);
        }

        // Apply non-maximum suppression
        let filtered = self.non_maximum_suppression(&detections);

        Ok(filtered)
    }

    /// Detect objects at a specific scale.
    fn detect_at_scale(
        &self,
        gray: &[f32],
        width: usize,
        height: usize,
        scale: f32,
    ) -> SceneResult<Vec<ObjectDetection>> {
        let mut detections = Vec::new();
        let window_size = (64.0 * scale) as usize;
        let stride = (window_size / 4).max(1);

        if window_size < self.config.min_size || window_size > self.config.max_size {
            return Ok(detections);
        }

        for y in (0..height.saturating_sub(window_size)).step_by(stride) {
            for x in (0..width.saturating_sub(window_size)).step_by(stride) {
                // Extract window
                let window = self.extract_window(gray, width, x, y, window_size);

                // Compute HOG features
                let features = self.hog.compute(&window, window_size, window_size);

                // Classify window
                if let Some((object_type, confidence)) = self.classify_window(&features) {
                    if confidence.value() >= self.config.confidence_threshold {
                        let bbox =
                            Rect::new(x as f32, y as f32, window_size as f32, window_size as f32);

                        let properties = ObjectProperties {
                            aspect_ratio: 1.0,
                            relative_size: (window_size * window_size) as f32
                                / (width * height) as f32,
                            position: (
                                (x + window_size / 2) as f32 / width as f32,
                                (y + window_size / 2) as f32 / height as f32,
                            ),
                        };

                        detections.push(ObjectDetection {
                            object_type,
                            bbox,
                            confidence,
                            properties,
                        });
                    }
                }
            }
        }

        Ok(detections)
    }

    /// Extract a window from grayscale image.
    fn extract_window(
        &self,
        gray: &[f32],
        width: usize,
        x: usize,
        y: usize,
        size: usize,
    ) -> Vec<f32> {
        let mut window = Vec::with_capacity(size * size);
        for dy in 0..size {
            for dx in 0..size {
                let idx = (y + dy) * width + (x + dx);
                window.push(gray[idx]);
            }
        }
        window
    }

    /// Classify a window based on HOG features.
    fn classify_window(&self, features: &[f32]) -> Option<(ObjectType, Confidence)> {
        // Simple heuristic classifier based on HOG feature statistics
        let feature_sum: f32 = features.iter().sum();
        let feature_mean = feature_sum / features.len() as f32;
        let feature_max = features.iter().copied().fold(f32::MIN, f32::max);

        // These are simplified heuristics - in practice, you'd use trained classifiers
        if feature_mean > 0.1 && feature_max > 0.5 {
            // Strong edges suggest person or vehicle
            if feature_max > 0.8 {
                return Some((ObjectType::Person, Confidence::new(0.7)));
            }
            return Some((ObjectType::Vehicle, Confidence::new(0.6)));
        } else if feature_mean > 0.05 {
            // Moderate edges might be building or furniture
            return Some((ObjectType::Building, Confidence::new(0.5)));
        }

        None
    }

    /// Apply non-maximum suppression to remove overlapping detections.
    fn non_maximum_suppression(&self, detections: &[ObjectDetection]) -> Vec<ObjectDetection> {
        let mut result = Vec::new();
        let mut sorted = detections.to_vec();

        // Sort by confidence descending
        sorted.sort_by(|a, b| {
            b.confidence
                .value()
                .partial_cmp(&a.confidence.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut suppressed = vec![false; sorted.len()];

        for i in 0..sorted.len() {
            if suppressed[i] {
                continue;
            }

            result.push(sorted[i].clone());

            // Suppress overlapping detections
            for j in (i + 1)..sorted.len() {
                if suppressed[j] {
                    continue;
                }

                let iou = sorted[i].bbox.iou(&sorted[j].bbox);
                if iou > self.config.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        result
    }

    /// Convert RGB to grayscale.
    fn rgb_to_gray(&self, rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut gray = Vec::with_capacity(width * height);
        for i in (0..rgb.len()).step_by(3) {
            let r = rgb[i] as f32;
            let g = rgb[i + 1] as f32;
            let b = rgb[i + 2] as f32;
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            gray.push(y / 255.0);
        }
        gray
    }
}

impl Default for ObjectDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_type_name() {
        assert_eq!(ObjectType::Person.name(), "Person");
        assert_eq!(ObjectType::Vehicle.name(), "Vehicle");
    }

    #[test]
    fn test_object_detector() {
        let detector = ObjectDetector::new();
        let width = 256;
        let height = 256;
        let rgb_data = vec![128u8; width * height * 3];

        let result = detector.detect(&rgb_data, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nms() {
        let detector = ObjectDetector::new();
        let detections = vec![
            ObjectDetection {
                object_type: ObjectType::Person,
                bbox: Rect::new(10.0, 10.0, 50.0, 50.0),
                confidence: Confidence::new(0.9),
                properties: ObjectProperties::default(),
            },
            ObjectDetection {
                object_type: ObjectType::Person,
                bbox: Rect::new(15.0, 15.0, 50.0, 50.0),
                confidence: Confidence::new(0.7),
                properties: ObjectProperties::default(),
            },
        ];

        let filtered = detector.non_maximum_suppression(&detections);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].confidence.value() - 0.9).abs() < f32::EPSILON);
    }
}
