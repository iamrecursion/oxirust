//! Emotion recognition module for analyzing facial expressions.
//!
//! Provides lightweight, patent-free facial expression analysis using geometric
//! features derived from face region statistics. Classifies into 7 basic emotions
//! (Ekman's universal emotions) plus neutral.

use crate::common::{Confidence, Rect};
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Basic emotion categories based on Ekman's universal emotions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Emotion {
    /// Neutral expression.
    Neutral,
    /// Happy/smiling.
    Happy,
    /// Sad expression.
    Sad,
    /// Angry expression.
    Angry,
    /// Surprised expression.
    Surprised,
    /// Disgusted expression.
    Disgusted,
    /// Fearful expression.
    Fearful,
    /// Contemptuous expression.
    Contemptuous,
}

impl Emotion {
    /// Get all emotion types.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Neutral,
            Self::Happy,
            Self::Sad,
            Self::Angry,
            Self::Surprised,
            Self::Disgusted,
            Self::Fearful,
            Self::Contemptuous,
        ]
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Happy => "happy",
            Self::Sad => "sad",
            Self::Angry => "angry",
            Self::Surprised => "surprised",
            Self::Disgusted => "disgusted",
            Self::Fearful => "fearful",
            Self::Contemptuous => "contemptuous",
        }
    }

    /// Valence (pleasure dimension): -1.0 (negative) to 1.0 (positive).
    #[must_use]
    pub const fn valence(&self) -> f32 {
        match self {
            Self::Neutral => 0.0,
            Self::Happy => 0.8,
            Self::Sad => -0.7,
            Self::Angry => -0.6,
            Self::Surprised => 0.1,
            Self::Disgusted => -0.5,
            Self::Fearful => -0.6,
            Self::Contemptuous => -0.3,
        }
    }

    /// Arousal (activation dimension): 0.0 (calm) to 1.0 (excited).
    #[must_use]
    pub const fn arousal(&self) -> f32 {
        match self {
            Self::Neutral => 0.1,
            Self::Happy => 0.6,
            Self::Sad => 0.3,
            Self::Angry => 0.9,
            Self::Surprised => 0.9,
            Self::Disgusted => 0.4,
            Self::Fearful => 0.8,
            Self::Contemptuous => 0.3,
        }
    }
}

/// Result of emotion recognition for a single face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionResult {
    /// Face bounding box.
    pub face_bbox: Rect,
    /// Dominant emotion.
    pub dominant_emotion: Emotion,
    /// Confidence in the dominant emotion.
    pub confidence: Confidence,
    /// Scores for all emotions.
    pub scores: Vec<(Emotion, f32)>,
    /// Valence-arousal coordinates.
    pub valence: f32,
    /// Arousal level.
    pub arousal: f32,
}

/// Facial expression features extracted from the face region.
#[derive(Debug, Clone)]
struct FacialFeatures {
    /// Average brightness of upper face (forehead/eyes).
    upper_brightness: f32,
    /// Average brightness of lower face (mouth/chin).
    lower_brightness: f32,
    /// Horizontal edge density in eye region.
    eye_region_edges: f32,
    /// Horizontal edge density in mouth region.
    mouth_region_edges: f32,
    /// Vertical gradient strength (brow furrow indicator).
    brow_gradient: f32,
    /// Symmetry of face region.
    face_symmetry: f32,
    /// Contrast in face region.
    face_contrast: f32,
    /// Variance of intensity in face region.
    intensity_variance: f32,
}

/// Configuration for emotion recognition.
#[derive(Debug, Clone)]
pub struct EmotionRecognizerConfig {
    /// Minimum confidence threshold for reporting an emotion.
    pub confidence_threshold: f32,
    /// Whether to compute valence-arousal coordinates.
    pub compute_valence_arousal: bool,
}

impl Default for EmotionRecognizerConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
            compute_valence_arousal: true,
        }
    }
}

/// Emotion recognizer that analyzes facial expressions from face regions.
pub struct EmotionRecognizer {
    config: EmotionRecognizerConfig,
}

impl EmotionRecognizer {
    /// Create a new emotion recognizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EmotionRecognizerConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: EmotionRecognizerConfig) -> Self {
        Self { config }
    }

    /// Recognize emotion from a face region in an RGB image.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - Full RGB image data
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `face_bbox` - Bounding box of the detected face
    ///
    /// # Errors
    ///
    /// Returns error if the face region is invalid.
    pub fn recognize(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        face_bbox: &Rect,
    ) -> SceneResult<EmotionResult> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        if face_bbox.width < 2.0 || face_bbox.height < 2.0 {
            return Err(SceneError::InvalidParameter(
                "Face bounding box too small".to_string(),
            ));
        }

        let features = self.extract_facial_features(rgb_data, width, height, face_bbox);
        let scores = self.classify_emotion(&features);

        let (dominant_emotion, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((Emotion::Neutral, 0.0), |(e, s)| (*e, *s));

        let valence = if self.config.compute_valence_arousal {
            scores.iter().map(|(e, s)| e.valence() * s).sum::<f32>()
        } else {
            0.0
        };

        let arousal = if self.config.compute_valence_arousal {
            scores.iter().map(|(e, s)| e.arousal() * s).sum::<f32>()
        } else {
            0.0
        };

        Ok(EmotionResult {
            face_bbox: *face_bbox,
            dominant_emotion,
            confidence: Confidence::new(best_score),
            scores,
            valence,
            arousal,
        })
    }

    /// Recognize emotions for multiple faces.
    ///
    /// # Errors
    ///
    /// Returns error if recognition fails.
    pub fn recognize_batch(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        face_bboxes: &[Rect],
    ) -> SceneResult<Vec<EmotionResult>> {
        let mut results = Vec::with_capacity(face_bboxes.len());
        for bbox in face_bboxes {
            results.push(self.recognize(rgb_data, width, height, bbox)?);
        }
        Ok(results)
    }

    /// Extract facial expression features from the face region.
    fn extract_facial_features(
        &self,
        rgb_data: &[u8],
        width: usize,
        _height: usize,
        bbox: &Rect,
    ) -> FacialFeatures {
        let x0 = bbox.x as usize;
        let y0 = bbox.y as usize;
        let fw = bbox.width as usize;
        let fh = bbox.height as usize;

        // Divide face into upper and lower halves
        let mid_y = y0 + fh / 2;

        let (upper_brightness, lower_brightness) = {
            let mut upper_sum = 0.0_f32;
            let mut lower_sum = 0.0_f32;
            let mut upper_count = 0;
            let mut lower_count = 0;

            for y in y0..(y0 + fh) {
                for x in x0..(x0 + fw) {
                    let idx = (y * width + x) * 3;
                    if idx + 2 < rgb_data.len() {
                        let lum = 0.299 * rgb_data[idx] as f32
                            + 0.587 * rgb_data[idx + 1] as f32
                            + 0.114 * rgb_data[idx + 2] as f32;
                        if y < mid_y {
                            upper_sum += lum;
                            upper_count += 1;
                        } else {
                            lower_sum += lum;
                            lower_count += 1;
                        }
                    }
                }
            }

            let ub = if upper_count > 0 {
                upper_sum / upper_count as f32 / 255.0
            } else {
                0.5
            };
            let lb = if lower_count > 0 {
                lower_sum / lower_count as f32 / 255.0
            } else {
                0.5
            };
            (ub, lb)
        };

        // Edge density in eye region (upper 1/3)
        let eye_y_start = y0 + fh / 6;
        let eye_y_end = y0 + fh / 3;
        let eye_region_edges = self.compute_edge_density(
            rgb_data,
            width,
            x0,
            eye_y_start,
            fw,
            eye_y_end - eye_y_start,
        );

        // Edge density in mouth region (lower 1/3)
        let mouth_y_start = y0 + fh * 2 / 3;
        let mouth_y_end = y0 + fh;
        let mouth_region_edges = self.compute_edge_density(
            rgb_data,
            width,
            x0,
            mouth_y_start,
            fw,
            mouth_y_end - mouth_y_start,
        );

        // Brow gradient (vertical edges in upper face)
        let brow_gradient = self.compute_vertical_gradient(rgb_data, width, x0, y0, fw, fh / 3);

        // Face symmetry
        let face_symmetry = self.compute_face_symmetry(rgb_data, width, x0, y0, fw, fh);

        // Contrast and variance
        let (face_contrast, intensity_variance) =
            self.compute_contrast_variance(rgb_data, width, x0, y0, fw, fh);

        FacialFeatures {
            upper_brightness,
            lower_brightness,
            eye_region_edges,
            mouth_region_edges,
            brow_gradient,
            face_symmetry,
            face_contrast,
            intensity_variance,
        }
    }

    /// Compute horizontal edge density in a region.
    fn compute_edge_density(
        &self,
        rgb_data: &[u8],
        width: usize,
        rx: usize,
        ry: usize,
        rw: usize,
        rh: usize,
    ) -> f32 {
        let mut edge_count = 0;
        let mut total = 0;

        for y in ry..(ry + rh) {
            for x in rx..(rx + rw).saturating_sub(1) {
                let idx = (y * width + x) * 3;
                let idx_next = (y * width + x + 1) * 3;
                if idx + 2 < rgb_data.len() && idx_next + 2 < rgb_data.len() {
                    let diff = (rgb_data[idx] as i32 - rgb_data[idx_next] as i32).abs()
                        + (rgb_data[idx + 1] as i32 - rgb_data[idx_next + 1] as i32).abs()
                        + (rgb_data[idx + 2] as i32 - rgb_data[idx_next + 2] as i32).abs();
                    if diff > 30 {
                        edge_count += 1;
                    }
                    total += 1;
                }
            }
        }

        if total > 0 {
            edge_count as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Compute vertical gradient strength.
    fn compute_vertical_gradient(
        &self,
        rgb_data: &[u8],
        width: usize,
        rx: usize,
        ry: usize,
        rw: usize,
        rh: usize,
    ) -> f32 {
        let mut gradient_sum = 0.0_f32;
        let mut count = 0;

        for y in ry..(ry + rh).saturating_sub(1) {
            for x in rx..(rx + rw) {
                let idx = (y * width + x) * 3;
                let idx_below = ((y + 1) * width + x) * 3;
                if idx + 2 < rgb_data.len() && idx_below + 2 < rgb_data.len() {
                    let diff = (rgb_data[idx] as i32 - rgb_data[idx_below] as i32).abs()
                        + (rgb_data[idx + 1] as i32 - rgb_data[idx_below + 1] as i32).abs()
                        + (rgb_data[idx + 2] as i32 - rgb_data[idx_below + 2] as i32).abs();
                    gradient_sum += diff as f32;
                    count += 1;
                }
            }
        }

        if count > 0 {
            gradient_sum / count as f32 / 255.0
        } else {
            0.0
        }
    }

    /// Compute face symmetry (left-right).
    fn compute_face_symmetry(
        &self,
        rgb_data: &[u8],
        width: usize,
        rx: usize,
        ry: usize,
        rw: usize,
        rh: usize,
    ) -> f32 {
        let mut diff_sum = 0.0_f32;
        let mut count = 0;

        for y in ry..(ry + rh) {
            for dx in 0..(rw / 2) {
                let left_x = rx + dx;
                let right_x = rx + rw - 1 - dx;
                let left_idx = (y * width + left_x) * 3;
                let right_idx = (y * width + right_x) * 3;

                if left_idx + 2 < rgb_data.len() && right_idx + 2 < rgb_data.len() {
                    for c in 0..3 {
                        diff_sum +=
                            (rgb_data[left_idx + c] as f32 - rgb_data[right_idx + c] as f32).abs();
                    }
                    count += 3;
                }
            }
        }

        if count > 0 {
            (1.0 - diff_sum / count as f32 / 255.0).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Compute contrast and intensity variance.
    fn compute_contrast_variance(
        &self,
        rgb_data: &[u8],
        width: usize,
        rx: usize,
        ry: usize,
        rw: usize,
        rh: usize,
    ) -> (f32, f32) {
        let mut values = Vec::new();

        for y in ry..(ry + rh) {
            for x in rx..(rx + rw) {
                let idx = (y * width + x) * 3;
                if idx + 2 < rgb_data.len() {
                    let lum = 0.299 * rgb_data[idx] as f32
                        + 0.587 * rgb_data[idx + 1] as f32
                        + 0.114 * rgb_data[idx + 2] as f32;
                    values.push(lum);
                }
            }
        }

        if values.is_empty() {
            return (0.0, 0.0);
        }

        let min_val = values.iter().copied().fold(f32::MAX, f32::min);
        let max_val = values.iter().copied().fold(f32::MIN, f32::max);
        let contrast = (max_val - min_val) / 255.0;

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        (contrast, variance / (255.0 * 255.0))
    }

    /// Classify emotion from facial features using heuristic rules.
    fn classify_emotion(&self, features: &FacialFeatures) -> Vec<(Emotion, f32)> {
        let mut scores = Vec::new();

        // Neutral: symmetric, moderate contrast, low edge density
        let neutral_score = features.face_symmetry * 0.4
            + (1.0 - features.mouth_region_edges) * 0.3
            + (1.0 - features.brow_gradient.min(1.0)) * 0.3;
        scores.push((Emotion::Neutral, neutral_score.clamp(0.0, 1.0)));

        // Happy: high mouth edges (smile lines), high symmetry
        let happy_score = features.mouth_region_edges * 0.5
            + features.face_symmetry * 0.3
            + features.upper_brightness * 0.2;
        scores.push((Emotion::Happy, happy_score.clamp(0.0, 1.0)));

        // Sad: lower brightness in lower face, low mouth edges
        let sad_score = (1.0 - features.lower_brightness) * 0.4
            + (1.0 - features.mouth_region_edges) * 0.3
            + (1.0 - features.face_contrast) * 0.3;
        scores.push((Emotion::Sad, sad_score.clamp(0.0, 1.0)));

        // Angry: strong brow gradient (furrow), low symmetry
        let angry_score = features.brow_gradient.min(1.0) * 0.5
            + (1.0 - features.face_symmetry) * 0.3
            + features.face_contrast * 0.2;
        scores.push((Emotion::Angry, angry_score.clamp(0.0, 1.0)));

        // Surprised: high eye region edges (wide eyes), high contrast
        let surprised_score = features.eye_region_edges * 0.5
            + features.face_contrast * 0.3
            + features.mouth_region_edges * 0.2;
        scores.push((Emotion::Surprised, surprised_score.clamp(0.0, 1.0)));

        // Disgusted: asymmetric, moderate brow, moderate mouth
        let disgusted_score = (1.0 - features.face_symmetry) * 0.4
            + features.brow_gradient.min(1.0) * 0.3
            + features.mouth_region_edges * 0.3;
        scores.push((Emotion::Disgusted, disgusted_score.clamp(0.0, 1.0)));

        // Fearful: high variance, wide eyes
        let fearful_score = features.intensity_variance.min(1.0) * 0.4
            + features.eye_region_edges * 0.4
            + (1.0 - features.face_symmetry) * 0.2;
        scores.push((Emotion::Fearful, fearful_score.clamp(0.0, 1.0)));

        // Contemptuous: strong asymmetry, moderate mouth
        let contemptuous_score = (1.0 - features.face_symmetry) * 0.5
            + features.mouth_region_edges * 0.3
            + (1.0 - features.brow_gradient.min(1.0)) * 0.2;
        scores.push((Emotion::Contemptuous, contemptuous_score.clamp(0.0, 1.0)));

        // Normalize scores to sum to 1
        let total: f32 = scores.iter().map(|(_, s)| s).sum();
        if total > 0.0 {
            for entry in &mut scores {
                entry.1 /= total;
            }
        }

        scores
    }
}

impl Default for EmotionRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_labels() {
        assert_eq!(Emotion::Happy.label(), "happy");
        assert_eq!(Emotion::Neutral.label(), "neutral");
        assert_eq!(Emotion::Angry.label(), "angry");
    }

    #[test]
    fn test_emotion_all() {
        assert_eq!(Emotion::all().len(), 8);
    }

    #[test]
    fn test_emotion_valence_arousal() {
        assert!(Emotion::Happy.valence() > 0.0);
        assert!(Emotion::Sad.valence() < 0.0);
        assert!(Emotion::Angry.arousal() > 0.5);
        assert!(Emotion::Neutral.arousal() < 0.5);
    }

    #[test]
    fn test_recognize_basic() {
        let recognizer = EmotionRecognizer::new();
        let width = 200;
        let height = 200;
        let rgb_data = vec![128u8; width * height * 3];
        let face_bbox = Rect::new(50.0, 50.0, 100.0, 100.0);

        let result = recognizer.recognize(&rgb_data, width, height, &face_bbox);
        assert!(result.is_ok());
        let emotion = result.expect("should succeed");
        assert!(!emotion.scores.is_empty());
        assert!(emotion.confidence.value() >= 0.0);
        assert!(emotion.confidence.value() <= 1.0);
    }

    #[test]
    fn test_recognize_batch() {
        let recognizer = EmotionRecognizer::new();
        let width = 200;
        let height = 200;
        let rgb_data = vec![128u8; width * height * 3];
        let faces = vec![
            Rect::new(10.0, 10.0, 50.0, 50.0),
            Rect::new(100.0, 100.0, 50.0, 50.0),
        ];

        let result = recognizer.recognize_batch(&rgb_data, width, height, &faces);
        assert!(result.is_ok());
        let results = result.expect("should succeed");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_recognize_invalid_dimensions() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer.recognize(&[0u8; 10], 100, 100, &Rect::new(0.0, 0.0, 50.0, 50.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_recognize_tiny_face() {
        let recognizer = EmotionRecognizer::new();
        let width = 100;
        let height = 100;
        let rgb_data = vec![128u8; width * height * 3];
        let result = recognizer.recognize(&rgb_data, width, height, &Rect::new(0.0, 0.0, 1.0, 1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_scores_sum_to_one() {
        let recognizer = EmotionRecognizer::new();
        let width = 200;
        let height = 200;
        let rgb_data = vec![128u8; width * height * 3];
        let face_bbox = Rect::new(50.0, 50.0, 100.0, 100.0);

        let result = recognizer
            .recognize(&rgb_data, width, height, &face_bbox)
            .expect("should succeed");
        let total: f32 = result.scores.iter().map(|(_, s)| s).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn test_valence_arousal_computed() {
        let recognizer = EmotionRecognizer::new();
        let width = 200;
        let height = 200;
        let rgb_data = vec![128u8; width * height * 3];
        let face_bbox = Rect::new(50.0, 50.0, 100.0, 100.0);

        let result = recognizer
            .recognize(&rgb_data, width, height, &face_bbox)
            .expect("should succeed");
        // Valence and arousal should be computed
        assert!(result.valence.is_finite());
        assert!(result.arousal.is_finite());
    }

    #[test]
    fn test_custom_config() {
        let config = EmotionRecognizerConfig {
            confidence_threshold: 0.5,
            compute_valence_arousal: false,
        };
        let recognizer = EmotionRecognizer::with_config(config);
        let width = 200;
        let height = 200;
        let rgb_data = vec![128u8; width * height * 3];
        let face_bbox = Rect::new(50.0, 50.0, 100.0, 100.0);

        let result = recognizer
            .recognize(&rgb_data, width, height, &face_bbox)
            .expect("should succeed");
        // Without valence-arousal, they should be 0.0
        assert!((result.valence - 0.0).abs() < f32::EPSILON);
        assert!((result.arousal - 0.0).abs() < f32::EPSILON);
    }
}
