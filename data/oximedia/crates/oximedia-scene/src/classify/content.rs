//! Content type classification (sports, news, drama, etc.).

use crate::common::Confidence;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Type of video content detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    /// Sports content.
    Sports,
    /// News or documentary.
    News,
    /// Drama or narrative content.
    Drama,
    /// Action or fast-paced content.
    Action,
    /// Animation or cartoon.
    Animation,
    /// Music video or concert.
    Music,
    /// Static or slideshow content.
    Static,
    /// Talking head (interview, vlog).
    TalkingHead,
    /// Unknown content type.
    Unknown,
}

impl ContentType {
    /// Get all content types.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Sports,
            Self::News,
            Self::Drama,
            Self::Action,
            Self::Animation,
            Self::Music,
            Self::Static,
            Self::TalkingHead,
            Self::Unknown,
        ]
    }

    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sports => "Sports",
            Self::News => "News",
            Self::Drama => "Drama",
            Self::Action => "Action",
            Self::Animation => "Animation",
            Self::Music => "Music",
            Self::Static => "Static",
            Self::TalkingHead => "Talking Head",
            Self::Unknown => "Unknown",
        }
    }
}

/// Content classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentClassification {
    /// Primary content type.
    pub content_type: ContentType,
    /// Confidence score.
    pub confidence: Confidence,
    /// Scores for all content types.
    pub scores: Vec<(ContentType, f32)>,
    /// Features used for classification.
    pub features: ContentFeatures,
}

/// Features extracted for content classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFeatures {
    /// Motion intensity (0.0-1.0).
    pub motion_intensity: f32,
    /// Motion uniformity (0.0-1.0).
    pub motion_uniformity: f32,
    /// Color diversity (0.0-1.0).
    pub color_diversity: f32,
    /// Edge density (0.0-1.0).
    pub edge_density: f32,
    /// Temporal stability (0.0-1.0).
    pub temporal_stability: f32,
    /// Face presence probability (0.0-1.0).
    pub face_presence: f32,
    /// Text presence probability (0.0-1.0).
    pub text_presence: f32,
}

impl Default for ContentFeatures {
    fn default() -> Self {
        Self {
            motion_intensity: 0.0,
            motion_uniformity: 0.5,
            color_diversity: 0.5,
            edge_density: 0.5,
            temporal_stability: 0.5,
            face_presence: 0.0,
            text_presence: 0.0,
        }
    }
}

/// Content classifier using motion and temporal analysis.
pub struct ContentClassifier {
    min_frames: usize,
}

impl ContentClassifier {
    /// Create a new content classifier.
    #[must_use]
    pub fn new() -> Self {
        Self { min_frames: 3 }
    }

    /// Classify content from a sequence of frames.
    ///
    /// # Arguments
    ///
    /// * `frames` - Sequence of RGB frame data
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Errors
    ///
    /// Returns error if insufficient frames or invalid dimensions.
    pub fn classify(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<ContentClassification> {
        if frames.len() < self.min_frames {
            return Err(SceneError::InsufficientData(format!(
                "Need at least {} frames, got {}",
                self.min_frames,
                frames.len()
            )));
        }

        for frame in frames {
            if frame.len() != width * height * 3 {
                return Err(SceneError::InvalidDimensions(
                    "Frame size mismatch".to_string(),
                ));
            }
        }

        // Extract temporal features
        let features = self.extract_features(frames, width, height)?;

        // Compute scores for each content type
        let mut scores = Vec::new();
        scores.push((ContentType::Sports, self.score_sports(&features)));
        scores.push((ContentType::News, self.score_news(&features)));
        scores.push((ContentType::Drama, self.score_drama(&features)));
        scores.push((ContentType::Action, self.score_action(&features)));
        scores.push((ContentType::Animation, self.score_animation(&features)));
        scores.push((ContentType::Music, self.score_music(&features)));
        scores.push((ContentType::Static, self.score_static(&features)));
        scores.push((ContentType::TalkingHead, self.score_talking_head(&features)));

        // Find highest score
        let (content_type, confidence) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((ContentType::Unknown, 0.0), |(t, s)| (*t, *s));

        Ok(ContentClassification {
            content_type,
            confidence: Confidence::new(confidence),
            scores,
            features,
        })
    }

    /// Extract content features from frame sequence.
    fn extract_features(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<ContentFeatures> {
        let mut motion_sum = 0.0;
        let mut motion_variance = 0.0;
        let pixel_count = width * height;

        // Calculate frame-to-frame differences
        for i in 1..frames.len() {
            let mut frame_diff = 0.0;
            for j in 0..pixel_count * 3 {
                let diff = (frames[i][j] as i32 - frames[i - 1][j] as i32).unsigned_abs() as f32;
                frame_diff += diff;
            }
            motion_sum += frame_diff;
        }

        let motion_intensity =
            (motion_sum / ((frames.len() - 1) as f32 * pixel_count as f32 * 255.0)).clamp(0.0, 1.0);

        // Calculate motion uniformity (lower variance = more uniform)
        let mean_motion = motion_sum / (frames.len() - 1) as f32;
        for i in 1..frames.len() {
            let mut frame_diff = 0.0;
            for j in 0..pixel_count * 3 {
                let diff = (frames[i][j] as i32 - frames[i - 1][j] as i32).unsigned_abs() as f32;
                frame_diff += diff;
            }
            motion_variance += (frame_diff - mean_motion).powi(2);
        }
        let motion_uniformity =
            (1.0 - (motion_variance.sqrt() / (pixel_count as f32 * 255.0))).clamp(0.0, 1.0);

        // Analyze first frame for static features
        let first_frame = frames[0];
        let color_diversity = self.calculate_color_diversity(first_frame, width, height);
        let edge_density = self.calculate_edge_density(first_frame, width, height);
        let face_presence = self.detect_face_regions(first_frame, width, height);
        let text_presence = self.detect_text_regions(first_frame, width, height);

        // Temporal stability
        let temporal_stability = (1.0 - motion_intensity).clamp(0.0, 1.0);

        Ok(ContentFeatures {
            motion_intensity,
            motion_uniformity,
            color_diversity,
            edge_density,
            temporal_stability,
            face_presence,
            text_presence,
        })
    }

    /// Calculate color diversity using histogram.
    fn calculate_color_diversity(&self, frame: &[u8], _width: usize, _height: usize) -> f32 {
        let mut histogram = vec![0u32; 256];
        for &pixel in frame.iter().step_by(3) {
            histogram[pixel as usize] += 1;
        }

        // Calculate entropy as measure of diversity
        let total = histogram.iter().sum::<u32>() as f32;
        let mut entropy = 0.0;
        for &count in &histogram {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        (entropy / 8.0).clamp(0.0, 1.0) // Normalize by max entropy
    }

    /// Calculate edge density using simple gradient.
    fn calculate_edge_density(&self, frame: &[u8], width: usize, height: usize) -> f32 {
        let mut edge_count = 0;
        let threshold = 30;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                let idx_right = (y * width + x + 1) * 3;
                let idx_down = ((y + 1) * width + x) * 3;

                for c in 0..3 {
                    let gx = (frame[idx_right + c] as i32 - frame[idx + c] as i32).unsigned_abs();
                    let gy = (frame[idx_down + c] as i32 - frame[idx + c] as i32).unsigned_abs();
                    if gx > threshold || gy > threshold {
                        edge_count += 1;
                        break;
                    }
                }
            }
        }

        (edge_count as f32 / ((width - 2) * (height - 2)) as f32).clamp(0.0, 1.0)
    }

    /// Detect face-like regions (very simple skin tone detection).
    fn detect_face_regions(&self, frame: &[u8], _width: usize, _height: usize) -> f32 {
        let mut skin_pixels = 0;
        for i in (0..frame.len()).step_by(3) {
            let r = frame[i];
            let g = frame[i + 1];
            let b = frame[i + 2];

            // Simple skin tone heuristic
            if r > 95 && g > 40 && b > 20 && r > g && r > b && r.abs_diff(g) > 15 {
                skin_pixels += 1;
            }
        }

        (skin_pixels as f32 / (frame.len() / 3) as f32).clamp(0.0, 1.0)
    }

    /// Detect text-like regions (high contrast edges).
    fn detect_text_regions(&self, frame: &[u8], width: usize, height: usize) -> f32 {
        let mut text_regions = 0;
        let threshold = 100;

        for y in 0..height.saturating_sub(10) {
            for x in 0..width.saturating_sub(10) {
                let mut high_contrast = 0;
                for dy in 0..10 {
                    for dx in 0..10 {
                        let idx = ((y + dy) * width + (x + dx)) * 3;
                        let idx_next = ((y + dy) * width + (x + dx + 1)) * 3;
                        if x + dx + 1 < width {
                            let diff = (frame[idx] as i32 - frame[idx_next] as i32).unsigned_abs();
                            if diff > threshold {
                                high_contrast += 1;
                            }
                        }
                    }
                }
                if high_contrast > 20 {
                    text_regions += 1;
                }
            }
        }

        (text_regions as f32 / ((width / 10) * (height / 10)) as f32).clamp(0.0, 1.0)
    }

    // Scoring functions for each content type
    fn score_sports(&self, features: &ContentFeatures) -> f32 {
        let mut score = 0.0;
        // High motion, high uniformity (camera follows action)
        score += features.motion_intensity * 0.4;
        score += features.motion_uniformity * 0.3;
        // Good color diversity (green field, etc.)
        score += features.color_diversity * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_news(&self, features: &ContentFeatures) -> f32 {
        let mut score = 0.0;
        // Low motion, face present, text present
        score += features.temporal_stability * 0.3;
        score += features.face_presence * 0.4;
        score += features.text_presence * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_drama(&self, features: &ContentFeatures) -> f32 {
        let mut score = 0.0;
        // Moderate motion, face present, good color diversity
        score += (1.0 - (features.motion_intensity - 0.3).abs() / 0.7) * 0.3;
        score += features.face_presence * 0.4;
        score += features.color_diversity * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_action(&self, features: &ContentFeatures) -> f32 {
        // Very high motion, high edge density
        (features.motion_intensity * 0.6 + features.edge_density * 0.4).clamp(0.0, 1.0)
    }

    fn score_animation(&self, features: &ContentFeatures) -> f32 {
        let mut score = 0.0;
        // High color diversity, high edge density, moderate motion
        score += features.color_diversity * 0.4;
        score += features.edge_density * 0.3;
        score += (1.0 - (features.motion_intensity - 0.4).abs() / 0.6) * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_music(&self, features: &ContentFeatures) -> f32 {
        let mut score = 0.0;
        // High motion, high color diversity, low uniformity (cuts)
        score += features.motion_intensity * 0.4;
        score += features.color_diversity * 0.3;
        score += (1.0 - features.motion_uniformity) * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_static(&self, features: &ContentFeatures) -> f32 {
        // Very low motion, high stability
        features.temporal_stability
    }

    fn score_talking_head(&self, features: &ContentFeatures) -> f32 {
        let mut score = 0.0;
        // High face presence, low motion, moderate stability
        score += features.face_presence * 0.6;
        score += features.temporal_stability * 0.4;
        score.clamp(0.0, 1.0)
    }
}

impl Default for ContentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_name() {
        assert_eq!(ContentType::Sports.name(), "Sports");
        assert_eq!(ContentType::News.name(), "News");
    }

    #[test]
    fn test_content_classifier() {
        let classifier = ContentClassifier::new();
        let width = 100;
        let height = 100;
        let frame1 = vec![0u8; width * height * 3];
        let frame2 = vec![128u8; width * height * 3];
        let frame3 = vec![255u8; width * height * 3];

        let frames = vec![&frame1[..], &frame2[..], &frame3[..]];
        let result = classifier.classify(&frames, width, height);
        assert!(result.is_ok());

        let classification = result.expect("should succeed in test");
        assert!(classification.confidence.value() > 0.0);
    }

    #[test]
    fn test_insufficient_frames() {
        let classifier = ContentClassifier::new();
        let frame = vec![0u8; 100 * 100 * 3];
        let frames = vec![&frame[..]];
        let result = classifier.classify(&frames, 100, 100);
        assert!(result.is_err());
    }
}
