//! Angle scoring system for automatic camera selection.

use super::SelectionCriteria;
use crate::{AngleId, Result};

/// Angle score
#[derive(Debug, Clone, Copy)]
pub struct AngleScore {
    /// Angle identifier
    pub angle: AngleId,
    /// Face detection score (0.0 to 1.0)
    pub face_score: f32,
    /// Composition quality score (0.0 to 1.0)
    pub composition_score: f32,
    /// Audio activity score (0.0 to 1.0)
    pub audio_score: f32,
    /// Motion detection score (0.0 to 1.0)
    pub motion_score: f32,
    /// Speaker detection score (0.0 to 1.0)
    pub speaker_score: f32,
    /// Total weighted score
    pub total_score: f32,
}

impl AngleScore {
    /// Create a new angle score
    #[must_use]
    pub fn new(angle: AngleId) -> Self {
        Self {
            angle,
            face_score: 0.0,
            composition_score: 0.0,
            audio_score: 0.0,
            motion_score: 0.0,
            speaker_score: 0.0,
            total_score: 0.0,
        }
    }

    /// Calculate total score with weights
    pub fn calculate_total(&mut self, weights: &ScoringWeights) {
        self.total_score = self.face_score * weights.face_weight
            + self.composition_score * weights.composition_weight
            + self.audio_score * weights.audio_weight
            + self.motion_score * weights.motion_weight
            + self.speaker_score * weights.speaker_weight;

        // Normalize to 0.0-1.0 range
        let total_weight = weights.face_weight
            + weights.composition_weight
            + weights.audio_weight
            + weights.motion_weight
            + weights.speaker_weight;

        if total_weight > 0.0 {
            self.total_score /= total_weight;
        }
    }
}

/// Scoring weights
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    /// Face detection weight
    pub face_weight: f32,
    /// Composition quality weight
    pub composition_weight: f32,
    /// Audio activity weight
    pub audio_weight: f32,
    /// Motion detection weight
    pub motion_weight: f32,
    /// Speaker detection weight
    pub speaker_weight: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            face_weight: 1.0,
            composition_weight: 0.8,
            audio_weight: 1.2,
            motion_weight: 0.6,
            speaker_weight: 1.5,
        }
    }
}

/// Angle scorer
#[derive(Debug)]
pub struct AngleScorer {
    /// Scoring weights
    weights: ScoringWeights,
    /// Face detection data (mock)
    face_data: Vec<FaceData>,
    /// Audio data (mock)
    audio_data: Vec<AudioData>,
    /// Motion data (mock)
    motion_data: Vec<MotionData>,
}

/// Face detection data
#[derive(Debug, Clone)]
struct FaceData {
    /// Number of faces detected
    face_count: usize,
    /// Average face size
    avg_face_size: f32,
    /// Face positions
    positions: Vec<(f32, f32)>,
}

/// Audio activity data
#[derive(Debug, Clone)]
struct AudioData {
    /// Audio level (dB)
    level_db: f32,
    /// Speech detected
    speech_detected: bool,
    /// Speaker identifier
    speaker_id: Option<usize>,
}

/// Motion detection data
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MotionData {
    /// Motion magnitude
    magnitude: f32,
    /// Motion vectors
    vectors: Vec<(f32, f32)>,
}

impl AngleScorer {
    /// Create a new angle scorer
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: ScoringWeights::default(),
            face_data: Vec::new(),
            audio_data: Vec::new(),
            motion_data: Vec::new(),
        }
    }

    /// Set scoring weights
    pub fn set_weights(&mut self, weights: ScoringWeights) {
        self.weights = weights;
    }

    /// Get scoring weights
    #[must_use]
    pub fn weights(&self) -> &ScoringWeights {
        &self.weights
    }

    /// Score an angle
    ///
    /// # Errors
    ///
    /// Returns an error if scoring fails
    pub fn score_angle(&self, angle: AngleId, criteria: &SelectionCriteria) -> Result<AngleScore> {
        let mut score = AngleScore::new(angle);

        if criteria.face_detection {
            score.face_score = self.score_face_detection(angle);
        }

        if criteria.composition_quality {
            score.composition_score = self.score_composition(angle);
        }

        if criteria.audio_activity {
            score.audio_score = self.score_audio(angle);
        }

        if criteria.motion_detection {
            score.motion_score = self.score_motion(angle);
        }

        if criteria.speaker_detection {
            score.speaker_score = self.score_speaker(angle);
        }

        score.calculate_total(&self.weights);

        Ok(score)
    }

    /// Score face detection
    fn score_face_detection(&self, angle: AngleId) -> f32 {
        if angle < self.face_data.len() {
            let data = &self.face_data[angle];
            let face_score = (data.face_count as f32 * 0.3).min(1.0);
            let size_score = (data.avg_face_size / 200.0).min(1.0);
            (face_score + size_score) / 2.0
        } else {
            0.5 // Default score
        }
    }

    /// Score composition quality
    fn score_composition(&self, angle: AngleId) -> f32 {
        if angle < self.face_data.len() {
            let data = &self.face_data[angle];
            // Simple rule of thirds check
            let mut composition_score = 0.0;
            for (x, y) in &data.positions {
                // Check if face is near rule of thirds intersections
                let x_thirds = (x - 0.33).abs().min((x - 0.67).abs());
                let y_thirds = (y - 0.33).abs().min((y - 0.67).abs());
                let distance = (x_thirds * x_thirds + y_thirds * y_thirds).sqrt();
                composition_score += (1.0 - distance).max(0.0);
            }
            if data.positions.is_empty() {
                0.5
            } else {
                composition_score / data.positions.len() as f32
            }
        } else {
            0.5
        }
    }

    /// Score audio activity
    fn score_audio(&self, angle: AngleId) -> f32 {
        if angle < self.audio_data.len() {
            let data = &self.audio_data[angle];
            let level_score = ((data.level_db + 60.0) / 60.0).clamp(0.0, 1.0);
            let speech_score = if data.speech_detected { 1.0 } else { 0.0 };
            (level_score + speech_score) / 2.0
        } else {
            0.3
        }
    }

    /// Score motion
    fn score_motion(&self, angle: AngleId) -> f32 {
        if angle < self.motion_data.len() {
            let data = &self.motion_data[angle];
            data.magnitude.min(1.0)
        } else {
            0.3
        }
    }

    /// Score speaker detection
    fn score_speaker(&self, angle: AngleId) -> f32 {
        if angle < self.audio_data.len() {
            let data = &self.audio_data[angle];
            if data.speech_detected && data.speaker_id.is_some() {
                1.0
            } else if data.speech_detected {
                0.7
            } else {
                0.0
            }
        } else {
            0.3
        }
    }

    /// Add face detection data for an angle
    pub fn add_face_data(
        &mut self,
        angle: AngleId,
        face_count: usize,
        avg_size: f32,
        positions: Vec<(f32, f32)>,
    ) {
        while self.face_data.len() <= angle {
            self.face_data.push(FaceData {
                face_count: 0,
                avg_face_size: 0.0,
                positions: Vec::new(),
            });
        }
        self.face_data[angle] = FaceData {
            face_count,
            avg_face_size: avg_size,
            positions,
        };
    }

    /// Add audio data for an angle
    pub fn add_audio_data(
        &mut self,
        angle: AngleId,
        level_db: f32,
        speech: bool,
        speaker_id: Option<usize>,
    ) {
        while self.audio_data.len() <= angle {
            self.audio_data.push(AudioData {
                level_db: -60.0,
                speech_detected: false,
                speaker_id: None,
            });
        }
        self.audio_data[angle] = AudioData {
            level_db,
            speech_detected: speech,
            speaker_id,
        };
    }

    /// Add motion data for an angle
    pub fn add_motion_data(&mut self, angle: AngleId, magnitude: f32, vectors: Vec<(f32, f32)>) {
        while self.motion_data.len() <= angle {
            self.motion_data.push(MotionData {
                magnitude: 0.0,
                vectors: Vec::new(),
            });
        }
        self.motion_data[angle] = MotionData { magnitude, vectors };
    }

    /// Clear all scoring data
    pub fn clear_data(&mut self) {
        self.face_data.clear();
        self.audio_data.clear();
        self.motion_data.clear();
    }
}

impl Default for AngleScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoring criteria
#[derive(Debug, Clone)]
pub struct ScoringCriteria {
    /// Weights for different factors
    pub weights: ScoringWeights,
    /// Minimum face size for detection
    pub min_face_size: f32,
    /// Minimum audio level (dB)
    pub min_audio_level: f32,
    /// Minimum motion magnitude
    pub min_motion: f32,
}

impl Default for ScoringCriteria {
    fn default() -> Self {
        Self {
            weights: ScoringWeights::default(),
            min_face_size: 50.0,
            min_audio_level: -40.0,
            min_motion: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_score_creation() {
        let score = AngleScore::new(0);
        assert_eq!(score.angle, 0);
        assert_eq!(score.total_score, 0.0);
    }

    #[test]
    fn test_calculate_total() {
        let mut score = AngleScore::new(0);
        score.face_score = 0.8;
        score.composition_score = 0.6;
        score.audio_score = 0.9;

        let weights = ScoringWeights::default();
        score.calculate_total(&weights);

        assert!(score.total_score > 0.0);
        assert!(score.total_score <= 1.0);
    }

    #[test]
    fn test_scorer_creation() {
        let scorer = AngleScorer::new();
        assert!(scorer.face_data.is_empty());
        assert!(scorer.audio_data.is_empty());
    }

    #[test]
    fn test_add_face_data() {
        let mut scorer = AngleScorer::new();
        scorer.add_face_data(0, 2, 150.0, vec![(0.5, 0.5)]);
        assert_eq!(scorer.face_data.len(), 1);
        assert_eq!(scorer.face_data[0].face_count, 2);
    }

    #[test]
    fn test_add_audio_data() {
        let mut scorer = AngleScorer::new();
        scorer.add_audio_data(0, -20.0, true, Some(1));
        assert_eq!(scorer.audio_data.len(), 1);
        assert!(scorer.audio_data[0].speech_detected);
    }

    #[test]
    fn test_add_motion_data() {
        let mut scorer = AngleScorer::new();
        scorer.add_motion_data(0, 0.7, vec![(1.0, 0.5)]);
        assert_eq!(scorer.motion_data.len(), 1);
        assert_eq!(scorer.motion_data[0].magnitude, 0.7);
    }

    #[test]
    fn test_clear_data() {
        let mut scorer = AngleScorer::new();
        scorer.add_face_data(0, 1, 100.0, vec![]);
        scorer.clear_data();
        assert!(scorer.face_data.is_empty());
    }

    #[test]
    fn test_default_weights() {
        let weights = ScoringWeights::default();
        assert!(weights.speaker_weight > weights.audio_weight);
        assert!(weights.audio_weight > weights.composition_weight);
    }
}
