//! General activity recognition from motion patterns.

use crate::common::Confidence;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Type of activity detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivityType {
    /// Walking.
    Walking,
    /// Running.
    Running,
    /// Sitting/stationary.
    Sitting,
    /// Standing.
    Standing,
    /// Jumping.
    Jumping,
    /// Waving.
    Waving,
    /// Dancing.
    Dancing,
    /// Cycling.
    Cycling,
    /// Driving.
    Driving,
    /// Unknown activity.
    Unknown,
}

impl ActivityType {
    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Walking => "Walking",
            Self::Running => "Running",
            Self::Sitting => "Sitting",
            Self::Standing => "Standing",
            Self::Jumping => "Jumping",
            Self::Waving => "Waving",
            Self::Dancing => "Dancing",
            Self::Cycling => "Cycling",
            Self::Driving => "Driving",
            Self::Unknown => "Unknown",
        }
    }
}

/// Recognized activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizedActivity {
    /// Activity type.
    pub activity: ActivityType,
    /// Detection confidence.
    pub confidence: Confidence,
    /// Motion features.
    pub features: MotionFeatures,
}

/// Motion features for activity recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionFeatures {
    /// Average motion magnitude (0.0-1.0).
    pub motion_magnitude: f32,
    /// Motion direction variance.
    pub direction_variance: f32,
    /// Motion periodicity (0.0-1.0).
    pub periodicity: f32,
    /// Vertical motion component.
    pub vertical_motion: f32,
    /// Horizontal motion component.
    pub horizontal_motion: f32,
}

/// Activity recognizer.
pub struct ActivityRecognizer {
    min_frames: usize,
}

impl ActivityRecognizer {
    /// Create a new activity recognizer.
    #[must_use]
    pub fn new() -> Self {
        Self { min_frames: 5 }
    }

    /// Recognize activity from frame sequence.
    ///
    /// # Errors
    ///
    /// Returns error if insufficient frames.
    pub fn recognize(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<RecognizedActivity> {
        if frames.len() < self.min_frames {
            return Err(SceneError::InsufficientData(format!(
                "Need at least {} frames",
                self.min_frames
            )));
        }

        // Extract motion features
        let features = self.extract_motion_features(frames, width, height)?;

        // Classify activity
        let (activity, confidence) = self.classify_activity(&features);

        Ok(RecognizedActivity {
            activity,
            confidence,
            features,
        })
    }

    /// Extract motion features from frames.
    fn extract_motion_features(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
    ) -> SceneResult<MotionFeatures> {
        let mut total_motion = 0.0;
        let mut vertical_motion = 0.0;
        let mut horizontal_motion = 0.0;
        let mut motion_magnitudes = Vec::new();

        for i in 1..frames.len() {
            let (h_motion, v_motion, magnitude) =
                self.compute_frame_motion(frames[i - 1], frames[i], width, height);

            total_motion += magnitude;
            horizontal_motion += h_motion;
            vertical_motion += v_motion;
            motion_magnitudes.push(magnitude);
        }

        let frame_count = (frames.len() - 1) as f32;
        let motion_magnitude = (total_motion / frame_count).clamp(0.0, 1.0);
        let avg_horizontal = horizontal_motion / frame_count;
        let avg_vertical = vertical_motion / frame_count;

        // Calculate motion variance for direction analysis
        let mean_magnitude = total_motion / frame_count;
        let direction_variance = motion_magnitudes
            .iter()
            .map(|&m| (m - mean_magnitude).powi(2))
            .sum::<f32>()
            / frame_count;

        // Detect periodicity using autocorrelation
        let periodicity = self.detect_periodicity(&motion_magnitudes);

        Ok(MotionFeatures {
            motion_magnitude,
            direction_variance,
            periodicity,
            vertical_motion: avg_vertical,
            horizontal_motion: avg_horizontal,
        })
    }

    /// Compute motion between two frames.
    fn compute_frame_motion(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
    ) -> (f32, f32, f32) {
        let block_size = 16;
        let mut h_motion = 0.0;
        let mut v_motion = 0.0;
        let mut total_diff = 0.0;
        let mut block_count = 0;

        for y in (0..height - block_size).step_by(block_size) {
            for x in (0..width - block_size).step_by(block_size) {
                let (bh, bv, bdiff) =
                    self.compute_block_motion(frame1, frame2, width, x, y, block_size);
                h_motion += bh;
                v_motion += bv;
                total_diff += bdiff;
                block_count += 1;
            }
        }

        if block_count > 0 {
            h_motion /= block_count as f32;
            v_motion /= block_count as f32;
            total_diff /= block_count as f32;
        }

        (h_motion, v_motion, total_diff)
    }

    /// Compute motion for a block.
    fn compute_block_motion(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        x: usize,
        y: usize,
        block_size: usize,
    ) -> (f32, f32, f32) {
        let mut best_diff = f32::MAX;
        let mut best_dx = 0;
        let mut best_dy = 0;
        let search_range = 8;

        // Block matching
        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && ny >= 0 {
                    let nx = nx as usize;
                    let ny = ny as usize;

                    if nx + block_size < width && ny + block_size < width {
                        let diff =
                            self.block_difference(frame1, frame2, width, x, y, nx, ny, block_size);

                        if diff < best_diff {
                            best_diff = diff;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }
            }
        }

        (
            best_dx as f32 / search_range as f32,
            best_dy as f32 / search_range as f32,
            (best_diff / (block_size * block_size) as f32 / 255.0).clamp(0.0, 1.0),
        )
    }

    /// Compute SAD between two blocks.
    fn block_difference(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        size: usize,
    ) -> f32 {
        let mut diff = 0.0;

        for dy in 0..size {
            for dx in 0..size {
                let idx1 = ((y1 + dy) * width + (x1 + dx)) * 3;
                let idx2 = ((y2 + dy) * width + (x2 + dx)) * 3;

                if idx1 + 2 < frame1.len() && idx2 + 2 < frame2.len() {
                    for c in 0..3 {
                        diff += (frame1[idx1 + c] as i32 - frame2[idx2 + c] as i32).unsigned_abs()
                            as f32;
                    }
                }
            }
        }

        diff
    }

    /// Detect periodicity in motion.
    fn detect_periodicity(&self, magnitudes: &[f32]) -> f32 {
        if magnitudes.len() < 4 {
            return 0.0;
        }

        // Simple autocorrelation for period detection
        let mut max_correlation: f32 = 0.0;

        for lag in 1..magnitudes.len() / 2 {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..magnitudes.len() - lag {
                correlation += magnitudes[i] * magnitudes[i + lag];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f32;
                max_correlation = max_correlation.max(correlation);
            }
        }

        (max_correlation / 1.0).clamp(0.0, 1.0)
    }

    /// Classify activity from features.
    fn classify_activity(&self, features: &MotionFeatures) -> (ActivityType, Confidence) {
        let mut scores = Vec::new();

        scores.push((ActivityType::Walking, self.score_walking(features)));
        scores.push((ActivityType::Running, self.score_running(features)));
        scores.push((ActivityType::Sitting, self.score_sitting(features)));
        scores.push((ActivityType::Standing, self.score_standing(features)));
        scores.push((ActivityType::Jumping, self.score_jumping(features)));
        scores.push((ActivityType::Waving, self.score_waving(features)));
        scores.push((ActivityType::Dancing, self.score_dancing(features)));
        scores.push((ActivityType::Cycling, self.score_cycling(features)));
        scores.push((ActivityType::Driving, self.score_driving(features)));

        let (activity, score) = scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((ActivityType::Unknown, 0.0));

        (activity, Confidence::new(score))
    }

    fn score_walking(&self, f: &MotionFeatures) -> f32 {
        let mut score: f32 = 0.0;
        if f.motion_magnitude > 0.2 && f.motion_magnitude < 0.5 {
            score += 0.4;
        }
        if f.periodicity > 0.3 {
            score += 0.3;
        }
        if f.horizontal_motion.abs() > f.vertical_motion.abs() {
            score += 0.3;
        }
        score.clamp(0.0, 1.0)
    }

    fn score_running(&self, f: &MotionFeatures) -> f32 {
        let mut score: f32 = 0.0;
        if f.motion_magnitude > 0.5 {
            score += 0.5;
        }
        if f.periodicity > 0.4 {
            score += 0.3;
        }
        if f.vertical_motion > 0.2 {
            score += 0.2;
        }
        score.clamp(0.0, 1.0)
    }

    fn score_sitting(&self, f: &MotionFeatures) -> f32 {
        if f.motion_magnitude < 0.1 {
            0.8
        } else {
            0.2
        }
    }

    fn score_standing(&self, f: &MotionFeatures) -> f32 {
        if f.motion_magnitude < 0.15 && f.motion_magnitude > 0.05 {
            0.7
        } else {
            0.3
        }
    }

    fn score_jumping(&self, f: &MotionFeatures) -> f32 {
        if f.vertical_motion > 0.5 && f.periodicity > 0.3 {
            0.8
        } else {
            0.2
        }
    }

    fn score_waving(&self, f: &MotionFeatures) -> f32 {
        if f.periodicity > 0.5 && f.motion_magnitude > 0.3 && f.motion_magnitude < 0.6 {
            0.7
        } else {
            0.2
        }
    }

    fn score_dancing(&self, f: &MotionFeatures) -> f32 {
        let mut score: f32 = 0.0;
        if f.motion_magnitude > 0.4 {
            score += 0.4;
        }
        if f.direction_variance > 0.1 {
            score += 0.3;
        }
        if f.periodicity > 0.2 {
            score += 0.3;
        }
        score.clamp(0.0, 1.0)
    }

    fn score_cycling(&self, f: &MotionFeatures) -> f32 {
        if f.periodicity > 0.5 && f.horizontal_motion.abs() > 0.3 {
            0.7
        } else {
            0.2
        }
    }

    fn score_driving(&self, f: &MotionFeatures) -> f32 {
        if f.motion_magnitude > 0.3 && f.periodicity < 0.2 {
            0.6
        } else {
            0.2
        }
    }
}

impl Default for ActivityRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_type_name() {
        assert_eq!(ActivityType::Walking.name(), "Walking");
        assert_eq!(ActivityType::Running.name(), "Running");
    }

    #[test]
    fn test_activity_recognizer() {
        let recognizer = ActivityRecognizer::new();
        // Use minimal frame dimensions to keep block-matching O(N) tractable.
        // 64×48 gives (64/16)*(48/16) = 4*3 = 12 blocks, each with 17*17 candidates
        // of 16*16*3 pixels — orders of magnitude less than 320×240.
        let width = 64;
        let height = 48;
        let frame = vec![128u8; width * height * 3];
        let frames: Vec<&[u8]> = (0..10).map(|_| &frame[..]).collect();

        let result = recognizer.recognize(&frames, width, height);
        assert!(result.is_ok());
    }
}
