//! Scene segmentation: shot boundary detection, scene grouping, and transition types.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Type of transition between two shots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionKind {
    /// Immediate cut between shots.
    Cut,
    /// Gradual blend from one shot to the next.
    Dissolve,
    /// Luminance fades to black then back up.
    FadeThrough,
    /// Frame wiped away by an edge.
    Wipe,
    /// Flashing white / overexposed frame.
    Flash,
    /// Unknown or ambiguous transition.
    Unknown,
}

/// A single shot boundary within the video stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotBoundary {
    /// Frame index where the new shot begins.
    pub frame_index: u64,
    /// Type of transition.
    pub transition: TransitionKind,
    /// Score from the boundary detector (higher = more confident).
    pub score: f32,
}

/// A group of shots that together form a semantic scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGroup {
    /// Sequential index of this scene (0-based).
    pub scene_index: usize,
    /// Frame indices of all shots belonging to this scene.
    pub shot_frames: Vec<u64>,
    /// Estimated duration in seconds.
    pub duration_secs: f64,
}

impl SceneGroup {
    /// Number of shots in this scene.
    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.shot_frames.len()
    }
}

/// Statistics for a complete segmentation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationStats {
    /// Total number of frames analysed.
    pub total_frames: u64,
    /// Number of shot boundaries detected.
    pub shot_count: usize,
    /// Number of semantic scene groups formed.
    pub scene_count: usize,
    /// Average shots per scene.
    pub avg_shots_per_scene: f32,
}

/// Configuration for the segmentation algorithm.
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Minimum histogram-difference score to call a cut (0–1).
    pub cut_threshold: f32,
    /// Minimum histogram-difference score to call a dissolve (0–1).
    pub dissolve_threshold: f32,
    /// Minimum number of shots required to form a scene group.
    pub min_shots_per_scene: usize,
    /// Target frame rate used to convert frame counts to seconds.
    pub fps: f64,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            cut_threshold: 0.4,
            dissolve_threshold: 0.15,
            min_shots_per_scene: 1,
            fps: 25.0,
        }
    }
}

/// Shot-boundary detector and scene grouper.
#[derive(Debug)]
pub struct SceneSegmenter {
    config: SegmentationConfig,
}

impl Default for SceneSegmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneSegmenter {
    /// Create a segmenter with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SegmentationConfig::default(),
        }
    }

    /// Create a segmenter with custom configuration.
    #[must_use]
    pub fn with_config(config: SegmentationConfig) -> Self {
        Self { config }
    }

    /// Detect shot boundaries from a sequence of per-frame histogram difference scores.
    ///
    /// `diff_scores` is a slice where `diff_scores[i]` represents the histogram
    /// difference between frame `i` and frame `i+1`.
    #[must_use]
    pub fn detect_boundaries(&self, diff_scores: &[f32]) -> Vec<ShotBoundary> {
        let mut boundaries = Vec::new();
        for (i, &score) in diff_scores.iter().enumerate() {
            let frame_index = (i + 1) as u64;
            if score >= self.config.cut_threshold {
                boundaries.push(ShotBoundary {
                    frame_index,
                    transition: TransitionKind::Cut,
                    score,
                });
            } else if score >= self.config.dissolve_threshold {
                boundaries.push(ShotBoundary {
                    frame_index,
                    transition: TransitionKind::Dissolve,
                    score,
                });
            }
        }
        boundaries
    }

    /// Group detected shot boundaries into semantic scenes.
    #[must_use]
    pub fn group_into_scenes(
        &self,
        boundaries: &[ShotBoundary],
        total_frames: u64,
    ) -> Vec<SceneGroup> {
        if boundaries.is_empty() {
            let duration = total_frames as f64 / self.config.fps;
            return vec![SceneGroup {
                scene_index: 0,
                shot_frames: vec![0],
                duration_secs: duration,
            }];
        }

        let mut groups: Vec<SceneGroup> = Vec::new();
        let mut current_shots: Vec<u64> = vec![0];
        let mut scene_start_frame: u64 = 0;

        for boundary in boundaries {
            current_shots.push(boundary.frame_index);
            // New scene every time we hit a hard cut.
            if boundary.transition == TransitionKind::Cut
                && current_shots.len() >= self.config.min_shots_per_scene
            {
                let duration = (boundary.frame_index - scene_start_frame) as f64 / self.config.fps;
                groups.push(SceneGroup {
                    scene_index: groups.len(),
                    shot_frames: current_shots.clone(),
                    duration_secs: duration,
                });
                scene_start_frame = boundary.frame_index;
                current_shots.clear();
                current_shots.push(boundary.frame_index);
            }
        }

        // Flush remaining shots
        if !current_shots.is_empty() {
            let duration = (total_frames - scene_start_frame) as f64 / self.config.fps;
            groups.push(SceneGroup {
                scene_index: groups.len(),
                shot_frames: current_shots,
                duration_secs: duration,
            });
        }

        groups
    }

    /// Produce summary statistics for a segmentation run.
    #[must_use]
    pub fn compute_stats(
        &self,
        boundaries: &[ShotBoundary],
        scenes: &[SceneGroup],
        total_frames: u64,
    ) -> SegmentationStats {
        let avg = if scenes.is_empty() {
            0.0
        } else {
            let total_shots: usize = scenes.iter().map(SceneGroup::shot_count).sum();
            total_shots as f32 / scenes.len() as f32
        };
        SegmentationStats {
            total_frames,
            shot_count: boundaries.len(),
            scene_count: scenes.len(),
            avg_shots_per_scene: avg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diff_scores() -> Vec<f32> {
        // frame 2 → cut, frame 5 → dissolve, frame 9 → cut
        vec![0.1, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.6, 0.0]
    }

    #[test]
    fn test_default_config() {
        let cfg = SegmentationConfig::default();
        assert!((cfg.cut_threshold - 0.4).abs() < f32::EPSILON);
        assert!((cfg.fps - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segmenter_new() {
        let s = SceneSegmenter::new();
        assert!((s.config.cut_threshold - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_boundaries_count() {
        let s = SceneSegmenter::new();
        let scores = make_diff_scores();
        let boundaries = s.detect_boundaries(&scores);
        // Cuts at index 1 (score 0.5) and 7 (score 0.6), dissolve at 4 (score 0.2)
        assert_eq!(boundaries.len(), 3);
    }

    #[test]
    fn test_detect_cuts_have_correct_transition() {
        let s = SceneSegmenter::new();
        let scores = vec![0.0, 0.5, 0.0];
        let b = s.detect_boundaries(&scores);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].transition, TransitionKind::Cut);
    }

    #[test]
    fn test_detect_dissolve_transition() {
        let s = SceneSegmenter::new();
        let scores = vec![0.0, 0.2, 0.0];
        let b = s.detect_boundaries(&scores);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].transition, TransitionKind::Dissolve);
    }

    #[test]
    fn test_detect_no_boundaries_below_threshold() {
        let s = SceneSegmenter::new();
        let scores = vec![0.0, 0.1, 0.05];
        let b = s.detect_boundaries(&scores);
        assert!(b.is_empty());
    }

    #[test]
    fn test_group_into_scenes_no_boundaries() {
        let s = SceneSegmenter::new();
        let scenes = s.group_into_scenes(&[], 100);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].scene_index, 0);
    }

    #[test]
    fn test_group_into_scenes_with_boundaries() {
        let s = SceneSegmenter::new();
        let scores = make_diff_scores();
        let b = s.detect_boundaries(&scores);
        let scenes = s.group_into_scenes(&b, 100);
        assert!(!scenes.is_empty());
    }

    #[test]
    fn test_scene_group_shot_count() {
        let sg = SceneGroup {
            scene_index: 0,
            shot_frames: vec![0, 10, 20],
            duration_secs: 2.0,
        };
        assert_eq!(sg.shot_count(), 3);
    }

    #[test]
    fn test_stats_total_frames() {
        let s = SceneSegmenter::new();
        let scores = make_diff_scores();
        let b = s.detect_boundaries(&scores);
        let scenes = s.group_into_scenes(&b, 500);
        let stats = s.compute_stats(&b, &scenes, 500);
        assert_eq!(stats.total_frames, 500);
    }

    #[test]
    fn test_stats_shot_count_matches_boundaries() {
        let s = SceneSegmenter::new();
        let scores = make_diff_scores();
        let b = s.detect_boundaries(&scores);
        let scenes = s.group_into_scenes(&b, 100);
        let stats = s.compute_stats(&b, &scenes, 100);
        assert_eq!(stats.shot_count, b.len());
    }

    #[test]
    fn test_stats_avg_shots_positive() {
        let s = SceneSegmenter::new();
        let scores = make_diff_scores();
        let b = s.detect_boundaries(&scores);
        let scenes = s.group_into_scenes(&b, 100);
        let stats = s.compute_stats(&b, &scenes, 100);
        assert!(stats.avg_shots_per_scene > 0.0);
    }

    #[test]
    fn test_with_config() {
        let cfg = SegmentationConfig {
            cut_threshold: 0.5,
            dissolve_threshold: 0.2,
            min_shots_per_scene: 2,
            fps: 30.0,
        };
        let s = SceneSegmenter::with_config(cfg);
        assert!((s.config.fps - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scene_duration_positive() {
        let s = SceneSegmenter::new();
        let scenes = s.group_into_scenes(&[], 250);
        assert!(scenes[0].duration_secs > 0.0);
    }

    #[test]
    fn test_transition_kinds_are_distinct() {
        assert_ne!(TransitionKind::Cut, TransitionKind::Dissolve);
        assert_ne!(TransitionKind::Wipe, TransitionKind::Flash);
    }
}
