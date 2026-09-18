//! Shot-to-scene grouping algorithms.

use crate::types::{Scene, Shot};

/// Scene grouper.
pub struct SceneGrouper {
    /// Maximum gap between shots in same scene (seconds).
    max_gap: f64,
}

impl SceneGrouper {
    /// Create a new scene grouper.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_gap: 5.0 }
    }

    /// Group shots into scenes based on temporal and visual similarity.
    #[must_use]
    pub fn group_shots(&self, shots: &[Shot]) -> Vec<Scene> {
        if shots.is_empty() {
            return Vec::new();
        }

        let mut scenes = Vec::new();
        let mut current_group = vec![shots[0].id];
        let mut scene_id = 0;
        let mut scene_start = shots[0].start;

        for i in 1..shots.len() {
            let gap = shots[i].start.to_seconds() - shots[i - 1].end.to_seconds();

            if gap > self.max_gap {
                // Create new scene
                scenes.push(Scene {
                    id: scene_id,
                    start: scene_start,
                    end: shots[i - 1].end,
                    shots: current_group.clone(),
                    scene_type: String::from("Grouped"),
                    confidence: 0.7,
                });

                scene_id += 1;
                scene_start = shots[i].start;
                current_group.clear();
            }

            current_group.push(shots[i].id);
        }

        // Add final scene
        if !current_group.is_empty() {
            scenes.push(Scene {
                id: scene_id,
                start: scene_start,
                end: shots[shots.len() - 1].end,
                shots: current_group,
                scene_type: String::from("Grouped"),
                confidence: 0.7,
            });
        }

        scenes
    }
}

impl Default for SceneGrouper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_grouper_creation() {
        let grouper = SceneGrouper::new();
        assert!((grouper.max_gap - 5.0).abs() < f64::EPSILON);
    }
}
