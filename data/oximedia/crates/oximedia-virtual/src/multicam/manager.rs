//! Multi-camera coordination manager
//!
//! Provides multi-camera management with automatic camera selection
//! based on talent tracking position.

use super::{CameraId, MultiCameraState};
use crate::math::{Point3, Vector3};
use crate::{tracking::CameraPose, Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Multi-camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCameraConfig {
    /// Number of cameras
    pub num_cameras: usize,
    /// Enable auto-switching
    pub auto_switch: bool,
}

impl Default for MultiCameraConfig {
    fn default() -> Self {
        Self {
            num_cameras: 1,
            auto_switch: false,
        }
    }
}

/// Criteria used for automatic camera selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoSwitchCriteria {
    /// Select the camera whose optical axis is most aligned with the
    /// talent's position (smallest angle between forward vector and
    /// direction to talent).
    BestAngle,
    /// Select the camera closest to the talent.
    NearestDistance,
    /// Select based on a weighted score combining angle and distance.
    /// The score = (1 - w) * normalized_angle + w * normalized_distance
    /// where w is the `distance_weight` in `AutoSwitchConfig`.
    WeightedScore,
    /// Select the camera that has the talent most centered in its
    /// field of view (closest to optical axis in screen space).
    CenteredFraming,
}

/// Configuration for automatic camera selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoSwitchConfig {
    /// Selection criteria.
    pub criteria: AutoSwitchCriteria,
    /// Minimum time between automatic switches (milliseconds).
    /// Prevents rapid ping-ponging between cameras.
    pub min_switch_interval_ms: u64,
    /// Hysteresis threshold: a new camera must score at least this
    /// much better (as a fraction, e.g. 0.1 = 10%) than the current
    /// camera to trigger a switch.
    pub hysteresis: f64,
    /// Distance weight for `WeightedScore` criteria (0.0 to 1.0).
    pub distance_weight: f64,
    /// Camera horizontal field of view in radians (used for `CenteredFraming`).
    pub camera_fov_h: f64,
}

impl Default for AutoSwitchConfig {
    fn default() -> Self {
        Self {
            criteria: AutoSwitchCriteria::BestAngle,
            min_switch_interval_ms: 2000,
            hysteresis: 0.15,
            distance_weight: 0.3,
            camera_fov_h: std::f64::consts::PI / 3.0, // 60 degrees
        }
    }
}

/// Result of evaluating a camera for talent coverage.
#[derive(Debug, Clone, Copy)]
pub struct CameraScore {
    /// Camera identifier.
    pub camera_id: CameraId,
    /// Angle between camera forward and direction to talent (radians).
    pub angle_to_talent: f64,
    /// Distance from camera to talent (meters).
    pub distance_to_talent: f64,
    /// Normalized score (0.0 = best, 1.0 = worst). Lower is better.
    pub score: f64,
    /// Whether the talent is within the camera's field of view.
    pub in_fov: bool,
}

/// Multi-camera manager
pub struct MultiCameraManager {
    config: MultiCameraConfig,
    state: MultiCameraState,
    /// Auto-switch configuration (used when auto_switch is enabled).
    auto_switch_config: AutoSwitchConfig,
    /// Timestamp (ns) of the last auto-switch. `None` if no switch has occurred.
    last_switch_timestamp_ns: Option<u64>,
    /// History of automatic switches for diagnostics.
    switch_history: Vec<SwitchEvent>,
}

/// Record of a camera switch event.
#[derive(Debug, Clone)]
pub struct SwitchEvent {
    /// Timestamp of the switch.
    pub timestamp_ns: u64,
    /// Camera switched from.
    pub from: CameraId,
    /// Camera switched to.
    pub to: CameraId,
    /// Score of the selected camera.
    pub score: f64,
    /// Reason for the switch.
    pub reason: String,
}

impl MultiCameraManager {
    /// Create new multi-camera manager
    pub fn new(config: MultiCameraConfig) -> Result<Self> {
        if config.num_cameras == 0 {
            return Err(VirtualProductionError::MultiCamera(
                "Number of cameras must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config,
            state: MultiCameraState::new(),
            auto_switch_config: AutoSwitchConfig::default(),
            last_switch_timestamp_ns: None,
            switch_history: Vec::new(),
        })
    }

    /// Create with auto-switch configuration.
    pub fn with_auto_switch(
        config: MultiCameraConfig,
        auto_switch_config: AutoSwitchConfig,
    ) -> Result<Self> {
        if config.num_cameras == 0 {
            return Err(VirtualProductionError::MultiCamera(
                "Number of cameras must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config: MultiCameraConfig {
                auto_switch: true,
                ..config
            },
            state: MultiCameraState::new(),
            auto_switch_config,
            last_switch_timestamp_ns: None,
            switch_history: Vec::new(),
        })
    }

    /// Update camera pose
    pub fn update_camera(&mut self, camera_id: CameraId, pose: CameraPose) {
        if let Some(entry) = self.state.poses.iter_mut().find(|(id, _)| *id == camera_id) {
            entry.1 = pose;
        } else {
            self.state.poses.push((camera_id, pose));
        }
    }

    /// Set active camera
    pub fn set_active_camera(&mut self, camera_id: CameraId) {
        self.state.active_camera = camera_id;
    }

    /// Get active camera
    #[must_use]
    pub fn active_camera(&self) -> CameraId {
        self.state.active_camera
    }

    /// Get active camera pose
    #[must_use]
    pub fn active_pose(&self) -> Option<&CameraPose> {
        self.state.active_pose()
    }

    /// Get all camera poses
    #[must_use]
    pub fn all_poses(&self) -> &[(CameraId, CameraPose)] {
        &self.state.poses
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MultiCameraConfig {
        &self.config
    }

    /// Get the auto-switch configuration.
    #[must_use]
    pub fn auto_switch_config(&self) -> &AutoSwitchConfig {
        &self.auto_switch_config
    }

    /// Set the auto-switch configuration.
    pub fn set_auto_switch_config(&mut self, config: AutoSwitchConfig) {
        self.auto_switch_config = config;
    }

    /// Evaluate all cameras and score them for a given talent position.
    ///
    /// Returns scores sorted best-first (lowest score first).
    #[must_use]
    pub fn evaluate_cameras(&self, talent_position: &Point3<f64>) -> Vec<CameraScore> {
        let mut scores: Vec<CameraScore> = self
            .state
            .poses
            .iter()
            .map(|(camera_id, pose)| self.score_camera(*camera_id, pose, talent_position))
            .collect();

        scores.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores
    }

    /// Automatically select the best camera for the given talent position.
    ///
    /// Applies hysteresis to avoid rapid switching. Returns `Some(CameraId)`
    /// if a switch is recommended, `None` if the current camera is still best
    /// (or the switch interval hasn't elapsed).
    pub fn auto_select(
        &mut self,
        talent_position: &Point3<f64>,
        current_timestamp_ns: u64,
    ) -> Option<CameraId> {
        if !self.config.auto_switch {
            return None;
        }

        if self.state.poses.is_empty() {
            return None;
        }

        // Check minimum switch interval (skip check if no switch has occurred yet)
        if let Some(last_ts) = self.last_switch_timestamp_ns {
            let elapsed_ns = current_timestamp_ns.saturating_sub(last_ts);
            let min_interval_ns = self.auto_switch_config.min_switch_interval_ms * 1_000_000;
            if elapsed_ns < min_interval_ns {
                return None;
            }
        }

        let scores = self.evaluate_cameras(talent_position);
        if scores.is_empty() {
            return None;
        }

        let best = &scores[0];
        let current_id = self.state.active_camera;

        // If best is already active, no switch needed
        if best.camera_id == current_id {
            return None;
        }

        // Find current camera's score
        let current_score = scores
            .iter()
            .find(|s| s.camera_id == current_id)
            .map(|s| s.score)
            .unwrap_or(f64::MAX);

        // Hysteresis: only switch if the improvement exceeds the threshold
        let improvement = if current_score > 1e-10 {
            (current_score - best.score) / current_score
        } else {
            1.0
        };

        if improvement < self.auto_switch_config.hysteresis {
            return None;
        }

        // Perform the switch
        let previous_camera = self.state.active_camera;
        self.state.active_camera = best.camera_id;
        self.last_switch_timestamp_ns = Some(current_timestamp_ns);

        self.switch_history.push(SwitchEvent {
            timestamp_ns: current_timestamp_ns,
            from: previous_camera,
            to: best.camera_id,
            score: best.score,
            reason: format!(
                "{:?}: improvement {:.1}%",
                self.auto_switch_config.criteria,
                improvement * 100.0
            ),
        });

        Some(best.camera_id)
    }

    /// Get the switch history.
    #[must_use]
    pub fn switch_history(&self) -> &[SwitchEvent] {
        &self.switch_history
    }

    /// Clear switch history.
    pub fn clear_switch_history(&mut self) {
        self.switch_history.clear();
    }

    // -----------------------------------------------------------------------
    // Internal scoring
    // -----------------------------------------------------------------------

    fn score_camera(
        &self,
        camera_id: CameraId,
        pose: &CameraPose,
        talent_position: &Point3<f64>,
    ) -> CameraScore {
        let cam_pos = pose.position;
        let direction_to_talent = Vector3::new(
            talent_position.x - cam_pos.x,
            talent_position.y - cam_pos.y,
            talent_position.z - cam_pos.z,
        );

        let distance = direction_to_talent.norm();
        let dir_normalized = if distance > 1e-10 {
            Vector3::new(
                direction_to_talent.x / distance,
                direction_to_talent.y / distance,
                direction_to_talent.z / distance,
            )
        } else {
            Vector3::new(0.0, 0.0, -1.0)
        };

        // Camera forward vector
        let forward = pose.forward();

        // Angle between forward and direction to talent
        let cos_angle = forward.x * dir_normalized.x
            + forward.y * dir_normalized.y
            + forward.z * dir_normalized.z;
        let angle = cos_angle.clamp(-1.0, 1.0).acos();

        let in_fov = angle < self.auto_switch_config.camera_fov_h * 0.5;

        let score = match self.auto_switch_config.criteria {
            AutoSwitchCriteria::BestAngle => {
                // Normalize angle to [0, 1] where 0 is best
                angle / std::f64::consts::PI
            }
            AutoSwitchCriteria::NearestDistance => {
                // Normalize distance; assume max reasonable distance is 20m
                (distance / 20.0).min(1.0)
            }
            AutoSwitchCriteria::WeightedScore => {
                let w = self.auto_switch_config.distance_weight;
                let angle_norm = angle / std::f64::consts::PI;
                let dist_norm = (distance / 20.0).min(1.0);
                (1.0 - w) * angle_norm + w * dist_norm
            }
            AutoSwitchCriteria::CenteredFraming => {
                // Score based on how centered the talent is in the FOV
                if !in_fov {
                    1.0 // Worst score if out of FOV
                } else {
                    let half_fov = self.auto_switch_config.camera_fov_h * 0.5;
                    if half_fov > 1e-10 {
                        angle / half_fov
                    } else {
                        0.0
                    }
                }
            }
        };

        CameraScore {
            camera_id,
            angle_to_talent: angle,
            distance_to_talent: distance,
            score,
            in_fov,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::UnitQuaternion;

    #[test]
    fn test_multicam_manager() {
        let config = MultiCameraConfig {
            num_cameras: 4,
            auto_switch: false,
        };
        let manager = MultiCameraManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_multicam_update() {
        let config = MultiCameraConfig::default();
        let mut manager = MultiCameraManager::new(config).expect("should succeed in test");

        let pose = CameraPose::new(Point3::origin(), UnitQuaternion::identity(), 0);

        manager.update_camera(CameraId(0), pose);
        assert!(manager.active_pose().is_some());
    }

    #[test]
    fn test_multicam_switch() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: false,
        };
        let mut manager = MultiCameraManager::new(config).expect("should succeed in test");

        manager.set_active_camera(CameraId(1));
        assert_eq!(manager.active_camera(), CameraId(1));
    }

    // --- Auto camera selection tests ---

    fn make_camera_pose(x: f64, y: f64, z: f64, look_z: f64) -> CameraPose {
        // Camera at (x, y, z), looking along -Z by default
        let _ = look_z; // orientation is identity (looks along -Z)
        CameraPose::new(Point3::new(x, y, z), UnitQuaternion::identity(), 0)
    }

    #[test]
    fn test_evaluate_cameras_best_angle() {
        let config = MultiCameraConfig {
            num_cameras: 3,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::BestAngle,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        // Camera 0 at origin looking -Z
        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        // Camera 1 offset to the right
        manager.update_camera(CameraId(1), make_camera_pose(5.0, 0.0, 0.0, -1.0));
        // Camera 2 far away
        manager.update_camera(CameraId(2), make_camera_pose(10.0, 0.0, 0.0, -1.0));

        // Talent directly in front of camera 0 at (0, 0, -5)
        let talent_pos = Point3::new(0.0, 0.0, -5.0);
        let scores = manager.evaluate_cameras(&talent_pos);

        assert_eq!(scores.len(), 3);
        // Camera 0 should have the best (lowest) score - talent is on its axis
        assert_eq!(scores[0].camera_id, CameraId(0));
        assert!(
            scores[0].score < scores[1].score,
            "cam0 score {} should be < cam1 score {}",
            scores[0].score,
            scores[1].score
        );
    }

    #[test]
    fn test_evaluate_cameras_nearest_distance() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::NearestDistance,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(0.0, 0.0, -4.0, -1.0));

        // Talent at (0, 0, -5) - closer to camera 1
        let talent_pos = Point3::new(0.0, 0.0, -5.0);
        let scores = manager.evaluate_cameras(&talent_pos);

        assert_eq!(scores[0].camera_id, CameraId(1));
        assert!(
            scores[0].distance_to_talent < scores[1].distance_to_talent,
            "cam1 should be closer"
        );
    }

    #[test]
    fn test_auto_select_switches_camera() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::BestAngle,
                min_switch_interval_ms: 0, // allow immediate switching
                hysteresis: 0.05,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(5.0, 0.0, 0.0, -1.0));

        // Start with camera 0 active
        manager.set_active_camera(CameraId(0));

        // Talent moves to be directly in front of camera 1
        // Camera 1 at (5,0,0) looking -Z, talent at (5, 0, -5)
        let talent_pos = Point3::new(5.0, 0.0, -5.0);
        let result = manager.auto_select(&talent_pos, 1_000_000_000);

        // Should switch to camera 1
        assert_eq!(result, Some(CameraId(1)));
        assert_eq!(manager.active_camera(), CameraId(1));
    }

    #[test]
    fn test_auto_select_respects_min_interval() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::NearestDistance,
                min_switch_interval_ms: 2000,
                hysteresis: 0.0,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        // Camera 0 at origin, camera 1 at (10, 0, 0)
        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(10.0, 0.0, 0.0, -1.0));
        manager.set_active_camera(CameraId(0));

        // Talent right next to camera 1 => should switch to cam 1
        let talent_near_cam1 = Point3::new(10.0, 0.0, -1.0);
        let r1 = manager.auto_select(&talent_near_cam1, 0);
        assert!(r1.is_some(), "should switch to nearer camera");

        // Now talent moves back near cam 0, but interval not elapsed
        let talent_near_cam0 = Point3::new(0.0, 0.0, -1.0);
        let r2 = manager.auto_select(&talent_near_cam0, 500_000_000); // 0.5s later
        assert!(r2.is_none(), "should respect min switch interval");

        // After interval elapses, should be able to switch again
        let r3 = manager.auto_select(&talent_near_cam0, 3_000_000_000); // 3s later
        assert!(r3.is_some(), "should allow switch after interval");
    }

    #[test]
    fn test_auto_select_hysteresis() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::BestAngle,
                min_switch_interval_ms: 0,
                hysteresis: 0.5, // very high hysteresis
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(1.0, 0.0, 0.0, -1.0));
        manager.set_active_camera(CameraId(0));

        // Talent slightly favors camera 1, but not by 50%
        let talent_pos = Point3::new(0.5, 0.0, -5.0);
        let result = manager.auto_select(&talent_pos, 1_000_000_000);

        // High hysteresis should prevent switching for a marginal improvement
        assert!(result.is_none(), "hysteresis should prevent switch");
    }

    #[test]
    fn test_auto_select_disabled() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: false,
        };
        let mut manager = MultiCameraManager::new(config).expect("should succeed in test");
        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(5.0, 0.0, 0.0, -1.0));

        let talent_pos = Point3::new(5.0, 0.0, -5.0);
        let result = manager.auto_select(&talent_pos, 1_000_000_000);
        assert!(result.is_none(), "auto_select should be disabled");
    }

    #[test]
    fn test_switch_history() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::BestAngle,
                min_switch_interval_ms: 0,
                hysteresis: 0.0,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(5.0, 0.0, 0.0, -1.0));
        manager.set_active_camera(CameraId(0));

        let talent_pos = Point3::new(5.0, 0.0, -5.0);
        manager.auto_select(&talent_pos, 1_000_000_000);

        assert_eq!(manager.switch_history().len(), 1);
        assert_eq!(manager.switch_history()[0].from, CameraId(0));
        assert_eq!(manager.switch_history()[0].to, CameraId(1));

        manager.clear_switch_history();
        assert!(manager.switch_history().is_empty());
    }

    #[test]
    fn test_camera_score_in_fov() {
        let config = MultiCameraConfig {
            num_cameras: 1,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                camera_fov_h: std::f64::consts::PI / 3.0, // 60 deg
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));

        // Talent directly ahead - should be in FOV
        let scores_ahead = manager.evaluate_cameras(&Point3::new(0.0, 0.0, -5.0));
        assert!(scores_ahead[0].in_fov, "talent ahead should be in FOV");

        // Talent behind camera - should NOT be in FOV
        let scores_behind = manager.evaluate_cameras(&Point3::new(0.0, 0.0, 5.0));
        assert!(
            !scores_behind[0].in_fov,
            "talent behind should not be in FOV"
        );
    }

    #[test]
    fn test_weighted_score_criteria() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::WeightedScore,
                distance_weight: 0.5,
                min_switch_interval_ms: 0,
                hysteresis: 0.0,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(2.0, 0.0, 0.0, -1.0));

        let scores = manager.evaluate_cameras(&Point3::new(1.0, 0.0, -3.0));
        // Both cameras should have valid scores
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(
                s.score >= 0.0 && s.score <= 1.0,
                "score out of range: {}",
                s.score
            );
        }
    }

    #[test]
    fn test_centered_framing_criteria() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(
            config,
            AutoSwitchConfig {
                criteria: AutoSwitchCriteria::CenteredFraming,
                min_switch_interval_ms: 0,
                hysteresis: 0.0,
                camera_fov_h: std::f64::consts::PI / 3.0,
                ..AutoSwitchConfig::default()
            },
        )
        .expect("should succeed in test");

        manager.update_camera(CameraId(0), make_camera_pose(0.0, 0.0, 0.0, -1.0));
        manager.update_camera(CameraId(1), make_camera_pose(5.0, 0.0, 0.0, -1.0));

        // Talent at (0, 0, -5) - perfectly centered for camera 0
        let scores = manager.evaluate_cameras(&Point3::new(0.0, 0.0, -5.0));
        assert_eq!(scores[0].camera_id, CameraId(0));
        assert!(
            scores[0].score < 0.05,
            "perfectly centered should have very low score: {}",
            scores[0].score
        );
    }

    #[test]
    fn test_auto_select_no_cameras() {
        let config = MultiCameraConfig {
            num_cameras: 1,
            auto_switch: true,
        };
        let mut manager = MultiCameraManager::with_auto_switch(config, AutoSwitchConfig::default())
            .expect("should succeed in test");

        // No cameras registered yet
        let talent_pos = Point3::new(0.0, 0.0, -5.0);
        let result = manager.auto_select(&talent_pos, 1_000_000_000);
        assert!(result.is_none());
    }
}
