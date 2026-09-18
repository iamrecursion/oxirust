//! Pre-visualization (previz) module for virtual production.
//!
//! Provides storyboard-to-virtual-set blocking workflows, including shot
//! sequencing, camera blocking, talent blocking, and timeline export.
//! Enables directors and DPs to plan and visualise shots before principal
//! photography.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// A 3D position and orientation (pose) for blocking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlockingPose {
    /// World-space position [x, y, z] in meters.
    pub position: [f64; 3],
    /// Yaw angle in degrees (rotation around Y axis).
    pub yaw_deg: f64,
}

impl BlockingPose {
    /// Create a new blocking pose.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64, yaw_deg: f64) -> Self {
        Self {
            position: [x, y, z],
            yaw_deg,
        }
    }

    /// Create at origin.
    #[must_use]
    pub const fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Camera move type in a previz shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraMoveType {
    /// Camera remains stationary.
    Static,
    /// Smooth dolly move on a straight track.
    Dolly,
    /// Pan (rotation around vertical axis only).
    Pan,
    /// Tilt (rotation around horizontal axis).
    Tilt,
    /// Handheld organic movement.
    Handheld,
    /// Crane/jib arm sweep.
    Crane,
    /// Aerial/drone movement.
    Drone,
}

/// Shot type (framing convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShotType {
    ExtremeWide,
    Wide,
    Medium,
    MediumClose,
    Close,
    ExtremeClose,
    TwoShot,
    OverTheShoulder,
}

impl ShotType {
    /// Return a human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ExtremeWide => "Extreme Wide Shot",
            Self::Wide => "Wide Shot",
            Self::Medium => "Medium Shot",
            Self::MediumClose => "Medium Close-Up",
            Self::Close => "Close-Up",
            Self::ExtremeClose => "Extreme Close-Up",
            Self::TwoShot => "Two Shot",
            Self::OverTheShoulder => "Over the Shoulder",
        }
    }
}

/// A single camera blocking keyframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraKeyframe {
    /// Timecode in seconds from the start of the shot.
    pub time_s: f64,
    /// Camera world pose at this keyframe.
    pub pose: BlockingPose,
    /// Lens focal length in mm.
    pub focal_length_mm: f64,
    /// Camera move type from previous keyframe to this one.
    pub move_type: CameraMoveType,
}

/// A talent (actor) blocking keyframe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalentKeyframe {
    /// Actor identifier.
    pub actor_id: String,
    /// Timecode in seconds.
    pub time_s: f64,
    /// Actor world pose.
    pub pose: BlockingPose,
    /// Optional action/emotion note.
    pub note: Option<String>,
}

/// A single storyboard panel that maps to a virtual-set shot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryboardShot {
    /// Unique shot identifier (e.g. "1A", "2B_CLOSE").
    pub id: String,
    /// Shot description from the script.
    pub description: String,
    /// Shot type (framing).
    pub shot_type: ShotType,
    /// Duration in seconds.
    pub duration_s: f64,
    /// Camera keyframes for this shot.
    pub camera_keys: Vec<CameraKeyframe>,
    /// Talent blocking for this shot.
    pub talent_keys: Vec<TalentKeyframe>,
    /// Virtual set scene reference.
    pub scene_id: Option<String>,
    /// Notes from the director.
    pub director_notes: Option<String>,
    /// Estimated render frame count (for scheduling).
    pub frame_count: u32,
}

impl StoryboardShot {
    /// Create a new storyboard shot.
    #[must_use]
    pub fn new(id: &str, description: &str, shot_type: ShotType, duration_s: f64) -> Self {
        let fps = 24.0_f64;
        let frame_count = (duration_s * fps) as u32;
        Self {
            id: id.to_string(),
            description: description.to_string(),
            shot_type,
            duration_s,
            camera_keys: Vec::new(),
            talent_keys: Vec::new(),
            scene_id: None,
            director_notes: None,
            frame_count,
        }
    }

    /// Add a camera keyframe.
    pub fn add_camera_key(&mut self, key: CameraKeyframe) {
        self.camera_keys.push(key);
    }

    /// Add a talent keyframe.
    pub fn add_talent_key(&mut self, key: TalentKeyframe) {
        self.talent_keys.push(key);
    }

    /// Set scene id.
    #[must_use]
    pub fn with_scene(mut self, scene_id: &str) -> Self {
        self.scene_id = Some(scene_id.to_string());
        self
    }

    /// Set director notes.
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.director_notes = Some(notes.to_string());
        self
    }

    /// Interpolate camera pose at a given time using linear interpolation.
    ///
    /// Returns `None` if no keyframes exist.
    #[must_use]
    pub fn interpolate_camera(&self, time_s: f64) -> Option<BlockingPose> {
        if self.camera_keys.is_empty() {
            return None;
        }

        // Find surrounding keyframes
        let mut before: Option<&CameraKeyframe> = None;
        let mut after: Option<&CameraKeyframe> = None;

        for key in &self.camera_keys {
            if key.time_s <= time_s {
                before = Some(key);
            } else if after.is_none() && key.time_s > time_s {
                after = Some(key);
            }
        }

        match (before, after) {
            (Some(b), Some(a)) => {
                let t = if (a.time_s - b.time_s).abs() < 1e-9 {
                    0.0
                } else {
                    (time_s - b.time_s) / (a.time_s - b.time_s)
                };
                Some(lerp_pose(b.pose, a.pose, t))
            }
            (Some(b), None) => Some(b.pose),
            (None, Some(a)) => Some(a.pose),
            (None, None) => None,
        }
    }

    /// Total number of unique actors in the blocking.
    #[must_use]
    pub fn actor_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for key in &self.talent_keys {
            seen.insert(key.actor_id.clone());
        }
        seen.len()
    }
}

/// Linear interpolation of two blocking poses.
fn lerp_pose(a: BlockingPose, b: BlockingPose, t: f64) -> BlockingPose {
    BlockingPose {
        position: [
            a.position[0] + (b.position[0] - a.position[0]) * t,
            a.position[1] + (b.position[1] - a.position[1]) * t,
            a.position[2] + (b.position[2] - a.position[2]) * t,
        ],
        yaw_deg: a.yaw_deg + (b.yaw_deg - a.yaw_deg) * t,
    }
}

/// Full previz sequence: ordered list of storyboard shots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrevizSequence {
    /// Sequence title.
    pub title: String,
    /// Ordered list of shots.
    pub shots: Vec<StoryboardShot>,
    /// Default frames-per-second for the sequence.
    pub fps: f64,
    /// Creator/director name.
    pub director: Option<String>,
}

impl PrevizSequence {
    /// Create a new sequence.
    #[must_use]
    pub fn new(title: &str, fps: f64) -> Self {
        Self {
            title: title.to_string(),
            shots: Vec::new(),
            fps,
            director: None,
        }
    }

    /// Add a shot.
    pub fn add_shot(&mut self, shot: StoryboardShot) {
        self.shots.push(shot);
    }

    /// Total duration in seconds.
    #[must_use]
    pub fn total_duration_s(&self) -> f64 {
        self.shots.iter().map(|s| s.duration_s).sum()
    }

    /// Total frame count.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.shots.iter().map(|s| s.frame_count as u64).sum()
    }

    /// Find a shot by ID.
    #[must_use]
    pub fn find_shot(&self, id: &str) -> Option<&StoryboardShot> {
        self.shots.iter().find(|s| s.id == id)
    }

    /// Find a mutable shot by ID.
    pub fn find_shot_mut(&mut self, id: &str) -> Option<&mut StoryboardShot> {
        self.shots.iter_mut().find(|s| s.id == id)
    }

    /// Remove a shot by ID.
    pub fn remove_shot(&mut self, id: &str) {
        self.shots.retain(|s| s.id != id);
    }

    /// Export to a simple JSON-like string representation.
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| VirtualProductionError::InvalidConfig(format!("JSON export error: {e}")))
    }

    /// Import from JSON.
    pub fn import_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| VirtualProductionError::InvalidConfig(format!("JSON import error: {e}")))
    }

    /// Generate a simple text breakdown for printing.
    #[must_use]
    pub fn text_breakdown(&self) -> String {
        let mut s = format!("PREVIZ: {}\n", self.title);
        if let Some(dir) = &self.director {
            s.push_str(&format!("Director: {dir}\n"));
        }
        s.push_str(&format!("Total shots: {}\n", self.shots.len()));
        s.push_str(&format!(
            "Total duration: {:.1}s\n",
            self.total_duration_s()
        ));
        s.push_str("---\n");
        for (i, shot) in self.shots.iter().enumerate() {
            s.push_str(&format!(
                "[{}] {} - {} ({:.1}s) {}\n",
                i + 1,
                shot.id,
                shot.shot_type.name(),
                shot.duration_s,
                shot.description,
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_previz_sequence_creation() {
        let seq = PrevizSequence::new("Test Sequence", 24.0);
        assert_eq!(seq.title, "Test Sequence");
        assert_eq!(seq.fps, 24.0);
        assert!(seq.shots.is_empty());
    }

    #[test]
    fn test_add_shot() {
        let mut seq = PrevizSequence::new("Test", 24.0);
        let shot = StoryboardShot::new("1A", "Wide establishing shot", ShotType::Wide, 3.0);
        seq.add_shot(shot);
        assert_eq!(seq.shots.len(), 1);
    }

    #[test]
    fn test_total_duration() {
        let mut seq = PrevizSequence::new("Test", 24.0);
        seq.add_shot(StoryboardShot::new("1A", "desc", ShotType::Wide, 3.0));
        seq.add_shot(StoryboardShot::new("1B", "desc", ShotType::Medium, 2.5));
        seq.add_shot(StoryboardShot::new("1C", "desc", ShotType::Close, 1.0));
        assert!((seq.total_duration_s() - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_total_frames() {
        let mut seq = PrevizSequence::new("Test", 24.0);
        // 3s * 24fps = 72 frames
        seq.add_shot(StoryboardShot::new("1A", "desc", ShotType::Wide, 3.0));
        assert_eq!(seq.total_frames(), 72);
    }

    #[test]
    fn test_find_shot() {
        let mut seq = PrevizSequence::new("Test", 24.0);
        seq.add_shot(StoryboardShot::new("2A", "close", ShotType::Close, 2.0));
        seq.add_shot(StoryboardShot::new("2B", "medium", ShotType::Medium, 3.0));

        assert!(seq.find_shot("2A").is_some());
        assert!(seq.find_shot("3X").is_none());
    }

    #[test]
    fn test_remove_shot() {
        let mut seq = PrevizSequence::new("Test", 24.0);
        seq.add_shot(StoryboardShot::new("1A", "wide", ShotType::Wide, 3.0));
        seq.add_shot(StoryboardShot::new("1B", "close", ShotType::Close, 2.0));
        seq.remove_shot("1A");
        assert_eq!(seq.shots.len(), 1);
        assert_eq!(seq.shots[0].id, "1B");
    }

    #[test]
    fn test_storyboard_shot_with_scene_and_notes() {
        let shot = StoryboardShot::new("3A", "desc", ShotType::TwoShot, 4.0)
            .with_scene("INT_OFFICE_DAY")
            .with_notes("Tension builds here");
        assert_eq!(shot.scene_id.as_deref(), Some("INT_OFFICE_DAY"));
        assert!(shot.director_notes.is_some());
    }

    #[test]
    fn test_camera_interpolation_no_keys() {
        let shot = StoryboardShot::new("1A", "desc", ShotType::Wide, 3.0);
        assert!(shot.interpolate_camera(1.5).is_none());
    }

    #[test]
    fn test_camera_interpolation_single_key() {
        let mut shot = StoryboardShot::new("1A", "desc", ShotType::Wide, 3.0);
        shot.add_camera_key(CameraKeyframe {
            time_s: 0.0,
            pose: BlockingPose::new(1.0, 0.0, 3.0, 45.0),
            focal_length_mm: 35.0,
            move_type: CameraMoveType::Static,
        });
        let pose = shot.interpolate_camera(1.5);
        assert!(pose.is_some());
        let p = pose.expect("some");
        assert!((p.position[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_camera_interpolation_two_keys() {
        let mut shot = StoryboardShot::new("1A", "desc", ShotType::Wide, 4.0);
        shot.add_camera_key(CameraKeyframe {
            time_s: 0.0,
            pose: BlockingPose::new(0.0, 0.0, 0.0, 0.0),
            focal_length_mm: 35.0,
            move_type: CameraMoveType::Dolly,
        });
        shot.add_camera_key(CameraKeyframe {
            time_s: 4.0,
            pose: BlockingPose::new(4.0, 0.0, 0.0, 90.0),
            focal_length_mm: 50.0,
            move_type: CameraMoveType::Dolly,
        });

        let pose = shot.interpolate_camera(2.0).expect("should interpolate");
        assert!(
            (pose.position[0] - 2.0).abs() < 1e-6,
            "x should be 2.0: {}",
            pose.position[0]
        );
        assert!(
            (pose.yaw_deg - 45.0).abs() < 1e-6,
            "yaw should be 45: {}",
            pose.yaw_deg
        );
    }

    #[test]
    fn test_blocking_pose_lerp() {
        let a = BlockingPose::new(0.0, 0.0, 0.0, 0.0);
        let b = BlockingPose::new(10.0, 5.0, 2.0, 180.0);
        let mid = lerp_pose(a, b, 0.5);
        assert!((mid.position[0] - 5.0).abs() < 1e-9);
        assert!((mid.position[1] - 2.5).abs() < 1e-9);
        assert!((mid.yaw_deg - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_actor_count() {
        let mut shot = StoryboardShot::new("1A", "desc", ShotType::TwoShot, 3.0);
        shot.add_talent_key(TalentKeyframe {
            actor_id: "ALICE".to_string(),
            time_s: 0.0,
            pose: BlockingPose::origin(),
            note: None,
        });
        shot.add_talent_key(TalentKeyframe {
            actor_id: "BOB".to_string(),
            time_s: 0.0,
            pose: BlockingPose::new(1.5, 0.0, 0.0, 180.0),
            note: None,
        });
        shot.add_talent_key(TalentKeyframe {
            actor_id: "ALICE".to_string(),
            time_s: 2.0,
            pose: BlockingPose::new(0.5, 0.0, 0.0, 0.0),
            note: Some("Moves toward Bob".to_string()),
        });
        assert_eq!(shot.actor_count(), 2);
    }

    #[test]
    fn test_export_import_json() {
        let mut seq = PrevizSequence::new("Export Test", 25.0);
        seq.director = Some("Test Director".to_string());
        seq.add_shot(StoryboardShot::new(
            "1A",
            "Opening shot",
            ShotType::ExtremeWide,
            5.0,
        ));
        seq.add_shot(StoryboardShot::new(
            "1B",
            "Close on face",
            ShotType::Close,
            2.0,
        ));

        let json = seq.export_json().expect("should export");
        assert!(json.contains("Export Test"));

        let imported = PrevizSequence::import_json(&json).expect("should import");
        assert_eq!(imported.title, "Export Test");
        assert_eq!(imported.shots.len(), 2);
        assert_eq!(imported.shots[0].id, "1A");
    }

    #[test]
    fn test_text_breakdown() {
        let mut seq = PrevizSequence::new("My Film", 24.0);
        seq.add_shot(StoryboardShot::new(
            "1A",
            "Scene opens",
            ShotType::Wide,
            3.0,
        ));
        let breakdown = seq.text_breakdown();
        assert!(breakdown.contains("My Film"));
        assert!(breakdown.contains("1A"));
        assert!(breakdown.contains("Wide Shot"));
    }

    #[test]
    fn test_shot_type_names() {
        assert!(!ShotType::ExtremeWide.name().is_empty());
        assert!(!ShotType::OverTheShoulder.name().is_empty());
    }

    #[test]
    fn test_camera_move_types_exist() {
        let moves = [
            CameraMoveType::Static,
            CameraMoveType::Dolly,
            CameraMoveType::Pan,
            CameraMoveType::Tilt,
            CameraMoveType::Handheld,
            CameraMoveType::Crane,
            CameraMoveType::Drone,
        ];
        assert_eq!(moves.len(), 7);
    }
}
