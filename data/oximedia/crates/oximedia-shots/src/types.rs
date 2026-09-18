//! Common types for shot detection and classification.

use oximedia_core::types::Timestamp;
use serde::{Deserialize, Serialize};

/// A detected shot in a video.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    /// Shot ID (unique within a video).
    pub id: u64,
    /// Start time of the shot.
    pub start: Timestamp,
    /// End time of the shot.
    pub end: Timestamp,
    /// Shot type classification.
    pub shot_type: ShotType,
    /// Camera angle.
    pub angle: CameraAngle,
    /// Camera movements detected in this shot.
    pub movements: Vec<CameraMovement>,
    /// Shot composition analysis.
    pub composition: CompositionAnalysis,
    /// Coverage type.
    pub coverage: CoverageType,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Transition from previous shot.
    pub transition: TransitionType,
}

impl Shot {
    /// Create a new shot.
    #[must_use]
    pub const fn new(id: u64, start: Timestamp, end: Timestamp) -> Self {
        Self {
            id,
            start,
            end,
            shot_type: ShotType::MediumShot,
            angle: CameraAngle::EyeLevel,
            movements: Vec::new(),
            composition: CompositionAnalysis {
                rule_of_thirds: 0.0,
                symmetry: 0.0,
                balance: 0.0,
                leading_lines: 0.0,
                depth: 0.0,
            },
            coverage: CoverageType::Master,
            confidence: 0.0,
            transition: TransitionType::Cut,
        }
    }

    /// Get duration of the shot.
    #[must_use]
    pub fn duration(&self) -> Timestamp {
        Timestamp::new(self.end.pts - self.start.pts, self.start.timebase)
    }

    /// Get duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.duration().to_seconds()
    }
}

/// Shot type classification based on framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShotType {
    /// Extreme Close-up (ECU) - Face details, eyes, lips.
    ExtremeCloseUp,
    /// Close-up (CU) - Head and shoulders.
    CloseUp,
    /// Medium Close-up (MCU) - Waist up.
    MediumCloseUp,
    /// Medium Shot (MS) - Knees up.
    MediumShot,
    /// Medium Long Shot (MLS) - Full body with space.
    MediumLongShot,
    /// Long Shot (LS) - Full body in environment.
    LongShot,
    /// Extreme Long Shot (ELS) - Establishing shot.
    ExtremeLongShot,
    /// Unknown or unclassified.
    Unknown,
}

impl ShotType {
    /// Get a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ExtremeCloseUp => "Extreme Close-up (ECU)",
            Self::CloseUp => "Close-up (CU)",
            Self::MediumCloseUp => "Medium Close-up (MCU)",
            Self::MediumShot => "Medium Shot (MS)",
            Self::MediumLongShot => "Medium Long Shot (MLS)",
            Self::LongShot => "Long Shot (LS)",
            Self::ExtremeLongShot => "Extreme Long Shot (ELS)",
            Self::Unknown => "Unknown",
        }
    }

    /// Get the abbreviation.
    #[must_use]
    pub const fn abbreviation(&self) -> &'static str {
        match self {
            Self::ExtremeCloseUp => "ECU",
            Self::CloseUp => "CU",
            Self::MediumCloseUp => "MCU",
            Self::MediumShot => "MS",
            Self::MediumLongShot => "MLS",
            Self::LongShot => "LS",
            Self::ExtremeLongShot => "ELS",
            Self::Unknown => "UNK",
        }
    }
}

/// Camera angle classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CameraAngle {
    /// High angle (looking down).
    High,
    /// Eye level (neutral).
    EyeLevel,
    /// Low angle (looking up).
    Low,
    /// Bird's eye view (directly above).
    BirdsEye,
    /// Dutch angle (tilted).
    Dutch,
    /// Unknown.
    Unknown,
}

impl CameraAngle {
    /// Get a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::High => "High Angle",
            Self::EyeLevel => "Eye Level",
            Self::Low => "Low Angle",
            Self::BirdsEye => "Bird's Eye",
            Self::Dutch => "Dutch Angle",
            Self::Unknown => "Unknown",
        }
    }
}

/// Camera movement detected in a shot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraMovement {
    /// Type of movement.
    pub movement_type: MovementType,
    /// Start time relative to shot start.
    pub start: f64,
    /// End time relative to shot start.
    pub end: f64,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Movement speed (pixels per frame or degrees per second).
    pub speed: f32,
}

/// Type of camera movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MovementType {
    /// Pan left.
    PanLeft,
    /// Pan right.
    PanRight,
    /// Tilt up.
    TiltUp,
    /// Tilt down.
    TiltDown,
    /// Zoom in.
    ZoomIn,
    /// Zoom out.
    ZoomOut,
    /// Dolly in (camera moves toward subject).
    DollyIn,
    /// Dolly out (camera moves away from subject).
    DollyOut,
    /// Track left (lateral movement).
    TrackLeft,
    /// Track right (lateral movement).
    TrackRight,
    /// Handheld shake.
    Handheld,
    /// Static (no movement).
    Static,
}

impl MovementType {
    /// Get a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::PanLeft => "Pan Left",
            Self::PanRight => "Pan Right",
            Self::TiltUp => "Tilt Up",
            Self::TiltDown => "Tilt Down",
            Self::ZoomIn => "Zoom In",
            Self::ZoomOut => "Zoom Out",
            Self::DollyIn => "Dolly In",
            Self::DollyOut => "Dolly Out",
            Self::TrackLeft => "Track Left",
            Self::TrackRight => "Track Right",
            Self::Handheld => "Handheld",
            Self::Static => "Static",
        }
    }
}

/// Composition analysis of a shot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositionAnalysis {
    /// Rule of thirds score (0.0 to 1.0).
    pub rule_of_thirds: f32,
    /// Symmetry score (0.0 to 1.0).
    pub symmetry: f32,
    /// Balance score (0.0 to 1.0).
    pub balance: f32,
    /// Leading lines score (0.0 to 1.0).
    pub leading_lines: f32,
    /// Depth perception score (0.0 to 1.0).
    pub depth: f32,
}

impl CompositionAnalysis {
    /// Calculate overall composition score.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        (self.rule_of_thirds + self.symmetry + self.balance + self.leading_lines + self.depth) / 5.0
    }
}

/// Coverage type (cinematography terminology).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageType {
    /// Master shot (wide shot establishing the scene).
    Master,
    /// Single (one person in frame).
    Single,
    /// Two-shot (two people in frame).
    TwoShot,
    /// Three-shot (three people in frame).
    ThreeShot,
    /// Over-the-shoulder shot.
    OverTheShoulder,
    /// Point of view shot.
    PointOfView,
    /// Insert shot (detail shot).
    Insert,
    /// Cutaway (shot away from main action).
    Cutaway,
    /// Unknown.
    Unknown,
}

impl CoverageType {
    /// Get a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Master => "Master Shot",
            Self::Single => "Single",
            Self::TwoShot => "Two-Shot",
            Self::ThreeShot => "Three-Shot",
            Self::OverTheShoulder => "Over-the-Shoulder",
            Self::PointOfView => "Point of View",
            Self::Insert => "Insert",
            Self::Cutaway => "Cutaway",
            Self::Unknown => "Unknown",
        }
    }
}

/// Type of transition between shots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionType {
    /// Hard cut.
    Cut,
    /// Dissolve/crossfade.
    Dissolve,
    /// Fade to black.
    FadeToBlack,
    /// Fade from black.
    FadeFromBlack,
    /// Fade to white.
    FadeToWhite,
    /// Fade from white.
    FadeFromWhite,
    /// Wipe (left to right).
    WipeLeft,
    /// Wipe (right to left).
    WipeRight,
    /// Wipe (top to bottom).
    WipeDown,
    /// Wipe (bottom to top).
    WipeUp,
    /// Unknown transition.
    Unknown,
}

impl TransitionType {
    /// Get a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Cut => "Cut",
            Self::Dissolve => "Dissolve",
            Self::FadeToBlack => "Fade to Black",
            Self::FadeFromBlack => "Fade from Black",
            Self::FadeToWhite => "Fade to White",
            Self::FadeFromWhite => "Fade from White",
            Self::WipeLeft => "Wipe Left",
            Self::WipeRight => "Wipe Right",
            Self::WipeDown => "Wipe Down",
            Self::WipeUp => "Wipe Up",
            Self::Unknown => "Unknown",
        }
    }
}

/// A scene (collection of related shots).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// Scene ID.
    pub id: u64,
    /// Start time of the scene.
    pub start: Timestamp,
    /// End time of the scene.
    pub end: Timestamp,
    /// Shots in this scene.
    pub shots: Vec<u64>,
    /// Scene type/location.
    pub scene_type: String,
    /// Confidence score.
    pub confidence: f32,
}

impl Scene {
    /// Create a new scene.
    #[must_use]
    pub const fn new(id: u64, start: Timestamp, end: Timestamp) -> Self {
        Self {
            id,
            start,
            end,
            shots: Vec::new(),
            scene_type: String::new(),
            confidence: 0.0,
        }
    }

    /// Get duration of the scene.
    #[must_use]
    pub fn duration(&self) -> Timestamp {
        Timestamp::new(self.end.pts - self.start.pts, self.start.timebase)
    }

    /// Get shot count.
    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.shots.len()
    }
}

/// Shot detection statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotStatistics {
    /// Total number of shots.
    pub total_shots: usize,
    /// Total number of scenes.
    pub total_scenes: usize,
    /// Average shot duration in seconds.
    pub average_shot_duration: f64,
    /// Median shot duration in seconds.
    pub median_shot_duration: f64,
    /// Minimum shot duration in seconds.
    pub min_shot_duration: f64,
    /// Maximum shot duration in seconds.
    pub max_shot_duration: f64,
    /// Shot type distribution.
    pub shot_type_distribution: Vec<(ShotType, usize)>,
    /// Coverage type distribution.
    pub coverage_distribution: Vec<(CoverageType, usize)>,
    /// Transition type distribution.
    pub transition_distribution: Vec<(TransitionType, usize)>,
    /// Average shots per scene.
    pub average_shots_per_scene: f64,
}

impl Default for ShotStatistics {
    fn default() -> Self {
        Self {
            total_shots: 0,
            total_scenes: 0,
            average_shot_duration: 0.0,
            median_shot_duration: 0.0,
            min_shot_duration: 0.0,
            max_shot_duration: 0.0,
            shot_type_distribution: Vec::new(),
            coverage_distribution: Vec::new(),
            transition_distribution: Vec::new(),
            average_shots_per_scene: 0.0,
        }
    }
}
