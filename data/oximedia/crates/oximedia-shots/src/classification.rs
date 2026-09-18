//! Shot type classification using cinematographic criteria.
//!
//! Provides enumerations for shot size, camera angle, and camera movement,
//! together with a unified `ShotDescriptor` and a metrics-driven classifier.

#![allow(dead_code)]

/// The framing size of a shot, from extreme wide to extreme close-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotSize {
    /// Extreme Wide Shot — vast landscape or environment.
    ExtremeWide,
    /// Wide Shot — full body visible with significant environment.
    Wide,
    /// Medium Wide Shot — full body with some environment.
    MediumWide,
    /// Medium Shot — waist to head.
    Medium,
    /// Medium Close-Up — chest to head.
    MediumClose,
    /// Close-Up — face fills the frame.
    CloseUp,
    /// Extreme Close-Up — a single feature (eyes, mouth, etc.).
    ExtremeCloseUp,
}

impl ShotSize {
    /// Return a short human-readable description of this shot size.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::ExtremeWide => "Extreme Wide Shot — vast environment",
            Self::Wide => "Wide Shot — full body, large environment",
            Self::MediumWide => "Medium Wide Shot — full body",
            Self::Medium => "Medium Shot — waist up",
            Self::MediumClose => "Medium Close-Up — chest up",
            Self::CloseUp => "Close-Up — face fills frame",
            Self::ExtremeCloseUp => "Extreme Close-Up — single feature",
        }
    }

    /// Return the face-area proportion threshold that characterises this size.
    ///
    /// Face area is expressed as a fraction of total frame area (0.0 – 1.0).
    #[must_use]
    pub fn face_area_threshold(&self) -> f32 {
        match self {
            Self::ExtremeWide => 0.0,
            Self::Wide => 0.01,
            Self::MediumWide => 0.03,
            Self::Medium => 0.06,
            Self::MediumClose => 0.12,
            Self::CloseUp => 0.25,
            Self::ExtremeCloseUp => 0.50,
        }
    }
}

/// The vertical angle of the camera relative to the subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotAngle {
    /// Looking straight down at the subject.
    BirdsEye,
    /// Camera above eye level, looking down.
    HighAngle,
    /// Camera at the subject's eye level.
    EyeLevel,
    /// Camera below eye level, looking up.
    LowAngle,
    /// Camera at ground level, looking up at a steep angle.
    WormEye,
}

impl ShotAngle {
    /// Return a description of the typical psychological effect of this angle.
    #[must_use]
    pub fn psychological_effect(&self) -> &str {
        match self {
            Self::BirdsEye => "Omniscience; the subject appears diminished or map-like",
            Self::HighAngle => "Vulnerability or weakness; subject feels small",
            Self::EyeLevel => "Neutral and objective; viewer equals subject",
            Self::LowAngle => "Power and dominance; subject feels imposing",
            Self::WormEye => "Extreme power; subject towers over the viewer",
        }
    }
}

/// The type of camera movement present in a shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotMovement {
    /// No camera movement — the camera is fixed.
    Static,
    /// Horizontal rotation around the camera's vertical axis.
    Pan,
    /// Vertical rotation around the camera's horizontal axis.
    Tilt,
    /// Optical zoom in or out.
    Zoom,
    /// Physical movement toward or away from the subject.
    Track,
    /// Unstabilised, free-hand camera movement.
    Handheld,
}

impl ShotMovement {
    /// Return `true` if this movement involves the camera physically moving
    /// through space (as opposed to a rotation or optical change).
    #[must_use]
    pub fn is_camera_movement(&self) -> bool {
        matches!(self, Self::Track | Self::Handheld)
    }
}

/// A complete description of a shot's cinematographic properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShotDescriptor {
    /// Framing size.
    pub size: ShotSize,
    /// Camera angle.
    pub angle: ShotAngle,
    /// Dominant camera movement.
    pub movement: ShotMovement,
}

impl ShotDescriptor {
    /// Return `true` if this shot qualifies as an establishing shot.
    ///
    /// An establishing shot is a wide or extreme-wide shot taken from an
    /// eye-level or high angle with no physical camera movement.
    #[must_use]
    pub fn is_establishing(&self) -> bool {
        matches!(self.size, ShotSize::ExtremeWide | ShotSize::Wide)
            && matches!(self.angle, ShotAngle::EyeLevel | ShotAngle::HighAngle)
            && !self.movement.is_camera_movement()
    }
}

/// Classifies shots from simple numeric metrics.
pub struct ShotClassifier;

impl ShotClassifier {
    /// Derive a `ShotDescriptor` from low-level frame metrics.
    ///
    /// # Parameters
    ///
    /// * `face_area`  — face area as fraction of frame area (0.0 – 1.0).
    /// * `horizon_y`  — normalised Y position of the horizon (0.0 = top, 1.0 = bottom).
    /// * `motion_mag` — magnitude of inter-frame optical-flow (pixels/frame).
    /// * `zoom_delta` — change in optical-flow divergence (positive = zoom in).
    #[must_use]
    pub fn classify_from_metrics(
        face_area: f32,
        horizon_y: f32,
        motion_mag: f32,
        zoom_delta: f32,
    ) -> ShotDescriptor {
        let size = Self::classify_size(face_area);
        let angle = Self::classify_angle(horizon_y);
        let movement = Self::classify_movement(motion_mag, zoom_delta);
        ShotDescriptor {
            size,
            angle,
            movement,
        }
    }

    fn classify_size(face_area: f32) -> ShotSize {
        if face_area >= ShotSize::ExtremeCloseUp.face_area_threshold() {
            ShotSize::ExtremeCloseUp
        } else if face_area >= ShotSize::CloseUp.face_area_threshold() {
            ShotSize::CloseUp
        } else if face_area >= ShotSize::MediumClose.face_area_threshold() {
            ShotSize::MediumClose
        } else if face_area >= ShotSize::Medium.face_area_threshold() {
            ShotSize::Medium
        } else if face_area >= ShotSize::MediumWide.face_area_threshold() {
            ShotSize::MediumWide
        } else if face_area >= ShotSize::Wide.face_area_threshold() {
            ShotSize::Wide
        } else {
            ShotSize::ExtremeWide
        }
    }

    fn classify_angle(horizon_y: f32) -> ShotAngle {
        if horizon_y < 0.15 {
            ShotAngle::WormEye
        } else if horizon_y < 0.35 {
            ShotAngle::LowAngle
        } else if horizon_y < 0.65 {
            ShotAngle::EyeLevel
        } else if horizon_y < 0.85 {
            ShotAngle::HighAngle
        } else {
            ShotAngle::BirdsEye
        }
    }

    fn classify_movement(motion_mag: f32, zoom_delta: f32) -> ShotMovement {
        if motion_mag > 20.0 {
            ShotMovement::Handheld
        } else if zoom_delta.abs() > 5.0 {
            ShotMovement::Zoom
        } else if motion_mag > 8.0 {
            ShotMovement::Track
        } else if motion_mag > 3.0 {
            ShotMovement::Pan
        } else if motion_mag > 1.0 {
            ShotMovement::Tilt
        } else {
            ShotMovement::Static
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ShotSize ---

    #[test]
    fn test_shot_size_description_not_empty() {
        for size in [
            ShotSize::ExtremeWide,
            ShotSize::Wide,
            ShotSize::MediumWide,
            ShotSize::Medium,
            ShotSize::MediumClose,
            ShotSize::CloseUp,
            ShotSize::ExtremeCloseUp,
        ] {
            assert!(!size.description().is_empty());
        }
    }

    #[test]
    fn test_face_area_thresholds_ordered() {
        let sizes = [
            ShotSize::ExtremeWide,
            ShotSize::Wide,
            ShotSize::MediumWide,
            ShotSize::Medium,
            ShotSize::MediumClose,
            ShotSize::CloseUp,
            ShotSize::ExtremeCloseUp,
        ];
        for w in sizes.windows(2) {
            assert!(w[0].face_area_threshold() < w[1].face_area_threshold());
        }
    }

    // --- ShotAngle ---

    #[test]
    fn test_angle_psychological_effect_not_empty() {
        for angle in [
            ShotAngle::BirdsEye,
            ShotAngle::HighAngle,
            ShotAngle::EyeLevel,
            ShotAngle::LowAngle,
            ShotAngle::WormEye,
        ] {
            assert!(!angle.psychological_effect().is_empty());
        }
    }

    // --- ShotMovement ---

    #[test]
    fn test_is_camera_movement_track() {
        assert!(ShotMovement::Track.is_camera_movement());
    }

    #[test]
    fn test_is_camera_movement_handheld() {
        assert!(ShotMovement::Handheld.is_camera_movement());
    }

    #[test]
    fn test_is_not_camera_movement_static() {
        assert!(!ShotMovement::Static.is_camera_movement());
    }

    #[test]
    fn test_is_not_camera_movement_pan() {
        assert!(!ShotMovement::Pan.is_camera_movement());
    }

    #[test]
    fn test_is_not_camera_movement_zoom() {
        assert!(!ShotMovement::Zoom.is_camera_movement());
    }

    // --- ShotDescriptor ---

    #[test]
    fn test_is_establishing_true() {
        let d = ShotDescriptor {
            size: ShotSize::Wide,
            angle: ShotAngle::EyeLevel,
            movement: ShotMovement::Static,
        };
        assert!(d.is_establishing());
    }

    #[test]
    fn test_is_establishing_false_wrong_size() {
        let d = ShotDescriptor {
            size: ShotSize::CloseUp,
            angle: ShotAngle::EyeLevel,
            movement: ShotMovement::Static,
        };
        assert!(!d.is_establishing());
    }

    #[test]
    fn test_is_establishing_false_camera_movement() {
        let d = ShotDescriptor {
            size: ShotSize::ExtremeWide,
            angle: ShotAngle::HighAngle,
            movement: ShotMovement::Track,
        };
        assert!(!d.is_establishing());
    }

    // --- ShotClassifier ---

    #[test]
    fn test_classify_extreme_close_up() {
        let d = ShotClassifier::classify_from_metrics(0.6, 0.5, 0.0, 0.0);
        assert_eq!(d.size, ShotSize::ExtremeCloseUp);
    }

    #[test]
    fn test_classify_extreme_wide() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.5, 0.0, 0.0);
        assert_eq!(d.size, ShotSize::ExtremeWide);
    }

    #[test]
    fn test_classify_handheld_movement() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.5, 25.0, 0.0);
        assert_eq!(d.movement, ShotMovement::Handheld);
    }

    #[test]
    fn test_classify_zoom_movement() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.5, 0.0, 8.0);
        assert_eq!(d.movement, ShotMovement::Zoom);
    }

    #[test]
    fn test_classify_static_movement() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.5, 0.0, 0.0);
        assert_eq!(d.movement, ShotMovement::Static);
    }

    #[test]
    fn test_classify_birds_eye_angle() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.95, 0.0, 0.0);
        assert_eq!(d.angle, ShotAngle::BirdsEye);
    }

    #[test]
    fn test_classify_worm_eye_angle() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.05, 0.0, 0.0);
        assert_eq!(d.angle, ShotAngle::WormEye);
    }

    #[test]
    fn test_classify_eye_level_angle() {
        let d = ShotClassifier::classify_from_metrics(0.0, 0.5, 0.0, 0.0);
        assert_eq!(d.angle, ShotAngle::EyeLevel);
    }
}
