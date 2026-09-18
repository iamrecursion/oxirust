//! Camera-movement detection for `oximedia-shots`.
//!
//! Detects the dominant camera movement type from per-frame optical-flow
//! vectors (represented here as simple `(dx, dy, dz)` tuples so that this
//! module stays dependency-free for unit-testing purposes).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Movement type ─────────────────────────────────────────────────────────────

/// Canonical camera-movement classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MovementType {
    /// Camera pans left or right about its vertical axis.
    Pan,
    /// Camera tilts up or down about its horizontal axis.
    Tilt,
    /// Lens zoom-in or zoom-out (no physical camera displacement).
    Zoom,
    /// Camera physically moves toward or away from the subject (dolly/truck).
    Dolly,
    /// No significant camera movement – the frame is locked off.
    Static,
}

impl MovementType {
    /// Returns a short description of the movement.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::Pan => "Horizontal rotation of camera on its axis",
            Self::Tilt => "Vertical rotation of camera on its axis",
            Self::Zoom => "Lens focal-length change without physical movement",
            Self::Dolly => "Physical camera displacement toward/away from subject",
            Self::Static => "No significant camera movement",
        }
    }

    /// Returns `true` when the movement involves physical camera displacement.
    #[must_use]
    pub fn is_physical(self) -> bool {
        matches!(self, Self::Dolly)
    }

    /// Returns `true` when the movement is a rotation-only operation.
    #[must_use]
    pub fn is_rotation(self) -> bool {
        matches!(self, Self::Pan | Self::Tilt)
    }
}

impl std::fmt::Display for MovementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Pan => "Pan",
            Self::Tilt => "Tilt",
            Self::Zoom => "Zoom",
            Self::Dolly => "Dolly",
            Self::Static => "Static",
        };
        write!(f, "{s}")
    }
}

// ── Per-frame flow sample ─────────────────────────────────────────────────────

/// A simplified optical-flow sample for one inter-frame interval.
///
/// * `dx` – mean horizontal pixel displacement (positive = right)
/// * `dy` – mean vertical pixel displacement (positive = down)
/// * `dz` – estimated depth-change proxy (positive = zoom-in / dolly-in)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlowSample {
    /// Mean horizontal displacement in normalised pixels.
    pub dx: f32,
    /// Mean vertical displacement in normalised pixels.
    pub dy: f32,
    /// Depth change proxy (zoom / dolly).
    pub dz: f32,
}

impl FlowSample {
    /// Creates a new flow sample.
    #[must_use]
    pub fn new(dx: f32, dy: f32, dz: f32) -> Self {
        Self { dx, dy, dz }
    }

    /// Euclidean magnitude of the in-plane (dx, dy) component.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

// ── Detector ──────────────────────────────────────────────────────────────────

/// Detects the dominant camera movement from a sequence of [`FlowSample`]s.
#[derive(Debug, Clone)]
pub struct MovementDetector {
    /// Minimum mean displacement (pixels) required to be classified as moving.
    pub static_threshold: f32,
    /// Ratio by which one axis must dominate the other to be classified as
    /// Pan or Tilt (rather than a compound movement).
    pub axis_dominance_ratio: f32,
}

impl Default for MovementDetector {
    fn default() -> Self {
        Self {
            static_threshold: 0.5,
            axis_dominance_ratio: 2.0,
        }
    }
}

impl MovementDetector {
    /// Creates a new detector with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Classifies the dominant movement from a slice of [`FlowSample`]s.
    ///
    /// Returns [`MovementType::Static`] for an empty slice.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn classify(&self, samples: &[FlowSample]) -> MovementType {
        if samples.is_empty() {
            return MovementType::Static;
        }

        let n = samples.len() as f32;
        let mean_dx: f32 = samples.iter().map(|s| s.dx).sum::<f32>() / n;
        let mean_dy: f32 = samples.iter().map(|s| s.dy).sum::<f32>() / n;
        let mean_dz: f32 = samples.iter().map(|s| s.dz).sum::<f32>() / n;

        let abs_dx = mean_dx.abs();
        let abs_dy = mean_dy.abs();
        let abs_dz = mean_dz.abs();

        // Check for static first
        let in_plane_magnitude = (abs_dx * abs_dx + abs_dy * abs_dy).sqrt();
        if in_plane_magnitude < self.static_threshold && abs_dz < self.static_threshold {
            return MovementType::Static;
        }

        // Depth / zoom dominates?
        if abs_dz > in_plane_magnitude * self.axis_dominance_ratio {
            return MovementType::Zoom;
        }

        // Pan vs Tilt
        if abs_dx >= abs_dy * self.axis_dominance_ratio {
            MovementType::Pan
        } else if abs_dy >= abs_dx * self.axis_dominance_ratio {
            MovementType::Tilt
        } else {
            // Neither axis dominates clearly – pick the larger
            if abs_dx >= abs_dy {
                MovementType::Pan
            } else {
                MovementType::Tilt
            }
        }
    }

    /// Classifies movement and also returns the mean flow magnitude.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn classify_with_magnitude(&self, samples: &[FlowSample]) -> (MovementType, f32) {
        let movement = self.classify(samples);
        let magnitude = if samples.is_empty() {
            0.0
        } else {
            let n = samples.len() as f32;
            samples.iter().map(|s| s.magnitude()).sum::<f32>() / n
        };
        (movement, magnitude)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn det() -> MovementDetector {
        MovementDetector::new()
    }

    #[test]
    fn test_static_empty() {
        assert_eq!(det().classify(&[]), MovementType::Static);
    }

    #[test]
    fn test_static_small_motion() {
        let samples = vec![FlowSample::new(0.1, 0.1, 0.0)];
        assert_eq!(det().classify(&samples), MovementType::Static);
    }

    #[test]
    fn test_pan_right() {
        let samples = vec![
            FlowSample::new(5.0, 0.0, 0.0),
            FlowSample::new(5.0, 0.2, 0.0),
        ];
        assert_eq!(det().classify(&samples), MovementType::Pan);
    }

    #[test]
    fn test_pan_left() {
        let samples = vec![FlowSample::new(-6.0, 0.1, 0.0)];
        assert_eq!(det().classify(&samples), MovementType::Pan);
    }

    #[test]
    fn test_tilt_up() {
        let samples = vec![FlowSample::new(0.1, -5.0, 0.0)];
        assert_eq!(det().classify(&samples), MovementType::Tilt);
    }

    #[test]
    fn test_tilt_down() {
        let samples = vec![FlowSample::new(0.0, 7.0, 0.0)];
        assert_eq!(det().classify(&samples), MovementType::Tilt);
    }

    #[test]
    fn test_zoom_in() {
        let samples = vec![FlowSample::new(0.0, 0.0, 8.0)];
        assert_eq!(det().classify(&samples), MovementType::Zoom);
    }

    #[test]
    fn test_zoom_out() {
        let samples = vec![FlowSample::new(0.1, 0.1, -9.0)];
        assert_eq!(det().classify(&samples), MovementType::Zoom);
    }

    #[test]
    fn test_classify_with_magnitude_static() {
        let (mv, mag) = det().classify_with_magnitude(&[]);
        assert_eq!(mv, MovementType::Static);
        assert!((mag - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_with_magnitude_pan() {
        let samples = vec![FlowSample::new(4.0, 0.0, 0.0)];
        let (mv, mag) = det().classify_with_magnitude(&samples);
        assert_eq!(mv, MovementType::Pan);
        assert!((mag - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_movement_type_description_static() {
        assert!(!MovementType::Static.description().is_empty());
    }

    #[test]
    fn test_movement_type_is_physical() {
        assert!(MovementType::Dolly.is_physical());
        assert!(!MovementType::Pan.is_physical());
        assert!(!MovementType::Zoom.is_physical());
    }

    #[test]
    fn test_movement_type_is_rotation() {
        assert!(MovementType::Pan.is_rotation());
        assert!(MovementType::Tilt.is_rotation());
        assert!(!MovementType::Zoom.is_rotation());
        assert!(!MovementType::Dolly.is_rotation());
    }

    #[test]
    fn test_flow_sample_magnitude() {
        let s = FlowSample::new(3.0, 4.0, 0.0);
        assert!((s.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_display_movement_types() {
        assert_eq!(format!("{}", MovementType::Pan), "Pan");
        assert_eq!(format!("{}", MovementType::Tilt), "Tilt");
        assert_eq!(format!("{}", MovementType::Zoom), "Zoom");
        assert_eq!(format!("{}", MovementType::Dolly), "Dolly");
        assert_eq!(format!("{}", MovementType::Static), "Static");
    }

    #[test]
    fn test_multi_sample_pan_average() {
        let samples = vec![
            FlowSample::new(4.0, 0.0, 0.0),
            FlowSample::new(6.0, 0.5, 0.0),
            FlowSample::new(5.0, 0.2, 0.0),
        ];
        assert_eq!(det().classify(&samples), MovementType::Pan);
    }
}
