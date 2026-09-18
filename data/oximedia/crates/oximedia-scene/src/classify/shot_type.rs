//! Shot type classification (close-up, medium, wide, establishing).
//!
//! Classifies the camera shot type based on composition and content cues:
//! face/skin pixel density, depth-of-field indicators, and frame coverage ratios.

use crate::common::Confidence;
use crate::error::SceneResult;
use serde::{Deserialize, Serialize};

/// Camera shot type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShotType {
    /// Extreme close-up: single feature fills frame (eye, lips).
    ExtremeCloseUp,
    /// Close-up: face fills most of frame.
    CloseUp,
    /// Medium close-up: head and shoulders.
    MediumCloseUp,
    /// Medium shot: waist up.
    Medium,
    /// Medium wide: full body visible.
    MediumWide,
    /// Wide shot: full body with environment.
    Wide,
    /// Establishing shot: wide view of location, tiny or no people.
    Establishing,
    /// Two-shot: two people in frame.
    TwoShot,
    /// Over-the-shoulder shot: OTS interview/conversation.
    OverTheShoulder,
    /// Point-of-view shot: subjective camera.
    PointOfView,
}

impl ShotType {
    /// Get human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::ExtremeCloseUp => "Extreme Close-Up",
            Self::CloseUp => "Close-Up",
            Self::MediumCloseUp => "Medium Close-Up",
            Self::Medium => "Medium",
            Self::MediumWide => "Medium Wide",
            Self::Wide => "Wide",
            Self::Establishing => "Establishing",
            Self::TwoShot => "Two-Shot",
            Self::OverTheShoulder => "Over-The-Shoulder",
            Self::PointOfView => "Point of View",
        }
    }

    /// Whether this is a close-range shot.
    #[must_use]
    pub const fn is_close(&self) -> bool {
        matches!(
            self,
            Self::ExtremeCloseUp | Self::CloseUp | Self::MediumCloseUp
        )
    }

    /// Whether this is a wide-range shot.
    #[must_use]
    pub const fn is_wide(&self) -> bool {
        matches!(self, Self::Wide | Self::Establishing | Self::MediumWide)
    }
}

/// Features extracted for shot classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotFeatures {
    /// Fraction of pixels with skin-tone color.
    pub skin_ratio: f32,
    /// Largest connected skin region as fraction of frame area.
    pub max_skin_region: f32,
    /// Vertical position of dominant face-like region (0=top, 1=bottom).
    pub face_vertical_pos: f32,
    /// Depth-of-field blur indicator (0=sharp, 1=blurred background).
    pub dof_blur: f32,
    /// Horizontal symmetry index (0=asymmetric, 1=symmetric).
    pub symmetry: f32,
    /// Edge density in center third vs frame average.
    pub center_edge_ratio: f32,
    /// Estimated number of face-like blobs.
    pub face_count_estimate: u32,
}

/// Shot type classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotClassification {
    /// Classified shot type.
    pub shot_type: ShotType,
    /// Confidence score.
    pub confidence: Confidence,
    /// Raw extracted features.
    pub features: ShotFeatures,
    /// Scores for all shot types.
    pub scores: Vec<(ShotType, f32)>,
}

/// Classifies the shot type of a single video frame.
pub struct ShotTypeClassifier {
    skin_close_up_threshold: f32,
    skin_medium_threshold: f32,
}

impl ShotTypeClassifier {
    /// Create a new shot type classifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skin_close_up_threshold: 0.30,
            skin_medium_threshold: 0.10,
        }
    }

    /// Classify the shot type of a single RGB frame.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Raw RGB pixel data
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions are inconsistent.
    pub fn classify(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<ShotClassification> {
        crate::classify::validate_frame(rgb, width, height)?;

        let features = self.extract_features(rgb, width, height);
        let mut scores: Vec<(ShotType, f32)> = vec![
            (ShotType::ExtremeCloseUp, self.score_ecu(&features)),
            (ShotType::CloseUp, self.score_cu(&features)),
            (ShotType::MediumCloseUp, self.score_mcu(&features)),
            (ShotType::Medium, self.score_medium(&features)),
            (ShotType::MediumWide, self.score_medium_wide(&features)),
            (ShotType::Wide, self.score_wide(&features)),
            (ShotType::Establishing, self.score_establishing(&features)),
            (ShotType::TwoShot, self.score_two_shot(&features)),
            (ShotType::OverTheShoulder, self.score_ots(&features)),
            (ShotType::PointOfView, self.score_pov(&features)),
        ];

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (shot_type, conf) = scores[0];
        Ok(ShotClassification {
            shot_type,
            confidence: Confidence::new(conf),
            features,
            scores,
        })
    }

    fn extract_features(&self, rgb: &[u8], width: usize, height: usize) -> ShotFeatures {
        let pixel_count = width * height;
        let mut skin_pixels = 0u32;
        let mut skin_row_min = height;
        let mut skin_row_max = 0usize;
        let mut skin_y_sum = 0u64;

        // Skin tone detection + vertical extent
        for y in 0..height {
            let mut row_skin = 0u32;
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let r = rgb[idx] as f32;
                let g = rgb[idx + 1] as f32;
                let b = rgb[idx + 2] as f32;
                if Self::is_skin(r, g, b) {
                    skin_pixels += 1;
                    row_skin += 1;
                }
            }
            if row_skin > 0 {
                skin_y_sum += y as u64 * row_skin as u64;
                if y < skin_row_min {
                    skin_row_min = y;
                }
                if y > skin_row_max {
                    skin_row_max = y;
                }
            }
        }

        let skin_ratio = skin_pixels as f32 / pixel_count as f32;
        let max_skin_region = if skin_pixels > 0 {
            let skin_height =
                (skin_row_max + 1).saturating_sub(skin_row_min) as f32 / height as f32;
            skin_height.min(skin_ratio * 4.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let face_vertical_pos = if skin_pixels > 0 {
            (skin_y_sum as f32 / (skin_pixels as f32 * height as f32)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Estimate face count by scanning for disconnected skin blobs
        let face_count_estimate = self.estimate_face_blobs(rgb, width, height, skin_ratio);

        // Depth-of-field blur: compare edge density in center vs periphery
        let (center_edges, peripheral_edges) = self.measure_depth_blur(rgb, width, height);
        let dof_blur = if center_edges + peripheral_edges > 0.0 {
            (1.0 - center_edges / (center_edges + peripheral_edges + f32::EPSILON)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let center_edge_ratio = if peripheral_edges > 0.0 {
            center_edges / peripheral_edges
        } else {
            1.0
        };

        // Horizontal symmetry using half-frame comparison
        let symmetry = self.compute_symmetry(rgb, width, height);

        ShotFeatures {
            skin_ratio,
            max_skin_region,
            face_vertical_pos,
            dof_blur,
            symmetry,
            center_edge_ratio,
            face_count_estimate,
        }
    }

    fn is_skin(r: f32, g: f32, b: f32) -> bool {
        // Based on empirical skin-tone heuristic in RGB space
        r > 95.0
            && g > 40.0
            && b > 20.0
            && r > g
            && r > b
            && (r - g) > 15.0
            && r < 250.0
            && g < 220.0
    }

    fn estimate_face_blobs(&self, rgb: &[u8], width: usize, height: usize, skin_ratio: f32) -> u32 {
        if skin_ratio < 0.01 {
            return 0;
        }
        // Simple column-scan: count transitions in skin-density horizontal bands
        let band_h = (height / 8).max(1);
        let mut face_like = 0u32;
        let mut y = 0;
        while y + band_h <= height {
            let mut band_skin = 0u32;
            for row in y..y + band_h {
                for x in 0..width {
                    let idx = (row * width + x) * 3;
                    let r = rgb[idx] as f32;
                    let g = rgb[idx + 1] as f32;
                    let b = rgb[idx + 2] as f32;
                    if Self::is_skin(r, g, b) {
                        band_skin += 1;
                    }
                }
            }
            let band_density = band_skin as f32 / (band_h * width) as f32;
            if band_density > 0.15 {
                face_like += 1;
            }
            y += band_h;
        }
        face_like.min(4)
    }

    fn measure_depth_blur(&self, rgb: &[u8], width: usize, height: usize) -> (f32, f32) {
        // Center third vs outer third edges using simple gradient magnitude
        let cx_start = width / 3;
        let cx_end = 2 * width / 3;

        let mut center_edge_sum = 0.0f32;
        let mut center_count = 0u32;
        let mut outer_edge_sum = 0.0f32;
        let mut outer_count = 0u32;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                let idx_r = (y * width + x + 1) * 3;
                let idx_d = ((y + 1) * width + x) * 3;

                let gx = (rgb[idx_r] as i32 - rgb[idx] as i32).unsigned_abs() as f32;
                let gy = (rgb[idx_d] as i32 - rgb[idx] as i32).unsigned_abs() as f32;
                let mag = (gx * gx + gy * gy).sqrt();

                if x >= cx_start && x < cx_end {
                    center_edge_sum += mag;
                    center_count += 1;
                } else if x < width / 6 || x > 5 * width / 6 {
                    outer_edge_sum += mag;
                    outer_count += 1;
                }
            }
        }

        let center = if center_count > 0 {
            center_edge_sum / center_count as f32
        } else {
            0.0
        };
        let outer = if outer_count > 0 {
            outer_edge_sum / outer_count as f32
        } else {
            0.0
        };
        (center, outer)
    }

    fn compute_symmetry(&self, rgb: &[u8], width: usize, height: usize) -> f32 {
        let mut diff_sum = 0.0f64;
        let sample_rows = (height / 4).max(1);
        let step = (height / sample_rows).max(1);
        let mut count = 0u64;

        for y in (0..height).step_by(step) {
            for x in 0..width / 2 {
                let mirror_x = width - 1 - x;
                let idx_l = (y * width + x) * 3;
                let idx_r = (y * width + mirror_x) * 3;
                for c in 0..3 {
                    let diff =
                        (rgb[idx_l + c] as i32 - rgb[idx_r + c] as i32).unsigned_abs() as f64;
                    diff_sum += diff;
                }
                count += 1;
            }
        }

        if count == 0 {
            return 0.5;
        }
        let mean_diff = diff_sum / (count as f64 * 255.0 * 3.0);
        (1.0 - mean_diff).clamp(0.0, 1.0) as f32
    }

    // Scoring functions
    fn score_ecu(&self, f: &ShotFeatures) -> f32 {
        // Very high skin ratio, single large region, high DOF blur
        (f.skin_ratio.min(1.0) * 2.0 - 1.0).max(0.0) * 0.5
            + f.dof_blur * 0.3
            + (f.max_skin_region * 0.2)
    }

    fn score_cu(&self, f: &ShotFeatures) -> f32 {
        let skin_score = if f.skin_ratio >= self.skin_close_up_threshold {
            ((f.skin_ratio - self.skin_close_up_threshold) / 0.3 + 0.4).clamp(0.0, 1.0)
        } else {
            0.0
        };
        skin_score * 0.6 + f.dof_blur * 0.4
    }

    fn score_mcu(&self, f: &ShotFeatures) -> f32 {
        let skin_in_range = (1.0 - (f.skin_ratio - 0.18).abs() / 0.15).clamp(0.0, 1.0);
        skin_in_range * 0.5 + f.dof_blur * 0.3 + f.symmetry * 0.2
    }

    fn score_medium(&self, f: &ShotFeatures) -> f32 {
        let skin_in_range = (1.0 - (f.skin_ratio - 0.08).abs() / 0.10).clamp(0.0, 1.0);
        skin_in_range * 0.5 + f.symmetry * 0.3 + (1.0 - f.dof_blur) * 0.2
    }

    fn score_medium_wide(&self, f: &ShotFeatures) -> f32 {
        let skin_in_range = (1.0 - (f.skin_ratio - 0.04).abs() / 0.06).clamp(0.0, 1.0);
        skin_in_range * 0.4 + (1.0 - f.dof_blur) * 0.4 + f.center_edge_ratio.min(1.0) * 0.2
    }

    fn score_wide(&self, f: &ShotFeatures) -> f32 {
        let low_skin = (1.0 - f.skin_ratio / self.skin_medium_threshold).clamp(0.0, 1.0);
        low_skin * 0.6 + (1.0 - f.dof_blur) * 0.4
    }

    fn score_establishing(&self, f: &ShotFeatures) -> f32 {
        let very_low_skin =
            (self.skin_medium_threshold - f.skin_ratio).max(0.0) / self.skin_medium_threshold;
        very_low_skin * 0.7 + (1.0 - f.dof_blur) * 0.3
    }

    fn score_two_shot(&self, f: &ShotFeatures) -> f32 {
        let two_faces = if f.face_count_estimate == 2 { 1.0 } else { 0.0 };
        two_faces * 0.6 + f.skin_ratio.clamp(0.0, 0.25) * 4.0 * 0.2 + f.dof_blur * 0.2
    }

    fn score_ots(&self, f: &ShotFeatures) -> f32 {
        // Asymmetric framing, moderate skin in one side
        let asymmetry = 1.0 - f.symmetry;
        asymmetry * 0.5 + f.skin_ratio.clamp(0.0, 0.2) * 5.0 * 0.3 + f.dof_blur * 0.2
    }

    fn score_pov(&self, f: &ShotFeatures) -> f32 {
        // Low skin, high edge density, slight blur
        let low_skin = (1.0 - f.skin_ratio / 0.05).clamp(0.0, 1.0);
        low_skin * 0.4 + f.center_edge_ratio.min(2.0) / 2.0 * 0.4 + (1.0 - f.symmetry) * 0.2
    }
}

impl Default for ShotTypeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    #[test]
    fn test_shot_type_labels() {
        assert_eq!(ShotType::CloseUp.label(), "Close-Up");
        assert_eq!(ShotType::Medium.label(), "Medium");
        assert_eq!(ShotType::Wide.label(), "Wide");
        assert_eq!(ShotType::Establishing.label(), "Establishing");
        assert_eq!(ShotType::TwoShot.label(), "Two-Shot");
    }

    #[test]
    fn test_is_close_is_wide() {
        assert!(ShotType::CloseUp.is_close());
        assert!(ShotType::ExtremeCloseUp.is_close());
        assert!(ShotType::Wide.is_wide());
        assert!(ShotType::Establishing.is_wide());
        assert!(!ShotType::Medium.is_close());
        assert!(!ShotType::Medium.is_wide());
    }

    #[test]
    fn test_classify_establishing_no_skin() {
        let classifier = ShotTypeClassifier::new();
        let frame = make_frame(30, 100, 60, 64, 64); // Green: no skin
        let result = classifier
            .classify(&frame, 64, 64)
            .expect("should succeed in test");
        assert!(result.shot_type.is_wide() || result.shot_type == ShotType::PointOfView);
    }

    #[test]
    fn test_classify_close_up_skin() {
        let classifier = ShotTypeClassifier::new();
        // Skin-like pixels: high R, moderate G, low B
        let mut frame = Vec::with_capacity(64 * 64 * 3);
        for _ in 0..64 * 64 {
            frame.push(200u8); // R
            frame.push(130u8); // G
            frame.push(80u8); // B
        }
        let result = classifier
            .classify(&frame, 64, 64)
            .expect("should succeed in test");
        assert!(result.features.skin_ratio > 0.1);
        assert!(
            result.shot_type.is_close()
                || result.shot_type == ShotType::MediumCloseUp
                || result.shot_type == ShotType::Medium
        );
    }

    #[test]
    fn test_features_ranges() {
        let classifier = ShotTypeClassifier::new();
        let frame = make_frame(100, 100, 100, 32, 32);
        let result = classifier
            .classify(&frame, 32, 32)
            .expect("should succeed in test");
        assert!(result.features.skin_ratio >= 0.0 && result.features.skin_ratio <= 1.0);
        assert!(result.features.dof_blur >= 0.0 && result.features.dof_blur <= 1.0);
        assert!(result.features.symmetry >= 0.0 && result.features.symmetry <= 1.0);
    }

    #[test]
    fn test_confidence_range() {
        let classifier = ShotTypeClassifier::new();
        let frame = make_frame(50, 50, 50, 32, 32);
        let result = classifier
            .classify(&frame, 32, 32)
            .expect("should succeed in test");
        assert!(result.confidence.value() >= 0.0 && result.confidence.value() <= 1.0);
    }

    #[test]
    fn test_invalid_frame() {
        let classifier = ShotTypeClassifier::new();
        let frame = vec![0u8; 5];
        assert!(classifier.classify(&frame, 64, 64).is_err());
    }

    #[test]
    fn test_scores_count() {
        let classifier = ShotTypeClassifier::new();
        let frame = make_frame(100, 100, 100, 32, 32);
        let result = classifier
            .classify(&frame, 32, 32)
            .expect("should succeed in test");
        assert_eq!(result.scores.len(), 10);
    }
}
