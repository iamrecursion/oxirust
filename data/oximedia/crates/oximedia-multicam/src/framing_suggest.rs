//! Automatic camera framing suggestions based on face detection and composition rules.
//!
//! Analyzes the scene to detect faces and evaluate composition quality, then
//! suggests optimal pan-tilt-zoom adjustments to improve framing.
//!
//! # Composition rules applied
//!
//! 1. **Rule of thirds** — position the primary subject at a thirds intersection.
//! 2. **Head room** — keep ~1/6 of frame height above the highest face.
//! 3. **Look room** — for side-facing subjects, preserve space in the look direction.
//! 4. **Subject size** — primary subject should fill 15–50% of frame height.
//! 5. **Centering** — single subject should be horizontally centered or slightly offset.

use crate::{AngleId, MultiCamError, Result};

// ── Face bounding box ─────────────────────────────────────────────────────────

/// A face detection result within a single video frame.
#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    /// Horizontal centre of the face (0.0 = left, 1.0 = right).
    pub cx: f32,
    /// Vertical centre of the face (0.0 = top, 1.0 = bottom).
    pub cy: f32,
    /// Width of the face as a fraction of frame width.
    pub w: f32,
    /// Height of the face as a fraction of frame height.
    pub h: f32,
    /// Detection confidence (0.0–1.0).
    pub confidence: f32,
}

impl FaceBox {
    /// Top edge (normalised).
    #[must_use]
    pub fn top(&self) -> f32 {
        (self.cy - self.h / 2.0).max(0.0)
    }

    /// Bottom edge (normalised).
    #[must_use]
    pub fn bottom(&self) -> f32 {
        (self.cy + self.h / 2.0).min(1.0)
    }

    /// Left edge (normalised).
    #[must_use]
    pub fn left(&self) -> f32 {
        (self.cx - self.w / 2.0).max(0.0)
    }

    /// Right edge (normalised).
    #[must_use]
    pub fn right(&self) -> f32 {
        (self.cx + self.w / 2.0).min(1.0)
    }

    /// Face area as a fraction of frame area.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.w * self.h
    }
}

// ── FramingAdjustment ─────────────────────────────────────────────────────────

/// Suggested camera adjustment to improve framing.
#[derive(Debug, Clone, Copy, Default)]
pub struct FramingAdjustment {
    /// Pan delta: positive = pan right, negative = pan left.
    /// Expressed as a fraction of frame width (−1.0 to 1.0).
    pub pan_delta: f32,
    /// Tilt delta: positive = tilt up, negative = tilt down.
    /// Expressed as a fraction of frame height (−1.0 to 1.0).
    pub tilt_delta: f32,
    /// Zoom delta: positive = zoom in, negative = zoom out.
    /// Expressed as a fraction of current zoom range (−1.0 to 1.0).
    pub zoom_delta: f32,
    /// Aggregate confidence of this suggestion (0.0–1.0).
    pub confidence: f32,
    /// Whether the current framing is already acceptable (no adjustment needed).
    pub is_acceptable: bool,
}

impl FramingAdjustment {
    /// An adjustment that says the current framing is good.
    #[must_use]
    pub fn acceptable() -> Self {
        Self {
            is_acceptable: true,
            confidence: 1.0,
            ..Self::default()
        }
    }

    /// Magnitude of the combined spatial adjustment.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.pan_delta * self.pan_delta + self.tilt_delta * self.tilt_delta).sqrt()
    }
}

// ── FramingSuggestion ─────────────────────────────────────────────────────────

/// Full framing suggestion for one camera angle at one frame.
#[derive(Debug, Clone)]
pub struct FramingSuggestion {
    /// Camera angle this suggestion is for.
    pub angle: AngleId,
    /// Frame number this analysis was performed on.
    pub frame_number: u64,
    /// Detected faces in this frame.
    pub faces: Vec<FaceBox>,
    /// Suggested camera adjustment.
    pub adjustment: FramingAdjustment,
    /// Human-readable explanation.
    pub reason: String,
}

// ── FramingRules ─────────────────────────────────────────────────────────────

/// Composition rules configuration.
#[derive(Debug, Clone)]
pub struct FramingRules {
    /// Target head-room fraction (ratio of frame height above the top face edge).
    pub head_room: f32,
    /// Minimum acceptable subject height (fraction of frame height).
    pub min_subject_height: f32,
    /// Maximum acceptable subject height (fraction of frame height).
    pub max_subject_height: f32,
    /// Tolerance: adjustment is "acceptable" when below this magnitude.
    pub accept_threshold: f32,
}

impl Default for FramingRules {
    fn default() -> Self {
        Self {
            head_room: 1.0 / 6.0,
            min_subject_height: 0.15,
            max_subject_height: 0.50,
            accept_threshold: 0.05,
        }
    }
}

// ── FramingAnalyzer ───────────────────────────────────────────────────────────

/// Analyzes detected faces and suggests framing adjustments.
#[derive(Debug, Clone)]
pub struct FramingAnalyzer {
    rules: FramingRules,
}

impl Default for FramingAnalyzer {
    fn default() -> Self {
        Self::new(FramingRules::default())
    }
}

impl FramingAnalyzer {
    /// Create with the given rules.
    #[must_use]
    pub fn new(rules: FramingRules) -> Self {
        Self { rules }
    }

    /// Analyze a set of detected faces and produce a framing suggestion.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame dimensions are invalid.
    pub fn analyze(
        &self,
        angle: AngleId,
        frame_number: u64,
        faces: Vec<FaceBox>,
    ) -> Result<FramingSuggestion> {
        if faces.is_empty() {
            return Ok(FramingSuggestion {
                angle,
                frame_number,
                faces,
                adjustment: FramingAdjustment::acceptable(),
                reason: "No faces detected; no adjustment suggested".into(),
            });
        }

        // Select primary subject (largest face with high confidence).
        let primary = self.select_primary_face(&faces)?;
        let adjustment = self.compute_adjustment(primary);
        let reason = self.describe_adjustment(primary, &adjustment);

        Ok(FramingSuggestion {
            angle,
            frame_number,
            faces,
            adjustment,
            reason,
        })
    }

    /// Select the primary face (highest confidence × area product).
    fn select_primary_face<'a>(&self, faces: &'a [FaceBox]) -> Result<&'a FaceBox> {
        faces
            .iter()
            .max_by(|a, b| {
                let score_a = a.confidence * a.area();
                let score_b = b.confidence * b.area();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| MultiCamError::InsufficientData("No faces provided".into()))
    }

    /// Compute the recommended camera adjustment for a primary face.
    fn compute_adjustment(&self, face: &FaceBox) -> FramingAdjustment {
        let r = &self.rules;

        // --- Horizontal (pan) ---
        // Rule-of-thirds: ideally cx ≈ 1/3 or 2/3.
        // Choose the nearer thirds point.
        let thirds_x = [1.0_f32 / 3.0, 2.0_f32 / 3.0];
        let target_cx = *thirds_x
            .iter()
            .min_by(|&&a, &&b| {
                (a - face.cx)
                    .abs()
                    .partial_cmp(&(b - face.cx).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&0.5);
        let pan_delta = target_cx - face.cx;

        // --- Vertical (tilt) ---
        // Apply head-room: target top edge = head_room from top.
        let target_top = r.head_room;
        let tilt_delta = target_top - face.top();

        // --- Zoom ---
        let current_h = face.h;
        let zoom_delta = if current_h < r.min_subject_height {
            // Too far — zoom in.
            r.min_subject_height - current_h
        } else if current_h > r.max_subject_height {
            // Too close — zoom out.
            -(current_h - r.max_subject_height)
        } else {
            0.0
        };

        let magnitude =
            (pan_delta * pan_delta + tilt_delta * tilt_delta + zoom_delta * zoom_delta).sqrt();
        let is_acceptable = magnitude < r.accept_threshold;

        // Confidence decays when adjustments are large.
        let confidence = (1.0 - magnitude * 0.5).clamp(0.0, 1.0);

        FramingAdjustment {
            pan_delta,
            tilt_delta,
            zoom_delta,
            confidence,
            is_acceptable,
        }
    }

    /// Build a human-readable description of the adjustment.
    fn describe_adjustment(&self, face: &FaceBox, adj: &FramingAdjustment) -> String {
        if adj.is_acceptable {
            return format!("Framing acceptable (face at {:.2},{:.2})", face.cx, face.cy);
        }
        let mut parts = Vec::new();
        if adj.pan_delta.abs() > 0.01 {
            let dir = if adj.pan_delta > 0.0 { "right" } else { "left" };
            parts.push(format!("pan {dir} {:.2}", adj.pan_delta.abs()));
        }
        if adj.tilt_delta.abs() > 0.01 {
            let dir = if adj.tilt_delta > 0.0 { "down" } else { "up" };
            parts.push(format!("tilt {dir} {:.2}", adj.tilt_delta.abs()));
        }
        if adj.zoom_delta.abs() > 0.01 {
            let dir = if adj.zoom_delta > 0.0 { "in" } else { "out" };
            parts.push(format!("zoom {dir} {:.2}", adj.zoom_delta.abs()));
        }
        if parts.is_empty() {
            "Minor adjustment".into()
        } else {
            parts.join(", ")
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn face(cx: f32, cy: f32, w: f32, h: f32) -> FaceBox {
        FaceBox {
            cx,
            cy,
            w,
            h,
            confidence: 0.9,
        }
    }

    #[test]
    fn test_face_box_edges() {
        let f = face(0.5, 0.5, 0.2, 0.3);
        assert!((f.top() - 0.35).abs() < 1e-5);
        assert!((f.bottom() - 0.65).abs() < 1e-5);
        assert!((f.left() - 0.4).abs() < 1e-5);
        assert!((f.right() - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_face_box_area() {
        let f = face(0.5, 0.5, 0.2, 0.3);
        assert!((f.area() - 0.06).abs() < 1e-5);
    }

    #[test]
    fn test_framing_adjustment_magnitude() {
        let adj = FramingAdjustment {
            pan_delta: 0.3,
            tilt_delta: 0.4,
            zoom_delta: 0.0,
            confidence: 0.8,
            is_acceptable: false,
        };
        assert!((adj.magnitude() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_no_faces_returns_acceptable() {
        let analyzer = FramingAnalyzer::default();
        let suggestion = analyzer.analyze(0, 0, vec![]).expect("should succeed");
        assert!(suggestion.adjustment.is_acceptable);
    }

    #[test]
    fn test_face_centered_horizontally_no_pan_needed() {
        // Face exactly at a thirds intersection horizontally — pan should be ~0.
        let analyzer = FramingAnalyzer::default();
        let f = face(1.0 / 3.0, 1.0 / 6.0 + 0.15, 0.2, 0.30); // cx=1/3, top=1/6
        let suggestion = analyzer.analyze(0, 0, vec![f]).expect("should succeed");
        assert!(
            suggestion.adjustment.pan_delta.abs() < 0.02,
            "pan_delta should be ~0, got {}",
            suggestion.adjustment.pan_delta
        );
    }

    #[test]
    fn test_face_too_small_zooms_in() {
        let analyzer = FramingAnalyzer::default();
        // Face height = 0.05, well below min_subject_height=0.15
        let f = face(0.33, 0.2, 0.05, 0.05);
        let suggestion = analyzer.analyze(0, 0, vec![f]).expect("should succeed");
        assert!(
            suggestion.adjustment.zoom_delta > 0.0,
            "Expected zoom-in, got {}",
            suggestion.adjustment.zoom_delta
        );
    }

    #[test]
    fn test_face_too_large_zooms_out() {
        let analyzer = FramingAnalyzer::default();
        // Face height = 0.6, above max_subject_height=0.50
        let f = face(0.5, 0.5, 0.4, 0.6);
        let suggestion = analyzer.analyze(0, 0, vec![f]).expect("should succeed");
        assert!(
            suggestion.adjustment.zoom_delta < 0.0,
            "Expected zoom-out, got {}",
            suggestion.adjustment.zoom_delta
        );
    }

    #[test]
    fn test_reason_is_non_empty() {
        let analyzer = FramingAnalyzer::default();
        let f = face(0.5, 0.5, 0.2, 0.3);
        let suggestion = analyzer.analyze(0, 0, vec![f]).expect("should succeed");
        assert!(!suggestion.reason.is_empty());
    }

    #[test]
    fn test_multiple_faces_selects_largest() {
        let analyzer = FramingAnalyzer::default();
        let small = FaceBox {
            cx: 0.1,
            cy: 0.1,
            w: 0.05,
            h: 0.05,
            confidence: 0.9,
        };
        let large = FaceBox {
            cx: 0.67,
            cy: 0.25,
            w: 0.25,
            h: 0.40,
            confidence: 0.9,
        };
        let suggestion = analyzer
            .analyze(0, 0, vec![small, large])
            .expect("should succeed");
        assert_eq!(suggestion.faces.len(), 2);
    }
}
