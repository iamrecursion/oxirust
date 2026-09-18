//! Frame composition and framing guide analysis for `oximedia-shots`.
//!
//! Provides [`FrameComposition`] classification, [`AspectRatioGuide`] helpers,
//! and a [`FramingAnalyzer`] that scores how well a given shot adheres to
//! classical framing rules (rule-of-thirds, headroom, lead-room, etc.).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Frame composition style
// ---------------------------------------------------------------------------

/// High-level framing style detected in a shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameComposition {
    /// Subject is centred in the frame.
    Centered,
    /// Subject follows the rule-of-thirds grid.
    RuleOfThirds,
    /// Frame exhibits strong bilateral symmetry.
    Symmetrical,
    /// Subjects positioned to create diagonal tension.
    DiagonalTension,
    /// Frame makes use of natural leading lines.
    LeadingLines,
    /// Deep staging with distinct foreground, mid, and background.
    DeepStaging,
    /// Framing that does not clearly match any canonical style.
    Unclassified,
}

impl FrameComposition {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Centered => "Centered",
            Self::RuleOfThirds => "Rule of Thirds",
            Self::Symmetrical => "Symmetrical",
            Self::DiagonalTension => "Diagonal Tension",
            Self::LeadingLines => "Leading Lines",
            Self::DeepStaging => "Deep Staging",
            Self::Unclassified => "Unclassified",
        }
    }

    /// Returns all known composition variants.
    #[must_use]
    pub const fn all() -> &'static [FrameComposition] {
        &[
            Self::Centered,
            Self::RuleOfThirds,
            Self::Symmetrical,
            Self::DiagonalTension,
            Self::LeadingLines,
            Self::DeepStaging,
            Self::Unclassified,
        ]
    }
}

impl std::fmt::Display for FrameComposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Aspect-ratio guide
// ---------------------------------------------------------------------------

/// Common aspect-ratio presets used for framing guides.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AspectRatioGuide {
    /// Horizontal component of the ratio (e.g. 16 in 16:9).
    pub width: f64,
    /// Vertical component of the ratio (e.g. 9 in 16:9).
    pub height: f64,
    /// Short descriptive name (e.g. `"Cinemascope"`).
    pub name: &'static str,
}

impl AspectRatioGuide {
    /// Standard 16:9 widescreen.
    pub const WIDESCREEN: Self = Self {
        width: 16.0,
        height: 9.0,
        name: "16:9 Widescreen",
    };
    /// Cinemascope / anamorphic 2.39:1.
    pub const CINEMASCOPE: Self = Self {
        width: 2.39,
        height: 1.0,
        name: "Cinemascope 2.39:1",
    };
    /// Classic 4:3 / academy.
    pub const ACADEMY: Self = Self {
        width: 4.0,
        height: 3.0,
        name: "4:3 Academy",
    };
    /// 1:1 square (social media).
    pub const SQUARE: Self = Self {
        width: 1.0,
        height: 1.0,
        name: "1:1 Square",
    };
    /// 9:16 vertical (mobile-first).
    pub const VERTICAL: Self = Self {
        width: 9.0,
        height: 16.0,
        name: "9:16 Vertical",
    };

    /// Returns the aspect ratio as a single floating-point number.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio(&self) -> f64 {
        if self.height.abs() < f64::EPSILON {
            return 0.0;
        }
        self.width / self.height
    }

    /// Checks whether a given pixel dimension matches this guide within a
    /// tolerance (default 2 %).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn matches(&self, pixel_w: u32, pixel_h: u32, tolerance: f64) -> bool {
        if pixel_h == 0 {
            return false;
        }
        let actual = f64::from(pixel_w) / f64::from(pixel_h);
        (actual - self.ratio()).abs() / self.ratio() <= tolerance
    }
}

// ---------------------------------------------------------------------------
// Framing analyzer
// ---------------------------------------------------------------------------

/// Scores representing how well a frame adheres to common composition rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FramingScores {
    /// Rule-of-thirds alignment score in `[0.0, 1.0]`.
    pub thirds_score: f64,
    /// Headroom adequacy score in `[0.0, 1.0]`.
    pub headroom_score: f64,
    /// Lead-room adequacy score in `[0.0, 1.0]`.
    pub lead_room_score: f64,
    /// Symmetry score in `[0.0, 1.0]`.
    pub symmetry_score: f64,
    /// Overall weighted composite score in `[0.0, 1.0]`.
    pub overall: f64,
}

impl Default for FramingScores {
    fn default() -> Self {
        Self {
            thirds_score: 0.0,
            headroom_score: 0.0,
            lead_room_score: 0.0,
            symmetry_score: 0.0,
            overall: 0.0,
        }
    }
}

impl FramingScores {
    /// Returns `true` if the overall score exceeds the given threshold.
    #[must_use]
    pub fn is_good(&self, threshold: f64) -> bool {
        self.overall >= threshold
    }
}

/// Analyzes frame-level composition from subject position data.
///
/// The analyzer uses normalised coordinates `(x, y)` where both axes are
/// in `[0.0, 1.0]` (top-left origin).
#[derive(Debug, Clone)]
pub struct FramingAnalyzer {
    /// Weight for thirds alignment in composite score.
    thirds_weight: f64,
    /// Weight for headroom in composite score.
    headroom_weight: f64,
    /// Weight for lead-room in composite score.
    lead_room_weight: f64,
    /// Weight for symmetry in composite score.
    symmetry_weight: f64,
}

impl Default for FramingAnalyzer {
    fn default() -> Self {
        Self {
            thirds_weight: 0.35,
            headroom_weight: 0.25,
            lead_room_weight: 0.25,
            symmetry_weight: 0.15,
        }
    }
}

impl FramingAnalyzer {
    /// Creates a new analyzer with custom weights.
    #[must_use]
    pub fn new(
        thirds_weight: f64,
        headroom_weight: f64,
        lead_room_weight: f64,
        symmetry_weight: f64,
    ) -> Self {
        Self {
            thirds_weight,
            headroom_weight,
            lead_room_weight,
            symmetry_weight,
        }
    }

    /// Score thirds alignment: how close `x` is to 1/3 or 2/3 of the frame.
    #[must_use]
    fn thirds_alignment(x: f64, y: f64) -> f64 {
        let dx = (x - 1.0 / 3.0).abs().min((x - 2.0 / 3.0).abs());
        let dy = (y - 1.0 / 3.0).abs().min((y - 2.0 / 3.0).abs());
        let dist = (dx * dx + dy * dy).sqrt();
        (1.0 - dist * 3.0).max(0.0)
    }

    /// Score headroom: ideal headroom places the top-of-head near y ~ 0.08..0.15.
    #[must_use]
    fn headroom_score(subject_top_y: f64) -> f64 {
        let ideal = 0.10;
        let deviation = (subject_top_y - ideal).abs();
        (1.0 - deviation * 5.0).max(0.0)
    }

    /// Score lead-room: if subject faces right, more room on the right is better.
    #[must_use]
    fn lead_room_score(subject_x: f64, facing_right: bool) -> f64 {
        if facing_right {
            // More room on right → subject_x should be small
            (1.0 - subject_x).min(1.0).max(0.0)
        } else {
            subject_x.min(1.0).max(0.0)
        }
    }

    /// Score symmetry: how close a subject is to the horizontal centre.
    #[must_use]
    fn symmetry_score(subject_x: f64) -> f64 {
        let dev = (subject_x - 0.5).abs();
        (1.0 - dev * 2.0).max(0.0)
    }

    /// Analyzes a frame given normalised subject coordinates.
    ///
    /// * `subject_x`, `subject_y` - centre of primary subject `[0..1]`.
    /// * `subject_top_y` - top of subject bounding box `[0..1]`.
    /// * `facing_right` - whether the subject faces camera-right.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(
        &self,
        subject_x: f64,
        subject_y: f64,
        subject_top_y: f64,
        facing_right: bool,
    ) -> FramingScores {
        let thirds = Self::thirds_alignment(subject_x, subject_y);
        let headroom = Self::headroom_score(subject_top_y);
        let lead = Self::lead_room_score(subject_x, facing_right);
        let sym = Self::symmetry_score(subject_x);

        let total_weight = self.thirds_weight
            + self.headroom_weight
            + self.lead_room_weight
            + self.symmetry_weight;
        let overall = if total_weight.abs() < f64::EPSILON {
            0.0
        } else {
            (thirds * self.thirds_weight
                + headroom * self.headroom_weight
                + lead * self.lead_room_weight
                + sym * self.symmetry_weight)
                / total_weight
        };

        FramingScores {
            thirds_score: thirds,
            headroom_score: headroom,
            lead_room_score: lead,
            symmetry_score: sym,
            overall,
        }
    }

    /// Classifies the dominant framing style from scores.
    #[must_use]
    pub fn classify(&self, scores: &FramingScores) -> FrameComposition {
        if scores.symmetry_score > 0.85 {
            FrameComposition::Symmetrical
        } else if scores.thirds_score > 0.75 {
            FrameComposition::RuleOfThirds
        } else if scores.thirds_score < 0.3 && scores.symmetry_score > 0.6 {
            FrameComposition::Centered
        } else {
            FrameComposition::Unclassified
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- FrameComposition ---------------------------------------------------

    #[test]
    fn test_frame_composition_label() {
        assert_eq!(FrameComposition::Centered.label(), "Centered");
        assert_eq!(FrameComposition::RuleOfThirds.label(), "Rule of Thirds");
        assert_eq!(FrameComposition::DeepStaging.label(), "Deep Staging");
    }

    #[test]
    fn test_frame_composition_display() {
        assert_eq!(
            format!("{}", FrameComposition::LeadingLines),
            "Leading Lines"
        );
    }

    #[test]
    fn test_frame_composition_all_variants() {
        let all = FrameComposition::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&FrameComposition::Unclassified));
    }

    // -- AspectRatioGuide ---------------------------------------------------

    #[test]
    fn test_aspect_ratio_widescreen() {
        let r = AspectRatioGuide::WIDESCREEN.ratio();
        assert!((r - 16.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_aspect_ratio_square() {
        let r = AspectRatioGuide::SQUARE.ratio();
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_aspect_ratio_zero_height() {
        let guide = AspectRatioGuide {
            width: 16.0,
            height: 0.0,
            name: "bad",
        };
        assert_eq!(guide.ratio(), 0.0);
    }

    #[test]
    fn test_aspect_ratio_matches_true() {
        assert!(AspectRatioGuide::WIDESCREEN.matches(1920, 1080, 0.02));
    }

    #[test]
    fn test_aspect_ratio_matches_false() {
        assert!(!AspectRatioGuide::WIDESCREEN.matches(1080, 1080, 0.02));
    }

    #[test]
    fn test_aspect_ratio_matches_zero_pixel_h() {
        assert!(!AspectRatioGuide::WIDESCREEN.matches(1920, 0, 0.02));
    }

    // -- FramingScores ------------------------------------------------------

    #[test]
    fn test_framing_scores_default() {
        let s = FramingScores::default();
        assert_eq!(s.overall, 0.0);
    }

    #[test]
    fn test_framing_scores_is_good() {
        let s = FramingScores {
            overall: 0.8,
            ..Default::default()
        };
        assert!(s.is_good(0.7));
        assert!(!s.is_good(0.9));
    }

    // -- FramingAnalyzer ----------------------------------------------------

    #[test]
    fn test_analyzer_default_creation() {
        let a = FramingAnalyzer::default();
        assert!((a.thirds_weight - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_thirds_alignment_perfect() {
        let score = FramingAnalyzer::thirds_alignment(1.0 / 3.0, 1.0 / 3.0);
        assert!(score > 0.95, "score was {score}");
    }

    #[test]
    fn test_thirds_alignment_centre() {
        // Centre (0.5, 0.5) is away from thirds intersections.
        let score = FramingAnalyzer::thirds_alignment(0.5, 0.5);
        assert!(score < 0.7, "score was {score}");
    }

    #[test]
    fn test_headroom_ideal() {
        let score = FramingAnalyzer::headroom_score(0.10);
        assert!(score > 0.95, "score was {score}");
    }

    #[test]
    fn test_lead_room_facing_right() {
        // Subject on left side facing right → good lead room.
        let score = FramingAnalyzer::lead_room_score(0.2, true);
        assert!(score > 0.7, "score was {score}");
    }

    #[test]
    fn test_classify_symmetrical() {
        let a = FramingAnalyzer::default();
        let scores = FramingScores {
            thirds_score: 0.4,
            headroom_score: 0.8,
            lead_room_score: 0.5,
            symmetry_score: 0.9,
            overall: 0.7,
        };
        assert_eq!(a.classify(&scores), FrameComposition::Symmetrical);
    }

    #[test]
    fn test_classify_rule_of_thirds() {
        let a = FramingAnalyzer::default();
        let scores = FramingScores {
            thirds_score: 0.8,
            headroom_score: 0.6,
            lead_room_score: 0.5,
            symmetry_score: 0.3,
            overall: 0.55,
        };
        assert_eq!(a.classify(&scores), FrameComposition::RuleOfThirds);
    }

    #[test]
    fn test_analyze_composite_score_bounded() {
        let a = FramingAnalyzer::default();
        let s = a.analyze(0.33, 0.33, 0.10, true);
        assert!(
            s.overall >= 0.0 && s.overall <= 1.0,
            "overall={}",
            s.overall
        );
    }
}
