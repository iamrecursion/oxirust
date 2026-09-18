#![allow(dead_code)]

//! Framing composition guides: rule of thirds, golden ratio, center weighting,
//! and safe-area analysis for professional video production.

// ---------------------------------------------------------------------------
// Coordinate types
// ---------------------------------------------------------------------------

/// A 2D point in normalized coordinates (0.0 to 1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormPoint {
    /// Horizontal position (0.0 = left, 1.0 = right).
    pub x: f64,
    /// Vertical position (0.0 = top, 1.0 = bottom).
    pub y: f64,
}

impl NormPoint {
    /// Create a new normalized point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &NormPoint) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A rectangle in normalized coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormRect {
    /// Top-left X.
    pub x: f64,
    /// Top-left Y.
    pub y: f64,
    /// Width (0.0 to 1.0).
    pub w: f64,
    /// Height (0.0 to 1.0).
    pub h: f64,
}

impl NormRect {
    /// Create a new normalized rectangle.
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    /// Center point of the rectangle.
    pub fn center(&self) -> NormPoint {
        NormPoint::new(self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// Area of the rectangle.
    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    /// Check whether a point is inside the rectangle.
    pub fn contains(&self, p: &NormPoint) -> bool {
        p.x >= self.x && p.x <= self.x + self.w && p.y >= self.y && p.y <= self.y + self.h
    }
}

// ---------------------------------------------------------------------------
// Rule of Thirds
// ---------------------------------------------------------------------------

/// The four intersection points of a rule-of-thirds grid.
pub const THIRDS_POINTS: [NormPoint; 4] = [
    NormPoint {
        x: 1.0 / 3.0,
        y: 1.0 / 3.0,
    },
    NormPoint {
        x: 2.0 / 3.0,
        y: 1.0 / 3.0,
    },
    NormPoint {
        x: 1.0 / 3.0,
        y: 2.0 / 3.0,
    },
    NormPoint {
        x: 2.0 / 3.0,
        y: 2.0 / 3.0,
    },
];

/// Compute the rule-of-thirds score for a point of interest.
///
/// Returns a score in [0.0, 1.0] where 1.0 means the point sits
/// exactly on a thirds intersection.
pub fn thirds_score(point: &NormPoint) -> f64 {
    let min_dist = THIRDS_POINTS
        .iter()
        .map(|tp| point.distance_to(tp))
        .fold(f64::INFINITY, f64::min);

    // Maximum possible distance is ~0.943 (corner to far intersection).
    // Normalise so that distance 0 => score 1.0.
    let max_relevant = 0.25; // within 25% of frame dimension
    (1.0 - (min_dist / max_relevant).min(1.0)).max(0.0)
}

/// Evaluate rule-of-thirds adherence for multiple subjects.
pub fn thirds_score_multi(points: &[NormPoint]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let sum: f64 = points.iter().map(|p| thirds_score(p)).sum();
    sum / points.len() as f64
}

// ---------------------------------------------------------------------------
// Golden ratio
// ---------------------------------------------------------------------------

/// The golden ratio constant.
const PHI: f64 = 1.618_033_988_749_895;

/// Golden ratio intersection points (phi-based grid).
pub fn golden_ratio_points() -> [NormPoint; 4] {
    let a = 1.0 / (1.0 + PHI); // ~0.382
    let b = PHI / (1.0 + PHI); // ~0.618
    [
        NormPoint::new(a, a),
        NormPoint::new(b, a),
        NormPoint::new(a, b),
        NormPoint::new(b, b),
    ]
}

/// Compute golden ratio adherence score for a point of interest.
pub fn golden_ratio_score(point: &NormPoint) -> f64 {
    let gr_points = golden_ratio_points();
    let min_dist = gr_points
        .iter()
        .map(|gp| point.distance_to(gp))
        .fold(f64::INFINITY, f64::min);

    let max_relevant = 0.25;
    (1.0 - (min_dist / max_relevant).min(1.0)).max(0.0)
}

// ---------------------------------------------------------------------------
// Center weighting
// ---------------------------------------------------------------------------

/// Compute a center-weighted score for a subject position.
///
/// Returns 1.0 if the point is at the exact center, decaying towards 0.
pub fn center_weight_score(point: &NormPoint) -> f64 {
    let center = NormPoint::new(0.5, 0.5);
    let dist = point.distance_to(&center);
    // Max distance from center to corner is ~0.707
    let max_dist = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
    (1.0 - dist / max_dist).max(0.0)
}

// ---------------------------------------------------------------------------
// Safe area
// ---------------------------------------------------------------------------

/// Standard safe-area definitions for broadcast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeArea {
    /// Title-safe area (inner margin as fraction of frame, e.g. 0.1 = 10%).
    pub title_margin: f64,
    /// Action-safe area margin.
    pub action_margin: f64,
}

impl Default for SafeArea {
    fn default() -> Self {
        Self {
            title_margin: 0.1,   // 10% title-safe
            action_margin: 0.05, // 5% action-safe
        }
    }
}

impl SafeArea {
    /// Create safe area with custom margins.
    pub fn new(title_margin: f64, action_margin: f64) -> Self {
        Self {
            title_margin,
            action_margin,
        }
    }

    /// Get the title-safe rectangle.
    pub fn title_safe_rect(&self) -> NormRect {
        let m = self.title_margin;
        NormRect::new(m, m, 1.0 - 2.0 * m, 1.0 - 2.0 * m)
    }

    /// Get the action-safe rectangle.
    pub fn action_safe_rect(&self) -> NormRect {
        let m = self.action_margin;
        NormRect::new(m, m, 1.0 - 2.0 * m, 1.0 - 2.0 * m)
    }

    /// Check whether a point is within the title-safe area.
    pub fn is_title_safe(&self, point: &NormPoint) -> bool {
        self.title_safe_rect().contains(point)
    }

    /// Check whether a point is within the action-safe area.
    pub fn is_action_safe(&self, point: &NormPoint) -> bool {
        self.action_safe_rect().contains(point)
    }

    /// Compute what fraction of a rectangle lies within the action-safe area.
    ///
    /// Approximated by checking corner and center points.
    pub fn safe_coverage(&self, rect: &NormRect) -> f64 {
        let safe = self.action_safe_rect();
        let test_points = [
            NormPoint::new(rect.x, rect.y),
            NormPoint::new(rect.x + rect.w, rect.y),
            NormPoint::new(rect.x, rect.y + rect.h),
            NormPoint::new(rect.x + rect.w, rect.y + rect.h),
            rect.center(),
        ];
        let inside = test_points.iter().filter(|p| safe.contains(p)).count();
        inside as f64 / test_points.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Composite framing score
// ---------------------------------------------------------------------------

/// Weights for combining different composition scores.
#[derive(Debug, Clone)]
pub struct CompositionWeights {
    /// Weight for rule-of-thirds score.
    pub thirds: f64,
    /// Weight for golden ratio score.
    pub golden: f64,
    /// Weight for center weight score.
    pub center: f64,
}

impl Default for CompositionWeights {
    fn default() -> Self {
        Self {
            thirds: 0.5,
            golden: 0.3,
            center: 0.2,
        }
    }
}

/// Compute a composite framing score for a subject position.
pub fn composite_framing_score(point: &NormPoint, weights: &CompositionWeights) -> f64 {
    let ts = thirds_score(point);
    let gs = golden_ratio_score(point);
    let cs = center_weight_score(point);
    let total_weight = weights.thirds + weights.golden + weights.center;
    if total_weight <= 0.0 {
        return 0.0;
    }
    (ts * weights.thirds + gs * weights.golden + cs * weights.center) / total_weight
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- NormPoint tests --

    #[test]
    fn test_point_distance() {
        let a = NormPoint::new(0.0, 0.0);
        let b = NormPoint::new(1.0, 0.0);
        assert!((a.distance_to(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_distance_zero() {
        let a = NormPoint::new(0.5, 0.5);
        assert!((a.distance_to(&a) - 0.0).abs() < 1e-9);
    }

    // -- NormRect tests --

    #[test]
    fn test_rect_center() {
        let r = NormRect::new(0.0, 0.0, 1.0, 1.0);
        let c = r.center();
        assert!((c.x - 0.5).abs() < 1e-9);
        assert!((c.y - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_rect_area() {
        let r = NormRect::new(0.1, 0.2, 0.5, 0.3);
        assert!((r.area() - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_rect_contains() {
        let r = NormRect::new(0.2, 0.2, 0.6, 0.6);
        assert!(r.contains(&NormPoint::new(0.5, 0.5)));
        assert!(!r.contains(&NormPoint::new(0.1, 0.1)));
    }

    // -- Rule of thirds tests --

    #[test]
    fn test_thirds_score_at_intersection() {
        let p = NormPoint::new(1.0 / 3.0, 1.0 / 3.0);
        let score = thirds_score(&p);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_thirds_score_at_center() {
        let p = NormPoint::new(0.5, 0.5);
        let score = thirds_score(&p);
        // Center is NOT on a thirds intersection, but close
        assert!(score > 0.0);
        assert!(score < 1.0);
    }

    #[test]
    fn test_thirds_multi_empty() {
        assert!((thirds_score_multi(&[]) - 0.0).abs() < f64::EPSILON);
    }

    // -- Golden ratio tests --

    #[test]
    fn test_golden_ratio_points_valid() {
        let pts = golden_ratio_points();
        for p in &pts {
            assert!(p.x > 0.0 && p.x < 1.0);
            assert!(p.y > 0.0 && p.y < 1.0);
        }
    }

    #[test]
    fn test_golden_ratio_score_at_point() {
        let pts = golden_ratio_points();
        let score = golden_ratio_score(&pts[0]);
        assert!((score - 1.0).abs() < 1e-9);
    }

    // -- Center weight tests --

    #[test]
    fn test_center_weight_at_center() {
        let score = center_weight_score(&NormPoint::new(0.5, 0.5));
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_center_weight_at_corner() {
        let score = center_weight_score(&NormPoint::new(0.0, 0.0));
        assert!((score - 0.0).abs() < 1e-6);
    }

    // -- Safe area tests --

    #[test]
    fn test_safe_area_default() {
        let sa = SafeArea::default();
        assert!((sa.title_margin - 0.1).abs() < f64::EPSILON);
        assert!((sa.action_margin - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_title_safe_center() {
        let sa = SafeArea::default();
        assert!(sa.is_title_safe(&NormPoint::new(0.5, 0.5)));
    }

    #[test]
    fn test_title_safe_corner_outside() {
        let sa = SafeArea::default();
        assert!(!sa.is_title_safe(&NormPoint::new(0.01, 0.01)));
    }

    #[test]
    fn test_action_safe_rect() {
        let sa = SafeArea::new(0.1, 0.05);
        let rect = sa.action_safe_rect();
        assert!((rect.x - 0.05).abs() < 1e-9);
        assert!((rect.w - 0.9).abs() < 1e-9);
    }

    // -- Composite score tests --

    #[test]
    fn test_composite_at_center() {
        let p = NormPoint::new(0.5, 0.5);
        let score = composite_framing_score(&p, &CompositionWeights::default());
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_composite_zero_weights() {
        let p = NormPoint::new(0.5, 0.5);
        let w = CompositionWeights {
            thirds: 0.0,
            golden: 0.0,
            center: 0.0,
        };
        let score = composite_framing_score(&p, &w);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }
}
