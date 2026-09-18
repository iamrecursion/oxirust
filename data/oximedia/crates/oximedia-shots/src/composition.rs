//! Shot composition analysis.
//!
//! Provides tools for evaluating classic cinematographic composition techniques:
//! rule of thirds, leading lines, symmetry, and an overall quality score.

#![allow(dead_code)]

/// The four "power points" of the rule-of-thirds grid.
///
/// Each intersection of the grid lines is a natural focal point.
#[derive(Debug, Clone, Copy)]
pub struct RuleOfThirds {
    /// Four power-point coordinates as (x, y) in normalised [0, 1] space.
    pub power_points: [(f32, f32); 4],
}

impl RuleOfThirds {
    /// Return the standard rule-of-thirds power points.
    ///
    /// The grid divides the frame at 1/3 and 2/3 along each axis, yielding
    /// four intersections.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            power_points: [
                (1.0 / 3.0, 1.0 / 3.0),
                (2.0 / 3.0, 1.0 / 3.0),
                (1.0 / 3.0, 2.0 / 3.0),
                (2.0 / 3.0, 2.0 / 3.0),
            ],
        }
    }

    /// Return the power point closest to `(x, y)`.
    #[must_use]
    pub fn nearest_power_point(&self, x: f32, y: f32) -> (f32, f32) {
        let mut best = self.power_points[0];
        let mut best_dist = dist_sq(x, y, best.0, best.1);
        for &pp in &self.power_points[1..] {
            let d = dist_sq(x, y, pp.0, pp.1);
            if d < best_dist {
                best_dist = d;
                best = pp;
            }
        }
        best
    }

    /// Score how well a subject at `(subject_x, subject_y)` aligns with the
    /// rule of thirds.
    ///
    /// Returns a value in `[0, 1]` where `1.0` means the subject is exactly on
    /// a power point, and `0.0` means it is in the furthest possible position
    /// (centre of the frame, distance ≈ 0.47 from the nearest power point).
    #[must_use]
    pub fn score(&self, subject_x: f32, subject_y: f32) -> f32 {
        let nearest = self.nearest_power_point(subject_x, subject_y);
        let d = dist_sq(subject_x, subject_y, nearest.0, nearest.1).sqrt();
        // Maximum possible distance from any power point is approximately 0.47
        // (from frame centre to corner), but we use 0.5 as a practical cap.
        let normalised = (d / 0.5).min(1.0);
        1.0 - normalised
    }
}

fn dist_sq(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    (x2 - x1).powi(2) + (y2 - y1).powi(2)
}

/// A set of leading lines detected in a frame.
#[derive(Debug, Clone, Default)]
pub struct LeadingLines {
    /// Each element is `((x1, y1), (x2, y2))` in normalised [0, 1] frame space.
    pub lines: Vec<((f32, f32), (f32, f32))>,
}

impl LeadingLines {
    /// Create an empty `LeadingLines` set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a line from `(x1, y1)` to `(x2, y2)`.
    pub fn add_line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.lines.push(((x1, y1), (x2, y2)));
    }

    /// Return the average intersection point of all pairs of lines, or `None`
    /// if there are fewer than two lines.
    ///
    /// Lines that are parallel (or near-parallel) are skipped.
    #[must_use]
    pub fn converge_at(&self) -> Option<(f32, f32)> {
        if self.lines.len() < 2 {
            return None;
        }

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut count = 0u32;

        for i in 0..self.lines.len() {
            for j in (i + 1)..self.lines.len() {
                if let Some((ix, iy)) = line_intersect(&self.lines[i], &self.lines[j]) {
                    sum_x += ix;
                    sum_y += iy;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return None;
        }
        Some((sum_x / count as f32, sum_y / count as f32))
    }

    /// Return the average angle of all lines in degrees (0 – 180).
    ///
    /// Returns `0.0` if there are no lines.
    #[must_use]
    pub fn dominant_direction(&self) -> f32 {
        if self.lines.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .lines
            .iter()
            .map(|&((x1, y1), (x2, y2))| {
                let angle_rad = (y2 - y1).atan2(x2 - x1);
                let degrees = angle_rad.to_degrees();
                // Normalise to [0, 180)
                if degrees < 0.0 {
                    degrees + 180.0
                } else {
                    degrees
                }
            })
            .sum();
        sum / self.lines.len() as f32
    }
}

/// Return the intersection of two line segments, or `None` if they are
/// parallel / near-parallel.
fn line_intersect(
    a: &((f32, f32), (f32, f32)),
    b: &((f32, f32), (f32, f32)),
) -> Option<(f32, f32)> {
    let ((x1, y1), (x2, y2)) = *a;
    let ((x3, y3), (x4, y4)) = *b;

    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denom.abs() < 1e-6 {
        return None; // parallel
    }
    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    let ix = x1 + t * (x2 - x1);
    let iy = y1 + t * (y2 - y1);
    Some((ix, iy))
}

/// Computes a symmetry score by comparing two equal-length byte slices.
pub struct SymmetryScore;

impl SymmetryScore {
    /// Compute a normalised cross-correlation score between `left_half` and
    /// `right_half`.
    ///
    /// Returns a value in `[0, 1]` where `1.0` indicates perfect symmetry.
    /// Returns `0.0` if either slice is empty.
    #[must_use]
    pub fn compute(left_half: &[u8], right_half: &[u8]) -> f32 {
        let n = left_half.len().min(right_half.len());
        if n == 0 {
            return 0.0;
        }

        let mut dot = 0.0f64;
        let mut norm_l = 0.0f64;
        let mut norm_r = 0.0f64;

        for i in 0..n {
            let l = f64::from(left_half[i]);
            let r = f64::from(right_half[i]);
            dot += l * r;
            norm_l += l * l;
            norm_r += r * r;
        }

        let denom = (norm_l * norm_r).sqrt();
        if denom < 1e-12 {
            return 0.0;
        }
        (dot / denom) as f32
    }
}

/// A summary of composition quality metrics for a single frame.
#[derive(Debug, Clone, Copy)]
pub struct CompositionReport {
    /// Rule-of-thirds alignment score (0 – 1).
    pub thirds_score: f32,
    /// Symmetry score (0 – 1).
    pub symmetry: f32,
    /// Whether significant leading lines were detected.
    pub has_leading_lines: bool,
}

impl CompositionReport {
    /// Create a new report.
    #[must_use]
    pub fn new(thirds_score: f32, symmetry: f32, has_leading_lines: bool) -> Self {
        Self {
            thirds_score,
            symmetry,
            has_leading_lines,
        }
    }

    /// Compute an overall composition quality score.
    ///
    /// Weights: thirds 40 %, symmetry 40 %, leading lines bonus 20 %.
    #[must_use]
    pub fn overall_quality(&self) -> f32 {
        let lines_bonus = if self.has_leading_lines { 1.0 } else { 0.0 };
        (self.thirds_score * 0.4 + self.symmetry * 0.4 + lines_bonus * 0.2).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RuleOfThirds ---

    #[test]
    fn test_standard_power_points_count() {
        let rot = RuleOfThirds::standard();
        assert_eq!(rot.power_points.len(), 4);
    }

    #[test]
    fn test_nearest_power_point_top_left() {
        let rot = RuleOfThirds::standard();
        let (px, py) = rot.nearest_power_point(0.3, 0.3);
        assert!((px - 1.0 / 3.0).abs() < 0.01);
        assert!((py - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_score_at_power_point_is_one() {
        let rot = RuleOfThirds::standard();
        let score = rot.score(1.0 / 3.0, 1.0 / 3.0);
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_score_at_centre_is_low() {
        let rot = RuleOfThirds::standard();
        let score = rot.score(0.5, 0.5);
        // Centre is ~0.24 from nearest pp; score should be less than 0.6
        assert!(score < 0.6);
    }

    // --- LeadingLines ---

    #[test]
    fn test_add_line_and_count() {
        let mut ll = LeadingLines::new();
        ll.add_line(0.0, 0.0, 1.0, 1.0);
        assert_eq!(ll.lines.len(), 1);
    }

    #[test]
    fn test_converge_at_none_single_line() {
        let mut ll = LeadingLines::new();
        ll.add_line(0.0, 0.0, 1.0, 1.0);
        assert!(ll.converge_at().is_none());
    }

    #[test]
    fn test_converge_at_none_empty() {
        let ll = LeadingLines::new();
        assert!(ll.converge_at().is_none());
    }

    #[test]
    fn test_converge_at_two_lines() {
        let mut ll = LeadingLines::new();
        // Diagonal from top-left to bottom-right
        ll.add_line(0.0, 0.0, 1.0, 1.0);
        // Diagonal from top-right to bottom-left
        ll.add_line(1.0, 0.0, 0.0, 1.0);
        // These intersect at (0.5, 0.5)
        let conv = ll.converge_at().expect("should succeed in test");
        assert!((conv.0 - 0.5).abs() < 0.01);
        assert!((conv.1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_converge_at_parallel_lines_none() {
        let mut ll = LeadingLines::new();
        ll.add_line(0.0, 0.0, 1.0, 0.0); // horizontal
        ll.add_line(0.0, 1.0, 1.0, 1.0); // parallel horizontal
        assert!(ll.converge_at().is_none());
    }

    #[test]
    fn test_dominant_direction_empty() {
        let ll = LeadingLines::new();
        assert_eq!(ll.dominant_direction(), 0.0);
    }

    #[test]
    fn test_dominant_direction_horizontal() {
        let mut ll = LeadingLines::new();
        ll.add_line(0.0, 0.5, 1.0, 0.5); // perfectly horizontal → 0°
        assert!((ll.dominant_direction()).abs() < 1.0);
    }

    // --- SymmetryScore ---

    #[test]
    fn test_symmetry_identical_slices() {
        let data = vec![128u8; 100];
        let score = SymmetryScore::compute(&data, &data);
        assert!((score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_symmetry_empty_slices() {
        assert_eq!(SymmetryScore::compute(&[], &[]), 0.0);
    }

    #[test]
    fn test_symmetry_different_slices() {
        let left = vec![255u8; 10];
        let right = vec![0u8; 10];
        let score = SymmetryScore::compute(&left, &right);
        assert!(score < 0.1);
    }

    // --- CompositionReport ---

    #[test]
    fn test_overall_quality_max() {
        let r = CompositionReport::new(1.0, 1.0, true);
        assert!((r.overall_quality() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_overall_quality_no_lines() {
        let r = CompositionReport::new(1.0, 1.0, false);
        // 0.4 + 0.4 + 0.0 = 0.8
        assert!((r.overall_quality() - 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_overall_quality_zero() {
        let r = CompositionReport::new(0.0, 0.0, false);
        assert_eq!(r.overall_quality(), 0.0);
    }
}
