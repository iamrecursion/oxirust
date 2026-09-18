//! Bezier curve-based mask tools.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// A point on a Bezier curve with control handles.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BezierPoint {
    /// Point X position.
    pub x: f32,
    /// Point Y position.
    pub y: f32,
    /// In-tangent handle X (relative to point).
    pub handle_in_x: f32,
    /// In-tangent handle Y (relative to point).
    pub handle_in_y: f32,
    /// Out-tangent handle X (relative to point).
    pub handle_out_x: f32,
    /// Out-tangent handle Y (relative to point).
    pub handle_out_y: f32,
}

impl BezierPoint {
    /// Create a new Bezier point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            handle_in_x: 0.0,
            handle_in_y: 0.0,
            handle_out_x: 0.0,
            handle_out_y: 0.0,
        }
    }

    /// Create with handles.
    #[must_use]
    pub const fn with_handles(mut self, in_x: f32, in_y: f32, out_x: f32, out_y: f32) -> Self {
        self.handle_in_x = in_x;
        self.handle_in_y = in_y;
        self.handle_out_x = out_x;
        self.handle_out_y = out_y;
        self
    }

    /// Make handles symmetric.
    pub fn make_symmetric(&mut self) {
        self.handle_out_x = -self.handle_in_x;
        self.handle_out_y = -self.handle_in_y;
    }

    /// Make handles smooth (aligned but independent lengths).
    pub fn make_smooth(&mut self) {
        let in_len =
            (self.handle_in_x * self.handle_in_x + self.handle_in_y * self.handle_in_y).sqrt();
        let out_len =
            (self.handle_out_x * self.handle_out_x + self.handle_out_y * self.handle_out_y).sqrt();

        if in_len > 0.0 {
            let scale = out_len / in_len;
            self.handle_out_x = -self.handle_in_x * scale;
            self.handle_out_y = -self.handle_in_y * scale;
        }
    }
}

/// A Bezier curve segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BezierCurve {
    points: Vec<BezierPoint>,
    closed: bool,
}

impl BezierCurve {
    /// Create a new empty curve.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            points: Vec::new(),
            closed: false,
        }
    }

    /// Create from points.
    #[must_use]
    pub fn from_points(points: Vec<BezierPoint>) -> Self {
        Self {
            points,
            closed: false,
        }
    }

    /// Add a point to the curve.
    pub fn add_point(&mut self, point: BezierPoint) {
        self.points.push(point);
    }

    /// Set whether curve is closed.
    pub fn set_closed(&mut self, closed: bool) {
        self.closed = closed;
    }

    /// Get number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if curve is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get reference to points.
    #[must_use]
    pub fn points(&self) -> &[BezierPoint] {
        &self.points
    }

    /// Get mutable reference to points.
    #[must_use]
    pub fn points_mut(&mut self) -> &mut Vec<BezierPoint> {
        &mut self.points
    }

    /// Evaluate curve at parameter t [0, 1] for segment i.
    #[must_use]
    pub fn evaluate(&self, segment: usize, t: f32) -> Option<(f32, f32)> {
        if segment >= self.segment_count() {
            return None;
        }

        let p0 = &self.points[segment];
        let p1 = if self.closed && segment == self.points.len() - 1 {
            &self.points[0]
        } else {
            &self.points[segment + 1]
        };

        // Cubic Bezier: P(t) = (1-t)³P0 + 3(1-t)²t·H1 + 3(1-t)t²·H2 + t³P1
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        let h1_x = p0.x + p0.handle_out_x;
        let h1_y = p0.y + p0.handle_out_y;
        let h2_x = p1.x + p1.handle_in_x;
        let h2_y = p1.y + p1.handle_in_y;

        let x = mt3 * p0.x + 3.0 * mt2 * t * h1_x + 3.0 * mt * t2 * h2_x + t3 * p1.x;
        let y = mt3 * p0.y + 3.0 * mt2 * t * h1_y + 3.0 * mt * t2 * h2_y + t3 * p1.y;

        Some((x, y))
    }

    /// Get number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        if self.points.len() < 2 {
            0
        } else if self.closed {
            self.points.len()
        } else {
            self.points.len() - 1
        }
    }

    /// Sample curve into line segments.
    #[must_use]
    pub fn sample(&self, steps_per_segment: usize) -> Vec<(f32, f32)> {
        let mut samples = Vec::new();

        for i in 0..self.segment_count() {
            for step in 0..steps_per_segment {
                let t = step as f32 / steps_per_segment as f32;
                if let Some(point) = self.evaluate(i, t) {
                    samples.push(point);
                }
            }
        }

        // Add final point
        if !self.closed && !self.points.is_empty() {
            let last = &self.points[self.points.len() - 1];
            samples.push((last.x, last.y));
        }

        samples
    }
}

impl Default for BezierCurve {
    fn default() -> Self {
        Self::new()
    }
}

/// A complete mask shape with feather.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BezierMask {
    /// Mask curve.
    pub curve: BezierCurve,
    /// Feather amount.
    pub feather: f32,
    /// Mask opacity.
    pub opacity: f32,
    /// Invert mask.
    pub inverted: bool,
}

impl BezierMask {
    /// Create a new mask.
    #[must_use]
    pub fn new(curve: BezierCurve) -> Self {
        Self {
            curve,
            feather: 0.0,
            opacity: 1.0,
            inverted: false,
        }
    }

    /// Set feather amount.
    #[must_use]
    pub fn with_feather(mut self, feather: f32) -> Self {
        self.feather = feather.max(0.0);
        self
    }

    /// Set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set inverted flag.
    #[must_use]
    pub const fn with_inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    /// Render mask to frame.
    pub fn render(&self, output: &mut Frame) -> VfxResult<()> {
        // Sample curve
        let samples = self.curve.sample(10);
        if samples.len() < 3 {
            return Ok(());
        }

        // Rasterize polygon
        for y in 0..output.height {
            for x in 0..output.width {
                let inside = self.is_point_inside(x as f32, y as f32, &samples);
                let distance = if inside {
                    -self.distance_to_curve(x as f32, y as f32, &samples)
                } else {
                    self.distance_to_curve(x as f32, y as f32, &samples)
                };

                // Apply feather
                let alpha = if self.feather > 0.0 {
                    let normalized = (-distance / self.feather).clamp(0.0, 1.0);
                    normalized
                } else if inside {
                    1.0
                } else {
                    0.0
                };

                // Apply opacity
                let alpha = alpha * self.opacity;

                // Apply inversion
                let alpha = if self.inverted { 1.0 - alpha } else { alpha };

                let pixel = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                output.set_pixel(x, y, [pixel[0], pixel[1], pixel[2], (alpha * 255.0) as u8]);
            }
        }

        Ok(())
    }

    fn is_point_inside(&self, x: f32, y: f32, samples: &[(f32, f32)]) -> bool {
        // Ray casting algorithm
        let mut inside = false;
        let mut j = samples.len() - 1;

        for i in 0..samples.len() {
            let (xi, yi) = samples[i];
            let (xj, yj) = samples[j];

            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    fn distance_to_curve(&self, x: f32, y: f32, samples: &[(f32, f32)]) -> f32 {
        let mut min_dist = f32::MAX;

        for i in 0..samples.len() {
            let (x1, y1) = samples[i];
            let (x2, y2) = if i + 1 < samples.len() {
                samples[i + 1]
            } else if self.curve.closed {
                samples[0]
            } else {
                continue;
            };

            let dist = self.point_to_segment_distance(x, y, x1, y1, x2, y2);
            min_dist = min_dist.min(dist);
        }

        min_dist
    }

    fn point_to_segment_distance(
        &self,
        px: f32,
        py: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> f32 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        if len_sq == 0.0 {
            return ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
        }

        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        let t = t.clamp(0.0, 1.0);

        let proj_x = x1 + t * dx;
        let proj_y = y1 + t * dy;

        ((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezier_point() {
        let point = BezierPoint::new(10.0, 20.0);
        assert_eq!(point.x, 10.0);
        assert_eq!(point.y, 20.0);
    }

    #[test]
    fn test_bezier_curve_creation() {
        let mut curve = BezierCurve::new();
        curve.add_point(BezierPoint::new(0.0, 0.0));
        curve.add_point(BezierPoint::new(10.0, 10.0));
        assert_eq!(curve.len(), 2);
        assert_eq!(curve.segment_count(), 1);
    }

    #[test]
    fn test_bezier_curve_closed() {
        let mut curve = BezierCurve::new();
        curve.add_point(BezierPoint::new(0.0, 0.0));
        curve.add_point(BezierPoint::new(10.0, 0.0));
        curve.add_point(BezierPoint::new(10.0, 10.0));
        curve.set_closed(true);
        assert_eq!(curve.segment_count(), 3);
    }

    #[test]
    fn test_bezier_curve_evaluate() {
        let mut curve = BezierCurve::new();
        curve.add_point(BezierPoint::new(0.0, 0.0));
        curve.add_point(BezierPoint::new(10.0, 10.0));

        let (x, y) = curve.evaluate(0, 0.5).expect("should succeed in test");
        assert!(x > 0.0 && x < 10.0);
        assert!(y > 0.0 && y < 10.0);
    }

    #[test]
    fn test_bezier_mask() -> VfxResult<()> {
        let mut curve = BezierCurve::new();
        curve.add_point(BezierPoint::new(10.0, 10.0));
        curve.add_point(BezierPoint::new(90.0, 10.0));
        curve.add_point(BezierPoint::new(90.0, 90.0));
        curve.add_point(BezierPoint::new(10.0, 90.0));
        curve.set_closed(true);

        let mask = BezierMask::new(curve).with_feather(5.0);
        let mut frame = Frame::new(100, 100)?;
        mask.render(&mut frame)?;
        Ok(())
    }
}
