#![allow(dead_code)]
//! Programmatic path construction for vector graphics.
//!
//! Provides a builder API for constructing 2D paths composed of lines, arcs,
//! quadratic and cubic Bezier curves, used in broadcast graphics overlays.

/// A 2D point.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2D {
    /// Create a new point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Linearly interpolate between this point and another.
    pub fn lerp(&self, other: &Point2D, t: f64) -> Point2D {
        Point2D {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

/// Path segment types.
#[derive(Clone, Debug, PartialEq)]
pub enum PathSegment {
    /// Move to a point without drawing.
    MoveTo(Point2D),
    /// Draw a line to a point.
    LineTo(Point2D),
    /// Quadratic Bezier curve with a control point and endpoint.
    QuadTo {
        /// Control point.
        control: Point2D,
        /// End point.
        end: Point2D,
    },
    /// Cubic Bezier curve with two control points and endpoint.
    CubicTo {
        /// First control point.
        c1: Point2D,
        /// Second control point.
        c2: Point2D,
        /// End point.
        end: Point2D,
    },
    /// Close the current sub-path.
    Close,
}

/// A constructed path consisting of segments.
#[derive(Clone, Debug, Default)]
pub struct Path2D {
    /// The path segments.
    segments: Vec<PathSegment>,
}

impl Path2D {
    /// Get the segments as a slice.
    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// Get the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Check if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Check if the path is closed (last segment is Close).
    pub fn is_closed(&self) -> bool {
        matches!(self.segments.last(), Some(PathSegment::Close))
    }

    /// Compute the bounding box of the path as `(min_x, min_y, max_x, max_y)`.
    pub fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for seg in &self.segments {
            let points: Vec<Point2D> = match seg {
                PathSegment::MoveTo(p) | PathSegment::LineTo(p) => vec![*p],
                PathSegment::QuadTo { control, end } => vec![*control, *end],
                PathSegment::CubicTo { c1, c2, end } => vec![*c1, *c2, *end],
                PathSegment::Close => vec![],
            };
            for p in points {
                if p.x < min_x {
                    min_x = p.x;
                }
                if p.y < min_y {
                    min_y = p.y;
                }
                if p.x > max_x {
                    max_x = p.x;
                }
                if p.y > max_y {
                    max_y = p.y;
                }
            }
        }
        if min_x > max_x {
            return (0.0, 0.0, 0.0, 0.0);
        }
        (min_x, min_y, max_x, max_y)
    }

    /// Approximate the total length of the path by linearizing curves.
    pub fn approximate_length(&self) -> f64 {
        let mut length = 0.0;
        let mut current = Point2D::new(0.0, 0.0);
        let mut sub_start = current;

        for seg in &self.segments {
            match seg {
                PathSegment::MoveTo(p) => {
                    current = *p;
                    sub_start = *p;
                }
                PathSegment::LineTo(p) => {
                    length += current.distance_to(p);
                    current = *p;
                }
                PathSegment::QuadTo { control, end } => {
                    // Approximate with 10 line segments
                    let steps = 10;
                    let mut prev = current;
                    for i in 1..=steps {
                        let t = i as f64 / steps as f64;
                        let a = current.lerp(control, t);
                        let b = control.lerp(end, t);
                        let pt = a.lerp(&b, t);
                        length += prev.distance_to(&pt);
                        prev = pt;
                    }
                    current = *end;
                }
                PathSegment::CubicTo { c1, c2, end } => {
                    let steps = 16;
                    let mut prev = current;
                    for i in 1..=steps {
                        let t = i as f64 / steps as f64;
                        let t2 = t * t;
                        let t3 = t2 * t;
                        let mt = 1.0 - t;
                        let mt2 = mt * mt;
                        let mt3 = mt2 * mt;
                        let pt = Point2D {
                            x: mt3 * current.x
                                + 3.0 * mt2 * t * c1.x
                                + 3.0 * mt * t2 * c2.x
                                + t3 * end.x,
                            y: mt3 * current.y
                                + 3.0 * mt2 * t * c1.y
                                + 3.0 * mt * t2 * c2.y
                                + t3 * end.y,
                        };
                        length += prev.distance_to(&pt);
                        prev = pt;
                    }
                    current = *end;
                }
                PathSegment::Close => {
                    length += current.distance_to(&sub_start);
                    current = sub_start;
                }
            }
        }
        length
    }
}

/// Builder for constructing 2D paths step by step.
#[derive(Clone, Debug, Default)]
pub struct PathBuilder {
    /// Accumulated segments.
    segments: Vec<PathSegment>,
    /// Current position.
    current: Point2D,
}

impl PathBuilder {
    /// Create a new path builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the pen to a point.
    pub fn move_to(mut self, x: f64, y: f64) -> Self {
        let p = Point2D::new(x, y);
        self.segments.push(PathSegment::MoveTo(p));
        self.current = p;
        self
    }

    /// Draw a line to a point.
    pub fn line_to(mut self, x: f64, y: f64) -> Self {
        let p = Point2D::new(x, y);
        self.segments.push(PathSegment::LineTo(p));
        self.current = p;
        self
    }

    /// Draw a quadratic Bezier curve.
    pub fn quad_to(mut self, cx: f64, cy: f64, x: f64, y: f64) -> Self {
        let control = Point2D::new(cx, cy);
        let end = Point2D::new(x, y);
        self.segments.push(PathSegment::QuadTo { control, end });
        self.current = end;
        self
    }

    /// Draw a cubic Bezier curve.
    pub fn cubic_to(mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64) -> Self {
        let c1 = Point2D::new(c1x, c1y);
        let c2 = Point2D::new(c2x, c2y);
        let end = Point2D::new(x, y);
        self.segments.push(PathSegment::CubicTo { c1, c2, end });
        self.current = end;
        self
    }

    /// Close the current sub-path.
    pub fn close(mut self) -> Self {
        self.segments.push(PathSegment::Close);
        self
    }

    /// Add a rectangle sub-path.
    pub fn rect(self, x: f64, y: f64, w: f64, h: f64) -> Self {
        self.move_to(x, y)
            .line_to(x + w, y)
            .line_to(x + w, y + h)
            .line_to(x, y + h)
            .close()
    }

    /// Add a rounded rectangle sub-path.
    #[allow(clippy::too_many_arguments)]
    pub fn rounded_rect(self, x: f64, y: f64, w: f64, h: f64, r: f64) -> Self {
        let r = r.min(w / 2.0).min(h / 2.0);
        self.move_to(x + r, y)
            .line_to(x + w - r, y)
            .quad_to(x + w, y, x + w, y + r)
            .line_to(x + w, y + h - r)
            .quad_to(x + w, y + h, x + w - r, y + h)
            .line_to(x + r, y + h)
            .quad_to(x, y + h, x, y + h - r)
            .line_to(x, y + r)
            .quad_to(x, y, x + r, y)
            .close()
    }

    /// Add a circle sub-path approximated with cubic Bezier curves.
    #[allow(clippy::cast_precision_loss)]
    pub fn circle(self, cx: f64, cy: f64, radius: f64) -> Self {
        // Approximate a circle with 4 cubic Bezier segments.
        // Magic number: 4*(sqrt(2)-1)/3
        let k = radius * 0.5522847498;
        self.move_to(cx + radius, cy)
            .cubic_to(cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius)
            .cubic_to(cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy)
            .cubic_to(cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius)
            .cubic_to(cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy)
            .close()
    }

    /// Build the final path.
    pub fn build(self) -> Path2D {
        Path2D {
            segments: self.segments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_lerp() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(10.0, 20.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
        assert!((mid.y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_path() {
        let p = PathBuilder::new().build();
        assert!(p.is_empty());
        assert_eq!(p.segment_count(), 0);
    }

    #[test]
    fn test_line_path() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .build();
        assert_eq!(p.segment_count(), 2);
        assert!(!p.is_closed());
    }

    #[test]
    fn test_closed_path() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(10.0, 10.0)
            .close()
            .build();
        assert!(p.is_closed());
    }

    #[test]
    fn test_rect_path() {
        let p = PathBuilder::new().rect(0.0, 0.0, 100.0, 50.0).build();
        assert!(p.is_closed());
        assert_eq!(p.segment_count(), 5); // move + 3 line + close
    }

    #[test]
    fn test_rounded_rect_path() {
        let p = PathBuilder::new()
            .rounded_rect(0.0, 0.0, 100.0, 50.0, 10.0)
            .build();
        assert!(p.is_closed());
        // move + line + quad + line + quad + line + quad + line + quad + close = 10
        assert_eq!(p.segment_count(), 10);
    }

    #[test]
    fn test_circle_path() {
        let p = PathBuilder::new().circle(50.0, 50.0, 25.0).build();
        assert!(p.is_closed());
        // move + 4 cubic + close = 6
        assert_eq!(p.segment_count(), 6);
    }

    #[test]
    fn test_bounding_box_line() {
        let p = PathBuilder::new()
            .move_to(10.0, 20.0)
            .line_to(30.0, 40.0)
            .build();
        let (min_x, min_y, max_x, max_y) = p.bounding_box();
        assert!((min_x - 10.0).abs() < 1e-10);
        assert!((min_y - 20.0).abs() < 1e-10);
        assert!((max_x - 30.0).abs() < 1e-10);
        assert!((max_y - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_empty() {
        let p = PathBuilder::new().build();
        let bb = p.bounding_box();
        assert_eq!(bb, (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_approximate_length_line() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(3.0, 4.0)
            .build();
        assert!((p.approximate_length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_approximate_length_closed_triangle() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(10.0, 0.0)
            .line_to(5.0, 5.0)
            .close()
            .build();
        let len = p.approximate_length();
        // Should be > 20 (perimeter of the triangle)
        assert!(len > 20.0);
    }

    #[test]
    fn test_quad_to() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .quad_to(5.0, 10.0, 10.0, 0.0)
            .build();
        assert_eq!(p.segment_count(), 2);
        let len = p.approximate_length();
        // Should be longer than straight line (10)
        assert!(len > 10.0);
    }

    #[test]
    fn test_cubic_to() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .cubic_to(5.0, 20.0, 15.0, 20.0, 20.0, 0.0)
            .build();
        assert_eq!(p.segment_count(), 2);
        let len = p.approximate_length();
        // Should be longer than straight line (20)
        assert!(len > 20.0);
    }

    #[test]
    fn test_circle_approximate_length() {
        let r = 50.0;
        let p = PathBuilder::new().circle(0.0, 0.0, r).build();
        let len = p.approximate_length();
        let expected = 2.0 * std::f64::consts::PI * r;
        // Bezier circle approximation should be within 0.5% of true circumference
        assert!((len - expected).abs() / expected < 0.005);
    }

    #[test]
    fn test_path_segments_slice() {
        let p = PathBuilder::new()
            .move_to(0.0, 0.0)
            .line_to(1.0, 1.0)
            .build();
        let segs = p.segments();
        assert_eq!(segs.len(), 2);
        assert!(matches!(segs[0], PathSegment::MoveTo(_)));
        assert!(matches!(segs[1], PathSegment::LineTo(_)));
    }
}
