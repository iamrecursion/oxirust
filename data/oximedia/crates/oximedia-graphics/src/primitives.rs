//! 2D graphics primitives

use crate::color::{Color, Gradient};
use kurbo::{BezPath, Circle as KurboCircle, Line as KurboLine, Rect as KurboRect};
use serde::{Deserialize, Serialize};

/// 2D point
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
}

impl Point {
    /// Create a new point
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Distance to another point
    #[must_use]
    pub fn distance(&self, other: &Point) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Midpoint between two points
    #[must_use]
    pub fn midpoint(&self, other: &Point) -> Point {
        Point::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }
}

impl From<(f32, f32)> for Point {
    fn from((x, y): (f32, f32)) -> Self {
        Self::new(x, y)
    }
}

/// 2D size
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Size {
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl Size {
    /// Create a new size
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Area
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// Rectangle
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl Rect {
    /// Create a new rectangle
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create from position and size
    #[must_use]
    pub fn from_pos_size(pos: Point, size: Size) -> Self {
        Self::new(pos.x, pos.y, size.width, size.height)
    }

    /// Get center point
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Check if contains point
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }

    /// Inset by amount
    #[must_use]
    pub fn inset(&self, amount: f32) -> Self {
        Self::new(
            self.x + amount,
            self.y + amount,
            self.width - amount * 2.0,
            self.height - amount * 2.0,
        )
    }

    /// Convert to kurbo rect
    #[must_use]
    pub fn to_kurbo(&self) -> KurboRect {
        KurboRect::new(
            f64::from(self.x),
            f64::from(self.y),
            f64::from(self.x + self.width),
            f64::from(self.y + self.height),
        )
    }
}

/// Circle
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    /// Center X
    pub x: f32,
    /// Center Y
    pub y: f32,
    /// Radius
    pub radius: f32,
}

impl Circle {
    /// Create a new circle
    #[must_use]
    pub const fn new(x: f32, y: f32, radius: f32) -> Self {
        Self { x, y, radius }
    }

    /// Create from center point
    #[must_use]
    pub fn from_center(center: Point, radius: f32) -> Self {
        Self::new(center.x, center.y, radius)
    }

    /// Get center point
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(self.x, self.y)
    }

    /// Check if contains point
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        let dx = point.x - self.x;
        let dy = point.y - self.y;
        dx * dx + dy * dy <= self.radius * self.radius
    }

    /// Convert to kurbo circle
    #[must_use]
    pub fn to_kurbo(&self) -> KurboCircle {
        KurboCircle::new(
            (f64::from(self.x), f64::from(self.y)),
            f64::from(self.radius),
        )
    }
}

/// Line
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Line {
    /// Start point
    pub start: Point,
    /// End point
    pub end: Point,
}

impl Line {
    /// Create a new line
    #[must_use]
    pub const fn new(start: Point, end: Point) -> Self {
        Self { start, end }
    }

    /// Length of the line
    #[must_use]
    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }

    /// Convert to kurbo line
    #[must_use]
    pub fn to_kurbo(&self) -> KurboLine {
        KurboLine::new(
            (f64::from(self.start.x), f64::from(self.start.y)),
            (f64::from(self.end.x), f64::from(self.end.y)),
        )
    }
}

/// Bezier curve control points
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BezierCurve {
    /// Start point
    pub start: Point,
    /// First control point
    pub control1: Point,
    /// Second control point
    pub control2: Point,
    /// End point
    pub end: Point,
}

impl BezierCurve {
    /// Create a new cubic Bezier curve
    #[must_use]
    pub const fn new(start: Point, control1: Point, control2: Point, end: Point) -> Self {
        Self {
            start,
            control1,
            control2,
            end,
        }
    }

    /// Evaluate curve at t (0.0 to 1.0)
    #[must_use]
    pub fn eval(&self, t: f32) -> Point {
        let t = t.clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        Point::new(
            mt3 * self.start.x
                + 3.0 * mt2 * t * self.control1.x
                + 3.0 * mt * t2 * self.control2.x
                + t3 * self.end.x,
            mt3 * self.start.y
                + 3.0 * mt2 * t * self.control1.y
                + 3.0 * mt * t2 * self.control2.y
                + t3 * self.end.y,
        )
    }
}

/// Vector path for complex shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// Path commands
    pub commands: Vec<PathCommand>,
}

/// Path command
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PathCommand {
    /// Move to point
    MoveTo(Point),
    /// Line to point
    LineTo(Point),
    /// Quadratic curve
    QuadTo(Point, Point),
    /// Cubic curve
    CubicTo(Point, Point, Point),
    /// Close path
    Close,
}

impl Path {
    /// Create a new empty path
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Move to point
    pub fn move_to(&mut self, point: Point) -> &mut Self {
        self.commands.push(PathCommand::MoveTo(point));
        self
    }

    /// Line to point
    pub fn line_to(&mut self, point: Point) -> &mut Self {
        self.commands.push(PathCommand::LineTo(point));
        self
    }

    /// Quadratic curve
    pub fn quad_to(&mut self, control: Point, end: Point) -> &mut Self {
        self.commands.push(PathCommand::QuadTo(control, end));
        self
    }

    /// Cubic curve
    pub fn cubic_to(&mut self, control1: Point, control2: Point, end: Point) -> &mut Self {
        self.commands
            .push(PathCommand::CubicTo(control1, control2, end));
        self
    }

    /// Close path
    pub fn close(&mut self) -> &mut Self {
        self.commands.push(PathCommand::Close);
        self
    }

    /// Convert to kurbo path
    #[must_use]
    pub fn to_kurbo(&self) -> BezPath {
        let mut path = BezPath::new();
        for cmd in &self.commands {
            match cmd {
                PathCommand::MoveTo(p) => {
                    path.move_to((f64::from(p.x), f64::from(p.y)));
                }
                PathCommand::LineTo(p) => {
                    path.line_to((f64::from(p.x), f64::from(p.y)));
                }
                PathCommand::QuadTo(c, e) => {
                    path.quad_to(
                        (f64::from(c.x), f64::from(c.y)),
                        (f64::from(e.x), f64::from(e.y)),
                    );
                }
                PathCommand::CubicTo(c1, c2, e) => {
                    path.curve_to(
                        (f64::from(c1.x), f64::from(c1.y)),
                        (f64::from(c2.x), f64::from(c2.y)),
                        (f64::from(e.x), f64::from(e.y)),
                    );
                }
                PathCommand::Close => {
                    path.close_path();
                }
            }
        }
        path
    }

    /// Create a rounded rectangle path
    #[must_use]
    pub fn rounded_rect(rect: Rect, radius: f32) -> Self {
        let mut path = Self::new();

        let r = radius.min(rect.width / 2.0).min(rect.height / 2.0);

        path.move_to(Point::new(rect.x + r, rect.y));
        path.line_to(Point::new(rect.x + rect.width - r, rect.y));
        path.quad_to(
            Point::new(rect.x + rect.width, rect.y),
            Point::new(rect.x + rect.width, rect.y + r),
        );
        path.line_to(Point::new(rect.x + rect.width, rect.y + rect.height - r));
        path.quad_to(
            Point::new(rect.x + rect.width, rect.y + rect.height),
            Point::new(rect.x + rect.width - r, rect.y + rect.height),
        );
        path.line_to(Point::new(rect.x + r, rect.y + rect.height));
        path.quad_to(
            Point::new(rect.x, rect.y + rect.height),
            Point::new(rect.x, rect.y + rect.height - r),
        );
        path.line_to(Point::new(rect.x, rect.y + r));
        path.quad_to(Point::new(rect.x, rect.y), Point::new(rect.x + r, rect.y));
        path.close();

        path
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

/// Fill style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Fill {
    /// Solid color
    Solid(Color),
    /// Gradient fill
    Gradient(Gradient),
}

impl Fill {
    /// Create solid fill
    #[must_use]
    pub fn solid(color: Color) -> Self {
        Self::Solid(color)
    }

    /// Create gradient fill
    #[must_use]
    pub fn gradient(gradient: Gradient) -> Self {
        Self::Gradient(gradient)
    }
}

/// Stroke style
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    /// Stroke color
    pub color: Color,
    /// Stroke width
    pub width: f32,
    /// Line cap
    pub cap: LineCap,
    /// Line join
    pub join: LineJoin,
}

impl Stroke {
    /// Create a new stroke
    #[must_use]
    pub fn new(color: Color, width: f32) -> Self {
        Self {
            color,
            width,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        }
    }

    /// Set line cap
    #[must_use]
    pub fn with_cap(mut self, cap: LineCap) -> Self {
        self.cap = cap;
        self
    }

    /// Set line join
    #[must_use]
    pub fn with_join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }
}

/// Line cap style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineCap {
    /// Butt cap
    Butt,
    /// Round cap
    Round,
    /// Square cap
    Square,
}

/// Line join style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineJoin {
    /// Miter join
    Miter,
    /// Round join
    Round,
    /// Bevel join
    Bevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1.distance(&p2), 5.0);

        let mid = p1.midpoint(&p2);
        assert_eq!(mid, Point::new(1.5, 2.0));
    }

    #[test]
    fn test_size() {
        let size = Size::new(10.0, 20.0);
        assert_eq!(size.area(), 200.0);
    }

    #[test]
    fn test_rect() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.center(), Point::new(60.0, 45.0));
        assert!(rect.contains(Point::new(50.0, 40.0)));
        assert!(!rect.contains(Point::new(5.0, 5.0)));

        let inset = rect.inset(10.0);
        assert_eq!(inset.width, 80.0);
        assert_eq!(inset.height, 30.0);
    }

    #[test]
    fn test_circle() {
        let circle = Circle::new(50.0, 50.0, 25.0);
        assert_eq!(circle.center(), Point::new(50.0, 50.0));
        assert!(circle.contains(Point::new(50.0, 50.0)));
        assert!(circle.contains(Point::new(60.0, 50.0)));
        assert!(!circle.contains(Point::new(100.0, 50.0)));
    }

    #[test]
    fn test_line() {
        let line = Line::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0));
        assert_eq!(line.length(), 5.0);
    }

    #[test]
    fn test_bezier_curve() {
        let curve = BezierCurve::new(
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(3.0, 2.0),
            Point::new(4.0, 0.0),
        );

        let start = curve.eval(0.0);
        assert_eq!(start, Point::new(0.0, 0.0));

        let end = curve.eval(1.0);
        assert_eq!(end, Point::new(4.0, 0.0));

        let mid = curve.eval(0.5);
        assert!(mid.y > 0.0); // Should be above the line
    }

    #[test]
    fn test_path() {
        let mut path = Path::new();
        path.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(100.0, 0.0))
            .line_to(Point::new(100.0, 100.0))
            .close();

        assert_eq!(path.commands.len(), 4);
    }

    #[test]
    fn test_rounded_rect_path() {
        let rect = Rect::new(10.0, 10.0, 100.0, 50.0);
        let path = Path::rounded_rect(rect, 5.0);
        assert!(!path.commands.is_empty());
    }

    #[test]
    fn test_fill() {
        let fill = Fill::solid(Color::RED);
        assert!(matches!(fill, Fill::Solid(_)));

        let gradient = Gradient::linear(
            (0.0, 0.0),
            (100.0, 0.0),
            vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
        );
        let fill = Fill::gradient(gradient);
        assert!(matches!(fill, Fill::Gradient(_)));
    }

    #[test]
    fn test_stroke() {
        let stroke = Stroke::new(Color::BLACK, 2.0)
            .with_cap(LineCap::Round)
            .with_join(LineJoin::Round);

        assert_eq!(stroke.width, 2.0);
        assert_eq!(stroke.cap, LineCap::Round);
        assert_eq!(stroke.join, LineJoin::Round);
    }
}
