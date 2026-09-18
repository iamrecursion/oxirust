#![allow(dead_code)]
//! Vector shape rendering primitives for broadcast graphics.
//!
//! Provides a set of drawable shapes including rectangles, rounded rectangles,
//! circles, ellipses, polygons, and paths for composing broadcast graphics
//! elements. Shapes support stroke, fill, and shadow properties.

use std::f32::consts::PI;

/// RGBA color for shape rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeColor {
    /// Red channel (0.0..=1.0).
    pub r: f32,
    /// Green channel (0.0..=1.0).
    pub g: f32,
    /// Blue channel (0.0..=1.0).
    pub b: f32,
    /// Alpha channel (0.0..=1.0).
    pub a: f32,
}

impl ShapeColor {
    /// Create a new color with the given RGBA values.
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
            a: a.clamp(0.0, 1.0),
        }
    }

    /// Create an opaque color.
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    /// Create a fully transparent color.
    pub fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Black color.
    pub fn black() -> Self {
        Self::rgb(0.0, 0.0, 0.0)
    }

    /// White color.
    pub fn white() -> Self {
        Self::rgb(1.0, 1.0, 1.0)
    }

    /// Premultiply alpha.
    pub fn premultiplied(&self) -> Self {
        Self {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }

    /// Linear interpolation between two colors.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

impl Default for ShapeColor {
    fn default() -> Self {
        Self::white()
    }
}

/// Line cap style.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineCap {
    /// Flat cap (no extension).
    #[default]
    Butt,
    /// Rounded cap.
    Round,
    /// Square cap (extends by half stroke width).
    Square,
}

/// Line join style.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineJoin {
    /// Mitered join.
    #[default]
    Miter,
    /// Rounded join.
    Round,
    /// Beveled join.
    Bevel,
}

/// Stroke style for shapes.
#[derive(Clone, Debug)]
pub struct StrokeStyle {
    /// Stroke color.
    pub color: ShapeColor,
    /// Stroke width in pixels.
    pub width: f32,
    /// Line cap style.
    pub cap: LineCap,
    /// Line join style.
    pub join: LineJoin,
    /// Miter limit for miter joins.
    pub miter_limit: f32,
    /// Dash pattern (alternating dash/gap lengths). Empty means solid.
    pub dash_pattern: Vec<f32>,
    /// Dash offset.
    pub dash_offset: f32,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            color: ShapeColor::black(),
            width: 1.0,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 4.0,
            dash_pattern: Vec::new(),
            dash_offset: 0.0,
        }
    }
}

impl StrokeStyle {
    /// Create a solid stroke with the given color and width.
    pub fn solid(color: ShapeColor, width: f32) -> Self {
        Self {
            color,
            width,
            ..Default::default()
        }
    }

    /// Create a dashed stroke.
    pub fn dashed(color: ShapeColor, width: f32, dash: f32, gap: f32) -> Self {
        Self {
            color,
            width,
            dash_pattern: vec![dash, gap],
            ..Default::default()
        }
    }

    /// Check if the stroke uses a dash pattern.
    pub fn is_dashed(&self) -> bool {
        !self.dash_pattern.is_empty()
    }
}

/// Shadow effect for shapes.
#[derive(Clone, Debug)]
pub struct ShapeShadow {
    /// Shadow color.
    pub color: ShapeColor,
    /// Horizontal offset in pixels.
    pub offset_x: f32,
    /// Vertical offset in pixels.
    pub offset_y: f32,
    /// Blur radius in pixels.
    pub blur_radius: f32,
    /// Spread distance in pixels.
    pub spread: f32,
}

impl Default for ShapeShadow {
    fn default() -> Self {
        Self {
            color: ShapeColor::new(0.0, 0.0, 0.0, 0.5),
            offset_x: 2.0,
            offset_y: 2.0,
            blur_radius: 4.0,
            spread: 0.0,
        }
    }
}

/// A 2D point.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Point2D {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Point2D {
    /// Create a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Midpoint between this and another point.
    pub fn midpoint(&self, other: &Self) -> Self {
        Self {
            x: (self.x + other.x) / 2.0,
            y: (self.y + other.y) / 2.0,
        }
    }

    /// Linear interpolation between two points.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BoundingBox {
    /// Minimum X coordinate.
    pub min_x: f32,
    /// Minimum Y coordinate.
    pub min_y: f32,
    /// Maximum X coordinate.
    pub max_x: f32,
    /// Maximum Y coordinate.
    pub max_y: f32,
}

impl BoundingBox {
    /// Create a bounding box from position and size.
    pub fn from_xywh(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x + w,
            max_y: y + h,
        }
    }

    /// Width of the bounding box.
    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Point2D {
        Point2D {
            x: (self.min_x + self.max_x) / 2.0,
            y: (self.min_y + self.max_y) / 2.0,
        }
    }

    /// Check if a point is inside this bounding box.
    pub fn contains(&self, p: &Point2D) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }

    /// Check if this bounding box intersects another.
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Compute the union of this bounding box with another.
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// A drawable shape.
#[derive(Clone, Debug)]
pub enum Shape {
    /// Rectangle with position, width, and height.
    Rect {
        /// X coordinate.
        x: f32,
        /// Y coordinate.
        y: f32,
        /// Width.
        width: f32,
        /// Height.
        height: f32,
    },
    /// Rounded rectangle.
    RoundedRect {
        /// X coordinate.
        x: f32,
        /// Y coordinate.
        y: f32,
        /// Width.
        width: f32,
        /// Height.
        height: f32,
        /// Corner radius (uniform).
        radius: f32,
    },
    /// Circle with center and radius.
    Circle {
        /// Center X.
        cx: f32,
        /// Center Y.
        cy: f32,
        /// Radius.
        radius: f32,
    },
    /// Ellipse with center and radii.
    Ellipse {
        /// Center X.
        cx: f32,
        /// Center Y.
        cy: f32,
        /// Horizontal radius.
        rx: f32,
        /// Vertical radius.
        ry: f32,
    },
    /// Line between two points.
    Line {
        /// Start X.
        x1: f32,
        /// Start Y.
        y1: f32,
        /// End X.
        x2: f32,
        /// End Y.
        y2: f32,
    },
    /// Polygon defined by vertices.
    Polygon {
        /// Vertices.
        points: Vec<Point2D>,
    },
    /// Star shape.
    Star {
        /// Center X.
        cx: f32,
        /// Center Y.
        cy: f32,
        /// Outer radius.
        outer_radius: f32,
        /// Inner radius.
        inner_radius: f32,
        /// Number of points.
        num_points: u32,
    },
}

impl Shape {
    /// Get the bounding box of this shape.
    #[allow(clippy::cast_precision_loss)]
    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Self::Rect {
                x,
                y,
                width,
                height,
            } => BoundingBox::from_xywh(*x, *y, *width, *height),
            Self::RoundedRect {
                x,
                y,
                width,
                height,
                ..
            } => BoundingBox::from_xywh(*x, *y, *width, *height),
            Self::Circle { cx, cy, radius } => {
                BoundingBox::from_xywh(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
            }
            Self::Ellipse { cx, cy, rx, ry } => {
                BoundingBox::from_xywh(cx - rx, cy - ry, rx * 2.0, ry * 2.0)
            }
            Self::Line { x1, y1, x2, y2 } => BoundingBox {
                min_x: x1.min(*x2),
                min_y: y1.min(*y2),
                max_x: x1.max(*x2),
                max_y: y1.max(*y2),
            },
            Self::Polygon { points } => {
                if points.is_empty() {
                    return BoundingBox::default();
                }
                let mut bb = BoundingBox {
                    min_x: points[0].x,
                    min_y: points[0].y,
                    max_x: points[0].x,
                    max_y: points[0].y,
                };
                for p in points.iter().skip(1) {
                    bb.min_x = bb.min_x.min(p.x);
                    bb.min_y = bb.min_y.min(p.y);
                    bb.max_x = bb.max_x.max(p.x);
                    bb.max_y = bb.max_y.max(p.y);
                }
                bb
            }
            Self::Star {
                cx,
                cy,
                outer_radius,
                ..
            } => BoundingBox::from_xywh(
                cx - outer_radius,
                cy - outer_radius,
                outer_radius * 2.0,
                outer_radius * 2.0,
            ),
        }
    }

    /// Generate vertices for the shape outline.
    #[allow(clippy::cast_precision_loss)]
    pub fn vertices(&self, segments: u32) -> Vec<Point2D> {
        match self {
            Self::Rect {
                x,
                y,
                width,
                height,
            } => {
                vec![
                    Point2D::new(*x, *y),
                    Point2D::new(x + width, *y),
                    Point2D::new(x + width, y + height),
                    Point2D::new(*x, y + height),
                ]
            }
            Self::RoundedRect {
                x,
                y,
                width,
                height,
                radius,
            } => {
                let r = radius.min(width / 2.0).min(height / 2.0);
                let steps = (segments / 4).max(4);
                let mut pts = Vec::new();
                // Top-right corner arc.
                for i in 0..=steps {
                    let angle = -PI / 2.0 + (PI / 2.0) * (i as f32 / steps as f32);
                    pts.push(Point2D::new(
                        x + width - r + r * angle.cos(),
                        y + r + r * angle.sin(),
                    ));
                }
                // Bottom-right corner arc.
                for i in 0..=steps {
                    let angle = (PI / 2.0) * (i as f32 / steps as f32);
                    pts.push(Point2D::new(
                        x + width - r + r * angle.cos(),
                        y + height - r + r * angle.sin(),
                    ));
                }
                // Bottom-left corner arc.
                for i in 0..=steps {
                    let angle = PI / 2.0 + (PI / 2.0) * (i as f32 / steps as f32);
                    pts.push(Point2D::new(
                        x + r + r * angle.cos(),
                        y + height - r + r * angle.sin(),
                    ));
                }
                // Top-left corner arc.
                for i in 0..=steps {
                    let angle = PI + (PI / 2.0) * (i as f32 / steps as f32);
                    pts.push(Point2D::new(
                        x + r + r * angle.cos(),
                        y + r + r * angle.sin(),
                    ));
                }
                pts
            }
            Self::Circle { cx, cy, radius } => {
                let mut pts = Vec::with_capacity(segments as usize);
                for i in 0..segments {
                    let angle = 2.0 * PI * (i as f32 / segments as f32);
                    pts.push(Point2D::new(
                        cx + radius * angle.cos(),
                        cy + radius * angle.sin(),
                    ));
                }
                pts
            }
            Self::Ellipse { cx, cy, rx, ry } => {
                let mut pts = Vec::with_capacity(segments as usize);
                for i in 0..segments {
                    let angle = 2.0 * PI * (i as f32 / segments as f32);
                    pts.push(Point2D::new(cx + rx * angle.cos(), cy + ry * angle.sin()));
                }
                pts
            }
            Self::Line { x1, y1, x2, y2 } => {
                vec![Point2D::new(*x1, *y1), Point2D::new(*x2, *y2)]
            }
            Self::Polygon { points } => points.clone(),
            Self::Star {
                cx,
                cy,
                outer_radius,
                inner_radius,
                num_points,
            } => {
                let n = *num_points as usize;
                let mut pts = Vec::with_capacity(n * 2);
                for i in 0..(n * 2) {
                    let angle = -PI / 2.0 + 2.0 * PI * (i as f32 / (n as f32 * 2.0));
                    let r = if i % 2 == 0 {
                        *outer_radius
                    } else {
                        *inner_radius
                    };
                    pts.push(Point2D::new(cx + r * angle.cos(), cy + r * angle.sin()));
                }
                pts
            }
        }
    }

    /// Check if the shape contains a point (approximate for curved shapes).
    pub fn contains_point(&self, p: &Point2D) -> bool {
        match self {
            Self::Rect {
                x,
                y,
                width,
                height,
            } => p.x >= *x && p.x <= x + width && p.y >= *y && p.y <= y + height,
            Self::RoundedRect {
                x,
                y,
                width,
                height,
                radius,
            } => {
                // Simple check: inside rect first.
                if p.x < *x || p.x > x + width || p.y < *y || p.y > y + height {
                    return false;
                }
                let r = *radius;
                // Check corners.
                let corners = [
                    Point2D::new(x + r, y + r),
                    Point2D::new(x + width - r, y + r),
                    Point2D::new(x + r, y + height - r),
                    Point2D::new(x + width - r, y + height - r),
                ];
                for corner in &corners {
                    if (p.x - corner.x).abs() > *width / 2.0 - r {
                        continue;
                    }
                    if p.distance_to(corner) > r
                        && ((p.x < x + r || p.x > x + width - r)
                            && (p.y < y + r || p.y > y + height - r))
                    {
                        return false;
                    }
                }
                true
            }
            Self::Circle { cx, cy, radius } => {
                let dx = p.x - cx;
                let dy = p.y - cy;
                dx * dx + dy * dy <= radius * radius
            }
            Self::Ellipse { cx, cy, rx, ry } => {
                if *rx == 0.0 || *ry == 0.0 {
                    return false;
                }
                let dx = (p.x - cx) / rx;
                let dy = (p.y - cy) / ry;
                dx * dx + dy * dy <= 1.0
            }
            _ => self.bounding_box().contains(p),
        }
    }

    /// Compute the perimeter (approximate for curved shapes).
    #[allow(clippy::cast_precision_loss)]
    pub fn perimeter(&self) -> f32 {
        match self {
            Self::Rect { width, height, .. } => 2.0 * (width + height),
            Self::RoundedRect {
                width,
                height,
                radius,
                ..
            } => {
                let r = radius.min(width / 2.0).min(height / 2.0);
                2.0 * (width - 2.0 * r) + 2.0 * (height - 2.0 * r) + 2.0 * PI * r
            }
            Self::Circle { radius, .. } => 2.0 * PI * radius,
            Self::Ellipse { rx, ry, .. } => {
                // Ramanujan approximation.
                let a = *rx;
                let b = *ry;
                PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
            }
            Self::Line { x1, y1, x2, y2 } => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                (dx * dx + dy * dy).sqrt()
            }
            Self::Polygon { points } => {
                if points.len() < 2 {
                    return 0.0;
                }
                let mut peri = 0.0;
                for i in 0..points.len() {
                    let j = (i + 1) % points.len();
                    peri += points[i].distance_to(&points[j]);
                }
                peri
            }
            Self::Star {
                outer_radius,
                inner_radius,
                num_points,
                ..
            } => {
                // Approximate by summing edges of the star polygon.
                let n = *num_points as usize;
                let mut peri = 0.0_f32;
                let verts = self.vertices(64);
                for i in 0..verts.len() {
                    let j = (i + 1) % verts.len();
                    peri += verts[i].distance_to(&verts[j]);
                }
                let _ = (outer_radius, inner_radius, n);
                peri
            }
        }
    }

    /// Compute the area of the shape.
    pub fn area(&self) -> f32 {
        match self {
            Self::Rect { width, height, .. } => width * height,
            Self::RoundedRect {
                width,
                height,
                radius,
                ..
            } => {
                let r = radius.min(width / 2.0).min(height / 2.0);
                width * height - (4.0 - PI) * r * r
            }
            Self::Circle { radius, .. } => PI * radius * radius,
            Self::Ellipse { rx, ry, .. } => PI * rx * ry,
            Self::Line { .. } => 0.0,
            Self::Polygon { points } => {
                // Shoelace formula.
                if points.len() < 3 {
                    return 0.0;
                }
                let mut sum = 0.0_f32;
                for i in 0..points.len() {
                    let j = (i + 1) % points.len();
                    sum += points[i].x * points[j].y - points[j].x * points[i].y;
                }
                sum.abs() / 2.0
            }
            Self::Star {
                outer_radius,
                inner_radius,
                num_points,
                ..
            } => {
                // Area of a regular star polygon.
                let n = *num_points as f32;
                let r_out = *outer_radius;
                let r_in = *inner_radius;
                n * r_out * r_in * (PI / n).sin()
            }
        }
    }
}

/// A renderable shape with style properties.
#[derive(Clone, Debug)]
pub struct StyledShape {
    /// The shape geometry.
    pub shape: Shape,
    /// Fill color (None = no fill).
    pub fill: Option<ShapeColor>,
    /// Stroke style (None = no stroke).
    pub stroke: Option<StrokeStyle>,
    /// Shadow (None = no shadow).
    pub shadow: Option<ShapeShadow>,
    /// Opacity (0.0..=1.0).
    pub opacity: f32,
}

impl StyledShape {
    /// Create a filled shape.
    pub fn filled(shape: Shape, color: ShapeColor) -> Self {
        Self {
            shape,
            fill: Some(color),
            stroke: None,
            shadow: None,
            opacity: 1.0,
        }
    }

    /// Create a stroked shape.
    pub fn stroked(shape: Shape, stroke: StrokeStyle) -> Self {
        Self {
            shape,
            fill: None,
            stroke: Some(stroke),
            shadow: None,
            opacity: 1.0,
        }
    }

    /// Create a shape with both fill and stroke.
    pub fn filled_and_stroked(shape: Shape, fill: ShapeColor, stroke: StrokeStyle) -> Self {
        Self {
            shape,
            fill: Some(fill),
            stroke: Some(stroke),
            shadow: None,
            opacity: 1.0,
        }
    }

    /// Set the shadow.
    pub fn with_shadow(mut self, shadow: ShapeShadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Set the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Get the bounding box including shadow offset.
    pub fn bounding_box(&self) -> BoundingBox {
        let mut bb = self.shape.bounding_box();
        if let Some(ref shadow) = self.shadow {
            let sx = shadow.offset_x + shadow.blur_radius + shadow.spread;
            let sy = shadow.offset_y + shadow.blur_radius + shadow.spread;
            bb.max_x += sx.max(0.0);
            bb.max_y += sy.max(0.0);
            bb.min_x += sx.min(0.0);
            bb.min_y += sy.min(0.0);
        }
        if let Some(ref stroke) = self.stroke {
            let half = stroke.width / 2.0;
            bb.min_x -= half;
            bb.min_y -= half;
            bb.max_x += half;
            bb.max_y += half;
        }
        bb
    }
}

/// Regular polygon generator.
pub struct RegularPolygon;

impl RegularPolygon {
    /// Generate a regular polygon (equilateral triangle, square, pentagon, etc.).
    #[allow(clippy::cast_precision_loss)]
    pub fn generate(cx: f32, cy: f32, radius: f32, sides: u32) -> Shape {
        let n = sides.max(3);
        let mut points = Vec::with_capacity(n as usize);
        for i in 0..n {
            let angle = -PI / 2.0 + 2.0 * PI * (i as f32 / n as f32);
            points.push(Point2D::new(
                cx + radius * angle.cos(),
                cy + radius * angle.sin(),
            ));
        }
        Shape::Polygon { points }
    }
}

// ===========================================================================
// Anti-aliased rendering
// ===========================================================================

/// Anti-aliased rendering quality level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AntiAliasQuality {
    /// No anti-aliasing (alias/jagged edges).
    None,
    /// 2x2 MSAA (4 sub-samples per pixel).
    Low,
    /// 4x4 MSAA (16 sub-samples per pixel).
    Medium,
    /// 8x8 MSAA (64 sub-samples per pixel).
    High,
}

impl AntiAliasQuality {
    /// Number of sub-samples per axis.
    fn samples_per_axis(self) -> u32 {
        match self {
            Self::None => 1,
            Self::Low => 2,
            Self::Medium => 4,
            Self::High => 8,
        }
    }

    /// Total sub-samples per pixel.
    pub fn total_samples(self) -> u32 {
        let s = self.samples_per_axis();
        s * s
    }
}

impl Default for AntiAliasQuality {
    fn default() -> Self {
        Self::Medium
    }
}

/// An anti-aliased software rasterizer for shapes.
///
/// Renders shapes into an RGBA pixel buffer using multi-sample anti-aliasing.
/// Each pixel is sampled at multiple sub-pixel locations to determine coverage,
/// producing smooth edges at broadcast resolution.
pub struct AntiAliasedRenderer {
    /// Width of the render buffer.
    pub width: u32,
    /// Height of the render buffer.
    pub height: u32,
    /// AA quality level.
    pub quality: AntiAliasQuality,
}

impl AntiAliasedRenderer {
    /// Create a new anti-aliased renderer.
    pub fn new(width: u32, height: u32, quality: AntiAliasQuality) -> Self {
        Self {
            width,
            height,
            quality,
        }
    }

    /// Render a filled circle with anti-aliasing into an RGBA buffer.
    ///
    /// The buffer must be `width * height * 4` bytes.
    pub fn render_circle(
        &self,
        buffer: &mut [u8],
        cx: f32,
        cy: f32,
        radius: f32,
        color: &ShapeColor,
    ) {
        let samples = self.quality.samples_per_axis();
        let _total = self.quality.total_samples() as f32;
        let r2 = radius * radius;

        let bb = BoundingBox::from_xywh(cx - radius, cy - radius, radius * 2.0, radius * 2.0);
        let x_start = (bb.min_x.floor() as i32).max(0) as u32;
        let y_start = (bb.min_y.floor() as i32).max(0) as u32;
        let x_end = (bb.max_x.ceil() as u32 + 1).min(self.width);
        let y_end = (bb.max_y.ceil() as u32 + 1).min(self.height);

        for py in y_start..y_end {
            for px in x_start..x_end {
                let coverage = self.compute_circle_coverage(px, py, cx, cy, r2, samples);

                if coverage > 0.0 {
                    let alpha = color.a * coverage;
                    self.blend_pixel(buffer, px, py, color, alpha);
                }
            }
        }
    }

    /// Render a filled rectangle with anti-aliased edges.
    pub fn render_rect(
        &self,
        buffer: &mut [u8],
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: &ShapeColor,
    ) {
        let samples = self.quality.samples_per_axis();

        let x_start = (x.floor() as i32).max(0) as u32;
        let y_start = (y.floor() as i32).max(0) as u32;
        let x_end = ((x + w).ceil() as u32 + 1).min(self.width);
        let y_end = ((y + h).ceil() as u32 + 1).min(self.height);

        for py in y_start..y_end {
            for px in x_start..x_end {
                let coverage = self.compute_rect_coverage(px, py, x, y, w, h, samples);

                if coverage > 0.0 {
                    let alpha = color.a * coverage;
                    self.blend_pixel(buffer, px, py, color, alpha);
                }
            }
        }
    }

    /// Render a filled polygon with anti-aliased edges.
    pub fn render_polygon(&self, buffer: &mut [u8], points: &[Point2D], color: &ShapeColor) {
        if points.len() < 3 {
            return;
        }

        let samples = self.quality.samples_per_axis();

        // Compute bounding box
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        let x_start = (min_x.floor() as i32).max(0) as u32;
        let y_start = (min_y.floor() as i32).max(0) as u32;
        let x_end = (max_x.ceil() as u32 + 1).min(self.width);
        let y_end = (max_y.ceil() as u32 + 1).min(self.height);

        for py in y_start..y_end {
            for px in x_start..x_end {
                let coverage = self.compute_polygon_coverage(px, py, points, samples);

                if coverage > 0.0 {
                    let alpha = color.a * coverage;
                    self.blend_pixel(buffer, px, py, color, alpha);
                }
            }
        }
    }

    /// Render an anti-aliased line (Xiaolin Wu style).
    pub fn render_line(
        &self,
        buffer: &mut [u8],
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: &ShapeColor,
        width: f32,
    ) {
        // Use MSAA sampling along the line's perpendicular
        let dx = x2 - x1;
        let dy = y2 - y1;
        let length = (dx * dx + dy * dy).sqrt();
        if length < f32::EPSILON {
            return;
        }

        // Normal to the line
        let _nx = -dy / length;
        let _ny = dx / length;
        let half_w = width / 2.0;

        let samples = self.quality.samples_per_axis();

        // Bounding box
        let expand = half_w + 1.0;
        let min_x = x1.min(x2) - expand;
        let min_y = y1.min(y2) - expand;
        let max_x = x1.max(x2) + expand;
        let max_y = y1.max(y2) + expand;

        let px_start = (min_x.floor() as i32).max(0) as u32;
        let py_start = (min_y.floor() as i32).max(0) as u32;
        let px_end = (max_x.ceil() as u32 + 1).min(self.width);
        let py_end = (max_y.ceil() as u32 + 1).min(self.height);

        for py in py_start..py_end {
            for px in px_start..px_end {
                let mut hit = 0u32;
                let total = self.quality.total_samples() as f32;

                for sy in 0..samples {
                    for sx in 0..samples {
                        let sub_x = px as f32 + (sx as f32 + 0.5) / samples as f32;
                        let sub_y = py as f32 + (sy as f32 + 0.5) / samples as f32;

                        // Project onto line
                        let ax = sub_x - x1;
                        let ay = sub_y - y1;
                        let t = (ax * dx + ay * dy) / (length * length);
                        let t_clamped = t.clamp(0.0, 1.0);

                        let closest_x = x1 + t_clamped * dx;
                        let closest_y = y1 + t_clamped * dy;

                        let dist =
                            ((sub_x - closest_x).powi(2) + (sub_y - closest_y).powi(2)).sqrt();
                        if dist <= half_w {
                            hit += 1;
                        }
                    }
                }

                if hit > 0 {
                    let coverage = hit as f32 / total;
                    let alpha = color.a * coverage;
                    self.blend_pixel(buffer, px, py, color, alpha);
                }
            }
        }
    }

    /// Compute circle coverage for a pixel via MSAA.
    fn compute_circle_coverage(
        &self,
        px: u32,
        py: u32,
        cx: f32,
        cy: f32,
        r2: f32,
        samples: u32,
    ) -> f32 {
        let mut hit = 0u32;
        let total = (samples * samples) as f32;

        for sy in 0..samples {
            for sx in 0..samples {
                let sub_x = px as f32 + (sx as f32 + 0.5) / samples as f32;
                let sub_y = py as f32 + (sy as f32 + 0.5) / samples as f32;
                let dx = sub_x - cx;
                let dy = sub_y - cy;
                if dx * dx + dy * dy <= r2 {
                    hit += 1;
                }
            }
        }

        hit as f32 / total
    }

    /// Compute rectangle coverage for a pixel via MSAA.
    fn compute_rect_coverage(
        &self,
        px: u32,
        py: u32,
        rx: f32,
        ry: f32,
        rw: f32,
        rh: f32,
        samples: u32,
    ) -> f32 {
        let mut hit = 0u32;
        let total = (samples * samples) as f32;

        for sy in 0..samples {
            for sx in 0..samples {
                let sub_x = px as f32 + (sx as f32 + 0.5) / samples as f32;
                let sub_y = py as f32 + (sy as f32 + 0.5) / samples as f32;
                if sub_x >= rx && sub_x <= rx + rw && sub_y >= ry && sub_y <= ry + rh {
                    hit += 1;
                }
            }
        }

        hit as f32 / total
    }

    /// Compute polygon coverage via MSAA using ray-casting point-in-polygon.
    fn compute_polygon_coverage(&self, px: u32, py: u32, points: &[Point2D], samples: u32) -> f32 {
        let mut hit = 0u32;
        let total = (samples * samples) as f32;

        for sy in 0..samples {
            for sx in 0..samples {
                let sub_x = px as f32 + (sx as f32 + 0.5) / samples as f32;
                let sub_y = py as f32 + (sy as f32 + 0.5) / samples as f32;
                if point_in_polygon(sub_x, sub_y, points) {
                    hit += 1;
                }
            }
        }

        hit as f32 / total
    }

    /// Alpha-blend a pixel into the buffer.
    fn blend_pixel(&self, buffer: &mut [u8], px: u32, py: u32, color: &ShapeColor, alpha: f32) {
        if px >= self.width || py >= self.height {
            return;
        }
        let idx = ((py * self.width + px) * 4) as usize;
        if idx + 3 >= buffer.len() {
            return;
        }

        let inv = 1.0 - alpha;
        buffer[idx] =
            (color.r * alpha * 255.0 + f32::from(buffer[idx]) * inv).clamp(0.0, 255.0) as u8;
        buffer[idx + 1] =
            (color.g * alpha * 255.0 + f32::from(buffer[idx + 1]) * inv).clamp(0.0, 255.0) as u8;
        buffer[idx + 2] =
            (color.b * alpha * 255.0 + f32::from(buffer[idx + 2]) * inv).clamp(0.0, 255.0) as u8;
        buffer[idx + 3] =
            ((alpha * 255.0) + f32::from(buffer[idx + 3]) * inv).clamp(0.0, 255.0) as u8;
    }
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(px: f32, py: f32, polygon: &[Point2D]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = polygon[i].y;
        let yj = polygon[j].y;
        let xi = polygon[i].x;
        let xj = polygon[j].x;

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_color_new() {
        let c = ShapeColor::new(0.5, 0.6, 0.7, 0.8);
        assert!((c.r - 0.5).abs() < f32::EPSILON);
        assert!((c.g - 0.6).abs() < f32::EPSILON);
        assert!((c.b - 0.7).abs() < f32::EPSILON);
        assert!((c.a - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_shape_color_clamp() {
        let c = ShapeColor::new(1.5, -0.1, 0.5, 2.0);
        assert!((c.r - 1.0).abs() < f32::EPSILON);
        assert!((c.g).abs() < f32::EPSILON);
        assert!((c.a - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_premultiplied() {
        let c = ShapeColor::new(1.0, 0.5, 0.0, 0.5);
        let pm = c.premultiplied();
        assert!((pm.r - 0.5).abs() < f32::EPSILON);
        assert!((pm.g - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_lerp() {
        let a = ShapeColor::black();
        let b = ShapeColor::white();
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.g - 0.5).abs() < 0.01);
        assert!((mid.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_point_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_point_midpoint() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(10.0, 10.0);
        let mid = a.midpoint(&b);
        assert!((mid.x - 5.0).abs() < f32::EPSILON);
        assert!((mid.y - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bb = BoundingBox::from_xywh(0.0, 0.0, 100.0, 100.0);
        assert!(bb.contains(&Point2D::new(50.0, 50.0)));
        assert!(!bb.contains(&Point2D::new(150.0, 50.0)));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let a = BoundingBox::from_xywh(0.0, 0.0, 50.0, 50.0);
        let b = BoundingBox::from_xywh(25.0, 25.0, 50.0, 50.0);
        let c = BoundingBox::from_xywh(100.0, 100.0, 50.0, 50.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bounding_box_union() {
        let a = BoundingBox::from_xywh(0.0, 0.0, 50.0, 50.0);
        let b = BoundingBox::from_xywh(25.0, 25.0, 50.0, 50.0);
        let u = a.union(&b);
        assert!((u.min_x).abs() < f32::EPSILON);
        assert!((u.min_y).abs() < f32::EPSILON);
        assert!((u.max_x - 75.0).abs() < f32::EPSILON);
        assert!((u.max_y - 75.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_area() {
        let r = Shape::Rect {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 20.0,
        };
        assert!((r.area() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_circle_area() {
        let c = Shape::Circle {
            cx: 0.0,
            cy: 0.0,
            radius: 10.0,
        };
        let expected = PI * 100.0;
        assert!((c.area() - expected).abs() < 0.01);
    }

    #[test]
    fn test_circle_perimeter() {
        let c = Shape::Circle {
            cx: 0.0,
            cy: 0.0,
            radius: 10.0,
        };
        let expected = 2.0 * PI * 10.0;
        assert!((c.perimeter() - expected).abs() < 0.01);
    }

    #[test]
    fn test_circle_contains_point() {
        let c = Shape::Circle {
            cx: 50.0,
            cy: 50.0,
            radius: 25.0,
        };
        assert!(c.contains_point(&Point2D::new(50.0, 50.0)));
        assert!(!c.contains_point(&Point2D::new(100.0, 100.0)));
    }

    #[test]
    fn test_rect_bounding_box() {
        let r = Shape::Rect {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
        };
        let bb = r.bounding_box();
        assert!((bb.min_x - 10.0).abs() < f32::EPSILON);
        assert!((bb.min_y - 20.0).abs() < f32::EPSILON);
        assert!((bb.width() - 30.0).abs() < f32::EPSILON);
        assert!((bb.height() - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_vertices() {
        let r = Shape::Rect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
        };
        let verts = r.vertices(32);
        assert_eq!(verts.len(), 4);
    }

    #[test]
    fn test_polygon_area_triangle() {
        let tri = Shape::Polygon {
            points: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(10.0, 0.0),
                Point2D::new(5.0, 10.0),
            ],
        };
        // Triangle area = 0.5 * base * height = 50.
        assert!((tri.area() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_stroke_style_dashed() {
        let stroke = StrokeStyle::dashed(ShapeColor::black(), 2.0, 5.0, 3.0);
        assert!(stroke.is_dashed());
        assert_eq!(stroke.dash_pattern.len(), 2);
    }

    #[test]
    fn test_styled_shape_filled() {
        let shape = Shape::Circle {
            cx: 0.0,
            cy: 0.0,
            radius: 10.0,
        };
        let styled = StyledShape::filled(shape, ShapeColor::white());
        assert!(styled.fill.is_some());
        assert!(styled.stroke.is_none());
        assert!((styled.opacity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_styled_shape_with_shadow_bb() {
        let shape = Shape::Rect {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
        };
        let styled =
            StyledShape::filled(shape, ShapeColor::white()).with_shadow(ShapeShadow::default());
        let bb = styled.bounding_box();
        // Shadow should extend the bounding box.
        assert!(bb.max_x > 100.0);
        assert!(bb.max_y > 100.0);
    }

    #[test]
    fn test_regular_polygon_pentagon() {
        let shape = RegularPolygon::generate(0.0, 0.0, 50.0, 5);
        if let Shape::Polygon { points } = shape {
            assert_eq!(points.len(), 5);
        } else {
            panic!("Expected polygon");
        }
    }

    #[test]
    fn test_star_vertices() {
        let star = Shape::Star {
            cx: 0.0,
            cy: 0.0,
            outer_radius: 50.0,
            inner_radius: 25.0,
            num_points: 5,
        };
        let verts = star.vertices(64);
        assert_eq!(verts.len(), 10); // 5 points * 2 (outer + inner)
    }

    #[test]
    fn test_line_perimeter() {
        let line = Shape::Line {
            x1: 0.0,
            y1: 0.0,
            x2: 3.0,
            y2: 4.0,
        };
        assert!((line.perimeter() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_ellipse_contains() {
        let e = Shape::Ellipse {
            cx: 50.0,
            cy: 50.0,
            rx: 30.0,
            ry: 20.0,
        };
        assert!(e.contains_point(&Point2D::new(50.0, 50.0)));
        assert!(!e.contains_point(&Point2D::new(0.0, 0.0)));
    }

    // --- Anti-aliased rendering tests ---

    #[test]
    fn test_aa_quality_samples() {
        assert_eq!(AntiAliasQuality::None.total_samples(), 1);
        assert_eq!(AntiAliasQuality::Low.total_samples(), 4);
        assert_eq!(AntiAliasQuality::Medium.total_samples(), 16);
        assert_eq!(AntiAliasQuality::High.total_samples(), 64);
    }

    #[test]
    fn test_aa_renderer_creation() {
        let r = AntiAliasedRenderer::new(100, 100, AntiAliasQuality::Medium);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 100);
    }

    #[test]
    fn test_aa_render_circle_center_pixel() {
        let renderer = AntiAliasedRenderer::new(20, 20, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 20 * 20 * 4];
        let color = ShapeColor::rgb(1.0, 0.0, 0.0);
        renderer.render_circle(&mut buffer, 10.0, 10.0, 5.0, &color);
        let idx = (10 * 20 + 10) * 4;
        assert!(
            buffer[idx] > 200,
            "Center R should be high: {}",
            buffer[idx]
        );
        assert!(
            buffer[idx + 3] > 200,
            "Center A should be high: {}",
            buffer[idx + 3]
        );
    }

    #[test]
    fn test_aa_render_circle_outside_empty() {
        let renderer = AntiAliasedRenderer::new(20, 20, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 20 * 20 * 4];
        let color = ShapeColor::rgb(1.0, 1.0, 1.0);
        renderer.render_circle(&mut buffer, 10.0, 10.0, 3.0, &color);
        assert_eq!(buffer[3], 0, "Far corner should be empty");
    }

    #[test]
    fn test_aa_render_rect() {
        let renderer = AntiAliasedRenderer::new(20, 20, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 20 * 20 * 4];
        let color = ShapeColor::rgb(0.0, 0.0, 1.0);
        renderer.render_rect(&mut buffer, 5.0, 5.0, 10.0, 10.0, &color);
        let idx = (10 * 20 + 10) * 4;
        assert!(buffer[idx + 2] > 200, "Center should be blue");
        assert_eq!(buffer[3], 0, "Top-left should be empty");
    }

    #[test]
    fn test_aa_render_rect_fractional() {
        let renderer = AntiAliasedRenderer::new(10, 10, AntiAliasQuality::High);
        let mut buffer = vec![0u8; 10 * 10 * 4];
        let color = ShapeColor::rgb(1.0, 1.0, 0.0);
        renderer.render_rect(&mut buffer, 2.3, 2.7, 5.0, 5.0, &color);
        let idx = (3 * 10 + 2) * 4;
        let alpha = buffer[idx + 3];
        assert!(
            alpha > 0 && alpha < 255,
            "Edge should be partial, got {alpha}"
        );
    }

    #[test]
    fn test_aa_render_polygon() {
        let renderer = AntiAliasedRenderer::new(20, 20, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 20 * 20 * 4];
        let color = ShapeColor::rgb(1.0, 0.5, 0.0);
        let triangle = vec![
            Point2D::new(10.0, 2.0),
            Point2D::new(18.0, 18.0),
            Point2D::new(2.0, 18.0),
        ];
        renderer.render_polygon(&mut buffer, &triangle, &color);
        let idx = (12 * 20 + 10) * 4;
        assert!(
            buffer[idx + 3] > 100,
            "Triangle center should have coverage"
        );
    }

    #[test]
    fn test_aa_render_line() {
        let renderer = AntiAliasedRenderer::new(20, 20, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 20 * 20 * 4];
        let color = ShapeColor::rgb(1.0, 1.0, 1.0);
        renderer.render_line(&mut buffer, 2.0, 2.0, 18.0, 18.0, &color, 2.0);
        let idx = (10 * 20 + 10) * 4;
        assert!(buffer[idx + 3] > 0, "Line midpoint should have coverage");
    }

    #[test]
    fn test_aa_render_line_zero_length() {
        let renderer = AntiAliasedRenderer::new(10, 10, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 10 * 10 * 4];
        let color = ShapeColor::rgb(1.0, 0.0, 0.0);
        renderer.render_line(&mut buffer, 5.0, 5.0, 5.0, 5.0, &color, 1.0);
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        let tri = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(5.0, 10.0),
        ];
        assert!(point_in_polygon(5.0, 3.0, &tri));
        assert!(!point_in_polygon(20.0, 20.0, &tri));
    }

    #[test]
    fn test_point_in_polygon_square() {
        let sq = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(0.0, 10.0),
        ];
        assert!(point_in_polygon(5.0, 5.0, &sq));
        assert!(!point_in_polygon(-1.0, 5.0, &sq));
    }

    #[test]
    fn test_point_in_polygon_degenerate() {
        let line = vec![Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0)];
        assert!(!point_in_polygon(5.0, 0.0, &line));
    }

    #[test]
    fn test_aa_none_no_smoothing() {
        let renderer = AntiAliasedRenderer::new(20, 20, AntiAliasQuality::None);
        let mut buffer = vec![0u8; 20 * 20 * 4];
        let color = ShapeColor::rgb(1.0, 0.0, 0.0);
        renderer.render_circle(&mut buffer, 10.0, 10.0, 5.0, &color);
        let mut has_partial = false;
        for pixel in buffer.chunks_exact(4) {
            if pixel[3] > 0 && pixel[3] < 255 {
                has_partial = true;
            }
        }
        assert!(!has_partial, "None quality should produce binary edges");
    }

    #[test]
    fn test_aa_medium_has_smoothing() {
        let renderer = AntiAliasedRenderer::new(30, 30, AntiAliasQuality::Medium);
        let mut buffer = vec![0u8; 30 * 30 * 4];
        let color = ShapeColor::rgb(1.0, 0.0, 0.0);
        renderer.render_circle(&mut buffer, 15.0, 15.0, 10.0, &color);
        let mut has_partial = false;
        for pixel in buffer.chunks_exact(4) {
            if pixel[3] > 0 && pixel[3] < 250 {
                has_partial = true;
                break;
            }
        }
        assert!(has_partial, "Medium quality should produce smooth edges");
    }
}
