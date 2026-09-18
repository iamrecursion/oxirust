//! Shape drawing.

use crate::{Color, EffectParams, Frame, Rect, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Shape type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeType {
    /// Rectangle.
    Rectangle,
    /// Circle.
    Circle,
    /// Line.
    Line,
    /// Polygon.
    Polygon,
}

/// Shape definition.
#[derive(Debug, Clone)]
pub struct Shape {
    /// Shape type.
    pub shape_type: ShapeType,
    /// Fill color.
    pub fill_color: Color,
    /// Stroke color.
    pub stroke_color: Color,
    /// Stroke width.
    pub stroke_width: f32,
    /// Bounding rectangle.
    pub bounds: Rect,
    /// Number of sides (for polygon).
    pub sides: u32,
}

impl Shape {
    /// Create a new rectangle shape.
    #[must_use]
    pub fn rectangle(bounds: Rect) -> Self {
        Self {
            shape_type: ShapeType::Rectangle,
            fill_color: Color::white(),
            stroke_color: Color::black(),
            stroke_width: 1.0,
            bounds,
            sides: 0,
        }
    }

    /// Create a new circle shape.
    #[must_use]
    pub fn circle(center_x: f32, center_y: f32, radius: f32) -> Self {
        Self {
            shape_type: ShapeType::Circle,
            fill_color: Color::white(),
            stroke_color: Color::black(),
            stroke_width: 1.0,
            bounds: Rect::new(
                center_x - radius,
                center_y - radius,
                radius * 2.0,
                radius * 2.0,
            ),
            sides: 0,
        }
    }

    /// Create a new polygon shape.
    #[must_use]
    pub fn polygon(center_x: f32, center_y: f32, radius: f32, sides: u32) -> Self {
        Self {
            shape_type: ShapeType::Polygon,
            fill_color: Color::white(),
            stroke_color: Color::black(),
            stroke_width: 1.0,
            bounds: Rect::new(
                center_x - radius,
                center_y - radius,
                radius * 2.0,
                radius * 2.0,
            ),
            sides,
        }
    }

    /// Set fill color.
    #[must_use]
    pub const fn with_fill(mut self, color: Color) -> Self {
        self.fill_color = color;
        self
    }

    /// Set stroke.
    #[must_use]
    pub const fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke_color = color;
        self.stroke_width = width;
        self
    }
}

/// Shape drawer.
pub struct ShapeDrawer {
    shapes: Vec<Shape>,
}

impl ShapeDrawer {
    /// Create a new shape drawer.
    #[must_use]
    pub const fn new() -> Self {
        Self { shapes: Vec::new() }
    }

    /// Add a shape.
    pub fn add_shape(&mut self, shape: Shape) {
        self.shapes.push(shape);
    }

    /// Clear all shapes.
    pub fn clear(&mut self) {
        self.shapes.clear();
    }

    fn draw_rectangle(&self, frame: &mut Frame, shape: &Shape) {
        let x1 = shape.bounds.x as i32;
        let y1 = shape.bounds.y as i32;
        let x2 = (shape.bounds.x + shape.bounds.width) as i32;
        let y2 = (shape.bounds.y + shape.bounds.height) as i32;

        for y in y1..=y2 {
            for x in x1..=x2 {
                if x >= 0 && x < frame.width as i32 && y >= 0 && y < frame.height as i32 {
                    // Check if on edge (stroke)
                    let is_edge = x == x1 || x == x2 || y == y1 || y == y2;
                    let color = if is_edge && shape.stroke_width > 0.0 {
                        shape.stroke_color
                    } else {
                        shape.fill_color
                    };

                    frame.set_pixel(x as u32, y as u32, color.to_rgba());
                }
            }
        }
    }

    fn draw_circle(&self, frame: &mut Frame, shape: &Shape) {
        let cx = shape.bounds.x + shape.bounds.width / 2.0;
        let cy = shape.bounds.y + shape.bounds.height / 2.0;
        let radius = shape.bounds.width / 2.0;

        let x1 = (cx - radius) as i32;
        let y1 = (cy - radius) as i32;
        let x2 = (cx + radius) as i32;
        let y2 = (cy + radius) as i32;

        for y in y1..=y2 {
            for x in x1..=x2 {
                if x >= 0 && x < frame.width as i32 && y >= 0 && y < frame.height as i32 {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();

                    if dist <= radius {
                        let is_edge =
                            dist >= radius - shape.stroke_width && shape.stroke_width > 0.0;
                        let color = if is_edge {
                            shape.stroke_color
                        } else {
                            shape.fill_color
                        };

                        frame.set_pixel(x as u32, y as u32, color.to_rgba());
                    }
                }
            }
        }
    }

    fn draw_polygon(&self, frame: &mut Frame, shape: &Shape) {
        let cx = shape.bounds.x + shape.bounds.width / 2.0;
        let cy = shape.bounds.y + shape.bounds.height / 2.0;
        let radius = shape.bounds.width / 2.0;

        // Calculate polygon vertices
        let mut vertices = Vec::new();
        for i in 0..shape.sides {
            let angle = (i as f32 / shape.sides as f32) * std::f32::consts::TAU
                - std::f32::consts::FRAC_PI_2;
            let x = cx + angle.cos() * radius;
            let y = cy + angle.sin() * radius;
            vertices.push((x, y));
        }

        // Simple fill using scanline
        let x1 = (cx - radius) as i32;
        let y1 = (cy - radius) as i32;
        let x2 = (cx + radius) as i32;
        let y2 = (cy + radius) as i32;

        for y in y1..=y2 {
            for x in x1..=x2 {
                if x >= 0 && x < frame.width as i32 && y >= 0 && y < frame.height as i32 {
                    // Simple point-in-polygon test
                    let px = x as f32;
                    let py = y as f32;

                    let mut inside = false;
                    for i in 0..vertices.len() {
                        let j = (i + 1) % vertices.len();
                        let (x1, y1) = vertices[i];
                        let (x2, y2) = vertices[j];

                        if ((y1 > py) != (y2 > py)) && (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)
                        {
                            inside = !inside;
                        }
                    }

                    if inside {
                        frame.set_pixel(x as u32, y as u32, shape.fill_color.to_rgba());
                    }
                }
            }
        }
    }
}

impl Default for ShapeDrawer {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoEffect for ShapeDrawer {
    fn name(&self) -> &'static str {
        "Shape Drawer"
    }

    fn description(&self) -> &'static str {
        "Draw shapes on frames"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        // Copy input to output
        for y in 0..output.height {
            for x in 0..output.width {
                output.set_pixel(x, y, input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]));
            }
        }

        // Draw all shapes
        for shape in &self.shapes {
            match shape.shape_type {
                ShapeType::Rectangle => self.draw_rectangle(output, shape),
                ShapeType::Circle => self.draw_circle(output, shape),
                ShapeType::Polygon => self.draw_polygon(output, shape),
                ShapeType::Line => {
                    // Line drawing would go here
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_drawer() {
        let mut drawer = ShapeDrawer::new();
        drawer.add_shape(
            Shape::rectangle(Rect::new(10.0, 10.0, 50.0, 50.0)).with_fill(Color::rgb(255, 0, 0)),
        );

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        drawer
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
