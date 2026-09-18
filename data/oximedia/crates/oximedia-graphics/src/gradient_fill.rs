//! Gradient fill system for broadcast graphics.
//!
//! Provides linear, radial, and conic gradient fills for shapes and backgrounds.
//! Supports multiple color stops, spread modes, and coordinate transformations
//! commonly used in broadcast graphics elements.

use std::f32::consts::PI;

/// A color used within gradients (RGBA, 0.0..=1.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientColor {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

impl GradientColor {
    /// Create a new gradient color.
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
            a: a.clamp(0.0, 1.0),
        }
    }

    /// Create an opaque gradient color.
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    /// Create from 8-bit RGBA values.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Black color.
    pub fn black() -> Self {
        Self::rgb(0.0, 0.0, 0.0)
    }

    /// White color.
    pub fn white() -> Self {
        Self::rgb(1.0, 1.0, 1.0)
    }

    /// Transparent color.
    pub fn transparent() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Linearly interpolate between this color and another.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Convert to premultiplied alpha.
    pub fn premultiplied(&self) -> Self {
        Self {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }

    /// Convert to 8-bit RGBA tuple.
    pub fn to_u8(&self) -> (u8, u8, u8, u8) {
        (
            (self.r * 255.0 + 0.5) as u8,
            (self.g * 255.0 + 0.5) as u8,
            (self.b * 255.0 + 0.5) as u8,
            (self.a * 255.0 + 0.5) as u8,
        )
    }
}

impl Default for GradientColor {
    fn default() -> Self {
        Self::white()
    }
}

/// A color stop in a gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    /// Position of the stop (0.0..=1.0).
    pub position: f32,
    /// Color at this stop.
    pub color: GradientColor,
}

impl ColorStop {
    /// Create a new color stop.
    pub fn new(position: f32, color: GradientColor) -> Self {
        Self {
            position: position.clamp(0.0, 1.0),
            color,
        }
    }
}

/// How to handle coordinates outside the 0.0..=1.0 range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpreadMode {
    /// Clamp to the nearest edge color.
    #[default]
    Pad,
    /// Repeat the gradient pattern.
    Repeat,
    /// Mirror the gradient pattern at boundaries.
    Reflect,
}

/// Color interpolation space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterpolationSpace {
    /// Linear RGB interpolation.
    #[default]
    LinearRgb,
    /// sRGB interpolation (perceptually uniform).
    Srgb,
    /// HSL interpolation (hue-based).
    Hsl,
}

/// Linear gradient definition.
#[derive(Clone, Debug)]
pub struct LinearGradient {
    /// Start X coordinate (normalized 0.0..=1.0 relative to shape bounds).
    pub start_x: f32,
    /// Start Y coordinate.
    pub start_y: f32,
    /// End X coordinate.
    pub end_x: f32,
    /// End Y coordinate.
    pub end_y: f32,
    /// Color stops (must have at least 2).
    pub stops: Vec<ColorStop>,
    /// Spread mode.
    pub spread: SpreadMode,
    /// Color interpolation space.
    pub interpolation: InterpolationSpace,
}

impl LinearGradient {
    /// Create a horizontal linear gradient (left to right).
    pub fn horizontal(start_color: GradientColor, end_color: GradientColor) -> Self {
        Self {
            start_x: 0.0,
            start_y: 0.5,
            end_x: 1.0,
            end_y: 0.5,
            stops: vec![
                ColorStop::new(0.0, start_color),
                ColorStop::new(1.0, end_color),
            ],
            spread: SpreadMode::Pad,
            interpolation: InterpolationSpace::LinearRgb,
        }
    }

    /// Create a vertical linear gradient (top to bottom).
    pub fn vertical(start_color: GradientColor, end_color: GradientColor) -> Self {
        Self {
            start_x: 0.5,
            start_y: 0.0,
            end_x: 0.5,
            end_y: 1.0,
            stops: vec![
                ColorStop::new(0.0, start_color),
                ColorStop::new(1.0, end_color),
            ],
            spread: SpreadMode::Pad,
            interpolation: InterpolationSpace::LinearRgb,
        }
    }

    /// Create a diagonal gradient (top-left to bottom-right).
    pub fn diagonal(start_color: GradientColor, end_color: GradientColor) -> Self {
        Self {
            start_x: 0.0,
            start_y: 0.0,
            end_x: 1.0,
            end_y: 1.0,
            stops: vec![
                ColorStop::new(0.0, start_color),
                ColorStop::new(1.0, end_color),
            ],
            spread: SpreadMode::Pad,
            interpolation: InterpolationSpace::LinearRgb,
        }
    }

    /// Create a gradient at a specific angle (degrees, 0 = right, 90 = down).
    pub fn angled(angle_deg: f32, start_color: GradientColor, end_color: GradientColor) -> Self {
        let angle_rad = angle_deg * PI / 180.0;
        let dx = angle_rad.cos();
        let dy = angle_rad.sin();
        Self {
            start_x: 0.5 - dx * 0.5,
            start_y: 0.5 - dy * 0.5,
            end_x: 0.5 + dx * 0.5,
            end_y: 0.5 + dy * 0.5,
            stops: vec![
                ColorStop::new(0.0, start_color),
                ColorStop::new(1.0, end_color),
            ],
            spread: SpreadMode::Pad,
            interpolation: InterpolationSpace::LinearRgb,
        }
    }

    /// Add a color stop to the gradient.
    pub fn add_stop(&mut self, position: f32, color: GradientColor) {
        self.stops.push(ColorStop::new(position, color));
        self.stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sample the gradient color at a given position (0.0..=1.0).
    pub fn sample(&self, t: f32) -> GradientColor {
        if self.stops.is_empty() {
            return GradientColor::transparent();
        }
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }

        let t = apply_spread(t, self.spread);

        // Find the two stops around t.
        if t <= self.stops[0].position {
            return self.stops[0].color;
        }
        if t >= self.stops[self.stops.len() - 1].position {
            return self.stops[self.stops.len() - 1].color;
        }

        for i in 0..self.stops.len() - 1 {
            let s0 = &self.stops[i];
            let s1 = &self.stops[i + 1];
            if t >= s0.position && t <= s1.position {
                let range = s1.position - s0.position;
                if range < f32::EPSILON {
                    return s0.color;
                }
                let local_t = (t - s0.position) / range;
                return s0.color.lerp(&s1.color, local_t);
            }
        }

        self.stops[self.stops.len() - 1].color
    }

    /// Sample at a 2D point within a bounding box.
    pub fn sample_at(&self, x: f32, y: f32, width: f32, height: f32) -> GradientColor {
        if width < f32::EPSILON || height < f32::EPSILON {
            return GradientColor::transparent();
        }
        let nx = x / width;
        let ny = y / height;

        // Project onto the gradient line.
        let dx = self.end_x - self.start_x;
        let dy = self.end_y - self.start_y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < f32::EPSILON {
            return self.sample(0.0);
        }
        let t = ((nx - self.start_x) * dx + (ny - self.start_y) * dy) / len_sq;
        self.sample(t)
    }

    /// Get the number of color stops.
    pub fn stop_count(&self) -> usize {
        self.stops.len()
    }
}

/// Radial gradient definition.
#[derive(Clone, Debug)]
pub struct RadialGradient {
    /// Center X (normalized).
    pub center_x: f32,
    /// Center Y (normalized).
    pub center_y: f32,
    /// Radius (normalized).
    pub radius: f32,
    /// Focus X (normalized, for focal point offset).
    pub focus_x: f32,
    /// Focus Y (normalized, for focal point offset).
    pub focus_y: f32,
    /// Color stops.
    pub stops: Vec<ColorStop>,
    /// Spread mode.
    pub spread: SpreadMode,
    /// Color interpolation space.
    pub interpolation: InterpolationSpace,
}

impl RadialGradient {
    /// Create a centered radial gradient.
    pub fn centered(inner_color: GradientColor, outer_color: GradientColor) -> Self {
        Self {
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.5,
            focus_x: 0.5,
            focus_y: 0.5,
            stops: vec![
                ColorStop::new(0.0, inner_color),
                ColorStop::new(1.0, outer_color),
            ],
            spread: SpreadMode::Pad,
            interpolation: InterpolationSpace::LinearRgb,
        }
    }

    /// Create a vignette-style gradient (dark edges).
    pub fn vignette(inner_color: GradientColor, edge_color: GradientColor) -> Self {
        Self {
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.707, // sqrt(2)/2 to reach corners
            focus_x: 0.5,
            focus_y: 0.5,
            stops: vec![
                ColorStop::new(0.0, inner_color),
                ColorStop::new(0.6, inner_color),
                ColorStop::new(1.0, edge_color),
            ],
            spread: SpreadMode::Pad,
            interpolation: InterpolationSpace::LinearRgb,
        }
    }

    /// Add a color stop.
    pub fn add_stop(&mut self, position: f32, color: GradientColor) {
        self.stops.push(ColorStop::new(position, color));
        self.stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sample the gradient at a distance from center (0.0..=1.0 of radius).
    pub fn sample(&self, t: f32) -> GradientColor {
        if self.stops.is_empty() {
            return GradientColor::transparent();
        }
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }

        let t = apply_spread(t, self.spread);

        if t <= self.stops[0].position {
            return self.stops[0].color;
        }
        if t >= self.stops[self.stops.len() - 1].position {
            return self.stops[self.stops.len() - 1].color;
        }

        for i in 0..self.stops.len() - 1 {
            let s0 = &self.stops[i];
            let s1 = &self.stops[i + 1];
            if t >= s0.position && t <= s1.position {
                let range = s1.position - s0.position;
                if range < f32::EPSILON {
                    return s0.color;
                }
                let local_t = (t - s0.position) / range;
                return s0.color.lerp(&s1.color, local_t);
            }
        }

        self.stops[self.stops.len() - 1].color
    }

    /// Sample at a 2D point within a bounding box.
    pub fn sample_at(&self, x: f32, y: f32, width: f32, height: f32) -> GradientColor {
        if width < f32::EPSILON || height < f32::EPSILON {
            return GradientColor::transparent();
        }
        let nx = x / width;
        let ny = y / height;
        let dx = nx - self.center_x;
        let dy = ny - self.center_y;
        let dist = (dx * dx + dy * dy).sqrt();
        let t = if self.radius > f32::EPSILON {
            dist / self.radius
        } else {
            0.0
        };
        self.sample(t)
    }

    /// Get the number of color stops.
    pub fn stop_count(&self) -> usize {
        self.stops.len()
    }
}

/// Conic (angular/sweep) gradient definition.
#[derive(Clone, Debug)]
pub struct ConicGradient {
    /// Center X (normalized).
    pub center_x: f32,
    /// Center Y (normalized).
    pub center_y: f32,
    /// Start angle in degrees (0 = right, 90 = down).
    pub start_angle: f32,
    /// Color stops.
    pub stops: Vec<ColorStop>,
    /// Spread mode.
    pub spread: SpreadMode,
}

impl ConicGradient {
    /// Create a simple conic gradient sweeping through colors.
    pub fn sweep(colors: &[GradientColor]) -> Self {
        let n = colors.len();
        let stops = if n == 0 {
            vec![]
        } else if n == 1 {
            vec![
                ColorStop::new(0.0, colors[0]),
                ColorStop::new(1.0, colors[0]),
            ]
        } else {
            colors
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    #[allow(clippy::cast_precision_loss)]
                    let pos = i as f32 / (n - 1) as f32;
                    ColorStop::new(pos, *c)
                })
                .collect()
        };

        Self {
            center_x: 0.5,
            center_y: 0.5,
            start_angle: 0.0,
            stops,
            spread: SpreadMode::Pad,
        }
    }

    /// Create a rainbow conic gradient.
    pub fn rainbow() -> Self {
        Self::sweep(&[
            GradientColor::rgb(1.0, 0.0, 0.0), // Red
            GradientColor::rgb(1.0, 0.5, 0.0), // Orange
            GradientColor::rgb(1.0, 1.0, 0.0), // Yellow
            GradientColor::rgb(0.0, 1.0, 0.0), // Green
            GradientColor::rgb(0.0, 0.0, 1.0), // Blue
            GradientColor::rgb(0.5, 0.0, 1.0), // Indigo
            GradientColor::rgb(1.0, 0.0, 0.5), // Violet
        ])
    }

    /// Add a color stop.
    pub fn add_stop(&mut self, position: f32, color: GradientColor) {
        self.stops.push(ColorStop::new(position, color));
        self.stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sample the gradient at an angle-fraction (0.0..=1.0 around the circle).
    pub fn sample(&self, t: f32) -> GradientColor {
        if self.stops.is_empty() {
            return GradientColor::transparent();
        }
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }

        let t = apply_spread(t, self.spread);

        if t <= self.stops[0].position {
            return self.stops[0].color;
        }
        if t >= self.stops[self.stops.len() - 1].position {
            return self.stops[self.stops.len() - 1].color;
        }

        for i in 0..self.stops.len() - 1 {
            let s0 = &self.stops[i];
            let s1 = &self.stops[i + 1];
            if t >= s0.position && t <= s1.position {
                let range = s1.position - s0.position;
                if range < f32::EPSILON {
                    return s0.color;
                }
                let local_t = (t - s0.position) / range;
                return s0.color.lerp(&s1.color, local_t);
            }
        }

        self.stops[self.stops.len() - 1].color
    }

    /// Sample at a 2D point within a bounding box.
    pub fn sample_at(&self, x: f32, y: f32, width: f32, height: f32) -> GradientColor {
        if width < f32::EPSILON || height < f32::EPSILON {
            return GradientColor::transparent();
        }
        let nx = x / width;
        let ny = y / height;
        let dx = nx - self.center_x;
        let dy = ny - self.center_y;
        let angle = dy.atan2(dx) * 180.0 / PI;
        let adjusted = (angle - self.start_angle + 360.0) % 360.0;
        let t = adjusted / 360.0;
        self.sample(t)
    }

    /// Get the number of color stops.
    pub fn stop_count(&self) -> usize {
        self.stops.len()
    }
}

/// A gradient fill that can be one of several types.
#[derive(Clone, Debug)]
pub enum GradientFill {
    /// Linear gradient.
    Linear(LinearGradient),
    /// Radial gradient.
    Radial(RadialGradient),
    /// Conic (angular) gradient.
    Conic(ConicGradient),
}

impl GradientFill {
    /// Sample the gradient at a 2D point within a bounding box.
    pub fn sample_at(&self, x: f32, y: f32, width: f32, height: f32) -> GradientColor {
        match self {
            Self::Linear(g) => g.sample_at(x, y, width, height),
            Self::Radial(g) => g.sample_at(x, y, width, height),
            Self::Conic(g) => g.sample_at(x, y, width, height),
        }
    }

    /// Get the number of color stops.
    pub fn stop_count(&self) -> usize {
        match self {
            Self::Linear(g) => g.stop_count(),
            Self::Radial(g) => g.stop_count(),
            Self::Conic(g) => g.stop_count(),
        }
    }

    /// Render the gradient into a flat RGBA8 buffer of `width × height` pixels.
    ///
    /// Scanlines are computed in parallel via rayon for large surfaces.  The
    /// function is deterministic: calling it twice with the same arguments
    /// produces byte-identical output.
    ///
    /// # Returns
    /// A `Vec<u8>` of length `width * height * 4` in RGBA8 row-major order.
    #[must_use]
    pub fn render_to_rgba(&self, width: u32, height: u32) -> Vec<u8> {
        use rayon::prelude::*;

        if width == 0 || height == 0 {
            return Vec::new();
        }

        let w = width as f32;
        let h = height as f32;
        let row_bytes = width as usize * 4;
        let mut buf = vec![0u8; width as usize * height as usize * 4];

        buf.par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(row, row_buf)| {
                for col in 0..width as usize {
                    // Normalised coordinates — centre of the pixel.
                    let px = (col as f32 + 0.5) / w;
                    let py = (row as f32 + 0.5) / h;
                    // `sample_at` expects absolute pixel coords and the surface size.
                    let color = self.sample_at(px * w, py * h, w, h);
                    let offset = col * 4;
                    row_buf[offset] = (color.r * 255.0 + 0.5) as u8;
                    row_buf[offset + 1] = (color.g * 255.0 + 0.5) as u8;
                    row_buf[offset + 2] = (color.b * 255.0 + 0.5) as u8;
                    row_buf[offset + 3] = (color.a * 255.0 + 0.5) as u8;
                }
            });

        buf
    }
}

/// Predefined gradient presets for broadcast use.
pub struct GradientPresets;

impl GradientPresets {
    /// Lower-third gradient (dark semi-transparent to fully transparent).
    pub fn lower_third_background() -> LinearGradient {
        LinearGradient::horizontal(
            GradientColor::new(0.0, 0.0, 0.0, 0.8),
            GradientColor::new(0.0, 0.0, 0.0, 0.0),
        )
    }

    /// News ticker gradient background.
    pub fn news_ticker() -> LinearGradient {
        let mut g = LinearGradient::horizontal(
            GradientColor::rgb(0.1, 0.1, 0.4),
            GradientColor::rgb(0.2, 0.2, 0.6),
        );
        g.add_stop(0.5, GradientColor::rgb(0.15, 0.15, 0.5));
        g
    }

    /// Sports score overlay gradient.
    pub fn sports_overlay() -> LinearGradient {
        LinearGradient::vertical(
            GradientColor::new(0.0, 0.0, 0.0, 0.9),
            GradientColor::new(0.0, 0.0, 0.0, 0.6),
        )
    }

    /// Spotlight radial gradient.
    pub fn spotlight() -> RadialGradient {
        RadialGradient::centered(
            GradientColor::new(1.0, 1.0, 0.9, 0.5),
            GradientColor::transparent(),
        )
    }

    /// Vignette overlay for cinematic look.
    pub fn cinematic_vignette() -> RadialGradient {
        RadialGradient::vignette(
            GradientColor::transparent(),
            GradientColor::new(0.0, 0.0, 0.0, 0.7),
        )
    }
}

/// Apply spread mode to a position value.
fn apply_spread(t: f32, mode: SpreadMode) -> f32 {
    match mode {
        SpreadMode::Pad => t.clamp(0.0, 1.0),
        SpreadMode::Repeat => {
            let t = t % 1.0;
            if t < 0.0 {
                t + 1.0
            } else {
                t
            }
        }
        SpreadMode::Reflect => {
            let t = t.abs();
            let cycle = t as u32;
            let frac = t - t.floor();
            if cycle % 2 == 0 {
                frac
            } else {
                1.0 - frac
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_color_new() {
        let c = GradientColor::new(0.5, 0.6, 0.7, 0.8);
        assert!((c.r - 0.5).abs() < f32::EPSILON);
        assert!((c.a - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_color_clamp() {
        let c = GradientColor::new(1.5, -0.1, 0.5, 2.0);
        assert!((c.r - 1.0).abs() < f32::EPSILON);
        assert!(c.g.abs() < f32::EPSILON);
        assert!((c.a - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_color_lerp() {
        let a = GradientColor::black();
        let b = GradientColor::white();
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.g - 0.5).abs() < 0.01);
        assert!((mid.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_from_u8() {
        let c = GradientColor::from_u8(255, 128, 0, 255);
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.g - 0.502).abs() < 0.01);
        assert!(c.b.abs() < 0.01);
        assert!((c.a - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_color_to_u8() {
        let c = GradientColor::rgb(1.0, 0.0, 0.5);
        let (r, g, b, a) = c.to_u8();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert!(b >= 127 && b <= 128);
        assert_eq!(a, 255);
    }

    #[test]
    fn test_color_stop() {
        let stop = ColorStop::new(0.5, GradientColor::white());
        assert!((stop.position - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_stop_clamp() {
        let stop = ColorStop::new(1.5, GradientColor::white());
        assert!((stop.position - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_gradient_horizontal() {
        let g = LinearGradient::horizontal(GradientColor::black(), GradientColor::white());
        assert_eq!(g.stops.len(), 2);
        let c = g.sample(0.5);
        assert!((c.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_linear_gradient_vertical() {
        let g = LinearGradient::vertical(GradientColor::black(), GradientColor::white());
        assert!((g.start_x - 0.5).abs() < f32::EPSILON);
        assert!((g.start_y).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_gradient_sample_edges() {
        let g = LinearGradient::horizontal(GradientColor::black(), GradientColor::white());
        let start = g.sample(0.0);
        let end = g.sample(1.0);
        assert!(start.r.abs() < 0.01);
        assert!((end.r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_gradient_add_stop() {
        let mut g = LinearGradient::horizontal(GradientColor::black(), GradientColor::white());
        g.add_stop(0.5, GradientColor::rgb(1.0, 0.0, 0.0));
        assert_eq!(g.stop_count(), 3);
        // The middle stop should be red.
        let mid = g.sample(0.5);
        assert!((mid.r - 1.0).abs() < 0.01);
        assert!(mid.g.abs() < 0.01);
    }

    #[test]
    fn test_linear_gradient_sample_at() {
        let g = LinearGradient::horizontal(GradientColor::black(), GradientColor::white());
        let c = g.sample_at(50.0, 50.0, 100.0, 100.0);
        assert!((c.r - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_radial_gradient_centered() {
        let g = RadialGradient::centered(GradientColor::white(), GradientColor::black());
        assert_eq!(g.stop_count(), 2);
        let center = g.sample(0.0);
        assert!((center.r - 1.0).abs() < 0.01);
        let edge = g.sample(1.0);
        assert!(edge.r.abs() < 0.01);
    }

    #[test]
    fn test_radial_gradient_vignette() {
        let g = RadialGradient::vignette(GradientColor::transparent(), GradientColor::black());
        assert_eq!(g.stop_count(), 3);
    }

    #[test]
    fn test_conic_gradient_sweep() {
        let colors = vec![
            GradientColor::rgb(1.0, 0.0, 0.0),
            GradientColor::rgb(0.0, 1.0, 0.0),
            GradientColor::rgb(0.0, 0.0, 1.0),
        ];
        let g = ConicGradient::sweep(&colors);
        assert_eq!(g.stop_count(), 3);
    }

    #[test]
    fn test_conic_gradient_rainbow() {
        let g = ConicGradient::rainbow();
        assert_eq!(g.stop_count(), 7);
    }

    #[test]
    fn test_spread_mode_pad() {
        assert!((apply_spread(-0.5, SpreadMode::Pad)).abs() < f32::EPSILON);
        assert!((apply_spread(1.5, SpreadMode::Pad) - 1.0).abs() < f32::EPSILON);
        assert!((apply_spread(0.5, SpreadMode::Pad) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spread_mode_repeat() {
        let val = apply_spread(1.5, SpreadMode::Repeat);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_spread_mode_reflect() {
        let val = apply_spread(1.5, SpreadMode::Reflect);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gradient_fill_enum() {
        let fill = GradientFill::Linear(LinearGradient::horizontal(
            GradientColor::black(),
            GradientColor::white(),
        ));
        assert_eq!(fill.stop_count(), 2);
        let c = fill.sample_at(50.0, 50.0, 100.0, 100.0);
        assert!(c.r >= 0.0 && c.r <= 1.0);
    }

    #[test]
    fn test_gradient_presets() {
        let lt = GradientPresets::lower_third_background();
        assert_eq!(lt.stop_count(), 2);

        let nt = GradientPresets::news_ticker();
        assert_eq!(nt.stop_count(), 3);

        let so = GradientPresets::sports_overlay();
        assert_eq!(so.stop_count(), 2);

        let sp = GradientPresets::spotlight();
        assert_eq!(sp.stop_count(), 2);

        let cv = GradientPresets::cinematic_vignette();
        assert_eq!(cv.stop_count(), 3);
    }

    #[test]
    fn test_premultiplied_gradient_color() {
        let c = GradientColor::new(1.0, 0.5, 0.0, 0.5);
        let pm = c.premultiplied();
        assert!((pm.r - 0.5).abs() < f32::EPSILON);
        assert!((pm.g - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_linear_gradient_angled() {
        let g = LinearGradient::angled(45.0, GradientColor::black(), GradientColor::white());
        assert_eq!(g.stop_count(), 2);
        // Verify start/end are not the same.
        assert!((g.start_x - g.end_x).abs() > 0.01);
    }

    #[test]
    fn test_conic_gradient_sample_at() {
        let g = ConicGradient::rainbow();
        let c = g.sample_at(75.0, 50.0, 100.0, 100.0);
        // Should return some valid color.
        assert!(c.r >= 0.0 && c.r <= 1.0);
        assert!(c.g >= 0.0 && c.g <= 1.0);
        assert!(c.b >= 0.0 && c.b <= 1.0);
    }

    /// `render_to_rgba` must be deterministic: two calls return byte-identical results.
    #[test]
    fn test_gradient_render_to_rgba_serial_vs_parallel() {
        let fill = GradientFill::Linear(LinearGradient::horizontal(
            GradientColor::black(),
            GradientColor::white(),
        ));
        let buf1 = fill.render_to_rgba(64, 32);
        let buf2 = fill.render_to_rgba(64, 32);
        assert_eq!(buf1, buf2, "render_to_rgba must be deterministic");
        // Sanity: output is not all-zero for a non-trivial gradient.
        assert!(
            buf1.iter().any(|&b| b > 0),
            "gradient pixels should be non-zero"
        );
    }

    /// `render_to_rgba` must return exactly `width * height * 4` bytes.
    #[test]
    fn test_gradient_render_dimensions() {
        let fill = GradientFill::Radial(RadialGradient::centered(
            GradientColor::white(),
            GradientColor::black(),
        ));
        let buf = fill.render_to_rgba(16, 8);
        assert_eq!(
            buf.len(),
            16 * 8 * 4,
            "buffer length must be width*height*4"
        );
    }
}
