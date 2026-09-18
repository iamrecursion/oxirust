#![allow(dead_code)]
//! Text rendering primitives for broadcast graphics.
//!
//! Provides text alignment, font weight, and style types along with a
//! `TextRenderer` that measures text geometry and a `TextStyle` builder.

/// Horizontal or vertical alignment of rendered text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextAlignment {
    /// Left-align text within the bounding box.
    Left,
    /// Centre-align text horizontally.
    Center,
    /// Right-align text.
    Right,
    /// Justify text to both edges.
    Justify,
    /// Align text to the top of its container.
    Top,
    /// Align text to the vertical centre of its container.
    Middle,
    /// Align text to the bottom of its container.
    Bottom,
}

impl TextAlignment {
    /// Returns `true` for alignments that operate on the horizontal axis.
    pub fn is_horizontal(&self) -> bool {
        matches!(
            self,
            TextAlignment::Left
                | TextAlignment::Center
                | TextAlignment::Right
                | TextAlignment::Justify
        )
    }

    /// Returns `true` for alignments that operate on the vertical axis.
    pub fn is_vertical(&self) -> bool {
        !self.is_horizontal()
    }
}

/// Font weight descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontWeight {
    /// Thin (100).
    Thin,
    /// Extra-light (200).
    ExtraLight,
    /// Light (300).
    Light,
    /// Regular (400).
    Regular,
    /// Medium (500).
    Medium,
    /// Semi-bold (600).
    SemiBold,
    /// Bold (700).
    Bold,
    /// Extra-bold (800).
    ExtraBold,
    /// Black (900).
    Black,
}

impl FontWeight {
    /// Returns `true` for weights ≥ Bold (700).
    pub fn is_bold(&self) -> bool {
        matches!(
            self,
            FontWeight::Bold | FontWeight::ExtraBold | FontWeight::Black
        )
    }

    /// Numeric CSS-style weight value.
    pub fn numeric_value(&self) -> u16 {
        match self {
            FontWeight::Thin => 100,
            FontWeight::ExtraLight => 200,
            FontWeight::Light => 300,
            FontWeight::Regular => 400,
            FontWeight::Medium => 500,
            FontWeight::SemiBold => 600,
            FontWeight::Bold => 700,
            FontWeight::ExtraBold => 800,
            FontWeight::Black => 900,
        }
    }
}

/// Complete style specification for a block of rendered text.
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Font family name.
    pub font_family: String,
    /// Font size in points.
    pub font_size_pt: f32,
    /// Font weight.
    pub weight: FontWeight,
    /// Horizontal alignment.
    pub alignment: TextAlignment,
    /// Text opacity in `[0.0, 1.0]`.
    pub opacity: f32,
    /// Whether italic rendering is requested.
    pub italic: bool,
    /// RGBA packed colour.
    pub color: u32,
    /// Letter-spacing in pixels (positive = wider).
    pub letter_spacing_px: f32,
    /// Line-height multiplier (1.0 = normal).
    pub line_height: f32,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_family: "Arial".to_string(),
            font_size_pt: 24.0,
            weight: FontWeight::Regular,
            alignment: TextAlignment::Left,
            opacity: 1.0,
            italic: false,
            color: 0xFFFF_FFFF,
            letter_spacing_px: 0.0,
            line_height: 1.2,
        }
    }
}

impl TextStyle {
    /// Create a style with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when the text will be rendered visibly (opacity > 0).
    pub fn is_visible(&self) -> bool {
        self.opacity > 0.0
    }

    /// Returns `true` when the style uses a bold font weight.
    pub fn is_bold(&self) -> bool {
        self.weight.is_bold()
    }

    /// Returns `true` when the style uses italic rendering.
    pub fn is_italic(&self) -> bool {
        self.italic
    }

    /// Builder-style setter for font family.
    pub fn with_font(mut self, family: &str) -> Self {
        self.font_family = family.to_string();
        self
    }

    /// Builder-style setter for font size.
    pub fn with_size(mut self, pt: f32) -> Self {
        self.font_size_pt = pt;
        self
    }

    /// Builder-style setter for weight.
    pub fn with_weight(mut self, weight: FontWeight) -> Self {
        self.weight = weight;
        self
    }

    /// Builder-style setter for opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

/// Simple text renderer that provides measurement and layout utilities.
///
/// This is a lightweight, dependency-free model renderer. Actual pixel rasterisation
/// is delegated to the GPU layer; this struct provides geometry metrics only.
#[derive(Debug, Clone)]
pub struct TextRenderer {
    /// Pixels per point scaling factor.
    pub px_per_pt: f32,
    /// Average glyph width as a fraction of `font_size_pt` (simplistic model).
    pub avg_glyph_width_ratio: f32,
    /// Maximum bounding-box width in pixels.
    pub max_width_px: f32,
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self {
            px_per_pt: 1.333_333,
            avg_glyph_width_ratio: 0.55,
            max_width_px: 1920.0,
        }
    }
}

impl TextRenderer {
    /// Create a renderer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a renderer for a specific display.
    pub fn with_display(px_per_pt: f32, max_width_px: f32) -> Self {
        Self {
            px_per_pt,
            max_width_px,
            ..Default::default()
        }
    }

    /// Estimate the rendered pixel width of `text` with the given style.
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_width(&self, text: &str, style: &TextStyle) -> f32 {
        let char_count = text.chars().count() as f32;
        let glyph_width_px = style.font_size_pt * self.px_per_pt * self.avg_glyph_width_ratio;
        let spacing_total = style.letter_spacing_px * (char_count - 1.0).max(0.0);
        (glyph_width_px * char_count + spacing_total).min(self.max_width_px)
    }

    /// Estimate the number of wrapped lines for `text` inside `container_width_px`.
    ///
    /// Uses a word-wrap model: splits on whitespace and accumulates widths.
    #[allow(clippy::cast_precision_loss)]
    pub fn line_count(&self, text: &str, style: &TextStyle, container_width_px: f32) -> usize {
        if text.is_empty() {
            return 0;
        }
        let glyph_width_px = style.font_size_pt * self.px_per_pt * self.avg_glyph_width_ratio;
        let space_width = glyph_width_px;
        let mut lines = 1usize;
        let mut x = 0.0f32;
        for word in text.split_whitespace() {
            let word_width = word.chars().count() as f32 * glyph_width_px;
            if x > 0.0 && x + space_width + word_width > container_width_px {
                lines += 1;
                x = word_width;
            } else {
                if x > 0.0 {
                    x += space_width;
                }
                x += word_width;
            }
        }
        lines
    }

    /// Estimate the rendered pixel height of `text` using line metrics.
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_height(&self, text: &str, style: &TextStyle, container_width_px: f32) -> f32 {
        let lines = self.line_count(text, style, container_width_px) as f32;
        let line_height_px = style.font_size_pt * self.px_per_pt * style.line_height;
        lines * line_height_px
    }

    /// Returns `true` when the estimated text width fits within `container_width_px`.
    pub fn fits_on_one_line(&self, text: &str, style: &TextStyle, container_width_px: f32) -> bool {
        self.measure_width(text, style) <= container_width_px
    }
}

// ---------------------------------------------------------------------------
// Text outline configuration
// ---------------------------------------------------------------------------

/// Configuration for a text stroke/outline effect.
#[derive(Debug, Clone)]
pub struct TextOutline {
    /// RGBA color of the outline (packed as 0xRRGGBBAA).
    pub color: u32,
    /// Width of the outline in pixels.
    pub width_px: f32,
}

impl Default for TextOutline {
    fn default() -> Self {
        Self {
            color: 0x000000FF, // opaque black
            width_px: 1.5,
        }
    }
}

impl TextOutline {
    /// Create a text outline with an explicit color and width.
    pub fn new(color: u32, width_px: f32) -> Self {
        Self {
            color,
            width_px: width_px.max(0.0),
        }
    }

    /// Returns `true` if the outline is visible (positive width).
    pub fn is_visible(&self) -> bool {
        self.width_px > 0.0
    }

    /// Extract RGBA bytes from the packed color.
    pub fn rgba_bytes(&self) -> [u8; 4] {
        [
            ((self.color >> 24) & 0xFF) as u8,
            ((self.color >> 16) & 0xFF) as u8,
            ((self.color >> 8) & 0xFF) as u8,
            (self.color & 0xFF) as u8,
        ]
    }
}

// ---------------------------------------------------------------------------
// Drop shadow configuration
// ---------------------------------------------------------------------------

/// Configuration for a text drop shadow effect.
#[derive(Debug, Clone)]
pub struct DropShadow {
    /// RGBA color of the shadow (packed as 0xRRGGBBAA).
    pub color: u32,
    /// Horizontal shadow offset in pixels (positive = right).
    pub offset_x_px: f32,
    /// Vertical shadow offset in pixels (positive = down).
    pub offset_y_px: f32,
    /// Shadow blur radius in pixels.
    pub blur_radius_px: f32,
    /// Shadow opacity multiplier in [0.0, 1.0].
    pub opacity: f32,
}

impl Default for DropShadow {
    fn default() -> Self {
        Self {
            color: 0x000000AA, // semi-transparent black
            offset_x_px: 2.0,
            offset_y_px: 2.0,
            blur_radius_px: 3.0,
            opacity: 0.7,
        }
    }
}

impl DropShadow {
    /// Create a drop shadow with all parameters.
    pub fn new(
        color: u32,
        offset_x_px: f32,
        offset_y_px: f32,
        blur_radius_px: f32,
        opacity: f32,
    ) -> Self {
        Self {
            color,
            offset_x_px,
            offset_y_px,
            blur_radius_px: blur_radius_px.max(0.0),
            opacity: opacity.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` if the shadow is visible.
    pub fn is_visible(&self) -> bool {
        self.opacity > 0.0
            && (self.offset_x_px.abs() + self.offset_y_px.abs() + self.blur_radius_px) > 0.0
    }

    /// Extract RGBA bytes from the packed color with opacity applied.
    pub fn rgba_bytes(&self) -> [u8; 4] {
        let r = ((self.color >> 24) & 0xFF) as u8;
        let g = ((self.color >> 16) & 0xFF) as u8;
        let b = ((self.color >> 8) & 0xFF) as u8;
        let base_a = (self.color & 0xFF) as u8;
        let a = (base_a as f32 * self.opacity) as u8;
        [r, g, b, a]
    }

    /// Pixel area required around the text to accommodate the shadow.
    ///
    /// Returns `(left, top, right, bottom)` extra margins in pixels.
    pub fn required_margins(&self) -> (f32, f32, f32, f32) {
        let blur = self.blur_radius_px;
        let ox = self.offset_x_px;
        let oy = self.offset_y_px;
        let left = (-ox + blur).max(0.0);
        let right = (ox + blur).max(0.0);
        let top = (-oy + blur).max(0.0);
        let bottom = (oy + blur).max(0.0);
        (left, top, right, bottom)
    }
}

// ---------------------------------------------------------------------------
// Enhanced text style with outline and shadow
// ---------------------------------------------------------------------------

/// Enriched text style that carries optional outline and drop shadow.
#[derive(Debug, Clone)]
pub struct RichTextStyle {
    /// Base text style.
    pub base: TextStyle,
    /// Optional text outline rendered beneath the fill.
    pub outline: Option<TextOutline>,
    /// Optional drop shadow rendered beneath the outline.
    pub shadow: Option<DropShadow>,
}

impl Default for RichTextStyle {
    fn default() -> Self {
        Self {
            base: TextStyle::default(),
            outline: None,
            shadow: None,
        }
    }
}

impl RichTextStyle {
    /// Create from a base style.
    pub fn from_base(base: TextStyle) -> Self {
        Self {
            base,
            outline: None,
            shadow: None,
        }
    }

    /// Attach an outline to this style.
    pub fn with_outline(mut self, outline: TextOutline) -> Self {
        self.outline = Some(outline);
        self
    }

    /// Attach a drop shadow to this style.
    pub fn with_shadow(mut self, shadow: DropShadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Returns `true` if any outline is configured and visible.
    pub fn has_visible_outline(&self) -> bool {
        self.outline.as_ref().is_some_and(TextOutline::is_visible)
    }

    /// Returns `true` if any shadow is configured and visible.
    pub fn has_visible_shadow(&self) -> bool {
        self.shadow.as_ref().is_some_and(DropShadow::is_visible)
    }

    /// Compute the expanded bounding box that accommodates the shadow.
    ///
    /// Returns extra pixels `(left, top, right, bottom)` around the text.
    pub fn shadow_margins(&self) -> (f32, f32, f32, f32) {
        self.shadow
            .as_ref()
            .map(|s| s.required_margins())
            .unwrap_or((0.0, 0.0, 0.0, 0.0))
    }

    /// Compute the extra radius consumed by the outline on each side.
    pub fn outline_radius(&self) -> f32 {
        self.outline.as_ref().map_or(0.0, |o| o.width_px)
    }

    /// Total extra margin needed on each side for outline + shadow.
    ///
    /// Returns `(left, top, right, bottom)` expansions in pixels.
    pub fn total_margins(&self) -> (f32, f32, f32, f32) {
        let or_ = self.outline_radius();
        let (sl, st, sr, sb) = self.shadow_margins();
        (sl + or_, st + or_, sr + or_, sb + or_)
    }
}

// ---------------------------------------------------------------------------
// Stroke join style for outlines
// ---------------------------------------------------------------------------

/// Join style for text stroke/outline corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrokeJoinStyle {
    /// Miter join — sharp corners (default for broadcast text).
    #[default]
    Miter,
    /// Round join — smooth rounded corners.
    Round,
    /// Bevel join — flat cut corners.
    Bevel,
}

/// Enhanced text outline with configurable join style and miter limit.
#[derive(Debug, Clone)]
pub struct StyledTextOutline {
    /// RGBA color of the outline (packed as 0xRRGGBBAA).
    pub color: u32,
    /// Width of the outline in pixels.
    pub width_px: f32,
    /// Join style at corners.
    pub join_style: StrokeJoinStyle,
    /// Miter limit (only applies when `join_style` is `Miter`).
    /// When the miter length exceeds `width_px * miter_limit`, the join
    /// degrades to a bevel.
    pub miter_limit: f32,
}

impl Default for StyledTextOutline {
    fn default() -> Self {
        Self {
            color: 0x000000FF,
            width_px: 2.0,
            join_style: StrokeJoinStyle::Miter,
            miter_limit: 4.0,
        }
    }
}

impl StyledTextOutline {
    /// Create a new styled outline.
    pub fn new(color: u32, width_px: f32, join_style: StrokeJoinStyle) -> Self {
        Self {
            color,
            width_px: width_px.max(0.0),
            join_style,
            miter_limit: 4.0,
        }
    }

    /// Builder: set miter limit.
    pub fn with_miter_limit(mut self, limit: f32) -> Self {
        self.miter_limit = limit.max(1.0);
        self
    }

    /// Returns `true` if the outline is visible (positive width).
    pub fn is_visible(&self) -> bool {
        self.width_px > 0.0
    }

    /// Extract RGBA bytes from the packed color.
    pub fn rgba_bytes(&self) -> [u8; 4] {
        [
            ((self.color >> 24) & 0xFF) as u8,
            ((self.color >> 16) & 0xFF) as u8,
            ((self.color >> 8) & 0xFF) as u8,
            (self.color & 0xFF) as u8,
        ]
    }

    /// Effective miter length at the current width and limit.
    pub fn effective_miter_length(&self) -> f32 {
        self.width_px * self.miter_limit
    }

    /// Convert to a basic [`TextOutline`] (discards join info).
    pub fn to_basic_outline(&self) -> TextOutline {
        TextOutline::new(self.color, self.width_px)
    }
}

// ---------------------------------------------------------------------------
// Text glow effect
// ---------------------------------------------------------------------------

/// Glow direction: inner, outer, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlowDirection {
    /// Glow radiates outward from the text edge.
    #[default]
    Outer,
    /// Glow radiates inward into the text body.
    Inner,
    /// Both inner and outer glow.
    Both,
}

/// Configuration for a text glow effect.
///
/// A glow is similar to a shadow with zero offset; it radiates uniformly
/// from the text edges.
#[derive(Debug, Clone)]
pub struct TextGlow {
    /// RGBA color of the glow (packed as 0xRRGGBBAA).
    pub color: u32,
    /// Spread radius in pixels — how far the glow extends.
    pub spread_px: f32,
    /// Blur radius in pixels — softness of the glow edge.
    pub blur_radius_px: f32,
    /// Glow direction.
    pub direction: GlowDirection,
    /// Opacity multiplier in [0.0, 1.0].
    pub opacity: f32,
}

impl Default for TextGlow {
    fn default() -> Self {
        Self {
            color: 0xFFFFFFCC, // semi-transparent white
            spread_px: 4.0,
            blur_radius_px: 6.0,
            direction: GlowDirection::Outer,
            opacity: 0.8,
        }
    }
}

impl TextGlow {
    /// Create a new text glow.
    pub fn new(color: u32, spread_px: f32, blur_radius_px: f32, direction: GlowDirection) -> Self {
        Self {
            color,
            spread_px: spread_px.max(0.0),
            blur_radius_px: blur_radius_px.max(0.0),
            direction,
            opacity: 1.0,
        }
    }

    /// Builder: set opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Returns `true` if the glow is visible.
    pub fn is_visible(&self) -> bool {
        self.opacity > 0.0 && (self.spread_px + self.blur_radius_px) > 0.0
    }

    /// Extract RGBA bytes with opacity baked into the alpha channel.
    pub fn rgba_bytes(&self) -> [u8; 4] {
        let r = ((self.color >> 24) & 0xFF) as u8;
        let g = ((self.color >> 16) & 0xFF) as u8;
        let b = ((self.color >> 8) & 0xFF) as u8;
        let base_a = (self.color & 0xFF) as u8;
        let a = (base_a as f32 * self.opacity) as u8;
        [r, g, b, a]
    }

    /// Total extra margin required around text to accommodate the outer glow.
    ///
    /// Inner-only glows don't require extra margin.
    pub fn required_margin(&self) -> f32 {
        match self.direction {
            GlowDirection::Inner => 0.0,
            GlowDirection::Outer | GlowDirection::Both => self.spread_px + self.blur_radius_px,
        }
    }

    /// Convert to a [`DropShadow`] equivalent (zero offset, spread as blur).
    pub fn to_drop_shadow(&self) -> DropShadow {
        DropShadow::new(
            self.color,
            0.0,
            0.0,
            self.spread_px + self.blur_radius_px,
            self.opacity,
        )
    }
}

// ---------------------------------------------------------------------------
// Gradient text fill
// ---------------------------------------------------------------------------

/// A color stop in a gradient.
#[derive(Debug, Clone)]
pub struct GradientStop {
    /// Position along the gradient axis in [0.0, 1.0].
    pub position: f32,
    /// RGBA color at this stop (packed as 0xRRGGBBAA).
    pub color: u32,
}

impl GradientStop {
    /// Create a new gradient stop.
    pub fn new(position: f32, color: u32) -> Self {
        Self {
            position: position.clamp(0.0, 1.0),
            color,
        }
    }

    /// Extract RGBA bytes.
    pub fn rgba_bytes(&self) -> [u8; 4] {
        [
            ((self.color >> 24) & 0xFF) as u8,
            ((self.color >> 16) & 0xFF) as u8,
            ((self.color >> 8) & 0xFF) as u8,
            (self.color & 0xFF) as u8,
        ]
    }
}

/// Type of gradient applied to text fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradientType {
    /// Linear gradient along a direction vector.
    #[default]
    Linear,
    /// Radial gradient from a center point.
    Radial,
}

/// Configuration for a gradient text fill.
///
/// The gradient is applied across the text bounding box.  For `Linear`,
/// the angle (in degrees, 0 = left-to-right, 90 = top-to-bottom) defines
/// the direction.  For `Radial`, the center is at (center_x, center_y)
/// as fractions of the bounding box dimensions.
#[derive(Debug, Clone)]
pub struct GradientFill {
    /// Type of gradient.
    pub gradient_type: GradientType,
    /// Color stops (must be sorted by position; at least 2 required).
    pub stops: Vec<GradientStop>,
    /// Angle in degrees (for linear gradient; 0 = left-to-right).
    pub angle_deg: f32,
    /// Center X as fraction of bounding box width [0.0, 1.0] (for radial).
    pub center_x: f32,
    /// Center Y as fraction of bounding box height [0.0, 1.0] (for radial).
    pub center_y: f32,
}

impl Default for GradientFill {
    fn default() -> Self {
        Self {
            gradient_type: GradientType::Linear,
            stops: vec![
                GradientStop::new(0.0, 0xFFFFFFFF), // white
                GradientStop::new(1.0, 0x000000FF), // black
            ],
            angle_deg: 0.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }
}

impl GradientFill {
    /// Create a linear gradient between two colors.
    pub fn linear(color_start: u32, color_end: u32, angle_deg: f32) -> Self {
        Self {
            gradient_type: GradientType::Linear,
            stops: vec![
                GradientStop::new(0.0, color_start),
                GradientStop::new(1.0, color_end),
            ],
            angle_deg,
            center_x: 0.5,
            center_y: 0.5,
        }
    }

    /// Create a radial gradient between two colors.
    pub fn radial(color_center: u32, color_edge: u32) -> Self {
        Self {
            gradient_type: GradientType::Radial,
            stops: vec![
                GradientStop::new(0.0, color_center),
                GradientStop::new(1.0, color_edge),
            ],
            angle_deg: 0.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }

    /// Builder: add a color stop.
    pub fn with_stop(mut self, position: f32, color: u32) -> Self {
        self.stops.push(GradientStop::new(position, color));
        self.stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self
    }

    /// Builder: set radial center.
    pub fn with_center(mut self, cx: f32, cy: f32) -> Self {
        self.center_x = cx.clamp(0.0, 1.0);
        self.center_y = cy.clamp(0.0, 1.0);
        self
    }

    /// Returns `true` if the gradient has at least 2 stops.
    pub fn is_valid(&self) -> bool {
        self.stops.len() >= 2
    }

    /// Sample the gradient color at position `t` in [0.0, 1.0].
    ///
    /// Linearly interpolates between the nearest surrounding stops.
    /// Returns an RGBA `[u8; 4]`.
    pub fn sample(&self, t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        if self.stops.is_empty() {
            return [255, 255, 255, 255];
        }
        if self.stops.len() == 1 {
            return self.stops[0].rgba_bytes();
        }

        // Find surrounding stops
        let mut lower_idx = 0;
        let mut upper_idx = self.stops.len() - 1;
        for (i, stop) in self.stops.iter().enumerate() {
            if stop.position <= t {
                lower_idx = i;
            }
            if stop.position >= t && i < upper_idx {
                upper_idx = i;
                break;
            }
        }

        let lower = &self.stops[lower_idx];
        let upper = &self.stops[upper_idx];

        if (upper.position - lower.position).abs() < f32::EPSILON {
            return lower.rgba_bytes();
        }

        let frac = (t - lower.position) / (upper.position - lower.position);
        let lo = lower.rgba_bytes();
        let hi = upper.rgba_bytes();

        [
            (lo[0] as f32 + (hi[0] as f32 - lo[0] as f32) * frac) as u8,
            (lo[1] as f32 + (hi[1] as f32 - lo[1] as f32) * frac) as u8,
            (lo[2] as f32 + (hi[2] as f32 - lo[2] as f32) * frac) as u8,
            (lo[3] as f32 + (hi[3] as f32 - lo[3] as f32) * frac) as u8,
        ]
    }

    /// Compute the gradient parameter `t` for a point `(x, y)` within a
    /// bounding box of `(width, height)`.
    pub fn compute_t(&self, x: f32, y: f32, width: f32, height: f32) -> f32 {
        match self.gradient_type {
            GradientType::Linear => {
                let angle_rad = self.angle_deg.to_radians();
                let dx = angle_rad.cos();
                let dy = angle_rad.sin();
                // Normalize (x,y) to [0,1] range within bounding box
                let nx = if width > 0.0 { x / width } else { 0.0 };
                let ny = if height > 0.0 { y / height } else { 0.0 };
                // Project onto gradient direction
                let dot = nx * dx + ny * dy;
                // Normalize to [0,1] range
                let max_proj = dx.abs() + dy.abs();
                if max_proj > 0.0 {
                    (dot / max_proj).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            GradientType::Radial => {
                let cx = self.center_x * width;
                let cy = self.center_y * height;
                let dx = x - cx;
                let dy = y - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let max_dist = ((width * width + height * height) / 4.0).sqrt();
                if max_dist > 0.0 {
                    (dist / max_dist).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Convenience: sample the color at pixel `(x, y)` within a bounding box.
    pub fn color_at(&self, x: f32, y: f32, width: f32, height: f32) -> [u8; 4] {
        let t = self.compute_t(x, y, width, height);
        self.sample(t)
    }
}

// ---------------------------------------------------------------------------
// Fully enriched text style with all effects
// ---------------------------------------------------------------------------

/// A comprehensive text rendering style that supports outline, shadow, glow,
/// and gradient fill simultaneously.
#[derive(Debug, Clone)]
pub struct FullEffectTextStyle {
    /// Base text style.
    pub base: TextStyle,
    /// Optional styled outline (with join style).
    pub outline: Option<StyledTextOutline>,
    /// Optional drop shadow.
    pub shadow: Option<DropShadow>,
    /// Optional glow effect.
    pub glow: Option<TextGlow>,
    /// Optional gradient fill (replaces solid `base.color` when present).
    pub gradient: Option<GradientFill>,
}

impl Default for FullEffectTextStyle {
    fn default() -> Self {
        Self {
            base: TextStyle::default(),
            outline: None,
            shadow: None,
            glow: None,
            gradient: None,
        }
    }
}

impl FullEffectTextStyle {
    /// Create from a base style.
    pub fn from_base(base: TextStyle) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }

    /// Builder: attach a styled outline.
    pub fn with_outline(mut self, outline: StyledTextOutline) -> Self {
        self.outline = Some(outline);
        self
    }

    /// Builder: attach a drop shadow.
    pub fn with_shadow(mut self, shadow: DropShadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Builder: attach a glow.
    pub fn with_glow(mut self, glow: TextGlow) -> Self {
        self.glow = Some(glow);
        self
    }

    /// Builder: attach a gradient fill.
    pub fn with_gradient(mut self, gradient: GradientFill) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Returns `true` if any visual effect (outline/shadow/glow/gradient) is attached.
    pub fn has_effects(&self) -> bool {
        self.outline.is_some()
            || self.shadow.is_some()
            || self.glow.is_some()
            || self.gradient.is_some()
    }

    /// Total extra margin needed on each side for all effects combined.
    ///
    /// Returns `(left, top, right, bottom)`.
    pub fn total_margins(&self) -> (f32, f32, f32, f32) {
        let outline_r = self.outline.as_ref().map_or(0.0, |o| o.width_px);
        let shadow_margins = self
            .shadow
            .as_ref()
            .map(|s| s.required_margins())
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let glow_m = self.glow.as_ref().map_or(0.0, |g| g.required_margin());

        (
            shadow_margins.0 + outline_r + glow_m,
            shadow_margins.1 + outline_r + glow_m,
            shadow_margins.2 + outline_r + glow_m,
            shadow_margins.3 + outline_r + glow_m,
        )
    }

    /// Convert to a [`RichTextStyle`] (only carries outline + shadow, loses
    /// glow and gradient).
    pub fn to_rich_text_style(&self) -> RichTextStyle {
        let mut rts = RichTextStyle::from_base(self.base.clone());
        if let Some(ref o) = self.outline {
            rts = rts.with_outline(o.to_basic_outline());
        }
        if let Some(ref s) = self.shadow {
            rts = rts.with_shadow(s.clone());
        }
        rts
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_left_is_horizontal() {
        assert!(TextAlignment::Left.is_horizontal());
    }

    #[test]
    fn test_alignment_center_is_horizontal() {
        assert!(TextAlignment::Center.is_horizontal());
    }

    #[test]
    fn test_alignment_top_is_not_horizontal() {
        assert!(!TextAlignment::Top.is_horizontal());
    }

    #[test]
    fn test_alignment_middle_is_vertical() {
        assert!(TextAlignment::Middle.is_vertical());
    }

    #[test]
    fn test_font_weight_bold_is_bold() {
        assert!(FontWeight::Bold.is_bold());
    }

    #[test]
    fn test_font_weight_regular_not_bold() {
        assert!(!FontWeight::Regular.is_bold());
    }

    #[test]
    fn test_font_weight_black_is_bold() {
        assert!(FontWeight::Black.is_bold());
    }

    #[test]
    fn test_font_weight_numeric_regular() {
        assert_eq!(FontWeight::Regular.numeric_value(), 400);
    }

    #[test]
    fn test_font_weight_numeric_bold() {
        assert_eq!(FontWeight::Bold.numeric_value(), 700);
    }

    #[test]
    fn test_text_style_visible_default() {
        let style = TextStyle::default();
        assert!(style.is_visible());
    }

    #[test]
    fn test_text_style_invisible_when_zero_opacity() {
        let style = TextStyle::default().with_opacity(0.0);
        assert!(!style.is_visible());
    }

    #[test]
    fn test_text_style_builder_font() {
        let style = TextStyle::new().with_font("Helvetica");
        assert_eq!(style.font_family, "Helvetica");
    }

    #[test]
    fn test_text_style_builder_weight() {
        let style = TextStyle::new().with_weight(FontWeight::Bold);
        assert!(style.is_bold());
    }

    #[test]
    fn test_renderer_measure_width_empty() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        assert_eq!(r.measure_width("", &s), 0.0);
    }

    #[test]
    fn test_renderer_measure_width_positive() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        assert!(r.measure_width("Hello World", &s) > 0.0);
    }

    #[test]
    fn test_renderer_line_count_empty() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        assert_eq!(r.line_count("", &s, 1920.0), 0);
    }

    #[test]
    fn test_renderer_line_count_single_short_word() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        assert_eq!(r.line_count("Hi", &s, 1920.0), 1);
    }

    #[test]
    fn test_renderer_line_count_wraps_on_narrow_container() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        // Very narrow container forces wrapping
        let lines = r.line_count("This is a long line of text that should wrap", &s, 50.0);
        assert!(lines > 1);
    }

    #[test]
    fn test_renderer_fits_on_one_line_short_text() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        assert!(r.fits_on_one_line("Hi", &s, 1920.0));
    }

    #[test]
    fn test_renderer_does_not_fit_narrow_container() {
        let r = TextRenderer::new();
        let s = TextStyle::default();
        assert!(!r.fits_on_one_line("A very long title that will not fit", &s, 10.0));
    }

    // --- TextOutline tests ---

    #[test]
    fn test_text_outline_default_is_visible() {
        let o = TextOutline::default();
        assert!(o.is_visible());
    }

    #[test]
    fn test_text_outline_zero_width_not_visible() {
        let o = TextOutline::new(0xFF000000, 0.0);
        assert!(!o.is_visible());
    }

    #[test]
    fn test_text_outline_rgba_bytes() {
        // 0xRRGGBBAA = 0xFF_80_00_FF -> r=255, g=128, b=0, a=255
        let o = TextOutline::new(0xFF8000FF, 2.0);
        let bytes = o.rgba_bytes();
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x80);
        assert_eq!(bytes[2], 0x00);
        assert_eq!(bytes[3], 0xFF);
    }

    // --- DropShadow tests ---

    #[test]
    fn test_drop_shadow_default_is_visible() {
        let s = DropShadow::default();
        assert!(s.is_visible());
    }

    #[test]
    fn test_drop_shadow_zero_opacity_not_visible() {
        let s = DropShadow::new(0x000000FF, 2.0, 2.0, 3.0, 0.0);
        assert!(!s.is_visible());
    }

    #[test]
    fn test_drop_shadow_rgba_bytes_applies_opacity() {
        let s = DropShadow::new(0x000000FF, 2.0, 2.0, 0.0, 0.5);
        let bytes = s.rgba_bytes();
        // base_a=255, opacity=0.5 → a ≈ 127
        assert!(bytes[3] > 100 && bytes[3] < 140, "alpha={}", bytes[3]);
    }

    #[test]
    fn test_drop_shadow_required_margins() {
        let s = DropShadow::new(0x000000FF, 4.0, 6.0, 2.0, 1.0);
        let (l, t, r, b) = s.required_margins();
        // offset=(4,6), blur=2
        // left = max(0, -4+2) = 0, right = max(0, 4+2)=6
        // top  = max(0, -6+2) = 0, bottom= max(0, 6+2)=8
        assert!((l - 0.0).abs() < 0.01);
        assert!((t - 0.0).abs() < 0.01);
        assert!((r - 6.0).abs() < 0.01);
        assert!((b - 8.0).abs() < 0.01);
    }

    // --- RichTextStyle tests ---

    #[test]
    fn test_rich_text_style_default_no_effects() {
        let rts = RichTextStyle::default();
        assert!(!rts.has_visible_outline());
        assert!(!rts.has_visible_shadow());
    }

    #[test]
    fn test_rich_text_style_with_outline() {
        let rts = RichTextStyle::default().with_outline(TextOutline::default());
        assert!(rts.has_visible_outline());
    }

    #[test]
    fn test_rich_text_style_with_shadow() {
        let rts = RichTextStyle::default().with_shadow(DropShadow::default());
        assert!(rts.has_visible_shadow());
    }

    #[test]
    fn test_rich_text_style_outline_radius() {
        let o = TextOutline::new(0xFF, 3.0);
        let rts = RichTextStyle::default().with_outline(o);
        assert!((rts.outline_radius() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rich_text_style_total_margins_no_effects() {
        let rts = RichTextStyle::default();
        assert_eq!(rts.total_margins(), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_rich_text_style_total_margins_with_both() {
        let outline = TextOutline::new(0xFF, 2.0);
        let shadow = DropShadow::new(0x000000FF, 4.0, 4.0, 0.0, 1.0);
        let rts = RichTextStyle::default()
            .with_outline(outline)
            .with_shadow(shadow);
        let (l, t, r, b) = rts.total_margins();
        // shadow margins: left=0, top=0, right=4, bottom=4; outline adds 2 each side
        assert!((l - 2.0).abs() < 0.01, "left={}", l);
        assert!((t - 2.0).abs() < 0.01, "top={}", t);
        assert!((r - 6.0).abs() < 0.01, "right={}", r);
        assert!((b - 6.0).abs() < 0.01, "bottom={}", b);
    }
}

// ============================================================================
// TextShadow — caller-friendly shim over DropShadow
// ============================================================================

/// A caller-friendly text drop shadow specification.
///
/// Stores offsets, blur, and colour as `[u8; 4]` RGBA bytes.  Use
/// [`to_drop_shadow`](Self::to_drop_shadow) to obtain the underlying
/// [`DropShadow`] used for rendering calculations.
#[derive(Debug, Clone)]
pub struct TextShadow {
    /// Horizontal offset in pixels (positive = right).
    pub offset_x: f32,
    /// Vertical offset in pixels (positive = down).
    pub offset_y: f32,
    /// Gaussian blur radius in pixels.
    pub blur_radius: f32,
    /// RGBA colour as `[R, G, B, A]`.
    pub color: [u8; 4],
}

impl TextShadow {
    /// Create a new `TextShadow`.
    pub fn new(offset_x: f32, offset_y: f32, blur_radius: f32, color: [u8; 4]) -> Self {
        Self {
            offset_x,
            offset_y,
            blur_radius: blur_radius.max(0.0),
            color,
        }
    }

    /// Returns `true` if the shadow contributes any visible pixels.
    pub fn is_visible(&self) -> bool {
        self.color[3] > 0 && (self.offset_x.abs() + self.offset_y.abs() + self.blur_radius) > 0.0
    }

    /// Convert to a [`DropShadow`] for use with rendering calculations.
    pub fn to_drop_shadow(&self) -> DropShadow {
        let packed = (self.color[0] as u32) << 24
            | (self.color[1] as u32) << 16
            | (self.color[2] as u32) << 8
            | self.color[3] as u32;
        DropShadow::new(packed, self.offset_x, self.offset_y, self.blur_radius, 1.0)
    }
}

impl From<TextShadow> for DropShadow {
    fn from(s: TextShadow) -> Self {
        s.to_drop_shadow()
    }
}

impl From<DropShadow> for TextShadow {
    fn from(d: DropShadow) -> Self {
        let bytes = d.rgba_bytes();
        // rgba_bytes() includes opacity baked into alpha, so use those directly.
        Self {
            offset_x: d.offset_x_px,
            offset_y: d.offset_y_px,
            blur_radius: d.blur_radius_px,
            color: bytes,
        }
    }
}

// ============================================================================
// TextOutlineConfig — caller-friendly shim over TextOutline
// ============================================================================

/// A caller-friendly text outline specification.
///
/// Stores the stroke width and colour as `[u8; 4]` RGBA bytes.  Use
/// [`to_text_outline`](Self::to_text_outline) to obtain the underlying
/// [`TextOutline`] used for rendering calculations.
#[derive(Debug, Clone)]
pub struct TextOutlineConfig {
    /// Stroke width in pixels.
    pub width: f32,
    /// RGBA colour as `[R, G, B, A]`.
    pub color: [u8; 4],
}

impl TextOutlineConfig {
    /// Create a new `TextOutlineConfig`.
    pub fn new(width: f32, color: [u8; 4]) -> Self {
        Self {
            width: width.max(0.0),
            color,
        }
    }

    /// Returns `true` if the outline has a positive width (and is thus visible).
    pub fn is_visible(&self) -> bool {
        self.width > 0.0
    }

    /// Convert to a [`TextOutline`] for use with rendering calculations.
    pub fn to_text_outline(&self) -> TextOutline {
        let packed = (self.color[0] as u32) << 24
            | (self.color[1] as u32) << 16
            | (self.color[2] as u32) << 8
            | self.color[3] as u32;
        TextOutline::new(packed, self.width)
    }
}

impl From<TextOutlineConfig> for TextOutline {
    fn from(c: TextOutlineConfig) -> Self {
        c.to_text_outline()
    }
}

// ============================================================================
// FullTextRenderConfig — carries style + optional outline and shadow
// ============================================================================

/// An enhanced text render configuration that carries an optional outline and
/// drop shadow in addition to the base [`TextStyle`].
///
/// Use [`with_shadow`](Self::with_shadow) and [`with_outline`](Self::with_outline)
/// to attach effects, then call [`to_rich_style`](Self::to_rich_style) to
/// produce the underlying [`RichTextStyle`] used by the rendering engine.
#[derive(Debug, Clone)]
pub struct FullTextRenderConfig {
    /// Core text style.
    pub style: TextStyle,
    /// Optional drop shadow.
    pub shadow: Option<TextShadow>,
    /// Optional stroke outline.
    pub outline: Option<TextOutlineConfig>,
}

impl Default for FullTextRenderConfig {
    fn default() -> Self {
        Self {
            style: TextStyle::default(),
            shadow: None,
            outline: None,
        }
    }
}

impl FullTextRenderConfig {
    /// Attach a drop shadow.
    pub fn with_shadow(mut self, shadow: TextShadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Attach a text outline.
    pub fn with_outline(mut self, outline: TextOutlineConfig) -> Self {
        self.outline = Some(outline);
        self
    }

    /// Returns `true` if either shadow or outline is present.
    pub fn has_effects(&self) -> bool {
        self.shadow.is_some() || self.outline.is_some()
    }

    /// Convert to a [`RichTextStyle`] for use with the rendering engine.
    pub fn to_rich_style(&self) -> RichTextStyle {
        let mut rts = RichTextStyle::from_base(self.style.clone());
        if let Some(ref s) = self.shadow {
            rts = rts.with_shadow(s.to_drop_shadow());
        }
        if let Some(ref o) = self.outline {
            rts = rts.with_outline(o.to_text_outline());
        }
        rts
    }
}

// ============================================================================
// Unit tests for TextShadow, TextOutlineConfig, FullTextRenderConfig
// ============================================================================

#[cfg(test)]
mod shadow_outline_tests {
    use super::*;

    #[test]
    fn test_text_shadow_new_stores_fields() {
        let s = TextShadow::new(3.0, -2.0, 5.0, [255, 0, 0, 200]);
        assert!((s.offset_x - 3.0).abs() < f32::EPSILON);
        assert!((s.offset_y - (-2.0)).abs() < f32::EPSILON);
        assert!((s.blur_radius - 5.0).abs() < f32::EPSILON);
        assert_eq!(s.color, [255, 0, 0, 200]);
    }

    #[test]
    fn test_text_shadow_is_visible_with_offset() {
        let s = TextShadow::new(2.0, 2.0, 0.0, [0, 0, 0, 255]);
        assert!(s.is_visible());
    }

    #[test]
    fn test_text_shadow_not_visible_when_alpha_zero() {
        let s = TextShadow::new(2.0, 2.0, 1.0, [0, 0, 0, 0]);
        assert!(!s.is_visible());
    }

    #[test]
    fn test_text_shadow_not_visible_when_all_zero() {
        let s = TextShadow::new(0.0, 0.0, 0.0, [0, 0, 0, 128]);
        assert!(!s.is_visible());
    }

    #[test]
    fn test_text_shadow_to_drop_shadow_offset() {
        let s = TextShadow::new(4.0, 6.0, 2.0, [0, 0, 0, 200]);
        let ds = s.to_drop_shadow();
        assert!((ds.offset_x_px - 4.0).abs() < f32::EPSILON);
        assert!((ds.offset_y_px - 6.0).abs() < f32::EPSILON);
        assert!((ds.blur_radius_px - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_shadow_from_into_drop_shadow() {
        let s = TextShadow::new(1.0, 2.0, 3.0, [128, 64, 32, 255]);
        let ds: DropShadow = s.into();
        assert!(ds.is_visible());
    }

    #[test]
    fn test_text_shadow_from_drop_shadow() {
        let ds = DropShadow::new(0xFF0000FF_u32, 3.0, 4.0, 1.0, 1.0);
        let ts: TextShadow = ds.into();
        assert!((ts.offset_x - 3.0).abs() < f32::EPSILON);
        assert!((ts.offset_y - 4.0).abs() < f32::EPSILON);
        assert!((ts.blur_radius - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_outline_config_new_stores_fields() {
        let o = TextOutlineConfig::new(2.5, [255, 128, 0, 255]);
        assert!((o.width - 2.5).abs() < f32::EPSILON);
        assert_eq!(o.color, [255, 128, 0, 255]);
    }

    #[test]
    fn test_text_outline_config_is_visible_positive_width() {
        let o = TextOutlineConfig::new(1.0, [0, 0, 0, 255]);
        assert!(o.is_visible());
    }

    #[test]
    fn test_text_outline_config_not_visible_zero_width() {
        let o = TextOutlineConfig::new(0.0, [0, 0, 0, 255]);
        assert!(!o.is_visible());
    }

    #[test]
    fn test_text_outline_config_to_text_outline() {
        let o = TextOutlineConfig::new(3.0, [0, 0, 0, 255]);
        let to = o.to_text_outline();
        assert!(to.is_visible());
        assert!((to.width_px - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_outline_config_from_into_text_outline() {
        let o = TextOutlineConfig::new(2.0, [255, 255, 255, 200]);
        let to: TextOutline = o.into();
        assert!(to.is_visible());
    }

    #[test]
    fn test_text_outline_config_rgba_bytes_round_trip() {
        // R=0x10, G=0x20, B=0x30, A=0xFF
        let o = TextOutlineConfig::new(1.0, [0x10, 0x20, 0x30, 0xFF]);
        let to = o.to_text_outline();
        let bytes = to.rgba_bytes();
        assert_eq!(bytes[0], 0x10);
        assert_eq!(bytes[1], 0x20);
        assert_eq!(bytes[2], 0x30);
        assert_eq!(bytes[3], 0xFF);
    }

    #[test]
    fn test_full_text_render_config_default_no_effects() {
        let cfg = FullTextRenderConfig::default();
        assert!(!cfg.has_effects());
        assert!(cfg.shadow.is_none());
        assert!(cfg.outline.is_none());
    }

    #[test]
    fn test_full_text_render_config_with_shadow() {
        let shadow = TextShadow::new(2.0, 2.0, 1.0, [0, 0, 0, 200]);
        let cfg = FullTextRenderConfig::default().with_shadow(shadow);
        assert!(cfg.has_effects());
        assert!(cfg.shadow.is_some());
    }

    #[test]
    fn test_full_text_render_config_with_outline() {
        let outline = TextOutlineConfig::new(2.0, [0, 0, 0, 255]);
        let cfg = FullTextRenderConfig::default().with_outline(outline);
        assert!(cfg.has_effects());
        assert!(cfg.outline.is_some());
    }

    #[test]
    fn test_full_text_render_config_to_rich_style_no_effects() {
        let cfg = FullTextRenderConfig::default();
        let rts = cfg.to_rich_style();
        assert!(!rts.has_visible_outline());
        assert!(!rts.has_visible_shadow());
    }

    #[test]
    fn test_full_text_render_config_to_rich_style_with_outline() {
        let outline = TextOutlineConfig::new(2.0, [0, 0, 0, 255]);
        let cfg = FullTextRenderConfig::default().with_outline(outline);
        let rts = cfg.to_rich_style();
        assert!(rts.has_visible_outline());
    }

    #[test]
    fn test_full_text_render_config_to_rich_style_with_shadow() {
        let shadow = TextShadow::new(3.0, 3.0, 2.0, [0, 0, 0, 200]);
        let cfg = FullTextRenderConfig::default().with_shadow(shadow);
        let rts = cfg.to_rich_style();
        assert!(rts.has_visible_shadow());
    }

    #[test]
    fn test_full_text_render_config_to_rich_style_with_both() {
        let shadow = TextShadow::new(2.0, 2.0, 1.0, [0, 0, 0, 180]);
        let outline = TextOutlineConfig::new(1.5, [255, 255, 255, 255]);
        let cfg = FullTextRenderConfig::default()
            .with_shadow(shadow)
            .with_outline(outline);
        let rts = cfg.to_rich_style();
        assert!(rts.has_visible_outline());
        assert!(rts.has_visible_shadow());
    }

    #[test]
    fn test_text_shadow_blur_clamped_to_zero() {
        let s = TextShadow::new(0.0, 0.0, -5.0, [0, 0, 0, 255]);
        assert!((s.blur_radius - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_outline_config_width_clamped_to_zero() {
        let o = TextOutlineConfig::new(-3.0, [0, 0, 0, 255]);
        assert!((o.width - 0.0).abs() < f32::EPSILON);
    }
}

// ============================================================================
// Tests for StyledTextOutline, TextGlow, GradientFill, FullEffectTextStyle
// ============================================================================

#[cfg(test)]
mod effect_tests {
    use super::*;

    // --- StyledTextOutline ---

    #[test]
    fn test_styled_outline_default_visible() {
        let o = StyledTextOutline::default();
        assert!(o.is_visible());
        assert_eq!(o.join_style, StrokeJoinStyle::Miter);
    }

    #[test]
    fn test_styled_outline_zero_width_not_visible() {
        let o = StyledTextOutline::new(0xFF0000FF, 0.0, StrokeJoinStyle::Round);
        assert!(!o.is_visible());
    }

    #[test]
    fn test_styled_outline_rgba_bytes() {
        let o = StyledTextOutline::new(0xFF8040CC, 2.0, StrokeJoinStyle::Bevel);
        let bytes = o.rgba_bytes();
        assert_eq!(bytes, [0xFF, 0x80, 0x40, 0xCC]);
    }

    #[test]
    fn test_styled_outline_miter_limit() {
        let o = StyledTextOutline::default().with_miter_limit(8.0);
        assert!((o.miter_limit - 8.0).abs() < f32::EPSILON);
        assert!((o.effective_miter_length() - 16.0).abs() < f32::EPSILON); // 2.0 * 8.0
    }

    #[test]
    fn test_styled_outline_to_basic() {
        let o = StyledTextOutline::new(0xAABBCCDD, 3.5, StrokeJoinStyle::Round);
        let basic = o.to_basic_outline();
        assert!((basic.width_px - 3.5).abs() < f32::EPSILON);
        assert_eq!(basic.color, 0xAABBCCDD);
    }

    // --- TextGlow ---

    #[test]
    fn test_glow_default_visible() {
        let g = TextGlow::default();
        assert!(g.is_visible());
    }

    #[test]
    fn test_glow_zero_spread_and_blur_not_visible() {
        let g = TextGlow::new(0xFF0000FF, 0.0, 0.0, GlowDirection::Outer);
        assert!(!g.is_visible());
    }

    #[test]
    fn test_glow_opacity_zero_not_visible() {
        let g = TextGlow::default().with_opacity(0.0);
        assert!(!g.is_visible());
    }

    #[test]
    fn test_glow_rgba_bytes_applies_opacity() {
        let g = TextGlow::new(0x000000FF, 4.0, 4.0, GlowDirection::Outer).with_opacity(0.5);
        let bytes = g.rgba_bytes();
        // base_a=255, opacity=0.5 → a≈127
        assert!(bytes[3] > 100 && bytes[3] < 140, "alpha={}", bytes[3]);
    }

    #[test]
    fn test_glow_inner_no_margin() {
        let g = TextGlow::new(0xFFFFFFFF, 5.0, 3.0, GlowDirection::Inner);
        assert!((g.required_margin() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_glow_outer_margin() {
        let g = TextGlow::new(0xFFFFFFFF, 5.0, 3.0, GlowDirection::Outer);
        assert!((g.required_margin() - 8.0).abs() < f32::EPSILON); // 5+3
    }

    #[test]
    fn test_glow_both_margin() {
        let g = TextGlow::new(0xFFFFFFFF, 4.0, 2.0, GlowDirection::Both);
        assert!((g.required_margin() - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_glow_to_drop_shadow() {
        let g = TextGlow::new(0xFF0000FF, 3.0, 2.0, GlowDirection::Outer).with_opacity(0.9);
        let ds = g.to_drop_shadow();
        assert!((ds.offset_x_px).abs() < f32::EPSILON);
        assert!((ds.offset_y_px).abs() < f32::EPSILON);
        assert!((ds.blur_radius_px - 5.0).abs() < f32::EPSILON); // 3+2
    }

    // --- GradientFill ---

    #[test]
    fn test_gradient_default_is_valid() {
        let g = GradientFill::default();
        assert!(g.is_valid());
        assert_eq!(g.gradient_type, GradientType::Linear);
    }

    #[test]
    fn test_gradient_linear_two_stops() {
        let g = GradientFill::linear(0xFF0000FF, 0x0000FFFF, 90.0);
        assert!(g.is_valid());
        assert_eq!(g.stops.len(), 2);
    }

    #[test]
    fn test_gradient_radial_center() {
        let g = GradientFill::radial(0xFFFFFFFF, 0x000000FF).with_center(0.3, 0.7);
        assert_eq!(g.gradient_type, GradientType::Radial);
        assert!((g.center_x - 0.3).abs() < f32::EPSILON);
        assert!((g.center_y - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gradient_sample_endpoints() {
        let g = GradientFill::linear(0xFF0000FF, 0x0000FFFF, 0.0);
        let start = g.sample(0.0);
        assert_eq!(start[0], 0xFF); // red at start
        let end = g.sample(1.0);
        assert_eq!(end[2], 0xFF); // blue at end
    }

    #[test]
    fn test_gradient_sample_midpoint() {
        let g = GradientFill::linear(0x000000FF, 0xFEFEFEFF, 0.0);
        let mid = g.sample(0.5);
        // Should be roughly halfway: ~127
        assert!(mid[0] > 60 && mid[0] < 140, "r={}", mid[0]);
    }

    #[test]
    fn test_gradient_with_stop() {
        let g = GradientFill::linear(0xFF0000FF, 0x0000FFFF, 0.0).with_stop(0.5, 0x00FF00FF);
        assert_eq!(g.stops.len(), 3);
        // Stops should be sorted
        assert!(g.stops[0].position <= g.stops[1].position);
        assert!(g.stops[1].position <= g.stops[2].position);
    }

    #[test]
    fn test_gradient_compute_t_linear() {
        let g = GradientFill::linear(0xFF0000FF, 0x0000FFFF, 0.0);
        let t = g.compute_t(50.0, 0.0, 100.0, 100.0);
        assert!(t >= 0.0 && t <= 1.0);
    }

    #[test]
    fn test_gradient_compute_t_radial_center() {
        let g = GradientFill::radial(0xFFFFFFFF, 0x000000FF);
        let t = g.compute_t(50.0, 50.0, 100.0, 100.0); // at center
        assert!(t < 0.1, "Center should be near 0, got {}", t);
    }

    #[test]
    fn test_gradient_color_at() {
        let g = GradientFill::linear(0xFF0000FF, 0x0000FFFF, 0.0);
        let color = g.color_at(0.0, 0.0, 100.0, 100.0);
        // Should be valid RGBA
        assert_eq!(color[3], 0xFF);
    }

    // --- FullEffectTextStyle ---

    #[test]
    fn test_full_effect_default_no_effects() {
        let s = FullEffectTextStyle::default();
        assert!(!s.has_effects());
    }

    #[test]
    fn test_full_effect_with_all_effects() {
        let s = FullEffectTextStyle::from_base(TextStyle::default())
            .with_outline(StyledTextOutline::default())
            .with_shadow(DropShadow::default())
            .with_glow(TextGlow::default())
            .with_gradient(GradientFill::default());
        assert!(s.has_effects());
    }

    #[test]
    fn test_full_effect_total_margins_with_glow() {
        let s = FullEffectTextStyle::from_base(TextStyle::default()).with_glow(TextGlow::new(
            0xFFFFFFFF,
            5.0,
            3.0,
            GlowDirection::Outer,
        ));
        let (l, t, r, b) = s.total_margins();
        assert!((l - 8.0).abs() < 0.01);
        assert!((t - 8.0).abs() < 0.01);
        assert!((r - 8.0).abs() < 0.01);
        assert!((b - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_full_effect_to_rich_text_style() {
        let s = FullEffectTextStyle::from_base(TextStyle::default())
            .with_outline(StyledTextOutline::new(
                0xFF0000FF,
                2.0,
                StrokeJoinStyle::Round,
            ))
            .with_shadow(DropShadow::default());
        let rts = s.to_rich_text_style();
        assert!(rts.has_visible_outline());
        assert!(rts.has_visible_shadow());
    }

    #[test]
    fn test_stroke_join_style_default() {
        assert_eq!(StrokeJoinStyle::default(), StrokeJoinStyle::Miter);
    }

    #[test]
    fn test_glow_direction_default() {
        assert_eq!(GlowDirection::default(), GlowDirection::Outer);
    }

    #[test]
    fn test_gradient_type_default() {
        assert_eq!(GradientType::default(), GradientType::Linear);
    }
}
