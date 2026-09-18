//! Subtitle styling definitions.

/// RGBA color value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
    /// Alpha component (0-255, 0=transparent, 255=opaque).
    pub a: u8,
}

impl Color {
    /// Create a new color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// White color.
    #[must_use]
    pub const fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// Black color.
    #[must_use]
    pub const fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    /// Transparent color.
    #[must_use]
    pub const fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Parse from hex string (e.g., "#FFFFFF" or "#FFFFFFFF").
    ///
    /// # Errors
    ///
    /// Returns error if the string is not a valid hex color.
    pub fn from_hex(hex: &str) -> Result<Self, crate::SubtitleError> {
        let hex = hex.trim_start_matches('#');

        let (r, g, b, a) = match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                (r, g, b, 255)
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                let a = u8::from_str_radix(&hex[6..8], 16)
                    .map_err(|_| crate::SubtitleError::InvalidColor(hex.to_string()))?;
                (r, g, b, a)
            }
            _ => return Err(crate::SubtitleError::InvalidColor(hex.to_string())),
        };

        Ok(Self::new(r, g, b, a))
    }

    /// Blend this color with another using alpha compositing.
    #[must_use]
    pub fn blend_over(&self, background: Color) -> Color {
        if self.a == 255 {
            return *self;
        }
        if self.a == 0 {
            return background;
        }

        let alpha = f32::from(self.a) / 255.0;
        let inv_alpha = 1.0 - alpha;

        Color::new(
            (f32::from(self.r) * alpha + f32::from(background.r) * inv_alpha) as u8,
            (f32::from(self.g) * alpha + f32::from(background.g) * inv_alpha) as u8,
            (f32::from(self.b) * alpha + f32::from(background.b) * inv_alpha) as u8,
            255,
        )
    }

    /// Create a color with modified alpha.
    #[must_use]
    pub const fn with_alpha(&self, a: u8) -> Self {
        Self::new(self.r, self.g, self.b, a)
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::white()
    }
}

/// Text alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Alignment {
    /// Left aligned.
    Left,
    /// Center aligned.
    #[default]
    Center,
    /// Right aligned.
    Right,
}

/// Vertical alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerticalAlignment {
    /// Top aligned.
    Top,
    /// Middle aligned.
    Middle,
    /// Bottom aligned.
    #[default]
    Bottom,
}

/// Subtitle position on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    /// Horizontal position (0.0 = left, 1.0 = right).
    pub x: f32,
    /// Vertical position (0.0 = top, 1.0 = bottom).
    pub y: f32,
    /// Horizontal alignment.
    pub alignment: Alignment,
    /// Vertical alignment.
    pub vertical_alignment: VerticalAlignment,
}

impl Position {
    /// Create a new position.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            alignment: Alignment::Center,
            vertical_alignment: VerticalAlignment::Bottom,
        }
    }

    /// Bottom center position (default).
    #[must_use]
    pub const fn bottom_center() -> Self {
        Self::new(0.5, 0.9)
    }

    /// Top center position.
    #[must_use]
    pub const fn top_center() -> Self {
        Self::new(0.5, 0.1)
    }

    /// Middle center position.
    #[must_use]
    pub const fn middle_center() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Set alignment.
    #[must_use]
    pub const fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set vertical alignment.
    #[must_use]
    pub const fn with_vertical_alignment(mut self, vertical_alignment: VerticalAlignment) -> Self {
        self.vertical_alignment = vertical_alignment;
        self
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::bottom_center()
    }
}

/// Outline style for text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineStyle {
    /// Outline color.
    pub color: Color,
    /// Outline width in pixels.
    pub width: f32,
}

impl OutlineStyle {
    /// Create a new outline style.
    #[must_use]
    pub const fn new(color: Color, width: f32) -> Self {
        Self { color, width }
    }

    /// Default black outline.
    #[must_use]
    pub const fn black(width: f32) -> Self {
        Self::new(Color::black(), width)
    }
}

impl Default for OutlineStyle {
    fn default() -> Self {
        Self::black(2.0)
    }
}

/// Shadow style for text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowStyle {
    /// Shadow color.
    pub color: Color,
    /// Shadow offset X in pixels.
    pub offset_x: f32,
    /// Shadow offset Y in pixels.
    pub offset_y: f32,
    /// Shadow blur radius in pixels.
    pub blur: f32,
}

impl ShadowStyle {
    /// Create a new shadow style.
    #[must_use]
    pub const fn new(color: Color, offset_x: f32, offset_y: f32, blur: f32) -> Self {
        Self {
            color,
            offset_x,
            offset_y,
            blur,
        }
    }

    /// Default drop shadow.
    #[must_use]
    pub const fn default_shadow() -> Self {
        Self::new(Color::new(0, 0, 0, 128), 2.0, 2.0, 0.0)
    }
}

impl Default for ShadowStyle {
    fn default() -> Self {
        Self::default_shadow()
    }
}

/// Font weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FontWeight {
    /// Thin (100).
    Thin,
    /// Extra light (200).
    ExtraLight,
    /// Light (300).
    Light,
    /// Normal (400).
    #[default]
    Normal,
    /// Medium (500).
    Medium,
    /// Semi bold (600).
    SemiBold,
    /// Bold (700).
    Bold,
    /// Extra bold (800).
    ExtraBold,
    /// Black (900).
    Black,
}

/// Font style.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FontStyle {
    /// Normal/Roman style.
    #[default]
    Normal,
    /// Italic style.
    Italic,
    /// Oblique style.
    Oblique,
}

/// Animation effect for subtitles.
#[derive(Clone, Debug, PartialEq)]
pub enum Animation {
    /// Fade in over duration (milliseconds).
    FadeIn(i64),
    /// Fade out over duration (milliseconds).
    FadeOut(i64),
    /// Move from position to position over duration.
    Move {
        /// Start position.
        from: Position,
        /// End position.
        to: Position,
        /// Duration in milliseconds.
        duration: i64,
    },
    /// Karaoke effect - highlight words as they're sung.
    Karaoke {
        /// Highlight color.
        color: Color,
        /// Timing for each syllable/word in milliseconds.
        timings: Vec<i64>,
    },
    /// Scale animation.
    Scale {
        /// Start scale (1.0 = normal).
        from: f32,
        /// End scale.
        to: f32,
        /// Duration in milliseconds.
        duration: i64,
    },
    /// Rotation animation (degrees).
    Rotate {
        /// Start angle.
        from: f32,
        /// End angle.
        to: f32,
        /// Duration in milliseconds.
        duration: i64,
    },
}

/// Complete subtitle style configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct SubtitleStyle {
    /// Font size in pixels.
    pub font_size: f32,
    /// Font weight.
    pub font_weight: FontWeight,
    /// Font style.
    pub font_style: FontStyle,
    /// Primary text color.
    pub primary_color: Color,
    /// Secondary color (for karaoke, gradients).
    pub secondary_color: Color,
    /// Outline style.
    pub outline: Option<OutlineStyle>,
    /// Shadow style.
    pub shadow: Option<ShadowStyle>,
    /// Text alignment.
    pub alignment: Alignment,
    /// Vertical alignment.
    pub vertical_alignment: VerticalAlignment,
    /// Default position.
    pub position: Position,
    /// Margin from edges (in pixels).
    pub margin_left: u32,
    /// Right margin.
    pub margin_right: u32,
    /// Top margin.
    pub margin_top: u32,
    /// Bottom margin.
    pub margin_bottom: u32,
    /// Line spacing multiplier (1.0 = normal).
    pub line_spacing: f32,
    /// Enable word wrapping.
    pub word_wrap: bool,
    /// Maximum width for wrapping (0 = use frame width with margins).
    pub max_width: u32,
    /// Background box color (if any).
    pub background_color: Option<Color>,
    /// Background box padding.
    pub background_padding: f32,
}

impl SubtitleStyle {
    /// Create a new subtitle style with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set font size.
    #[must_use]
    pub const fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set primary color.
    #[must_use]
    pub fn with_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.primary_color = Color::new(r, g, b, a);
        self
    }

    /// Set outline.
    #[must_use]
    pub fn with_outline(mut self, outline: OutlineStyle) -> Self {
        self.outline = Some(outline);
        self
    }

    /// Set shadow.
    #[must_use]
    pub fn with_shadow(mut self, shadow: ShadowStyle) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Set alignment.
    #[must_use]
    pub const fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set position.
    #[must_use]
    pub const fn with_position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    /// Set margins.
    #[must_use]
    pub const fn with_margins(mut self, left: u32, right: u32, top: u32, bottom: u32) -> Self {
        self.margin_left = left;
        self.margin_right = right;
        self.margin_top = top;
        self.margin_bottom = bottom;
        self
    }

    /// Enable background box.
    #[must_use]
    pub fn with_background(mut self, color: Color, padding: f32) -> Self {
        self.background_color = Some(color);
        self.background_padding = padding;
        self
    }
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self {
            font_size: 48.0,
            font_weight: FontWeight::Normal,
            font_style: FontStyle::Normal,
            primary_color: Color::white(),
            secondary_color: Color::rgb(255, 255, 0), // Yellow for karaoke
            outline: Some(OutlineStyle::default()),
            shadow: Some(ShadowStyle::default()),
            alignment: Alignment::Center,
            vertical_alignment: VerticalAlignment::Bottom,
            position: Position::default(),
            margin_left: 40,
            margin_right: 40,
            margin_top: 40,
            margin_bottom: 40,
            line_spacing: 1.2,
            word_wrap: true,
            max_width: 0,
            background_color: None,
            background_padding: 4.0,
        }
    }
}
