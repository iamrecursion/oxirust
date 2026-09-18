//! Subtitle and caption burn-in transcoding.
//!
//! Provides position calculation, font rasterization mocking, and
//! subtitle timing/styling for burn-in operations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Subtitle position anchors on the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleAnchor {
    /// Bottom center (most common for subtitles).
    BottomCenter,
    /// Top center (for on-screen graphics / supers).
    TopCenter,
    /// Bottom left.
    BottomLeft,
    /// Bottom right.
    BottomRight,
    /// Custom pixel position.
    Custom(u32, u32),
}

/// Font weight for subtitle rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontWeight {
    /// Normal weight.
    Normal,
    /// Bold weight.
    Bold,
}

/// A font style specification for subtitle rendering.
#[derive(Debug, Clone)]
pub struct SubtitleFont {
    /// Font family name (e.g., "Arial", "Helvetica").
    pub family: String,
    /// Font size in pixels at the output resolution.
    pub size_px: u32,
    /// Font weight.
    pub weight: FontWeight,
    /// Whether italic is enabled.
    pub italic: bool,
    /// Text color as (R, G, B, A).
    pub color: (u8, u8, u8, u8),
    /// Outline/shadow color as (R, G, B, A).
    pub outline_color: (u8, u8, u8, u8),
    /// Outline thickness in pixels.
    pub outline_px: u32,
}

impl SubtitleFont {
    /// Creates a new subtitle font with default settings.
    #[must_use]
    pub fn new(family: impl Into<String>, size_px: u32) -> Self {
        Self {
            family: family.into(),
            size_px,
            weight: FontWeight::Normal,
            italic: false,
            color: (255, 255, 255, 255),
            outline_color: (0, 0, 0, 200),
            outline_px: 2,
        }
    }

    /// Sets the font weight.
    #[must_use]
    pub fn with_weight(mut self, weight: FontWeight) -> Self {
        self.weight = weight;
        self
    }

    /// Enables italic rendering.
    #[must_use]
    pub fn italic(mut self) -> Self {
        self.italic = true;
        self
    }

    /// Sets the text color.
    #[must_use]
    pub fn with_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.color = (r, g, b, a);
        self
    }

    /// Sets the outline color and thickness.
    #[must_use]
    pub fn with_outline(mut self, r: u8, g: u8, b: u8, a: u8, thickness_px: u32) -> Self {
        self.outline_color = (r, g, b, a);
        self.outline_px = thickness_px;
        self
    }
}

impl Default for SubtitleFont {
    fn default() -> Self {
        Self::new("Arial", 48)
    }
}

/// A single subtitle entry with text, timing, and style.
#[derive(Debug, Clone)]
pub struct SubtitleEntry {
    /// Subtitle text (may contain newlines for multi-line).
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Position anchor.
    pub anchor: SubtitleAnchor,
    /// Font style override (uses config default if None).
    pub font: Option<SubtitleFont>,
    /// Margin from the frame edge in pixels.
    pub margin_px: u32,
}

impl SubtitleEntry {
    /// Creates a new subtitle entry.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
            anchor: SubtitleAnchor::BottomCenter,
            font: None,
            margin_px: 20,
        }
    }

    /// Sets the position anchor.
    #[must_use]
    pub fn with_anchor(mut self, anchor: SubtitleAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Sets a font override for this entry.
    #[must_use]
    pub fn with_font(mut self, font: SubtitleFont) -> Self {
        self.font = Some(font);
        self
    }

    /// Returns the duration of this subtitle entry in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns true if this subtitle is active at the given timestamp.
    #[must_use]
    pub fn is_active_at(&self, timestamp_ms: u64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }
}

/// Configuration for the subtitle burn-in operation.
#[derive(Debug, Clone)]
pub struct BurnSubsConfig {
    /// All subtitle entries to burn in.
    pub entries: Vec<SubtitleEntry>,
    /// Default font for all entries without a font override.
    pub default_font: SubtitleFont,
    /// Video frame width in pixels.
    pub frame_width: u32,
    /// Video frame height in pixels.
    pub frame_height: u32,
    /// Whether to enable soft-shadow rendering.
    pub shadow_enabled: bool,
    /// Whether to enable anti-aliased text rendering.
    pub antialias: bool,
}

impl BurnSubsConfig {
    /// Creates a new burn-subs configuration.
    #[must_use]
    pub fn new(frame_width: u32, frame_height: u32) -> Self {
        Self {
            entries: Vec::new(),
            default_font: SubtitleFont::default(),
            frame_width,
            frame_height,
            shadow_enabled: true,
            antialias: true,
        }
    }

    /// Adds a subtitle entry.
    #[must_use]
    pub fn add_entry(mut self, entry: SubtitleEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Sets the default font.
    #[must_use]
    pub fn with_default_font(mut self, font: SubtitleFont) -> Self {
        self.default_font = font;
        self
    }

    /// Returns all entries active at the given timestamp.
    #[must_use]
    pub fn active_at(&self, timestamp_ms: u64) -> Vec<&SubtitleEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_active_at(timestamp_ms))
            .collect()
    }

    /// Computes the pixel position for an entry based on its anchor.
    #[must_use]
    pub fn compute_position(
        &self,
        entry: &SubtitleEntry,
        text_width: u32,
        text_height: u32,
    ) -> (u32, u32) {
        let m = entry.margin_px;
        let fw = self.frame_width;
        let fh = self.frame_height;
        match entry.anchor {
            SubtitleAnchor::BottomCenter => {
                let x = (fw.saturating_sub(text_width)) / 2;
                let y = fh.saturating_sub(text_height).saturating_sub(m);
                (x, y)
            }
            SubtitleAnchor::TopCenter => {
                let x = (fw.saturating_sub(text_width)) / 2;
                (x, m)
            }
            SubtitleAnchor::BottomLeft => (m, fh.saturating_sub(text_height).saturating_sub(m)),
            SubtitleAnchor::BottomRight => {
                let x = fw.saturating_sub(text_width).saturating_sub(m);
                let y = fh.saturating_sub(text_height).saturating_sub(m);
                (x, y)
            }
            SubtitleAnchor::Custom(cx, cy) => (cx, cy),
        }
    }

    /// Mock font rasterization: returns estimated text dimensions.
    ///
    /// In a real implementation this would call a font rendering library.
    #[must_use]
    pub fn estimate_text_size(&self, text: &str, font: &SubtitleFont) -> (u32, u32) {
        let char_width = font.size_px * 6 / 10;
        let line_height = font.size_px * 120 / 100;
        let max_line_len = text.lines().map(|l| l.chars().count()).max().unwrap_or(0);
        let line_count = text.lines().count().max(1);
        (
            char_width * max_line_len as u32,
            line_height * line_count as u32,
        )
    }

    /// Validates all subtitle entries.
    ///
    /// Returns a list of validation errors (empty if valid).
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.start_ms >= entry.end_ms {
                errors.push(format!(
                    "Entry {i}: start_ms ({}) >= end_ms ({})",
                    entry.start_ms, entry.end_ms
                ));
            }
            if entry.text.is_empty() {
                errors.push(format!("Entry {i}: text is empty"));
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_entry_duration() {
        let entry = SubtitleEntry::new("Hello", 1000, 4000);
        assert_eq!(entry.duration_ms(), 3000);
    }

    #[test]
    fn test_subtitle_entry_is_active() {
        let entry = SubtitleEntry::new("Hello", 1000, 4000);
        assert!(!entry.is_active_at(999));
        assert!(entry.is_active_at(1000));
        assert!(entry.is_active_at(3999));
        assert!(!entry.is_active_at(4000));
    }

    #[test]
    fn test_active_at_returns_correct_entries() {
        let config = BurnSubsConfig::new(1920, 1080)
            .add_entry(SubtitleEntry::new("A", 0, 2000))
            .add_entry(SubtitleEntry::new("B", 1500, 4000))
            .add_entry(SubtitleEntry::new("C", 5000, 7000));
        let active = config.active_at(1800);
        assert_eq!(active.len(), 2);
        let active_late = config.active_at(6000);
        assert_eq!(active_late.len(), 1);
        assert_eq!(active_late[0].text, "C");
    }

    #[test]
    fn test_position_bottom_center() {
        let config = BurnSubsConfig::new(1920, 1080);
        let entry = SubtitleEntry::new("Hello", 0, 1000);
        let (x, y) = config.compute_position(&entry, 400, 60);
        assert_eq!(x, (1920 - 400) / 2);
        assert_eq!(y, 1080 - 60 - 20);
    }

    #[test]
    fn test_position_top_center() {
        let config = BurnSubsConfig::new(1920, 1080);
        let entry = SubtitleEntry::new("Super", 0, 1000).with_anchor(SubtitleAnchor::TopCenter);
        let (_x, y) = config.compute_position(&entry, 400, 60);
        assert_eq!(y, 20);
    }

    #[test]
    fn test_position_custom() {
        let config = BurnSubsConfig::new(1920, 1080);
        let entry =
            SubtitleEntry::new("Custom", 0, 1000).with_anchor(SubtitleAnchor::Custom(100, 200));
        let (x, y) = config.compute_position(&entry, 400, 60);
        assert_eq!(x, 100);
        assert_eq!(y, 200);
    }

    #[test]
    fn test_estimate_text_size() {
        let config = BurnSubsConfig::new(1920, 1080);
        let font = SubtitleFont::new("Arial", 48);
        let (w, h) = config.estimate_text_size("Hello", &font);
        // char_width = 48 * 6 / 10 = 28 (integer truncation); 5 chars * 28 = 140
        // line_height = 48 * 120 / 100 = 57; 1 line * 57 = 57
        let char_width = 48_u32 * 6 / 10;
        let line_height = 48_u32 * 120 / 100;
        assert_eq!(w, 5 * char_width);
        assert_eq!(h, line_height);
    }

    #[test]
    fn test_estimate_text_size_multiline() {
        let config = BurnSubsConfig::new(1920, 1080);
        let font = SubtitleFont::new("Arial", 48);
        let (_, h) = config.estimate_text_size("Line1\nLine2", &font);
        // line_height = 48 * 120 / 100 = 57 (integer truncation); 2 lines * 57 = 114
        let line_height = 48_u32 * 120 / 100;
        assert_eq!(h, 2 * line_height);
    }

    #[test]
    fn test_validate_no_errors() {
        let config =
            BurnSubsConfig::new(1920, 1080).add_entry(SubtitleEntry::new("Hello", 0, 1000));
        let errors = config.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_invalid_timing() {
        let mut config = BurnSubsConfig::new(1920, 1080);
        config.entries.push(SubtitleEntry::new("Bad", 5000, 1000));
        let errors = config.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("start_ms"));
    }

    #[test]
    fn test_validate_empty_text() {
        let mut config = BurnSubsConfig::new(1920, 1080);
        config.entries.push(SubtitleEntry::new("", 0, 1000));
        let errors = config.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("empty"));
    }

    #[test]
    fn test_font_defaults() {
        let font = SubtitleFont::default();
        assert_eq!(font.family, "Arial");
        assert_eq!(font.size_px, 48);
        assert_eq!(font.color, (255, 255, 255, 255));
    }

    #[test]
    fn test_font_with_outline() {
        let font = SubtitleFont::new("Helvetica", 40).with_outline(255, 0, 0, 255, 4);
        assert_eq!(font.outline_color, (255, 0, 0, 255));
        assert_eq!(font.outline_px, 4);
    }

    #[test]
    fn test_font_italic_bold() {
        let font = SubtitleFont::new("Arial", 48)
            .with_weight(FontWeight::Bold)
            .italic();
        assert_eq!(font.weight, FontWeight::Bold);
        assert!(font.italic);
    }
}
