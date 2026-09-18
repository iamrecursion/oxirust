//! Font loading and glyph management with fallback font chain support.
//!
//! The `FontChain` type allows specifying a prioritized list of fonts so that
//! glyphs missing in the primary font (e.g. CJK, Arabic, Devanagari) are
//! automatically sourced from the next font in the chain.

use crate::{SubtitleError, SubtitleResult};
use fontdue::layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle};
use fontdue::{Font as FontdueFont, FontSettings};
use std::collections::HashMap;

/// Unicode block ranges used to classify glyphs for fallback routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnicodeScript {
    /// Standard Latin / ASCII range.
    Latin,
    /// CJK Unified Ideographs (U+4E00–U+9FFF and extensions).
    Cjk,
    /// Arabic script (U+0600–U+06FF).
    Arabic,
    /// Devanagari script (U+0900–U+097F).
    Devanagari,
    /// Other / unclassified scripts.
    Other,
}

/// Classify a codepoint into a broad Unicode script category.
#[must_use]
pub fn script_of(c: char) -> UnicodeScript {
    let cp = c as u32;
    match cp {
        // Basic Latin + Latin Extended
        0x0000..=0x024F => UnicodeScript::Latin,
        // Arabic
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
            UnicodeScript::Arabic
        }
        // Devanagari
        0x0900..=0x097F => UnicodeScript::Devanagari,
        // CJK Unified Ideographs (main block + extensions A/B/C/D/E/F)
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2CEAF
        | 0x2CEB0..=0x2EBEF
        | 0xF900..=0xFAFF => UnicodeScript::Cjk,
        // Katakana + Hiragana
        0x3040..=0x30FF => UnicodeScript::Cjk,
        // Hangul
        0xAC00..=0xD7AF | 0x1100..=0x11FF => UnicodeScript::Cjk,
        _ => UnicodeScript::Other,
    }
}

/// A loaded font for subtitle rendering.
pub struct Font {
    inner: FontdueFont,
}

impl Font {
    /// Load a font from bytes (TTF or OTF).
    ///
    /// # Errors
    ///
    /// Returns error if the font data is invalid.
    pub fn from_bytes(data: Vec<u8>) -> SubtitleResult<Self> {
        let inner = FontdueFont::from_bytes(data, FontSettings::default())
            .map_err(|e| SubtitleError::FontError(format!("Failed to load font: {e}")))?;

        Ok(Self { inner })
    }

    /// Load a font from a file.
    ///
    /// # Errors
    ///
    /// Returns error if the file cannot be read or the font is invalid.
    pub fn from_file(path: &str) -> SubtitleResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| SubtitleError::FontError(format!("Failed to read font file: {e}")))?;
        Self::from_bytes(data)
    }

    /// Get the font's internal representation.
    #[must_use]
    pub(crate) fn inner(&self) -> &FontdueFont {
        &self.inner
    }

    /// Get font metrics for a given size.
    #[must_use]
    pub fn metrics(&self, size: f32) -> FontMetrics {
        let metrics = self
            .inner
            .horizontal_line_metrics(size)
            .unwrap_or(fontdue::LineMetrics {
                ascent: 0.0,
                descent: 0.0,
                line_gap: 0.0,
                new_line_size: 0.0,
            });

        FontMetrics {
            ascent: metrics.ascent,
            descent: metrics.descent,
            line_gap: metrics.line_gap,
            new_line_size: metrics.new_line_size,
        }
    }

    /// Measure text width at given size.
    #[must_use]
    pub fn measure_text(&self, text: &str, size: f32) -> f32 {
        let mut width = 0.0;
        for c in text.chars() {
            let metrics = self.inner.metrics(c, size);
            width += metrics.advance_width;
        }
        width
    }
}

/// Font metrics for a specific size.
#[derive(Clone, Copy, Debug)]
pub struct FontMetrics {
    /// Ascent (pixels above baseline).
    pub ascent: f32,
    /// Descent (pixels below baseline).
    pub descent: f32,
    /// Gap between lines.
    pub line_gap: f32,
    /// Recommended line height.
    pub new_line_size: f32,
}

/// Cached glyph bitmap.
#[derive(Clone, Debug)]
pub struct CachedGlyph {
    /// Rasterized glyph bitmap (grayscale).
    pub bitmap: Vec<u8>,
    /// Bitmap width in pixels.
    pub width: usize,
    /// Bitmap height in pixels.
    pub height: usize,
    /// Horizontal offset from origin.
    pub offset_x: f32,
    /// Vertical offset from origin.
    pub offset_y: f32,
    /// Advance width to next glyph.
    pub advance_width: f32,
}

/// Glyph cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GlyphKey {
    /// Character code point.
    codepoint: u32,
    /// Font size (scaled to avoid floating point key).
    size_scaled: u32,
}

impl GlyphKey {
    fn new(codepoint: char, size: f32) -> Self {
        Self {
            codepoint: codepoint as u32,
            size_scaled: (size * 100.0) as u32,
        }
    }
}

/// Cache for rasterized glyphs to avoid re-rendering.
pub struct GlyphCache {
    cache: HashMap<GlyphKey, CachedGlyph>,
    font: Font,
}

impl GlyphCache {
    /// Create a new glyph cache for the given font.
    #[must_use]
    pub fn new(font: Font) -> Self {
        Self {
            cache: HashMap::new(),
            font,
        }
    }

    /// Get or rasterize a glyph.
    pub fn get_glyph(&mut self, c: char, size: f32) -> &CachedGlyph {
        let key = GlyphKey::new(c, size);

        self.cache.entry(key).or_insert_with(|| {
            let (metrics, bitmap) = self.font.inner.rasterize(c, size);

            CachedGlyph {
                bitmap,
                width: metrics.width,
                height: metrics.height,
                offset_x: metrics.xmin as f32,
                offset_y: metrics.ymin as f32,
                advance_width: metrics.advance_width,
            }
        })
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size (number of cached glyphs).
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the underlying font.
    #[must_use]
    pub fn font(&self) -> &Font {
        &self.font
    }
}

// ============================================================================
// Fallback font chain
// ============================================================================

/// A font with its associated script-coverage hints.
pub struct ChainedFont {
    /// The loaded font.
    pub font: Font,
    /// Preferred scripts this font covers well (empty = cover-all fallback).
    pub preferred_scripts: Vec<UnicodeScript>,
}

impl ChainedFont {
    /// Create a chain entry that covers all scripts.
    #[must_use]
    pub fn universal(font: Font) -> Self {
        Self {
            font,
            preferred_scripts: Vec::new(),
        }
    }

    /// Create a chain entry specialised for specific scripts.
    #[must_use]
    pub fn for_scripts(font: Font, scripts: Vec<UnicodeScript>) -> Self {
        Self {
            font,
            preferred_scripts: scripts,
        }
    }

    /// Return `true` if this font entry claims to cover `script`.
    /// A font with no preferred scripts is considered a universal fallback.
    #[must_use]
    pub fn covers(&self, script: UnicodeScript) -> bool {
        self.preferred_scripts.is_empty() || self.preferred_scripts.contains(&script)
    }
}

/// A prioritized chain of fonts used to resolve glyphs for rendering.
///
/// When a glyph is requested, the chain tries each font in order and returns
/// the first that contains a non-empty rasterization for the character.
/// This enables transparent CJK, Arabic, and Devanagari fallback.
///
/// # Example
///
/// ```rust,ignore
/// use oximedia_subtitle::font::{Font, FontChain, UnicodeScript, ChainedFont};
///
/// let latin_data = std::fs::read("latin.ttf").unwrap();
/// let cjk_data   = std::fs::read("noto-cjk.ttf").unwrap();
///
/// let mut chain = FontChain::new(Font::from_bytes(latin_data).unwrap());
/// chain.add(ChainedFont::for_scripts(
///     Font::from_bytes(cjk_data).unwrap(),
///     vec![UnicodeScript::Cjk],
/// ));
///
/// let (bitmap, width, height) = chain.rasterize('あ', 48.0);
/// ```
pub struct FontChain {
    /// Primary font (always present).
    primary: ChainedFont,
    /// Ordered list of fallback fonts tried after the primary.
    fallbacks: Vec<ChainedFont>,
    /// Per-character rasterization cache.
    cache: HashMap<(u32, u32), CachedGlyph>,
}

impl FontChain {
    /// Create a chain with a single primary font.
    #[must_use]
    pub fn new(primary: Font) -> Self {
        Self {
            primary: ChainedFont::universal(primary),
            fallbacks: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Add a fallback font to the end of the chain.
    pub fn add(&mut self, font: ChainedFont) {
        self.fallbacks.push(font);
    }

    /// Rasterize a character using the first font in the chain that covers it.
    ///
    /// Returns a `CachedGlyph` from the best available font for this character.
    pub fn rasterize(&mut self, c: char, size: f32) -> &CachedGlyph {
        let key = (c as u32, (size * 100.0) as u32);
        if !self.cache.contains_key(&key) {
            let script = script_of(c);
            let glyph = self.rasterize_inner(c, size, script);
            self.cache.insert(key, glyph);
        }
        // SAFETY: we either confirmed the key exists or just inserted it
        // Using index access via entry API would require different borrow patterns,
        // so we use get() which is guaranteed to succeed here.
        match self.cache.get(&key) {
            Some(glyph) => glyph,
            None => unreachable!("cache key was just confirmed or inserted"),
        }
    }

    /// Internal: find the best font and rasterize.
    fn rasterize_inner(&self, c: char, size: f32, script: UnicodeScript) -> CachedGlyph {
        // Try primary first if it covers this script
        if self.primary.covers(script) {
            let (metrics, bitmap) = self.primary.font.inner.rasterize(c, size);
            if !bitmap.is_empty() || metrics.width > 0 {
                return make_cached_glyph(metrics, bitmap);
            }
        }

        // Try script-matching fallbacks first (preference by specificity)
        for fb in &self.fallbacks {
            if fb.covers(script) && !fb.preferred_scripts.is_empty() {
                let (metrics, bitmap) = fb.font.inner.rasterize(c, size);
                if !bitmap.is_empty() || metrics.width > 0 {
                    return make_cached_glyph(metrics, bitmap);
                }
            }
        }

        // Try universal fallbacks (preferred_scripts empty = covers all)
        for fb in &self.fallbacks {
            if fb.preferred_scripts.is_empty() {
                let (metrics, bitmap) = fb.font.inner.rasterize(c, size);
                if !bitmap.is_empty() || metrics.width > 0 {
                    return make_cached_glyph(metrics, bitmap);
                }
            }
        }

        // Last resort: use primary font even if glyph is missing (shows .notdef)
        let (metrics, bitmap) = self.primary.font.inner.rasterize(c, size);
        make_cached_glyph(metrics, bitmap)
    }

    /// Measure the advance width of a string using the best font for each char.
    #[must_use]
    pub fn measure_text(&self, text: &str, size: f32) -> f32 {
        let mut total = 0.0_f32;
        for c in text.chars() {
            let script = script_of(c);
            let width = self.advance_width_for(c, size, script);
            total += width;
        }
        total
    }

    /// Get advance width for a single character from the best font.
    fn advance_width_for(&self, c: char, size: f32, script: UnicodeScript) -> f32 {
        if self.primary.covers(script) {
            let m = self.primary.font.inner.metrics(c, size);
            if m.advance_width > 0.0 {
                return m.advance_width;
            }
        }
        for fb in &self.fallbacks {
            if fb.covers(script) {
                let m = fb.font.inner.metrics(c, size);
                if m.advance_width > 0.0 {
                    return m.advance_width;
                }
            }
        }
        self.primary.font.inner.metrics(c, size).advance_width
    }

    /// Clear the rasterization cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of entries in the cache.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

/// Build a `CachedGlyph` from fontdue rasterization output.
fn make_cached_glyph(metrics: fontdue::Metrics, bitmap: Vec<u8>) -> CachedGlyph {
    CachedGlyph {
        bitmap,
        width: metrics.width,
        height: metrics.height,
        offset_x: metrics.xmin as f32,
        offset_y: metrics.ymin as f32,
        advance_width: metrics.advance_width,
    }
}

// ============================================================================
// Tests for font chain
// ============================================================================

#[cfg(test)]
mod font_chain_tests {
    use super::*;

    #[test]
    fn test_script_of_latin() {
        assert_eq!(script_of('A'), UnicodeScript::Latin);
        assert_eq!(script_of('z'), UnicodeScript::Latin);
        assert_eq!(script_of('0'), UnicodeScript::Latin);
    }

    #[test]
    fn test_script_of_cjk_main_block() {
        // U+4E2D = 中
        assert_eq!(script_of('\u{4E2D}'), UnicodeScript::Cjk);
    }

    #[test]
    fn test_script_of_cjk_hiragana() {
        // Hiragana あ U+3042
        assert_eq!(script_of('\u{3042}'), UnicodeScript::Cjk);
    }

    #[test]
    fn test_script_of_cjk_hangul() {
        // Hangul U+AC00
        assert_eq!(script_of('\u{AC00}'), UnicodeScript::Cjk);
    }

    #[test]
    fn test_script_of_arabic() {
        // U+0627 = ا (alef)
        assert_eq!(script_of('\u{0627}'), UnicodeScript::Arabic);
    }

    #[test]
    fn test_script_of_devanagari() {
        // U+0905 = अ
        assert_eq!(script_of('\u{0905}'), UnicodeScript::Devanagari);
    }

    #[test]
    fn test_script_of_other() {
        // U+2603 = snowman - falls in 'Other'
        assert_eq!(script_of('\u{2603}'), UnicodeScript::Other);
    }

    #[test]
    fn test_chained_font_universal_covers_all() {
        // A ChainedFont with no preferred_scripts is a universal fallback
        let covers_struct = ChainedFontCoversTester {
            preferred_scripts: vec![],
        };
        assert!(covers_struct.covers(UnicodeScript::Cjk));
        assert!(covers_struct.covers(UnicodeScript::Arabic));
        assert!(covers_struct.covers(UnicodeScript::Latin));
        assert!(covers_struct.covers(UnicodeScript::Other));
    }

    #[test]
    fn test_chained_font_specific_covers() {
        let covers_struct = ChainedFontCoversTester {
            preferred_scripts: vec![UnicodeScript::Cjk],
        };
        assert!(covers_struct.covers(UnicodeScript::Cjk));
        assert!(!covers_struct.covers(UnicodeScript::Arabic));
        assert!(!covers_struct.covers(UnicodeScript::Latin));
    }

    #[test]
    fn test_script_of_cjk_extension() {
        // CJK Extension A: U+3400
        assert_eq!(script_of('\u{3400}'), UnicodeScript::Cjk);
    }

    #[test]
    fn test_script_of_katakana() {
        // Katakana ア U+30A2
        assert_eq!(script_of('\u{30A2}'), UnicodeScript::Cjk);
    }

    #[test]
    fn test_script_of_arabic_presentation() {
        // Arabic Presentation Forms-A: U+FB50
        assert_eq!(script_of('\u{FB50}'), UnicodeScript::Arabic);
    }

    #[test]
    fn test_script_of_arabic_extended() {
        // Arabic Extended-A: U+08A0 is actually Other, FB50 is Arabic Presentation
        assert_eq!(script_of('\u{FE70}'), UnicodeScript::Arabic);
    }

    #[test]
    fn test_script_of_latin_extended() {
        // Latin Extended-B: U+0100
        assert_eq!(script_of('\u{0100}'), UnicodeScript::Latin);
    }

    // Test helper - mirrors covers() logic without needing a real Font
    struct ChainedFontCoversTester {
        preferred_scripts: Vec<UnicodeScript>,
    }
    impl ChainedFontCoversTester {
        fn covers(&self, script: UnicodeScript) -> bool {
            self.preferred_scripts.is_empty() || self.preferred_scripts.contains(&script)
        }
    }
}

// ============================================================================

/// Simple layout engine using fontdue's layout.
pub struct SimpleLayoutEngine {
    layout: Layout,
}

impl SimpleLayoutEngine {
    /// Create a new layout engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layout: Layout::new(CoordinateSystem::PositiveYDown),
        }
    }

    /// Layout text and return glyph positions.
    pub fn layout_text(
        &mut self,
        font: &Font,
        text: &str,
        size: f32,
        max_width: Option<f32>,
    ) -> Vec<GlyphPosition> {
        self.layout.reset(&LayoutSettings {
            x: 0.0,
            y: 0.0,
            max_width,
            max_height: None,
            horizontal_align: fontdue::layout::HorizontalAlign::Left,
            vertical_align: fontdue::layout::VerticalAlign::Top,
            line_height: size * 1.2,
            wrap_style: fontdue::layout::WrapStyle::Word,
            wrap_hard_breaks: true,
        });

        let fonts = &[font.inner()];
        self.layout.append(fonts, &TextStyle::new(text, size, 0));

        self.layout
            .glyphs()
            .iter()
            .map(|g| GlyphPosition {
                c: g.parent,
                x: g.x,
                y: g.y,
                width: g.width as f32,
                height: g.height as f32,
            })
            .collect()
    }

    /// Get layout bounds (width, height).
    #[must_use]
    pub fn bounds(&self) -> (f32, f32) {
        let glyphs = self.layout.glyphs();
        if glyphs.is_empty() {
            return (0.0, 0.0);
        }

        let max_x = glyphs
            .iter()
            .map(|g| g.x + g.width as f32)
            .fold(0.0_f32, f32::max);
        let max_y = glyphs
            .iter()
            .map(|g| g.y + g.height as f32)
            .fold(0.0_f32, f32::max);

        (max_x, max_y)
    }
}

impl Default for SimpleLayoutEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Positioned glyph from layout.
#[derive(Clone, Copy, Debug)]
pub struct GlyphPosition {
    /// Character.
    pub c: char,
    /// X position.
    pub x: f32,
    /// Y position.
    pub y: f32,
    /// Glyph width.
    pub width: f32,
    /// Glyph height.
    pub height: f32,
}
