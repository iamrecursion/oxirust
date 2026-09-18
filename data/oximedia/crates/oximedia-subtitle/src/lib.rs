//! Subtitle and closed caption rendering for OxiMedia.
//!
//! This crate provides comprehensive subtitle rendering support including:
//!
//! - **Subtitle Formats**: SubRip (SRT), WebVTT, SSA/ASS, CEA-608/708
//! - **Text Rendering**: Font loading, glyph caching, Unicode support, bidirectional text
//! - **Styling**: Font properties, colors, outlines, shadows, positioning
//! - **Advanced Features**: Burn-in, animations, collision detection, karaoke effects
//!
//! # Supported Formats
//!
//! | Format | Description | Features |
//! |--------|-------------|----------|
//! | SRT | SubRip | Basic text, simple HTML tags |
//! | WebVTT | Web Video Text Tracks | Positioning, cue settings |
//! | SSA/ASS | Advanced SubStation Alpha | Full styling, animations, karaoke |
//! | CEA-608/708 | Closed Captions | Real-time captions, pop-on, roll-up |
//!
//! # Example
//!
//! ```ignore
//! use oximedia_subtitle::{SubtitleRenderer, SubtitleStyle, Font};
//! use oximedia_codec::VideoFrame;
//!
//! // Load font
//! let font_data = std::fs::read("font.ttf")?;
//! let font = Font::from_bytes(font_data)?;
//!
//! // Create renderer with custom style
//! let style = SubtitleStyle::default()
//!     .with_font_size(48.0)
//!     .with_color(255, 255, 255, 255);
//!
//! let renderer = SubtitleRenderer::new(font, style);
//!
//! // Parse subtitles
//! let subtitle_data = std::fs::read("movie.srt")?;
//! let subtitles = parser::srt::parse(&subtitle_data)?;
//!
//! // Render onto frame
//! let mut frame = VideoFrame::new(...);
//! renderer.render_subtitle(&subtitles[0], &mut frame, timestamp)?;
//! ```

#![warn(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_self,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    dead_code,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::unnested_or_patterns,
    unused_imports,
    unused_variables,
    clippy::unnecessary_wraps,
    clippy::redundant_pattern_matching,
    clippy::pedantic,
    clippy::approx_constant,
    clippy::builtin_type_shadow
)]

pub mod burn_in;
pub mod error;
pub mod font;
pub mod format_convert;
pub mod format_converter;
pub mod overlay;
pub mod parser;
pub mod renderer;
pub mod soft_shadow;
pub mod style;
pub mod sub_style;
pub mod text;
pub mod timing;
pub mod timing_adjuster;

// CEA-608/708 encoding and embedding
pub mod cea;
pub mod convert;

// New accessibility and language modules
pub mod accessibility;
pub mod segmentation;
pub mod translation;

// Timing, line-breaking, and spell-check utilities
pub mod line_break;
pub mod spell_check;
pub mod timing_adjust;

// New parsing and validation modules
pub mod cue_parser;
pub mod subtitle_merge;
pub mod subtitle_validator;

// New reading-speed, style, and overlap modules
pub mod overlap_detect;
pub mod reading_speed;
pub mod subtitle_style_ext;

// Timestamp-indexed lookup, cue point annotations, and multi-format export
pub mod cue_point;
pub mod subtitle_export;
pub mod subtitle_index;

// Cue timing, position calculation, and subtitle diffing
pub mod cue_timing;
pub mod position_calc;
pub mod subtitle_diff;

// Full-text search, statistics, and sanitization
pub mod subtitle_sanitize;
pub mod subtitle_search;
pub mod subtitle_stats;

// Forced subtitle detection
pub mod forced_subtitle;

// Automatic subtitle timing alignment between two tracks
pub mod subtitle_alignment;

// IMSC1/TTML2 enhanced parser with regions, styles, and spans
pub mod ttml_v2;

// CEA-708 DTVCC decoder
pub mod cea708;

// ── Wave 6 / Slice F orphan modules ──────────────────────────────────────────

// ASS/SSA style override tag parser (\an, \pos, \clip, etc.)
pub mod ass_override;

// WebVTT / SRT cue positioning helpers (line, position, size, align)
pub mod cue_positioning;

// Real-time CEA-608 decoder for live streams (`Eia608Decoder`).
// `parser::eia608_realtime` is a separate module with `RealtimeCea608Decoder`.
pub mod eia608_realtime;

// Fallback font chain for missing glyphs (CJK, Arabic, Devanagari)
pub mod fallback_fonts;

// Glyph atlas packing and GPU-ready texture atlas for subtitle rendering
pub mod glyph_atlas;

// ASS karaoke timing engine with syllable-level highlight events
pub mod karaoke_engine;

// Real-time roll-up / paint-on / pop-on live caption display manager
pub mod live_caption;

// Subtitle cue position (x, y, origin) calculation helpers
pub mod position;

// Full-text subtitle search utilities (index build, query, highlight)
pub mod search;

// Sign language overlay region and picture-in-picture positioning
pub mod sign_language;

// SSA/ASS style cache for O(1) style lookups per dialogue line
pub mod ssa_style_cache;

// CSS-like subtitle style inheritance resolver
pub mod style_inherit;

// Chapter-level subtitle segmentation and chapter marker extraction
pub mod subtitle_chapters;

// Bitmap-based subtitle OCR for PGS / VobSub pattern matching
pub mod subtitle_ocr;

// DVB Teletext (ETS 300 706) subtitle extraction
pub mod teletext;

// Higher-level Teletext subtitle struct wrappers and utilities
pub mod teletext_subtitle;

// WCAG 2.1 AA/AAA contrast ratio validator for subtitle colours
pub mod wcag_validator;

// Re-export main types
pub use cea708::{CaptionWindow, Dtvcc708Command, Dtvcc708Decoder, Dtvcc708Packet};
pub use error::{SubtitleError, SubtitleResult};
pub use font::{Font, GlyphCache};
pub use overlay::overlay_subtitle;
pub use renderer::{DirtyRect, IncrementalSubtitleRenderer, SubtitleRenderer};
pub use style::{Alignment, Animation, Color, OutlineStyle, Position, ShadowStyle, SubtitleStyle};
pub use text::{BidiLevel, TextLayout, TextLayoutEngine};
pub use ttml_v2::{SubtitleEntry, TtmlParser, TtmlRegion, TtmlSpan, TtmlStyle};

/// A single subtitle cue with timing and content.
#[derive(Clone, Debug)]
pub struct Subtitle {
    /// Unique identifier (e.g. sequence number in SRT).
    pub id: Option<String>,
    /// Start time in milliseconds.
    pub start_time: i64,
    /// End time in milliseconds.
    pub end_time: i64,
    /// Subtitle text content.
    pub text: String,
    /// Optional styling override.
    pub style: Option<SubtitleStyle>,
    /// Position override.
    pub position: Option<Position>,
    /// Animation effects.
    pub animations: Vec<Animation>,
}

impl Subtitle {
    /// Create a new subtitle cue.
    #[must_use]
    pub fn new(start_time: i64, end_time: i64, text: String) -> Self {
        Self {
            id: None,
            start_time,
            end_time,
            text,
            style: None,
            position: None,
            animations: Vec::new(),
        }
    }

    /// Create a subtitle cue with an id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Check if this subtitle is active at the given timestamp.
    #[must_use]
    pub fn is_active(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_time && timestamp_ms < self.end_time
    }

    /// Get duration in milliseconds.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.end_time - self.start_time
    }

    /// Add an animation effect.
    pub fn with_animation(mut self, animation: Animation) -> Self {
        self.animations.push(animation);
        self
    }

    /// Set position override.
    #[must_use]
    pub fn with_position(mut self, position: Position) -> Self {
        self.position = Some(position);
        self
    }

    /// Set style override.
    #[must_use]
    pub fn with_style(mut self, style: SubtitleStyle) -> Self {
        self.style = Some(style);
        self
    }
}

// ============================================================================
// Simple Parser Structs API
// ============================================================================

/// High-level SRT parser struct.
///
/// # Example
///
/// ```ignore
/// use oximedia_subtitle::SrtParser;
/// let text = "1\n00:00:01,000 --> 00:00:04,000\nHello!\n\n";
/// let subs = SrtParser::parse(text).expect("should succeed in test");
/// ```
pub struct SrtParser;

impl SrtParser {
    /// Parse SRT subtitle text and return a vector of subtitles.
    ///
    /// # Errors
    ///
    /// Returns error if the text is not valid SRT format.
    pub fn parse(text: &str) -> SubtitleResult<Vec<Subtitle>> {
        parser::srt::parse_srt(text)
    }
}

/// High-level ASS/SSA parser struct.
pub struct AssParser;

impl AssParser {
    /// Parse ASS/SSA subtitle text and return a vector of subtitles.
    ///
    /// # Errors
    ///
    /// Returns error if the text is not valid ASS format.
    pub fn parse(text: &str) -> SubtitleResult<Vec<Subtitle>> {
        let file = parser::ssa::parse_ass(text)?;
        Ok(file.events)
    }
}

/// High-level WebVTT parser struct.
pub struct WebVttParser;

impl WebVttParser {
    /// Parse WebVTT subtitle text and return a vector of subtitles.
    ///
    /// # Errors
    ///
    /// Returns error if the text is not valid WebVTT format.
    pub fn parse(text: &str) -> SubtitleResult<Vec<Subtitle>> {
        parser::webvtt::parse_webvtt(text)
    }
}

#[cfg(test)]
mod subtitle_api_tests {
    use super::*;

    const SAMPLE_SRT: &str = "1\n00:00:01,000 --> 00:00:04,000\nHello, world!\n\n2\n00:00:05,000 --> 00:00:08,000\nSecond subtitle.\n\n";

    #[test]
    fn test_subtitle_id_field() {
        let sub = Subtitle::new(0, 1000, "test".to_string()).with_id("42");
        assert_eq!(sub.id, Some("42".to_string()));
    }

    #[test]
    fn test_subtitle_new_has_no_id() {
        let sub = Subtitle::new(0, 1000, "test".to_string());
        assert!(sub.id.is_none());
    }

    #[test]
    fn test_srt_parser_basic() {
        let subs = SrtParser::parse(SAMPLE_SRT).expect("should succeed in test");
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].text, "Hello, world!");
        assert_eq!(subs[0].start_time, 1000);
        assert_eq!(subs[0].end_time, 4000);
    }

    #[test]
    fn test_srt_parser_second_entry() {
        let subs = SrtParser::parse(SAMPLE_SRT).expect("should succeed in test");
        assert_eq!(subs[1].text, "Second subtitle.");
        assert_eq!(subs[1].start_time, 5000);
        assert_eq!(subs[1].end_time, 8000);
    }

    #[test]
    fn test_webvtt_parser_basic() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello VTT!\n\n";
        let subs = WebVttParser::parse(vtt).expect("should succeed in test");
        assert!(!subs.is_empty());
        assert_eq!(subs[0].text, "Hello VTT!");
    }

    #[test]
    fn test_webvtt_parser_timing() {
        let vtt = "WEBVTT\n\n00:00:05.500 --> 00:00:09.000\nTimed cue.\n\n";
        let subs = WebVttParser::parse(vtt).expect("should succeed in test");
        assert_eq!(subs[0].start_time, 5500);
        assert_eq!(subs[0].end_time, 9000);
    }

    #[test]
    fn test_subtitle_is_active() {
        let sub = Subtitle::new(1000, 4000, "test".to_string());
        assert!(sub.is_active(2000));
        assert!(!sub.is_active(500));
        assert!(!sub.is_active(5000));
    }

    #[test]
    fn test_subtitle_duration() {
        let sub = Subtitle::new(1000, 4000, "test".to_string());
        assert_eq!(sub.duration(), 3000);
    }

    #[test]
    fn test_ass_parser_basic() {
        let ass = "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello ASS!\n\n";
        let result = AssParser::parse(ass);
        assert!(result.is_ok());
        let subs = result.expect("should succeed in test");
        assert!(!subs.is_empty());
    }
}
